#!/usr/bin/env python3
"""
QuantBox Live Trader - 암호화폐 페어 트레이딩 실거래 시스템

실행 전 확인사항:
1. .env 파일에 바이낸스 API 키 설정 (cp .env.example .env)
2. --paper 또는 --once 플래그로 모드 선택
3. 초기 자본금 및 리스크 파라미터 확인

환경변수 (.env):
    BINANCE_API_KEY: 바이낸스 API 키 (필수)
    BINANCE_API_SECRET: 바이낸스 시크릿 키 (필수)
    BINANCE_TESTNET: 테스트넷 사용 여부 (기본값: false)
    LEVERAGE: 레버리지 배수 (기본값: 2)
    MAX_TRADERS: 최대 동시 트레이더 수 (기본값: 5)
    POSITION_SIZE_PCT: 슬롯당 자본 비율 (기본값: auto=1/max_traders)
    ENTRY_ZSCORE: 진입 z-score 임계값 (기본값: 2.0)
    EXIT_ZSCORE: 청산 z-score 임계값 (기본값: 0.5)
    STOP_LOSS_ZSCORE: 손절 z-score (기본값: 3.5)
    TRADING_FEE: 거래 수수료 (기본값: 0.0005)
    SLIPPAGE: 슬리피지 (기본값: 0.001)
    FUNDING_RATE_8H: 8시간 펀딩비 (기본값: 0.0001)
    N_COINS: 유니버스 코인 수 (기본값: 30)
    PAPER_BALANCE: 페이퍼 트레이딩 가상 잔고 (기본값: 1000)
    DISCORD_WEBHOOK_URL: Discord 웹훅 URL (미설정 시 알림 비활성화)

사용법:
    python live_trader.py [--paper] [--once]

    --paper: Paper trading 모드 (실제 주문 없이 시뮬레이션)
    --once: 한 번만 실행하고 종료
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests

# QuantBox imports
from quantbox.core.config import Config, TradingMode
from quantbox.core.settings import get_settings
from quantbox.core.trader_slot import (
    TraderSlotState,
    get_occupied_symbols,
    get_idle_slots,
    get_active_slots,
    calculate_slot_capital,
    calculate_margin_from_notional,
    is_pair_available,
)
from quantbox.strategy.pair_selector import PairSelector, PairCandidate
from quantbox.strategy.filters import (
    FilterPipeline, CorrelationFilter, CointegrationFilter,
    HalfLifeFilter, HurstFilter
)
from quantbox.data.preprocessor import DataPreprocessor
from quantbox.core.limit_chaser import (
    LimitOrderChaser, ChaseConfig, ChaseFillStatus, ChaseResult,
)
from quantbox.strategy.kalman_spread import OnlineKalmanSpread, KalmanState

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('live_trader.log', mode='a')
    ]
)
logger = logging.getLogger('LiveTrader')

# Suppress noisy loggers
for log_name in ['urllib3', 'requests', 'binance']:
    logging.getLogger(log_name).setLevel(logging.WARNING)


from quantbox.notifications.discord import DiscordNotifier
from quantbox.notifications.bot import (
    DiscordBotRunner, BotCommand, CommandType,
)


class LivePairTrader:
    """실시간 페어 트레이딩 시스템 - 멀티 트레이더 지원

    최대 max_traders명의 독립적인 트레이더 슬롯이 동시에 운영되며,
    각 슬롯은 독립적으로 페어를 탐색하고 포지션을 관리한다.
    심볼 중복은 슬롯 간 방지된다.
    """
    
    def __init__(
        self,
        paper_mode: bool = False,
        config: Dict = None
    ):
        self.paper_mode = paper_mode

        # pydantic-settings: .env 검증 (필수 변수 누락 시 즉시 에러)
        self._settings = get_settings()
        self.config = config or self._settings.to_trader_config()
        
        # Initialize Binance client
        self.client = Client(
            api_key=self._settings.binance_api_key,
            api_secret=self._settings.binance_api_secret,
            testnet=self._settings.binance_testnet,
        )
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.filter_pipeline = self._create_filter_pipeline()
        self.pair_selector = PairSelector(
            filter_pipeline=self.filter_pipeline,
            max_pairs=20
        )
        
        # Multi-trader state
        self.max_traders: int = self.config.get('max_traders', 5)
        # Auto-calculate position_size_pct if sentinel (-1)
        if self.config['position_size_pct'] < 0:
            self.config['position_size_pct'] = 1.0 / self.max_traders
        self.slots: list[TraderSlotState] = [
            TraderSlotState(slot_id=i) for i in range(self.max_traders)
        ]
        # slot_id -> position dict
        self.slot_positions: dict[int, Dict] = {}
        # slot_id -> entry time
        self.slot_entry_times: dict[int, datetime] = {}
        self.trade_history: List[Dict] = []
        self.prev_prices: Dict[str, float] = {}
        
        # Paper trading state
        self.paper_balance = self._settings.paper_balance
        self.paper_positions: List[Dict] = []
        
        # Discord notifications
        self.discord = DiscordNotifier(webhook_url=self._settings.discord_webhook_url)

        # Discord bot (양방향 원격 명령)
        self._bot_runner: DiscordBotRunner | None = None
        if self._settings.discord_bot_token and self._settings.discord_channel_id:
            self._bot_runner = DiscordBotRunner(
                token=self._settings.discord_bot_token,
                channel_id=self._settings.discord_channel_id,
            )

        # Limit order chaser (maker-only execution)
        self._chaser = LimitOrderChaser(
            client=self.client,
            config=ChaseConfig(
                max_chase_ticks=10,
                chase_interval_seconds=1.0,
                timeout_seconds=30.0,
                fallback_to_market=True,
                initial_offset_ticks=0,
            ),
            paper_mode=self.paper_mode,
        )

        # 실패한 심볼 블랙리스트 (사이클 간 지속, TTL 기반 자동 만료)
        # key: symbol, value: 실패 시각
        self.BLACKLIST_TTL_SECONDS: int = 3600  # 1시간 후 자동 만료
        self._failed_symbols: dict[str, datetime] = {}

        # Kalman filter instances per slot (adaptive z-score)
        # key: slot_id, value: OnlineKalmanSpread
        self._slot_kalman: dict[int, OnlineKalmanSpread] = {}

        # 일시정지 플래그 (Discord !pause / !resume)
        self.paused: bool = False
        # 종료 요청 플래그 (Discord !stop)
        self._stop_requested: bool = False
        # 알림 중복 방지 플래그 (상태 전이 시 1회만 발송)
        self._all_slots_occupied_notified: bool = False
        self._insufficient_balance_notified: bool = False

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"LivePairTrader initialized in {mode_str} mode")
        logger.info(f"Max concurrent traders: {self.max_traders}")
        logger.info(f"Leverage: {self.config['leverage']}x")
        logger.info(f"Testnet: {self._settings.binance_testnet}")
        if paper_mode:
            logger.info(f"Paper Balance: ${self.paper_balance:.2f} USDT")
        
    def _default_config(self) -> Dict:
        """기본 설정 — get_settings().to_trader_config() 사용 권장."""
        return get_settings().to_trader_config()
    
    def _create_filter_pipeline(self) -> FilterPipeline:
        """필터 파이프라인 생성"""
        return FilterPipeline([
            CorrelationFilter(
                min_correlation=self.config['min_correlation'],
                max_correlation=0.98,
                use_returns=True
            ),
            CointegrationFilter(
                max_pvalue=self.config['max_cointegration_pvalue']
            ),
            HalfLifeFilter(
                min_half_life=self.config['min_half_life'],
                max_half_life=self.config['max_half_life']
            ),
            HurstFilter(
                max_hurst=self.config['max_hurst']
            ),
        ])
    
    # ==================== Data Methods ====================
    
    def get_trading_coins(self) -> List[str]:
        """거래량 상위 코인 목록 조회"""
        try:
            exchange_info = self.client.futures_exchange_info()
            
            trading_coins = [
                info['symbol']
                for info in exchange_info['symbols']
                if info['symbol'].endswith(self.config['quote_currency'])
                and info['status'] == 'TRADING'
            ]
            
            # Get 24h volumes
            tickers = self.client.futures_ticker()
            volume_dict = {
                t['symbol']: float(t['quoteVolume'])
                for t in tickers
                if t['symbol'] in trading_coins
            }
            
            # Sort by volume
            sorted_coins = sorted(
                volume_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            result = [coin for coin, _ in sorted_coins[:self.config['n_coins']]]
            logger.info(f"Selected {len(result)} coins by volume")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get trading coins: {e}")
            return []
    
    def download_klines(
        self,
        symbol: str,
        interval: str,
        hours: int
    ) -> Optional[pd.DataFrame]:
        """캔들 데이터 다운로드"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                limit=1000
            )
            
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Volume', 'Trades', 'Taker Buy Base',
                'Taker Buy Quote', 'Ignore'
            ])
            
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df.set_index('Open Time', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Failed to download klines for {symbol}: {e}")
            return None
    
    def download_multi_symbol_data(
        self,
        symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """여러 심볼 데이터 다운로드"""
        data = {}
        for symbol in symbols:
            df = self.download_klines(
                symbol,
                self.config['interval'],
                self.config['formation_hours']
            )
            if df is not None and len(df) >= 100:
                data[symbol] = df
                logger.debug(f"Downloaded {len(df)} bars for {symbol}")
        
        logger.info(f"Downloaded data for {len(data)} symbols")
        return data
    
    # ==================== Cost Calculation ====================

    def _fetch_funding_rate(self, symbol: str) -> float:
        """바이낸스 선물 실시간 펀딩비 조회.

        Returns:
            현재 펀딩비 (8시간 기준). 실패 시 config fallback값.
        """
        try:
            info = self.client.futures_mark_price(symbol=symbol)
            rate = float(info.get('lastFundingRate', 0))
            if rate != 0:
                return abs(rate)
        except Exception as e:
            logger.debug(f"Failed to fetch funding rate for {symbol}: {e}")
        return self.config['funding_rate_8h']

    def _calculate_funding_cost(
        self,
        symbol1: str,
        symbol2: str,
        notional: float,
        holding_hours: float,
    ) -> float:
        """실제 펀딩비 기반 비용 계산.

        양 레그의 실시간 펀딩비를 조회하여 보유 시간에 비례한 비용을 반환.
        페이퍼 모드 또는 API 실패 시 config 고정값으로 fallback.

        Args:
            symbol1: Leg1 심볼.
            symbol2: Leg2 심볼.
            notional: 포지션 총 노셔널 가치.
            holding_hours: 보유 시간 (hours).

        Returns:
            추정 펀딩비 비용 (양수).
        """
        if self.paper_mode:
            rate = self.config['funding_rate_8h']
        else:
            rate1 = self._fetch_funding_rate(symbol1)
            rate2 = self._fetch_funding_rate(symbol2)
            # 페어 트레이딩은 양 레그이므로 두 펀딩비 합산
            # (한쪽은 long, 한쪽은 short → 부호 반대이지만 보수적으로 양쪽 절대값 합산)
            rate = (rate1 + rate2) / 2

        funding_periods = holding_hours / 8.0
        # 양 레그 × 펀딩비 × 기간
        cost = rate * 2 * notional * funding_periods
        return max(0.0, cost)

    # ==================== Trading Methods ====================
    
    def get_price(self, symbol: str) -> float:
        """현재가 조회"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            self.prev_prices[symbol] = price
            return price
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return self.prev_prices.get(symbol, 0.0)
    
    def get_available_balance(self, asset: str = 'USDT') -> float:
        """사용 가능 잔고 조회 (미사용 마진)"""
        if self.paper_mode:
            return self.paper_balance
        
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == asset:
                    return float(balance['availableBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_total_balance(self, asset: str = 'USDT') -> float:
        """총 자산 잔고 조회 (available + 미실현 PnL 포함).

        calculate_slot_capital()의 total_book_value 계산에 사용.
        백테스트의 current_capital + total_allocated 와 동일한 개념.

        Args:
            asset: 조회할 자산 (기본 USDT).

        Returns:
            총 자산 잔고. 실패 시 0.0.
        """
        if self.paper_mode:
            return self.paper_balance

        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == asset:
                    return float(balance['balance'])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get total balance: {e}")
            return 0.0
    
    def get_current_positions(self) -> List[Dict]:
        """현재 포지션 조회"""
        # Paper mode: use simulated positions
        if self.paper_mode:
            return self.paper_positions
        
        try:
            positions = self.client.futures_account()['positions']
            return [
                pos for pos in positions
                if float(pos['positionAmt']) != 0.0
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """심볼 정보 조회 (tick_size, price_precision 포함)."""
        try:
            info = self.client.futures_exchange_info()
            for sym_info in info['symbols']:
                if sym_info['symbol'] == symbol:
                    filters = sym_info['filters']
                    result = {
                        'quantity_precision': sym_info['quantityPrecision'],
                        'price_precision': sym_info['pricePrecision'],
                    }
                    for f in filters:
                        if f['filterType'] == 'LOT_SIZE':
                            result['step_size'] = float(f['stepSize'])
                            result['min_qty'] = float(f['minQty'])
                        elif f['filterType'] == 'PRICE_FILTER':
                            result['tick_size'] = float(f['tickSize'])
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            result['min_notional'] = float(f['notional'])
                    return result
            return {}
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
            return {}
    
    def round_quantity(self, quantity: float, step_size: float, precision: int) -> float:
        """수량 반올림"""
        import math
        rounded = math.floor(quantity / step_size) * step_size
        return round(rounded, precision)
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Optional[Dict]:
        """주문 실행 — LimitOrderChaser를 통한 maker-only 체결.

        Limit 지정가를 best bid/ask에 걸고, 미체결 시 가격을 추격한다.
        추격 실패 시 시장가(MARKET)로 fallback하여 체결을 보장한다.

        Returns:
            체결 정보 dict 또는 None (실패 시).
        """
        try:
            # Set leverage (live mode only)
            if not self.paper_mode:
                self.client.futures_change_leverage(
                    symbol=symbol,
                    leverage=self.config['leverage']
                )

            # Get tick_size for price rounding
            info = self.get_symbol_info(symbol)
            tick_size = info.get('tick_size', 0.01)
            price_precision = info.get('price_precision', 2)
            step_size = info.get('step_size', 0.0)
            quantity_precision = info.get('quantity_precision', 8)

            # Execute via limit chaser
            result: ChaseResult = self._chaser.chase_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                tick_size=tick_size,
                price_precision=price_precision,
                step_size=step_size,
                quantity_precision=quantity_precision,
            )

            if result.status in (ChaseFillStatus.FILLED, ChaseFillStatus.FALLBACK_MARKET):
                fill_type = "MAKER" if result.maker_fill else "TAKER(fallback)"

                # 안전 검증: 체결 가격이 0이면 실제 체결 여부 미확인 → 실패 처리
                if result.avg_price <= 0:
                    logger.error(
                        f"Order {fill_type}: {side} {result.filled_quantity:.6f} {symbol} "
                        f"reported FILLED but avg_price={result.avg_price:.4f} — "
                        f"TREATING AS FAILED (unverified fill). "
                        f"Order ID: {result.order_id}"
                    )
                    return None

                logger.info(
                    f"Order {fill_type}: {side} {result.filled_quantity:.6f} {symbol} "
                    f"@ {result.avg_price:.4f} (chase={result.chase_count}, "
                    f"{result.elapsed_seconds:.1f}s) - ID: {result.order_id}"
                )
                return {
                    'orderId': result.order_id,
                    'status': 'FILLED',
                    'executedQty': str(result.filled_quantity),
                    'avgPrice': str(result.avg_price),
                    'makerFill': result.maker_fill,
                    'chaseCount': result.chase_count,
                }
            else:
                logger.error(
                    f"Order FAILED: {side} {quantity:.6f} {symbol} "
                    f"(status={result.status.value}, chase={result.chase_count})"
                )
                return None

        except BinanceAPIException as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected order error for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, quantity: float) -> dict | None:
        """포지션 청산.

        Returns:
            체결 정보 dict {'avgPrice': float, 'executedQty': float} 또는 실패 시 None.
        """
        side = 'SELL' if quantity > 0 else 'BUY'
        quantity = abs(quantity)
        
        info = self.get_symbol_info(symbol)
        if not info:
            return None
        
        rounded_qty = self.round_quantity(
            quantity,
            info.get('step_size', 0.001),
            info.get('quantity_precision', 3)
        )
        
        if rounded_qty <= 0:
            logger.warning(f"Quantity too small to close for {symbol}")
            return None
        
        order = self.place_order(symbol, side, rounded_qty)
        return order
    
    def close_all_positions(self):
        """모든 포지션 청산"""
        positions = self.get_current_positions()
        if not positions:
            logger.info("No positions to close")
        else:
            for pos in positions:
                symbol = pos['symbol']
                quantity = float(pos['positionAmt'])
                logger.info(f"Closing position: {symbol} qty={quantity}")
                self.close_position(symbol, quantity)

            time.sleep(2)
            remaining = self.get_current_positions()
            if remaining:
                logger.warning(f"Some positions still open: {remaining}")
            else:
                logger.info("All positions closed successfully")

        # Reset all slots regardless of exchange positions
        for slot in self.slots:
            slot.reset()
        self.slot_positions.clear()
        self.slot_entry_times.clear()

    def close_slot_positions(self, slot_id: int) -> dict[str, float]:
        """특정 슬롯의 포지션만 청산.

        두 레그 모두 성공적으로 청산된 경우에만 슬롯을 리셋한다.
        청산 실패 시 슬롯 상태를 유지하고 CRITICAL 로그를 남기며,
        거래소에 유령 포지션이 남는 것을 방지한다.

        Args:
            slot_id: 청산할 슬롯 ID.

        Returns:
            {'exit_price1': float, 'exit_price2': float} 실제 체결가.
            체결 실패한 레그는 0.0.
        """
        exit_prices = {'exit_price1': 0.0, 'exit_price2': 0.0}

        if slot_id not in self.slot_positions:
            return exit_prices

        pos = self.slot_positions[slot_id]
        symbol1, symbol2 = pos['symbol1'], pos['symbol2']

        # Close both legs
        qty1 = pos.get('qty1', 0)
        qty2 = pos.get('qty2', 0)

        close1_result = None
        close2_result = None

        if abs(qty1) > 0:
            close1_result = self.close_position(symbol1, qty1)
        if abs(qty2) > 0:
            close2_result = self.close_position(symbol2, qty2)

        if close1_result:
            if isinstance(close1_result, dict):
                exit_prices['exit_price1'] = float(close1_result.get('avgPrice', 0))
        if close2_result:
            if isinstance(close2_result, dict):
                exit_prices['exit_price2'] = float(close2_result.get('avgPrice', 0))

        if not close1_result or not close2_result:
            failed = []
            if not close1_result:
                failed.append(f"{symbol1}(qty={qty1})")
            if not close2_result:
                failed.append(f"{symbol2}(qty={qty2})")
            logger.critical(
                f"Slot {slot_id}: CLOSE FAILED for {', '.join(failed)} — "
                f"ORPHANED POSITION on exchange! Manual intervention required."
            )
            self.discord.send(content=(
                f"🚨 **CRITICAL** Slot {slot_id}: 청산 실패\n"
                f"실패 레그: {', '.join(failed)}\n"
                f"거래소에 유령 포지션 확인 필수!"
            ))
            # 실패해도 슬롯은 리셋한다 — 다음 사이클에서 재시도 불가능하므로
            # 대신 CRITICAL 로그와 Discord 알림으로 수동 개입 유도

        # Reset slot
        self.slots[slot_id].reset()
        del self.slot_positions[slot_id]
        if slot_id in self.slot_entry_times:
            del self.slot_entry_times[slot_id]
        # Clean up Kalman filter for this slot
        self._slot_kalman.pop(slot_id, None)

        return exit_prices

    @property
    def occupied_symbols(self) -> set[str]:
        """현재 모든 활성 슬롯에서 사용 중인 심볼."""
        return get_occupied_symbols(self.slots)

    @property
    def active_slot_count(self) -> int:
        """현재 포지션을 보유 중인 슬롯 수."""
        return len(get_active_slots(self.slots))

    @property
    def idle_slot_ids(self) -> list[int]:
        """포지션이 없는 슬롯 ID 목록."""
        return [s.slot_id for s in get_idle_slots(self.slots)]
    
    # ==================== Strategy Methods ====================
    
    def find_best_pair(
        self,
        data: Dict[str, pd.DataFrame],
        excluded_symbols: set[str] = None,
        cached_candidates: list[PairCandidate] | None = None,
    ) -> Optional[Tuple[PairCandidate, Dict]]:
        """최적 페어 선택 (이미 사용 중인 심볼 제외).

        Args:
            data: 심볼별 OHLCV 데이터.
            excluded_symbols: 제외할 심볼 집합 (다른 슬롯이 사용 중).
            cached_candidates: 사전에 select_pairs()로 계산된 후보 리스트.
                None이면 내부에서 select_pairs()를 호출한다.
                run_cycle에서 1회 계산 후 재사용하여 CPU 낭비를 방지한다.

        Returns:
            (PairCandidate, pair_info) 또는 None.
        """
        excluded_symbols = excluded_symbols or set()
        logger.info(f"Searching for cointegrated pairs (excluded: {excluded_symbols or 'none'})...")

        if cached_candidates is not None:
            candidates = cached_candidates
            logger.debug(f"Using cached candidates: {len(candidates)} pairs")
        else:
            candidates = self.pair_selector.select_pairs(data, progress=False)

        if not candidates:
            logger.warning("No valid pairs found")
            return None
        
        # Filter by current z-score (ready to trade)
        tradeable = self.pair_selector.select_tradeable_pairs(
            candidates,
            min_z_score=self.config['entry_zscore'],
            max_z_score=5.0
        )
        
        if not tradeable:
            logger.info("No pairs meet entry criteria")
            return None

        # Find best pair that doesn't overlap with excluded symbols
        best = None
        for candidate in tradeable:
            if is_pair_available(
                (candidate.symbol1, candidate.symbol2), excluded_symbols
            ):
                best = candidate
                break

        if best is None:
            logger.info("No pairs available (all overlap with active positions)")
            return None
        
        logger.info(
            f"Best pair: {best.symbol1}/{best.symbol2} | "
            f"Z-Score: {best.current_z_score:.2f} | "
            f"Half-life: {best.half_life:.1f} | "
            f"Score: {best.combined_score:.4f}"
        )
        
        # Prepare pair info
        pair_info = {
            'hedge_ratio': best.hedge_ratio,
            'spread_mean': 0,  # Will recalculate
            'spread_std': best.spread_std,
            'half_life': best.half_life,
        }
        
        # Calculate spread statistics
        df1 = data[best.symbol1]
        df2 = data[best.symbol2]
        log1 = np.log(df1['Close'])
        log2 = np.log(df2['Close'])
        spread = log1 - best.hedge_ratio * log2
        pair_info['spread_mean'] = spread.mean()
        pair_info['spread_std'] = spread.std()

        # Attach raw DataFrames for KF warmup & vol gate
        pair_info['_df1'] = df1
        pair_info['_df2'] = df2
        
        return best, pair_info
    
    def calculate_zscore(
        self,
        price1: float,
        price2: float,
        hedge_ratio: float,
        spread_mean: float,
        spread_std: float
    ) -> float:
        """Z-Score 계산"""
        spread = np.log(price1) - hedge_ratio * np.log(price2)
        return (spread - spread_mean) / spread_std if spread_std > 0 else 0.0

    def _get_effective_exit_zscore(self, pos: Dict) -> float:
        """Half-life 기반 동적 exit z-score 계산.

        use_dynamic_exit=False이면 고정 exit_zscore 반환.
        True이면 half_life에 비례해 exit threshold를 조정:
          - 빠른 회귀 (low half_life) → 타이트한 exit (0에 가깝게)
          - 느린 회귀 (high half_life) → 느슨한 exit (일찍 청산)

        Args:
            pos: slot_positions의 position dict (half_life 포함).

        Returns:
            해당 포지션에 적용할 exit z-score 임계값.
        """
        if not self.config.get('use_dynamic_exit', False):
            return self.config['exit_zscore']

        half_life = pos.get('half_life', self.config['max_half_life'])
        min_hl = self.config['min_half_life']
        max_hl = self.config['max_half_life']
        hl_ratio = (half_life - min_hl) / (max_hl - min_hl)
        hl_ratio = max(0.0, min(1.0, hl_ratio))

        return self.config['dynamic_exit_base'] + hl_ratio * self.config['dynamic_exit_range']
    
    def execute_pair_trade(
        self,
        slot_id: int,
        pair: PairCandidate,
        pair_info: Dict,
        capital: float
    ) -> bool:
        """특정 슬롯에서 페어 트레이드 실행.

        capital은 calculate_slot_capital()에서 반환된 값으로,
        leverage가 적용된 notional exposure이다.
        실제 마진(cash)은 capital / leverage.

        Args:
            slot_id: 트레이드를 실행할 슬롯 ID.
            pair: 선택된 페어 후보.
            pair_info: 헤지 비율, 스프레드 통계 등.
            capital: 레버리지 적용된 노셔널 (calculate_slot_capital 반환값).

        Returns:
            True if 포지션 진입 성공.
        """
        symbol1, symbol2 = pair.symbol1, pair.symbol2
        hedge_ratio = pair_info['hedge_ratio']
        leverage = self.config['leverage']
        
        # Get current prices
        price1 = self.get_price(symbol1)
        price2 = self.get_price(symbol2)
        
        if price1 <= 0 or price2 <= 0:
            logger.error(f"Slot {slot_id}: Invalid prices")
            return False
        
        # Calculate z-score
        z_score = self.calculate_zscore(
            price1, price2, hedge_ratio,
            pair_info['spread_mean'], pair_info['spread_std']
        )
        
        logger.info(f"Slot {slot_id}: Current Z-Score: {z_score:.4f}")

        # ── Adaptive entry gates ──
        z_cap = self.config.get('z_score_cap', 4.0)
        if abs(z_score) > z_cap:
            logger.warning(
                f"Slot {slot_id}: Z-Score {z_score:.2f} exceeds cap "
                f"|{z_cap}| — regime anomaly, skipping entry"
            )
            return False

        vol_gate = self.config.get('vol_expansion_gate', 2.0)
        if vol_gate > 0:
            # Quick vol check: recent spread std vs formation std
            df1 = pair_info.get('_df1')
            df2 = pair_info.get('_df2')
            if df1 is not None and df2 is not None and len(df1) >= 8:
                recent_spread = (np.log(df1['Close'].iloc[-8:].values)
                                 - hedge_ratio * np.log(df2['Close'].iloc[-8:].values))
                recent_std = float(np.std(recent_spread))
                formation_std = pair_info['spread_std']
                vr = recent_std / formation_std if formation_std > 0 else 999
                if vr > vol_gate:
                    logger.warning(
                        f"Slot {slot_id}: Vol expansion {vr:.2f}x "
                        f"(recent_std={recent_std:.6f}, formation={formation_std:.6f}) "
                        f"> gate {vol_gate:.1f}x — skipping entry"
                    )
                    return False
                logger.debug(f"Slot {slot_id}: Vol ratio {vr:.2f}x — OK")

        if abs(z_score) < self.config['entry_zscore']:
            logger.info(f"Slot {slot_id}: Z-Score below entry threshold")
            return False
        
        # Determine signal: z < 0 -> Long spread, z > 0 -> Short spread
        signal = 1 if z_score < 0 else -1
        side1 = 'BUY' if signal == 1 else 'SELL'
        side2 = 'SELL' if signal == 1 else 'BUY'
        
        # capital is notional (leveraged). Use full notional for position sizing.
        margin = calculate_margin_from_notional(capital, leverage)
        effective_capital = capital * (1 - 2 * self.config['trading_fee'])
        
        s1_capital = effective_capital / (1 + hedge_ratio)
        s2_capital = hedge_ratio * s1_capital
        
        qty1 = s1_capital / price1
        qty2 = s2_capital / price2
        
        # Get symbol info and round quantities
        info1 = self.get_symbol_info(symbol1)
        info2 = self.get_symbol_info(symbol2)
        
        if not info1 or not info2:
            logger.error("Failed to get symbol info")
            return False
        
        qty1 = self.round_quantity(
            qty1, info1.get('step_size', 0.001), info1.get('quantity_precision', 3)
        )
        qty2 = self.round_quantity(
            qty2, info2.get('step_size', 0.001), info2.get('quantity_precision', 3)
        )
        
        # Check minimum notional
        notional1 = qty1 * price1
        notional2 = qty2 * price2
        
        min_notional1 = info1.get('min_notional', 5)
        min_notional2 = info2.get('min_notional', 5)
        
        if notional1 < min_notional1 or notional2 < min_notional2:
            logger.warning(f"Notional too small: {notional1:.2f}, {notional2:.2f}")
            return False
        
        logger.info(f"Slot {slot_id}: Opening position: {symbol1} {side1} {qty1:.6f}, {symbol2} {side2} {qty2:.6f}")
        logger.info(f"Slot {slot_id}: Capital usage: ${notional1 + notional2:.2f} ({(notional1 + notional2) / capital * 100:.1f}%)")
        
        # Place orders
        order1 = self.place_order(symbol1, side1, qty1)
        order2 = None

        if order1:
            order2 = self.place_order(symbol2, side2, qty2)

        if not order1 or not order2:
            # 한쪽 레그만 체결된 경우 — 거래소에 남은 포지션 직접 청산
            if order1 and not order2:
                logger.error(
                    f"Slot {slot_id}: Leg2 ({symbol2}) failed, "
                    f"closing orphaned leg1 ({symbol1}) directly"
                )
                filled_qty1 = float(order1.get('executedQty', qty1))
                close_side1 = 'SELL' if side1 == 'BUY' else 'BUY'
                close_ok = self.close_position(
                    symbol1, filled_qty1 if side1 == 'BUY' else -filled_qty1
                )
                if not close_ok:
                    logger.critical(
                        f"Slot {slot_id}: FAILED to close orphaned position "
                        f"{symbol1} qty={filled_qty1} — MANUAL CHECK REQUIRED"
                    )
                    self.discord.send(content=(
                        f"🚨 **CRITICAL** Slot {slot_id}: 유령 포지션 발생\n"
                        f"심볼: {symbol1} qty={filled_qty1}\n"
                        f"거래소에서 수동으로 청산 필요!"
                    ))
            elif not order1:
                logger.error(f"Slot {slot_id}: Leg1 ({symbol1}) failed, no cleanup needed")

            logger.info(
                f"Slot {slot_id}: Trade execution failed for "
                f"{symbol1}/{symbol2}, excluding from further search"
            )
            return False
        
        # Record position in slot — use actual fill prices for accurate PnL
        fill_price1 = float(order1.get('avgPrice', 0)) if order1 else price1
        fill_price2 = float(order2.get('avgPrice', 0)) if order2 else price2
        # Fallback to ticker price if fill price is missing (should not happen)
        if fill_price1 <= 0:
            logger.warning(f"Slot {slot_id}: Leg1 avgPrice missing, using ticker {price1}")
            fill_price1 = price1
        if fill_price2 <= 0:
            logger.warning(f"Slot {slot_id}: Leg2 avgPrice missing, using ticker {price2}")
            fill_price2 = price2

        position_data = {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'signal': signal,
            'qty1': qty1 if signal == 1 else -qty1,
            'qty2': -qty2 if signal == 1 else qty2,
            'entry_price1': fill_price1,
            'entry_price2': fill_price2,
            'entry_zscore': z_score,
            'hedge_ratio': hedge_ratio,
            'spread_mean': pair_info['spread_mean'],
            'spread_std': pair_info['spread_std'],
            'half_life': pair_info['half_life'],
            'capital': margin,  # Margin (cash locked), matches backtest slot.position_info['capital']
            'notional': capital,  # Notional exposure (margin × leverage)
            'leverage': leverage,
        }

        # Update slot state
        slot = self.slots[slot_id]
        slot.position_open = True
        slot.current_pair = (symbol1, symbol2)
        slot.position_info = position_data
        self.slot_positions[slot_id] = position_data
        self.slot_entry_times[slot_id] = datetime.now(timezone.utc)

        # ── Initialize Kalman filter for this slot ──
        try:
            kf = OnlineKalmanSpread(
                init_mu=position_data['spread_mean'],
                init_gamma=hedge_ratio,
                formation_spread_std=pair_info['spread_std'],
                delta=self.config.get('kalman_delta', 1e-5),
                obs_noise=self.config.get('kalman_obs_noise', 1e-3),
            )
            # Warmup with formation data
            df1_warmup = pair_info.get('_df1')
            df2_warmup = pair_info.get('_df2')
            if df1_warmup is not None and df2_warmup is not None:
                lp1 = np.log(df1_warmup['Close'].values).astype(np.float64)
                lp2 = np.log(df2_warmup['Close'].values).astype(np.float64)
                kf.warmup(lp1, lp2)
                logger.info(
                    f"Slot {slot_id}: KF warmed up ({len(lp1)} bars), "
                    f"gamma={kf.gamma:.4f}, mu={kf.mu:.6f}"
                )
            self._slot_kalman[slot_id] = kf
        except Exception as e:
            logger.warning(f"Slot {slot_id}: KF init failed: {e}, using fixed params")

        direction = "LONG Spread" if signal == 1 else "SHORT Spread"
        logger.info(f"Slot {slot_id}: Position opened: signal={signal}, z-score={z_score:.4f}")
        
        # Discord 알림
        self.discord.notify_open(
            symbol1, symbol2, f"[S{slot_id}] {direction}",
            z_score, capital,
            slot_id=slot_id,
            leverage=leverage,
            margin=margin,
            available_after=self.get_available_balance(),
        )
        
        return True
    
    def monitor_slot(self, slot_id: int) -> bool:
        """특정 슬롯의 포지션 모니터링 및 청산 조건 확인.

        Args:
            slot_id: 모니터링할 슬롯 ID.

        Returns:
            True if 포지션 유지, False if 청산됨.
        """
        if slot_id not in self.slot_positions:
            return False
        
        pos = self.slot_positions[slot_id]
        
        # Get current prices
        price1 = self.get_price(pos['symbol1'])
        price2 = self.get_price(pos['symbol2'])
        
        if price1 <= 0 or price2 <= 0:
            return True  # Keep position (price fetch error)
        
        # Calculate current z-score — prefer Kalman adaptive if available
        kf = self._slot_kalman.get(slot_id)
        kf_state: KalmanState | None = None
        if kf is not None:
            try:
                kf_state = kf.update(np.log(price1), np.log(price2))
                z_score = kf_state.z_score
            except Exception as e:
                logger.warning(f"Slot {slot_id}: KF update failed: {e}, falling back")
                kf_state = None

        if kf_state is None:
            z_score = self.calculate_zscore(
                price1, price2, pos['hedge_ratio'],
                pos['spread_mean'], pos['spread_std']
            )
        
        signal = pos['signal']
        entry_z = pos['entry_zscore']
        entry_time = self.slot_entry_times[slot_id]
        elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

        kf_info = ""
        if kf_state is not None:
            kf_info = (
                f", KF[gamma={kf_state.gamma:.4f}, "
                f"vol_r={kf_state.vol_ratio:.2f}]"
            )
        logger.debug(
            f"Slot {slot_id}: Monitor {pos['symbol1']}/{pos['symbol2']}: "
            f"Z={z_score:.4f} (entry={entry_z:.4f}), "
            f"Elapsed={elapsed:.1f}h{kf_info}"
        )
        
        should_close = False
        close_reason = ""

        # Determine effective exit threshold
        effective_exit_z = self._get_effective_exit_zscore(pos)
        
        # Exit conditions
        if signal == 1:  # Long spread
            if z_score >= -effective_exit_z:
                should_close = True
                close_reason = "mean_reversion"
            elif z_score < entry_z - self.config['stop_loss_zscore']:
                should_close = True
                close_reason = "stop_loss"
        else:  # Short spread
            if z_score <= effective_exit_z:
                should_close = True
                close_reason = "mean_reversion"
            elif z_score > entry_z + self.config['stop_loss_zscore']:
                should_close = True
                close_reason = "stop_loss"
        
        # Max holding time
        if elapsed >= self.config['max_holding_hours']:
            should_close = True
            close_reason = "max_holding_time"
        
        if should_close:
            logger.info(f"Slot {slot_id}: Closing position: {close_reason}")
            
            # Close positions FIRST to get actual fill prices
            exit_prices = self.close_slot_positions(slot_id)
            exit_p1 = exit_prices['exit_price1']
            exit_p2 = exit_prices['exit_price2']

            # Fallback to mark price if fill price unavailable (paper mode or close failure)
            if exit_p1 <= 0:
                exit_p1 = price1
            if exit_p2 <= 0:
                exit_p2 = price2

            # Calculate PnL using actual fill prices
            pnl1 = (exit_p1 - pos['entry_price1']) * pos['qty1']
            pnl2 = (exit_p2 - pos['entry_price2']) * pos['qty2']
            raw_pnl = pnl1 + pnl2
            
            # Deduct realistic costs
            entry_value = abs(pos['qty1'] * pos['entry_price1']) + abs(pos['qty2'] * pos['entry_price2'])
            exit_value = abs(pos['qty1'] * exit_p1) + abs(pos['qty2'] * exit_p2)
            fee_cost = (entry_value + exit_value) * self.config['trading_fee']  # entry + exit
            # Paper: simulate slippage as conservative estimate.
            # Live: entry/exit fill prices already include real slippage.
            slip_cost = (
                (entry_value + exit_value) * self.config['slippage']
                if self.paper_mode
                else 0.0
            )
            # Funding cost: fetch actual rates from Binance, fallback to config
            funding_cost = self._calculate_funding_cost(
                pos['symbol1'], pos['symbol2'], entry_value, elapsed,
            )
            total_cost = fee_cost + slip_cost + funding_cost
            total_pnl = raw_pnl - total_cost
            
            # Calculate PnL percentage (net of costs)
            pnl_pct = (total_pnl / entry_value * 100) if entry_value > 0 else 0
            
            # Record trade
            self.trade_history.append({
                'slot_id': slot_id,
                'symbol1': pos['symbol1'],
                'symbol2': pos['symbol2'],
                'entry_time': entry_time,
                'exit_time': datetime.now(timezone.utc),
                'entry_zscore': entry_z,
                'exit_zscore': z_score,
                'pnl': total_pnl,
                'pnl_pct': pnl_pct,
                'raw_pnl': raw_pnl,
                'fee_cost': fee_cost,
                'funding_cost': funding_cost,
                'slip_cost': slip_cost,
                'reason': close_reason,
            })
            
            logger.info(
                f"Slot {slot_id}: Trade closed: PnL=${total_pnl:.2f} ({pnl_pct:+.2f}%), "
                f"Reason={close_reason}, "
                f"costs=[fee=${fee_cost:.2f}, fund=${funding_cost:.2f}, slip=${slip_cost:.2f}], "
                f"raw=${raw_pnl:.2f}, "
                f"entry_p=({pos['entry_price1']:.4f},{pos['entry_price2']:.4f}), "
                f"exit_p=({exit_p1:.4f},{exit_p2:.4f})"
            )
            
            # Discord 알림
            elapsed_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            cumulative = sum(t['pnl'] for t in self.trade_history)
            self.discord.notify_close(
                pos['symbol1'], pos['symbol2'], total_pnl, pnl_pct, close_reason,
                slot_id=slot_id,
                holding_time=f'{elapsed_hours:.1f}h',
                cumulative_pnl=cumulative,
                balance_after=self.get_available_balance(),
            )
            
            return False
        
        return True  # Keep position
    
    # ==================== Helpers ====================

    def _build_slots_info(self) -> list[dict]:
        """슬롯 상태 정보 리스트 생성 (Discord 알림용).

        Returns:
            슬롯별 상태 딕셔너리 리스트.
        """
        info: list[dict] = []
        for slot in self.slots:
            if slot.position_open and slot.slot_id in self.slot_positions:
                pos = self.slot_positions[slot.slot_id]
                entry_time = self.slot_entry_times.get(slot.slot_id)
                hours = 0.0
                if entry_time:
                    hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

                # Unrealized PnL
                try:
                    p1 = self.get_price(pos['symbol1'])
                    p2 = self.get_price(pos['symbol2'])
                    pnl = (p1 - pos['entry_price1']) * pos['qty1'] + \
                          (p2 - pos['entry_price2']) * pos['qty2']
                except Exception:
                    pnl = 0.0

                # Current z-score (prefer KF adaptive z-score)
                try:
                    kf = self._slot_kalman.get(slot.slot_id)
                    if kf is not None:
                        kf_state = kf.update(np.log(p1), np.log(p2))
                        current_z = kf_state.z_score
                    else:
                        current_z = self.calculate_zscore(
                            p1, p2, pos['hedge_ratio'],
                            pos['spread_mean'], pos['spread_std'],
                        )
                except Exception:
                    current_z = 0.0

                info.append({
                    'slot_id': slot.slot_id,
                    'status': 'active',
                    'pair': f"{pos['symbol1']}/{pos['symbol2']}",
                    'unrealized_pnl': pnl,
                    'holding_hours': hours,
                    'entry_z': pos.get('entry_zscore', 0.0),
                    'current_z': current_z,
                })
            else:
                info.append({
                    'slot_id': slot.slot_id,
                    'status': 'idle',
                })
        return info

    # ==================== Main Loop ====================
    
    def run_cycle(self):
        """단일 트레이딩 사이클: 모든 슬롯 모니터링 + 빈 슬롯에 새 트레이드 탐색.

        비차단(non-blocking) 방식으로 모든 활성 슬롯을 순회하여 청산 조건을 확인하고,
        유휴 슬롯이 있으면 새로운 페어를 탐색하여 진입한다.

        Returns:
            True if 사이클이 정상 완료됨.
        """
        # 0. Discord 봇 명령 처리
        self._process_bot_commands()
        if self._stop_requested:
            return False  # run_continuous에서 루프 탈출

        active = get_active_slots(self.slots)
        idle = get_idle_slots(self.slots)
        logger.debug(
            f"Cycle: {len(active)} active, {len(idle)} idle "
            f"(occupied: {self.occupied_symbols or 'none'})"
        )

        # 사이클 내 이벤트 추적 (포지션 열림/닫힘)
        slots_before = {s.slot_id for s in active}

        # 1. Monitor all active slots
        for slot in active:
            self.monitor_slot(slot.slot_id)

        # 2. Refresh idle slot list (some may have closed above)
        idle = get_idle_slots(self.slots)
        if not idle:
            logger.info("All slots occupied, waiting...")
            if not self._all_slots_occupied_notified:
                self.discord.notify_all_slots_occupied(self._build_slots_info())
                self._all_slots_occupied_notified = True
            return True

        # 슬롯이 비었으므로 다음 점유 시 알림 재발송 허용
        self._all_slots_occupied_notified = False

        # 2.5. Paused 상태면 신규 진입 건너뛰기
        if self.paused:
            logger.info("Trading paused — monitoring only, skipping new entries")
            return True

        # 3. Get capital info (matches backtest pool model)
        available = self.get_available_balance()
        total = self.get_total_balance()
        logger.debug(f"Available balance: ${available:.2f}, Total: ${total:.2f}")

        if available < 100:
            logger.warning("Insufficient balance for new trades")
            if not self._insufficient_balance_notified:
                self.discord.notify_insufficient_balance(available, required=100.0)
                self._insufficient_balance_notified = True
            return True

        # 잔고 복구 시 다음 부족 알림 재발송 허용
        self._insufficient_balance_notified = False

        # 4. Get trading coins and download data
        coins = self.get_trading_coins()
        if not coins:
            logger.error("Failed to get trading coins")
            return True

        data = self.download_multi_symbol_data(coins)
        if len(data) < 4:
            logger.error("Not enough data for pair selection")
            return True

        # 5. select_pairs를 1회만 호출하고 결과를 캐싱
        #    모든 idle slot이 동일한 candidates를 재사용한다.
        #    (이전: idle slot마다 select_pairs 재호출 → 동일 결과에 ~9초씩 낭비)
        logger.info("Computing pair candidates (once per cycle)...")
        cached_candidates = self.pair_selector.select_pairs(data, progress=False)
        if not cached_candidates:
            logger.warning("No valid pair candidates found")
            return True
        logger.info(f"Found {len(cached_candidates)} pair candidates")

        # 6. For each idle slot, find and enter a new pair
        #    Capital is calculated per-slot using the same function as backtest
        self._clean_expired_failures()
        excluded = set(self.occupied_symbols) | set(self._failed_symbols.keys())
        if self._failed_symbols:
            logger.info(f"Blacklisted symbols ({len(self._failed_symbols)}): {set(self._failed_symbols.keys())}")
        for slot in idle:
            # Pool-based capital allocation (identical to backtest)
            # Returns notional (leveraged) amount
            slot_capital = calculate_slot_capital(
                available_capital=self.get_available_balance(),
                active_slots=get_active_slots(self.slots),
                max_traders=self.max_traders,
                position_size_pct=self.config['position_size_pct'],
                leverage=self.config['leverage'],
            )
            if slot_capital < 10:
                logger.info(f"Slot {slot.slot_id}: Insufficient slot capital (${slot_capital:.2f})")
                break

            result = self.find_best_pair(
                data, excluded_symbols=excluded,
                cached_candidates=cached_candidates,
            )
            if not result:
                logger.info(f"Slot {slot.slot_id}: No suitable pair found")
                break  # 캐시된 candidates에서 못 찾으면 나머지도 없음

            pair, pair_info = result

            if self.execute_pair_trade(slot.slot_id, pair, pair_info, slot_capital):
                # Update excluded symbols for next idle slot
                excluded = set(self.occupied_symbols)
                # Short delay between orders to not spam the API
                time.sleep(2)
            else:
                # 실패한 페어의 심볼을 블랙리스트에 추가 (사이클 간 지속)
                now = datetime.now(timezone.utc)
                self._failed_symbols[pair.symbol1] = now
                self._failed_symbols[pair.symbol2] = now
                excluded.add(pair.symbol1)
                excluded.add(pair.symbol2)
                logger.warning(
                    f"Slot {slot.slot_id}: Trade execution failed for "
                    f"{pair.symbol1}/{pair.symbol2}, blacklisted for "
                    f"{self.BLACKLIST_TTL_SECONDS}s"
                )

        logger.debug("Cycle completed")

        # ── Cycle Summary: 포지션 변동이 있을 때만 알림 ──
        slots_after = {s.slot_id for s in get_active_slots(self.slots)}
        positions_changed = (slots_before != slots_after)

        if positions_changed:
            slots_info = self._build_slots_info()
            available = self.get_available_balance()
            total = self.get_total_balance()
            total_unrealized = sum(
                s.get('unrealized_pnl', 0.0) for s in slots_info if s['status'] == 'active'
            )
            session_pnl = sum(t['pnl'] for t in self.trade_history) if self.trade_history else 0.0
            self.discord.notify_cycle_summary(
                slots_info=slots_info,
                available_capital=available,
                total_capital=total,
                total_unrealized_pnl=total_unrealized,
                session_pnl=session_pnl,
                n_trades_session=len(self.trade_history),
            )

        # ── Daily Report (24h 주기) ──
        if self.discord.should_send_daily_report():
            total_bal = self.get_total_balance()
            session_pnl_dr = sum(t['pnl'] for t in self.trade_history) if self.trade_history else 0.0
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            win_rate = (len(wins) / len(self.trade_history) * 100) if self.trade_history else 0.0
            active_count = len(get_active_slots(self.slots))
            self.discord.notify_daily_report(
                total_balance=total_bal,
                daily_pnl=session_pnl_dr,
                daily_return_pct=(session_pnl_dr / total_bal * 100) if total_bal > 0 else 0.0,
                cumulative_pnl=session_pnl_dr,
                cumulative_return_pct=(session_pnl_dr / total_bal * 100) if total_bal > 0 else 0.0,
                n_trades_today=len(self.trade_history),
                n_trades_total=len(self.trade_history),
                win_rate=win_rate,
                max_drawdown_pct=0.0,  # 라이브에선 별도 DD 트래킹 필요
                active_slots=active_count,
                max_slots=self.max_traders,
                leverage=self.config['leverage'],
            )

        return True

    # ==================== Discord Bot Command Processing ====================

    def _interruptible_sleep(self, total_seconds: int):
        """짧은 간격으로 sleep하면서 봇 명령을 중간중간 처리한다.

        Args:
            total_seconds: 총 sleep 시간 (초)
        """
        poll_interval = 2  # 2초마다 큐 체크
        elapsed = 0
        while elapsed < total_seconds:
            time.sleep(min(poll_interval, total_seconds - elapsed))
            elapsed += poll_interval
            self._process_bot_commands()
            if self._stop_requested:
                return

    def _process_bot_commands(self):
        """Discord 봇 큐에 쌓인 명령을 모두 처리한다.

        매 run_cycle() 시작 시 호출된다.
        각 명령의 결과를 set_response()로 Discord 채널에 응답한다.
        """
        if not self._bot_runner:
            return

        for cmd in self._bot_runner.get_all_commands():
            try:
                response = self._handle_command(cmd)
                cmd.set_response(response)
            except Exception as e:
                logger.error(f"Bot command error: {e}")
                cmd.set_response(f"❌ 명령 처리 오류: {e}")

    def _handle_command(self, cmd: BotCommand) -> str:
        """개별 명령을 처리하고 응답 문자열을 반환한다.

        Args:
            cmd: 처리할 BotCommand.

        Returns:
            Discord에 보낼 응답 문자열.
        """
        if cmd.command == CommandType.STATUS:
            return self._cmd_status()
        elif cmd.command == CommandType.CLOSE_ALL:
            return self._cmd_close_all()
        elif cmd.command == CommandType.CLOSE_SLOT:
            slot_id = cmd.args['slot_id']
            return self._cmd_close_slot(slot_id)
        elif cmd.command == CommandType.PAUSE:
            return self._cmd_pause()
        elif cmd.command == CommandType.RESUME:
            return self._cmd_resume()
        elif cmd.command == CommandType.STOP:
            return self._cmd_stop()
        elif cmd.command == CommandType.PNL:
            return self._cmd_pnl()
        elif cmd.command == CommandType.BLACKLIST:
            if cmd.args and cmd.args.get('action') == 'clear':
                return self._cmd_blacklist_clear()
            return self._cmd_blacklist()
        else:
            return "❓ 알 수 없는 명령"

    def _cmd_status(self) -> str:
        """슬롯 상태 + 잔고 요약."""
        active = get_active_slots(self.slots)
        idle = get_idle_slots(self.slots)
        available = self.get_available_balance()
        total = self.get_total_balance()
        paused_str = " ⏸️ **PAUSED**" if self.paused else ""

        lines = [f"📊 **QuantBox Status**{paused_str}"]
        lines.append(f"잔고: ${available:,.2f} / ${total:,.2f}")
        lines.append(f"슬롯: {len(active)} active / {len(idle)} idle")
        lines.append("")

        for slot in self.slots:
            if slot.position_open and slot.current_pair:
                pos = self.slot_positions.get(slot.slot_id, {})
                z = pos.get('entry_zscore', 0)
                capital = pos.get('capital', 0)
                sym1, sym2 = slot.current_pair
                lines.append(
                    f"  `[{slot.slot_id}]` {sym1}/{sym2} "
                    f"| z={z:.2f} | ${capital:.0f}"
                )
            else:
                lines.append(f"  `[{slot.slot_id}]` — idle")

        return "\n".join(lines)

    def _cmd_close_all(self) -> str:
        """전체 포지션 청산."""
        active = get_active_slots(self.slots)
        if not active:
            return "ℹ️ 청산할 포지션 없음"

        n = len(active)
        self.close_all_positions()
        return f"✅ {n}개 슬롯 전체 청산 완료"

    def _cmd_close_slot(self, slot_id: int) -> str:
        """특정 슬롯 청산."""
        if slot_id < 0 or slot_id >= self.max_traders:
            return f"❌ 유효하지 않은 슬롯 번호: {slot_id} (0~{self.max_traders - 1})"

        slot = self.slots[slot_id]
        if not slot.position_open:
            return f"ℹ️ 슬롯 {slot_id}는 이미 비어있음"

        pair = slot.current_pair
        self.close_slot_positions(slot_id)
        return f"✅ 슬롯 {slot_id} 청산 완료 ({pair[0]}/{pair[1]})"

    def _cmd_pause(self) -> str:
        """신규 진입 중지."""
        if self.paused:
            return "ℹ️ 이미 일시정지 상태"
        self.paused = True
        logger.info("Trading PAUSED via Discord")
        return "⏸️ **신규 진입 중지** — 기존 포지션 모니터링만 계속"

    def _cmd_resume(self) -> str:
        """신규 진입 재개."""
        if not self.paused:
            return "ℹ️ 이미 활성 상태"
        self.paused = False
        logger.info("Trading RESUMED via Discord")
        return "▶️ **트레이딩 재개** — 신규 진입 활성화"

    def _cmd_stop(self) -> str:
        """전체 청산 후 종료 요청."""
        self._stop_requested = True
        logger.info("STOP requested via Discord")
        return "🛑 **종료 요청 접수** — 현재 사이클 종료 후 전체 청산 및 프로세스 종료"

    def _cmd_blacklist(self) -> str:
        """블랙리스트 현황 조회."""
        if not self._failed_symbols:
            return "✅ 블랙리스트 비어있음"

        now = datetime.now(timezone.utc)
        lines = [f"🚫 **블랙리스트** ({len(self._failed_symbols)}개 심볼)"]
        for sym, ts in sorted(self._failed_symbols.items()):
            elapsed = (now - ts).total_seconds()
            remaining = max(0, self.BLACKLIST_TTL_SECONDS - elapsed)
            lines.append(f"  `{sym}` — {remaining:.0f}초 남음")
        return "\n".join(lines)

    def _cmd_blacklist_clear(self) -> str:
        """블랙리스트 수동 초기화."""
        n = len(self._failed_symbols)
        self._failed_symbols.clear()
        logger.info("Blacklist cleared via Discord command")
        return f"✅ 블랙리스트 초기화 ({n}개 심볼 제거)"

    def _cmd_pnl(self) -> str:
        """세션 손익 요약."""
        if not self.trade_history:
            return "ℹ️ 아직 완료된 트레이드 없음"

        total_pnl = sum(t['pnl'] for t in self.trade_history)
        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] <= 0]
        win_rate = len(wins) / len(self.trade_history) * 100

        lines = ["📈 **세션 손익 요약**"]
        lines.append(f"총 트레이드: {len(self.trade_history)}")
        lines.append(f"총 손익: **${total_pnl:+.2f}**")
        lines.append(f"승률: {win_rate:.0f}% ({len(wins)}W / {len(losses)}L)")

        if wins:
            best = max(t['pnl'] for t in wins)
            lines.append(f"최대 수익: ${best:+.2f}")
        if losses:
            worst = min(t['pnl'] for t in losses)
            lines.append(f"최대 손실: ${worst:+.2f}")

        return "\n".join(lines)

    def _clean_expired_failures(self):
        """TTL이 지난 실패 심볼을 블랙리스트에서 제거한다."""
        if not self._failed_symbols:
            return
        now = datetime.now(timezone.utc)
        expired = [
            sym for sym, ts in self._failed_symbols.items()
            if (now - ts).total_seconds() > self.BLACKLIST_TTL_SECONDS
        ]
        for sym in expired:
            del self._failed_symbols[sym]
            logger.info(f"Blacklist expired: {sym}")

    def run_once(self):
        """레거시 호환: 하나의 트레이드 사이클 (모든 슬롯)."""
        return self.run_cycle()
    
    def run_continuous(self):
        """연속 트레이딩 루프 - 멀티 트레이더 모드.

        매 check_interval마다 모든 슬롯을 순회하며:
        1. 활성 슬롯: 청산 조건 확인
        2. 유휴 슬롯: 새 페어 탐색 및 진입
        """
        logger.info(f"Starting continuous trading loop (max {self.max_traders} slots)...")
        self.print_balance()

        # Discord Bot 시작 (토큰이 설정되어 있을 때만)
        if self._bot_runner:
            self._bot_runner.start()
            logger.info("Discord Bot started — remote commands enabled")

        # Discord 시작 알림
        balance = self.get_available_balance()
        mode = "PAPER" if self.paper_mode else "LIVE"
        self.discord.notify_start(
            mode, balance,
            leverage=self.config['leverage'],
            max_traders=self.max_traders,
        )
        
        try:
            while True:
                try:
                    result = self.run_cycle()
                    if result is False:  # !stop 명령
                        logger.info("Stop requested via Discord")
                        break
                    self._interruptible_sleep(self.config['check_interval_seconds'])
                    if self._stop_requested:
                        logger.info("Stop requested via Discord (during sleep)")
                        break
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    self.discord.notify_error(str(e))
                    # 에러 시 모든 포지션 청산 (안전 조치)
                    self.close_all_positions()
                    self._interruptible_sleep(60)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        # 종료 처리
        total_pnl = sum(t['pnl'] for t in self.trade_history) if self.trade_history else 0.0
        reason = 'Discord !stop' if self._stop_requested else '사용자 종료'
        self.discord.notify_stop(
            reason=reason,
            total_pnl=total_pnl,
            n_trades=len(self.trade_history),
        )
        self.close_all_positions()
        self.print_summary()

        if self._bot_runner:
            self._bot_runner.stop()
    
    def print_balance(self):
        """잔고 출력"""
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b['asset'] == 'USDT':
                    logger.info("=" * 50)
                    logger.info("Futures Account Balance:")
                    logger.info(f"  Available: ${float(b['availableBalance']):.2f}")
                    logger.info(f"  Total: ${float(b['balance']):.2f}")
                    logger.info(f"  Unrealized PnL: ${float(b['crossUnPnl']):.2f}")
                    logger.info("=" * 50)
                    break
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
    
    def print_summary(self):
        """트레이딩 요약 출력"""
        logger.info("=" * 60)
        logger.info("Trading Summary")
        logger.info(f"Total trades: {len(self.trade_history)}")
        
        if self.trade_history:
            total_pnl = sum(t['pnl'] for t in self.trade_history)
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            win_rate = len(wins) / len(self.trade_history) * 100
            
            logger.info(f"Total PnL: ${total_pnl:.2f}")
            logger.info(f"Win Rate: {win_rate:.1f}%")
            logger.info(f"Winning: {len(wins)}, Losing: {len(self.trade_history) - len(wins)}")


def main():
    parser = argparse.ArgumentParser(description='QuantBox Live Trader')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()
    
    mode = "PAPER" if args.paper else "LIVE"
    logger.info(f"Starting QuantBox Live Trader in {mode} mode")
    
    if not args.paper:
        logger.warning("=" * 60)
        logger.warning("WARNING: LIVE TRADING MODE")
        logger.warning("Real money will be used!")
        logger.warning("=" * 60)
        
        response = input("Type 'YES' to confirm: ")
        if response != 'YES':
            logger.info("Cancelled by user")
            return
    
    trader = LivePairTrader(paper_mode=args.paper)
    
    if args.once:
        trader.run_once()
    else:
        trader.run_continuous()


if __name__ == '__main__':
    main()
