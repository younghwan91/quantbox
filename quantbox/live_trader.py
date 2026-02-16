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
import logging.handlers
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# QuantBox imports
from quantbox.core.settings import get_settings
from quantbox.core.trader_slot import (
    TraderSlotState,
    get_occupied_symbols,
    get_idle_slots,
    get_active_slots,
    calculate_slot_capital,
    calculate_margin_from_notional,
    is_pair_available,
    calculate_dynamic_leverage,
    estimate_cost_rate,
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
from quantbox.core.state_persistence import (
    save_state, load_state, clear_state,
    deserialize_entry_times, deserialize_positions,
)

# Load environment variables
load_dotenv()

# Configure logging with session-based log directory
def _setup_session_logging() -> Path:
    """Create session-based log directory and configure logging.

    Each session gets a timestamped log file under logs/ directory:
        logs/session_20260214_153000.log
        logs/session_20260214_180000.log

    Returns:
        Path to the session log file.
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_log = log_dir / f'session_{session_ts}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                str(session_log), mode='a',
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=5,
            ),
        ]
    )
    return session_log


_session_log_path = _setup_session_logging()
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

        # 손실 페어 세션 블랙리스트 (세션 전체 지속, 재진입 방지)
        self._session_loss_blacklist: set[str] = set()

        # 일시정지 플래그 (Discord !pause / !resume)
        self.paused: bool = False
        # 종료 요청 플래그 (Discord !stop)
        self._stop_requested: bool = False
        # 알림 중복 방지 플래그 (상태 전이 시 1회만 발송)
        self._all_slots_occupied_notified: bool = False
        self._insufficient_balance_notified: bool = False

        # ── Drawdown & Daily Loss Tracking ──
        self._initial_balance: float = 0.0  # 세션 시작 잔고 (run_continuous에서 설정)
        self._session_high_water_mark: float = 0.0  # 세션 최고 잔고
        self._daily_start_balance: float = 0.0  # 당일 시작 잔고
        self._daily_start_time: datetime = datetime.now(timezone.utc)
        self._drawdown_triggered: bool = False  # DD 브레이커 발동 시 True

        # Cycle counter for periodic logging
        self._cycle_count: int = 0
        self._last_no_pair_reason: str = ""  # Suppress duplicate "no pair" logs

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"LivePairTrader initialized in {mode_str} mode")
        logger.info(f"Max concurrent traders: {self.max_traders}")
        logger.info(f"Leverage: {self.config['leverage']}x")
        logger.info(f"Testnet: {self._settings.binance_testnet}")
        if paper_mode:
            logger.info(f"Paper Balance: ${self.paper_balance:.2f} USDT")

        # ── Session Config Dump (파라미터 최적화 비교용) ──
        c = self.config
        logger.info(
            f"[Config] entry_z={c.get('entry_zscore', '?')}, exit_z={c.get('exit_zscore', '?')}, "
            f"stop_z={c.get('stop_loss_zscore', '?')}, max_hold={c.get('max_holding_hours', '?')}h, "
            f"n_coins={c.get('n_coins', '?')}, pos_pct={c.get('position_size_pct', 0):.2f}"
        )
        logger.info(
            f"[Config] dynamic_exit={c.get('use_dynamic_exit', False)}, "
            f"base={c.get('dynamic_exit_base', '-')}, range={c.get('dynamic_exit_range', '-')} | "
            f"z_cap={c.get('z_score_cap', 4.0)}, vol_gate={c.get('vol_expansion_gate', 2.0)}"
        )
        logger.info(
            f"[Config] target_r={c.get('target_r_pct', 0)*100:.1f}%, "
            f"max_lev={c.get('max_leverage', 10)}x, "
            f"fee={c.get('trading_fee', '?')}, slip={c.get('slippage', '?')}, "
            f"funding={c.get('funding_rate_8h', '?')}"
        )
        logger.info(
            f"[Config] filters: corr>={c.get('min_correlation', 0.6)}, "
            f"coint_p<={c.get('max_cointegration_pvalue', 0.05)}, "
            f"hl=[{c.get('min_half_life', 5)},{c.get('max_half_life', 40)}], "
            f"hurst<={c.get('max_hurst', 0.5)}"
        )
        
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
    
    def check_btc_crash_gate(self) -> tuple[bool, float]:
        """BTC 급락/급등 게이트: 최근 N봉 수익률이 임계값을 초과하면 진입 차단.

        매크로 위험(시장 전체 상관관계 급등) 시 모든 알트 페어가 동시에
        깨지는 것을 사전 차단하는 레이어.

        Returns:
            (is_blocked, btc_return):
                is_blocked — True이면 진입 차단해야 함.
                btc_return — 측정된 BTC 수익률 (로깅/모니터링용).
        """
        if not self.config.get('btc_crash_gate_enabled', True):
            return False, 0.0

        window = self.config.get('btc_crash_window', 8)
        threshold = self.config.get('btc_crash_threshold', 0.03)

        try:
            klines = self.client.futures_klines(
                symbol='BTCUSDT',
                interval=self.config.get('interval', '15m'),
                limit=window + 1,  # +1: 수익률 계산에 시작점 필요
            )
            if not klines or len(klines) < 2:
                logger.warning("BTC crash gate: 캔들 데이터 부족, gate 통과")
                return False, 0.0

            # 가장 오래된 봉의 종가 vs 최신 봉의 종가 (backtest와 동일: Close-to-Close)
            open_price = float(klines[0][4])   # Close of oldest candle
            close_price = float(klines[-1][4])  # Close of newest candle

            if open_price <= 0:
                return False, 0.0

            btc_return = (close_price - open_price) / open_price

            if abs(btc_return) > threshold:
                logger.warning(
                    f"BTC crash gate BLOCKED: BTC {btc_return:+.2%} "
                    f"over {window} candles (threshold ±{threshold:.1%})"
                )
                return True, btc_return

            logger.debug(
                f"BTC crash gate OK: BTC {btc_return:+.2%} "
                f"(threshold ±{threshold:.1%})"
            )
            return False, btc_return

        except Exception as e:
            logger.error(f"BTC crash gate error: {e} — gate 통과 (fail-open)")
            return False, 0.0

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
    
    def get_binance_unrealized_pnl(self) -> dict[str, float]:
        """바이낸스에서 심볼별 실제 미실현 PnL 조회.

        Returns:
            {symbol: unrealizedProfit} 딕셔너리. paper_mode 시 빈 딕셔너리.
        """
        if self.paper_mode:
            return {}
        try:
            account = self.client.futures_account()
            return {
                p['symbol']: float(p['unrealizedProfit'])
                for p in account['positions']
                if float(p['positionAmt']) != 0.0
            }
        except Exception as e:
            logger.debug(f"Failed to get Binance uPnL: {e}")
            return {}

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
        quantity: float,
        leverage: float = 0.0,
    ) -> Optional[Dict]:
        """주문 실행 — LimitOrderChaser를 통한 maker-only 체결.

        Limit 지정가를 best bid/ask에 걸고, 미체결 시 가격을 추격한다.
        추격 실패 시 시장가(MARKET)로 fallback하여 체결을 보장한다.

        Args:
            symbol: 거래 심볼.
            side: 'BUY' 또는 'SELL'.
            quantity: 주문 수량.
            leverage: 동적 레버리지 (0이면 config fallback).

        Returns:
            체결 정보 dict 또는 None (실패 시).
        """
        try:
            # Set leverage (live mode only, entry orders)
            # Skip for close orders (leverage=0) to avoid margin requirement changes
            import math
            if leverage > 0 and not self.paper_mode:
                exchange_leverage = int(math.ceil(leverage))
                try:
                    self.client.futures_change_leverage(
                        symbol=symbol,
                        leverage=exchange_leverage,
                    )
                except Exception as lev_err:
                    err_msg = str(lev_err)
                    # Binance returns max allowed leverage in error message
                    # e.g. "leverage 36 is not valid" or contains max leverage info
                    import re
                    max_match = re.search(r'Max\s+leverage\s+(\d+)', err_msg, re.IGNORECASE)
                    if not max_match:
                        max_match = re.search(r'maximum.*?(\d+)', err_msg, re.IGNORECASE)
                    if max_match:
                        exchange_max = int(max_match.group(1))
                        logger.warning(
                            f"[{symbol}] Leverage {exchange_leverage}x exceeds exchange max "
                            f"{exchange_max}x — using {exchange_max}x"
                        )
                        self.client.futures_change_leverage(
                            symbol=symbol,
                            leverage=exchange_max,
                        )
                    else:
                        logger.warning(
                            f"[{symbol}] Failed to set leverage {exchange_leverage}x: {err_msg} "
                            f"— retrying with 20x fallback"
                        )
                        try:
                            self.client.futures_change_leverage(
                                symbol=symbol, leverage=20,
                            )
                        except Exception:
                            logger.error(f"[{symbol}] Leverage fallback also failed")
                            return None

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
            # 청산 실패 시 슬롯을 리셋하지 않는다.
            # 리셋하면 봇이 거래소 포지션을 잊어버리고, 다음 사이클에서
            # 유령 포지션이 방치된다. 슬롯은 점유 상태로 유지하여
            # 다음 monitor_slot 사이클에서 재시도할 수 있도록 한다.
            return exit_prices

        # Reset slot only if BOTH legs closed successfully
        self.slots[slot_id].reset()
        del self.slot_positions[slot_id]
        if slot_id in self.slot_entry_times:
            del self.slot_entry_times[slot_id]

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
            # 왜 진입 기준 미달인지 로깅 (best candidate의 z-score 표시)
            if candidates:
                best_z = max(candidates, key=lambda c: abs(c.current_z_score))
                logger.info(
                    f"No pairs meet entry criteria "
                    f"(best |z|={abs(best_z.current_z_score):.2f} < threshold {self.config['entry_zscore']}, "
                    f"{len(candidates)} candidates)"
                )
            else:
                logger.info("No pairs meet entry criteria (0 candidates)")
            return None

        # Find best pair that doesn't overlap with excluded symbols
        best = None
        overlap_count = 0
        for candidate in tradeable:
            if is_pair_available(
                (candidate.symbol1, candidate.symbol2), excluded_symbols
            ):
                best = candidate
                break
            else:
                overlap_count += 1

        if best is None:
            # 중복 스킵은 첫 발생 시에만 INFO, 반복 시 DEBUG
            if self._last_no_pair_reason != "overlap":
                logger.info(
                    f"No pairs available ({overlap_count} tradeable, all overlap with active positions)"
                )
                self._last_no_pair_reason = "overlap"
            else:
                logger.debug(
                    f"No pairs available ({overlap_count} tradeable, all overlap)"
                )
            return None

        self._last_no_pair_reason = ""  # Reset suppression

        # Top-3 tradeable candidates 로깅 (파라미터 최적화 비교용)
        top3 = tradeable[:3]
        for i, c in enumerate(top3):
            avail = "✓" if is_pair_available((c.symbol1, c.symbol2), excluded_symbols) else "✗overlap"
            logger.info(
                f"  #{i+1} {c.symbol1}/{c.symbol2}: "
                f"z={c.current_z_score:+.2f}, score={c.combined_score:.4f}, "
                f"hl={c.half_life:.1f}, hr={c.hedge_ratio:.3f}, "
                f"corr={c.correlation:.3f}, hurst={c.hurst:.3f} [{avail}]"
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
        capital: float,
        anchor_std: float = 0.0,
        leverage: float = 0.0,
    ) -> bool:
        """특정 슬롯에서 페어 트레이드 실행.

        capital은 calculate_slot_capital()에서 반환된 값으로,
        leverage가 적용된 notional exposure이다.
        실제 마진(cash)은 capital / leverage.

        Vol-inverse sizing: anchor_std > 0이면 capital을
        anchor_std / spread_std 비율로 스케일 다운하여
        모든 페어의 손절 dollar loss를 동일하게 만든다.

        Args:
            slot_id: 트레이드를 실행할 슬롯 ID.
            pair: 선택된 페어 후보.
            pair_info: 헤지 비율, 스프레드 통계 등.
            capital: 레버리지 적용된 노셔널 (calculate_slot_capital 반환값).
            anchor_std: 최소 spread_std (vol-inverse sizing 기준값).
            leverage: 동적 레버리지 (0이면 config fallback).

        Returns:
            True if 포지션 진입 성공.
        """
        symbol1, symbol2 = pair.symbol1, pair.symbol2
        hedge_ratio = pair_info['hedge_ratio']
        if leverage <= 0:
            leverage = float(self.config['leverage'])

        # ── BTC Crash Gate (매크로 방어, 최우선) ──
        btc_blocked, btc_ret = self.check_btc_crash_gate()
        if btc_blocked:
            logger.warning(
                f"Slot {slot_id}: BTC crash gate blocked entry "
                f"(BTC {btc_ret:+.2%}) — skipping {symbol1}/{symbol2}"
            )
            return False
        
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
        vol_ratio_val = 0.0  # for gate summary

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
                vol_ratio_val = recent_std / formation_std if formation_std > 0 else 999
                if vol_ratio_val > vol_gate:
                    logger.warning(
                        f"Slot {slot_id}: Vol expansion {vol_ratio_val:.2f}x "
                        f"(recent_std={recent_std:.6f}, formation={formation_std:.6f}) "
                        f"> gate {vol_gate:.1f}x — skipping entry"
                    )
                    return False

        if abs(z_score) < self.config['entry_zscore']:
            logger.info(f"Slot {slot_id}: Z-Score below entry threshold")
            return False

        # ── Gate Summary (모든 게이트 통과 시 한 줄로 기록) ──
        logger.info(
            f"Slot {slot_id}: Gates PASS — BTC={btc_ret:+.2%}, "
            f"|z|={abs(z_score):.2f}/{z_cap}, "
            f"vol={vol_ratio_val:.2f}x/{vol_gate:.1f}x"
        )
        
        # Determine signal: z < 0 -> Long spread, z > 0 -> Short spread
        signal = 1 if z_score < 0 else -1
        side1 = 'BUY' if signal == 1 else 'SELL'
        side2 = 'SELL' if signal == 1 else 'BUY'
        
        # capital is notional (leveraged). Use full notional for position sizing.
        # ── Vol-inverse sizing (risk parity) ──
        # When dynamic leverage is ON (target_r_pct > 0), risk parity is achieved
        # through leverage variation, NOT notional reduction → full capital utilisation.
        # Vol-inverse notional scaling only applies when dynamic leverage is OFF.
        spread_std = pair_info['spread_std']
        target_r = self.config.get('target_r_pct', 0)
        if target_r <= 0 and anchor_std > 0 and spread_std > 0:
            vol_ratio = min(anchor_std / spread_std, 1.0)  # anchor=min → ratio ≤ 1
            capital = capital * vol_ratio
            logger.info(
                f"Slot {slot_id}: Vol-inverse sizing: "
                f"anchor_std={anchor_std:.6f}, spread_std={spread_std:.6f}, "
                f"ratio={vol_ratio:.3f}, adjusted_capital=${capital:.2f}"
            )

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
        order1 = self.place_order(symbol1, side1, qty1, leverage=leverage)
        order2 = None

        if order1:
            order2 = self.place_order(symbol2, side2, qty2, leverage=leverage)

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
            'best_z': z_score,           # Trailing stop: best z-score seen (closest to 0)
            'trailing_active': False,    # Trailing stop: activated when Δz >= threshold
        }

        # Update slot state
        slot = self.slots[slot_id]
        slot.position_open = True
        slot.current_pair = (symbol1, symbol2)
        slot.position_info = position_data
        self.slot_positions[slot_id] = position_data
        self.slot_entry_times[slot_id] = datetime.now(timezone.utc)

        # Paper mode: deduct margin from pool (parity with backtest)
        if self.paper_mode:
            self.paper_balance -= margin
            logger.debug(f"Slot {slot_id}: paper_balance -= ${margin:.2f} → ${self.paper_balance:.2f}")

        direction = "LONG Spread" if signal == 1 else "SHORT Spread"
        logger.info(f"Slot {slot_id}: Position opened: signal={signal}, z-score={z_score:.4f}")

        # 상태 저장 (크래시 복구용)
        self._save_state()
        
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
        
        # Calculate current z-score using frozen formation statistics
        z_score = self.calculate_zscore(
            price1, price2, pos['hedge_ratio'],
            pos['spread_mean'], pos['spread_std']
        )
        
        signal = pos['signal']
        entry_z = pos['entry_zscore']
        entry_time = self.slot_entry_times[slot_id]
        elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

        # Determine effective exit threshold
        effective_exit_z = self._get_effective_exit_zscore(pos)

        # 평균회귀/손절까지 거리 계산
        if signal == 1:  # Long spread
            dist_exit = z_score - (-effective_exit_z)  # z가 -exit_z 이상이면 청산
            dist_stop = (entry_z - self.config['stop_loss_zscore']) - z_score
        else:  # Short spread
            dist_exit = effective_exit_z - z_score  # z가 exit_z 이하이면 청산
            dist_stop = z_score - (entry_z + self.config['stop_loss_zscore'])

        time_remaining = max(0, self.config['max_holding_hours'] - elapsed)

        logger.debug(
            f"Slot {slot_id}: {pos['symbol1']}/{pos['symbol2']} "
            f"z={z_score:+.4f} (entry={entry_z:+.4f}), "
            f"exit_z={effective_exit_z:.2f}, "
            f"dist_exit={dist_exit:+.3f}, dist_stop={dist_stop:+.3f}, "
            f"elapsed={elapsed:.1f}h/{self.config['max_holding_hours']}h"
        )

        # 주기적 INFO 스냅샷: 15사이클마다 (≈15분) 활성 포지션 상태 기록
        if self._cycle_count % 15 == 0:
            # Unrealized PnL: 바이낸스 실제 값 우선, fallback으로 자체 계산
            try:
                binance_upnl = self.get_binance_unrealized_pnl()
                s1, s2 = pos['symbol1'], pos['symbol2']
                if binance_upnl and (s1 in binance_upnl or s2 in binance_upnl):
                    upnl = binance_upnl.get(s1, 0.0) + binance_upnl.get(s2, 0.0)
                else:
                    upnl = (price1 - pos['entry_price1']) * pos.get('qty1', 0) + \
                           (price2 - pos['entry_price2']) * pos.get('qty2', 0)
            except Exception:
                upnl = 0.0
            logger.info(
                f"Slot {slot_id}: [MONITOR] {pos['symbol1']}/{pos['symbol2']} "
                f"z={z_score:+.3f} (entry={entry_z:+.3f}), "
                f"exit_z={effective_exit_z:.2f}, "
                f"uPnL=${upnl:+.2f}, "
                f"elapsed={elapsed:.1f}h, remain={time_remaining:.1f}h"
            )

        should_close = False
        close_reason = ""
        
        # ── Trailing stop: update best_z ──
        if self.config.get('trailing_stop_enabled', False):
            best_z = pos.get('best_z', entry_z)
            if signal == 1:  # Long spread: z should increase (become less negative)
                if z_score > best_z:
                    pos['best_z'] = z_score
                    best_z = z_score
            else:  # Short spread: z should decrease (become less positive)
                if z_score < best_z:
                    pos['best_z'] = z_score
                    best_z = z_score
            
            # Check activation
            delta = abs(best_z - entry_z)
            activate_delta = self.config.get('trailing_stop_activate_delta', 0.3)
            if delta >= activate_delta:
                pos['trailing_active'] = True

        # Exit conditions (priority: mean_reversion > trailing_stop > stop_loss > max_holding)
        if signal == 1:  # Long spread
            if z_score >= -effective_exit_z:
                should_close = True
                close_reason = "mean_reversion"
            elif self.config.get('trailing_stop_enabled', False) and pos.get('trailing_active', False):
                trail_z = self.config.get('trailing_stop_trail_z', 0.8)
                best_z = pos.get('best_z', entry_z)
                threshold = best_z - trail_z
                if z_score <= threshold:
                    should_close = True
                    close_reason = "trailing_stop"
                    logger.info(
                        f"Slot {slot_id}: Trailing stop triggered: "
                        f"z={z_score:+.3f}, best_z={best_z:+.3f}, "
                        f"threshold={threshold:+.3f}"
                    )
            elif z_score < entry_z - self.config['stop_loss_zscore']:
                should_close = True
                close_reason = "stop_loss"
        else:  # Short spread
            if z_score <= effective_exit_z:
                should_close = True
                close_reason = "mean_reversion"
            elif self.config.get('trailing_stop_enabled', False) and pos.get('trailing_active', False):
                trail_z = self.config.get('trailing_stop_trail_z', 0.8)
                best_z = pos.get('best_z', entry_z)
                threshold = best_z + trail_z
                if z_score >= threshold:
                    should_close = True
                    close_reason = "trailing_stop"
                    logger.info(
                        f"Slot {slot_id}: Trailing stop triggered: "
                        f"z={z_score:+.3f}, best_z={best_z:+.3f}, "
                        f"threshold={threshold:+.3f}"
                    )
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
            
            # Calculate PnL percentage (net of costs, relative to MARGIN)
            # Matches backtest convention: return % = net_pnl / margin × 100
            margin = pos.get('capital', entry_value)
            pnl_pct = (total_pnl / margin * 100) if margin > 0 else 0
            
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
                f"held={elapsed:.1f}h, "
                f"z: {entry_z:+.3f}→{z_score:+.3f} (exit_z={effective_exit_z:.2f}), "
                f"costs=[fee=${fee_cost:.2f}, fund=${funding_cost:.2f}, slip=${slip_cost:.2f}], "
                f"raw=${raw_pnl:.2f}, "
                f"entry_p=({pos['entry_price1']:.4f},{pos['entry_price2']:.4f}), "
                f"exit_p=({exit_p1:.4f},{exit_p2:.4f})"
            )
            
            # Paper mode: return margin + PnL to pool (parity with backtest)
            if self.paper_mode:
                margin = pos.get('capital', 0)
                self.paper_balance += margin + total_pnl
                logger.debug(
                    f"Slot {slot_id}: paper_balance += ${margin + total_pnl:.2f} "
                    f"→ ${self.paper_balance:.2f}"
                )

            # 상태 저장 (크래시 복구용)
            self._save_state()

            # 손실 거래 시 세션 블랙리스트에 추가 (동일 페어 재진입 방지)
            if total_pnl < 0:
                self._session_loss_blacklist.add(pos['symbol1'])
                self._session_loss_blacklist.add(pos['symbol2'])
                logger.info(
                    f"Slot {slot_id}: Session blacklisted {pos['symbol1']}, "
                    f"{pos['symbol2']} (loss=${total_pnl:.2f})"
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
        # 바이낸스 미실현 PnL 한 번에 조회 (슬롯별 API 호출 방지)
        binance_upnl = self.get_binance_unrealized_pnl()
        for slot in self.slots:
            if slot.position_open and slot.slot_id in self.slot_positions:
                pos = self.slot_positions[slot.slot_id]
                entry_time = self.slot_entry_times.get(slot.slot_id)
                hours = 0.0
                if entry_time:
                    hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

                # 현재 가격 + z-score (항상 필요)
                p1 = p2 = 0.0
                current_z = 0.0
                try:
                    p1 = self.get_price(pos['symbol1'])
                    p2 = self.get_price(pos['symbol2'])
                    current_z = self.calculate_zscore(
                        p1, p2, pos['hedge_ratio'],
                        pos['spread_mean'], pos['spread_std'],
                    )
                except Exception:
                    pass

                # Unrealized PnL: 바이낸스 실제 값 우선
                try:
                    s1, s2 = pos['symbol1'], pos['symbol2']
                    if binance_upnl and (s1 in binance_upnl or s2 in binance_upnl):
                        pnl = binance_upnl.get(s1, 0.0) + binance_upnl.get(s2, 0.0)
                    else:
                        pnl = (p1 - pos['entry_price1']) * pos['qty1'] + \
                              (p2 - pos['entry_price2']) * pos['qty2']
                except Exception:
                    pnl = 0.0

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

    # ==================== State Persistence ====================

    def _save_state(self) -> None:
        """현재 슬롯/포지션 상태를 디스크에 저장.

        포지션 진입/청산 시 호출하여 크래시 복구를 지원한다.
        """
        try:
            save_state(
                slots=self.slots,
                slot_positions=self.slot_positions,
                slot_entry_times=self.slot_entry_times,
                trade_history=self.trade_history,
                paper_balance=self.paper_balance,
                paper_mode=self.paper_mode,
                initial_balance=self._initial_balance,
                session_high_water_mark=self._session_high_water_mark,
                daily_start_balance=self._daily_start_balance,
                daily_start_time=self._daily_start_time.isoformat(),
            )
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _recover_state(self) -> bool:
        """디스크에서 이전 상태를 복구.

        크래시 후 재시작 시 호출. 저장된 포지션 상태를 복원하고
        거래소 실제 포지션과 대조(reconcile)한다.

        Returns:
            True if 상태 복구됨, False if 복구할 상태 없음.
        """
        state = load_state()
        if state is None:
            return False

        # 모드 불일치 체크
        saved_mode = state.get('paper_mode', False)
        if saved_mode != self.paper_mode:
            logger.warning(
                f"State file mode mismatch: saved={saved_mode}, current={self.paper_mode} — "
                f"ignoring saved state"
            )
            clear_state()
            return False

        # 포지션 복원
        saved_positions = deserialize_positions(state.get('positions', {}))
        saved_entry_times = deserialize_entry_times(state.get('entry_times', {}))

        if not saved_positions:
            logger.info("Saved state has no open positions — clean start")
            clear_state()
            return False

        # 거래소 실제 포지션과 대조
        exchange_positions = self.get_current_positions()
        exchange_symbols = set()
        for pos in exchange_positions:
            exchange_symbols.add(pos['symbol'])

        restored_count = 0
        orphaned_exchange = []

        for slot_id_int, pos_data in saved_positions.items():
            slot_id = int(slot_id_int)
            if slot_id >= len(self.slots):
                logger.warning(f"Slot {slot_id} exceeds max_traders — skipping")
                continue

            sym1 = pos_data.get('symbol1', '')
            sym2 = pos_data.get('symbol2', '')

            # Live mode: 거래소에 해당 심볼 포지션이 있는지 확인
            if not self.paper_mode:
                has_sym1 = sym1 in exchange_symbols
                has_sym2 = sym2 in exchange_symbols

                if not has_sym1 and not has_sym2:
                    logger.warning(
                        f"Slot {slot_id}: Saved position {sym1}/{sym2} "
                        f"not found on exchange — discarding"
                    )
                    continue

                if has_sym1 != has_sym2:
                    missing = sym2 if has_sym1 else sym1
                    logger.critical(
                        f"Slot {slot_id}: Partial position detected — "
                        f"exchange has {'sym1' if has_sym1 else 'sym2'} "
                        f"but missing {missing}. MANUAL CHECK REQUIRED."
                    )
                    self.discord.send(content=(
                        f"🚨 **CRITICAL** 부분 포지션 감지 (복구 시)\n"
                        f"Slot {slot_id}: {sym1}/{sym2}\n"
                        f"누락 심볼: {missing}\n"
                        f"거래소에서 확인 필요!"
                    ))

            # 슬롯 복원
            slot = self.slots[slot_id]
            slot.position_open = True
            slot.current_pair = (sym1, sym2)
            slot.position_info = pos_data
            self.slot_positions[slot_id] = pos_data
            if slot_id in saved_entry_times:
                self.slot_entry_times[slot_id] = saved_entry_times[slot_id]

            restored_count += 1
            logger.info(
                f"Slot {slot_id}: Restored position {sym1}/{sym2} "
                f"(entry_z={pos_data.get('entry_zscore', '?')})"
            )

        # 거래소에 있지만 저장된 상태에 없는 포지션 검출
        if not self.paper_mode:
            saved_symbols = set()
            for pos_data in saved_positions.values():
                saved_symbols.add(pos_data.get('symbol1', ''))
                saved_symbols.add(pos_data.get('symbol2', ''))

            for sym in exchange_symbols:
                if sym not in saved_symbols:
                    orphaned_exchange.append(sym)

            if orphaned_exchange:
                logger.critical(
                    f"Orphaned exchange positions (not in saved state): "
                    f"{orphaned_exchange} — MANUAL CHECK REQUIRED"
                )
                self.discord.send(content=(
                    f"🚨 **CRITICAL** 미확인 거래소 포지션 발견 (복구 시)\n"
                    f"심볼: {', '.join(orphaned_exchange)}\n"
                    f"저장된 상태에 없는 포지션 — 수동 확인 필요!"
                ))

        # Drawdown 상태 복원
        self._initial_balance = state.get('initial_balance', 0.0)
        self._session_high_water_mark = state.get('session_high_water_mark', 0.0)
        self._daily_start_balance = state.get('daily_start_balance', 0.0)
        daily_start_str = state.get('daily_start_time', '')
        if daily_start_str:
            try:
                self._daily_start_time = datetime.fromisoformat(daily_start_str)
                if self._daily_start_time.tzinfo is None:
                    self._daily_start_time = self._daily_start_time.replace(tzinfo=timezone.utc)
            except ValueError:
                self._daily_start_time = datetime.now(timezone.utc)

        # Paper balance 복원
        if self.paper_mode:
            self.paper_balance = state.get('paper_balance', self.paper_balance)

        # Trade history 복원
        saved_trades = state.get('trade_history', [])
        if saved_trades:
            self.trade_history = saved_trades
            logger.info(f"Restored {len(saved_trades)} trade history records")

        logger.info(
            f"State recovery complete: {restored_count} positions restored"
        )
        self.discord.send(content=(
            f"♻️ **상태 복구 완료**\n"
            f"복원된 포지션: {restored_count}개\n"
            f"{'⚠️ 미확인 포지션: ' + ', '.join(orphaned_exchange) if orphaned_exchange else '✅ 거래소 대조 정상'}"
        ))
        return restored_count > 0

    # ==================== Drawdown & Daily Loss ====================

    def _check_drawdown(self) -> bool:
        """세션 최고 잔고 대비 드로다운 체크.

        max_drawdown_pct 초과 시 전체 청산 + 일시정지.

        Returns:
            True if 드로다운 브레이커 발동됨.
        """
        max_dd = self.config.get('max_drawdown_pct', 0.10)
        if max_dd <= 0:
            return False

        current_balance = self.get_total_balance()
        if current_balance <= 0:
            return False

        # High water mark 갱신
        if current_balance > self._session_high_water_mark:
            self._session_high_water_mark = current_balance

        if self._session_high_water_mark <= 0:
            return False

        drawdown = (self._session_high_water_mark - current_balance) / self._session_high_water_mark

        if drawdown >= max_dd:
            logger.critical(
                f"DRAWDOWN CIRCUIT BREAKER: {drawdown:.1%} >= {max_dd:.1%} "
                f"(HWM=${self._session_high_water_mark:.2f}, current=${current_balance:.2f})"
            )
            self.discord.send(content=(
                f"🔴 **드로다운 서킷 브레이커 발동**\n"
                f"드로다운: {drawdown:.1%} (한도: {max_dd:.1%})\n"
                f"최고 잔고: ${self._session_high_water_mark:,.2f}\n"
                f"현재 잔고: ${current_balance:,.2f}\n"
                f"→ 전체 청산 후 신규 진입 중단"
            ))
            self.close_all_positions()
            self._drawdown_triggered = True
            self.paused = True
            self._save_state()
            return True

        return False

    def _check_daily_loss(self) -> bool:
        """당일 손실 한도 체크.

        daily_loss_limit_pct 초과 시 신규 진입 중단 (기존 포지션 유지).

        Returns:
            True if 일일 손실 한도 도달.
        """
        limit = self.config.get('daily_loss_limit_pct', 0.05)
        if limit <= 0:
            return False

        # 24시간마다 daily_start_balance 리셋
        now = datetime.now(timezone.utc)
        if (now - self._daily_start_time).total_seconds() >= 86400:
            self._daily_start_balance = self.get_total_balance()
            self._daily_start_time = now
            logger.info(f"Daily loss tracker reset: start_balance=${self._daily_start_balance:.2f}")

        if self._daily_start_balance <= 0:
            return False

        current = self.get_total_balance()
        daily_loss = (self._daily_start_balance - current) / self._daily_start_balance

        if daily_loss >= limit:
            logger.warning(
                f"DAILY LOSS LIMIT: {daily_loss:.1%} >= {limit:.1%} "
                f"(start=${self._daily_start_balance:.2f}, current=${current:.2f})"
            )
            if not self.paused:
                self.discord.send(content=(
                    f"⚠️ **일일 손실 한도 도달**\n"
                    f"일일 손실: {daily_loss:.1%} (한도: {limit:.1%})\n"
                    f"시작 잔고: ${self._daily_start_balance:,.2f}\n"
                    f"현재 잔고: ${current:,.2f}\n"
                    f"→ 신규 진입 중단 (기존 포지션 모니터링 계속)"
                ))
                self.paused = True
            return True

        return False

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

        # 0.5. 드로다운 체크 (최우선 — 모니터링보다 먼저)
        if self._check_drawdown():
            return True  # 브레이커 발동 — 이번 사이클 스킵

        self._cycle_count += 1
        active = get_active_slots(self.slots)
        idle = get_idle_slots(self.slots)
        logger.debug(
            f"Cycle #{self._cycle_count}: {len(active)} active, {len(idle)} idle "
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
            if self._cycle_count % 15 == 0:
                logger.info(f"All slots occupied (cycle #{self._cycle_count}), monitoring only")
            else:
                logger.debug("All slots occupied, waiting...")
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

        # 2.6. 일일 손실 한도 체크
        if self._check_daily_loss():
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
        excluded = set(self.occupied_symbols) | set(self._failed_symbols.keys()) | self._session_loss_blacklist
        if self._failed_symbols:
            logger.info(f"Blacklisted symbols ({len(self._failed_symbols)}): {set(self._failed_symbols.keys())}")
        if self._session_loss_blacklist:
            logger.info(f"Session-blacklisted symbols ({len(self._session_loss_blacklist)}): {self._session_loss_blacklist}")

        # ── Vol-inverse sizing: compute anchor_std from all candidates ──
        # Low-vol pair gets full capital, high-vol gets proportionally less
        anchor_std = min(
            (c.spread_std for c in cached_candidates if c.spread_std > 0),
            default=0.0,
        )

        for slot in idle:
            # Quick balance check (before pair search)
            avail_balance = self.get_available_balance()
            if avail_balance < 10:
                logger.info(f"Slot {slot.slot_id}: Insufficient available balance (${avail_balance:.2f})")
                break

            result = self.find_best_pair(
                data, excluded_symbols=excluded,
                cached_candidates=cached_candidates,
            )
            if not result:
                logger.info(f"Slot {slot.slot_id}: No suitable pair found")
                break  # 캐시된 candidates에서 못 찾으면 나머지도 없음

            pair, pair_info = result

            # ── Dynamic leverage: target R = target_r_pct × balance ──
            target_r = self.config.get('target_r_pct', 0.01)
            if target_r > 0:
                # When dynamic leverage is ON, always use each pair's own spread_std.
                # Risk parity is achieved through leverage variation, not notional reduction.
                risk_std = pair_info.get('spread_std', 0.01)
                cost_rate = estimate_cost_rate(
                    trading_fee=self.config['trading_fee'],
                    slippage=self.config['slippage'],
                    funding_rate_8h=self.config['funding_rate_8h'],
                    expected_holding_bars=self.config.get('max_holding_hours', 20) * 4 // 2,
                    bar_interval_minutes=15,
                )
                effective_leverage = calculate_dynamic_leverage(
                    target_r_pct=target_r,
                    position_size_pct=self.config['position_size_pct'],
                    stop_loss_zscore=self.config['stop_loss_zscore'],
                    risk_std=risk_std,
                    hedge_ratio=pair_info['hedge_ratio'],
                    cost_rate=cost_rate,
                    max_leverage=float(self.config.get('max_leverage', 10)),
                )
                logger.info(
                    f"Slot {slot.slot_id}: Dynamic leverage={effective_leverage:.2f}x "
                    f"(target_r={target_r*100:.1f}%, risk_std={risk_std:.6f}, "
                    f"HR={pair_info['hedge_ratio']:.3f})"
                )
            else:
                effective_leverage = float(self.config['leverage'])

            # Pool-based capital allocation (identical to backtest)
            # Returns notional (leveraged) amount
            slot_capital = calculate_slot_capital(
                available_capital=self.get_available_balance(),
                active_slots=get_active_slots(self.slots),
                max_traders=self.max_traders,
                position_size_pct=self.config['position_size_pct'],
                leverage=effective_leverage,
            )
            if slot_capital < 10:
                logger.info(f"Slot {slot.slot_id}: Insufficient slot capital (${slot_capital:.2f})")
                break

            if self.execute_pair_trade(
                slot.slot_id, pair, pair_info, slot_capital,
                anchor_std=anchor_std, leverage=effective_leverage,
            ):
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
            # 실제 드로다운 계산
            dd_pct = 0.0
            if self._session_high_water_mark > 0:
                dd_pct = (self._session_high_water_mark - total_bal) / self._session_high_water_mark * 100
                dd_pct = max(0.0, dd_pct)
            self.discord.notify_daily_report(
                total_balance=total_bal,
                daily_pnl=session_pnl_dr,
                daily_return_pct=(session_pnl_dr / total_bal * 100) if total_bal > 0 else 0.0,
                cumulative_pnl=session_pnl_dr,
                cumulative_return_pct=(session_pnl_dr / total_bal * 100) if total_bal > 0 else 0.0,
                n_trades_today=len(self.trade_history),
                n_trades_total=len(self.trade_history),
                win_rate=win_rate,
                max_drawdown_pct=dd_pct,
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

        # BTC crash gate status
        btc_blocked, btc_ret = self.check_btc_crash_gate()
        btc_icon = "🔴" if btc_blocked else "🟢"
        btc_str = f" | BTC {btc_icon} {btc_ret:+.2%}"

        lines = [f"📊 **QuantBox Status**{paused_str}{btc_str}"]
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
        lines = []
        if self._failed_symbols:
            now = datetime.now(timezone.utc)
            lines.append(f"🚫 **체결실패 블랙리스트** ({len(self._failed_symbols)}개 심볼)")
            for sym, ts in sorted(self._failed_symbols.items()):
                elapsed = (now - ts).total_seconds()
                remaining = max(0, self.BLACKLIST_TTL_SECONDS - elapsed)
                lines.append(f"  `{sym}` — {remaining:.0f}초 남음")
        if self._session_loss_blacklist:
            lines.append(f"🔒 **세션 손실 블랙리스트** ({len(self._session_loss_blacklist)}개 심볼)")
            for sym in sorted(self._session_loss_blacklist):
                lines.append(f"  `{sym}` — 세션 종료까지")
        if not lines:
            return "✅ 블랙리스트 비어있음"
        return "\n".join(lines)

    def _cmd_blacklist_clear(self) -> str:
        """블랙리스트 수동 초기화."""
        n_failed = len(self._failed_symbols)
        n_session = len(self._session_loss_blacklist)
        self._failed_symbols.clear()
        self._session_loss_blacklist.clear()
        logger.info("Blacklist cleared via Discord command")
        return f"✅ 블랙리스트 초기화 (체결실패 {n_failed}개 + 세션손실 {n_session}개 제거)"

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

        # ── 상태 복구 (크래시 후 재시작) ──
        recovered = self._recover_state()
        if recovered:
            logger.info("Continuing from recovered state")

        # ── Drawdown / Daily Loss 초기화 ──
        balance = self.get_total_balance()
        if self._initial_balance <= 0:
            self._initial_balance = balance
        if self._session_high_water_mark <= 0:
            self._session_high_water_mark = balance
        if self._daily_start_balance <= 0:
            self._daily_start_balance = balance
            self._daily_start_time = datetime.now(timezone.utc)

        # Discord Bot 시작 (토큰이 설정되어 있을 때만)
        if self._bot_runner:
            self._bot_runner.start()
            logger.info("Discord Bot started — remote commands enabled")

        # Discord 시작 알림
        mode = "PAPER" if self.paper_mode else "LIVE"
        self.discord.notify_start(
            mode, balance,
            leverage=self.config['leverage'],
            max_traders=self.max_traders,
        )
        
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5  # 연속 5회 실패 시에만 전체 청산

        try:
            while True:
                try:
                    result = self.run_cycle()
                    consecutive_errors = 0  # 성공 시 카운터 리셋
                    if result is False:  # !stop 명령
                        logger.info("Stop requested via Discord")
                        break
                    self._interruptible_sleep(self.config['check_interval_seconds'])
                    if self._stop_requested:
                        logger.info("Stop requested via Discord (during sleep)")
                        break
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"Error in trading cycle ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}",
                        exc_info=True,
                    )
                    self.discord.notify_error(
                        f"[{consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}] {e}"
                    )

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.critical(
                            f"Consecutive errors reached {MAX_CONSECUTIVE_ERRORS} — "
                            f"closing all positions and pausing"
                        )
                        self.discord.send(content=(
                            f"🚨 **CRITICAL** 연속 {MAX_CONSECUTIVE_ERRORS}회 에러 발생\n"
                            f"전체 포지션 청산 후 대기 (5분)\n"
                            f"최근 에러: {e}"
                        ))
                        self.close_all_positions()
                        consecutive_errors = 0
                        self._interruptible_sleep(300)  # 5분 대기
                    else:
                        self._interruptible_sleep(60)  # 1분 대기 후 재시도
                    
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

        # 정상 종료: 상태 파일 삭제 (포지션 없으므로 복구 불필요)
        clear_state()

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
