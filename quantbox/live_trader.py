#!/usr/bin/env python3
"""
QuantBox Live Trader - 암호화폐 페어 트레이딩 실거래 시스템

실행 전 확인사항:
1. .env 파일에 바이낸스 API 키 설정
2. TRADING_MODE를 'live' 또는 'paper'로 설정
3. 초기 자본금 및 리스크 파라미터 확인

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
from quantbox.strategy.pair_selector import PairSelector, PairCandidate
from quantbox.strategy.filters import (
    FilterPipeline, CorrelationFilter, CointegrationFilter,
    HalfLifeFilter, HurstFilter
)
from quantbox.data.preprocessor import DataPreprocessor

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


class DiscordNotifier:
    """Discord 웹훅 알림 전송"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        if self.enabled:
            logger.info("Discord notifications enabled")
    
    def send(self, content: str = None, embed: Dict = None) -> bool:
        """메시지 전송"""
        if not self.enabled:
            return False
        
        try:
            data = {}
            if content:
                data['content'] = content
            if embed:
                data['embeds'] = [embed]
            
            r = requests.post(self.webhook_url, json=data, timeout=5)
            return r.status_code == 204
        except Exception as e:
            logger.warning(f"Discord notification failed: {e}")
            return False
    
    def notify_start(self, mode: str, balance: float):
        """봇 시작 알림"""
        self.send(embed={
            'title': '🚀 QuantBox 트레이딩 시작',
            'color': 0x00ff00,
            'fields': [
                {'name': '모드', 'value': mode, 'inline': True},
                {'name': '잔고', 'value': f'${balance:,.2f}', 'inline': True},
                {'name': '시작 시간', 'value': datetime.now().strftime('%Y-%m-%d %H:%M'), 'inline': True}
            ]
        })
    
    def notify_open(self, symbol1: str, symbol2: str, direction: str, zscore: float, capital: float):
        """포지션 오픈 알림"""
        color = 0x00ff00 if direction == 'LONG' else 0xff0000
        self.send(embed={
            'title': f'📈 포지션 오픈',
            'color': color,
            'fields': [
                {'name': '페어', 'value': f'{symbol1}/{symbol2}', 'inline': True},
                {'name': '방향', 'value': direction, 'inline': True},
                {'name': 'Z-Score', 'value': f'{zscore:.2f}', 'inline': True},
                {'name': '투자금', 'value': f'${capital:,.2f}', 'inline': True}
            ]
        })
    
    def notify_close(self, symbol1: str, symbol2: str, pnl: float, pnl_pct: float, reason: str):
        """포지션 청산 알림"""
        color = 0x00ff00 if pnl >= 0 else 0xff0000
        emoji = '✅' if pnl >= 0 else '❌'
        self.send(embed={
            'title': f'{emoji} 포지션 청산',
            'color': color,
            'fields': [
                {'name': '페어', 'value': f'{symbol1}/{symbol2}', 'inline': True},
                {'name': 'PnL', 'value': f'${pnl:+,.2f} ({pnl_pct:+.2f}%)', 'inline': True},
                {'name': '청산 사유', 'value': reason, 'inline': True}
            ]
        })
    
    def notify_error(self, error: str):
        """에러 알림"""
        self.send(embed={
            'title': '⚠️ 에러 발생',
            'color': 0xff9900,
            'description': error[:500]
        })


class LivePairTrader:
    """실시간 페어 트레이딩 시스템"""
    
    def __init__(
        self,
        paper_mode: bool = False,
        config: Dict = None
    ):
        self.paper_mode = paper_mode
        self.config = config or self._default_config()
        
        # Initialize Binance client
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
        
        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.filter_pipeline = self._create_filter_pipeline()
        self.pair_selector = PairSelector(
            filter_pipeline=self.filter_pipeline,
            max_pairs=20
        )
        
        # State
        self.current_position: Optional[Dict] = None
        self.position_entry_time: Optional[datetime] = None
        self.trade_history: List[Dict] = []
        self.prev_prices: Dict[str, float] = {}
        
        # Paper trading state
        self.paper_balance = float(os.getenv('PAPER_BALANCE', 1000.0))  # Default 1000 USDT
        self.paper_positions: List[Dict] = []
        
        # Discord notifications
        self.discord = DiscordNotifier()
        
        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"LivePairTrader initialized in {mode_str} mode")
        logger.info(f"Testnet: {testnet}")
        if paper_mode:
            logger.info(f"Paper Balance: ${self.paper_balance:.2f} USDT")
        
    def _default_config(self) -> Dict:
        """기본 설정"""
        return {
            # Data settings
            'quote_currency': 'USDT',
            'n_coins': int(os.getenv('N_COINS', 20)),
            'interval': '15m',
            'formation_hours': 168,  # 7 days
            
            # Strategy parameters
            'entry_zscore': float(os.getenv('ENTRY_ZSCORE', 2.0)),
            'exit_zscore': float(os.getenv('EXIT_ZSCORE', 0.5)),
            'stop_loss_zscore': float(os.getenv('STOP_LOSS_ZSCORE', 3.5)),
            'max_holding_hours': 24,
            
            # Risk parameters (Half Kelly: 12.5%)
            'max_position_pct': 0.125,  # Half Kelly criterion
            'trading_fee': float(os.getenv('TRADING_FEE', 0.0005)),
            'slippage': float(os.getenv('SLIPPAGE', 0.001)),
            'leverage': 1,
            
            # Filter parameters
            'min_correlation': 0.5,
            'max_cointegration_pvalue': 0.05,
            'min_half_life': 3,
            'max_half_life': 60,
            'max_hurst': 0.5,
            
            # Execution settings
            'check_interval_seconds': 60,
            'retry_attempts': 3,
            'retry_delay': 2,
        }
    
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
        """사용 가능 잔고 조회"""
        # Paper mode: use simulated balance
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
        """심볼 정보 조회"""
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
        """주문 실행"""
        if self.paper_mode:
            logger.info(f"[PAPER] Order: {side} {quantity:.6f} {symbol}")
            return {'orderId': f'paper_{int(time.time())}', 'status': 'FILLED'}
        
        try:
            # Set leverage
            self.client.futures_change_leverage(
                symbol=symbol,
                leverage=self.config['leverage']
            )
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"Order placed: {side} {quantity:.6f} {symbol} - ID: {order.get('orderId')}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, quantity: float) -> bool:
        """포지션 청산"""
        side = 'SELL' if quantity > 0 else 'BUY'
        quantity = abs(quantity)
        
        info = self.get_symbol_info(symbol)
        if not info:
            return False
        
        rounded_qty = self.round_quantity(
            quantity,
            info.get('step_size', 0.001),
            info.get('quantity_precision', 3)
        )
        
        if rounded_qty <= 0:
            logger.warning(f"Quantity too small to close for {symbol}")
            return False
        
        order = self.place_order(symbol, side, rounded_qty)
        return order is not None
    
    def close_all_positions(self):
        """모든 포지션 청산"""
        positions = self.get_current_positions()
        if not positions:
            logger.info("No positions to close")
            return
        
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
        
        self.current_position = None
        self.position_entry_time = None
    
    # ==================== Strategy Methods ====================
    
    def find_best_pair(self, data: Dict[str, pd.DataFrame]) -> Optional[Tuple[PairCandidate, Dict]]:
        """최적 페어 선택"""
        logger.info("Searching for cointegrated pairs...")
        
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
        
        best = tradeable[0]
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
    
    def execute_pair_trade(
        self,
        pair: PairCandidate,
        pair_info: Dict,
        capital: float
    ) -> bool:
        """페어 트레이드 실행"""
        symbol1, symbol2 = pair.symbol1, pair.symbol2
        hedge_ratio = pair_info['hedge_ratio']
        
        # Get current prices
        price1 = self.get_price(symbol1)
        price2 = self.get_price(symbol2)
        
        if price1 <= 0 or price2 <= 0:
            logger.error("Invalid prices")
            return False
        
        # Calculate z-score
        z_score = self.calculate_zscore(
            price1, price2, hedge_ratio,
            pair_info['spread_mean'], pair_info['spread_std']
        )
        
        logger.info(f"Current Z-Score: {z_score:.4f}")
        
        if abs(z_score) < self.config['entry_zscore']:
            logger.info("Z-Score below entry threshold")
            return False
        
        # Determine signal: z < 0 -> Long spread, z > 0 -> Short spread
        signal = 1 if z_score < 0 else -1
        side1 = 'BUY' if signal == 1 else 'SELL'
        side2 = 'SELL' if signal == 1 else 'BUY'
        
        # Calculate position sizes
        trade_capital = capital * self.config['max_position_pct']
        effective_capital = trade_capital * (1 - 2 * self.config['trading_fee'])
        
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
        
        logger.info(f"Opening position: {symbol1} {side1} {qty1:.6f}, {symbol2} {side2} {qty2:.6f}")
        logger.info(f"Capital usage: ${notional1 + notional2:.2f} ({(notional1 + notional2) / capital * 100:.1f}%)")
        
        # Place orders
        order1 = self.place_order(symbol1, side1, qty1)
        order2 = self.place_order(symbol2, side2, qty2)
        
        if not order1 or not order2:
            logger.error("Order failed, closing positions")
            self.close_all_positions()
            return False
        
        # Record position
        self.current_position = {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'signal': signal,
            'qty1': qty1 if signal == 1 else -qty1,
            'qty2': -qty2 if signal == 1 else qty2,
            'entry_price1': price1,
            'entry_price2': price2,
            'entry_zscore': z_score,
            'hedge_ratio': hedge_ratio,
            'spread_mean': pair_info['spread_mean'],
            'spread_std': pair_info['spread_std'],
        }
        self.position_entry_time = datetime.now(timezone.utc)
        
        direction = "LONG Spread" if signal == 1 else "SHORT Spread"
        logger.info(f"Position opened: signal={signal}, z-score={z_score:.4f}")
        
        # Discord 알림
        self.discord.notify_open(
            symbol1, symbol2, direction, z_score, notional1 + notional2
        )
        
        return True
    
    def monitor_position(self) -> bool:
        """포지션 모니터링 및 청산 조건 확인"""
        if not self.current_position:
            return False
        
        pos = self.current_position
        
        # Get current prices
        price1 = self.get_price(pos['symbol1'])
        price2 = self.get_price(pos['symbol2'])
        
        if price1 <= 0 or price2 <= 0:
            return True  # Keep position
        
        # Calculate current z-score
        z_score = self.calculate_zscore(
            price1, price2, pos['hedge_ratio'],
            pos['spread_mean'], pos['spread_std']
        )
        
        signal = pos['signal']
        entry_z = pos['entry_zscore']
        elapsed = (datetime.now(timezone.utc) - self.position_entry_time).total_seconds() / 3600
        
        logger.info(
            f"Monitor: Z-Score={z_score:.4f} (entry={entry_z:.4f}), "
            f"Elapsed={elapsed:.1f}h, Signal={signal}"
        )
        
        should_close = False
        close_reason = ""
        
        # Exit conditions
        if signal == 1:  # Long spread
            if z_score >= -self.config['exit_zscore']:
                should_close = True
                close_reason = "mean_reversion"
            elif z_score < entry_z - self.config['stop_loss_zscore']:
                should_close = True
                close_reason = "stop_loss"
        else:  # Short spread
            if z_score <= self.config['exit_zscore']:
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
            logger.info(f"Closing position: {close_reason}")
            
            # Calculate PnL
            pnl1 = (price1 - pos['entry_price1']) * pos['qty1']
            pnl2 = (price2 - pos['entry_price2']) * pos['qty2']
            total_pnl = pnl1 + pnl2
            
            # Calculate PnL percentage
            entry_value = abs(pos['qty1'] * pos['entry_price1']) + abs(pos['qty2'] * pos['entry_price2'])
            pnl_pct = (total_pnl / entry_value * 100) if entry_value > 0 else 0
            
            self.close_all_positions()
            
            # Record trade
            self.trade_history.append({
                'symbol1': pos['symbol1'],
                'symbol2': pos['symbol2'],
                'entry_time': self.position_entry_time,
                'exit_time': datetime.now(timezone.utc),
                'entry_zscore': entry_z,
                'exit_zscore': z_score,
                'pnl': total_pnl,
                'reason': close_reason,
            })
            
            logger.info(f"Trade closed: PnL=${total_pnl:.2f}, Reason={close_reason}")
            
            # Discord 알림
            self.discord.notify_close(
                pos['symbol1'], pos['symbol2'], total_pnl, pnl_pct, close_reason
            )
            
            return False
        
        return True  # Keep position
    
    # ==================== Main Loop ====================
    
    def run_once(self):
        """단일 트레이딩 사이클 실행"""
        logger.info("=" * 60)
        logger.info("Starting trading cycle...")
        
        # Check and close existing positions
        positions = self.get_current_positions()
        if positions:
            logger.info(f"Found {len(positions)} existing positions, closing...")
            self.close_all_positions()
        
        # Get available capital
        capital = self.get_available_balance()
        logger.info(f"Available balance: ${capital:.2f}")
        
        if capital < 100:
            logger.warning("Insufficient balance")
            return False
        
        # Get trading coins and download data
        coins = self.get_trading_coins()
        if not coins:
            logger.error("Failed to get trading coins")
            return False
        
        data = self.download_multi_symbol_data(coins)
        if len(data) < 4:
            logger.error("Not enough data for pair selection")
            return False
        
        # Find best pair
        result = self.find_best_pair(data)
        if not result:
            logger.info("No suitable pairs found")
            return False
        
        pair, pair_info = result
        
        # Execute trade
        if not self.execute_pair_trade(pair, pair_info, capital):
            return False
        
        # Monitor position
        while self.current_position:
            time.sleep(self.config['check_interval_seconds'])
            if not self.monitor_position():
                break
        
        logger.info("Trading cycle completed")
        return True
    
    def run_continuous(self):
        """연속 트레이딩 루프"""
        logger.info("Starting continuous trading loop...")
        self.print_balance()
        
        # Discord 시작 알림
        balance = self.get_available_balance()
        mode = "PAPER" if self.paper_mode else "LIVE"
        self.discord.notify_start(mode, balance)
        
        try:
            while True:
                try:
                    self.run_once()
                    time.sleep(60)  # Wait before next cycle
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    self.discord.notify_error(str(e))
                    self.close_all_positions()
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.close_all_positions()
            self.print_summary()
    
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
