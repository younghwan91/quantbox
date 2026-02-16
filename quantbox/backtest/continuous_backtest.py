"""
Continuous Pair Trading Backtest Engine

This module implements a high-frequency pair trading backtest system that:
1. Continuously searches for new pairs after each position close
2. Maximizes market participation
3. Strictly prevents lookahead bias
4. Accurately models transaction costs

Key Design Principles:
- Formation period and trading period are completely separated
- Hedge ratio and spread statistics are FIXED at position entry time
- Only information available at time T is used for decisions at time T
- Realistic fee and slippage modeling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from tqdm import tqdm

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

logger = logging.getLogger('ContinuousBacktest')


class ExitReason(Enum):
    """Trade exit reasons"""
    MEAN_REVERSION = "mean_reversion"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    MAX_HOLDING = "max_holding"
    END_OF_PERIOD = "end_of_period"
    PAIR_UNAVAILABLE = "pair_unavailable"


@dataclass
class TradeResult:
    """Individual trade result"""
    pair: Tuple[str, str]
    entry_time: datetime
    exit_time: datetime
    entry_z: float
    exit_z: float
    entry_spread: float
    exit_spread: float
    signal: int  # 1=long spread, -1=short spread
    capital_allocated: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    funding_cost: float
    net_pnl: float
    net_return_pct: float
    bars_held: int
    exit_reason: ExitReason
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    slot_id: int = 0  # Which trader slot executed this trade
    best_z: float = 0.0  # Best (most favorable) z-score reached during trade


@dataclass
class BacktestState:
    """Backtest state tracking - supports multiple concurrent trader slots.

    current_capital represents unallocated cash. When a slot enters a position,
    capital is deducted; when it exits, capital + PnL is returned.
    """
    current_capital: float
    peak_capital: float
    max_traders: int = 1
    slots: list[TraderSlotState] = field(default_factory=list)
    trades: list[TradeResult] = field(default_factory=list)
    equity_curve: list[Dict] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    search_count: int = 0  # How many times we searched for pairs
    no_trade_periods: int = 0  # Periods with no suitable pair found

    def __post_init__(self):
        if not self.slots:
            self.slots = [
                TraderSlotState(slot_id=i) for i in range(self.max_traders)
            ]

    @property
    def occupied_symbols(self) -> set[str]:
        """All symbols currently in use across all active slots."""
        return get_occupied_symbols(self.slots)

    @property
    def idle_slots(self) -> list[TraderSlotState]:
        """All slots not currently holding a position."""
        return get_idle_slots(self.slots)

    @property
    def active_slots(self) -> list[TraderSlotState]:
        """All slots currently holding a position."""
        return get_active_slots(self.slots)

    @property
    def total_allocated(self) -> float:
        """Total capital allocated in active positions."""
        return sum(
            s.position_info['capital']
            for s in self.active_slots
            if s.position_info
        )

    @property
    def total_equity(self) -> float:
        """Total book value (cash + allocated, excluding unrealized PnL)."""
        return self.current_capital + self.total_allocated


@dataclass 
class ContinuousBacktestConfig:
    """Configuration for continuous backtest"""
    # Capital
    initial_capital: float = 10000.0
    
    # Transaction costs (per side)
    trading_fee: float = 0.0005  # 0.05% per trade (Binance maker)
    slippage: float = 0.0010    # 0.10% slippage estimate
    funding_rate_8h: float = 0.0001  # 0.01% per 8h (Binance perpetual futures)
    bar_interval_minutes: int = 15  # Bar interval for time calculations
    
    # Formation window for pair selection
    formation_bars: int = 240   # Bars to analyze for pair selection
    
    # Entry/Exit thresholds
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5    # Exit when z-score returns to this level
    stop_loss_zscore: float = 3.5  # Max adverse z-score movement

    # Dynamic exit: half-life 기반 exit_zscore 조정
    use_dynamic_exit: bool = False
    dynamic_exit_base: float = 0.3   # 가장 빠른 페어의 exit z-score
    dynamic_exit_range: float = 0.7  # base + range = 가장 느린 페어의 exit z-score
    
    # Holding limits (20h = 80 bars @ 15min interval)
    max_holding_bars: int = 80
    
    # Pair selection criteria
    min_correlation: float = 0.6
    max_cointegration_pvalue: float = 0.05
    min_half_life: float = 2.0   # Minimum half-life in bars
    max_half_life: float = 50.0  # Maximum half-life in bars
    max_hurst: float = 0.5  # Maximum Hurst exponent (< 0.5 = mean-reverting)
    
    # Regime protection gates (parity with live_trader)
    z_score_cap: float = 4.0           # |z| > cap → regime anomaly, reject entry
    vol_expansion_gate: float = 2.0    # recent_std/formation_std > gate → reject entry
    vol_expansion_window: int = 8      # Number of recent bars for vol gate

    # Trailing stop (ratchet stop): z가 best_z에서 trail_z만큼 되돌림 시 이익 보존 청산
    trailing_stop_enabled: bool = False         # Trailing stop 활성화 여부
    trailing_stop_activate_delta: float = 0.3   # entry 대비 Δz 이 이상 개선 시 활성화
    trailing_stop_trail_z: float = 0.8          # best_z에서 trail_z만큼 되돌림 시 발동

    # BTC crash gate (parity with live_trader)
    btc_crash_gate_enabled: bool = True   # Enable BTC crash gate in backtest
    btc_crash_window: int = 8             # Number of bars to measure BTC return
    btc_crash_threshold: float = 0.03    # |return| > 3% → block all entries

    # Risk management
    position_size_pct: float = -1.0  # Fraction of capital per trade (auto: 1/max_traders)
    leverage: int = 1  # Fallback leverage (used when target_r_pct=0, i.e. dynamic disabled)
    target_r_pct: float = 0.01  # Target risk per trade as fraction of balance (0.01=1%, 0=disable dynamic)
    max_leverage: float = 125.0  # Safety cap for dynamic leverage (Binance absolute max)
    vol_inverse_sizing: bool = True   # Position size ∝ 1/spread_std (risk parity)
    vol_sizing_max_multiplier: float = 3.0  # Cap: notional ≤ base × multiplier
    
    # Multi-trader
    max_traders: int = 5  # Maximum concurrent independent traders
    
    # Pair universe
    min_volume_rank: int = 50   # Only consider top N coins by volume
    
    def __post_init__(self):
        """Auto-calculate position_size_pct if not explicitly set."""
        if self.position_size_pct < 0:
            self.position_size_pct = 1.0 / self.max_traders
    
    @property
    def total_round_trip_cost(self) -> float:
        """Total cost for entry + exit (excluding funding)"""
        return 2 * (self.trading_fee + self.slippage)

    @property
    def bars_per_funding_period(self) -> int:
        """Number of bars in one 8-hour funding period.

        Binance charges funding every 8 hours.
        For 15-min bars: 8h * 60 / 15 = 32 bars.
        """
        return (8 * 60) // self.bar_interval_minutes


class PairStatistics:
    """Calculate pair statistics with strict lookahead prevention"""
    
    @staticmethod
    def calculate_hedge_ratio(prices1: np.ndarray, prices2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate hedge ratio using OLS regression on log prices.
        
        log(P1) = alpha + beta * log(P2) + epsilon
        
        Returns:
            (hedge_ratio, intercept)
        """
        log_p1 = np.log(prices1)
        log_p2 = np.log(prices2)
        
        # OLS: minimize sum((log_p1 - alpha - beta*log_p2)^2)
        X = np.column_stack([np.ones(len(log_p2)), log_p2])
        coeffs = np.linalg.lstsq(X, log_p1, rcond=None)[0]
        
        intercept, hedge_ratio = coeffs
        return hedge_ratio, intercept
    
    @staticmethod
    def calculate_spread(
        prices1: np.ndarray, 
        prices2: np.ndarray,
        hedge_ratio: float,
        intercept: float
    ) -> np.ndarray:
        """Calculate spread series"""
        log_p1 = np.log(prices1)
        log_p2 = np.log(prices2)
        return log_p1 - (hedge_ratio * log_p2 + intercept)
    
    @staticmethod
    def calculate_zscore(
        spread: np.ndarray,
        mean: float,
        std: float
    ) -> np.ndarray:
        """Calculate z-score series"""
        if std <= 0:
            return np.zeros_like(spread)
        return (spread - mean) / std
    
    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using OLS.
        
        Fits: delta_spread = lambda * spread_lag + epsilon
        Half-life = -ln(2) / ln(1 + lambda)
        """
        spread_arr = spread.values if isinstance(spread, pd.Series) else spread
        
        spread_lag = spread_arr[:-1]
        spread_diff = spread_arr[1:] - spread_arr[:-1]
        
        # OLS regression
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        try:
            coeffs = np.linalg.lstsq(X, spread_diff, rcond=None)[0]
            lambda_param = coeffs[1]
            
            if lambda_param >= 0:
                return np.inf  # Not mean-reverting
            
            half_life = -np.log(2) / np.log(1 + lambda_param)
            return max(0, half_life)
        except:
            return np.inf
    
    @staticmethod
    def cointegration_test(prices1: np.ndarray, prices2: np.ndarray) -> Tuple[float, float]:
        """
        Perform Engle-Granger cointegration test.
        
        Returns:
            (t_statistic, p_value)
        """
        try:
            from statsmodels.tsa.stattools import coint
            t_stat, p_value, _ = coint(np.log(prices1), np.log(prices2))
            return t_stat, p_value
        except:
            return 0.0, 1.0
    
    @staticmethod
    def pearson_correlation(prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate correlation of log returns"""
        log_ret1 = np.diff(np.log(prices1))
        log_ret2 = np.diff(np.log(prices2))
        
        if len(log_ret1) < 2:
            return 0.0
        
        return np.corrcoef(log_ret1, log_ret2)[0, 1]

    @staticmethod
    def calculate_hurst(spread: np.ndarray, max_lag: int = 100) -> float:
        """Calculate Hurst exponent using R/S analysis.
        
        Same implementation as HurstFilter in strategy/filters.py.
        H < 0.5 → mean-reverting, H > 0.5 → trending.
        """
        lags = range(2, min(max_lag, len(spread) // 4))
        if len(list(lags)) < 2:
            return 0.5  # not enough data, assume random walk
        tau = [np.std(np.subtract(spread[lag:], spread[:-lag])) for lag in lags]
        # Avoid log(0)
        tau = [t for t in tau if t > 0]
        if len(tau) < 2:
            return 0.5
        valid_lags = list(range(2, 2 + len(tau)))
        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return poly[0] / 2.0


class ContinuousPairBacktest:
    """
    Continuous pair trading backtest engine.
    
    Key Features:
    - Searches for new pairs after each position close
    - Strictly prevents lookahead bias
    - Accurate transaction cost modeling
    - Comprehensive performance tracking
    """
    
    def __init__(self, config: ContinuousBacktestConfig = None):
        self.config = config or ContinuousBacktestConfig()
        self.stats = PairStatistics()
        
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_idx: int = None,
        end_idx: int = None,
        progress: bool = True
    ) -> Dict:
        """
        Run continuous backtest.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
                  Must have 'Close' column, optionally 'Volume'
            start_idx: Starting bar index (default: after formation period)
            end_idx: Ending bar index
            progress: Show progress bar
            
        Returns:
            Dictionary with results, metrics, trades
        """
        # Validate and prepare data
        data = self._prepare_data(data)
        if not data:
            raise ValueError("No valid data provided")
        
        # Get aligned timestamps
        symbols = list(data.keys())
        timestamps = data[symbols[0]].index.tolist()
        
        # Determine trading range
        start_idx = start_idx or self.config.formation_bars
        end_idx = end_idx or len(timestamps)
        
        if start_idx >= end_idx:
            raise ValueError("Not enough data for backtest")
        
        # Initialize state
        state = BacktestState(
            current_capital=self.config.initial_capital,
            peak_capital=self.config.initial_capital,
            max_traders=self.config.max_traders,
        )
        
        logger.info(f"Starting continuous backtest")
        logger.info(f"  Period: {timestamps[start_idx]} to {timestamps[end_idx-1]}")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Initial capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"  Max concurrent traders: {self.config.max_traders}")
        logger.info(f"  Position size per trader: {self.config.position_size_pct*100:.1f}%")
        if self.config.target_r_pct > 0:
            logger.info(f"  Dynamic leverage: target R={self.config.target_r_pct*100:.1f}%, max={self.config.max_leverage:.0f}x")
        else:
            logger.info(f"  Leverage: {self.config.leverage}x (fixed)")
        logger.info(f"  Round-trip cost: {self.config.total_round_trip_cost*100:.2f}%")
        logger.info(f"  Funding rate: {self.config.funding_rate_8h*100:.4f}% per 8h")
        
        # Main backtest loop
        iterator = range(start_idx, end_idx)
        if progress:
            iterator = tqdm(iterator, desc="Backtesting")
        
        for bar_idx in iterator:
            timestamp = timestamps[bar_idx]
            
            # Get current bar data
            bar_data = self._get_bar_data(data, bar_idx)
            
            # 1. Update ALL active positions (may close some)
            for slot in list(state.active_slots):  # copy list since closing modifies it
                self._update_position(state, slot, bar_data, bar_idx, timestamp)
            
            # 2. Search and enter for ALL idle slots at once
            idle = state.idle_slots
            if idle:
                self._search_and_enter_multi(state, idle, data, bar_idx, timestamp)
            
            # Record equity
            self._record_equity(state, bar_data, timestamp)
        
        # Close any remaining positions at end
        final_bar = self._get_bar_data(data, end_idx - 1)
        for slot in list(state.active_slots):
            self._close_position(
                state, slot, final_bar, end_idx - 1,
                timestamps[end_idx - 1], ExitReason.END_OF_PERIOD
            )
        
        # Compile results
        return self._compile_results(state)
    
    def _prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare and validate data"""
        prepared = {}
        
        # Find common timestamps
        common_idx = None
        for symbol, df in data.items():
            if 'Close' not in df.columns:
                logger.warning(f"Skipping {symbol}: no Close column")
                continue
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)
        
        if common_idx is None or len(common_idx) < self.config.formation_bars + 10:
            return {}
        
        # Align all data to common index
        for symbol, df in data.items():
            if 'Close' not in df.columns:
                continue
            aligned = df.loc[common_idx].copy()
            if len(aligned) == len(common_idx):
                prepared[symbol] = aligned
        
        return prepared
    
    def _get_bar_data(
        self, 
        data: Dict[str, pd.DataFrame],
        bar_idx: int
    ) -> Dict[str, Dict]:
        """Get bar data for all symbols at given index"""
        bar_data = {}
        for symbol, df in data.items():
            if bar_idx < len(df):
                row = df.iloc[bar_idx]
                bar_data[symbol] = {
                    'close': row['Close'],
                    'high': row.get('High', row['Close']),
                    'low': row.get('Low', row['Close']),
                    'volume': row.get('Volume', 0)
                }
        return bar_data
    
    def _search_and_enter_multi(
        self,
        state: BacktestState,
        idle_slots: list[TraderSlotState],
        data: Dict[str, pd.DataFrame],
        bar_idx: int,
        timestamp: datetime,
    ):
        """Search for valid pairs and enter positions for all idle slots.

        Finds the top-N non-overlapping pair candidates and assigns
        one to each idle slot, ensuring no symbol duplication.

        Args:
            state: Current backtest state.
            idle_slots: List of slots available for new positions.
            data: Full price data.
            bar_idx: Current bar index.
            timestamp: Current timestamp.
        """
        state.search_count += 1

        # Get formation period data (ONLY past data, no lookahead)
        formation_start = bar_idx - self.config.formation_bars
        if formation_start < 0:
            state.no_trade_periods += 1
            return

        # Get currently occupied symbols
        excluded = state.occupied_symbols

        # Find top non-overlapping candidates
        candidates = self._find_top_pairs(
            data, formation_start, bar_idx - 1,
            excluded_symbols=excluded,
            n=len(idle_slots),
        )

        if not candidates:
            state.no_trade_periods += 1
            return

        # Compute anchor spread_std for vol-inverse sizing: min across candidates
        # Low-vol pair gets full slot capital, high-vol pairs get scaled down
        anchor_std = min(c['spread_std'] for c in candidates) if self.config.vol_inverse_sizing else 0.0

        # Try to enter each candidate into an idle slot
        entered_any = False
        for slot, candidate in zip(idle_slots, candidates):
            if self._enter_slot(state, slot, candidate, data, bar_idx, timestamp, anchor_std=anchor_std):
                entered_any = True

        if not entered_any:
            state.no_trade_periods += 1

    def _check_btc_crash_gate(
        self,
        data: Dict[str, pd.DataFrame],
        bar_idx: int,
    ) -> tuple[bool, float]:
        """BTC crash gate for backtest (parity with live_trader).

        Checks if BTCUSDT (if present in data) moved more than
        btc_crash_threshold over the last btc_crash_window bars.

        Args:
            data: Full price data dict.
            bar_idx: Current bar index.

        Returns:
            (is_blocked, btc_return)
        """
        if not self.config.btc_crash_gate_enabled:
            return False, 0.0

        # Find BTC symbol in data
        btc_sym = None
        for sym in data:
            if sym.startswith('BTC') and sym.endswith('USDT'):
                btc_sym = sym
                break

        if btc_sym is None:
            # No BTC data — fail-open (same as live on API error)
            return False, 0.0

        window = self.config.btc_crash_window
        if bar_idx < window:
            return False, 0.0

        btc_df = data[btc_sym]
        open_price = btc_df.iloc[bar_idx - window]['Close']
        close_price = btc_df.iloc[bar_idx]['Close']

        if open_price <= 0:
            return False, 0.0

        btc_return = (close_price - open_price) / open_price

        if abs(btc_return) > self.config.btc_crash_threshold:
            return True, btc_return

        return False, btc_return

    def _enter_slot(
        self,
        state: BacktestState,
        slot: TraderSlotState,
        candidate: Dict,
        data: Dict[str, pd.DataFrame],
        bar_idx: int,
        timestamp: datetime,
        anchor_std: float = 0.0,
    ) -> bool:
        """Attempt to enter a position in a specific slot.

        Checks regime gates (BTC crash, z-cap, vol expansion) and
        z-score threshold at the CURRENT bar, then allocates capital.
        Gate logic is identical to live_trader.execute_pair_trade().

        When vol_inverse_sizing is enabled, position notional is scaled by
        anchor_std / spread_std so that stop-loss dollar amounts are equalized
        across pairs with different volatilities.

        Args:
            state: Current backtest state.
            slot: The trader slot to use.
            candidate: Pair candidate dict with hedge_ratio, spread_mean, etc.
            data: Full price data.
            bar_idx: Current bar index.
            timestamp: Current timestamp.
            anchor_std: Min spread_std across candidates (for vol-inverse sizing).

        Returns:
            True if position was entered, False if any gate blocked.
        """
        symbol1 = candidate['symbol1']
        symbol2 = candidate['symbol2']

        # ── Gate 1: BTC Crash Gate (parity with live) ──
        btc_blocked, btc_ret = self._check_btc_crash_gate(data, bar_idx)
        if btc_blocked:
            logger.debug(
                f"Slot {slot.slot_id}: BTC crash gate blocked entry "
                f"(BTC {btc_ret:+.2%}) at bar {bar_idx}"
            )
            return False

        # Get entry prices at current bar
        entry_price1 = data[symbol1].iloc[bar_idx]['Close']
        entry_price2 = data[symbol2].iloc[bar_idx]['Close']

        # Calculate entry spread and z-score using FORMATION statistics (frozen)
        entry_spread = self.stats.calculate_spread(
            np.array([entry_price1]),
            np.array([entry_price2]),
            candidate['hedge_ratio'],
            candidate['intercept'],
        )[0]

        entry_z = (entry_spread - candidate['spread_mean']) / candidate['spread_std']

        # ── Gate 2: Z-Score Cap (parity with live) ──
        if abs(entry_z) > self.config.z_score_cap:
            logger.debug(
                f"Slot {slot.slot_id}: Z-Score {entry_z:.2f} exceeds cap "
                f"|{self.config.z_score_cap}| — regime anomaly, skipping"
            )
            return False

        # ── Gate 3: Vol Expansion Gate (parity with live) ──
        if self.config.vol_expansion_gate > 0:
            vol_window = self.config.vol_expansion_window
            if bar_idx >= vol_window:
                recent_prices1 = data[symbol1].iloc[bar_idx - vol_window + 1:bar_idx + 1]['Close'].values
                recent_prices2 = data[symbol2].iloc[bar_idx - vol_window + 1:bar_idx + 1]['Close'].values
                recent_spread = (
                    np.log(recent_prices1)
                    - candidate['hedge_ratio'] * np.log(recent_prices2)
                )
                recent_std = float(np.std(recent_spread))
                formation_std = candidate['spread_std']
                vol_ratio = recent_std / formation_std if formation_std > 0 else 999.0
                if vol_ratio > self.config.vol_expansion_gate:
                    logger.debug(
                        f"Slot {slot.slot_id}: Vol expansion {vol_ratio:.2f}x > "
                        f"gate {self.config.vol_expansion_gate:.1f}x — skipping"
                    )
                    return False

        # ── Gate 4: Z-Score Threshold ──
        # Determine signal direction
        if entry_z <= -self.config.entry_zscore:
            signal = 1  # Long spread (expect z to increase)
        elif entry_z >= self.config.entry_zscore:
            signal = -1  # Short spread (expect z to decrease)
        else:
            # Z-score not extreme enough (market moved since formation end)
            return False

        # ── Dynamic leverage: target R = target_r_pct × balance ──
        if self.config.target_r_pct > 0:
            # When dynamic leverage is ON, always use each pair's own spread_std.
            # This achieves risk parity through leverage (high-vol → lower leverage)
            # instead of through notional reduction, ensuring full capital utilisation.
            risk_std = candidate['spread_std']
            cost_rate = estimate_cost_rate(
                trading_fee=self.config.trading_fee,
                slippage=self.config.slippage,
                funding_rate_8h=self.config.funding_rate_8h,
                expected_holding_bars=self.config.max_holding_bars // 2,
                bar_interval_minutes=self.config.bar_interval_minutes,
            )
            effective_leverage = calculate_dynamic_leverage(
                target_r_pct=self.config.target_r_pct,
                position_size_pct=self.config.position_size_pct,
                stop_loss_zscore=self.config.stop_loss_zscore,
                risk_std=risk_std,
                hedge_ratio=candidate['hedge_ratio'],
                cost_rate=cost_rate,
                max_leverage=self.config.max_leverage,
            )
        else:
            # Dynamic leverage disabled — use fixed config value
            effective_leverage = float(self.config.leverage)

        # Calculate position size using pool-based capital allocation
        # With leverage, notional > margin. We deduct margin from cash pool.
        notional = calculate_slot_capital(
            available_capital=state.current_capital,
            active_slots=state.active_slots,
            max_traders=self.config.max_traders,
            position_size_pct=self.config.position_size_pct,
            leverage=effective_leverage,
        )

        if notional <= 0:
            logger.debug(f"Slot {slot.slot_id}: no capital available")
            return False

        # ── Vol-inverse sizing (risk parity) ──
        # When dynamic leverage is ON (target_r_pct > 0), risk parity is achieved
        # through leverage variation (spread_std → leverage), NOT notional reduction.
        # This ensures full capital utilisation while maintaining exact R = target_r_pct.
        # Vol-inverse notional scaling is only used when dynamic leverage is OFF.
        if self.config.target_r_pct <= 0 and self.config.vol_inverse_sizing and anchor_std > 0 and candidate['spread_std'] > 0:
            vol_ratio = anchor_std / candidate['spread_std']
            vol_ratio = min(vol_ratio, self.config.vol_sizing_max_multiplier)  # cap
            vol_ratio = min(vol_ratio, 1.0)  # anchor = min → ratio ≤ 1 always
            notional = notional * vol_ratio

        # Margin is the actual cash consumed from the pool
        margin = calculate_margin_from_notional(notional, effective_leverage)

        # Deduct margin (not notional) from pool
        state.current_capital -= margin

        # Store position info in slot
        slot.position_open = True
        slot.current_pair = (symbol1, symbol2)
        slot.position_info = {
            'entry_bar': bar_idx,
            'entry_time': timestamp,
            'entry_price1': entry_price1,
            'entry_price2': entry_price2,
            'entry_spread': entry_spread,
            'entry_z': entry_z,
            'signal': signal,
            'capital': margin,  # Actual cash locked (margin)
            'notional': notional,  # Effective exposure (margin × leverage)
            'leverage': effective_leverage,
            # Frozen formation statistics (NO UPDATES during trade)
            'hedge_ratio': candidate['hedge_ratio'],
            'intercept': candidate['intercept'],
            'spread_mean': candidate['spread_mean'],
            'spread_std': candidate['spread_std'],
            'half_life': candidate['half_life'],
            'best_z': entry_z,  # Track best (most favorable) z for trailing stop
            'trailing_active': False,  # Whether trailing stop has been activated
        }

        logger.debug(
            f"Slot {slot.slot_id}: Entered {symbol1}/{symbol2} at bar {bar_idx}: "
            f"z={entry_z:.2f}, signal={'LONG' if signal==1 else 'SHORT'}, "
            f"margin=${margin:.2f}, notional=${notional:.2f} ({effective_leverage:.1f}x)"
        )
        return True
    
    def _find_top_pairs(
        self,
        data: Dict[str, pd.DataFrame],
        start_idx: int,
        end_idx: int,
        excluded_symbols: set[str] = None,
        n: int = 1,
    ) -> list[Dict]:
        """
        Find top N non-overlapping tradeable pairs from formation period.

        Uses greedy selection: best-scored first, then skip any candidate
        whose symbols overlap with already-selected pairs or excluded symbols.

        CRITICAL: Only uses data from [start_idx, end_idx] - NO FUTURE DATA!

        Args:
            data: Full price data dict.
            start_idx: Formation period start index (inclusive).
            end_idx: Formation period end index (inclusive).
            excluded_symbols: Symbols already in use (by other active slots).
            n: Maximum number of pairs to return.

        Returns:
            List of candidate dicts, len <= n, with no symbol overlap.
        """
        excluded_symbols = excluded_symbols or set()
        symbols = list(data.keys())
        candidates = []
        
        # Generate all pairs
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                # Get formation period prices ONLY
                prices1 = data[s1].iloc[start_idx:end_idx+1]['Close'].values
                prices2 = data[s2].iloc[start_idx:end_idx+1]['Close'].values
                
                if len(prices1) < 30 or len(prices2) < 30:
                    continue
                
                # Check minimum criteria
                if (prices1 <= 0).any() or (prices2 <= 0).any():
                    continue
                
                # 1. Correlation filter
                corr = self.stats.pearson_correlation(prices1, prices2)
                if corr < self.config.min_correlation:
                    continue
                
                # 2. Cointegration filter
                t_stat, p_value = self.stats.cointegration_test(prices1, prices2)
                if p_value > self.config.max_cointegration_pvalue:
                    continue
                
                # 3. Calculate hedge ratio and spread statistics
                hedge_ratio, intercept = self.stats.calculate_hedge_ratio(prices1, prices2)
                
                # Check hedge ratio bounds (avoid extreme leverage)
                if not 0.25 < hedge_ratio < 4.0:
                    continue
                
                spread = self.stats.calculate_spread(prices1, prices2, hedge_ratio, intercept)
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                
                if spread_std < 0.001:  # Too stable (lowered threshold)
                    continue
                
                # 4. Half-life filter
                half_life = self.stats.calculate_half_life(pd.Series(spread))
                if not (self.config.min_half_life < half_life < self.config.max_half_life):
                    continue
                
                # 5. Hurst exponent filter
                hurst = self.stats.calculate_hurst(spread)
                if hurst >= self.config.max_hurst:
                    continue
                
                # Z-score at end of formation (for tracking, not filtering)
                current_z = (spread[-1] - spread_mean) / spread_std
                
                # Add candidate WITHOUT requiring z-score at entry level
                # Entry decision will be made based on CURRENT bar in _enter_slot
                candidates.append({
                    'symbol1': s1,
                    'symbol2': s2,
                    'hedge_ratio': hedge_ratio,
                    'intercept': intercept,
                    'spread_mean': spread_mean,
                    'spread_std': spread_std,
                    'half_life': half_life,
                    'correlation': corr,
                    'coint_pvalue': p_value,
                    'current_z': current_z,
                    # Score: lower = better (faster mean reversion + stronger cointegration)
                    'score': half_life * p_value
                })
        
        if not candidates:
            return []
        
        # Sort by score ascending (best first)
        candidates.sort(key=lambda x: x['score'])

        # Greedily select non-overlapping pairs
        selected: list[Dict] = []
        used_symbols = set(excluded_symbols)

        for c in candidates:
            s1, s2 = c['symbol1'], c['symbol2']
            if s1 not in used_symbols and s2 not in used_symbols:
                selected.append(c)
                used_symbols.add(s1)
                used_symbols.add(s2)
                if len(selected) >= n:
                    break

        return selected

    def _get_effective_exit_zscore(self, pos: Dict) -> float:
        """Half-life 기반 동적 exit z-score 계산.

        use_dynamic_exit=False이면 고정 exit_zscore 반환.
        True이면 half_life에 비례해 exit threshold를 조정:
          - 빠른 회귀 (low half_life) → 타이트한 exit (0에 가깝게)
          - 느린 회귀 (high half_life) → 느슨한 exit (일찍 청산)

        Args:
            pos: 슬롯의 position_info dict (half_life 포함).

        Returns:
            해당 포지션에 적용할 exit z-score 임계값.
        """
        if not self.config.use_dynamic_exit:
            return self.config.exit_zscore

        half_life = pos.get('half_life', self.config.max_half_life)
        hl_ratio = (half_life - self.config.min_half_life) / (
            self.config.max_half_life - self.config.min_half_life
        )
        hl_ratio = max(0.0, min(1.0, hl_ratio))  # clamp [0, 1]

        return self.config.dynamic_exit_base + hl_ratio * self.config.dynamic_exit_range
    
    def _update_position(
        self,
        state: BacktestState,
        slot: TraderSlotState,
        bar_data: Dict[str, Dict],
        bar_idx: int,
        timestamp: datetime
    ):
        """Update existing position in a slot, check for exit conditions.

        Args:
            state: Current backtest state.
            slot: The trader slot holding the position.
            bar_data: Current bar data for all symbols.
            bar_idx: Current bar index.
            timestamp: Current timestamp.
        """
        pos = slot.position_info
        s1, s2 = slot.current_pair
        
        # Check data availability
        if s1 not in bar_data or s2 not in bar_data:
            self._close_position(state, slot, bar_data, bar_idx, timestamp, 
                               ExitReason.PAIR_UNAVAILABLE)
            return
        
        # Calculate current spread using FROZEN hedge ratio
        current_price1 = bar_data[s1]['close']
        current_price2 = bar_data[s2]['close']
        
        current_spread = self.stats.calculate_spread(
            np.array([current_price1]),
            np.array([current_price2]),
            pos['hedge_ratio'],
            pos['intercept']
        )[0]
        
        # Calculate current z-score using frozen formation statistics
        current_z = (current_spread - pos['spread_mean']) / pos['spread_std']
        
        # Calculate bars held
        bars_held = bar_idx - pos['entry_bar']
        
        # Update best_z (most favorable z reached)
        if pos['signal'] == 1:  # Long spread: z should increase (become less negative)
            if current_z > pos['best_z']:
                pos['best_z'] = current_z
        else:  # Short spread: z should decrease (become less positive)
            if current_z < pos['best_z']:
                pos['best_z'] = current_z

        # Check exit conditions
        exit_reason = None

        # Determine effective exit threshold
        effective_exit_z = self._get_effective_exit_zscore(pos)
        
        # 1. Mean reversion target
        if pos['signal'] == 1:  # Long spread
            if current_z >= -effective_exit_z:
                exit_reason = ExitReason.MEAN_REVERSION
        else:  # Short spread
            if current_z <= effective_exit_z:
                exit_reason = ExitReason.MEAN_REVERSION
        
        # 2. Trailing stop (ratchet stop)
        # z가 entry 대비 충분히 개선된 후, best_z에서 trail_z만큼 되돌림 → 이익 보존
        if self.config.trailing_stop_enabled and exit_reason is None:
            entry_z = pos['entry_z']
            delta = abs(pos['best_z'] - entry_z)
            if delta >= self.config.trailing_stop_activate_delta:
                pos['trailing_active'] = True
            if pos['trailing_active']:
                best_z = pos['best_z']
                trail_z = self.config.trailing_stop_trail_z
                if pos['signal'] == 1:  # Long: z was improving (increasing), now worsened
                    if current_z <= best_z - trail_z:
                        exit_reason = ExitReason.TRAILING_STOP
                else:  # Short: z was improving (decreasing), now worsened
                    if current_z >= best_z + trail_z:
                        exit_reason = ExitReason.TRAILING_STOP

        # 3. Stop loss
        z_change = current_z - pos['entry_z']
        if pos['signal'] == 1:  # Long spread expects z to increase
            if z_change < -self.config.stop_loss_zscore:
                exit_reason = ExitReason.STOP_LOSS
        else:  # Short spread expects z to decrease
            if z_change > self.config.stop_loss_zscore:
                exit_reason = ExitReason.STOP_LOSS
        
        # 4. Max holding period
        if bars_held >= self.config.max_holding_bars:
            exit_reason = ExitReason.MAX_HOLDING
        
        if exit_reason:
            self._close_position(state, slot, bar_data, bar_idx, timestamp, exit_reason)
    
    def _close_position(
        self,
        state: BacktestState,
        slot: TraderSlotState,
        bar_data: Dict[str, Dict],
        bar_idx: int,
        timestamp: datetime,
        exit_reason: ExitReason
    ):
        """Close position in a specific slot and return capital to pool.

        Args:
            state: Current backtest state.
            slot: The trader slot holding the position.
            bar_data: Current bar data for all symbols.
            bar_idx: Current bar index.
            timestamp: Current timestamp.
            exit_reason: Why the position is being closed.
        """
        pos = slot.position_info
        s1, s2 = slot.current_pair
        
        # Get exit prices
        exit_price1 = bar_data.get(s1, {}).get('close', pos['entry_price1'])
        exit_price2 = bar_data.get(s2, {}).get('close', pos['entry_price2'])
        
        # Calculate exit spread using FROZEN hedge ratio
        exit_spread = self.stats.calculate_spread(
            np.array([exit_price1]),
            np.array([exit_price2]),
            pos['hedge_ratio'],
            pos['intercept']
        )[0]
        
        exit_z = (exit_spread - pos['spread_mean']) / pos['spread_std']
        
        # Calculate PnL from spread change
        # PnL is proportional to NOTIONAL (leveraged exposure), not margin
        # Must divide by (1 + hedge_ratio) because capital is split between legs:
        #   s1_capital = notional / (1 + HR), s2_capital = HR * notional / (1 + HR)
        #   actual PnL = signal * spread_change * notional / (1 + HR)
        notional = pos.get('notional', pos['capital'])  # backward compat
        spread_change = exit_spread - pos['entry_spread']
        hr = pos['hedge_ratio']
        gross_pnl = pos['signal'] * spread_change * notional / (1 + hr)
        
        # Calculate transaction costs (also on notional, not margin)
        # Fees: entry + exit both sides
        total_fees = self.config.trading_fee * 2 * notional
        
        # Slippage: proportional to notional size
        total_slippage = self.config.slippage * 2 * notional
        
        # Funding fee: both legs pay funding proportional to notional
        # Conservative: assume BOTH legs pay (worst-case for pair trade)
        # funding_cost = funding_rate_8h × 2 (legs) × notional × (bars_held / bars_per_8h)
        bars_held = bar_idx - pos['entry_bar']
        funding_periods = bars_held / self.config.bars_per_funding_period
        total_funding = self.config.funding_rate_8h * 2 * notional * funding_periods
        
        net_pnl = gross_pnl - total_fees - total_slippage - total_funding
        # Return percentage is relative to MARGIN (actual cash invested)
        margin = pos['capital']
        net_return_pct = (net_pnl / margin) * 100 if margin > 0 else 0.0
        
        # Record trade
        trade = TradeResult(
            pair=slot.current_pair,
            entry_time=pos['entry_time'],
            exit_time=timestamp,
            entry_z=pos['entry_z'],
            exit_z=exit_z,
            entry_spread=pos['entry_spread'],
            exit_spread=exit_spread,
            signal=pos['signal'],
            capital_allocated=margin,
            gross_pnl=gross_pnl,
            fees=total_fees,
            slippage_cost=total_slippage,
            funding_cost=total_funding,
            net_pnl=net_pnl,
            net_return_pct=net_return_pct,
            bars_held=bars_held,
            exit_reason=exit_reason,
            hedge_ratio=pos['hedge_ratio'],
            spread_mean=pos['spread_mean'],
            spread_std=pos['spread_std'],
            slot_id=slot.slot_id,
            best_z=pos.get('best_z', pos['entry_z']),
        )
        
        state.trades.append(trade)
        
        # Return margin + PnL to pool (margin is what was deducted on entry)
        state.current_capital += margin + net_pnl
        state.peak_capital = max(state.peak_capital, state.total_equity)
        
        # Reset slot state
        slot.reset()
        
        logger.debug(
            f"Slot {trade.slot_id}: Closed {s1}/{s2}: reason={exit_reason.value}, "
            f"PnL=${net_pnl:.2f} ({net_return_pct:.2f}%)"
        )
    
    def _record_equity(
        self,
        state: BacktestState,
        bar_data: Dict[str, Dict],
        timestamp: datetime
    ):
        """Record equity curve point, aggregating across all active slots.

        Args:
            state: Current backtest state.
            bar_data: Current bar data for all symbols.
            timestamp: Current timestamp.
        """
        # Calculate unrealized PnL summed across all active slots
        unrealized_pnl = 0.0
        active_count = 0

        for slot in state.active_slots:
            pos = slot.position_info
            s1, s2 = slot.current_pair
            active_count += 1

            if s1 in bar_data and s2 in bar_data:
                current_price1 = bar_data[s1]['close']
                current_price2 = bar_data[s2]['close']

                current_spread = self.stats.calculate_spread(
                    np.array([current_price1]),
                    np.array([current_price2]),
                    pos['hedge_ratio'],
                    pos['intercept']
                )[0]

                spread_change = current_spread - pos['entry_spread']
                # Unrealized PnL is based on notional (leveraged) exposure
                # Divided by (1 + HR) to match actual leg-split capital allocation
                notional = pos.get('notional', pos['capital'])
                hr = pos.get('hedge_ratio', 1.0)
                slot_unrealized = pos['signal'] * spread_change * notional / (1 + hr)

                # Deduct accrued funding cost from unrealized PnL
                bars_held = len(state.equity_curve) - pos.get('entry_bar', 0)
                if bars_held < 0:
                    bars_held = 0
                funding_periods = bars_held / self.config.bars_per_funding_period
                accrued_funding = self.config.funding_rate_8h * 2 * notional * funding_periods
                slot_unrealized -= accrued_funding

                unrealized_pnl += slot_unrealized

        total_equity = state.current_capital + state.total_allocated + unrealized_pnl
        drawdown = (
            (state.peak_capital - total_equity) / state.peak_capital
            if state.peak_capital > 0 else 0
        )

        # Update peak capital with mark-to-market equity
        if total_equity > state.peak_capital:
            state.peak_capital = total_equity

        state.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': state.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'drawdown': drawdown,
            'in_position': active_count > 0,
            'active_slots': active_count,
        })
    
    def _compile_results(self, state: BacktestState) -> Dict:
        """Compile final backtest results"""
        # Create equity DataFrame
        equity_df = pd.DataFrame(state.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        # Calculate metrics
        metrics = self._calculate_metrics(state, equity_df)
        
        # Create trades DataFrame
        trades_data = []
        for t in state.trades:
            trades_data.append({
                'pair': f"{t.pair[0]}/{t.pair[1]}",
                'slot_id': t.slot_id,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'signal': 'LONG' if t.signal == 1 else 'SHORT',
                'entry_z': t.entry_z,
                'exit_z': t.exit_z,
                'capital': t.capital_allocated,
                'leverage': self.config.leverage,
                'gross_pnl': t.gross_pnl,
                'fees': t.fees,
                'slippage': t.slippage_cost,
                'funding_cost': t.funding_cost,
                'net_pnl': t.net_pnl,
                'return_pct': t.net_return_pct,
                'bars_held': t.bars_held,
                'exit_reason': t.exit_reason.value,
                'hedge_ratio': t.hedge_ratio
            })
        trades_df = pd.DataFrame(trades_data)
        
        return {
            'metrics': metrics,
            'equity_curve': equity_df,
            'trades': trades_df,
            'state': state,
            'config': self.config
        }
    
    def _calculate_metrics(
        self, 
        state: BacktestState,
        equity_df: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics"""
        trades = state.trades
        
        # Basic metrics
        total_return = state.current_capital - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100
        
        # Trade statistics
        n_trades = len(trades)
        if n_trades == 0:
            return {
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'n_trades': 0,
                'win_rate': 0,
                'avg_trade': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'message': 'No trades executed'
            }
        
        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]
        
        win_rate = len(winning) / n_trades * 100
        avg_trade = np.mean([t.net_pnl for t in trades])
        avg_winner = np.mean([t.net_pnl for t in winning]) if winning else 0
        avg_loser = np.mean([t.net_pnl for t in losing]) if losing else 0
        
        # Profit factor
        gross_profit = sum(t.net_pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.net_pnl for t in losing)) if losing else 1e-10
        profit_factor = gross_profit / gross_loss
        
        # Drawdown
        max_dd = equity_df['drawdown'].max() * 100 if not equity_df.empty else 0
        
        # Calculate Sharpe ratio
        if not equity_df.empty and len(equity_df) > 1:
            equity_vals = equity_df['equity'].values
            returns = np.diff(equity_vals) / equity_vals[:-1]
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            
            if len(returns) > 0 and np.std(returns) > 0:
                bars_per_year = 365.25 * 24 * (60 / self.config.bar_interval_minutes)
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason.value
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Total fees and slippage paid
        total_fees = sum(t.fees for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        
        # Market participation rate
        if not equity_df.empty:
            in_market_pct = equity_df['in_position'].mean() * 100
            if 'active_slots' in equity_df.columns:
                avg_slot_utilization = (
                    equity_df['active_slots'].mean() / self.config.max_traders * 100
                )
            else:
                avg_slot_utilization = in_market_pct
        else:
            in_market_pct = 0
            avg_slot_utilization = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_capital': state.current_capital,
            'peak_capital': state.peak_capital,
            
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            
            'avg_trade': avg_trade,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'largest_win': max(t.net_pnl for t in trades),
            'largest_loss': min(t.net_pnl for t in trades),
            
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            
            'avg_holding_bars': np.mean([t.bars_held for t in trades]),
            'exit_reasons': exit_reasons,
            
            'total_fees_paid': total_fees,
            'total_slippage_paid': total_slippage,
            'total_costs': total_fees + total_slippage,
            
            'in_market_pct': in_market_pct,
            'avg_slot_utilization': avg_slot_utilization,
            'max_traders': self.config.max_traders,
            'search_count': state.search_count,
            'no_trade_periods': state.no_trade_periods,
        }


def run_continuous_backtest(
    data: Dict[str, pd.DataFrame],
    config: Dict = None
) -> Dict:
    """
    Convenience function to run continuous backtest.
    
    Args:
        data: Symbol -> OHLCV DataFrame
        config: Configuration overrides
        
    Returns:
        Backtest results
    """
    cfg = ContinuousBacktestConfig(**(config or {}))
    engine = ContinuousPairBacktest(cfg)
    return engine.run(data)


def compute_kelly_from_backtest(
    results: Dict,
    fraction: float = 0.5,
) -> Dict:
    """Compute Kelly-optimal position sizing from backtest results.

    Uses actual per-trade net return distribution to calculate the
    optimal fraction of capital per trade. This replaces naive
    fixed-fraction sizing with data-driven sizing.

    Args:
        results: Output from ContinuousPairBacktest.run() or
            run_continuous_backtest().
        fraction: Kelly fraction (0.5 = half-Kelly, default).

    Returns:
        Dict with:
            kelly_position_size_pct: Optimal position_size_pct
            kelly_raw: Raw (full) Kelly fraction
            n_trades: Number of trades used
            win_rate: Win rate from trades
            avg_win_pct: Average winning return %
            avg_loss_pct: Average losing return % (absolute)
            mean_return_pct: Mean per-trade return %
            std_return_pct: Std dev of per-trade return %
            continuous_kelly: Continuous Kelly result
            discrete_kelly: Discrete Kelly result
    """
    from quantbox.core.trader_slot import (
        calculate_kelly_from_trades,
        calculate_kelly_discrete,
    )

    trades_df = results.get('trades')
    if trades_df is None or trades_df.empty:
        return {
            'kelly_position_size_pct': 0.0,
            'kelly_raw': 0.0,
            'n_trades': 0,
            'message': 'No trades available for Kelly calculation',
        }

    returns = trades_df['return_pct'].values
    n_trades = len(returns)

    # Continuous Kelly (uses full distribution)
    continuous = calculate_kelly_from_trades(returns, fraction=fraction)

    # Discrete Kelly (uses summary stats)
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0.0
    avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losers))) if len(losers) > 0 else 0.0

    discrete = calculate_kelly_discrete(
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        fraction=fraction,
    )

    # Use the more conservative of the two estimates
    kelly_pct = min(continuous, discrete)

    return {
        'kelly_position_size_pct': kelly_pct,
        'continuous_kelly': continuous,
        'discrete_kelly': discrete,
        'kelly_raw': kelly_pct / fraction if fraction > 0 else 0.0,
        'n_trades': n_trades,
        'win_rate': win_rate * 100,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'mean_return_pct': float(np.mean(returns)),
        'std_return_pct': float(np.std(returns, ddof=1)) if n_trades > 1 else 0.0,
        'fraction': fraction,
    }
