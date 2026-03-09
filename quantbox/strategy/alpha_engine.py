"""
Alpha Engine — Multi-Factor Signal Generator (Numba Compiled)

Combines multiple alpha sources for hyper-exponential edge:
  1. Rubber Band (fade extreme moves) — base signal
  2. Volume Confirmation — only enter on liquidity surges
  3. Momentum Alignment — trade with the trend, not against it
  4. Adaptive Threshold — scale to recent volatility (ATR-based)
  5. Enhanced Trailing Stop — lock in profits dynamically
  6. Taker Buy Ratio — sentiment filter (aggression asymmetry)

Each factor can be toggled independently for A/B testing.
All heavy computation is Numba-compiled for ~100K bars/sec throughput.

Key insight: The original rubber band strategy has alpha (PF 1.09)
but it's too thin because it enters on noise. Volume + momentum
filters remove low-conviction entries → fewer but much better trades.
"""

import numpy as np
import numba
from dataclasses import dataclass


@dataclass
class AlphaConfig:
    """Multi-factor alpha configuration.

    Three layers of filtering:
      Layer 1: Rubber band base signal (extreme move detection)
      Layer 2: Volume + momentum confirmation
      Layer 3: Smart exit management (trailing stop)
    """

    # ── Rubber Band Base ──
    lookback: int = 8              # Bars to measure move (2h at 15m)
    threshold: float = 0.025      # Minimum move to trigger (2.5%)
    target_retrace: float = 0.6   # Take profit at 60% retracement (was 50%)
    stop_mult: float = 1.0        # Stop at 1x the move size
    max_hold: int = 12            # Max holding period (was 8, now 3h)

    # ── Volume Filter ──
    vol_filter: bool = True        # Enable volume confirmation
    vol_lookback: int = 48         # 12h volume average baseline
    vol_mult: float = 1.5          # Require 1.5x average volume

    # ── Momentum Filter ──
    momentum_filter: bool = True   # Only trade with the trend
    momentum_lookback: int = 96    # 24h momentum direction
    momentum_mode: int = 0         # 0=with_trend, 1=counter_trend

    # ── Adaptive Threshold ──
    adaptive_threshold: bool = True
    atr_lookback: int = 48         # 12h for ATR calculation
    atr_mult: float = 2.5          # threshold = atr_mult × ATR

    # ── Enhanced Trailing Stop ──
    trailing_stop: bool = True
    trail_activation: float = 0.3  # Activate after 30% of target reached
    trail_distance: float = 0.4    # Trail at 40% of peak unrealized PnL

    # ── Taker Buy Ratio Filter ──
    taker_filter: bool = False     # Taker buy ratio confirmation
    taker_lookback: int = 16       # 4h for taker ratio baseline
    taker_threshold: float = 0.55  # 55% taker buy = bullish bias

    # ── Momentum Breakout (alternative strategy) ──
    breakout_mode: bool = False    # Pure momentum breakout
    breakout_lookback: int = 32    # 8h high/low breakout

    # ── Costs ──
    fee_rate: float = 0.0002
    slippage_rate: float = 0.0002
    funding_rate_8h: float = 0.0001
    bar_interval_minutes: int = 15

    # ── Risk ──
    risk_per_trade: float = 0.01
    max_concurrent: int = 5
    max_leverage: float = 20.0

    @property
    def cost_per_side(self) -> float:
        return self.fee_rate + self.slippage_rate

    @property
    def bars_per_funding(self) -> int:
        return (8 * 60) // self.bar_interval_minutes

    @property
    def funding_per_bar(self) -> float:
        return self.funding_rate_8h / self.bars_per_funding


# ═══════════════════════════════════════════════════════════════════════
#  NUMBA SIGNAL ENGINES
# ═══════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int) -> np.ndarray:
    """Compute Average True Range (rolling)."""
    n = len(close)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i - 1]),
                 abs(low[i] - close[i - 1]))
        if i < period:
            # Simple average for warmup
            total = 0.0
            for j in range(max(1, i - period + 1), i + 1):
                total += max(high[j] - low[j],
                             abs(high[j] - close[j - 1]),
                             abs(low[j] - close[j - 1]))
            atr[i] = total / min(i, period)
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr


@numba.jit(nopython=True, cache=True)
def _compute_vol_ratio(volume: np.ndarray, lookback: int) -> np.ndarray:
    """Compute volume relative to rolling average."""
    n = len(volume)
    ratio = np.ones(n)
    for i in range(lookback, n):
        avg = 0.0
        for j in range(i - lookback, i):
            avg += volume[j]
        avg /= lookback
        if avg > 0:
            ratio[i] = volume[i] / avg
    return ratio


@numba.jit(nopython=True, cache=True)
def _compute_taker_ratio(taker_buy_vol: np.ndarray, total_vol: np.ndarray,
                          lookback: int) -> np.ndarray:
    """Compute rolling taker buy ratio."""
    n = len(taker_buy_vol)
    ratio = np.full(n, 0.5)
    for i in range(lookback, n):
        tb = 0.0
        tv = 0.0
        for j in range(i - lookback, i + 1):
            tb += taker_buy_vol[j]
            tv += total_vol[j]
        if tv > 0:
            ratio[i] = tb / tv
    return ratio


@numba.jit(nopython=True, cache=True)
def compute_alpha_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    taker_buy_volume: np.ndarray,
    # Rubber band params
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    # Volume filter
    vol_filter: bool,
    vol_lookback: int,
    vol_mult: float,
    # Momentum filter
    momentum_filter: bool,
    momentum_lookback: int,
    momentum_mode: int,
    # Adaptive threshold
    adaptive_threshold: bool,
    atr_lookback: int,
    atr_mult: float,
    # Enhanced trailing stop
    trailing_stop: bool,
    trail_activation: float,
    trail_distance: float,
    # Taker filter
    taker_filter: bool,
    taker_lookback: int,
    taker_threshold: float,
    # Breakout mode
    breakout_mode: bool,
    breakout_lookback: int,
) -> np.ndarray:
    """Multi-factor alpha signal engine.

    Args:
        close, high, low, volume, taker_buy_volume: Price/volume arrays.
        lookback..max_hold: Rubber band base parameters.
        vol_filter..vol_mult: Volume confirmation filter.
        momentum_filter..momentum_mode: Trend alignment filter.
        adaptive_threshold..atr_mult: Volatility-adaptive threshold.
        trailing_stop..trail_distance: Enhanced trailing stop parameters.
        taker_filter..taker_threshold: Taker buy ratio filter.
        breakout_mode, breakout_lookback: Momentum breakout alternative.

    Returns:
        Signal array: -1 (short), 0 (flat), +1 (long).
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute indicators
    atr = _compute_atr(high, low, close, atr_lookback) if adaptive_threshold else np.zeros(n)
    vol_ratio = _compute_vol_ratio(volume, vol_lookback) if vol_filter else np.ones(n)
    taker_ratio = _compute_taker_ratio(taker_buy_volume, volume, taker_lookback) if taker_filter else np.full(n, 0.5)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    warmup = max(lookback, momentum_lookback, vol_lookback, atr_lookback, breakout_lookback)

    for i in range(warmup, n):
        ret = close[i] / close[i - lookback] - 1.0

        # Adaptive threshold: scale to recent volatility
        if adaptive_threshold and atr[i] > 0 and close[i] > 0:
            current_threshold = atr[i] / close[i] * atr_mult
            # Floor at 60% of base threshold, cap at 200%
            current_threshold = max(threshold * 0.6, min(current_threshold, threshold * 2.0))
        else:
            current_threshold = threshold

        if position == 0:
            # ── Entry logic ──
            entry_signal = 0

            if breakout_mode:
                # Momentum breakout: break above/below N-bar high/low
                highest = close[i]
                lowest = close[i]
                for k in range(1, breakout_lookback + 1):
                    if close[i - k] > highest:
                        highest = close[i - k]
                    if close[i - k] < lowest:
                        lowest = close[i - k]

                if close[i] > highest and ret > current_threshold * 0.5:
                    entry_signal = 1  # Long breakout
                elif close[i] < lowest and ret < -current_threshold * 0.5:
                    entry_signal = -1  # Short breakout
            else:
                # Rubber band: fade extreme moves
                if ret < -current_threshold:
                    entry_signal = 1   # Long: fade the drop
                elif ret > current_threshold:
                    entry_signal = -1  # Short: fade the spike

            if entry_signal != 0:
                # ── Apply filters ──
                take_trade = True

                # Volume filter: require above-average volume
                if vol_filter and vol_ratio[i] < vol_mult:
                    take_trade = False

                # Momentum filter
                if momentum_filter and take_trade:
                    mom = close[i] / close[i - momentum_lookback] - 1.0
                    if momentum_mode == 0:
                        # With trend: only long if momentum up, short if down
                        if entry_signal == 1 and mom < 0:
                            take_trade = False
                        elif entry_signal == -1 and mom > 0:
                            take_trade = False
                    else:
                        # Counter trend (original rubber band is counter-trend)
                        # Allow entries opposite to momentum
                        if entry_signal == 1 and mom > 0.02:
                            take_trade = False  # Don't long if already ripping up
                        elif entry_signal == -1 and mom < -0.02:
                            take_trade = False

                # Taker buy ratio filter
                if taker_filter and take_trade:
                    if entry_signal == 1 and taker_ratio[i] < (1.0 - taker_threshold):
                        take_trade = False  # Don't long if sellers dominate
                    elif entry_signal == -1 and taker_ratio[i] > taker_threshold:
                        take_trade = False  # Don't short if buyers dominate

                if take_trade:
                    position = entry_signal
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
        else:
            # ── Exit logic ──
            hold_count += 1

            if position == 1:
                gain = (close[i] - entry_price) / entry_price  # noqa: F841
                worst_gain = (low[i] - entry_price) / entry_price
                best_gain = (high[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price  # noqa: F841
                worst_gain = (entry_price - high[i]) / entry_price
                best_gain = (entry_price - low[i]) / entry_price

            if best_gain > best_pnl:
                best_pnl = best_gain

            target = move_size * target_retrace

            # 1. Target hit (intra-bar best)
            if best_gain >= target:
                position = 0
            # 2. Stop loss (intra-bar worst)
            elif worst_gain < -move_size * stop_mult:
                position = 0
            # 3. Enhanced trailing stop (v12.5.3: worst_gain for intra-bar parity)
            elif trailing_stop:
                if best_pnl >= target * trail_activation:
                    # Trail: exit if we give back too much
                    trail_level = best_pnl * (1.0 - trail_distance)
                    if worst_gain < trail_level:
                        position = 0
                # Fallback: old trailing stop for positions without enough gain
                elif best_pnl > target * 0.5 and worst_gain < best_pnl * 0.3:
                    position = 0
            else:
                # Original trailing logic
                if best_pnl > target * 0.5 and worst_gain < best_pnl * 0.3:
                    position = 0
            # 4. Max holding period
            if hold_count >= max_hold and position != 0:
                position = 0

        sig[i] = position

    return sig


@numba.jit(nopython=True, cache=True)
def compute_momentum_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    # Momentum params
    fast_lookback: int,
    slow_lookback: int,
    breakout_lookback: int,
    atr_lookback: int,
    vol_lookback: int,
    vol_mult: float,
    # Exit params
    trail_atr_mult: float,
    max_hold: int,
) -> np.ndarray:
    """Pure momentum strategy — ride trends, not fade them.

    Enters on breakout above/below N-bar high/low with
    moving average confirmation and volume surge.

    Args:
        close, high, low, volume: OHLCV data.
        fast_lookback: Fast MA period.
        slow_lookback: Slow MA period.
        breakout_lookback: N-bar high/low for breakout.
        atr_lookback: ATR period for trailing stop.
        vol_lookback: Volume average period.
        vol_mult: Volume threshold multiplier.
        trail_atr_mult: ATR mult for trailing stop.
        max_hold: Maximum bars to hold.

    Returns:
        Signal array: -1/0/+1.
    """
    n = len(close)
    sig = np.zeros(n)

    atr = _compute_atr(high, low, close, atr_lookback)
    vol_ratio = _compute_vol_ratio(volume, vol_lookback)

    position = 0
    entry_price = 0.0
    hold_count = 0
    best_pnl = 0.0
    trail_stop = 0.0

    warmup = max(fast_lookback, slow_lookback, breakout_lookback, vol_lookback, atr_lookback)

    for i in range(warmup, n):
        # Compute moving averages
        fast_ma = 0.0
        for j in range(i - fast_lookback, i):
            fast_ma += close[j]
        fast_ma /= fast_lookback

        slow_ma = 0.0
        for j in range(i - slow_lookback, i):
            slow_ma += close[j]
        slow_ma /= slow_lookback

        if position == 0:
            # Entry: breakout + MA alignment + volume
            highest = close[i - 1]
            lowest = close[i - 1]
            for k in range(1, breakout_lookback):
                if close[i - 1 - k] > highest:
                    highest = close[i - 1 - k]
                if close[i - 1 - k] < lowest:
                    lowest = close[i - 1 - k]

            vol_ok = vol_ratio[i] >= vol_mult

            if close[i] > highest and fast_ma > slow_ma and vol_ok:
                position = 1
                entry_price = close[i]
                hold_count = 0
                best_pnl = 0.0
                trail_stop = close[i] - atr[i] * trail_atr_mult
            elif close[i] < lowest and fast_ma < slow_ma and vol_ok:
                position = -1
                entry_price = close[i]
                hold_count = 0
                best_pnl = 0.0
                trail_stop = close[i] + atr[i] * trail_atr_mult
        else:
            hold_count += 1

            if position == 1:
                gain = (close[i] - entry_price) / entry_price  # noqa: F841
                # Update trailing stop
                new_trail = close[i] - atr[i] * trail_atr_mult
                if new_trail > trail_stop:
                    trail_stop = new_trail
                # Exit
                if close[i] < trail_stop:
                    position = 0
            else:
                gain = (entry_price - close[i]) / entry_price  # noqa: F841
                new_trail = close[i] + atr[i] * trail_atr_mult
                if new_trail < trail_stop:
                    trail_stop = new_trail
                if close[i] > trail_stop:
                    position = 0

            if gain > best_pnl:
                best_pnl = gain

            if hold_count >= max_hold and position != 0:
                position = 0

        sig[i] = position

    return sig


@numba.jit(nopython=True, cache=True)
def compute_hybrid_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    taker_buy_volume: np.ndarray,
    # Trend params
    trend_lookback: int,
    trend_threshold: float,
    # Rubber band params
    rb_lookback: int,
    rb_threshold: float,
    rb_target: float,
    rb_stop: float,
    rb_max_hold: int,
    # Volume filter
    vol_lookback: int,
    vol_mult: float,
    # ATR for adaptive
    atr_lookback: int,
    atr_mult: float,
    # Trailing stop
    trail_activation: float,
    trail_distance: float,
) -> np.ndarray:
    """Hybrid Strategy: Momentum direction + Rubber band timing.

    The best of both worlds:
      - Determine trend direction from longer timeframe
      - Only take rubber band entries IN the trend direction
      - Wait for the dip/rally to enter with the trend

    This is the "buy the dip in an uptrend" strategy, Numba-compiled.

    Args:
        trend_lookback: Bars for trend determination.
        trend_threshold: Minimum trend strength to confirm direction.
        rb_*: Rubber band entry/exit parameters.
        vol_*: Volume filter parameters.
        atr_*: ATR adaptivity parameters.
        trail_*: Trailing stop parameters.

    Returns:
        Signal array: -1/0/+1.
    """
    n = len(close)
    sig = np.zeros(n)

    atr = _compute_atr(high, low, close, atr_lookback)
    vol_ratio = _compute_vol_ratio(volume, vol_lookback)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    warmup = max(trend_lookback, rb_lookback, vol_lookback, atr_lookback)

    for i in range(warmup, n):
        # Determine trend direction
        trend_ret = close[i] / close[i - trend_lookback] - 1.0

        # Also check shorter trend for momentum
        short_trend = close[i] / close[i - rb_lookback] - 1.0

        # Adaptive threshold
        if atr[i] > 0 and close[i] > 0:
            adaptive_thr = atr[i] / close[i] * atr_mult
            adaptive_thr = max(rb_threshold * 0.5, min(adaptive_thr, rb_threshold * 2.5))
        else:
            adaptive_thr = rb_threshold

        if position == 0:
            # Only enter rubber band dips in the trend direction
            entry_signal = 0

            if trend_ret > trend_threshold:
                # UPTREND: buy dips only
                if short_trend < -adaptive_thr:
                    entry_signal = 1  # Buy the dip in uptrend

            elif trend_ret < -trend_threshold:
                # DOWNTREND: sell rallies only
                if short_trend > adaptive_thr:
                    entry_signal = -1  # Sell the rally in downtrend

            # No entry in neutral/sideways — avoid chop

            if entry_signal != 0:
                # Volume confirmation
                if vol_ratio[i] >= vol_mult:
                    position = entry_signal
                    entry_price = close[i]
                    move_size = abs(short_trend)
                    hold_count = 0
                    best_pnl = 0.0
        else:
            hold_count += 1

            if position == 1:
                gain = (close[i] - entry_price) / entry_price  # noqa: F841
            else:
                gain = (entry_price - close[i]) / entry_price  # noqa: F841

            if gain > best_pnl:
                best_pnl = gain

            target = move_size * rb_target

            # Exit conditions
            if gain >= target:
                position = 0
            elif gain < -move_size * rb_stop:
                position = 0
            elif best_pnl >= target * trail_activation:
                trail_level = best_pnl * (1.0 - trail_distance)
                if gain < trail_level:
                    position = 0
            elif hold_count >= rb_max_hold:
                position = 0

        sig[i] = position

    return sig


# ═══════════════════════════════════════════════════════════════════════
#  LIVE STRATEGY WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class AlphaStrategy:
    """Stateful wrapper for live trading with any alpha signal engine."""

    def __init__(self, config: AlphaConfig = None):
        self.config = config or AlphaConfig()
        self._positions: dict[str, dict] = {}

    def on_bar(
        self,
        symbol: str,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        taker_buy_volume: np.ndarray | None = None,
    ) -> int:
        """Process a new bar and return signal."""
        cfg = self.config
        n = len(close)
        if n < max(cfg.lookback, cfg.momentum_lookback, cfg.vol_lookback,
                   cfg.atr_lookback) + 1:
            return 0

        if taker_buy_volume is None:
            taker_buy_volume = volume * 0.5  # Default 50/50 split

        sigs = compute_alpha_signals(
            close, high, low, volume, taker_buy_volume,
            cfg.lookback, cfg.threshold, cfg.target_retrace,
            cfg.stop_mult, cfg.max_hold,
            cfg.vol_filter, cfg.vol_lookback, cfg.vol_mult,
            cfg.momentum_filter, cfg.momentum_lookback, cfg.momentum_mode,
            cfg.adaptive_threshold, cfg.atr_lookback, cfg.atr_mult,
            cfg.trailing_stop, cfg.trail_activation, cfg.trail_distance,
            cfg.taker_filter, cfg.taker_lookback, cfg.taker_threshold,
            cfg.breakout_mode, cfg.breakout_lookback,
        )
        return int(sigs[-1])

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def open_position(self, symbol: str, signal: int, entry_price: float,
                      notional: float, leverage: float, move_size: float):
        self._positions[symbol] = {
            'signal': signal, 'entry_price': entry_price,
            'notional': notional, 'leverage': leverage,
            'move_size': move_size, 'bars_held': 0, 'best_pnl': 0.0,
        }

    def close_position(self, symbol: str) -> dict | None:
        return self._positions.pop(symbol, None)

    @property
    def active_positions(self) -> dict[str, dict]:
        return dict(self._positions)

    @property
    def n_active(self) -> int:
        return len(self._positions)
