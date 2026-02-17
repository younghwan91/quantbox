"""
Elliott Wave Strategy — Maeoknam Championship Method

Captures new impulse waves emerging after ABC corrective wave completion.
Implements the exact method from winning_strategy.md:

Alpha source:
  - ABC corrective wave completion at Fibonacci coordinates (0.618, 1.618)
  - Confluence: RSI/MACD divergence + Stochastic cross confirmation
  - Asymmetric R:R via strict 10% margin cap + pyramiding on Wave 3/C

Capital management:
  - 10% max margin of total capital (effective leverage 2x-5x)
  - 2% exploratory entry -> 8% primary on confirmation
  - 50% partial TP -> trail remainder to trend exhaustion
  - Pyramiding only on Wave 3/C with trailing stop at BE

Execution flow:
  1. BTC 200/500 EMA macro bias (Long/Short filter)
  2. ZigZag -> ABC corrective wave completion detection
  3. Fibonacci validation (B: 0.5-0.618 retrace, C: 1.0-1.618 extension)
  4. Divergence confluence (RSI failure swing + MACD/Stochastic)
  5. 2% exploratory entry -> 8% primary on signal validation
  6. 50% partial TP at primary target -> trail remaining

Timeframes:
  - Macro bias: BTC 4h vs 200/500 EMA
  - Wave detection: symbol 4h OHLC (ZigZag + Fibonacci)
  - Entry timing: 15m OHLC (divergence + confluence verification)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numba
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ElliottConfig:
    """Elliott Wave strategy parameters.

    All values calibrated to winning_strategy.md specification.
    """

    # ── ZigZag swing detection ──
    zigzag_threshold: float = 0.03       # 3% min swing on 4h
    zigzag_threshold_15m: float = 0.015  # 1.5% on 15m (divergence check)

    # ── EMA periods (winning_strategy §2.1) ──
    ema_periods: tuple[int, ...] = (20, 50, 100, 200, 500)
    ema_macro_fast: int = 200   # BTC macro trend — fast line
    ema_macro_slow: int = 500   # BTC macro trend — slow line

    # ── RSI (winning_strategy §2.1) ──
    rsi_period: int = 14
    rsi_dynamic_period: int = 9   # High-vol dynamic adjustment
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # ── MACD (standard 12/26/9) ──
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9

    # ── Stochastic ──
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # ── Fibonacci ratios (winning_strategy §3.1) ──
    # ABC B retrace of A: valid range
    fib_b_retrace_min: float = 0.40
    fib_b_retrace_max: float = 0.72
    # ABC C vs A length: valid range
    fib_c_vs_a_min: float = 0.80
    fib_c_vs_a_max: float = 1.80
    # C termination near 0.618 Fib of prior impulse: max distance tolerance
    fib_c_impulse_tolerance: float = 0.12

    # ── Capital management (winning_strategy §4) ──
    max_margin_pct: float = 0.10     # 10% max of total capital
    exploratory_pct: float = 0.02    # 2% initial pilot
    primary_pct: float = 0.08        # 8% on confirmation
    partial_tp_fraction: float = 0.50  # 50% take-profit

    # ── Leverage ──
    nominal_leverage: int = 20  # Exchange-set (effective = nominal × margin%)

    # ── Confirmation & position management ──
    confirmation_bars: int = 4   # Max bars to wait for primary confirmation
    max_hold_bars: int = 96      # Max hold ~24h at 15m (4h-scale trades)
    trail_atr_mult: float = 2.0  # Trailing stop distance in ATR units

    # ── Pyramiding (winning_strategy §4.2) ──
    pyramid_enabled: bool = True
    pyramid_min_profit_pct: float = 0.02  # Min 2% unrealized before pyramid

    # ── Costs ──
    fee_rate: float = 0.0005       # 0.05% taker
    slippage_rate: float = 0.0002  # 0.02% slippage
    funding_rate_8h: float = 0.0001

    # ── Risk limits ──
    max_concurrent: int = 3

    # ── Divergence lookback (bars of 15m for failure swing detection) ──
    divergence_lookback: int = 30

    # ── Wedge/triangle detection ──
    wedge_min_pivots: int = 6    # Minimum alternating pivots for wedge
    wedge_converge_ratio: float = 0.70  # High-range/Low-range ratio < this

    # ── Bar interval ──
    bar_interval_minutes: int = 15


# ═══════════════════════════════════════════════════════════════════════
#  Numba-compiled indicator functions
# ═══════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.

    Args:
        close: Close price array.
        period: EMA period.

    Returns:
        EMA array of same length as close.
    """
    n = len(close)
    ema = np.empty(n)
    ema[0] = close[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]
    return ema


@numba.jit(nopython=True, cache=True)
def compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI using Wilder smoothing.

    Args:
        close: Close price array.
        period: RSI period (default 14).

    Returns:
        RSI array of same length (first `period` bars are approximate).
    """
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    # Seed averages
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        delta = close[i] - close[i - 1]
        if delta > 0:
            avg_gain += delta
        else:
            avg_loss -= delta
    avg_gain /= period
    avg_loss /= period

    if avg_gain == 0.0 and avg_loss == 0.0:
        rsi[period] = 50.0  # No movement → neutral
    elif avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    else:
        rsi[period] = 100.0

    # Wilder smoothing
    for i in range(period + 1, n):
        delta = close[i] - close[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_gain == 0.0 and avg_loss == 0.0:
            rsi[i] = 50.0
        elif avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi[i] = 100.0

    return rsi


@numba.jit(nopython=True, cache=True)
def compute_macd(
    close: np.ndarray, fast: int, slow: int, signal_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, and histogram.

    Args:
        close: Close price array.
        fast: Fast EMA period (12).
        slow: Slow EMA period (26).
        signal_period: Signal line EMA period (9).

    Returns:
        (macd_line, signal_line, histogram) — all same length as close.
    """
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@numba.jit(nopython=True, cache=True)
def compute_stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    k_period: int, d_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic %K and %D.

    Args:
        high: High price array.
        low: Low price array.
        close: Close price array.
        k_period: %K lookback period.
        d_period: %D smoothing period (SMA of %K).

    Returns:
        (k, d) arrays of same length as close.
    """
    n = len(close)
    k = np.full(n, 50.0)

    for i in range(k_period - 1, n):
        hh = high[i]
        ll = low[i]
        for j in range(i - k_period + 1, i + 1):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        if hh > ll:
            k[i] = (close[i] - ll) / (hh - ll) * 100.0
        else:
            k[i] = 50.0

    # %D = SMA of %K
    d = np.full(n, 50.0)
    for i in range(k_period + d_period - 2, n):
        s = 0.0
        for j in range(i - d_period + 1, i + 1):
            s += k[j]
        d[i] = s / d_period

    return k, d


@numba.jit(nopython=True, cache=True)
def compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int,
) -> np.ndarray:
    """Average True Range (Wilder smoothing).

    Args:
        high: High price array.
        low: Low price array.
        close: Close price array.
        period: ATR period.

    Returns:
        ATR array of same length as close.
    """
    n = len(close)
    atr = np.zeros(n)
    if n < 2:
        return atr

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    # Seed
    if n < period + 1:
        s = 0.0
        for i in range(n):
            s += tr[i]
        atr[-1] = s / n
        return atr

    s = 0.0
    for i in range(period):
        s += tr[i]
    atr[period - 1] = s / period

    # Wilder smooth
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ═══════════════════════════════════════════════════════════════════════
#  ZigZag swing detection (Numba)
# ═══════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def zigzag_pivots(
    high: np.ndarray, low: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Standard ZigZag pivot detection.

    Identifies alternating swing highs (+1) and swing lows (-1) with a
    minimum percentage reversal threshold.

    Args:
        high: High price array.
        low: Low price array.
        threshold: Minimum % reversal to register new pivot (e.g. 0.03 = 3%).

    Returns:
        (pivot_idx, pivot_price, pivot_type, n_pivots):
        - pivot_idx:   int64 bar indices of each pivot.
        - pivot_price: Price at each pivot.
        - pivot_type:  +1 = swing high, -1 = swing low.
        - n_pivots:    Number of pivots found.
    """
    n = len(high)
    pivot_idx = np.empty(n, dtype=np.int64)
    pivot_price = np.empty(n)
    pivot_type = np.empty(n, dtype=np.int64)
    count = 0

    if n < 2:
        return pivot_idx[:0], pivot_price[:0], pivot_type[:0], 0

    # Track current extremes
    cur_high = high[0]
    cur_high_idx = 0
    cur_low = low[0]
    cur_low_idx = 0
    trend = 0  # 0=undefined, 1=up, -1=down

    for i in range(1, n):
        if high[i] > cur_high:
            cur_high = high[i]
            cur_high_idx = i
        if low[i] < cur_low:
            cur_low = low[i]
            cur_low_idx = i

        if trend == 0:
            # Determine initial direction
            up_move = (cur_high - low[0]) / low[0] if low[0] > 0 else 0.0
            down_move = (high[0] - cur_low) / high[0] if high[0] > 0 else 0.0

            if up_move >= threshold and up_move >= down_move:
                trend = 1
                pivot_idx[0] = 0
                pivot_price[0] = low[0]
                pivot_type[0] = -1
                count = 1
            elif down_move >= threshold:
                trend = -1
                pivot_idx[0] = 0
                pivot_price[0] = high[0]
                pivot_type[0] = 1
                count = 1

        elif trend == 1:
            # Uptrend: track highest, check reversal down
            drop = (cur_high - low[i]) / cur_high if cur_high > 0 else 0.0
            if drop >= threshold:
                # Register swing high
                pivot_idx[count] = cur_high_idx
                pivot_price[count] = cur_high
                pivot_type[count] = 1
                count += 1
                # Switch to downtrend
                trend = -1
                cur_low = low[i]
                cur_low_idx = i
                cur_high = high[i]
                cur_high_idx = i

        else:  # trend == -1
            # Downtrend: track lowest, check reversal up
            rise = (high[i] - cur_low) / cur_low if cur_low > 0 else 0.0
            if rise >= threshold:
                # Register swing low
                pivot_idx[count] = cur_low_idx
                pivot_price[count] = cur_low
                pivot_type[count] = -1
                count += 1
                # Switch to uptrend
                trend = 1
                cur_high = high[i]
                cur_high_idx = i
                cur_low = low[i]
                cur_low_idx = i

    # Register pending final pivot
    if count > 0:
        if trend == 1 and count < n:
            pivot_idx[count] = cur_high_idx
            pivot_price[count] = cur_high
            pivot_type[count] = 1
            count += 1
        elif trend == -1 and count < n:
            pivot_idx[count] = cur_low_idx
            pivot_price[count] = cur_low
            pivot_type[count] = -1
            count += 1

    return pivot_idx[:count], pivot_price[:count], pivot_type[:count], count


# ═══════════════════════════════════════════════════════════════════════
#  Divergence detection (Numba)
# ═══════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def detect_bullish_divergence(
    close: np.ndarray, rsi: np.ndarray, lookback: int,
) -> bool:
    """Bullish RSI divergence: lower low in price + higher low in RSI.

    Winning_strategy §2.2: "If the price prints a lower low while the RSI
    registers a higher low, the algorithm classifies downward momentum
    as exhausted."

    Args:
        close: Close price array.
        rsi: RSI array (same length).
        lookback: Number of bars to search for local minima.

    Returns:
        True if bullish divergence detected near the last bar.
    """
    n = len(close)
    if n < lookback + 2:
        return False

    start = n - lookback
    low_idx_1 = -1  # Most recent local minimum
    low_idx_2 = -1  # Previous local minimum

    for i in range(n - 2, start, -1):
        if close[i] <= close[i - 1] and close[i] <= close[i + 1]:
            if low_idx_1 == -1:
                low_idx_1 = i
            elif low_idx_2 == -1:
                low_idx_2 = i
                break

    if low_idx_1 == -1 or low_idx_2 == -1:
        return False

    # Price made lower low, RSI made higher low
    return close[low_idx_1] < close[low_idx_2] and rsi[low_idx_1] > rsi[low_idx_2]


@numba.jit(nopython=True, cache=True)
def detect_bearish_divergence(
    close: np.ndarray, rsi: np.ndarray, lookback: int,
) -> bool:
    """Bearish RSI divergence: higher high in price + lower high in RSI.

    Args:
        close: Close price array.
        rsi: RSI array (same length).
        lookback: Number of bars to search for local maxima.

    Returns:
        True if bearish divergence detected near the last bar.
    """
    n = len(close)
    if n < lookback + 2:
        return False

    start = n - lookback
    high_idx_1 = -1
    high_idx_2 = -1

    for i in range(n - 2, start, -1):
        if close[i] >= close[i - 1] and close[i] >= close[i + 1]:
            if high_idx_1 == -1:
                high_idx_1 = i
            elif high_idx_2 == -1:
                high_idx_2 = i
                break

    if high_idx_1 == -1 or high_idx_2 == -1:
        return False

    # Price made higher high, RSI made lower high
    return close[high_idx_1] > close[high_idx_2] and rsi[high_idx_1] < rsi[high_idx_2]


@numba.jit(nopython=True, cache=True)
def check_macd_reversal(histogram: np.ndarray, direction: int) -> bool:
    """Check MACD histogram reversal.

    Winning_strategy §2.2: "verify a reversal in the MACD histogram"

    Args:
        histogram: MACD histogram array.
        direction: +1 = bullish (histogram turning up from negative),
                   -1 = bearish (turning down from positive).

    Returns:
        True if histogram is reversing in the specified direction.
    """
    n = len(histogram)
    if n < 3:
        return False

    if direction == 1:
        # Bullish: histogram in negative territory and slope turning up
        return histogram[n - 2] < 0 and histogram[n - 1] > histogram[n - 2]
    else:
        # Bearish: histogram in positive territory and slope turning down
        return histogram[n - 2] > 0 and histogram[n - 1] < histogram[n - 2]


@numba.jit(nopython=True, cache=True)
def check_stoch_cross(
    k: np.ndarray, d: np.ndarray, direction: int,
) -> bool:
    """Check Stochastic %K/%D crossover.

    Winning_strategy §2.2: "golden/dead cross in the Stochastic oscillator"

    Args:
        k: Stochastic %K array.
        d: Stochastic %D array.
        direction: +1 = bullish (K crosses above D in oversold zone),
                   -1 = bearish (K crosses below D in overbought zone).

    Returns:
        True if crossover confirmed at the last bar.
    """
    n = len(k)
    if n < 2:
        return False

    if direction == 1:
        # Bullish: K was below D, now above or equal; in oversold zone
        return k[n - 2] < d[n - 2] and k[n - 1] >= d[n - 1] and k[n - 1] < 30.0
    else:
        # Bearish: K was above D, now below or equal; in overbought zone
        return k[n - 2] > d[n - 2] and k[n - 1] <= d[n - 1] and k[n - 1] > 70.0


# ═══════════════════════════════════════════════════════════════════════
#  ABC corrective wave pattern detection
# ═══════════════════════════════════════════════════════════════════════


def detect_abc_patterns(
    pivot_idx: np.ndarray,
    pivot_price: np.ndarray,
    pivot_type: np.ndarray,
    n_pivots: int,
    config: ElliottConfig,
) -> list[dict]:
    """Detect completed ABC corrective wave patterns.

    Requires 5 consecutive alternating pivots to identify:
      Prior impulse (L→H or H→L) + ABC correction (3 swings)

    Winning_strategy §3.1:
      - ABC B retraces 0.5-0.618 of Wave A
      - ABC C extends 1.0-1.618 of Wave A length
      - C termination near 0.618 Fib of prior impulse

    Long setup (bearish ABC completing):
      Low(-1) → High(+1) → Low_A(-1) → High_B(+1) → Low_C(-1)
      Entry at C termination, expecting new impulse UP.

    Short setup (bullish ABC completing):
      High(+1) → Low(-1) → High_A(+1) → Low_B(-1) → High_C(+1)
      Entry at C termination, expecting new impulse DOWN.

    Args:
        pivot_idx: Bar indices of zigzag pivots.
        pivot_price: Prices at pivots.
        pivot_type: +1 (swing high) or -1 (swing low).
        n_pivots: Number of valid pivots.
        config: Strategy configuration.

    Returns:
        List of pattern dicts with entry signal, targets, and metadata.
    """
    patterns: list[dict] = []
    if n_pivots < 5:
        return patterns

    # Scan all 5-pivot sequences
    for i in range(n_pivots - 4):
        p_types = (
            pivot_type[i], pivot_type[i + 1], pivot_type[i + 2],
            pivot_type[i + 3], pivot_type[i + 4],
        )
        p_prices = (
            pivot_price[i], pivot_price[i + 1], pivot_price[i + 2],
            pivot_price[i + 3], pivot_price[i + 4],
        )
        p_indices = (
            pivot_idx[i], pivot_idx[i + 1], pivot_idx[i + 2],
            pivot_idx[i + 3], pivot_idx[i + 4],
        )

        # ── Long setup: Low → High → Low(A) → High(B) → Low(C) ──
        if p_types == (-1, 1, -1, 1, -1):
            L, H, A, B, C = p_prices
            impulse = H - L  # Prior impulse UP
            wave_a_len = H - A  # A decline (correction down)
            wave_b_len = B - A  # B rally (partial recovery)
            wave_c_len = B - C  # C decline (final leg down)

            if impulse <= 0 or wave_a_len <= 0 or wave_b_len <= 0 or wave_c_len <= 0:
                continue

            # B must not exceed H (winning_strategy §3.1)
            if B > H:
                continue

            # B retracement of A (0.5-0.618 target)
            b_retrace = wave_b_len / wave_a_len

            # C vs A length ratio (1.0-1.618 target)
            c_vs_a = wave_c_len / wave_a_len

            # C near 0.618 Fib retracement of prior impulse
            fib_618 = H - 0.618 * impulse
            c_fib_dist = abs(C - fib_618) / impulse if impulse > 0 else 1.0

            if (config.fib_b_retrace_min <= b_retrace <= config.fib_b_retrace_max
                    and config.fib_c_vs_a_min <= c_vs_a <= config.fib_c_vs_a_max
                    and c_fib_dist <= config.fib_c_impulse_tolerance):
                patterns.append({
                    'signal': 1,  # Long
                    'type': 'bearish_abc_long',
                    'impulse_start': float(L),
                    'impulse_end': float(H),
                    'a_price': float(A),
                    'b_price': float(B),
                    'c_price': float(C),
                    'c_idx': int(p_indices[4]),
                    'b_retrace': float(b_retrace),
                    'c_vs_a': float(c_vs_a),
                    'c_fib_dist': float(c_fib_dist),
                    # Targets: B level (primary), H level (extended)
                    'target_1': float(B),
                    'target_2': float(H),
                    # Stop below C by 10% of wave A length
                    'stop': float(C - wave_a_len * 0.10),
                    'impulse_size': float(impulse),
                })

        # ── Short setup: High → Low → High(A) → Low(B) → High(C) ──
        elif p_types == (1, -1, 1, -1, 1):
            H, L, A, B, C = p_prices
            impulse = H - L  # Prior impulse DOWN (H to L)
            wave_a_len = A - L  # A rally (correction up)
            wave_b_len = A - B  # B decline (partial retracement)
            wave_c_len = C - B  # C rally (final leg up)

            if impulse <= 0 or wave_a_len <= 0 or wave_b_len <= 0 or wave_c_len <= 0:
                continue

            # B must not go below L
            if B < L:
                continue

            b_retrace = wave_b_len / wave_a_len
            c_vs_a = wave_c_len / wave_a_len

            # C near 0.618 Fib retracement of prior impulse (from the low)
            fib_618 = L + 0.618 * impulse
            c_fib_dist = abs(C - fib_618) / impulse if impulse > 0 else 1.0

            if (config.fib_b_retrace_min <= b_retrace <= config.fib_b_retrace_max
                    and config.fib_c_vs_a_min <= c_vs_a <= config.fib_c_vs_a_max
                    and c_fib_dist <= config.fib_c_impulse_tolerance):
                patterns.append({
                    'signal': -1,  # Short
                    'type': 'bullish_abc_short',
                    'impulse_start': float(H),
                    'impulse_end': float(L),
                    'a_price': float(A),
                    'b_price': float(B),
                    'c_price': float(C),
                    'c_idx': int(p_indices[4]),
                    'b_retrace': float(b_retrace),
                    'c_vs_a': float(c_vs_a),
                    'c_fib_dist': float(c_fib_dist),
                    'target_1': float(B),
                    'target_2': float(L),
                    'stop': float(C + wave_a_len * 0.10),
                    'impulse_size': float(impulse),
                })

    return patterns


# ═══════════════════════════════════════════════════════════════════════
#  Ending Diagonal (Wedge) detection
# ═══════════════════════════════════════════════════════════════════════


def detect_ending_diagonal(
    pivot_price: np.ndarray,
    pivot_type: np.ndarray,
    n_pivots: int,
    min_pivots: int = 6,
    converge_ratio: float = 0.70,
) -> bool:
    """Detect ending diagonal (wedge) pattern.

    Winning_strategy §3.2: "Occurs at the exhaustion point of a trend.
    The agent utilizes this as a termination signal for Wave 5 or Wave C."

    A wedge is a sequence of alternating swing highs and lows where
    the range between successive highs and successive lows narrows
    (converging trendlines).

    Args:
        pivot_price: Prices at zigzag pivots.
        pivot_type: +1 or -1 for each pivot.
        n_pivots: Number of valid pivots.
        min_pivots: Minimum pivots for detection.
        converge_ratio: Last range / first range must be below this.

    Returns:
        True if converging wedge detected in the latest pivots.
    """
    if n_pivots < min_pivots:
        return False

    # Collect recent swing highs and lows
    recent = n_pivots
    highs: list[float] = []
    lows: list[float] = []
    for i in range(max(0, recent - 10), recent):
        if pivot_type[i] == 1:
            highs.append(float(pivot_price[i]))
        else:
            lows.append(float(pivot_price[i]))

    if len(highs) < 3 or len(lows) < 3:
        return False

    # Check convergence: range between latest highs vs earliest highs shrinking
    # AND range between latest lows vs earliest lows shrinking
    high_range_first = abs(highs[1] - highs[0])
    high_range_last = abs(highs[-1] - highs[-2])
    low_range_first = abs(lows[1] - lows[0])
    low_range_last = abs(lows[-1] - lows[-2])

    if high_range_first <= 0 or low_range_first <= 0:
        return False

    h_ratio = high_range_last / high_range_first
    l_ratio = low_range_last / low_range_first

    return h_ratio < converge_ratio and l_ratio < converge_ratio


def detect_triangle(
    pivot_price: np.ndarray,
    pivot_type: np.ndarray,
    n_pivots: int,
) -> bool:
    """Detect triangle (consolidation) pattern.

    Winning_strategy §3.2: "Triangles typically manifest in Wave 4 or Wave B.
    The agent must either exclude the E wave from trading or deploy highly
    conservative parameters."

    A triangle shows contracting swing highs AND expanding swing lows
    (symmetrical triangle), or both contracting (ascending/descending).

    Args:
        pivot_price: Prices at zigzag pivots.
        pivot_type: +1 or -1 for each pivot.
        n_pivots: Number of valid pivots.

    Returns:
        True if triangle pattern detected (signals caution / skip entry).
    """
    if n_pivots < 8:
        return False

    # Last 8 pivots
    highs: list[float] = []
    lows: list[float] = []
    for i in range(n_pivots - 8, n_pivots):
        if pivot_type[i] == 1:
            highs.append(float(pivot_price[i]))
        else:
            lows.append(float(pivot_price[i]))

    if len(highs) < 3 or len(lows) < 3:
        return False

    # Contracting: swing highs decreasing AND swing lows increasing
    highs_decreasing = all(highs[i] >= highs[i + 1] for i in range(len(highs) - 1))
    lows_increasing = all(lows[i] <= lows[i + 1] for i in range(len(lows) - 1))

    return highs_decreasing and lows_increasing


# ═══════════════════════════════════════════════════════════════════════
#  ElliottStrategy — stateful wrapper for live trading
# ═══════════════════════════════════════════════════════════════════════


class ElliottStrategy:
    """Elliott Wave strategy for live trading.

    Mirrors the RubberBandStrategy interface so the scalper can use either
    strategy interchangeably via duck typing.

    Position lifecycle:
      1. on_bar() detects ABC completion → returns ±1 (exploratory entry)
      2. Scalper opens 2% position
      3. update_position() fires 'confirm_entry' on momentum confirmation
      4. Scalper adds 8% (upgrades to primary 10% margin)
      5. update_position() fires 'partial_close' at target_1 (50% TP)
      6. Remaining trails until 'trailing_stop' or 'counter_divergence'

    Args:
        config: ElliottConfig with strategy parameters.
    """

    def __init__(self, config: ElliottConfig | None = None):
        self.config = config or ElliottConfig()
        self._positions: dict[str, dict] = {}

        # Per-symbol market data (15m)
        self._close_data: dict[str, np.ndarray] = {}
        self._high_data: dict[str, np.ndarray] = {}
        self._low_data: dict[str, np.ndarray] = {}
        self._volume_data: dict[str, np.ndarray] = {}
        self._taker_data: dict[str, np.ndarray] = {}

        # Per-symbol 4h OHLC data
        self._htf_close: dict[str, np.ndarray] = {}
        self._htf_high: dict[str, np.ndarray] = {}
        self._htf_low: dict[str, np.ndarray] = {}

        # BTC 4h for macro bias
        self._btc_close_4h: np.ndarray | None = None

        # Pending pattern cache (after on_bar signals, before position open)
        self._pending_patterns: dict[str, dict] = {}

        # Track MTF confirmed (compatibility with scalper API)
        self._mtf_confirmed: dict[str, bool] = {}

    # ── Data update methods (same signature as RubberBandStrategy) ──

    def update_market_data(
        self,
        symbol: str,
        volume: np.ndarray,
        taker_buy: np.ndarray,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
    ) -> None:
        """Update 15m OHLCV data for divergence/confluence checks.

        Args:
            symbol: Trading symbol.
            volume: Volume array.
            taker_buy: Taker buy volume array.
            high: 15m high prices.
            low: 15m low prices.
        """
        self._volume_data[symbol] = volume
        self._taker_data[symbol] = taker_buy
        if high is not None:
            self._high_data[symbol] = high
        if low is not None:
            self._low_data[symbol] = low

    def update_htf_data(
        self,
        symbol: str,
        close_4h: np.ndarray,
        high_4h: np.ndarray | None = None,
        low_4h: np.ndarray | None = None,
    ) -> None:
        """Update 4h OHLC data for wave detection.

        Args:
            symbol: Trading symbol.
            close_4h: 4h close prices (need 500+ bars for 500 EMA).
            high_4h: 4h high prices (needed for ZigZag).
            low_4h: 4h low prices (needed for ZigZag).
        """
        self._htf_close[symbol] = close_4h
        if high_4h is not None:
            self._htf_high[symbol] = high_4h
        if low_4h is not None:
            self._htf_low[symbol] = low_4h

    def update_btc_data(self, btc_close_4h: np.ndarray) -> None:
        """Update BTC 4h close for macro bias.

        Args:
            btc_close_4h: BTC 4h close array (500+ bars for 500 EMA).
        """
        self._btc_close_4h = btc_close_4h

    # ── Macro bias ──

    def macro_bias(self) -> str:
        """Determine macro trend direction from BTC vs 200/500 EMAs.

        Winning_strategy §5.1: "Determine the macroeconomic bias (Long/Short)
        by analyzing BTC's position relative to the 200/500 EMAs."

        Returns:
            'long': BTC above both 200 and 500 EMA → only long entries.
            'short': BTC below both 200 and 500 EMA → only short entries.
            'neutral': Between the two → both directions allowed.
        """
        if self._btc_close_4h is None or len(self._btc_close_4h) < self.config.ema_macro_slow + 50:
            return 'neutral'

        ema200 = compute_ema(self._btc_close_4h, self.config.ema_macro_fast)
        ema500 = compute_ema(self._btc_close_4h, self.config.ema_macro_slow)
        current = float(self._btc_close_4h[-1])

        if current > float(ema200[-1]) and current > float(ema500[-1]):
            return 'long'
        elif current < float(ema200[-1]) and current < float(ema500[-1]):
            return 'short'
        return 'neutral'

    # ── Entry signal ──

    def on_bar(
        self,
        symbol: str,
        close_prices: np.ndarray,
        btc_close_prices: np.ndarray | None = None,
    ) -> int:
        """Process a new 15m bar and return entry signal.

        Execution flow (winning_strategy §5):
          1. Check macro bias (BTC vs 200/500 EMA) — directional filter
          2. ZigZag on 4h → detect ABC corrective wave completion
          3. Fibonacci validation (B retrace, C extension, C near 0.618 Fib)
          4. Triangle filter (skip if contracting triangle detected)
          5. Exploratory entry signal ±1

        Divergence confirmation happens post-entry in update_position()
        to match the 2%→8% phased execution model.

        Args:
            symbol: Trading symbol.
            close_prices: 15m close prices (200+ bars).
            btc_close_prices: BTC 15m close prices (unused here, kept for API).

        Returns:
            +1 (long), -1 (short), or 0 (flat).
        """
        cfg = self.config

        # Store latest 15m close for later divergence checks
        self._close_data[symbol] = close_prices

        # 1. Macro bias filter
        bias = self.macro_bias()

        # 2. Get 4h data for wave detection
        close_4h = self._htf_close.get(symbol)
        high_4h = self._htf_high.get(symbol)
        low_4h = self._htf_low.get(symbol)

        if close_4h is None or high_4h is None or low_4h is None:
            return 0
        if len(close_4h) < 100:
            return 0

        # 3. ZigZag on 4h data
        pidx, pprice, ptype, n_piv = zigzag_pivots(
            high_4h, low_4h, cfg.zigzag_threshold,
        )
        if n_piv < 5:
            return 0

        # 4. Triangle filter (winning_strategy §3.2)
        if detect_triangle(pprice, ptype, n_piv):
            logger.debug(f"[{symbol}] Triangle detected — skipping entry")
            return 0

        # 5. Detect ABC patterns with Fibonacci validation
        patterns = detect_abc_patterns(pidx, pprice, ptype, n_piv, cfg)
        if not patterns:
            return 0

        # Only consider patterns where C completed recently (last 3 bars of 4h)
        n_4h = len(close_4h)
        recent = [p for p in patterns if p['c_idx'] >= n_4h - 3]
        if not recent:
            return 0

        # Pick the most recent pattern
        pattern = recent[-1]

        # 6. Check macro bias alignment
        if bias == 'long' and pattern['signal'] != 1:
            return 0
        if bias == 'short' and pattern['signal'] != -1:
            return 0

        # 7. Store for position creation
        self._pending_patterns[symbol] = pattern
        self._mtf_confirmed[symbol] = True

        logger.info(
            f"[{symbol}] ABC pattern detected: {pattern['type']} | "
            f"B_retrace={pattern['b_retrace']:.3f} | "
            f"C/A={pattern['c_vs_a']:.3f} | "
            f"C_fib_dist={pattern['c_fib_dist']:.3f} | "
            f"target_1={pattern['target_1']:.4f}"
        )

        return pattern['signal']

    def on_bar_4h(
        self,
        symbol: str,
        close_4h: np.ndarray,
        btc_close_4h: np.ndarray | None = None,
    ) -> int:
        """No independent 4h signal for Elliott (uses 4h for wave detection only)."""
        return 0

    def was_mtf_confirmed(self, symbol: str) -> bool:
        """Whether the last signal had full wave + Fib confirmation."""
        return self._mtf_confirmed.get(symbol, False)

    # ── Position management ──

    def open_position(
        self,
        symbol: str,
        signal: int,
        entry_price: float,
        notional: float,
        leverage: float,
        move_size: float,
        entry_stop_mult: float | None = None,
        server_sl_order_id: str | None = None,
        server_tp_order_id: str | None = None,
    ) -> None:
        """Record a new position.

        Initializes in 'exploratory' phase (2% capital).
        Will upgrade to 'primary' (10%) via update_position() → 'confirm_entry'.

        Args:
            symbol: Trading symbol.
            signal: +1 (long) or -1 (short).
            entry_price: Fill price.
            notional: Position notional.
            leverage: Applied leverage.
            move_size: Move magnitude that triggered the signal.
            entry_stop_mult: Effective stop multiplier.
            server_sl_order_id: Exchange stop-loss order ID.
            server_tp_order_id: Exchange take-profit order ID.
        """
        pattern = self._pending_patterns.pop(symbol, {})

        self._positions[symbol] = {
            'signal': signal,
            'entry_price': entry_price,
            'notional': notional,
            'leverage': leverage,
            'move_size': move_size,
            'bars_held': 0,
            'best_pnl': 0.0,
            'entry_stop_mult': entry_stop_mult or 0.03,
            'partial_closed': False,
            'server_sl_order_id': server_sl_order_id,
            'server_tp_order_id': server_tp_order_id,
            # ── Elliott-specific ──
            'phase': 'exploratory',       # exploratory → primary → trailing
            'confirmation_countdown': self.config.confirmation_bars,
            'target_1': pattern.get('target_1', entry_price * 1.02),
            'target_2': pattern.get('target_2', entry_price * 1.05),
            'stop_price': pattern.get('stop', entry_price * 0.97),
            'impulse_size': pattern.get('impulse_size', move_size),
            'pyramid_done': False,
            'wedge_detected': False,
        }

    def close_position(self, symbol: str) -> dict | None:
        """Remove position and return its info."""
        return self._positions.pop(symbol, None)

    def get_effective_stop_mult(self, close_prices: np.ndarray) -> float:
        """Return a fixed stop multiplier for Elliott.

        Elliott uses pattern-based stops (stop_price), not dynamic mult.
        This method exists for scalper API compatibility.
        """
        return 0.03

    def update_position(
        self, symbol: str, current_price: float,
    ) -> str | None:
        """Check position status and return action string.

        Position lifecycle:
          1. Exploratory phase: wait for divergence confirmation
             → 'confirm_entry' or 'cancel_exploration'
          2. Primary phase: watch for target_1 → 'partial_close'
          3. After partial: trail to BE → 'breakeven_stop'
          4. Pyramid opportunity → 'pyramid'
          5. Wedge exit → 'wedge_exit'
          6. Counter-divergence → 'counter_divergence'
          7. Max hold → 'max_hold'

        Args:
            symbol: Trading symbol.
            current_price: Current market price.

        Returns:
            Action string or None (hold).
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        pos['bars_held'] += 1
        cfg = self.config

        # Calculate unrealized PnL
        if pos['signal'] == 1:
            gain = (current_price - pos['entry_price']) / pos['entry_price']
        else:
            gain = (pos['entry_price'] - current_price) / pos['entry_price']

        if gain > pos['best_pnl']:
            pos['best_pnl'] = gain

        # ── Phase 1: Exploratory — waiting for confirmation ──
        if pos['phase'] == 'exploratory':
            pos['confirmation_countdown'] -= 1

            # Check stop loss
            if pos['signal'] == 1 and current_price <= pos['stop_price']:
                return 'stop_loss'
            if pos['signal'] == -1 and current_price >= pos['stop_price']:
                return 'stop_loss'

            # Check divergence confirmation on 15m data
            if self._check_confirmation(symbol, pos):
                pos['phase'] = 'primary'
                pos['best_pnl'] = gain
                return 'confirm_entry'

            # Confirmation timeout → cancel
            if pos['confirmation_countdown'] <= 0:
                return 'cancel_exploration'

            return None

        # ── Phase 2+: Primary / Trailing ──

        # After partial close: breakeven stop takes priority
        if pos['partial_closed']:
            if gain <= 0:
                return 'breakeven_stop'

        # Hard stop loss (before partial close)
        if not pos['partial_closed']:
            if pos['signal'] == 1 and current_price <= pos['stop_price']:
                return 'stop_loss'
            if pos['signal'] == -1 and current_price >= pos['stop_price']:
                return 'stop_loss'

        # 50% partial TP at target_1 (winning_strategy §5.5)
        if not pos['partial_closed']:
            hit_target = False
            if pos['signal'] == 1 and current_price >= pos['target_1']:
                hit_target = True
            elif pos['signal'] == -1 and current_price <= pos['target_1']:
                hit_target = True

            if hit_target:
                pos['partial_closed'] = True
                pos['phase'] = 'trailing'
                # Trail stop to breakeven
                pos['stop_price'] = pos['entry_price']
                pos['best_pnl'] = gain
                return 'partial_close'

        # Trailing stop: protect 50% of best unrealized gain
        if pos['partial_closed'] and pos['best_pnl'] > 0.01:
            trail_level = pos['best_pnl'] * 0.50
            if gain < trail_level:
                return 'trailing_stop'

        # Extended target_2 (full exit, rare — trend exhaustion)
        if pos['partial_closed']:
            hit_ext = False
            if pos['signal'] == 1 and current_price >= pos['target_2']:
                hit_ext = True
            elif pos['signal'] == -1 and current_price <= pos['target_2']:
                hit_ext = True
            if hit_ext:
                return 'target'

        # Pyramiding check (winning_strategy §4.2)
        if (cfg.pyramid_enabled
                and not pos['pyramid_done']
                and not pos['partial_closed']
                and gain >= cfg.pyramid_min_profit_pct):
            # Check for Wave 3 extension confirmation via 4h pivots
            if self._check_pyramid_opportunity(symbol, pos, current_price):
                pos['pyramid_done'] = True
                pos['stop_price'] = pos['entry_price']  # Trail to BE
                return 'pyramid'

        # Wedge detection (winning_strategy §3.2 — exit signal)
        if not pos['wedge_detected']:
            if self._check_wedge(symbol):
                pos['wedge_detected'] = True
                if not pos['partial_closed']:
                    pos['partial_closed'] = True
                    pos['phase'] = 'trailing'
                    pos['stop_price'] = pos['entry_price']
                    return 'partial_close'

        # Counter-divergence exit (trend exhaustion)
        if pos['bars_held'] > 8 and self._check_counter_divergence(symbol, pos):
            return 'counter_divergence'

        # Max hold
        if pos['bars_held'] >= cfg.max_hold_bars:
            return 'max_hold'

        return None

    # ── Internal checks ──

    def _check_confirmation(self, symbol: str, pos: dict) -> bool:
        """Check if divergence + momentum confirms the exploratory entry.

        Winning_strategy §5.3: "Verified RSI/MACD divergence +
        Stochastic %K/%D crossover confirmed."

        Uses 15m data for confirmation granularity.

        Returns:
            True if at least 2 of 3 confirmation signals fire.
        """
        close = self._close_data.get(symbol)
        high = self._high_data.get(symbol)
        low = self._low_data.get(symbol)

        if close is None or high is None or low is None:
            return False
        if len(close) < 30:
            return False

        cfg = self.config
        direction = pos['signal']

        rsi = compute_rsi(close, cfg.rsi_period)
        _, _, histogram = compute_macd(
            close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
        )
        k, d = compute_stochastic(
            high, low, close, cfg.stoch_k_period, cfg.stoch_d_period,
        )

        confirmations = 0

        # RSI divergence
        if direction == 1:
            if detect_bullish_divergence(close, rsi, cfg.divergence_lookback):
                confirmations += 1
        else:
            if detect_bearish_divergence(close, rsi, cfg.divergence_lookback):
                confirmations += 1

        # MACD histogram reversal
        if check_macd_reversal(histogram, direction):
            confirmations += 1

        # Stochastic cross
        if check_stoch_cross(k, d, direction):
            confirmations += 1

        return confirmations >= 2

    def _check_pyramid_opportunity(
        self, symbol: str, pos: dict, current_price: float,
    ) -> bool:
        """Check if current price action suggests Wave 3/C momentum.

        Simple heuristic: price has moved beyond the B level of the original
        ABC pattern (momentum carrying through) and all EMAs align.

        Returns:
            True if pyramiding is warranted.
        """
        target_1 = pos.get('target_1', 0)
        if target_1 <= 0:
            return False

        # Price must be beyond midpoint to target_1
        entry = pos['entry_price']
        midpoint = (entry + target_1) / 2.0

        if pos['signal'] == 1 and current_price < midpoint:
            return False
        if pos['signal'] == -1 and current_price > midpoint:
            return False

        # Check EMA alignment on 4h
        close_4h = self._htf_close.get(symbol)
        if close_4h is None or len(close_4h) < 50:
            return False

        ema20 = compute_ema(close_4h, 20)
        ema50 = compute_ema(close_4h, 50)

        if pos['signal'] == 1:
            return float(ema20[-1]) > float(ema50[-1])  # Bullish alignment
        else:
            return float(ema20[-1]) < float(ema50[-1])  # Bearish alignment

    def _check_wedge(self, symbol: str) -> bool:
        """Check for ending diagonal / wedge on 4h data."""
        close_4h = self._htf_close.get(symbol)
        high_4h = self._htf_high.get(symbol)
        low_4h = self._htf_low.get(symbol)

        if close_4h is None or high_4h is None or low_4h is None:
            return False
        if len(close_4h) < 50:
            return False

        cfg = self.config
        pidx, pprice, ptype, n_piv = zigzag_pivots(
            high_4h, low_4h, cfg.zigzag_threshold,
        )
        return detect_ending_diagonal(
            pprice, ptype, n_piv,
            cfg.wedge_min_pivots, cfg.wedge_converge_ratio,
        )

    def _check_counter_divergence(self, symbol: str, pos: dict) -> bool:
        """Check for divergence against the current position direction.

        Winning_strategy §5.5: "Trail until trend exhaustion metrics
        (e.g., counter-divergence) are triggered."

        If we are long and bearish divergence appears → exit signal.
        If we are short and bullish divergence appears → exit signal.
        """
        close = self._close_data.get(symbol)
        if close is None or len(close) < 30:
            return False

        rsi = compute_rsi(close, self.config.rsi_period)

        if pos['signal'] == 1:
            return detect_bearish_divergence(
                close, rsi, self.config.divergence_lookback,
            )
        else:
            return detect_bullish_divergence(
                close, rsi, self.config.divergence_lookback,
            )

    # ── Position query (same interface as RubberBandStrategy) ──

    def has_position(self, symbol: str) -> bool:
        """Check if symbol has an active position."""
        return symbol in self._positions

    @property
    def active_positions(self) -> dict[str, dict]:
        """All active positions."""
        return dict(self._positions)

    @property
    def n_active(self) -> int:
        """Number of active positions."""
        return len(self._positions)


# ═══════════════════════════════════════════════════════════════════════
#  Backtest signal generation
# ═══════════════════════════════════════════════════════════════════════


def compute_elliott_signals(
    close_4h: np.ndarray,
    high_4h: np.ndarray,
    low_4h: np.ndarray,
    btc_close_4h: np.ndarray,
    config: ElliottConfig | None = None,
) -> np.ndarray:
    """Generate Elliott wave signals at 4h granularity for backtesting.

    Processes the full 4h dataset and returns a signal array where:
      +1 = long entry (ABC bearish correction completed)
      -1 = short entry (ABC bullish correction completed)
       0 = flat

    Includes macro bias filtering (BTC vs 200/500 EMA).

    Args:
        close_4h: Symbol 4h close prices.
        high_4h: Symbol 4h high prices.
        low_4h: Symbol 4h low prices.
        btc_close_4h: BTC 4h close prices.
        config: Strategy configuration.

    Returns:
        Signal array aligned to close_4h (same length).
    """
    cfg = config or ElliottConfig()
    n = len(close_4h)
    signals = np.zeros(n, dtype=np.float64)

    if n < cfg.ema_macro_slow + 50:
        return signals

    # Pre-compute BTC EMAs for macro bias
    btc_ema200 = compute_ema(btc_close_4h, cfg.ema_macro_fast)
    btc_ema500 = compute_ema(btc_close_4h, cfg.ema_macro_slow)

    # Pre-compute RSI on symbol 4h for divergence
    rsi_4h = compute_rsi(close_4h, cfg.rsi_period)

    # Walk-forward: at each bar, run ZigZag on data up to that bar
    min_start = cfg.ema_macro_slow + 50  # Need enough warmup

    for bar in range(min_start, n):
        # Macro bias at this bar
        btc_price = float(btc_close_4h[bar])
        above_200 = btc_price > float(btc_ema200[bar])
        above_500 = btc_price > float(btc_ema500[bar])

        if above_200 and above_500:
            bias = 'long'
        elif not above_200 and not above_500:
            bias = 'short'
        else:
            bias = 'neutral'

        # ZigZag on data up to this bar
        pidx, pprice, ptype, n_piv = zigzag_pivots(
            high_4h[:bar + 1], low_4h[:bar + 1], cfg.zigzag_threshold,
        )
        if n_piv < 5:
            continue

        # Triangle filter
        if detect_triangle(pprice, ptype, n_piv):
            continue

        # ABC pattern detection
        patterns = detect_abc_patterns(pidx, pprice, ptype, n_piv, cfg)
        if not patterns:
            continue

        # Only consider patterns completing at or near the current bar
        recent = [p for p in patterns if p['c_idx'] >= bar - 2]
        if not recent:
            continue

        pattern = recent[-1]

        # Macro bias filter
        if bias == 'long' and pattern['signal'] != 1:
            continue
        if bias == 'short' and pattern['signal'] != -1:
            continue

        # Simple divergence check on 4h
        direction = pattern['signal']
        lookback = min(30, bar)
        if direction == 1:
            has_div = detect_bullish_divergence(
                close_4h[:bar + 1], rsi_4h[:bar + 1], lookback,
            )
        else:
            has_div = detect_bearish_divergence(
                close_4h[:bar + 1], rsi_4h[:bar + 1], lookback,
            )

        if has_div:
            signals[bar] = float(direction)

    return signals
