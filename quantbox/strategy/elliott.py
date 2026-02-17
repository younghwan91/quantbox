"""
Elliott Wave Strategy 

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

    # ── 5-Wave Impulse Fibonacci ratios (winning_strategy §3.1 table) ──
    # Wave 2 retraces Wave 1: doc says 0.5, 0.618, 0.764
    w2_retrace_min: float = 0.382
    w2_retrace_max: float = 0.886
    # Wave 3 extends Wave 1: doc says 1.618, 2.618
    w3_extension_min: float = 1.272
    w3_extension_max: float = 4.236
    # Wave 4 retraces Wave 3: doc says 0.236, 0.382
    w4_retrace_min: float = 0.146
    w4_retrace_max: float = 0.500

    # ── ABC Corrective Fibonacci ratios (winning_strategy §3.1 table) ──
    # Wave B retraces Wave A: doc says 0.5, 0.618
    fib_b_retrace_min: float = 0.382
    fib_b_retrace_max: float = 0.786
    # Wave C extends Wave A: doc says 1.0, 1.618
    fib_c_vs_a_min: float = 0.786
    fib_c_vs_a_max: float = 2.000
    # C termination near 0.618 Fib of entire impulse: distance tolerance
    fib_c_impulse_tolerance: float = 0.20

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

    # ── Effective leverage for backtest (winning_strategy §4.1) ──
    # 10% margin × 20x nominal = 2.0x effective leverage on total equity
    effective_leverage: float = 2.0

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
#  Full 5-Wave Impulse + ABC Corrective Pattern Detection
# ═══════════════════════════════════════════════════════════════════════


def detect_impulse_abc_patterns(
    pivot_idx: np.ndarray,
    pivot_price: np.ndarray,
    pivot_type: np.ndarray,
    n_pivots: int,
    config: ElliottConfig,
) -> list[dict]:
    """Detect completed 5-wave impulse + ABC corrective wave patterns.

    Requires 9 consecutive alternating pivots:
      5-wave impulse (6 pivots: origin + 5 wave endpoints)
      + ABC correction (3 more pivots: A, B, C endpoints)

    Winning_strategy §3.1 Fibonacci validation:
      Impulse:
        - Wave 2 retraces 0.5/0.618/0.764 of Wave 1 (not beyond origin)
        - Wave 3 extends 1.618/2.618 of Wave 1 (strongest phase)
        - Wave 4 retraces 0.236/0.382 of Wave 3 (must NOT enter Wave 1 territory)

      ABC Correction:
        - Wave B retraces 0.5/0.618 of Wave A (cannot exceed Wave A origin)
        - Wave C extends 1.0/1.618 of Wave A length
        - Wave C terminates near 0.618 Fibonacci retracement of entire impulse

    Bullish setup (bearish ABC completing after bullish impulse):
      Impulse UP: W0(L) → W1(H) → W2(L) → W3(H) → W4(L) → W5(H)
      ABC DOWN:   W5(H) = A_origin → A(L) → B(H) → C(L)
      Pivots: -1, +1, -1, +1, -1, +1, -1, +1, -1
      Entry at C completion → new impulse UP expected.

    Bearish setup (bullish ABC completing after bearish impulse):
      Impulse DOWN: W0(H) → W1(L) → W2(H) → W3(L) → W4(H) → W5(L)
      ABC UP:       W5(L) = A_origin → A(H) → B(L) → C(H)
      Pivots: +1, -1, +1, -1, +1, -1, +1, -1, +1
      Entry at C completion → new impulse DOWN expected.

    Targets use Fibonacci extensions of Wave 1 from point C:
      TP1: C ± 1.618 × Wave1_length (conservative)
      TP2: C ± 2.618 × Wave1_length (extended)

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
    if n_pivots < 9:
        return patterns

    for i in range(n_pivots - 8):
        # ── Bullish setup: 9 pivots L,H,L,H,L,H,L,H,L ──
        types_bull = True
        expected_bull = -1
        for k in range(9):
            if pivot_type[i + k] != expected_bull:
                types_bull = False
                break
            expected_bull = -expected_bull

        if types_bull:
            p = [float(pivot_price[i + j]) for j in range(9)]
            W0, W1, W2, W3, W4, W5, A, B, C = p

            pattern = _validate_bullish_impulse_abc(
                W0, W1, W2, W3, W4, W5, A, B, C,
                int(pivot_idx[i + 8]), config,
            )
            if pattern is not None:
                patterns.append(pattern)
            continue

        # ── Bearish setup: 9 pivots H,L,H,L,H,L,H,L,H ──
        types_bear = True
        expected_bear = 1
        for k in range(9):
            if pivot_type[i + k] != expected_bear:
                types_bear = False
                break
            expected_bear = -expected_bear

        if types_bear:
            p = [float(pivot_price[i + j]) for j in range(9)]
            W0, W1, W2, W3, W4, W5, A, B, C = p

            pattern = _validate_bearish_impulse_abc(
                W0, W1, W2, W3, W4, W5, A, B, C,
                int(pivot_idx[i + 8]), config,
            )
            if pattern is not None:
                patterns.append(pattern)

    return patterns


def _validate_bullish_impulse_abc(
    W0: float, W1: float, W2: float, W3: float, W4: float,
    W5: float, A: float, B: float, C: float,
    c_idx: int, config: ElliottConfig,
) -> dict | None:
    """Validate bullish impulse (UP) + bearish ABC (DOWN) → Long entry.

    Impulse UP: W0(low) → W1(high) → W2(low) → W3(high) → W4(low) → W5(high)
    ABC DOWN:   W5(high) → A(low) → B(high) → C(low)

    Returns pattern dict or None if validation fails.
    """
    # ── Wave lengths ──
    wave1 = W1 - W0   # UP
    wave2_drop = W1 - W2  # DOWN (retrace)
    wave3 = W3 - W2   # UP (extension)
    wave4_drop = W3 - W4  # DOWN (retrace)
    wave5 = W5 - W4   # UP

    # Basic direction checks
    if wave1 <= 0 or wave2_drop <= 0 or wave3 <= 0 or wave4_drop <= 0 or wave5 <= 0:
        return None

    # ── Impulse validation (§3.1) ──

    # Wave 2 retraces Wave 1: 0.382-0.886
    w2_retrace = wave2_drop / wave1
    if not (config.w2_retrace_min <= w2_retrace <= config.w2_retrace_max):
        return None

    # Wave 2 must not go below Wave 0 origin
    if W2 <= W0:
        return None

    # Wave 3 extends Wave 1: 1.272-4.236
    w3_ext = wave3 / wave1
    if not (config.w3_extension_min <= w3_ext <= config.w3_extension_max):
        return None

    # Wave 3 must exceed Wave 1 peak
    if W3 <= W1:
        return None

    # Wave 4 retraces Wave 3: 0.146-0.500
    w4_retrace = wave4_drop / wave3
    if not (config.w4_retrace_min <= w4_retrace <= config.w4_retrace_max):
        return None

    # CARDINAL RULE: Wave 4 must NOT intrude into Wave 1 territory (§3.1)
    if W4 <= W1:
        return None

    # ── ABC correction validation (§3.1) ──
    wave_a_len = W5 - A   # A decline
    wave_b_len = B - A     # B recovery
    wave_c_len = B - C     # C decline

    if wave_a_len <= 0 or wave_b_len <= 0 or wave_c_len <= 0:
        return None

    # B must not exceed W5 (Wave A origin)
    if B > W5:
        return None

    # B retracement of A: 0.382-0.786
    b_retrace = wave_b_len / wave_a_len
    if not (config.fib_b_retrace_min <= b_retrace <= config.fib_b_retrace_max):
        return None

    # C vs A length: 0.786-2.000
    c_vs_a = wave_c_len / wave_a_len
    if not (config.fib_c_vs_a_min <= c_vs_a <= config.fib_c_vs_a_max):
        return None

    # C near 0.618 Fib of ENTIRE impulse (W0 to W5)
    total_impulse = W5 - W0
    fib_618 = W5 - 0.618 * total_impulse
    c_fib_dist = abs(C - fib_618) / total_impulse if total_impulse > 0 else 1.0
    if c_fib_dist > config.fib_c_impulse_tolerance:
        return None

    # ── Targets: Fibonacci extensions of Wave 1 from C (§6) ──
    # TP1: 1.618 × Wave 1 from C (conservative — Wave 3 of new impulse)
    # TP2: 2.618 × Wave 1 from C (extended — full new impulse)
    target_1 = C + 1.618 * wave1
    target_2 = C + 2.618 * wave1

    # Stop: below C by 10% of Wave A length
    stop = C - wave_a_len * 0.10

    return {
        'signal': 1,  # Long
        'type': 'bullish_impulse_abc_long',
        'w0': W0, 'w1': W1, 'w2': W2, 'w3': W3, 'w4': W4, 'w5': W5,
        'a_price': A, 'b_price': B, 'c_price': C,
        'c_idx': c_idx,
        'w2_retrace': w2_retrace, 'w3_extension': w3_ext,
        'w4_retrace': w4_retrace,
        'b_retrace': b_retrace, 'c_vs_a': c_vs_a,
        'c_fib_dist': c_fib_dist,
        'target_1': target_1, 'target_2': target_2,
        'stop': stop,
        'wave1_length': wave1,
        'impulse_size': total_impulse,
    }


def _validate_bearish_impulse_abc(
    W0: float, W1: float, W2: float, W3: float, W4: float,
    W5: float, A: float, B: float, C: float,
    c_idx: int, config: ElliottConfig,
) -> dict | None:
    """Validate bearish impulse (DOWN) + bullish ABC (UP) → Short entry.

    Impulse DOWN: W0(high) → W1(low) → W2(high) → W3(low) → W4(high) → W5(low)
    ABC UP:       W5(low) → A(high) → B(low) → C(high)

    Returns pattern dict or None if validation fails.
    """
    # ── Wave lengths (all positive for valid bearish impulse) ──
    wave1 = W0 - W1   # DOWN
    wave2_rise = W2 - W1  # UP (retrace)
    wave3 = W2 - W3   # DOWN (extension)
    wave4_rise = W4 - W3  # UP (retrace)
    wave5 = W4 - W5   # DOWN

    if wave1 <= 0 or wave2_rise <= 0 or wave3 <= 0 or wave4_rise <= 0 or wave5 <= 0:
        return None

    # ── Impulse validation (§3.1) ──

    # Wave 2 retraces Wave 1
    w2_retrace = wave2_rise / wave1
    if not (config.w2_retrace_min <= w2_retrace <= config.w2_retrace_max):
        return None

    # Wave 2 must not go above Wave 0 origin
    if W2 >= W0:
        return None

    # Wave 3 extends Wave 1
    w3_ext = wave3 / wave1
    if not (config.w3_extension_min <= w3_ext <= config.w3_extension_max):
        return None

    # Wave 3 must go below Wave 1 trough
    if W3 >= W1:
        return None

    # Wave 4 retraces Wave 3
    w4_retrace = wave4_rise / wave3
    if not (config.w4_retrace_min <= w4_retrace <= config.w4_retrace_max):
        return None

    # CARDINAL RULE: Wave 4 must NOT intrude into Wave 1 territory
    if W4 >= W1:
        return None

    # ── ABC correction validation ──
    wave_a_len = A - W5   # A rise
    wave_b_len = A - B     # B decline
    wave_c_len = C - B     # C rise

    if wave_a_len <= 0 or wave_b_len <= 0 or wave_c_len <= 0:
        return None

    # B must not go below W5
    if B < W5:
        return None

    # B retracement of A
    b_retrace = wave_b_len / wave_a_len
    if not (config.fib_b_retrace_min <= b_retrace <= config.fib_b_retrace_max):
        return None

    # C vs A length
    c_vs_a = wave_c_len / wave_a_len
    if not (config.fib_c_vs_a_min <= c_vs_a <= config.fib_c_vs_a_max):
        return None

    # C near 0.618 Fib of entire impulse
    total_impulse = W0 - W5
    fib_618 = W5 + 0.618 * total_impulse
    c_fib_dist = abs(C - fib_618) / total_impulse if total_impulse > 0 else 1.0
    if c_fib_dist > config.fib_c_impulse_tolerance:
        return None

    # ── Targets: Fibonacci extensions of Wave 1 from C ──
    target_1 = C - 1.618 * wave1
    target_2 = C - 2.618 * wave1

    stop = C + wave_a_len * 0.10

    return {
        'signal': -1,  # Short
        'type': 'bearish_impulse_abc_short',
        'w0': W0, 'w1': W1, 'w2': W2, 'w3': W3, 'w4': W4, 'w5': W5,
        'a_price': A, 'b_price': B, 'c_price': C,
        'c_idx': c_idx,
        'w2_retrace': w2_retrace, 'w3_extension': w3_ext,
        'w4_retrace': w4_retrace,
        'b_retrace': b_retrace, 'c_vs_a': c_vs_a,
        'c_fib_dist': c_fib_dist,
        'target_1': target_1, 'target_2': target_2,
        'stop': stop,
        'wave1_length': wave1,
        'impulse_size': total_impulse,
    }


# Keep backward compatibility alias
detect_abc_patterns = detect_impulse_abc_patterns


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
        if n_piv < 9:
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

    Implements the full championship strategy flow (winning_strategy §5):
      1. BTC 200/500 EMA macro bias filter
      2. ZigZag → 5-wave impulse + ABC correction detection (9 pivots)
      3. Fibonacci validation for all waves
      4. Triangle filter
      5. Divergence (mandatory) + corroboration (MACD or Stoch) per §2.2
      6. Signal scaled by effective leverage (§4.1: 10% margin × 20x = 2x)

    Confirmation Protocol (§2.2): "Divergence signals require mandatory
    corroboration. The agent must verify a reversal in the MACD histogram
    OR a golden/dead cross in the Stochastic oscillator."
    → Divergence REQUIRED + at least 1 of {MACD reversal, Stoch cross}

    Args:
        close_4h: Symbol 4h close prices.
        high_4h: Symbol 4h high prices.
        low_4h: Symbol 4h low prices.
        btc_close_4h: BTC 4h close prices.
        config: Strategy configuration.

    Returns:
        Signal array aligned to close_4h. Values are scaled by
        effective_leverage (default 2.0) for realistic P&L modeling.
    """
    cfg = config or ElliottConfig()
    n = len(close_4h)
    signals = np.zeros(n, dtype=np.float64)

    if n < cfg.ema_macro_slow + 50:
        return signals

    # Pre-compute BTC EMAs for macro bias
    btc_ema200 = compute_ema(btc_close_4h, cfg.ema_macro_fast)
    btc_ema500 = compute_ema(btc_close_4h, cfg.ema_macro_slow)

    # Pre-compute indicators on symbol 4h
    rsi_4h = compute_rsi(close_4h, cfg.rsi_period)
    _, _, macd_hist_4h = compute_macd(
        close_4h, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
    )
    stoch_k_4h, stoch_d_4h = compute_stochastic(
        high_4h, low_4h, close_4h, cfg.stoch_k_period, cfg.stoch_d_period,
    )

    lev = cfg.effective_leverage

    # Walk-forward: at each bar, detect patterns on data up to that bar
    min_start = cfg.ema_macro_slow + 50

    for bar in range(min_start, n):
        # 1. Macro bias at this bar
        btc_price = float(btc_close_4h[bar])
        above_200 = btc_price > float(btc_ema200[bar])
        above_500 = btc_price > float(btc_ema500[bar])

        if above_200 and above_500:
            bias = 'long'
        elif not above_200 and not above_500:
            bias = 'short'
        else:
            bias = 'neutral'

        # 2. ZigZag on data up to this bar
        pidx, pprice, ptype, n_piv = zigzag_pivots(
            high_4h[:bar + 1], low_4h[:bar + 1], cfg.zigzag_threshold,
        )
        if n_piv < 9:
            continue

        # 3. Triangle filter (§3.2)
        if detect_triangle(pprice, ptype, n_piv):
            continue

        # 4. 5-wave impulse + ABC detection with Fibonacci validation
        patterns = detect_impulse_abc_patterns(pidx, pprice, ptype, n_piv, cfg)
        if not patterns:
            continue

        # Only consider patterns completing at or near the current bar
        recent = [p for p in patterns if p['c_idx'] >= bar - 2]
        if not recent:
            continue

        pattern = recent[-1]

        # 5. Macro bias filter
        if bias == 'long' and pattern['signal'] != 1:
            continue
        if bias == 'short' and pattern['signal'] != -1:
            continue

        # 6. Confirmation Protocol (§2.2):
        #    Divergence (REQUIRED) + at least 1 corroboration (MACD or Stoch)
        direction = pattern['signal']
        lookback = min(30, bar)

        # Check divergence (mandatory)
        if direction == 1:
            has_div = detect_bullish_divergence(
                close_4h[:bar + 1], rsi_4h[:bar + 1], lookback,
            )
        else:
            has_div = detect_bearish_divergence(
                close_4h[:bar + 1], rsi_4h[:bar + 1], lookback,
            )

        if not has_div:
            continue  # Divergence is mandatory

        # Check corroboration (at least 1 of MACD or Stoch)
        has_macd = check_macd_reversal(macd_hist_4h[:bar + 1], direction)
        has_stoch = check_stoch_cross(
            stoch_k_4h[:bar + 1], stoch_d_4h[:bar + 1], direction,
        )

        if not (has_macd or has_stoch):
            continue  # Corroboration is mandatory

        # 7. Signal with effective leverage
        signals[bar] = float(direction) * lev

    return signals
