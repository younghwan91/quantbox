#!/usr/bin/env python3
"""
TOP-DOWN MULTI-INDICATOR A/B TEST — Competition Winner's Principles

대회 우승자의 4대 원칙을 Rubber Band Scalper에 적용:
  1. Top-Down: BTC(시장 전체)의 방향을 먼저 확인 → 개별 종목 진입
  2. Multi-Indicator: RSI + Volume + S/R + BB 등 3개 이상 지표 동의 시만 진입
  3. Risk-Reward ≥ 2:1: 잠재 수익이 잠재 손실의 2배 이상인 자리에서만 매매
  4. Backtesting: 365일 실제 데이터로 검증

핵심 적응:
  - 우리 전략은 "평균회귀(Counter-Trend)"이므로, Top-Down 필터는
    "BTC가 강하게 추세 중일 때 역추세 매매를 자제"하는 방식으로 적용.
  - Multi-Indicator는 "반전 확률이 높은 극단 상황"을 확인하는 용도.

Phase 1: BTC Top-Down 필터 (시장 전체 방향 확인)
Phase 2: RSI 극단값 확인 (과매수/과매도 확인)
Phase 3: 거래량 급증 확인 (캐피탈레이션 = 높은 반전 확률)
Phase 4: 지지/저항 + 볼린저 밴드 필터
Phase 5: RR 필터 (손익비 ≥ 2:1)
Phase 6: Multi-Indicator 복합 (3개 이상 지표 동의)
Phase 7: Top-10 + 최적 조합 + Dynamic Leverage
Phase 8: Rolling 90일 안정성 검증

Data: 30 symbols × 35,040 bars (15min), 365 days real Binance USDT-M
"""

import sys, os, gzip, pickle, time, warnings
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import numba
from quantbox.strategy.rubberband import compute_signals
from quantbox.backtest.vectorized_backtest import _backtest_core

# ── Constants ──
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32
BARS_PER_DAY = 96

# ── v4.0 validated parameters ──
LOOKBACK = 8
THRESHOLD = 0.045
TARGET = 0.3
STOP = 1.0
MAX_HOLD = 10
STOP_DECAY = 0.0

# ── Top-10 curated (from leverage A/B test) ──
TOP10 = [
    "FETUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT", "DOTUSDT",
    "OPUSDT", "SHIBUSDT", "DOGEUSDT", "LTCUSDT", "ATOMUSDT",
]


def log(msg=""):
    print(msg)
    out.write(msg + "\n")
    out.flush()


def load_all_data():
    """Load OHLCV data for all symbols."""
    cache_dir = Path("data/cache_extended")
    all_data = {}
    for f in sorted(cache_dir.glob("*_15m_365d.pkl.gz")):
        symbol = f.name.split("_15m_")[0]
        with gzip.open(f, "rb") as fh:
            df = pickle.load(fh)
        if "Close" in df.columns and len(df) >= 200:
            all_data[symbol] = {
                "open": df["Open"].values.astype(np.float64),
                "high": df["High"].values.astype(np.float64),
                "low": df["Low"].values.astype(np.float64),
                "close": df["Close"].values.astype(np.float64),
                "volume": df["Volume"].values.astype(np.float64),
            }
    return all_data


def fast_metrics(eq, start=0):
    s = eq[start:]
    if len(s) < 2:
        return 0, 0, 0, 0
    ret = (s[-1] / s[0] - 1) * 100
    pk = np.maximum.accumulate(s)
    dd = np.max((pk - s) / pk) * 100
    cal = ret / dd if dd > 1e-9 else (999 if ret > 0 else -999)
    rs = np.diff(s) / s[:-1]
    sh = np.mean(rs) / np.std(rs) * np.sqrt(96 * 365.25) if np.std(rs) > 1e-12 else 0
    return ret, dd, cal, sh


def portfolio_eq(eqs, n_bars):
    """Equal-weight portfolio equity (arithmetic mean).

    Each symbol gets 1/n of capital. If one symbol liquidates (eq→0),
    you lose 1/n of portfolio — NOT 100% (which geometric mean implies).
    """
    n = len(eqs)
    port = np.zeros(n_bars)
    for eq in eqs:
        e = eq[:n_bars]
        port += e / n
    return port


# ═══════════════════════════════════════════════════════════════════════
# NUMBA INDICATOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def compute_rsi(close, period=14):
    """Compute RSI (Relative Strength Index)."""
    n = len(close)
    rsi = np.full(n, 50.0)  # Default to neutral
    if n < period + 1:
        return rsi

    # Initial average gain/loss
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        diff = close[i] - close[i - 1]
        if diff > 0:
            avg_gain += diff
        else:
            avg_loss -= diff
    avg_gain /= period
    avg_loss /= period

    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    else:
        rsi[period] = 100.0

    # Exponential smoothing
    for i in range(period + 1, n):
        diff = close[i] - close[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi[i] = 100.0

    return rsi


@numba.jit(nopython=True, cache=True)
def compute_sma(close, period):
    """Simple Moving Average."""
    n = len(close)
    sma = np.zeros(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += close[i]
        if i >= period:
            cumsum -= close[i - period]
            sma[i] = cumsum / period
        elif i == period - 1:
            sma[i] = cumsum / period
        else:
            sma[i] = close[i]
    return sma


@numba.jit(nopython=True, cache=True)
def compute_bollinger(close, period=20, num_std=2.0):
    """Compute Bollinger Bands (middle, upper, lower)."""
    n = len(close)
    mid = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)

    for i in range(n):
        if i < period - 1:
            mid[i] = close[i]
            upper[i] = close[i]
            lower[i] = close[i]
        else:
            s = 0.0
            for j in range(i - period + 1, i + 1):
                s += close[j]
            mean = s / period
            var = 0.0
            for j in range(i - period + 1, i + 1):
                var += (close[j] - mean) ** 2
            std = (var / period) ** 0.5
            mid[i] = mean
            upper[i] = mean + num_std * std
            lower[i] = mean - num_std * std

    return mid, upper, lower


@numba.jit(nopython=True, cache=True)
def compute_volume_ratio(volume, period=20):
    """Volume relative to its moving average."""
    n = len(volume)
    ratio = np.ones(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += volume[i]
        if i >= period:
            cumsum -= volume[i - period]
            avg = cumsum / period
            if avg > 0:
                ratio[i] = volume[i] / avg
        elif i == period - 1:
            avg = cumsum / period
            if avg > 0:
                ratio[i] = volume[i] / avg
    return ratio


@numba.jit(nopython=True, cache=True)
def compute_support_resistance(high, low, close, lookback=96):
    """Rolling support/resistance levels from recent highs/lows."""
    n = len(close)
    support = np.zeros(n)
    resistance = np.zeros(n)
    for i in range(n):
        start = max(0, i - lookback)
        local_low = low[start]
        local_high = high[start]
        for j in range(start + 1, i + 1):
            if low[j] < local_low:
                local_low = low[j]
            if high[j] > local_high:
                local_high = high[j]
        support[i] = local_low
        resistance[i] = local_high
    return support, resistance


# ═══════════════════════════════════════════════════════════════════════
# PHASE SIGNAL GENERATORS — All Numba-compiled
# ═══════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def signals_with_btc_topdown(
    close, btc_close,
    lookback, threshold, target, stop, max_hold, stop_decay,
    btc_sma_period, btc_trend_max,
):
    """Phase 1: BTC Top-Down filter.

    For mean-reversion: block entries when BTC has a STRONG unidirectional
    trend (either up or down beyond btc_trend_max). When BTC is trending
    hard, counter-trend bets on alts get steamrolled.

    btc_sma_period: SMA lookback for BTC trend detection (e.g. 96=24h)
    btc_trend_max: Max |BTC trend| to allow entry (e.g. 0.02=2%)
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    btc_sma = compute_sma(btc_close, btc_sma_period)
    start = max(lookback, btc_sma_period)

    for i in range(start, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            # Top-Down check: is BTC trending too hard?
            btc_trend = (btc_close[i] - btc_sma[i]) / btc_sma[i] if btc_sma[i] > 0 else 0.0
            if abs(btc_trend) > btc_trend_max:
                sig[i] = 0
                continue

            if ret < -threshold:
                position = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold:
                position = -1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop * time_factor
            tgt = move_size * target
            if gain >= tgt:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > tgt * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_with_btc_directional(
    close, btc_close,
    lookback, threshold, target, stop, max_hold, stop_decay,
    btc_mom_period,
):
    """Phase 1b: BTC directional alignment.

    Only allow long when BTC is falling (sympathy oversold),
    only allow short when BTC is rising (sympathy overbought).
    Rationale: When BTC dumps, alts dump harder → bigger snap-back.
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start = max(lookback, btc_mom_period)

    for i in range(start, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            btc_ret = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

            if ret < -threshold:
                # Want to go long (fade drop)
                # BTC also dropping = sympathy dump = stronger conviction
                if btc_ret < 0:
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
            elif ret > threshold:
                # Want to go short (fade spike)
                # BTC also rising = sympathy pump = stronger conviction
                if btc_ret > 0:
                    position = -1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop * time_factor
            tgt = move_size * target
            if gain >= tgt:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > tgt * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_with_rsi(
    close,
    lookback, threshold, target, stop, max_hold, stop_decay,
    rsi_period, rsi_long_max, rsi_short_min,
):
    """Phase 2: RSI extreme confirmation.

    For mean-reversion:
      - Long only when RSI < rsi_long_max (oversold → higher bounce chance)
      - Short only when RSI > rsi_short_min (overbought → higher fade chance)
    """
    n = len(close)
    sig = np.zeros(n)
    rsi = compute_rsi(close, rsi_period)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if ret < -threshold and rsi[i] < rsi_long_max:
                position = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold and rsi[i] > rsi_short_min:
                position = -1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop * time_factor
            tgt = move_size * target
            if gain >= tgt:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > tgt * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_with_volume(
    close, volume,
    lookback, threshold, target, stop, max_hold, stop_decay,
    vol_period, vol_min_ratio,
):
    """Phase 3: Volume spike confirmation.

    Only enter when volume > vol_min_ratio × avg(volume).
    High volume during extreme moves = capitulation/euphoria = higher reversal probability.
    """
    n = len(close)
    sig = np.zeros(n)
    vol_ratio = compute_volume_ratio(volume, vol_period)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if ret < -threshold and vol_ratio[i] >= vol_min_ratio:
                position = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold and vol_ratio[i] >= vol_min_ratio:
                position = -1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop * time_factor
            tgt = move_size * target
            if gain >= tgt:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > tgt * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_with_bb_sr(
    close, high, low,
    lookback, threshold, target, stop, max_hold, stop_decay,
    bb_period, bb_std, sr_lookback,
):
    """Phase 4: Bollinger Band + Support/Resistance filter.

    For mean-reversion:
      - Long only when price near/below lower BB AND near support
      - Short only when price near/above upper BB AND near resistance
    "Near" = within 0.5% of the level.
    """
    n = len(close)
    sig = np.zeros(n)
    _, bb_upper, bb_lower = compute_bollinger(close, bb_period, bb_std)
    support, resistance = compute_support_resistance(high, low, close, sr_lookback)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    for i in range(max(lookback, bb_period, sr_lookback), n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            near_lower_bb = close[i] <= bb_lower[i] * 1.005
            near_upper_bb = close[i] >= bb_upper[i] * 0.995
            # Support: price within 1% of rolling low
            near_support = close[i] <= support[i] * 1.01
            near_resistance = close[i] >= resistance[i] * 0.99

            if ret < -threshold and (near_lower_bb or near_support):
                position = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold and (near_upper_bb or near_resistance):
                position = -1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop * time_factor
            tgt = move_size * target
            if gain >= tgt:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > tgt * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_with_rr_filter(
    close,
    lookback, threshold, target, stop, max_hold, stop_decay,
    min_rr_ratio,
):
    """Phase 5: Risk-Reward ratio filter.

    Only enter when expected profit / expected loss >= min_rr_ratio.
    For rubber band: profit = move_size * target_retrace, loss = move_size * stop_mult.
    R:R = target_retrace / stop_mult.

    But since target_retrace and stop_mult are fixed, this actually
    filters on move_size: bigger moves have better R:R because the
    target (absolute) is larger while cost is fixed.

    We compute different R:R measures:
      - Adjusted R:R = (target * ms - cost) / (stop * ms + cost)
      - Only enter when adjusted R:R >= min_rr_ratio
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    round_trip_cost = 2 * 0.0004  # 0.08% round-trip

    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                ms = abs(ret)
                expected_profit = ms * target - round_trip_cost  # Net gain
                expected_loss = ms * stop + round_trip_cost  # Total loss
                rr = expected_profit / expected_loss if expected_loss > 0 else 0

                if rr >= min_rr_ratio:
                    if ret < -threshold:
                        position = 1
                        entry_price = close[i]
                        move_size = ms
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold:
                        position = -1
                        entry_price = close[i]
                        move_size = ms
                        hold_count = 0
                        best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop * time_factor
            tgt = move_size * target
            if gain >= tgt:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > tgt * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_multi_indicator(
    close, high, low, volume, btc_close,
    lookback, threshold, target, stop, max_hold, stop_decay,
    rsi_period, rsi_long_max, rsi_short_min,
    vol_period, vol_min_ratio,
    bb_period, bb_std,
    btc_sma_period, btc_trend_max,
    min_confirms,
):
    """Phase 6: Multi-indicator confirmation (≥ min_confirms to enter).

    Indicators checked:
      1. RSI extreme (oversold/overbought)
      2. Volume spike (> vol_min_ratio × avg)
      3. Bollinger Band penetration
      4. BTC top-down (not trending too hard)
    Enter only when >= min_confirms indicators agree.
    """
    n = len(close)
    sig = np.zeros(n)
    rsi = compute_rsi(close, rsi_period)
    vol_ratio = compute_volume_ratio(volume, vol_period)
    _, bb_upper, bb_lower = compute_bollinger(close, bb_period, bb_std)
    btc_sma = compute_sma(btc_close, btc_sma_period)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start = max(lookback, rsi_period, vol_period, bb_period, btc_sma_period)

    for i in range(start, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) < threshold:
                sig[i] = 0
                continue

            confirms = 0

            if ret < -threshold:
                # Long candidate — count confirmations
                # 1. RSI oversold
                if rsi[i] < rsi_long_max:
                    confirms += 1
                # 2. Volume spike
                if vol_ratio[i] >= vol_min_ratio:
                    confirms += 1
                # 3. Near lower BB
                if close[i] <= bb_lower[i] * 1.005:
                    confirms += 1
                # 4. BTC not trending hard up (safe for counter-trend)
                btc_trend = (btc_close[i] - btc_sma[i]) / btc_sma[i] if btc_sma[i] > 0 else 0
                if abs(btc_trend) <= btc_trend_max:
                    confirms += 1

                if confirms >= min_confirms:
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0

            elif ret > threshold:
                # Short candidate
                if rsi[i] > rsi_short_min:
                    confirms += 1
                if vol_ratio[i] >= vol_min_ratio:
                    confirms += 1
                if close[i] >= bb_upper[i] * 0.995:
                    confirms += 1
                btc_trend = (btc_close[i] - btc_sma[i]) / btc_sma[i] if btc_sma[i] > 0 else 0
                if abs(btc_trend) <= btc_trend_max:
                    confirms += 1

                if confirms >= min_confirms:
                    position = -1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop * time_factor
            tgt = move_size * target
            if gain >= tgt:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > tgt * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def apply_confidence_leverage_numba(
    close, signals, lookback, threshold,
    base_lev, lev_scale, max_lev,
):
    """Scale ±1 signals by move-confidence leverage."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = signals[i]
        if s == 0.0 or i < lookback:
            out[i] = s
            continue
        move = abs(close[i] / close[i - lookback] - 1.0)
        excess = (move - threshold) / threshold
        lev = base_lev + lev_scale * max(excess, 0.0)
        if lev > max_lev:
            lev = max_lev
        if lev < base_lev:
            lev = base_lev
        out[i] = lev if s > 0 else -lev
    return out


# ═══════════════════════════════════════════════════════════════════════
# MAIN A/B TEST
# ═══════════════════════════════════════════════════════════════════════

def main():
    global out
    out = open("/tmp/topdown_ab.txt", "w")
    t0 = time.time()

    log("=" * 72)
    log("  TOP-DOWN MULTI-INDICATOR A/B TEST — 대회 우승자의 4대 원칙")
    log("  365일 실데이터 × 30 심볼 × 35,040 바")
    log("=" * 72)

    data = load_all_data()
    symbols = sorted(data.keys())
    btc_close = data["BTCUSDT"]["close"]
    n_bars = min(len(data[s]["close"]) for s in symbols)
    log(f"\n로드 완료: {len(symbols)} 심볼 × {n_bars} 바")

    # ── Baseline signals ──
    log("\n기준선(Baseline) 계산 중...")
    baseline_eqs = {}
    for sym in symbols:
        sigs = compute_signals(data[sym]["close"][:n_bars], LOOKBACK, THRESHOLD,
                               TARGET, STOP, MAX_HOLD, STOP_DECAY)
        eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                 COST_PER_SIDE, FUNDING_PER_BAR)
        baseline_eqs[sym] = eq

    base_port = portfolio_eq([baseline_eqs[s] for s in symbols], n_bars)
    br, bd, bc, bs = fast_metrics(base_port)
    log(f"  Baseline (All-30 EW 1x): Ret={br:+.1f}%, DD={bd:.1f}%, Calmar={bc:.2f}, Sharpe={bs:.2f}")

    base_top10_port = portfolio_eq([baseline_eqs[s] for s in TOP10 if s in baseline_eqs], n_bars)
    br10, bd10, bc10, bs10 = fast_metrics(base_top10_port)
    log(f"  Baseline (Top-10 EW 1x): Ret={br10:+.1f}%, DD={bd10:.1f}%, Calmar={bc10:.2f}, Sharpe={bs10:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: BTC TOP-DOWN FILTER
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 1: BTC Top-Down 시장 방향 필터")
    log("═" * 72)

    # 1a: BTC SMA trend gate
    log("\n  1a) BTC SMA 트렌드 게이트 (|BTC trend| > max → 진입 차단)")
    for sma_per in [48, 96, 192]:
        for btc_max in [0.01, 0.02, 0.03, 0.05]:
            eqs = []
            for sym in symbols:
                sigs = signals_with_btc_topdown(
                    data[sym]["close"][:n_bars], btc_close[:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    sma_per, btc_max)
                eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                         COST_PER_SIDE, FUNDING_PER_BAR)
                eqs.append(eq)
            port = portfolio_eq(eqs, n_bars)
            r, d, c, sh = fast_metrics(port)
            flag = " ★" if c > bc else ""
            log(f"    SMA={sma_per:>3d}, MaxTrend={btc_max:.0%}: "
                f"Ret={r:+.1f}%, DD={d:.1f}%, Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # 1b: BTC directional alignment
    log("\n  1b) BTC 방향 정렬 (Long→BTC하락, Short→BTC상승)")
    for mom_per in [4, 8, 16, 32]:
        eqs = []
        for sym in symbols:
            sigs = signals_with_btc_directional(
                data[sym]["close"][:n_bars], btc_close[:n_bars],
                LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                mom_per)
            eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                     COST_PER_SIDE, FUNDING_PER_BAR)
            eqs.append(eq)
        port = portfolio_eq(eqs, n_bars)
        r, d, c, sh = fast_metrics(port)
        flag = " ★" if c > bc else ""
        log(f"    BTC_mom={mom_per:>2d}: Ret={r:+.1f}%, DD={d:.1f}%, Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: RSI EXTREME CONFIRMATION
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 2: RSI 극단값 확인 (과매수/과매도)")
    log("═" * 72)

    for rsi_per in [8, 14]:
        for rsi_lm, rsi_sm in [(40, 60), (35, 65), (30, 70), (25, 75)]:
            eqs = []
            for sym in symbols:
                sigs = signals_with_rsi(
                    data[sym]["close"][:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    rsi_per, float(rsi_lm), float(rsi_sm))
                eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                         COST_PER_SIDE, FUNDING_PER_BAR)
                eqs.append(eq)
            port = portfolio_eq(eqs, n_bars)
            r, d, c, sh = fast_metrics(port)
            flag = " ★" if c > bc else ""
            log(f"    RSI({rsi_per:>2d}) Long<{rsi_lm} Short>{rsi_sm}: "
                f"Ret={r:+.1f}%, DD={d:.1f}%, Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: VOLUME SPIKE CONFIRMATION
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 3: 거래량 급증 확인 (캐피탈레이션)")
    log("═" * 72)

    for vol_per in [20, 48]:
        for vol_min in [1.0, 1.2, 1.5, 2.0, 3.0]:
            eqs = []
            for sym in symbols:
                sigs = signals_with_volume(
                    data[sym]["close"][:n_bars], data[sym]["volume"][:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    vol_per, vol_min)
                eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                         COST_PER_SIDE, FUNDING_PER_BAR)
                eqs.append(eq)
            port = portfolio_eq(eqs, n_bars)
            r, d, c, sh = fast_metrics(port)
            flag = " ★" if c > bc else ""
            log(f"    VolPeriod={vol_per:>2d}, VolMin={vol_min:.1f}x: "
                f"Ret={r:+.1f}%, DD={d:.1f}%, Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: BOLLINGER BAND + SUPPORT/RESISTANCE
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 4: 볼린저 밴드 + 지지/저항")
    log("═" * 72)

    for bb_per in [20, 40]:
        for bb_s in [1.5, 2.0, 2.5]:
            for sr_lb in [48, 96]:
                eqs = []
                for sym in symbols:
                    sigs = signals_with_bb_sr(
                        data[sym]["close"][:n_bars],
                        data[sym]["high"][:n_bars],
                        data[sym]["low"][:n_bars],
                        LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                        bb_per, bb_s, sr_lb)
                    eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                             COST_PER_SIDE, FUNDING_PER_BAR)
                    eqs.append(eq)
                port = portfolio_eq(eqs, n_bars)
                r, d, c, sh = fast_metrics(port)
                flag = " ★" if c > bc else ""
                log(f"    BB({bb_per},{bb_s:.1f}σ) SR({sr_lb}): "
                    f"Ret={r:+.1f}%, DD={d:.1f}%, Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: RISK-REWARD FILTER
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 5: 손익비(R:R) 필터 — 통과해야만 진입")
    log("═" * 72)
    log(f"  참고: 현재 R:R = target/stop = {TARGET}/{STOP} = {TARGET/STOP:.2f}:1")
    log(f"  비용 포함 R:R = target*ms / (stop*ms + 0.08%) → 큰 움직임이면 R:R↑")

    for min_rr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]:
        eqs = []
        for sym in symbols:
            sigs = signals_with_rr_filter(
                data[sym]["close"][:n_bars],
                LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                min_rr)
            eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                     COST_PER_SIDE, FUNDING_PER_BAR)
            eqs.append(eq)
        port = portfolio_eq(eqs, n_bars)
        r, d, c, sh = fast_metrics(port)
        flag = " ★" if c > bc else ""
        log(f"    MinR:R={min_rr:.2f}:1: Ret={r:+.1f}%, DD={d:.1f}%, "
            f"Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: MULTI-INDICATOR COMPOSITE
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 6: 복합 지표 검증 (N개 이상 지표 동의 시만 진입)")
    log("═" * 72)

    for min_conf in [1, 2, 3, 4]:
        for rsi_lm, rsi_sm in [(35, 65), (40, 60)]:
            eqs = []
            for sym in symbols:
                sigs = signals_multi_indicator(
                    data[sym]["close"][:n_bars],
                    data[sym]["high"][:n_bars],
                    data[sym]["low"][:n_bars],
                    data[sym]["volume"][:n_bars],
                    btc_close[:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    14, float(rsi_lm), float(rsi_sm),   # RSI
                    20, 1.5,                              # Volume
                    20, 2.0,                              # BB
                    96, 0.03,                              # BTC top-down
                    min_conf)
                eq, *_ = _backtest_core(data[sym]["close"][:n_bars], sigs,
                                         COST_PER_SIDE, FUNDING_PER_BAR)
                eqs.append(eq)
            port = portfolio_eq(eqs, n_bars)
            r, d, c, sh = fast_metrics(port)
            flag = " ★" if c > bc else ""
            log(f"    ≥{min_conf}지표, RSI({rsi_lm}/{rsi_sm}): "
                f"Ret={r:+.1f}%, DD={d:.1f}%, Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: TOP-10 + BEST COMBO + DYNAMIC LEVERAGE
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 7: Top-10 + 최적 조합 + Dynamic Leverage")
    log("═" * 72)

    top10_syms = [s for s in TOP10 if s in data]

    # Test best individual filters on Top-10
    configs = [
        ("Baseline (1x)", None, False),
        ("BTC TopDown SMA96 3%", "btc_td", False),
        ("RSI(14) 35/65", "rsi", False),
        ("Volume 1.5x", "vol", False),
        ("BB(20,2σ)+SR(96)", "bb_sr", False),
        ("Multi ≥2 지표", "multi2", False),
        ("Multi ≥3 지표", "multi3", False),
        # Now with dynamic leverage
        ("Baseline + DynLev 3/10/7", None, True),
        ("BTC TD + DynLev", "btc_td", True),
        ("RSI + DynLev", "rsi", True),
        ("Vol + DynLev", "vol", True),
        ("Multi≥2 + DynLev", "multi2", True),
        ("Multi≥3 + DynLev", "multi3", True),
        # Aggressive leverage
        ("Baseline + DynLev 5/20/15", None, "aggressive"),
        ("Multi≥2 + DynLev 5/20/15", "multi2", "aggressive"),
    ]

    for label, filt, lev in configs:
        eqs = []
        for sym in top10_syms:
            c_ = data[sym]["close"][:n_bars]

            if filt == "btc_td":
                sigs = signals_with_btc_topdown(
                    c_, btc_close[:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    96, 0.03)
            elif filt == "rsi":
                sigs = signals_with_rsi(
                    c_, LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    14, 35.0, 65.0)
            elif filt == "vol":
                sigs = signals_with_volume(
                    c_, data[sym]["volume"][:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    20, 1.5)
            elif filt == "bb_sr":
                sigs = signals_with_bb_sr(
                    c_, data[sym]["high"][:n_bars], data[sym]["low"][:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    20, 2.0, 96)
            elif filt == "multi2":
                sigs = signals_multi_indicator(
                    c_, data[sym]["high"][:n_bars], data[sym]["low"][:n_bars],
                    data[sym]["volume"][:n_bars], btc_close[:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    14, 35.0, 65.0, 20, 1.5, 20, 2.0, 96, 0.03, 2)
            elif filt == "multi3":
                sigs = signals_multi_indicator(
                    c_, data[sym]["high"][:n_bars], data[sym]["low"][:n_bars],
                    data[sym]["volume"][:n_bars], btc_close[:n_bars],
                    LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                    14, 35.0, 65.0, 20, 1.5, 20, 2.0, 96, 0.03, 3)
            else:
                sigs = compute_signals(c_, LOOKBACK, THRESHOLD, TARGET,
                                       STOP, MAX_HOLD, STOP_DECAY)

            # Apply leverage
            if lev == "aggressive":
                sigs = apply_confidence_leverage_numba(
                    c_, sigs, LOOKBACK, THRESHOLD, 5.0, 20.0, 15.0)
            elif lev:
                sigs = apply_confidence_leverage_numba(
                    c_, sigs, LOOKBACK, THRESHOLD, 3.0, 10.0, 7.0)

            eq, *_ = _backtest_core(c_, sigs, COST_PER_SIDE, FUNDING_PER_BAR)
            eqs.append(eq)

        port = portfolio_eq(eqs, n_bars)
        r, d, c, sh = fast_metrics(port)
        flag = " ★★★" if r > 1000 else (" ★★" if c > bc10 * 1.5 else (" ★" if c > bc10 else ""))
        log(f"    {label:>30s}: Ret={r:+.1f}%, DD={d:.1f}%, Calmar={c:.2f}, Sharpe={sh:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 8: ROLLING 90-DAY STABILITY
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "═" * 72)
    log("  PHASE 8: Rolling 90일 안정성 검증 (최적 조합)")
    log("═" * 72)

    window = 90 * BARS_PER_DAY  # 90 days in bars
    n_windows = (n_bars - window) // (30 * BARS_PER_DAY) + 1  # 30-day step

    # Test best configs from Phase 7
    best_configs = [
        ("Baseline 1x", None, False),
        ("Multi≥2 + DynLev 3/10/7", "multi2", True),
        ("Baseline + DynLev 5/20/15", None, "aggressive"),
    ]

    for label, filt, lev in best_configs:
        w_rets = []
        for w in range(min(n_windows, 10)):
            start = w * 30 * BARS_PER_DAY
            end = start + window
            if end > n_bars:
                break

            eqs = []
            for sym in top10_syms:
                c_ = data[sym]["close"][start:end]
                btc_ = btc_close[start:end]

                if filt == "multi2":
                    sigs = signals_multi_indicator(
                        c_, data[sym]["high"][start:end],
                        data[sym]["low"][start:end],
                        data[sym]["volume"][start:end], btc_,
                        LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
                        14, 35.0, 65.0, 20, 1.5, 20, 2.0, 96, 0.03, 2)
                else:
                    sigs = compute_signals(c_, LOOKBACK, THRESHOLD, TARGET,
                                           STOP, MAX_HOLD, STOP_DECAY)

                if lev == "aggressive":
                    sigs = apply_confidence_leverage_numba(
                        c_, sigs, LOOKBACK, THRESHOLD, 5.0, 20.0, 15.0)
                elif lev:
                    sigs = apply_confidence_leverage_numba(
                        c_, sigs, LOOKBACK, THRESHOLD, 3.0, 10.0, 7.0)

                eq, *_ = _backtest_core(c_, sigs, COST_PER_SIDE, FUNDING_PER_BAR)
                eqs.append(eq)

            port = portfolio_eq(eqs, len(eqs[0]))
            w_ret = (port[-1] / port[0] - 1) * 100
            w_rets.append(w_ret)

        pos_count = sum(1 for r in w_rets if r > 0)
        log(f"\n  {label}:")
        log(f"    수익 윈도우: {pos_count}/{len(w_rets)} ({pos_count/max(len(w_rets),1)*100:.0f}%)")
        log(f"    최소: {min(w_rets):+.1f}%, 최대: {max(w_rets):+.1f}%, "
            f"중앙: {sorted(w_rets)[len(w_rets)//2]:+.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    log("\n" + "═" * 72)
    log(f"  완료! 소요시간: {elapsed:.1f}초")
    log("═" * 72)

    out.close()
    print(f"\n결과 저장: /tmp/topdown_ab.txt")


if __name__ == "__main__":
    main()
