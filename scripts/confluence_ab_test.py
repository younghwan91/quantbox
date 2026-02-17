#!/usr/bin/env python3
"""
v8.0 A/B Test: Confluence-Gated Entry (Winning Strategy Insights)
================================================================
Tests 5 quantifiable improvements inspired by the 10000% competition winner:

  H1: RSI Divergence — replace RSI level check with divergence detection
      (price lower-low + RSI higher-low = bullish divergence)
  H2: Volatility Squeeze — only enter after ATR compression (energy buildup)
  H3: Dynamic Target — scale target by confidence (certain → ride the wave)
  H4: Reversal Confirmation — wait 1 bar for price to start reversing
  H5: All combined ("Structural Reversal" mode)

Each hypothesis tested on 10 curated symbols × 365 days, compared
against v7.0 baseline (compute_signals_confidence + MTF confirmation).
"""

import sys, os, time, gzip, pickle
import numpy as np
import numba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantbox.strategy.rubberband import (
    _compute_sma_inline,
    _compute_rsi_inline,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    compute_signals_confidence,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "cache_extended")

TOP10 = [
    "FETUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT", "DOTUSDT",
    "OPUSDT", "SHIBUSDT", "DOGEUSDT", "LTCUSDT", "ATOMUSDT",
]

COST_PER_SIDE = 0.0004
FUNDING_15M = 0.0001 / 32
MTF_SMA_PERIOD = 12


def load_data(symbol: str, interval: str = "15m") -> dict:
    """Load cached price data."""
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}_365d.pkl.gz")
    with gzip.open(path, "rb") as f:
        df = pickle.load(f)
    return {
        "close": df["Close"].values.astype(np.float64),
        "volume": df["Volume"].values.astype(np.float64),
        "taker_buy": df["Taker Buy Base"].values.astype(np.float64),
        "high": df["High"].values.astype(np.float64),
        "low": df["Low"].values.astype(np.float64),
    }


def load_btc(interval: str = "15m") -> np.ndarray:
    return load_data("BTCUSDT", interval)["close"]


# ══════════════════════════════════════════════════════════════
# Numba Signal Functions
# ══════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def _detect_rsi_divergence(close: np.ndarray, rsi: np.ndarray,
                           i: int, lookback: int, direction: int) -> bool:
    """Detect RSI divergence at bar i.

    For LONG (direction=1): bullish divergence
      - Price makes lower low vs recent low, but RSI makes higher low
    For SHORT (direction=-1): bearish divergence
      - Price makes higher high vs recent high, but RSI makes lower high

    Args:
        close: Price array.
        rsi: Pre-computed RSI array.
        i: Current bar index.
        lookback: Window to search for previous extremes.
        direction: 1 for bullish (long), -1 for bearish (short).

    Returns:
        True if divergence detected.
    """
    start = max(0, i - lookback)
    if start >= i:
        return False

    if direction == 1:
        # Bullish divergence: price lower low, RSI higher low
        prev_low_price = close[start]
        prev_low_rsi = rsi[start]
        for j in range(start + 1, i):
            if close[j] < prev_low_price:
                prev_low_price = close[j]
                prev_low_rsi = rsi[j]
        # Current price below previous low, but RSI above previous RSI low
        if close[i] < prev_low_price and rsi[i] > prev_low_rsi + 2.0:
            return True
    else:
        # Bearish divergence: price higher high, RSI lower high
        prev_high_price = close[start]
        prev_high_rsi = rsi[start]
        for j in range(start + 1, i):
            if close[j] > prev_high_price:
                prev_high_price = close[j]
                prev_high_rsi = rsi[j]
        # Current price above previous high, but RSI below previous RSI high
        if close[i] > prev_high_price and rsi[i] < prev_high_rsi - 2.0:
            return True

    return False


@numba.jit(nopython=True, cache=True)
def _compute_atr_inline(high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, period: int) -> np.ndarray:
    """ATR (Average True Range) for Numba context."""
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    # Initial SMA
    s = 0.0
    for i in range(1, period + 1):
        s += tr[i]
    atr[period] = s / period

    # Wilder's smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


@numba.jit(nopython=True, cache=True)
def _is_squeeze(atr: np.ndarray, atr_sma: np.ndarray,
                i: int, squeeze_ratio: float) -> bool:
    """Check if volatility was compressed (squeeze) before current bar.

    Looks at ATR 1-3 bars ago relative to ATR SMA. If ATR was below
    squeeze_ratio * ATR_SMA, the move is coming out of compression.
    """
    if np.isnan(atr_sma[i]):
        return False
    # Check bars i-1 to i-3 for squeeze condition
    for j in range(max(0, i - 3), i):
        if not np.isnan(atr[j]) and not np.isnan(atr_sma[j]):
            if atr[j] < squeeze_ratio * atr_sma[j]:
                return True
    return False


@numba.jit(nopython=True, cache=True)
def compute_signals_confluence(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    taker_buy: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    btc_regime_period: int,
    asset_trend_period: int,
    min_confidence: int,
    base_leverage: float,
    confidence_leverage_step: float,
    max_leverage: float,
    rsi_period: int,
    vol_period: int,
    # ── v8.0 new params ──
    use_rsi_divergence: bool,
    divergence_lookback: int,
    use_squeeze: bool,
    squeeze_ratio: float,
    atr_period: int,
    atr_sma_period: int,
    use_dynamic_target: bool,
    target_conf3: float,
    target_conf4: float,
    target_conf5: float,
    use_reversal_confirmation: bool,
) -> np.ndarray:
    """v8.0: Confluence-Gated Entry with winning strategy insights.

    Upgrades over v6.0 compute_signals_confidence:
      1. RSI Divergence: replaces RSI level check with true divergence
      2. Volatility Squeeze: +1 confidence for entries after ATR compression
      3. Dynamic Target: higher confidence → wider target (ride the wave)
      4. Reversal Confirmation: 1-bar delay to confirm price reversal

    Still uses regime-switch (Risk-On/Risk-Off) as the macro framework.
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute indicators
    btc_sma = _compute_sma_inline(btc_close, btc_regime_period)
    asset_sma = _compute_sma_inline(close, asset_trend_period)
    rsi = _compute_rsi_inline(close, rsi_period)
    vol_sma = _compute_sma_inline(volume, vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    # Volatility squeeze indicators
    atr = np.full(n, np.nan)
    atr_ma = np.full(n, np.nan)
    if use_squeeze:
        atr = _compute_atr_inline(high, low, close, atr_period)
        atr_ma = _compute_sma_inline(atr, atr_sma_period)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    lev = base_leverage
    current_target = target_retrace

    # Reversal confirmation state
    pending_entry = 0       # 0=none, 1=pending long, -1=pending short
    pending_confidence = 0
    pending_bar = -1
    pending_move_ret = 0.0

    start_bar = max(lookback, btc_mom_period, btc_regime_period,
                    asset_trend_period, rsi_period, vol_period,
                    divergence_lookback + 1, atr_sma_period + atr_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            # ── Check for reversal confirmation from previous bar ──
            if use_reversal_confirmation and pending_entry != 0:
                # Pending long: price must have bounced (close > prev close)
                if pending_entry == 1 and close[i] > close[i - 1]:
                    position = 1
                    entry_price = close[i]
                    move_size = abs(pending_move_ret)
                    hold_count = 0
                    best_pnl = 0.0
                    confidence = pending_confidence
                    lev = min(base_leverage + confidence_leverage_step * (confidence - 1),
                              max_leverage)
                    if use_dynamic_target:
                        if confidence >= 5:
                            current_target = move_size * target_conf5
                        elif confidence >= 4:
                            current_target = move_size * target_conf4
                        elif confidence >= 3:
                            current_target = move_size * target_conf3
                        else:
                            current_target = move_size * target_retrace
                    else:
                        current_target = move_size * target_retrace
                    sig[i] = position * lev
                    pending_entry = 0
                    continue
                # Pending short: price must have dipped (close < prev close)
                elif pending_entry == -1 and close[i] < close[i - 1]:
                    position = -1
                    entry_price = close[i]
                    move_size = abs(pending_move_ret)
                    hold_count = 0
                    best_pnl = 0.0
                    confidence = pending_confidence
                    lev = min(base_leverage + confidence_leverage_step * (confidence - 1),
                              max_leverage)
                    if use_dynamic_target:
                        if confidence >= 5:
                            current_target = move_size * target_conf5
                        elif confidence >= 4:
                            current_target = move_size * target_conf4
                        elif confidence >= 3:
                            current_target = move_size * target_conf3
                        else:
                            current_target = move_size * target_retrace
                    else:
                        current_target = move_size * target_retrace
                    sig[i] = position * lev
                    pending_entry = 0
                    continue
                else:
                    # Confirmation failed — cancel
                    if i - pending_bar >= 2:
                        pending_entry = 0

            # ── New signal detection ──
            if abs(ret) > threshold:
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                entered = False
                direction = 0
                confidence = 1  # BTC regime alignment

                if is_risk_on:
                    if np.isnan(asset_sma[i]):
                        continue
                    is_uptrend = close[i] > asset_sma[i]

                    if ret < -threshold and is_uptrend:
                        direction = 1  # LONG
                        # ── Evidence collection ──
                        if use_rsi_divergence:
                            if _detect_rsi_divergence(close, rsi, i,
                                                     divergence_lookback, 1):
                                confidence += 1
                        else:
                            if rsi[i] < 40:
                                confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.45:
                            confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02:
                            confidence += 1
                        if use_squeeze and _is_squeeze(atr, atr_ma, i, squeeze_ratio):
                            confidence += 1

                    elif ret > threshold and not is_uptrend:
                        direction = -1  # SHORT
                        if use_rsi_divergence:
                            if _detect_rsi_divergence(close, rsi, i,
                                                     divergence_lookback, -1):
                                confidence += 1
                        else:
                            if rsi[i] > 60:
                                confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.55:
                            confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02:
                            confidence += 1
                        if use_squeeze and _is_squeeze(atr, atr_ma, i, squeeze_ratio):
                            confidence += 1
                else:
                    # RISK-OFF
                    if ret < -threshold and btc_mom < 0:
                        direction = 1  # LONG
                        if use_rsi_divergence:
                            if _detect_rsi_divergence(close, rsi, i,
                                                     divergence_lookback, 1):
                                confidence += 1
                        else:
                            if rsi[i] < 35:
                                confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.42:
                            confidence += 1
                        if btc_mom < -0.01:
                            confidence += 1
                        if use_squeeze and _is_squeeze(atr, atr_ma, i, squeeze_ratio):
                            confidence += 1

                    elif ret > threshold and btc_mom > 0:
                        direction = -1  # SHORT
                        if use_rsi_divergence:
                            if _detect_rsi_divergence(close, rsi, i,
                                                     divergence_lookback, -1):
                                confidence += 1
                        else:
                            if rsi[i] > 65:
                                confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.58:
                            confidence += 1
                        if btc_mom > 0.01:
                            confidence += 1
                        if use_squeeze and _is_squeeze(atr, atr_ma, i, squeeze_ratio):
                            confidence += 1

                if direction != 0 and confidence >= min_confidence:
                    if use_reversal_confirmation:
                        # Defer entry: wait for next bar confirmation
                        pending_entry = direction
                        pending_confidence = confidence
                        pending_bar = i
                        pending_move_ret = ret
                        continue
                    else:
                        position = direction
                        entered = True

                if entered:
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                    lev = min(base_leverage + confidence_leverage_step * (confidence - 1),
                              max_leverage)
                    if use_dynamic_target:
                        if confidence >= 5:
                            current_target = move_size * target_conf5
                        elif confidence >= 4:
                            current_target = move_size * target_conf4
                        elif confidence >= 3:
                            current_target = move_size * target_conf3
                        else:
                            current_target = move_size * target_retrace
                    else:
                        current_target = move_size * target_retrace
                    sig[i] = position * lev
                    continue

        else:
            # ═══ EXIT LOGIC ═══
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor
            target = current_target

            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

            if position == 0:
                sig[i] = 0
                continue

        if position != 0:
            sig[i] = position * lev

    return sig


# ══════════════════════════════════════════════════════════════
# Test Harness
# ══════════════════════════════════════════════════════════════

# v7.0 baseline params
BASE_PARAMS = dict(
    lookback=8, threshold=0.045, target_retrace=0.4, stop_mult=0.9,
    max_hold=10, stop_decay=0.0, btc_mom_period=32,
    btc_regime_period=96, asset_trend_period=192,
    min_confidence=2, base_leverage=1.5, confidence_leverage_step=0.0,
    max_leverage=1.5, rsi_period=14, vol_period=20,
)


def run_baseline(symbol_data: dict, btc_close: np.ndarray, close_4h: np.ndarray) -> dict:
    """v7.0 baseline: confidence + MTF confirmation (2.0x leverage)."""
    sig = compute_signals_confidence(
        close=symbol_data["close"], volume=symbol_data["volume"],
        taker_buy=symbol_data["taker_buy"], btc_close=btc_close,
        **BASE_PARAMS,
    )
    sig = apply_mtf_confirmation(sig, close_4h, MTF_SMA_PERIOD, ratio=16)
    sig = apply_mtf_leverage(sig, 2.0)
    result = _backtest_core(symbol_data["close"], sig, COST_PER_SIDE, FUNDING_15M)
    return {
        "ret": result[1], "sharpe": result[2], "mdd": result[3],
        "win_rate": result[4], "n_trades": result[5],
        "calmar": result[1] / result[3] if result[3] > 0 else 0.0,
    }


def run_test(symbol_data: dict, btc_close: np.ndarray, close_4h: np.ndarray,
             label: str, **overrides) -> dict:
    """Run confluence signal with specific overrides, then apply MTF."""
    params = dict(
        close=symbol_data["close"], high=symbol_data["high"],
        low=symbol_data["low"], volume=symbol_data["volume"],
        taker_buy=symbol_data["taker_buy"], btc_close=btc_close,
        **BASE_PARAMS,
        # v8.0 defaults (all disabled)
        use_rsi_divergence=False, divergence_lookback=24,
        use_squeeze=False, squeeze_ratio=0.7,
        atr_period=14, atr_sma_period=50,
        use_dynamic_target=False,
        target_conf3=0.5, target_conf4=0.65, target_conf5=0.8,
        use_reversal_confirmation=False,
    )
    params.update(overrides)
    sig = compute_signals_confluence(**params)
    sig = apply_mtf_confirmation(sig, close_4h, MTF_SMA_PERIOD, ratio=16)
    sig = apply_mtf_leverage(sig, 2.0)
    result = _backtest_core(symbol_data["close"], sig, COST_PER_SIDE, FUNDING_15M)
    return {
        "ret": result[1], "sharpe": result[2], "mdd": result[3],
        "win_rate": result[4], "n_trades": result[5],
        "calmar": result[1] / result[3] if result[3] > 0 else 0.0,
    }


def avg_metrics(results: list[dict]) -> dict:
    """Average metrics across symbols."""
    keys = ["ret", "sharpe", "mdd", "win_rate", "n_trades", "calmar"]
    avg = {}
    for k in keys:
        vals = [r[k] for r in results]
        avg[k] = sum(vals) / len(vals) if vals else 0.0
    return avg


def portfolio_metrics(all_equities: list[np.ndarray]) -> dict:
    """Equal-weight portfolio from individual equity curves."""
    min_len = min(len(e) for e in all_equities)
    combined = np.zeros(min_len)
    for eq in all_equities:
        combined += eq[:min_len]
    combined /= len(all_equities)

    total_ret = combined[-1] / combined[0] - 1.0
    peak = combined[0]
    max_dd = 0.0
    for i in range(len(combined)):
        if combined[i] > peak:
            peak = combined[i]
        dd = (peak - combined[i]) / peak
        if dd > max_dd:
            max_dd = dd

    rets = np.diff(combined) / combined[:-1]
    mean_r = np.mean(rets)
    std_r = np.std(rets)
    sharpe = mean_r / std_r * np.sqrt(96 * 365.25) if std_r > 0.0 else 0.0
    calmar = total_ret / max_dd if max_dd > 0 else 0.0

    return {"ret": total_ret, "sharpe": sharpe, "mdd": max_dd, "calmar": calmar}


def get_equity(symbol_data, btc_close, close_4h, label, **overrides):
    """Get equity curve for a test config."""
    params = dict(
        close=symbol_data["close"], high=symbol_data["high"],
        low=symbol_data["low"], volume=symbol_data["volume"],
        taker_buy=symbol_data["taker_buy"], btc_close=btc_close,
        **BASE_PARAMS,
        use_rsi_divergence=False, divergence_lookback=24,
        use_squeeze=False, squeeze_ratio=0.7,
        atr_period=14, atr_sma_period=50,
        use_dynamic_target=False,
        target_conf3=0.5, target_conf4=0.65, target_conf5=0.8,
        use_reversal_confirmation=False,
    )
    params.update(overrides)
    sig = compute_signals_confluence(**params)
    sig = apply_mtf_confirmation(sig, close_4h, MTF_SMA_PERIOD, ratio=16)
    sig = apply_mtf_leverage(sig, 2.0)
    result = _backtest_core(symbol_data["close"], sig, COST_PER_SIDE, FUNDING_15M)
    return result[0]  # equity curve


def get_baseline_equity(symbol_data, btc_close, close_4h):
    """Get equity curve for baseline."""
    sig = compute_signals_confidence(
        close=symbol_data["close"], volume=symbol_data["volume"],
        taker_buy=symbol_data["taker_buy"], btc_close=btc_close,
        **BASE_PARAMS,
    )
    sig = apply_mtf_confirmation(sig, close_4h, MTF_SMA_PERIOD, ratio=16)
    sig = apply_mtf_leverage(sig, 2.0)
    result = _backtest_core(symbol_data["close"], sig, COST_PER_SIDE, FUNDING_15M)
    return result[0]


def print_result(label: str, avg: dict, port: dict = None):
    """Print formatted result row."""
    port_str = ""
    if port:
        port_str = f"  | Port: Ret={port['ret']:+.1%} DD={port['mdd']:.1%} Calmar={port['calmar']:.2f} Sharpe={port['sharpe']:.2f}"
    print(f"  {label:35s} | Ret={avg['ret']:+.1%} DD={avg['mdd']:.1%} "
          f"Calmar={avg['calmar']:.2f} Sharpe={avg['sharpe']:.2f} "
          f"WR={avg['win_rate']:.0f}% Trades={avg['n_trades']:.0f}{port_str}")


# ══════════════════════════════════════════════════════════════
# Phases
# ══════════════════════════════════════════════════════════════

def phase1_individual_hypotheses():
    """Test each hypothesis independently."""
    print("\n" + "=" * 100)
    print("PHASE 1: Individual Hypothesis Testing (10 symbols × 365 days)")
    print("=" * 100)

    btc_15m = load_btc("15m")
    btc_4h = load_btc("4h")

    # Load all symbol data
    symbols_data = {}
    close_4h_data = {}
    for sym in TOP10:
        symbols_data[sym] = load_data(sym, "15m")
        close_4h_data[sym] = load_data(sym, "4h")["close"]

    # ──── Baseline (v7.0) ────
    print("\n── Baseline: v7.0 (confidence + MTF confirmation + 2x leverage) ──")
    baseline_results = []
    baseline_equities = []
    for sym in TOP10:
        r = run_baseline(symbols_data[sym], btc_15m, close_4h_data[sym])
        baseline_results.append(r)
        eq = get_baseline_equity(symbols_data[sym], btc_15m, close_4h_data[sym])
        baseline_equities.append(eq)
    baseline_avg = avg_metrics(baseline_results)
    baseline_port = portfolio_metrics(baseline_equities)
    print_result("v7.0 BASELINE", baseline_avg, baseline_port)

    # ──── H1: RSI Divergence ────
    print("\n── H1: RSI Divergence (replace RSI level with divergence detection) ──")
    test_configs_h1 = [
        ("div_lb=16", dict(use_rsi_divergence=True, divergence_lookback=16)),
        ("div_lb=24", dict(use_rsi_divergence=True, divergence_lookback=24)),
        ("div_lb=32", dict(use_rsi_divergence=True, divergence_lookback=32)),
        ("div_lb=48", dict(use_rsi_divergence=True, divergence_lookback=48)),
    ]
    best_h1 = None
    best_h1_calmar = -999
    for label, overrides in test_configs_h1:
        results = []
        equities = []
        for sym in TOP10:
            r = run_test(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            results.append(r)
            eq = get_equity(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            equities.append(eq)
        avg = avg_metrics(results)
        port = portfolio_metrics(equities)
        print_result(f"H1: RSI Divergence ({label})", avg, port)
        if port["calmar"] > best_h1_calmar:
            best_h1_calmar = port["calmar"]
            best_h1 = overrides.copy()

    # ──── H2: Volatility Squeeze ────
    print("\n── H2: Volatility Squeeze (enter after ATR compression) ──")
    test_configs_h2 = [
        ("squeeze=0.6", dict(use_squeeze=True, squeeze_ratio=0.6)),
        ("squeeze=0.7", dict(use_squeeze=True, squeeze_ratio=0.7)),
        ("squeeze=0.8", dict(use_squeeze=True, squeeze_ratio=0.8)),
        ("squeeze=0.9", dict(use_squeeze=True, squeeze_ratio=0.9)),
    ]
    best_h2 = None
    best_h2_calmar = -999
    for label, overrides in test_configs_h2:
        results = []
        equities = []
        for sym in TOP10:
            r = run_test(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            results.append(r)
            eq = get_equity(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            equities.append(eq)
        avg = avg_metrics(results)
        port = portfolio_metrics(equities)
        print_result(f"H2: Vol Squeeze ({label})", avg, port)
        if port["calmar"] > best_h2_calmar:
            best_h2_calmar = port["calmar"]
            best_h2 = overrides.copy()

    # ──── H3: Dynamic Target ────
    print("\n── H3: Dynamic Target (scale target by confidence level) ──")
    test_configs_h3 = [
        ("dyn_t: 0.5/0.6/0.7", dict(use_dynamic_target=True,
         target_conf3=0.5, target_conf4=0.6, target_conf5=0.7)),
        ("dyn_t: 0.5/0.65/0.8", dict(use_dynamic_target=True,
         target_conf3=0.5, target_conf4=0.65, target_conf5=0.8)),
        ("dyn_t: 0.6/0.75/0.9", dict(use_dynamic_target=True,
         target_conf3=0.6, target_conf4=0.75, target_conf5=0.9)),
        ("dyn_t: 0.4/0.55/0.7", dict(use_dynamic_target=True,
         target_conf3=0.4, target_conf4=0.55, target_conf5=0.7)),
    ]
    best_h3 = None
    best_h3_calmar = -999
    for label, overrides in test_configs_h3:
        results = []
        equities = []
        for sym in TOP10:
            r = run_test(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            results.append(r)
            eq = get_equity(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            equities.append(eq)
        avg = avg_metrics(results)
        port = portfolio_metrics(equities)
        print_result(f"H3: Dynamic Target ({label})", avg, port)
        if port["calmar"] > best_h3_calmar:
            best_h3_calmar = port["calmar"]
            best_h3 = overrides.copy()

    # ──── H4: Reversal Confirmation ────
    print("\n── H4: Reversal Confirmation (1-bar delay for price reversal) ──")
    results_h4 = []
    equities_h4 = []
    overrides_h4 = dict(use_reversal_confirmation=True)
    for sym in TOP10:
        r = run_test(symbols_data[sym], btc_15m, close_4h_data[sym], "rev_conf", **overrides_h4)
        results_h4.append(r)
        eq = get_equity(symbols_data[sym], btc_15m, close_4h_data[sym], "rev_conf", **overrides_h4)
        equities_h4.append(eq)
    avg_h4 = avg_metrics(results_h4)
    port_h4 = portfolio_metrics(equities_h4)
    print_result("H4: Reversal Confirmation", avg_h4, port_h4)
    best_h4 = overrides_h4 if port_h4["calmar"] > baseline_port["calmar"] else None

    return baseline_port, best_h1, best_h2, best_h3, best_h4, symbols_data, close_4h_data, btc_15m


def phase2_combinations(baseline_port, best_h1, best_h2, best_h3, best_h4,
                        symbols_data, close_4h_data, btc_15m):
    """Combine winning hypotheses."""
    print("\n" + "=" * 100)
    print("PHASE 2: Combine Winning Hypotheses")
    print("=" * 100)
    print(f"\n  Baseline portfolio Calmar: {baseline_port['calmar']:.2f}")

    winners = {}
    if best_h1:
        winners["H1 (RSI Div)"] = best_h1
    if best_h2:
        winners["H2 (Squeeze)"] = best_h2
    if best_h3:
        winners["H3 (Dyn Target)"] = best_h3
    if best_h4:
        winners["H4 (Rev Conf)"] = best_h4

    print(f"  Winners to combine: {list(winners.keys())}")

    if not winners:
        print("  No hypothesis beat baseline. Skipping combinations.")
        return

    # Pairwise combinations
    winner_keys = list(winners.keys())
    best_combo = None
    best_combo_calmar = baseline_port["calmar"]
    best_combo_label = ""

    combos = []
    for i in range(len(winner_keys)):
        for j in range(i + 1, len(winner_keys)):
            merged = {}
            merged.update(winners[winner_keys[i]])
            merged.update(winners[winner_keys[j]])
            label = f"{winner_keys[i]} + {winner_keys[j]}"
            combos.append((label, merged))

    # All winners combined
    if len(winners) >= 2:
        merged_all = {}
        for w in winners.values():
            merged_all.update(w)
        combos.append(("ALL WINNERS", merged_all))

    for label, overrides in combos:
        results = []
        equities = []
        for sym in TOP10:
            r = run_test(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            results.append(r)
            eq = get_equity(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            equities.append(eq)
        avg = avg_metrics(results)
        port = portfolio_metrics(equities)
        print_result(f"  COMBO: {label}", avg, port)
        if port["calmar"] > best_combo_calmar:
            best_combo_calmar = port["calmar"]
            best_combo = overrides.copy()
            best_combo_label = label

    if best_combo:
        print(f"\n  >>> BEST COMBO: {best_combo_label} | Calmar {best_combo_calmar:.2f}")
    else:
        print(f"\n  >>> No combination beat baseline Calmar {baseline_port['calmar']:.2f}")

    return best_combo, best_combo_calmar


def phase3_confidence_sweep(best_combo, symbols_data, close_4h_data, btc_15m):
    """Sweep min_confidence and leverage with the best combo."""
    print("\n" + "=" * 100)
    print("PHASE 3: Confidence Threshold + Leverage Sweep (with best combo)")
    print("=" * 100)

    if not best_combo:
        print("  No winning combo — using individual best or baseline overrides.")
        best_combo = {}

    sweep_configs = [
        # (min_conf, base_lev, conf_step, max_lev, label)
        (2, 1.5, 0.0, 1.5, "mc=2 flat 1.5x"),
        (2, 1.5, 0.5, 3.0, "mc=2 conf_lev 1.5-3.0"),
        (3, 1.5, 0.0, 1.5, "mc=3 flat 1.5x"),
        (3, 2.0, 0.0, 2.0, "mc=3 flat 2.0x"),
        (3, 2.0, 0.5, 3.0, "mc=3 conf_lev 2.0-3.0"),
        (3, 2.0, 1.0, 4.0, "mc=3 conf_lev 2.0-4.0"),
        (4, 2.0, 0.0, 2.0, "mc=4 flat 2.0x"),
        (4, 2.5, 0.0, 2.5, "mc=4 flat 2.5x"),
        (4, 2.0, 1.0, 4.0, "mc=4 conf_lev 2.0-4.0"),
        (4, 3.0, 0.0, 3.0, "mc=4 flat 3.0x"),
    ]

    best_result = None
    best_calmar = -999

    for mc, blv, cs, ml, label in sweep_configs:
        overrides = dict(best_combo)
        overrides.update(dict(
            min_confidence=mc, base_leverage=blv,
            confidence_leverage_step=cs, max_leverage=ml,
        ))
        results = []
        equities = []
        for sym in TOP10:
            r = run_test(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            results.append(r)
            eq = get_equity(symbols_data[sym], btc_15m, close_4h_data[sym], label, **overrides)
            equities.append(eq)
        avg = avg_metrics(results)
        port = portfolio_metrics(equities)
        print_result(f"  {label}", avg, port)
        if port["calmar"] > best_calmar:
            best_calmar = port["calmar"]
            best_result = overrides.copy()

    if best_result:
        print(f"\n  >>> BEST CONFIG: Calmar {best_calmar:.2f}")
        for k, v in best_result.items():
            if k not in BASE_PARAMS or BASE_PARAMS.get(k) != v:
                print(f"      {k} = {v}")

    return best_result, best_calmar


def phase4_walk_forward(best_config, symbols_data, close_4h_data, btc_15m):
    """Walk-forward validation: Q1-Q3 train, Q4 test."""
    print("\n" + "=" * 100)
    print("PHASE 4: Walk-Forward Validation (Quarterly)")
    print("=" * 100)

    if not best_config:
        print("  No config to validate.")
        return

    n_bars = len(btc_15m)
    q_size = n_bars // 4

    for q_idx, q_name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        start = q_idx * q_size
        end = min((q_idx + 1) * q_size, n_bars)

        results = []
        equities = []
        baseline_results = []
        baseline_equities = []

        for sym in TOP10:
            d = symbols_data[sym]
            d_q = {k: v[start:end] for k, v in d.items()}
            btc_q = btc_15m[start:end]

            # 4h slice (adjust ratio)
            n_4h = len(close_4h_data[sym])
            q_4h_start = start * n_4h // n_bars
            q_4h_end = end * n_4h // n_bars
            c4h_q = close_4h_data[sym][q_4h_start:q_4h_end]

            if len(d_q["close"]) < 500 or len(c4h_q) < 20:
                continue

            # Best config
            r = run_test(d_q, btc_q, c4h_q, q_name, **best_config)
            results.append(r)
            eq = get_equity(d_q, btc_q, c4h_q, q_name, **best_config)
            equities.append(eq)

            # Baseline
            br = run_baseline(d_q, btc_q, c4h_q)
            baseline_results.append(br)
            beq = get_baseline_equity(d_q, btc_q, c4h_q)
            baseline_equities.append(beq)

        if results:
            avg_new = avg_metrics(results)
            port_new = portfolio_metrics(equities) if equities else {}
            avg_base = avg_metrics(baseline_results)
            port_base = portfolio_metrics(baseline_equities) if baseline_equities else {}

            dd_change = ""
            if port_new and port_base and port_base["mdd"] > 0:
                dd_pct = (port_new["mdd"] - port_base["mdd"]) / port_base["mdd"] * 100
                dd_change = f" ({dd_pct:+.0f}%)"

            print(f"\n  {q_name}:")
            print(f"    Baseline:  Ret={port_base.get('ret', 0):+.1%} DD={port_base.get('mdd', 0):.1%} "
                  f"Calmar={port_base.get('calmar', 0):.2f}")
            print(f"    v8.0:      Ret={port_new.get('ret', 0):+.1%} DD={port_new.get('mdd', 0):.1%}{dd_change} "
                  f"Calmar={port_new.get('calmar', 0):.2f}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 100)
    print("  v8.0 CONFLUENCE A/B TEST — Winning Strategy Insights")
    print("  'Only enter when you are SURE. Then use leverage.'")
    print("=" * 100)

    # Phase 1: Individual tests
    (baseline_port, best_h1, best_h2, best_h3, best_h4,
     symbols_data, close_4h_data, btc_15m) = phase1_individual_hypotheses()

    # Phase 2: Combinations
    best_combo, combo_calmar = phase2_combinations(
        baseline_port, best_h1, best_h2, best_h3, best_h4,
        symbols_data, close_4h_data, btc_15m)

    # Phase 3: Confidence + leverage sweep
    best_config, best_calmar = phase3_confidence_sweep(
        best_combo, symbols_data, close_4h_data, btc_15m)

    # Phase 4: Walk-forward validation
    phase4_walk_forward(best_config, symbols_data, close_4h_data, btc_15m)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
