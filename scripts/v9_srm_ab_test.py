#!/usr/bin/env python3
"""
v9.0 A/B Test: S+R+M Confluence Pipeline (Winning Strategy Deep Integration)
=============================================================================
Tests 6 quantifiable improvements from the updated competition winner strategy:

  H1: Stochastic Cross — add %K/%D cross as confidence evidence
  H2: MACD Histogram Reversal — add MACD hist slope as confidence evidence
  H3: EMA Regime Detection — replace BTC SMA with EMA for faster response
  H4: Fibonacci Exit Levels — Fib-based target/stop instead of fixed ratios
  H5: Capital Preservation + MTF — re-test partial exit with MTF filtering
  H6: Sequential Confluence Gate — RSI + Stoch + MACD all must confirm

Each hypothesis tested on 10 curated symbols × 365 days.
Baseline: v8.0 (compute_signals_confidence + MTF + dynamic target).

"Be brave, challenging and persist." — User
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
from quantbox.backtest.vectorized_backtest import _backtest_core, _backtest_core_partial

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

# ── v8.0 baseline params ──
BASE_PARAMS = dict(
    lookback=8, threshold=0.045, target_retrace=0.4,
    stop_mult=0.9, max_hold=10, stop_decay=0.0,
    btc_mom_period=32, btc_regime_period=96, asset_trend_period=192,
    min_confidence=2, base_leverage=1.5,
    confidence_leverage_step=0.0, max_leverage=1.5,
    rsi_period=14, vol_period=20,
    dynamic_target_enabled=True, target_conf3=0.45,
    target_conf4=0.50, target_conf5=0.55,
)


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
# Numba Helpers: Stochastic, MACD, EMA
# ══════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def _compute_ema_inline(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA helper for Numba. Uses Wilder-style warm-up (SMA for first period)."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    # Warm-up: SMA of first `period` values
    s = 0.0
    for i in range(period):
        s += arr[i]
    out[period - 1] = s / period
    alpha = 2.0 / (period + 1.0)
    for i in range(period, n):
        out[i] = (arr[i] - out[i - 1]) * alpha + out[i - 1]
    return out


@numba.jit(nopython=True, cache=True)
def _compute_stochastic_inline(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple:
    """Stochastic %K and %D arrays. %K = raw stochastic, %D = SMA(%K, d_period)."""
    n = len(close)
    k = np.full(n, 50.0)
    d = np.full(n, 50.0)

    for i in range(k_period - 1, n):
        hh = high[i]
        ll = low[i]
        for j in range(i - k_period + 1, i):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        rng = hh - ll
        if rng > 0:
            k[i] = (close[i] - ll) / rng * 100.0
        else:
            k[i] = 50.0

    # %D = SMA of %K
    s = 0.0
    for i in range(n):
        s += k[i]
        if i >= d_period:
            s -= k[i - d_period]
        if i >= d_period - 1:
            d[i] = s / d_period

    return k, d


@numba.jit(nopython=True, cache=True)
def _compute_macd_inline(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple:
    """MACD line, signal line, histogram."""
    ema_fast = _compute_ema_inline(close, fast)
    ema_slow = _compute_ema_inline(close, slow)
    n = len(close)
    macd_line = np.full(n, 0.0)
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = _compute_ema_inline(macd_line, signal_period)
    hist = np.full(n, 0.0)
    for i in range(n):
        if not np.isnan(signal_line[i]):
            hist[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, hist


# ══════════════════════════════════════════════════════════════
# v9.0 Signal Function: Full Confluence Engine
# ══════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def compute_signals_v9(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    taker_buy: np.ndarray,
    btc_close: np.ndarray,
    # ── core params ──
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
    # ── v8.0 dynamic target ──
    dynamic_target_enabled: bool,
    target_conf3: float,
    target_conf4: float,
    target_conf5: float,
    # ── v9.0 new features (toggles) ──
    stoch_enabled: bool = False,        # H1: Stochastic cross
    stoch_k_period: int = 14,
    stoch_d_period: int = 3,
    macd_enabled: bool = False,         # H2: MACD histogram reversal
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    ema_regime: bool = False,           # H3: EMA regime (replace SMA)
    ema_regime_period: int = 96,        # EMA period for BTC regime
    fib_exits: bool = False,            # H4: Fibonacci exit levels
    fib_target: float = 0.618,          # Fib target level
    fib_stop: float = 0.886,            # Fib stop level (deeper than 0.9)
    partial_exit: bool = False,         # H5: Capital preservation
    confluence_gate: bool = False,      # H6: All 3 must confirm
) -> np.ndarray:
    """v9.0 Confluence Engine: S+R+M pipeline from winning strategy.

    Builds on v8.0 (confidence scoring + dynamic target) and adds:
      - Stochastic %K/%D crossover as evidence
      - MACD histogram slope reversal as evidence
      - EMA regime detection (replaces SMA)
      - Fibonacci-based stop/target
      - Capital preservation (50% partial → break-even)
      - Sequential confluence gate (RSI+Stoch+MACD all must confirm)

    Each feature has an independent toggle for A/B testing.

    Returns:
        Signal array with leverage encoded. For partial exit: ±0.5 = half position.
    """
    n = len(close)
    sig = np.zeros(n)

    # ── Pre-compute indicators ──
    if ema_regime:
        btc_trend = _compute_ema_inline(btc_close, ema_regime_period)
        asset_trend = _compute_ema_inline(close, asset_trend_period)
    else:
        btc_trend = _compute_sma_inline(btc_close, btc_regime_period)
        asset_trend = _compute_sma_inline(close, asset_trend_period)

    rsi = _compute_rsi_inline(close, rsi_period)
    vol_sma = _compute_sma_inline(volume, vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    # Stochastic
    stoch_k = np.full(n, 50.0)
    stoch_d = np.full(n, 50.0)
    if stoch_enabled or confluence_gate:
        stoch_k, stoch_d = _compute_stochastic_inline(high, low, close,
                                                       stoch_k_period, stoch_d_period)

    # MACD
    macd_hist = np.full(n, 0.0)
    if macd_enabled or confluence_gate:
        _, _, macd_hist = _compute_macd_inline(close, macd_fast, macd_slow, macd_signal)

    # ── State variables ──
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    lev = base_leverage
    trade_target_retrace = target_retrace
    trade_stop = stop_mult
    phase = 0  # 0=flat, 1=full, 2=half (partial exit)

    eff_regime_period = ema_regime_period if ema_regime else btc_regime_period
    start_bar = max(lookback, btc_mom_period, int(eff_regime_period),
                    asset_trend_period, rsi_period, vol_period, 30)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if phase == 0:
            if abs(ret) > threshold:
                if np.isnan(btc_trend[i]):
                    sig[i] = 0.0
                    continue

                is_risk_on = btc_close[i] > btc_trend[i]
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                entered = False
                confidence = 1  # BTC regime alignment

                if is_risk_on:
                    if np.isnan(asset_trend[i]):
                        sig[i] = 0.0
                        continue
                    is_uptrend = close[i] > asset_trend[i]

                    if ret < -threshold and is_uptrend:
                        # ═══ LONG in Risk-On ═══
                        # Evidence scoring
                        rsi_ok = rsi[i] < 40
                        if rsi_ok:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.45:
                            confidence += 1
                        dist = abs(close[i] - asset_trend[i]) / asset_trend[i]
                        if dist < 0.02:
                            confidence += 1

                        # H1: Stochastic cross evidence
                        if stoch_enabled and i > 0:
                            if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]:
                                confidence += 1

                        # H2: MACD histogram reversal evidence
                        if macd_enabled and i > 0:
                            if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                                confidence += 1

                        # H6: Confluence gate — RSI + Stoch + MACD all must confirm
                        if confluence_gate:
                            gate_pass = rsi_ok
                            gate_pass = gate_pass and (stoch_k[i] < 20)
                            gate_pass = gate_pass and (i > 0 and macd_hist[i] > macd_hist[i - 1])
                            if not gate_pass:
                                sig[i] = 0.0
                                continue

                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > threshold and not is_uptrend:
                        # ═══ SHORT in Risk-On ═══
                        rsi_ok = rsi[i] > 60
                        if rsi_ok:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.55:
                            confidence += 1
                        dist = abs(close[i] - asset_trend[i]) / asset_trend[i]
                        if dist < 0.02:
                            confidence += 1

                        # H1: Stochastic cross (short)
                        if stoch_enabled and i > 0:
                            if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]:
                                confidence += 1

                        # H2: MACD histogram (short)
                        if macd_enabled and i > 0:
                            if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                                confidence += 1

                        # H6: Confluence gate (short)
                        if confluence_gate:
                            gate_pass = rsi_ok
                            gate_pass = gate_pass and (stoch_k[i] > 80)
                            gate_pass = gate_pass and (i > 0 and macd_hist[i] < macd_hist[i - 1])
                            if not gate_pass:
                                sig[i] = 0.0
                                continue

                        if confidence >= min_confidence:
                            position = -1
                            entered = True

                else:
                    # ═══ RISK-OFF regime ═══
                    if ret < -threshold and btc_mom < 0:
                        # LONG sympathy
                        rsi_ok = rsi[i] < 35
                        if rsi_ok:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.42:
                            confidence += 1
                        if btc_mom < -0.01:
                            confidence += 1

                        if stoch_enabled and i > 0:
                            if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]:
                                confidence += 1

                        if macd_enabled and i > 0:
                            if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                                confidence += 1

                        if confluence_gate:
                            gate_pass = rsi_ok
                            gate_pass = gate_pass and (stoch_k[i] < 20)
                            gate_pass = gate_pass and (i > 0 and macd_hist[i] > macd_hist[i - 1])
                            if not gate_pass:
                                sig[i] = 0.0
                                continue

                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > threshold and btc_mom > 0:
                        # SHORT sympathy
                        rsi_ok = rsi[i] > 65
                        if rsi_ok:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.58:
                            confidence += 1
                        if btc_mom > 0.01:
                            confidence += 1

                        if stoch_enabled and i > 0:
                            if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]:
                                confidence += 1

                        if macd_enabled and i > 0:
                            if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                                confidence += 1

                        if confluence_gate:
                            gate_pass = rsi_ok
                            gate_pass = gate_pass and (stoch_k[i] > 80)
                            gate_pass = gate_pass and (i > 0 and macd_hist[i] < macd_hist[i - 1])
                            if not gate_pass:
                                sig[i] = 0.0
                                continue

                        if confidence >= min_confidence:
                            position = -1
                            entered = True

                if entered:
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                    phase = 1
                    lev = min(base_leverage + confidence_leverage_step * (confidence - 1),
                              max_leverage)

                    # v8.0 dynamic target
                    if dynamic_target_enabled:
                        if confidence >= 5:
                            trade_target_retrace = target_conf5
                        elif confidence >= 4:
                            trade_target_retrace = target_conf4
                        elif confidence >= 3:
                            trade_target_retrace = target_conf3
                        else:
                            trade_target_retrace = target_retrace
                    else:
                        trade_target_retrace = target_retrace

                    # H4: Fibonacci exit levels override
                    if fib_exits:
                        trade_target_retrace = fib_target
                        trade_stop = fib_stop
                    else:
                        trade_stop = stop_mult

                    sig[i] = position * lev
                    continue

        elif phase == 1:
            # ═══ Full position exit logic ═══
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * trade_stop * time_factor
            target = move_size * trade_target_retrace

            if gain >= target:
                position = 0
                phase = 0
            elif gain < -effective_stop:
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0
            elif partial_exit and gain > 0:
                # H5: Any profit → close 50%, enter Phase 2
                phase = 2
                best_pnl = gain

            if phase == 0:
                sig[i] = 0.0
                continue
            elif phase == 2:
                sig[i] = position * lev * 0.5  # Half position
                continue

        elif phase == 2:
            # ═══ Half position: break-even stop ═══
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            target = move_size * trade_target_retrace

            if gain >= target:
                position = 0
                phase = 0
            elif gain <= 0:
                # Break-even stop: $0 risk
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0

            if phase == 0:
                sig[i] = 0.0
                continue

        if position != 0:
            if phase == 2:
                sig[i] = position * lev * 0.5
            else:
                sig[i] = position * lev

    return sig


# ══════════════════════════════════════════════════════════════
# Backtest helpers
# ══════════════════════════════════════════════════════════════

def run_baseline(symbol_data: dict, btc_15m: np.ndarray, btc_4h: np.ndarray) -> tuple:
    """Run v8.0 baseline (compute_signals_confidence + MTF + dynamic target)."""
    sigs = compute_signals_confidence(
        symbol_data["close"], symbol_data["volume"], symbol_data["taker_buy"],
        btc_15m, **BASE_PARAMS,
    )
    # Apply MTF filtering
    sigs = apply_mtf_confirmation(sigs, btc_4h, MTF_SMA_PERIOD, 16)
    sigs = apply_mtf_leverage(sigs, 2.0)  # 2.0x MTF mult

    result = _backtest_core(symbol_data["close"], sigs, COST_PER_SIDE, FUNDING_15M)
    return result  # (equity, ret, sharpe, dd, wr, trades, bars, pf, avg, tretns)


def run_v9(symbol_data: dict, btc_15m: np.ndarray, btc_4h: np.ndarray,
           overrides: dict = None) -> tuple:
    """Run v9.0 with optional feature toggles."""
    params = dict(BASE_PARAMS)
    if overrides:
        params.update(overrides)

    # Extract v9-only params
    stoch_enabled = params.pop("stoch_enabled", False)
    stoch_k_period = params.pop("stoch_k_period", 14)
    stoch_d_period = params.pop("stoch_d_period", 3)
    macd_enabled = params.pop("macd_enabled", False)
    macd_fast = params.pop("macd_fast", 12)
    macd_slow = params.pop("macd_slow", 26)
    macd_signal_period = params.pop("macd_signal", 9)
    ema_regime = params.pop("ema_regime", False)
    ema_regime_period = params.pop("ema_regime_period", 96)
    fib_exits = params.pop("fib_exits", False)
    fib_target = params.pop("fib_target", 0.618)
    fib_stop = params.pop("fib_stop", 0.886)
    partial_exit = params.pop("partial_exit", False)
    confluence_gate = params.pop("confluence_gate", False)
    mtf_lev_mult = params.pop("mtf_leverage_mult", 2.0)

    sigs = compute_signals_v9(
        symbol_data["close"], symbol_data["high"], symbol_data["low"],
        symbol_data["volume"], symbol_data["taker_buy"], btc_15m,
        params["lookback"], params["threshold"], params["target_retrace"],
        params["stop_mult"], params["max_hold"], params["stop_decay"],
        params["btc_mom_period"], params["btc_regime_period"],
        params["asset_trend_period"], params["min_confidence"],
        params["base_leverage"], params["confidence_leverage_step"],
        params["max_leverage"], params["rsi_period"], params["vol_period"],
        params["dynamic_target_enabled"], params["target_conf3"],
        params["target_conf4"], params["target_conf5"],
        stoch_enabled, stoch_k_period, stoch_d_period,
        macd_enabled, macd_fast, macd_slow, macd_signal_period,
        ema_regime, ema_regime_period,
        fib_exits, fib_target, fib_stop,
        partial_exit, confluence_gate,
    )

    # Apply MTF filtering + MTF leverage
    sigs = apply_mtf_confirmation(sigs, btc_4h, MTF_SMA_PERIOD, 16)
    sigs = apply_mtf_leverage(sigs, mtf_lev_mult)

    if partial_exit:
        result = _backtest_core_partial(symbol_data["close"], sigs, COST_PER_SIDE, FUNDING_15M)
    else:
        result = _backtest_core(symbol_data["close"], sigs, COST_PER_SIDE, FUNDING_15M)
    return result


def portfolio_metrics(all_equities: list, all_results: list) -> dict:
    """Compute portfolio-level metrics from per-symbol results."""
    if not all_equities:
        return {"ret": 0, "dd": 0, "calmar": 0, "sharpe": 0, "trades": 0, "wr": 0}

    # Equal-weighted portfolio
    min_len = min(len(e) for e in all_equities)
    combined = np.zeros(min_len)
    for eq in all_equities:
        combined += eq[:min_len]
    combined /= len(all_equities)

    total_ret = combined[-1] / combined[0] - 1.0
    peak = combined[0]
    max_dd = 0.0
    for v in combined:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    rets = np.diff(combined) / combined[:-1]
    sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25)) if np.std(rets) > 0 else 0

    total_trades = sum(r[5] for r in all_results)
    total_wins = sum(r[5] * r[4] / 100 for r in all_results)  # trades * win_rate
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    calmar = (total_ret / max_dd) if max_dd > 0 else 999

    return {
        "ret": total_ret * 100,
        "dd": max_dd * 100,
        "calmar": calmar,
        "sharpe": sharpe,
        "trades": total_trades,
        "wr": wr,
    }


# ══════════════════════════════════════════════════════════════
# Phase 1: Individual Hypothesis Testing
# ══════════════════════════════════════════════════════════════

def test_hypothesis(name: str, overrides: dict, symbols: list,
                    btc_15m: np.ndarray, btc_4h: np.ndarray,
                    all_data: dict) -> dict:
    """Test a single hypothesis across all symbols."""
    equities = []
    results = []
    for sym in symbols:
        data = all_data[sym]
        r = run_v9(data, btc_15m, btc_4h, overrides)
        equities.append(r[0])
        results.append(r)
    return portfolio_metrics(equities, results)


def main():
    print("=" * 80)
    print("v9.0 A/B Test: S+R+M Confluence Pipeline (Winning Strategy)")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    t0 = time.time()
    btc_15m = load_btc("15m")
    btc_4h = load_btc("4h")
    all_data = {}
    for sym in TOP10:
        all_data[sym] = load_data(sym, "15m")
    print(f"    Loaded {len(all_data)} symbols in {time.time()-t0:.1f}s")

    # ── JIT warm-up ──
    print("\n[2] JIT warm-up (first run compiles Numba functions)...")
    t0 = time.time()
    warmup_sym = TOP10[0]
    warmup_data = all_data[warmup_sym]
    _ = run_baseline(warmup_data, btc_15m, btc_4h)
    _ = run_v9(warmup_data, btc_15m, btc_4h, {"stoch_enabled": True, "macd_enabled": True,
                                                "ema_regime": True, "fib_exits": True,
                                                "partial_exit": True, "confluence_gate": True})
    print(f"    JIT warm-up done in {time.time()-t0:.1f}s")

    # ── Baseline ──
    print("\n[3] Computing v8.0 Baseline...")
    t0 = time.time()
    base_eq = []
    base_res = []
    for sym in TOP10:
        r = run_baseline(all_data[sym], btc_15m, btc_4h)
        base_eq.append(r[0])
        base_res.append(r)
    baseline = portfolio_metrics(base_eq, base_res)
    print(f"    Baseline: Ret={baseline['ret']:.1f}%, DD={baseline['dd']:.1f}%, "
          f"Calmar={baseline['calmar']:.2f}, Sharpe={baseline['sharpe']:.2f}, "
          f"Trades={baseline['trades']}, WR={baseline['wr']:.1f}% [{time.time()-t0:.1f}s]")

    # ═══════════════════ Phase 1: Individual Tests ═══════════════════
    print("\n" + "=" * 80)
    print("PHASE 1: Individual Hypothesis Testing")
    print("=" * 80)

    hypotheses = {
        "H1: Stochastic Cross": {"stoch_enabled": True},
        "H2: MACD Histogram": {"macd_enabled": True},
        "H3: EMA Regime (96)": {"ema_regime": True, "ema_regime_period": 96},
        "H3b: EMA Regime (50)": {"ema_regime": True, "ema_regime_period": 50},
        "H3c: EMA Regime (200)": {"ema_regime": True, "ema_regime_period": 200},
        "H4: Fib Exits (0.618/0.886)": {"fib_exits": True, "fib_target": 0.618, "fib_stop": 0.886},
        "H4b: Fib Exits (0.5/0.786)": {"fib_exits": True, "fib_target": 0.5, "fib_stop": 0.786},
        "H4c: Fib Exits (0.382/0.618)": {"fib_exits": True, "fib_target": 0.382, "fib_stop": 0.618},
        "H5: Capital Preservation": {"partial_exit": True},
        "H6: Confluence Gate (RSI+Stoch+MACD)": {"confluence_gate": True},
        "H1+H2: Stoch+MACD": {"stoch_enabled": True, "macd_enabled": True},
    }

    results_p1 = {}
    for name, ov in hypotheses.items():
        t0 = time.time()
        m = test_hypothesis(name, ov, TOP10, btc_15m, btc_4h, all_data)
        dt = time.time() - t0

        calmar_delta = ((m["calmar"] / baseline["calmar"]) - 1) * 100 if baseline["calmar"] > 0 else 0
        dd_delta = ((m["dd"] / baseline["dd"]) - 1) * 100 if baseline["dd"] > 0 else 0
        verdict = "WINNER" if m["calmar"] > baseline["calmar"] * 1.01 and m["dd"] <= baseline["dd"] * 1.15 else \
                  "NEUTRAL" if abs(calmar_delta) < 3 else "REJECTED"

        print(f"\n  {name}:")
        print(f"    Ret={m['ret']:.1f}%  DD={m['dd']:.1f}%  Calmar={m['calmar']:.2f}  "
              f"Sharpe={m['sharpe']:.2f}  Trades={m['trades']}  WR={m['wr']:.1f}%")
        print(f"    Calmar Δ={calmar_delta:+.1f}%  DD Δ={dd_delta:+.1f}%  → {verdict}  [{dt:.1f}s]")

        results_p1[name] = {"metrics": m, "verdict": verdict, "calmar_delta": calmar_delta}

    # ═══════════════════ Phase 2: Winner Combinations ═══════════════════
    print("\n" + "=" * 80)
    print("PHASE 2: Winner Combinations")
    print("=" * 80)

    winners = [k for k, v in results_p1.items() if v["verdict"] == "WINNER"]
    print(f"\n  Winners from Phase 1: {winners if winners else 'NONE'}")

    # Test promising combinations regardless
    combos = {
        "Stoch+MACD+DynTarget": {"stoch_enabled": True, "macd_enabled": True},
        "EMA50+Stoch": {"ema_regime": True, "ema_regime_period": 50, "stoch_enabled": True},
        "EMA50+MACD": {"ema_regime": True, "ema_regime_period": 50, "macd_enabled": True},
        "EMA50+Stoch+MACD": {"ema_regime": True, "ema_regime_period": 50,
                              "stoch_enabled": True, "macd_enabled": True},
        "Fib0.5+Stoch": {"fib_exits": True, "fib_target": 0.5, "fib_stop": 0.786,
                          "stoch_enabled": True},
        "Partial+Stoch+MACD": {"partial_exit": True, "stoch_enabled": True, "macd_enabled": True},
        "AllFeatures": {"stoch_enabled": True, "macd_enabled": True,
                        "ema_regime": True, "ema_regime_period": 50,
                        "fib_exits": True, "fib_target": 0.5, "fib_stop": 0.786},
    }

    results_p2 = {}
    for name, ov in combos.items():
        t0 = time.time()
        m = test_hypothesis(name, ov, TOP10, btc_15m, btc_4h, all_data)
        dt = time.time() - t0

        calmar_delta = ((m["calmar"] / baseline["calmar"]) - 1) * 100 if baseline["calmar"] > 0 else 0
        dd_delta = ((m["dd"] / baseline["dd"]) - 1) * 100 if baseline["dd"] > 0 else 0

        print(f"\n  {name}:")
        print(f"    Ret={m['ret']:.1f}%  DD={m['dd']:.1f}%  Calmar={m['calmar']:.2f}  "
              f"Sharpe={m['sharpe']:.2f}  Trades={m['trades']}  WR={m['wr']:.1f}%")
        print(f"    Calmar Δ={calmar_delta:+.1f}%  DD Δ={dd_delta:+.1f}%  [{dt:.1f}s]")

        results_p2[name] = {"metrics": m, "calmar_delta": calmar_delta}

    # ═══════════════════ Phase 3: min_confidence sweep ═══════════════════
    print("\n" + "=" * 80)
    print("PHASE 3: Confidence Threshold Sweep (with best combo)")
    print("=" * 80)

    # Find best combo that beats baseline
    all_tested = {**results_p1, **results_p2}
    best_name = max(all_tested, key=lambda k: all_tested[k].get("metrics", all_tested[k].get("metrics", {})).get("calmar", 0))
    print(f"\n  Best so far: {best_name}")

    # Get overrides for best
    best_ov = hypotheses.get(best_name, combos.get(best_name, {}))

    for mc in [1, 2, 3, 4]:
        ov = dict(best_ov)
        ov["min_confidence"] = mc
        t0 = time.time()
        m = test_hypothesis(f"mc={mc}", ov, TOP10, btc_15m, btc_4h, all_data)
        dt = time.time() - t0
        calmar_delta = ((m["calmar"] / baseline["calmar"]) - 1) * 100 if baseline["calmar"] > 0 else 0
        print(f"    mc={mc}: Ret={m['ret']:.1f}%  DD={m['dd']:.1f}%  Calmar={m['calmar']:.2f}  "
              f"Sharpe={m['sharpe']:.2f}  Trades={m['trades']}  Calmar Δ={calmar_delta:+.1f}%  [{dt:.1f}s]")

    # ═══════════════════ Phase 4: Leverage Sweep ═══════════════════
    print("\n" + "=" * 80)
    print("PHASE 4: MTF Leverage Sweep (with best combo)")
    print("=" * 80)

    for lev_mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        ov = dict(best_ov)
        ov["mtf_leverage_mult"] = lev_mult
        t0 = time.time()
        m = test_hypothesis(f"lev={lev_mult}x", ov, TOP10, btc_15m, btc_4h, all_data)
        dt = time.time() - t0
        calmar_delta = ((m["calmar"] / baseline["calmar"]) - 1) * 100 if baseline["calmar"] > 0 else 0
        print(f"    lev={lev_mult}x: Ret={m['ret']:.1f}%  DD={m['dd']:.1f}%  Calmar={m['calmar']:.2f}  "
              f"Sharpe={m['sharpe']:.2f}  Trades={m['trades']}  Calmar Δ={calmar_delta:+.1f}%  [{dt:.1f}s]")

    # ═══════════════════ Phase 5: Walk-Forward Quarterly ═══════════════════
    print("\n" + "=" * 80)
    print("PHASE 5: Walk-Forward Quarterly Validation (best config)")
    print("=" * 80)

    # Get best overall config
    bars_per_quarter = 96 * 91  # ~91 days
    n_bars = len(btc_15m)
    n_quarters = n_bars // bars_per_quarter

    for q in range(min(n_quarters, 4)):
        start = q * bars_per_quarter
        end = min((q + 1) * bars_per_quarter, n_bars)

        q_eq = []
        q_res = []
        for sym in TOP10:
            data = all_data[sym]
            q_data = {k: v[start:end] for k, v in data.items()}
            q_btc = btc_15m[start:end]
            q_btc4h_start = start // 16
            q_btc4h_end = end // 16
            q_btc4h = btc_4h[q_btc4h_start:q_btc4h_end]
            if len(q_data["close"]) < 500:
                continue
            try:
                r = run_v9(q_data, q_btc, q_btc4h, best_ov)
                q_eq.append(r[0])
                q_res.append(r)
            except Exception:
                continue

        if q_eq:
            qm = portfolio_metrics(q_eq, q_res)
            print(f"    Q{q+1}: Ret={qm['ret']:.1f}%  DD={qm['dd']:.1f}%  "
                  f"Calmar={qm['calmar']:.2f}  Trades={qm['trades']}")
        else:
            print(f"    Q{q+1}: No data")

    # ═══════════════════ Summary ═══════════════════
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Baseline (v8.0): Ret={baseline['ret']:.1f}%, DD={baseline['dd']:.1f}%, "
          f"Calmar={baseline['calmar']:.2f}")
    print(f"\n  Phase 1 Results:")
    for name, r in sorted(results_p1.items(), key=lambda x: x[1]["metrics"]["calmar"], reverse=True):
        m = r["metrics"]
        print(f"    {r['verdict']:8s} {name}: Calmar={m['calmar']:.2f}  "
              f"Δ={r['calmar_delta']:+.1f}%  DD={m['dd']:.1f}%  Trades={m['trades']}")

    print(f"\n  Phase 2 Combinations:")
    for name, r in sorted(results_p2.items(), key=lambda x: x[1]["metrics"]["calmar"], reverse=True):
        m = r["metrics"]
        print(f"    {name}: Calmar={m['calmar']:.2f}  Δ={r['calmar_delta']:+.1f}%  "
              f"DD={m['dd']:.1f}%  Trades={m['trades']}")


if __name__ == "__main__":
    main()
