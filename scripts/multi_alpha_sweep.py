#!/usr/bin/env python3
"""
Multi-Alpha Signal Sweep — v12.2 Baseline vs Alternative Strategies
====================================================================
Tests fundamentally different signal generators against the same
portfolio simulator to find approaches that beat v12.2 on BOTH
Win Rate (>64.2%) AND R:R (>0.98).

Alpha Sources:
  A: v12.2 Baseline (compute_signals_v9 + MTF) — current production
  B: Momentum Breakout (MA crossover + N-bar breakout + ATR trail)
  C: Hybrid Trend+Pullback (trend direction + rubberband timing)
  D: Taker Flow Divergence (taker_buy ratio reversal + volume surge)
  E: Volatility Squeeze Breakout (BB squeeze → directional expansion)
  F: Multi-Factor Alpha (alpha_engine with vol+momentum+taker filters)

All tested on same 50 symbols × 1095 days × 15m with identical
portfolio sim, costs (0.04%/side), funding, and 3 slot limit.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/multi_alpha_sweep.py
"""

import sys
import time
import json
import subprocess
import numba
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_signals_v9,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    compute_atr_pct,
)
from quantbox.strategy.alpha_engine import (
    compute_alpha_signals,
    compute_momentum_signals,
    compute_hybrid_signals,
)

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    pattern = f'{symbol}_{interval}_{days}d_*.npz'
    matches = sorted(CACHE_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def load_15m(symbol: str, days: int) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is None or not cache.exists():
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str, days: int) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is None or not cache.exists():
        return None
    d = np.load(cache)
    return d['close']


# ═══════════════════════════════════════════════════════════════════════
#  NEW SIGNAL GENERATORS (Numba compiled)
# ═══════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def compute_taker_flow_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    taker_buy: np.ndarray,
    btc_close: np.ndarray,
    # Taker flow params
    flow_lookback: int,       # period for taker ratio smoothing
    divergence_bars: int,     # bars to detect price vs flow divergence
    vol_surge_mult: float,    # volume surge threshold
    vol_lookback: int,        # volume MA period
    # BTC regime
    btc_regime_period: int,
    # Exit params
    atr_period: int,
    target_atr_mult: float,   # target = N × ATR
    stop_atr_mult: float,     # stop = N × ATR
    trail_activation: float,  # activate trail after this % of target
    trail_distance: float,    # trail distance as fraction of best PnL
    max_hold: int,
) -> np.ndarray:
    """Taker Flow Divergence Strategy.

    Alpha: When aggressive buying (taker_buy ratio rising) diverges from
    price action (price dropping), it signals institutional accumulation.
    Enter in the flow direction, not the price direction.

    Entry conditions:
      1. Taker buy ratio momentum > threshold (buying pressure increasing)
      2. Price moved opposite (divergence)
      3. Volume surge confirms institutional activity
      4. BTC regime alignment

    Exit: ATR-based target + ATR-based stop + trailing stop.
    R:R designed to be > 1.0 by construction (target_atr > stop_atr).
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute rolling taker buy ratio
    taker_ratio = np.full(n, 0.5)
    for i in range(flow_lookback, n):
        tb = 0.0
        tv = 0.0
        for j in range(i - flow_lookback, i + 1):
            tb += taker_buy[j]
            tv += volume[j]
        if tv > 0:
            taker_ratio[i] = tb / tv

    # Pre-compute taker ratio momentum (delta over divergence_bars)
    taker_mom = np.zeros(n)
    for i in range(flow_lookback + divergence_bars, n):
        taker_mom[i] = taker_ratio[i] - taker_ratio[i - divergence_bars]

    # Pre-compute volume SMA
    vol_sma = np.zeros(n)
    s = 0.0
    for i in range(n):
        s += volume[i]
        if i >= vol_lookback:
            s -= volume[i - vol_lookback]
        if i >= vol_lookback - 1:
            vol_sma[i] = s / vol_lookback

    # Pre-compute BTC SMA for regime
    btc_sma = np.full(n, np.nan)
    s2 = 0.0
    for i in range(n):
        s2 += btc_close[i]
        if i >= btc_regime_period:
            s2 -= btc_close[i - btc_regime_period]
        if i >= btc_regime_period - 1:
            btc_sma[i] = s2 / btc_regime_period

    # Pre-compute ATR for exit
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i - 1]),
                 abs(low[i] - close[i - 1]))
        if i < atr_period:
            total = 0.0
            for j in range(max(1, i - atr_period + 1), i + 1):
                total += max(high[j] - low[j],
                             abs(high[j] - close[j - 1]),
                             abs(low[j] - close[j - 1]))
            atr[i] = total / min(i, atr_period)
        else:
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr) / atr_period

    position = 0
    entry_price = 0.0
    hold_count = 0
    best_pnl = 0.0
    target = 0.0
    stop = 0.0

    warmup = max(flow_lookback + divergence_bars, vol_lookback, btc_regime_period,
                 atr_period)

    for i in range(warmup, n):
        if position == 0:
            # Price return over divergence period
            price_ret = close[i] / close[i - divergence_bars] - 1.0
            flow_delta = taker_mom[i]

            # Volume surge check
            vol_ok = vol_sma[i] > 0 and volume[i] > vol_surge_mult * vol_sma[i]

            # BTC regime
            if np.isnan(btc_sma[i]):
                continue
            is_risk_on = btc_close[i] > btc_sma[i]

            entry_signal = 0

            # LONG: Price dropped but taker buy ratio rising → accumulation
            if price_ret < -0.005 and flow_delta > 0.02 and vol_ok:
                if is_risk_on:
                    entry_signal = 1

            # SHORT: Price pumped but taker buy ratio falling → distribution
            elif price_ret > 0.005 and flow_delta < -0.02 and vol_ok:
                if not is_risk_on:
                    entry_signal = -1

            if entry_signal != 0:
                position = entry_signal
                entry_price = close[i]
                hold_count = 0
                best_pnl = 0.0
                # ATR-based targets (R:R > 1 by design)
                atr_val = atr[i]
                target = target_atr_mult * atr_val / close[i]
                stop = stop_atr_mult * atr_val / close[i]
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            # Exit conditions
            if gain >= target:
                position = 0
            elif gain < -stop:
                position = 0
            elif best_pnl >= target * trail_activation and gain < best_pnl * (1.0 - trail_distance):
                position = 0
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position

    return sig


@numba.jit(nopython=True, cache=True)
def compute_squeeze_breakout_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    btc_close: np.ndarray,
    # Squeeze params
    bb_period: int,           # Bollinger Band period
    bb_mult: float,           # BB standard deviation multiplier
    kc_period: int,           # Keltner Channel ATR period
    kc_mult: float,           # KC ATR multiplier
    squeeze_min_bars: int,    # min consecutive bars in squeeze
    vol_lookback: int,        # volume average period
    vol_mult: float,          # volume surge multiplier
    # BTC regime
    btc_regime_period: int,
    # Exit params
    atr_period: int,
    trail_atr_mult: float,    # trailing stop = N × ATR
    max_hold: int,
) -> np.ndarray:
    """Volatility Squeeze Breakout Strategy.

    Alpha: Volatility is mean-reverting. After contraction (squeeze),
    expansion follows with directional momentum. The squeeze is detected
    when Bollinger Bands are INSIDE Keltner Channels.

    Entry: On squeeze release (BB exits KC) in the directional bias.
    Exit: ATR trailing stop (ride the expansion).

    This is the opposite of rubberband: instead of fading moves,
    we ride the beginning of new moves.
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute Bollinger Band width
    bb_upper = np.zeros(n)
    bb_lower = np.zeros(n)
    bb_sma = np.zeros(n)

    for i in range(bb_period - 1, n):
        mean = 0.0
        for j in range(i - bb_period + 1, i + 1):
            mean += close[j]
        mean /= bb_period

        var = 0.0
        for j in range(i - bb_period + 1, i + 1):
            diff = close[j] - mean
            var += diff * diff
        std = (var / bb_period) ** 0.5

        bb_sma[i] = mean
        bb_upper[i] = mean + bb_mult * std
        bb_lower[i] = mean - bb_mult * std

    # Pre-compute Keltner Channels
    kc_upper = np.zeros(n)
    kc_lower = np.zeros(n)

    # KC uses EMA + ATR
    ema = np.zeros(n)
    ema[0] = close[0]
    alpha_ema = 2.0 / (kc_period + 1.0)
    for i in range(1, n):
        ema[i] = alpha_ema * close[i] + (1.0 - alpha_ema) * ema[i - 1]

    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i - 1]),
                 abs(low[i] - close[i - 1]))
        if i < kc_period:
            total = 0.0
            for j in range(max(1, i - kc_period + 1), i + 1):
                total += max(high[j] - low[j],
                             abs(high[j] - close[j - 1]),
                             abs(low[j] - close[j - 1]))
            atr[i] = total / min(i, kc_period)
        else:
            atr[i] = (atr[i - 1] * (kc_period - 1) + tr) / kc_period

    for i in range(n):
        kc_upper[i] = ema[i] + kc_mult * atr[i]
        kc_lower[i] = ema[i] - kc_mult * atr[i]

    # Pre-compute squeeze state: BB inside KC
    in_squeeze = np.zeros(n)
    for i in range(bb_period, n):
        if bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
            in_squeeze[i] = 1.0

    # Count consecutive squeeze bars
    squeeze_count = np.zeros(n)
    for i in range(1, n):
        if in_squeeze[i] > 0.5:
            squeeze_count[i] = squeeze_count[i - 1] + 1
        else:
            squeeze_count[i] = 0

    # Volume SMA
    vol_sma = np.zeros(n)
    sv = 0.0
    for i in range(n):
        sv += volume[i]
        if i >= vol_lookback:
            sv -= volume[i - vol_lookback]
        if i >= vol_lookback - 1:
            vol_sma[i] = sv / vol_lookback

    # BTC SMA
    btc_sma = np.full(n, np.nan)
    sb = 0.0
    for i in range(n):
        sb += btc_close[i]
        if i >= btc_regime_period:
            sb -= btc_close[i - btc_regime_period]
        if i >= btc_regime_period - 1:
            btc_sma[i] = sb / btc_regime_period

    # Directional momentum (which way the squeeze breaks)
    mom_period = bb_period  # use same period
    position = 0
    entry_price = 0.0
    hold_count = 0
    best_price = 0.0  # for trailing stop
    entry_atr = 0.0

    warmup = max(bb_period, kc_period, vol_lookback, btc_regime_period, atr_period)

    for i in range(warmup, n):
        if position == 0:
            # Squeeze release: was in squeeze, now out
            was_squeezing = squeeze_count[i - 1] >= squeeze_min_bars
            now_free = in_squeeze[i] < 0.5

            if was_squeezing and now_free:
                # Determine direction from momentum
                mom = close[i] / close[i - mom_period] - 1.0
                vol_ok = vol_sma[i] > 0 and volume[i] > vol_mult * vol_sma[i]

                # BTC alignment
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]

                entry_signal = 0
                if mom > 0.005 and vol_ok:
                    if is_risk_on:
                        entry_signal = 1
                elif mom < -0.005 and vol_ok:
                    if not is_risk_on:
                        entry_signal = -1

                if entry_signal != 0:
                    position = entry_signal
                    entry_price = close[i]
                    hold_count = 0
                    best_price = close[i]
                    entry_atr = atr[i]
        else:
            hold_count += 1

            # Update trailing stop reference
            if position == 1:
                if close[i] > best_price:
                    best_price = close[i]
                # Trailing stop: trail from highest price
                trail_stop = best_price - trail_atr_mult * atr[i]
                if close[i] < trail_stop:
                    position = 0
            else:
                if close[i] < best_price:
                    best_price = close[i]
                trail_stop = best_price + trail_atr_mult * atr[i]
                if close[i] > trail_stop:
                    position = 0

            # Max hold
            if hold_count >= max_hold and position != 0:
                position = 0

        sig[i] = position

    return sig


# ═══════════════════════════════════════════════════════════════════════
#  Signal Generation Functions (one per variant)
# ═══════════════════════════════════════════════════════════════════════

def gen_baseline_v12(data: dict, btc: np.ndarray, close_4h, settings) -> np.ndarray:
    """A: v12.2 Baseline — current production."""
    cfg = RubberBandConfig()
    cfg.lookback = settings.rb_lookback
    cfg.threshold = settings.rb_threshold
    cfg.target_retrace = settings.rb_target_retrace
    cfg.stop_mult = settings.rb_stop_mult
    cfg.max_hold = settings.rb_max_hold
    cfg.stop_decay = settings.rb_stop_decay
    cfg.btc_directional_period = settings.rb_btc_directional_period
    cfg.btc_regime_period = settings.rb_btc_regime_period
    cfg.asset_trend_period = settings.rb_asset_trend_period
    cfg.base_leverage = settings.rb_base_leverage
    cfg.confidence_leverage_step = settings.rb_confidence_leverage_step
    cfg.max_leverage = settings.rb_max_leverage
    cfg.rsi_period = settings.rb_rsi_period
    cfg.vol_period = settings.rb_vol_period
    cfg.stoch_enabled = settings.rb_stoch_enabled
    cfg.stoch_k_period = settings.rb_stoch_k_period
    cfg.stoch_d_period = settings.rb_stoch_d_period
    cfg.macd_enabled = settings.rb_macd_enabled
    cfg.macd_fast = settings.rb_macd_fast
    cfg.macd_slow = settings.rb_macd_slow
    cfg.macd_signal_period = settings.rb_macd_signal_period
    cfg.dynamic_target_enabled = settings.rb_dynamic_target_enabled
    cfg.target_conf3 = settings.rb_target_conf3
    cfg.target_conf4 = settings.rb_target_conf4
    cfg.target_conf5 = settings.rb_target_conf5
    cfg.min_confidence = settings.rb_min_confidence

    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    t = data['taker_buy']
    n = min(len(c), len(btc))
    c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc[:n]

    sigs = compute_signals_v9(
        c, h, l, v, t, b,
        cfg.lookback, cfg.threshold, cfg.target_retrace, cfg.stop_mult,
        cfg.max_hold, cfg.stop_decay, cfg.btc_directional_period,
        cfg.btc_regime_period, cfg.asset_trend_period,
        cfg.min_confidence, cfg.base_leverage,
        cfg.confidence_leverage_step, cfg.max_leverage,
        cfg.rsi_period, cfg.vol_period,
        cfg.dynamic_target_enabled,
        cfg.target_conf3, cfg.target_conf4, cfg.target_conf5,
        cfg.stoch_enabled, cfg.stoch_k_period, cfg.stoch_d_period,
        cfg.macd_enabled, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
        atr_period=settings.rb_atr_period,
        atr_multiplier=settings.rb_atr_multiplier,
        threshold_floor=settings.rb_threshold_floor,
        threshold_cap=settings.rb_threshold_cap,
    )

    if close_4h is not None:
        sigs = apply_mtf_confirmation(
            sigs, close_4h, settings.rb_mtf_sma_period, ratio=16)
        sigs = apply_mtf_leverage(sigs, settings.rb_mtf_leverage_mult)

    return c, sigs


def gen_momentum_breakout(data: dict, btc: np.ndarray, close_4h, settings) -> np.ndarray:
    """B: Pure Momentum Breakout — ride trends, don't fade them."""
    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    n = min(len(c), len(btc))
    c, h, l, v = c[:n], h[:n], l[:n], v[:n]

    sigs = compute_momentum_signals(
        c, h, l, v,
        fast_lookback=12,       # 3h fast MA
        slow_lookback=48,       # 12h slow MA
        breakout_lookback=32,   # 8h breakout
        atr_lookback=48,        # 12h ATR
        vol_lookback=96,        # 24h volume baseline
        vol_mult=1.3,           # require 1.3× volume
        trail_atr_mult=2.0,     # trail at 2× ATR
        max_hold=40,            # 10h max hold
    )
    return c, sigs


def gen_hybrid_trend_pullback(data: dict, btc: np.ndarray, close_4h, settings) -> np.ndarray:
    """C: Hybrid — trend direction + rubberband dip-buying."""
    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    t = data['taker_buy']
    n = min(len(c), len(btc))
    c, h, l, v, t = c[:n], h[:n], l[:n], v[:n], t[:n]

    sigs = compute_hybrid_signals(
        c, h, l, v, t,
        trend_lookback=96,       # 24h for trend
        trend_threshold=0.01,    # 1% min trend strength
        rb_lookback=8,           # same as rubberband
        rb_threshold=0.02,       # 2% dip threshold
        rb_target=0.5,           # 50% retrace (higher R:R)
        rb_stop=0.786,           # same stop
        rb_max_hold=20,          # 5h max hold
        vol_lookback=48,         # 12h volume
        vol_mult=1.2,            # 1.2× vol
        atr_lookback=48,         # 12h ATR
        atr_mult=2.5,            # adaptive threshold
        trail_activation=0.3,    # trail at 30% of target
        trail_distance=0.4,      # 40% trail distance
    )
    return c, sigs


def gen_taker_flow(data: dict, btc: np.ndarray, close_4h, settings) -> np.ndarray:
    """D: Taker Flow Divergence — accumulation/distribution detection."""
    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    t = data['taker_buy']
    n = min(len(c), len(btc))
    c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc[:n]

    sigs = compute_taker_flow_signals(
        c, h, l, v, t, b,
        flow_lookback=16,        # 4h taker ratio smoothing
        divergence_bars=8,       # 2h divergence window
        vol_surge_mult=1.5,      # require 1.5× volume
        vol_lookback=96,         # 24h baseline
        btc_regime_period=96,    # 24h BTC SMA
        atr_period=48,           # 12h ATR
        target_atr_mult=3.0,     # target = 3× ATR (high R:R)
        stop_atr_mult=1.5,       # stop = 1.5× ATR → R:R = 2.0
        trail_activation=0.4,    # trail at 40% of target
        trail_distance=0.5,      # trail at 50%
        max_hold=32,             # 8h max hold
    )
    return c, sigs


def gen_squeeze_breakout(data: dict, btc: np.ndarray, close_4h, settings) -> np.ndarray:
    """E: Volatility Squeeze Breakout — BB inside KC → ride expansion."""
    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    n = min(len(c), len(btc))
    c, h, l, v, b = c[:n], h[:n], l[:n], v[:n], btc[:n]

    sigs = compute_squeeze_breakout_signals(
        c, h, l, v, b,
        bb_period=20,            # 5h BB
        bb_mult=2.0,             # standard BB
        kc_period=20,            # 5h KC
        kc_mult=1.5,             # KC multiplier
        squeeze_min_bars=6,      # require 1.5h in squeeze
        vol_lookback=96,         # 24h volume
        vol_mult=1.2,            # modest volume requirement
        btc_regime_period=96,    # 24h BTC regime
        atr_period=48,           # 12h ATR
        trail_atr_mult=2.0,     # trail at 2× ATR
        max_hold=40,             # 10h max hold
    )
    return c, sigs


def gen_alpha_multifactor(data: dict, btc: np.ndarray, close_4h, settings) -> np.ndarray:
    """F: Multi-Factor Alpha — volume + momentum + taker + adaptive."""
    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    t = data['taker_buy']
    n = min(len(c), len(btc))
    c, h, l, v, t = c[:n], h[:n], l[:n], v[:n], t[:n]

    sigs = compute_alpha_signals(
        c, h, l, v, t,
        lookback=8,
        threshold=0.025,
        target_retrace=0.5,     # higher target (0.5 vs 0.382)
        stop_mult=0.786,
        max_hold=20,
        vol_filter=True,
        vol_lookback=48,
        vol_mult=1.3,
        momentum_filter=False,  # counter-trend by nature
        momentum_lookback=96,
        momentum_mode=1,        # counter-trend mode
        adaptive_threshold=True,
        atr_lookback=48,
        atr_mult=2.0,
        trailing_stop=True,
        trail_activation=0.3,
        trail_distance=0.4,
        taker_filter=True,
        taker_lookback=16,
        taker_threshold=0.55,
        breakout_mode=False,
        breakout_lookback=32,
    )
    return c, sigs


# ═══════════════════════════════════════════════════════════════════════
#  Portfolio Simulator (standard, from A/B infrastructure)
# ═══════════════════════════════════════════════════════════════════════

def simulate_portfolio(
    all_close: dict[str, np.ndarray],
    all_signals: dict[str, np.ndarray],
    initial_capital: float = 2095.0,
    max_slots: int = 3,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
) -> dict:
    """Standard portfolio sim — same as all previous A/B tests."""
    symbols = sorted(all_close.keys())
    n_bars = min(len(all_close[s]) for s in symbols)

    balance = initial_capital
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance

    positions: dict[str, dict] = {}
    trades: list[dict] = []
    peak = balance
    max_dd = 0.0

    for i in range(1, n_bars):
        # Update existing positions
        symbols_to_close = []
        for sym, pos in list(positions.items()):
            c = all_close[sym]
            s = all_signals[sym]

            current_price = c[i]
            prev_price = c[i - 1]
            bar_ret = current_price / prev_price - 1.0

            direction = 1.0 if pos['signal'] > 0 else -1.0
            leverage = abs(pos['signal'])

            pnl = direction * leverage * bar_ret * pos['notional']
            pnl -= abs(leverage) * funding_per_bar * pos['notional']
            pos['unrealized'] += pnl

            pos['bars_held'] += 1

            # Check signal exit
            current_sig = s[i] if i < len(s) else 0.0
            prev_sig = s[i - 1] if i - 1 < len(s) else 0.0

            if current_sig == 0.0 and prev_sig != 0.0:
                symbols_to_close.append(sym)
            elif current_sig != 0.0 and np.sign(current_sig) != np.sign(pos['signal']):
                symbols_to_close.append(sym)

        # Close positions
        for sym in set(symbols_to_close):
            if sym not in positions:
                continue
            pos = positions.pop(sym)
            leverage = abs(pos['signal'])
            close_cost = leverage * cost_per_side * pos['notional']
            net_pnl = pos['unrealized'] - close_cost
            balance += net_pnl

            pnl_pct = net_pnl / pos['notional'] * 100
            trades.append({
                'symbol': sym, 'signal': pos['signal'],
                'entry_bar': pos['entry_bar'], 'exit_bar': i,
                'bars_held': pos['bars_held'],
                'notional': pos['notional'],
                'pnl_usd': net_pnl, 'pnl_pct': pnl_pct,
            })

        # New entries
        n_active = len(positions)
        if n_active < max_slots:
            candidates = []
            for sym in symbols:
                if sym in positions:
                    continue
                s = all_signals[sym]
                if i >= len(s):
                    continue
                current_sig = s[i]
                prev_sig = s[i - 1] if i > 0 else 0.0
                if current_sig != 0.0 and prev_sig == 0.0:
                    candidates.append((sym, current_sig))

            candidates.sort(key=lambda x: abs(x[1]), reverse=True)

            for sym, sig in candidates:
                if len(positions) >= max_slots:
                    break
                per_slot = initial_capital / max(1, len(positions) + 1)
                per_slot = min(per_slot, initial_capital * 0.5)
                if per_slot < 10.0:
                    continue

                leverage = abs(sig)
                notional = per_slot
                entry_cost = leverage * cost_per_side * notional
                balance -= entry_cost

                positions[sym] = {
                    'signal': sig,
                    'entry_bar': i,
                    'bars_held': 0,
                    'notional': notional,
                    'unrealized': 0.0,
                }

        # Equity curve
        total_unrealized = sum(p['unrealized'] for p in positions.values())
        equity_curve[i] = balance + total_unrealized

        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Stats
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    n_wins = len(wins)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0.0

    total_return = (equity_curve[-1] / initial_capital - 1.0) * 100
    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = sum(abs(t['pnl_usd']) for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    eq_rets = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    if len(eq_rets) > 0 and np.std(eq_rets) > 0:
        sharpe = np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)
    else:
        sharpe = 0.0

    calmar = total_return / (max_dd * 100) if max_dd > 0 else 0.0

    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 1
    actual_rr = avg_win / avg_loss if avg_loss > 0 else 999

    avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
    avg_bars = np.mean([t['bars_held'] for t in trades]) if trades else 0

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'total_return_pct': round(total_return, 1),
        'final_balance': round(equity_curve[-1], 2),
        'initial_capital': initial_capital,
        'max_dd_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 1),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'avg_pnl_pct': round(avg_pnl, 3),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'actual_rr': round(actual_rr, 2),
        'avg_bars_held': round(avg_bars, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    commit = _get_git_commit()
    settings = get_settings()

    print("=" * 90)
    print("  MULTI-ALPHA SIGNAL SWEEP")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Target: Beat v12.2 on BOTH WR (>64.2%) AND R:R (>0.98)")
    print("=" * 90)

    SYMBOLS_50 = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
        'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
        'ICPUSDT', 'FILUSDT',
        'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT', 'BCHUSDT',
        'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
        'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT',
        'EGLDUSDT', 'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT',
        'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'LDOUSDT', 'COMPUSDT',
        'MKRUSDT', 'SNXUSDT', 'VETUSDT', 'EOSUSDT', 'APEUSDT',
    ]

    DAYS = 1095
    CAPITAL = 2095.0
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32

    # Variant definitions: (name, desc, generator_func)
    variants = [
        ('A-v12.2',    'Baseline v12.2 (rubberband+MTF)',     gen_baseline_v12),
        ('B-Momentum', 'Pure Momentum Breakout (MA+ATR)',     gen_momentum_breakout),
        ('C-Hybrid',   'Trend Direction + RB Pullback',       gen_hybrid_trend_pullback),
        ('D-TakerFlow','Taker Flow Divergence',               gen_taker_flow),
        ('E-Squeeze',  'Volatility Squeeze Breakout',         gen_squeeze_breakout),
        ('F-Alpha',    'Multi-Factor Alpha (vol+mom+taker)',  gen_alpha_multifactor),
    ]

    # Load data
    print(f"\n📊 Loading 50-symbol data ({DAYS}d)...")
    btc_data = load_15m('BTCUSDT', DAYS)
    if btc_data is None:
        print("ERROR: BTC 15m data not found")
        sys.exit(1)
    btc_close = btc_data['close']

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym, DAYS)
        if d is None:
            print(f"  SKIP {sym}")
            continue
        h4 = load_4h(sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    # Run each variant
    results: dict[str, dict] = {}

    for name, desc, gen_func in variants:
        print(f"\n{'─' * 70}")
        print(f"  ⚡ {name}: {desc}")
        print(f"{'─' * 70}")

        t0 = time.time()

        # Generate signals per symbol
        sym_close: dict[str, np.ndarray] = {}
        sym_sigs: dict[str, np.ndarray] = {}

        for sym in available:
            data = all_data[sym]
            close_4h = all_4h.get(sym)
            try:
                c, sigs = gen_func(data, btc_close, close_4h, settings)
                sym_close[sym] = c
                sym_sigs[sym] = sigs
            except Exception as e:
                print(f"    WARN {sym}: {e}")
                continue

        if not sym_close:
            print("    No symbols generated signals")
            continue

        # Left-pad to uniform length
        max_bars = max(len(v) for v in sym_close.values())
        for sym in list(sym_close.keys()):
            for arr_dict, fill_val in [(sym_close, 'price'), (sym_sigs, 'zero')]:
                arr = arr_dict[sym]
                if len(arr) < max_bars:
                    pad_n = max_bars - len(arr)
                    fv = arr[0] if fill_val == 'price' else 0.0
                    arr_dict[sym] = np.concatenate([np.full(pad_n, fv), arr])

        # Run portfolio sim
        result = simulate_portfolio(
            sym_close, sym_sigs,
            initial_capital=CAPITAL, max_slots=3,
            cost_per_side=COST, funding_per_bar=FUNDING,
        )
        elapsed = time.time() - t0

        n_days = len(result['equity_curve']) / 96
        trades_per_day = round(result['n_trades'] / n_days, 2)

        summary = {
            'description': desc,
            'total_return_pct': result['total_return_pct'],
            'final_balance': result['final_balance'],
            'max_dd_pct': result['max_dd_pct'],
            'sharpe': result['sharpe'],
            'calmar': result['calmar'],
            'n_trades': result['n_trades'],
            'trades_per_day': trades_per_day,
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'avg_pnl_pct': result['avg_pnl_pct'],
            'avg_win_pct': result['avg_win_pct'],
            'avg_loss_pct': result['avg_loss_pct'],
            'actual_rr': result['actual_rr'],
            'avg_bars_held': result['avg_bars_held'],
            'elapsed': round(elapsed, 2),
        }
        results[name] = summary

        # Print delta markers for WR and R:R
        wr_tag = '✓' if result['win_rate'] > 64.2 else '✗'
        rr_tag = '✓' if result['actual_rr'] > 0.98 else '✗'

        print(f"  Return: {result['total_return_pct']:+,.1f}%  Sharpe: {result['sharpe']:.2f}  "
              f"DD: {result['max_dd_pct']:.1f}%  Calmar: {result['calmar']:.1f}")
        print(f"  Trades: {result['n_trades']} ({trades_per_day}/day)  "
              f"Bars: {result['avg_bars_held']:.1f}  "
              f"WR: {result['win_rate']:.1f}% [{wr_tag}]  PF: {result['profit_factor']:.2f}  "
              f"R:R: {result['actual_rr']:.2f} [{rr_tag}]  [{elapsed:.1f}s]")

    # ══════════════════════════════════════════════════════════════
    #   RESULTS TABLE
    # ══════════════════════════════════════════════════════════════
    names = list(results.keys())
    base = results.get('A-v12.2', {})

    print(f"\n\n{'=' * 100}")
    print("  MULTI-ALPHA SWEEP — RESULTS COMPARISON")
    print(f"  Target: WR > 64.2% AND R:R > 0.98 (both must improve)")
    print(f"{'=' * 100}\n")

    col_w = 12
    print(f"  {'Metric':<20}", end='')
    for n in names:
        print(f"  {n:>{col_w}}", end='')
    print()
    print("  " + "-" * (20 + (col_w + 2) * len(names)))

    rows = [
        ('Total Return', 'total_return_pct', '{:+,.1f}%'),
        ('Sharpe', 'sharpe', '{:.2f}'),
        ('Calmar', 'calmar', '{:.1f}'),
        ('Max Drawdown', 'max_dd_pct', '{:.1f}%'),
        ('', None, None),
        ('Trades', 'n_trades', '{:,d}'),
        ('Trades/Day', 'trades_per_day', '{:.2f}'),
        ('Avg Bars Held', 'avg_bars_held', '{:.1f}'),
        ('', None, None),
        ('Win Rate ★', 'win_rate', '{:.1f}%'),
        ('Avg Win %', 'avg_win_pct', '{:+.3f}%'),
        ('Avg Loss %', 'avg_loss_pct', '{:.3f}%'),
        ('R:R ★', 'actual_rr', '{:.2f}'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Avg PnL/Trade', 'avg_pnl_pct', '{:+.3f}%'),
    ]

    for label, key, fmt in rows:
        if key is None:
            print()
            continue
        print(f"  {label:<20}", end='')
        for n in names:
            val = results[n][key]
            s = fmt.format(val)
            # Mark improvements on target metrics
            if key == 'win_rate' and val > 64.2 and n != 'A-v12.2':
                s += ' ✓'
            elif key == 'actual_rr' and val > 0.98 and n != 'A-v12.2':
                s += ' ✓'
            print(f"  {s:>{col_w + 2}}", end='')
        print()

    # Dual improvement check
    print(f"\n{'─' * 100}")
    print("  DUAL-IMPROVEMENT CHECK (WR > 64.2% AND R:R > 0.98)")
    print(f"{'─' * 100}\n")
    any_winner = False
    for n, r in results.items():
        wr_ok = r['win_rate'] > 64.2
        rr_ok = r['actual_rr'] > 0.98
        both = wr_ok and rr_ok
        tag = '★ BOTH IMPROVED' if both and n != 'A-v12.2' else ''
        wr_s = f"WR={r['win_rate']:.1f}% {'✓' if wr_ok else '✗'}"
        rr_s = f"R:R={r['actual_rr']:.2f} {'✓' if rr_ok else '✗'}"
        print(f"  {n:<15} {wr_s:>15}  {rr_s:>12}  {tag}")
        if both and n != 'A-v12.2':
            any_winner = True

    if not any_winner:
        print(f"\n  → No variant achieves BOTH improvements simultaneously.")
        print(f"  → This suggests the WR/R:R tradeoff is fundamental to mean-reversion strategies.")
        print(f"  → Next steps: try ensemble (combine best R:R + best WR signals),")
        print(f"    or fundamentally different feature engineering (microstructure, cross-asset).")
    else:
        print(f"\n  → Winner found! See variant with ★ above.")

    # Save
    output_path = Path(__file__).parent.parent / 'logs' / 'multi_alpha_sweep_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
