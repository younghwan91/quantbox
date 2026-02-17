"""
Regime-Switching A/B Test — True Dual-Algorithm Entry

Thesis: "Enter only when sure, in BOTH market regimes."
  - BTC Downtrend → Sympathy Pullback: alt oversold WITH BTC → snaps back
  - BTC Uptrend  → Classic Pullback: alt dips in its OWN uptrend → buy the dip
  - Each algo is high-conviction in its regime → doubled opportunities

This is fundamentally different from "Either-Or" (loose OR filter).
Regime switching runs DIFFERENT algorithms in DIFFERENT market states.

Test Plan:
  Phase 1: Regime detection methods (SMA, EMA, Return — sweep periods)
  Phase 2: Head-to-head: Regime-Switch vs BTC-only vs Either-Or vs raw
  Phase 3: Per-regime param optimization (different trend_period per regime?)
  Phase 4: Regime transition handling (gap vs immediate switch)
  Phase 5: Leverage sweep on winner
"""

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import numba

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantbox.backtest.vectorized_backtest import _backtest_core
from quantbox.strategy.rubberband import (
    compute_signals,
    compute_signals_with_btc_directional,
    apply_confidence_leverage,
)


# ═══════════════════════════════════════════════════════════════
#  NUMBA SIGNAL FUNCTIONS — REGIME SWITCHING
# ═══════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    n = len(close)
    sma = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += close[i]
        if i >= period:
            s -= close[i - period]
        if i >= period - 1:
            sma[i] = s / period
    return sma


@numba.jit(nopython=True, cache=True)
def signals_regime_switch(
    close: np.ndarray,
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
) -> np.ndarray:
    """Regime-switching entry: different algo per BTC regime.

    BTC Regime Detection: BTC price vs SMA(btc_regime_period)
      - BTC < SMA → "Risk-Off" regime → Use SYMPATHY PULLBACK
        (Long when alt dips AND BTC is falling short-term)
      - BTC > SMA → "Risk-On" regime → Use CLASSIC PULLBACK
        (Long when alt dips AND alt's own trend is UP)

    This maximizes conviction: each algorithm works in its home regime.
    Combined, they cover both bull and bear markets.

    Long conditions:
      - Risk-Off:  ret < -threshold AND btc_mom < 0 (BTC falling → sympathy)
      - Risk-On:   ret < -threshold AND close > asset_SMA (dip in uptrend)

    Short conditions:
      - Risk-Off:  ret > threshold AND btc_mom > 0 (BTC rising → euphoria)
      - Risk-On:   ret > threshold AND close < asset_SMA (pop in downtrend)
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute BTC SMA for regime detection
    btc_sma = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += btc_close[i]
        if i >= btc_regime_period:
            s -= btc_close[i - btc_regime_period]
        if i >= btc_regime_period - 1:
            btc_sma[i] = s / btc_regime_period

    # Pre-compute asset SMA for classic pullback (risk-on)
    asset_sma = np.full(n, np.nan)
    s2 = 0.0
    for i in range(n):
        s2 += close[i]
        if i >= asset_trend_period:
            s2 -= close[i - asset_trend_period]
        if i >= asset_trend_period - 1:
            asset_sma[i] = s2 / asset_trend_period

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_mom_period, btc_regime_period, asset_trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                # Determine BTC regime
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]  # BTC above SMA = bull

                if is_risk_on:
                    # ═══ RISK-ON REGIME: Classic Pullback ═══
                    # BTC is bullish → alt dips are likely just pullbacks
                    # Buy if alt's own trend is UP (dip in uptrend)
                    if np.isnan(asset_sma[i]):
                        continue
                    is_uptrend = close[i] > asset_sma[i]

                    if ret < -threshold and is_uptrend:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and not is_uptrend:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                else:
                    # ═══ RISK-OFF REGIME: Sympathy Pullback ═══
                    # BTC is bearish → use BTC directional (proven v4.4)
                    btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                    if ret < -threshold and btc_mom < 0:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and btc_mom > 0:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
        else:
            # ═══ EXIT LOGIC (same regardless of regime) ═══
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * target_retrace

            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_regime_switch_independent_params(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    btc_regime_period: int,
    # Risk-On params (classic pullback)
    ro_trend_period: int,
    ro_target: float,
    ro_stop: float,
    # Risk-Off params (sympathy pullback) — can differ
    rf_target: float,
    rf_stop: float,
) -> np.ndarray:
    """Regime-switch with independent stop/target per regime.

    Allows different risk profiles per regime:
      - Risk-On (BTC bullish): maybe tighter stop, wider target
      - Risk-Off (BTC bearish): current v4.4 params
    """
    n = len(close)
    sig = np.zeros(n)

    btc_sma = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += btc_close[i]
        if i >= btc_regime_period:
            s -= btc_close[i - btc_regime_period]
        if i >= btc_regime_period - 1:
            btc_sma[i] = s / btc_regime_period

    asset_sma = np.full(n, np.nan)
    s2 = 0.0
    for i in range(n):
        s2 += close[i]
        if i >= ro_trend_period:
            s2 -= close[i - ro_trend_period]
        if i >= ro_trend_period - 1:
            asset_sma[i] = s2 / ro_trend_period

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    entry_regime = 0  # 0=none, 1=risk-on, 2=risk-off
    start_bar = max(lookback, btc_mom_period, btc_regime_period, ro_trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]

                if is_risk_on:
                    if np.isnan(asset_sma[i]):
                        continue
                    is_uptrend = close[i] > asset_sma[i]
                    if ret < -threshold and is_uptrend:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                        entry_regime = 1
                    elif ret > threshold and not is_uptrend:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                        entry_regime = 1
                else:
                    btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                    if ret < -threshold and btc_mom < 0:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                        entry_regime = 2
                    elif ret > threshold and btc_mom > 0:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                        entry_regime = 2
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain

            # Use regime-specific stop/target
            if entry_regime == 1:
                cur_target = ro_target
                cur_stop = ro_stop
            else:
                cur_target = rf_target
                cur_stop = rf_stop

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * cur_stop * time_factor
            target = move_size * cur_target

            if gain >= target:
                position = 0
                entry_regime = 0
            elif gain < -effective_stop:
                position = 0
                entry_regime = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                entry_regime = 0
            elif hold_count >= max_hold:
                position = 0
                entry_regime = 0

        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_regime_switch_ema(
    close: np.ndarray,
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
) -> np.ndarray:
    """Same regime-switch logic but using EMA for both BTC regime and asset trend."""
    n = len(close)
    sig = np.zeros(n)

    # BTC EMA for regime
    btc_ema = np.empty(n)
    btc_ema[0] = btc_close[0]
    alpha_btc = 2.0 / (btc_regime_period + 1.0)
    for i in range(1, n):
        btc_ema[i] = alpha_btc * btc_close[i] + (1.0 - alpha_btc) * btc_ema[i - 1]

    # Asset EMA for trend
    asset_ema = np.empty(n)
    asset_ema[0] = close[0]
    alpha_asset = 2.0 / (asset_trend_period + 1.0)
    for i in range(1, n):
        asset_ema[i] = alpha_asset * close[i] + (1.0 - alpha_asset) * asset_ema[i - 1]

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_mom_period, btc_regime_period, asset_trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                is_risk_on = btc_close[i] > btc_ema[i]

                if is_risk_on:
                    is_uptrend = close[i] > asset_ema[i]
                    if ret < -threshold and is_uptrend:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and not is_uptrend:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                else:
                    btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                    if ret < -threshold and btc_mom < 0:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and btc_mom > 0:
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
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * target_retrace
            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def signals_regime_switch_return(
    close: np.ndarray,
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
) -> np.ndarray:
    """Regime-switch using return-based detection for both BTC and asset."""
    n = len(close)
    sig = np.zeros(n)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_mom_period, btc_regime_period, asset_trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                # Return-based regime: BTC N-bar return > 0 = risk-on
                btc_regime_ret = btc_close[i] / btc_close[i - btc_regime_period] - 1.0
                is_risk_on = btc_regime_ret > 0

                if is_risk_on:
                    # Classic pullback: asset's own trend
                    asset_trend = close[i] / close[i - asset_trend_period] - 1.0
                    is_uptrend = asset_trend > 0

                    if ret < -threshold and is_uptrend:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and not is_uptrend:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                else:
                    # Sympathy pullback
                    btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                    if ret < -threshold and btc_mom < 0:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and btc_mom > 0:
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
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * target_retrace
            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position
    return sig


# ═══════════════════════════════════════════════════════════════
#  EITHER-OR (imported from pullback_ab_test for comparison)
# ═══════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def signals_either_or(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    trend_period: int,
) -> np.ndarray:
    """Either-or: Asset trend OR BTC directional (loose OR filter).

    Long: dip + (asset > SMA OR BTC falling)
    Short: pop + (asset < SMA OR BTC rising)
    """
    n = len(close)
    sig = np.zeros(n)

    trend_line = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += close[i]
        if i >= trend_period:
            s -= close[i - trend_period]
        if i >= trend_period - 1:
            trend_line[i] = s / trend_period

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_mom_period, trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                if np.isnan(trend_line[i]):
                    continue
                is_uptrend = close[i] > trend_line[i]

                if ret < -threshold and (is_uptrend or btc_mom < 0):
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold and (not is_uptrend or btc_mom > 0):
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
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * target_retrace
            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position
    return sig


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING & PORTFOLIO RUNNER
# ═══════════════════════════════════════════════════════════════

COST = 0.0004
FUND = 0.01 / 100 / 32
TOP10 = [
    'FETUSDT', 'FILUSDT', 'ICPUSDT', 'HBARUSDT', 'DOTUSDT',
    'OPUSDT', 'SHIBUSDT', 'DOGEUSDT', 'LTCUSDT', 'ATOMUSDT',
]
N_BARS = 35040

cache_dir = Path('data/cache_extended')
files = {
    f.name.split('_15m_')[0]: f
    for f in sorted(cache_dir.glob('*_15m_365d.pkl.gz'))
}


def load_data():
    """Load all symbols + BTC close prices."""
    with gzip.open(files['BTCUSDT'], 'rb') as fh:
        btc_df = pickle.load(fh)
    btc_close = btc_df['Close'].values[:N_BARS].astype(np.float64)

    data = {}
    for sym in TOP10:
        with gzip.open(files[sym], 'rb') as fh:
            df = pickle.load(fh)
        data[sym] = df['Close'].values[:N_BARS].astype(np.float64)

    return btc_close, data


def run_portfolio(data, btc_close, signal_func, leverage=1.5, label=''):
    """Run signal_func on Top-10 portfolio, return metrics."""
    port_eq = np.ones(N_BARS)
    n = len(TOP10)
    all_rets = []
    all_trades = 0

    for sym in TOP10:
        close = data[sym]
        sigs = signal_func(close, btc_close)

        if leverage > 1.0:
            lev_sigs = apply_confidence_leverage(
                close, sigs, 8, 0.045, leverage, 0.0, leverage,
            )
        else:
            lev_sigs = sigs

        eq, _, _, _, _, n_trades, _, _, _, trade_rets = _backtest_core(
            close, lev_sigs, COST, FUND,
        )
        port_eq *= eq ** (1.0 / n)
        all_rets.extend(trade_rets.tolist())
        all_trades += n_trades

    ret = (port_eq[-1] - 1) * 100
    pk = np.maximum.accumulate(port_eq)
    dd = np.max((pk - port_eq) / pk) * 100
    cal = ret / dd if dd > 0.01 else 999
    rs = np.diff(port_eq) / port_eq[:-1]
    sh = (np.mean(rs) / np.std(rs) * np.sqrt(96 * 365.25)
          if np.std(rs) > 1e-12 else 0)

    winners = [x for x in all_rets if x > 0]
    losers = [x for x in all_rets if x < 0]
    rr = (np.mean(winners) / abs(np.mean(losers))
          if losers and winners else 0)
    wr = len(winners) / len(all_rets) * 100 if all_rets else 0

    bar90 = 8640
    pos = sum(
        1 for s in range(0, len(port_eq) - bar90 + 1, bar90 // 4)
        if port_eq[min(s + bar90 - 1, len(port_eq) - 1)] / port_eq[s] > 1
    )
    total_w = len(range(0, len(port_eq) - bar90 + 1, bar90 // 4))

    print(
        f'{label:<60s} Ret {ret:+8.1f}%  DD {dd:5.1f}%  '
        f'Cal {cal:6.2f}  Sh {sh:5.2f}  WR {wr:5.1f}%  '
        f'R:R {rr:.2f}  T {all_trades:4d}  Roll {pos}/{total_w}'
    )
    return cal, ret, dd, sh, wr, rr, all_trades


def run_portfolio_detailed(data, btc_close, signal_func, leverage=1.5, label=''):
    """Same as run_portfolio but also returns per-regime trade counts."""
    port_eq = np.ones(N_BARS)
    n = len(TOP10)
    all_rets = []
    all_trades = 0

    for sym in TOP10:
        close = data[sym]
        sigs = signal_func(close, btc_close)

        if leverage > 1.0:
            lev_sigs = apply_confidence_leverage(
                close, sigs, 8, 0.045, leverage, 0.0, leverage,
            )
        else:
            lev_sigs = sigs

        eq, _, _, _, _, n_trades, _, _, _, trade_rets = _backtest_core(
            close, lev_sigs, COST, FUND,
        )
        port_eq *= eq ** (1.0 / n)
        all_rets.extend(trade_rets.tolist())
        all_trades += n_trades

    ret = (port_eq[-1] - 1) * 100
    pk = np.maximum.accumulate(port_eq)
    dd = np.max((pk - port_eq) / pk) * 100
    cal = ret / dd if dd > 0.01 else 999

    return port_eq, cal, ret, dd, all_trades, all_rets


# ═══════════════════════════════════════════════════════════════
#  MAIN — COMPREHENSIVE REGIME SWITCH A/B
# ═══════════════════════════════════════════════════════════════

def main():
    print('Loading data...')
    btc_close, data = load_data()
    print(f'Loaded {len(data)} symbols, {N_BARS} bars each.\n')

    # v4.4 base params
    LB, TH, TGT, STOP, MH, SD = 8, 0.045, 0.4, 0.9, 10, 0.0
    BTC_PER = 32
    LEV = 1.5

    # ──────────────────────────────────────────────
    # PHASE 0: BTC Regime Distribution (how much time in each regime?)
    # ──────────────────────────────────────────────
    print('=' * 100)
    print('PHASE 0: BTC Regime Distribution')
    print('=' * 100)

    for period in [48, 96, 192, 288, 384]:
        btc_sma = compute_sma(btc_close, period)
        valid_mask = ~np.isnan(btc_sma)
        risk_on = np.sum(btc_close[valid_mask] > btc_sma[valid_mask])
        total = np.sum(valid_mask)
        pct_on = risk_on / total * 100
        hours = period * 15 / 60
        print(f'  BTC SMA({period:3d}) = {hours:4.0f}h:  '
              f'Risk-On {pct_on:5.1f}%  Risk-Off {100-pct_on:5.1f}%  '
              f'({risk_on}/{total} bars)')

    # ──────────────────────────────────────────────
    # PHASE 1: Regime Detection Method + Period Sweep
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 1: Regime-Switch SMA — BTC Regime Period Sweep (asset_trend=96)')
    print('=' * 100)

    for btc_regime_p in [48, 96, 192, 288, 384]:
        hours = btc_regime_p * 15 / 60

        def mk(brp):
            def f(close, btc_close):
                return signals_regime_switch(
                    close, btc_close, LB, TH, TGT, STOP, MH, SD,
                    BTC_PER, brp, 96,
                )
            return f

        run_portfolio(
            data, btc_close, mk(btc_regime_p), LEV,
            f'RegimeSwitch SMA btc_regime={btc_regime_p}({hours:.0f}h) asset=96',
        )

    # ──────────────────────────────────────────────
    # PHASE 1B: Asset Trend Period Sweep (with best BTC regime period)
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 1B: Regime-Switch SMA — Asset Trend Period Sweep (btc_regime=192)')
    print('=' * 100)

    for asset_p in [48, 64, 96, 128, 192]:
        hours = asset_p * 15 / 60

        def mk(ap):
            def f(close, btc_close):
                return signals_regime_switch(
                    close, btc_close, LB, TH, TGT, STOP, MH, SD,
                    BTC_PER, 192, ap,
                )
            return f

        run_portfolio(
            data, btc_close, mk(asset_p), LEV,
            f'RegimeSwitch btc=192 asset_trend={asset_p}({hours:.0f}h)',
        )

    # ──────────────────────────────────────────────
    # PHASE 2: Regime Detection Type (SMA vs EMA vs Return)
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 2: Regime Detection Type Comparison (btc=192, asset=96)')
    print('=' * 100)

    def sig_regime_sma(close, btc_close):
        return signals_regime_switch(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 192, 96,
        )
    def sig_regime_ema(close, btc_close):
        return signals_regime_switch_ema(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 192, 96,
        )
    def sig_regime_return(close, btc_close):
        return signals_regime_switch_return(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 192, 96,
        )

    run_portfolio(data, btc_close, sig_regime_sma, LEV, 'RegimeSwitch SMA(192/96)')
    run_portfolio(data, btc_close, sig_regime_ema, LEV, 'RegimeSwitch EMA(192/96)')
    run_portfolio(data, btc_close, sig_regime_return, LEV, 'RegimeSwitch Return(192/96)')

    # ──────────────────────────────────────────────
    # PHASE 3: Head-to-Head — Regime-Switch vs competitors
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 3: Head-to-Head Comparison (all at 1.5x)')
    print('=' * 100)

    def sig_raw(close, btc_close):
        return compute_signals(close, LB, TH, TGT, STOP, MH, SD)

    def sig_btc_only(close, btc_close):
        return compute_signals_with_btc_directional(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER,
        )

    def sig_either_or(close, btc_close):
        return signals_either_or(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 96,
        )

    # Best regime periods from Phase 1 sweep
    best_btc_regimes = [96, 192, 288]
    best_configs = []

    run_portfolio(data, btc_close, sig_raw, LEV,
                  'A. Raw RB (no filter)')
    run_portfolio(data, btc_close, sig_btc_only, LEV,
                  'B. BTC Directional (v4.4)')
    run_portfolio(data, btc_close, sig_either_or, LEV,
                  'C. Either-Or SMA(96)')

    for brp in best_btc_regimes:
        for ap in [96, 192]:
            def mk(b, a):
                def f(close, btc_close):
                    return signals_regime_switch(
                        close, btc_close, LB, TH, TGT, STOP, MH, SD,
                        BTC_PER, b, a,
                    )
                return f
            label = f'D. RegimeSwitch SMA btc={brp} asset={ap}'
            cal, ret, dd, sh, wr, rr, trades = run_portfolio(
                data, btc_close, mk(brp, ap), LEV, label,
            )
            best_configs.append((cal, brp, ap, label))

    # ──────────────────────────────────────────────
    # PHASE 4: Per-regime param optimization
    # Can risk-on regime use different stop/target than risk-off?
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 4: Per-Regime Stop/Target Optimization (btc=192, asset=96)')
    print('=' * 100)

    # Base: both regimes use same params (TGT=0.4, STOP=0.9)
    base_label = 'Same params both (tgt=0.4, stop=0.9)'

    def sig_same(close, btc_close):
        return signals_regime_switch_independent_params(
            close, btc_close, LB, TH, TGT, STOP, MH, SD,
            BTC_PER, 192, 96,
            ro_target=0.4, ro_stop=0.9,
            rf_target=0.4, rf_stop=0.9,
        )
    run_portfolio(data, btc_close, sig_same, LEV, base_label)

    # Test variations: risk-on might need different params
    param_combos = [
        # (ro_target, ro_stop, rf_target, rf_stop, description)
        (0.3, 0.9, 0.4, 0.9, 'RiskOn: tighter target(0.3)'),
        (0.5, 0.9, 0.4, 0.9, 'RiskOn: wider target(0.5)'),
        (0.4, 0.7, 0.4, 0.9, 'RiskOn: tighter stop(0.7)'),
        (0.4, 1.0, 0.4, 0.9, 'RiskOn: wider stop(1.0)'),
        (0.3, 0.7, 0.4, 0.9, 'RiskOn: aggressive(tgt=0.3,stp=0.7)'),
        (0.5, 1.0, 0.4, 0.9, 'RiskOn: conservative(tgt=0.5,stp=1.0)'),
        (0.4, 0.9, 0.3, 0.9, 'RiskOff: tighter target(0.3)'),
        (0.4, 0.9, 0.5, 0.9, 'RiskOff: wider target(0.5)'),
        (0.4, 0.9, 0.4, 0.7, 'RiskOff: tighter stop(0.7)'),
        (0.4, 0.9, 0.4, 1.0, 'RiskOff: wider stop(1.0)'),
    ]

    for ro_tgt, ro_stp, rf_tgt, rf_stp, desc in param_combos:
        def mk(rot, ros, rft, rfs):
            def f(close, btc_close):
                return signals_regime_switch_independent_params(
                    close, btc_close, LB, TH, TGT, STOP, MH, SD,
                    BTC_PER, 192, 96,
                    ro_target=rot, ro_stop=ros,
                    rf_target=rft, rf_stop=rfs,
                )
            return f
        run_portfolio(data, btc_close, mk(ro_tgt, ro_stp, rf_tgt, rf_stp), LEV, desc)

    # ──────────────────────────────────────────────
    # PHASE 5: Leverage Sweep on top configs
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 5: Leverage Sweep — Best Regime-Switch configs')
    print('=' * 100)

    # BTC-only baseline
    for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
        run_portfolio(data, btc_close, sig_btc_only, lev,
                      f'BTC Directional (v4.4) lev={lev:.1f}x')

    print()  # separator

    # Either-Or baseline
    for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
        run_portfolio(data, btc_close, sig_either_or, lev,
                      f'Either-Or SMA(96) lev={lev:.1f}x')

    print()  # separator

    # Regime Switch best configs
    for brp in [192]:
        for ap in [96]:
            for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
                def mk(b, a):
                    def f(close, btc_close):
                        return signals_regime_switch(
                            close, btc_close, LB, TH, TGT, STOP, MH, SD,
                            BTC_PER, b, a,
                        )
                    return f
                run_portfolio(
                    data, btc_close, mk(brp, ap), lev,
                    f'RegimeSwitch SMA({brp}/{ap}) lev={lev:.1f}x',
                )

    # ──────────────────────────────────────────────
    # PHASE 6: Regime-specific trade analysis
    # Count how many trades come from each regime
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 6: Per-Regime Trade Distribution (btc_regime=192, asset=96)')
    print('=' * 100)

    btc_sma_192 = compute_sma(btc_close, 192)

    for sym in TOP10:
        close = data[sym]
        sigs = signals_regime_switch(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 192, 96,
        )

        # Count entries per regime
        risk_on_entries = 0
        risk_off_entries = 0
        for i in range(1, len(sigs)):
            if sigs[i] != 0 and sigs[i - 1] == 0:
                if not np.isnan(btc_sma_192[i]) and btc_close[i] > btc_sma_192[i]:
                    risk_on_entries += 1
                else:
                    risk_off_entries += 1

        total = risk_on_entries + risk_off_entries
        pct_on = risk_on_entries / total * 100 if total > 0 else 0
        print(f'  {sym:<12s}  Total {total:3d}  '
              f'Risk-On(pullback) {risk_on_entries:3d} ({pct_on:4.1f}%)  '
              f'Risk-Off(sympathy) {risk_off_entries:3d} ({100-pct_on:4.1f}%)')

    print('\nDone.')


if __name__ == '__main__':
    main()
