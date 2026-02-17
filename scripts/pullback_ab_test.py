"""
Pullback Alpha A/B Test — Classic Pullback vs Sympathy Pullback

Thesis: "Pullback = Momentum + Mean Reversion"
  - Current: BTC directional (sympathy pullback) — alt overreacts to BTC move
  - Classic: Asset's OWN trend is up → dip → buy in trend direction
  - Combined: Both asset trend + BTC directional must agree

Entry Modes to test:
  A. Baseline: No filter (raw rubber band)
  B. BTC Directional (current v4.4): Long only when BTC falling
  C. Asset Trend Pullback: Long only when asset's own trend is UP (buying the dip)
  D. Dual Confirm: Asset trend UP + BTC falling → Long
  E. Either-Or: Asset trend UP OR BTC falling → Long
  F. Momentum Alignment: Long only when asset & BTC BOTH falling (current)
     vs Long only when asset trend UP but short-term dip (classic pullback)
  G. Multi-period confluence: agree on 2+ timeframes

Asset Trend Filters:
  - SMA: price > SMA(N) → uptrend
  - Return: N-bar return > 0 → uptrend  
  - Higher-low: recent low > previous low → uptrend

All with v4.4 optimal: stop=0.9, target=0.4, flat 1.5x.
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
#  NUMBA SIGNAL FUNCTIONS — PULLBACK VARIANTS
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
def compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    n = len(close)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (period + 1.0)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]
    return ema


@numba.jit(nopython=True, cache=True)
def signals_asset_trend_pullback(
    close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    trend_period: int,
    trend_type: int,  # 0=SMA, 1=return, 2=EMA
) -> np.ndarray:
    """Classic pullback: enter dips only in trend direction.

    Long: price dipped 4.5%+ AND asset trend is UP (buying the dip in uptrend)
    Short: price spiked 4.5%+ AND asset trend is DOWN (shorting the pop in downtrend)

    trend_type:
      0 = SMA: close > SMA(trend_period) → uptrend
      1 = Return: close/close[i-trend_period] > 1 → uptrend
      2 = EMA: close > EMA(trend_period) → uptrend
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute trend indicator
    if trend_type == 0:
        # SMA
        trend_line = np.full(n, np.nan)
        s = 0.0
        for i in range(n):
            s += close[i]
            if i >= trend_period:
                s -= close[i - trend_period]
            if i >= trend_period - 1:
                trend_line[i] = s / trend_period
    elif trend_type == 2:
        # EMA
        trend_line = np.full(n, np.nan)
        alpha = 2.0 / (trend_period + 1.0)
        trend_line[0] = close[0]
        for i in range(1, n):
            trend_line[i] = alpha * close[i] + (1.0 - alpha) * trend_line[i - 1]
    else:
        trend_line = np.full(n, np.nan)  # unused for return type

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                # Determine asset trend
                if trend_type == 1:
                    # Return-based trend
                    trend_ret = close[i] / close[i - trend_period] - 1.0
                    is_uptrend = trend_ret > 0
                elif trend_type == 0 or trend_type == 2:
                    # SMA/EMA-based
                    if np.isnan(trend_line[i]):
                        continue
                    is_uptrend = close[i] > trend_line[i]
                else:
                    is_uptrend = True

                if ret < -threshold and is_uptrend:
                    # CLASSIC PULLBACK LONG: dip in an uptrend
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold and not is_uptrend:
                    # CLASSIC PULLBACK SHORT: pop in a downtrend
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
def signals_dual_confirm(
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
    trend_type: int,
) -> np.ndarray:
    """Dual confirmation: Asset trend + BTC directional must both agree.

    Long: dip + asset uptrend + BTC falling (triple confirm)
    Short: pop + asset downtrend + BTC rising (triple confirm)
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute trend
    if trend_type == 0:
        trend_line = np.full(n, np.nan)
        s = 0.0
        for i in range(n):
            s += close[i]
            if i >= trend_period:
                s -= close[i - trend_period]
            if i >= trend_period - 1:
                trend_line[i] = s / trend_period
    elif trend_type == 2:
        trend_line = np.full(n, np.nan)
        alpha = 2.0 / (trend_period + 1.0)
        trend_line[0] = close[0]
        for i in range(1, n):
            trend_line[i] = alpha * close[i] + (1.0 - alpha) * trend_line[i - 1]
    else:
        trend_line = np.full(n, np.nan)

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

                if trend_type == 1:
                    trend_ret = close[i] / close[i - trend_period] - 1.0
                    is_uptrend = trend_ret > 0
                else:
                    if np.isnan(trend_line[i]):
                        continue
                    is_uptrend = close[i] > trend_line[i]

                if ret < -threshold and btc_mom < 0 and is_uptrend:
                    # Triple: dip + BTC down + asset uptrend
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold and btc_mom > 0 and not is_uptrend:
                    # Triple: pop + BTC up + asset downtrend
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
    trend_type: int,
) -> np.ndarray:
    """Either-or: Asset trend OR BTC directional (more opportunities).

    Long: dip + (asset uptrend OR BTC falling)
    Short: pop + (asset downtrend OR BTC rising)
    """
    n = len(close)
    sig = np.zeros(n)

    if trend_type == 0:
        trend_line = np.full(n, np.nan)
        s = 0.0
        for i in range(n):
            s += close[i]
            if i >= trend_period:
                s -= close[i - trend_period]
            if i >= trend_period - 1:
                trend_line[i] = s / trend_period
    elif trend_type == 2:
        trend_line = np.full(n, np.nan)
        alpha = 2.0 / (trend_period + 1.0)
        trend_line[0] = close[0]
        for i in range(1, n):
            trend_line[i] = alpha * close[i] + (1.0 - alpha) * trend_line[i - 1]
    else:
        trend_line = np.full(n, np.nan)

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

                if trend_type == 1:
                    trend_ret = close[i] / close[i - trend_period] - 1.0
                    is_uptrend = trend_ret > 0
                else:
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


@numba.jit(nopython=True, cache=True)
def signals_counter_trend_pullback(
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
    trend_type: int,
) -> np.ndarray:
    """Counter-trend pullback: opposite of classic.

    Long: dip + (asset DOWNtrend → "oversold in downtrend" mean-reversion)
    This is the opposite — buying what's falling on assumption of snap-back.
    Essentially tests if "buying-in-trend" or "buying-against-trend" is better.
    No BTC filter here.
    """
    n = len(close)
    sig = np.zeros(n)

    if trend_type == 0:
        trend_line = np.full(n, np.nan)
        s = 0.0
        for i in range(n):
            s += close[i]
            if i >= trend_period:
                s -= close[i - trend_period]
            if i >= trend_period - 1:
                trend_line[i] = s / trend_period
    elif trend_type == 2:
        trend_line = np.full(n, np.nan)
        alpha = 2.0 / (trend_period + 1.0)
        trend_line[0] = close[0]
        for i in range(1, n):
            trend_line[i] = alpha * close[i] + (1.0 - alpha) * trend_line[i - 1]
    else:
        trend_line = np.full(n, np.nan)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                if trend_type == 1:
                    trend_ret = close[i] / close[i - trend_period] - 1.0
                    is_uptrend = trend_ret > 0
                else:
                    if np.isnan(trend_line[i]):
                        continue
                    is_uptrend = close[i] > trend_line[i]

                # COUNTER-TREND: buy dips in DOWNtrends
                if ret < -threshold and not is_uptrend:
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold and is_uptrend:
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
def signals_multi_tf_confluence(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    short_period: int,
    mid_period: int,
    long_period: int,
    min_agree: int,
) -> np.ndarray:
    """Multi-timeframe trend confluence.

    Check trend on 3 periods (short/mid/long). Enter only when
    ≥ min_agree of them agree + BTC directional.

    Long: dip + ≥min_agree trends UP + BTC falling
    Short: pop + ≥min_agree trends DOWN + BTC rising
    """
    n = len(close)
    sig = np.zeros(n)
    start_bar = max(lookback, btc_mom_period, long_period)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                # Count trend agreement on 3 timeframes
                s_ret = close[i] / close[i - short_period] - 1.0
                m_ret = close[i] / close[i - mid_period] - 1.0
                l_ret = close[i] / close[i - long_period] - 1.0

                up_count = 0
                if s_ret > 0:
                    up_count += 1
                if m_ret > 0:
                    up_count += 1
                if l_ret > 0:
                    up_count += 1

                down_count = 3 - up_count

                if ret < -threshold and btc_mom < 0 and up_count >= min_agree:
                    # Pullback: price dipped but ≥min_agree TFs still UP
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold and btc_mom > 0 and down_count >= min_agree:
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
#  DATA LOADING & BACKTEST HELPERS
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


# ═══════════════════════════════════════════════════════════════
#  MAIN — COMPREHENSIVE SWEEP
# ═══════════════════════════════════════════════════════════════

def main():
    print('Loading data...')
    btc_close, data = load_data()
    print(f'Loaded {len(data)} symbols, {N_BARS} bars each.\n')

    # v4.4 optimal params
    LB, TH, TGT, STOP, MH, SD = 8, 0.045, 0.4, 0.9, 10, 0.0
    BTC_PER = 32
    LEV = 1.5

    # ──────────────────────────────────────────────
    # PHASE 1: Entry Mode Comparison (fixed trend_period=96=24h, SMA)
    # ──────────────────────────────────────────────
    print('=' * 100)
    print('PHASE 1: Entry Mode Comparison (trend_period=96=24h, SMA, 1.5x)')
    print('=' * 100)

    def sig_no_filter(close, btc_close):
        return compute_signals(close, LB, TH, TGT, STOP, MH, SD)

    def sig_btc_only(close, btc_close):
        return compute_signals_with_btc_directional(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER,
        )

    def sig_asset_trend(close, btc_close):
        return signals_asset_trend_pullback(
            close, LB, TH, TGT, STOP, MH, SD, 96, 0,
        )

    def sig_dual(close, btc_close):
        return signals_dual_confirm(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 96, 0,
        )

    def sig_either(close, btc_close):
        return signals_either_or(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 96, 0,
        )

    def sig_counter(close, btc_close):
        return signals_counter_trend_pullback(
            close, btc_close, LB, TH, TGT, STOP, MH, SD, BTC_PER, 96, 0,
        )

    modes = [
        ('A. No filter (raw RB)', sig_no_filter),
        ('B. BTC Directional (v4.4 current)', sig_btc_only),
        ('C. Asset Trend Pullback (SMA-96)', sig_asset_trend),
        ('D. Dual Confirm (asset+BTC)', sig_dual),
        ('E. Either-Or (asset OR BTC)', sig_either),
        ('F. Counter-trend (buy downtrent dip)', sig_counter),
    ]

    results_phase1 = {}
    for label, func in modes:
        cal, ret, dd, sh, wr, rr, trades = run_portfolio(
            data, btc_close, func, LEV, label,
        )
        results_phase1[label] = (cal, ret, dd, rr, trades)

    # ──────────────────────────────────────────────
    # PHASE 2: Asset Trend Period Sweep (classic pullback)
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 2: Asset Trend Period Sweep (Classic Pullback, SMA, 1.5x)')
    print('=' * 100)

    trend_periods = [16, 24, 32, 48, 64, 96, 128, 192, 288, 384]
    # 16=4h, 24=6h, 32=8h, 48=12h, 64=16h, 96=24h, 128=32h,
    # 192=48h, 288=72h, 384=96h

    for tp in trend_periods:
        hours = tp * 15 / 60
        def mk(tperiod):
            def f(close, btc_close):
                return signals_asset_trend_pullback(
                    close, LB, TH, TGT, STOP, MH, SD, tperiod, 0,
                )
            return f
        run_portfolio(
            data, btc_close, mk(tp), LEV,
            f'Classic Pullback SMA({tp}) = {hours:.0f}h',
        )

    # ──────────────────────────────────────────────
    # PHASE 3: Trend Type Comparison (SMA vs Return vs EMA)
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 3: Trend Type Comparison (period=96, 1.5x)')
    print('=' * 100)

    for tt, name in [(0, 'SMA'), (1, 'Return'), (2, 'EMA')]:
        def mk(ttype):
            def f(close, btc_close):
                return signals_asset_trend_pullback(
                    close, LB, TH, TGT, STOP, MH, SD, 96, ttype,
                )
            return f
        run_portfolio(
            data, btc_close, mk(tt), LEV,
            f'Asset Pullback type={name}(96)',
        )

    # ──────────────────────────────────────────────
    # PHASE 4: Dual Confirm Period Sweep  
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 4: Dual Confirm (Asset+BTC) Period Sweep (SMA, 1.5x)')
    print('=' * 100)

    for tp in [16, 32, 48, 64, 96, 128, 192]:
        hours = tp * 15 / 60
        def mk(tperiod):
            def f(close, btc_close):
                return signals_dual_confirm(
                    close, btc_close, LB, TH, TGT, STOP, MH, SD,
                    BTC_PER, tperiod, 0,
                )
            return f
        run_portfolio(
            data, btc_close, mk(tp), LEV,
            f'Dual (BTC+SMA({tp})={hours:.0f}h)',
        )

    # ──────────────────────────────────────────────
    # PHASE 5: Either-Or Period Sweep
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 5: Either-Or (Asset OR BTC) Period Sweep (SMA, 1.5x)')
    print('=' * 100)

    for tp in [16, 32, 48, 64, 96, 128, 192]:
        hours = tp * 15 / 60
        def mk(tperiod):
            def f(close, btc_close):
                return signals_either_or(
                    close, btc_close, LB, TH, TGT, STOP, MH, SD,
                    BTC_PER, tperiod, 0,
                )
            return f
        run_portfolio(
            data, btc_close, mk(tp), LEV,
            f'Either (SMA({tp})={hours:.0f}h OR BTC)',
        )

    # ──────────────────────────────────────────────
    # PHASE 6: Multi-TF Confluence
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 6: Multi-TF Confluence (short/mid/long + BTC, 1.5x)')
    print('=' * 100)

    configs_mtf = [
        # (short, mid, long, min_agree)
        (16, 64, 192, 1),   # 4h/16h/48h, just 1 agrees
        (16, 64, 192, 2),   # 2 of 3 agree
        (16, 64, 192, 3),   # all 3 agree
        (24, 96, 288, 1),   # 6h/24h/72h, 1 agree
        (24, 96, 288, 2),   # 2 of 3
        (24, 96, 288, 3),   # all 3
        (32, 96, 384, 1),   # 8h/24h/96h
        (32, 96, 384, 2),
        (32, 96, 384, 3),
    ]

    for short_p, mid_p, long_p, min_ag in configs_mtf:
        def mk(s, m, l, ma):
            def f(close, btc_close):
                return signals_multi_tf_confluence(
                    close, btc_close, LB, TH, TGT, STOP, MH, SD,
                    BTC_PER, s, m, l, ma,
                )
            return f
        sh = short_p * 15 / 60
        mh = mid_p * 15 / 60
        lh = long_p * 15 / 60
        run_portfolio(
            data, btc_close, mk(short_p, mid_p, long_p, min_ag), LEV,
            f'MTF({sh:.0f}h/{mh:.0f}h/{lh:.0f}h) agree≥{min_ag}+BTC',
        )

    # ──────────────────────────────────────────────
    # PHASE 7: Best winners from above at different leverage
    # ──────────────────────────────────────────────
    print('\n' + '=' * 100)
    print('PHASE 7: Leverage Sweep on top modes (for comparison)')
    print('=' * 100)

    for lev in [1.0, 1.5, 2.0, 3.0]:
        run_portfolio(
            data, btc_close, sig_btc_only, lev,
            f'BTC Directional (v4.4) lev={lev:.1f}x',
        )

    for tp_best in [48, 96]:
        for lev in [1.0, 1.5, 2.0, 3.0]:
            def mk(tperiod):
                def f(close, btc_close):
                    return signals_asset_trend_pullback(
                        close, LB, TH, TGT, STOP, MH, SD, tperiod, 0,
                    )
                return f
            run_portfolio(
                data, btc_close, mk(tp_best), lev,
                f'Asset Pullback SMA({tp_best}) lev={lev:.1f}x',
            )

    print('\nDone.')


if __name__ == '__main__':
    main()
