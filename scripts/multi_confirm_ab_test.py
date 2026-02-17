#!/usr/bin/env python3
"""
Multi-Confirmation Entry A/B Test (v4.4 candidate)

"정말 확실한 여러 다중 프레임 및 복합 지표 검증이 될 때 들어가서
큰 손익비를 불러오고, 레버리지를 쓸 수 있다."

Current problem: target=0.3, stop=1.0 → R:R = 0.30 (terrible).
v4.3 "fix" was disabling leverage entirely. But the real fix is to
IMPROVE the R:R via multi-indicator confirmation gating, then leverage
becomes justified.

Confirmation indicators (5 total):
  1. Volume Spike:     vol > N × SMA(vol, 20) → liquidation cascade
  2. RSI Extreme:      RSI < 25 (long) or > 75 (short) → stretched
  3. ATR Extension:    move > N × ATR(14) → abnormal volatility
  4. Wick Rejection:   reversal wick > 30% of range → buyers/sellers stepping in
  5. Multi-Lookback:   extreme on lookback/2 AND lookback*2 → multi-TF confluence

BTC Directional alignment is a HARD gate (must pass, not in score).

When score >= min_confirmations:
  - Use confirmed_stop (tighter) + confirmed_target (wider) → R:R >> 0.3
  - With R:R > 1.0, leverage is justified

Test structure:
  Phase 1: Individual filter impact (which confirmations matter?)
  Phase 2: Combined filters + R:R improvement
  Phase 3: Leverage scaling on confirmed entries

Data: 30 symbols × 35,040 bars (15m × 365d), real Binance USDT-M
"""

import gzip
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import numba

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantbox.backtest.vectorized_backtest import _backtest_core, _backtest_core_partial
from quantbox.strategy.rubberband import (
    compute_signals_partial_exit_btc_directional,
    apply_rr_gated_leverage,
)

# ── Constants ──
LOOKBACK = 8
THRESHOLD = 0.045
BASE_TARGET = 0.3
BASE_STOP = 1.0
MAX_HOLD = 10
STOP_DECAY = 0.0
BTC_MOM_PERIOD = 32
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.01 / 100 / 32

TOP10 = [
    "FETUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT", "DOTUSDT",
    "OPUSDT", "SHIBUSDT", "DOGEUSDT", "LTCUSDT", "ATOMUSDT",
]

OUT_FILE = Path("logs/multi_confirm_ab_output.txt")
OUT_FILE.parent.mkdir(exist_ok=True)
out = open(OUT_FILE, "w")


def log(msg=""):
    print(msg)
    out.write(msg + "\n")
    out.flush()


# ═══════════════════════════════════════════════════════════════
#  Indicator Pre-computation (NumPy — run once per symbol)
# ═══════════════════════════════════════════════════════════════

def compute_rsi_array(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI array (Wilder's smoothing)."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - 100 / (1 + rs)
    else:
        rsi[period] = 100.0

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0

    return rsi


def compute_atr_array(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14,
) -> np.ndarray:
    """Compute ATR array (Wilder's smoothing)."""
    n = len(close)
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        atr[:period - 1] = atr[period - 1]
    else:
        atr[:] = np.mean(tr)

    return atr


def compute_vol_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume / SMA(volume, period)."""
    n = len(volume)
    ratio = np.ones(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += volume[i]
        if i >= period:
            cumsum -= volume[i - period]
            sma = cumsum / period
        elif i >= period - 1:
            sma = cumsum / period
        else:
            sma = cumsum / (i + 1)
        if sma > 0:
            ratio[i] = volume[i] / sma
    return ratio


def precompute_indicators(ohlcv: dict) -> dict:
    """Pre-compute all confirmation indicator arrays for one symbol."""
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    open_ = ohlcv["open"]
    volume = ohlcv["volume"]

    rsi = compute_rsi_array(close, 14)
    atr = compute_atr_array(high, low, close, 14)
    vol_ratio = compute_vol_ratio(volume, 20)

    # Wick fractions (per bar)
    n = len(close)
    lower_wick_frac = np.zeros(n)
    upper_wick_frac = np.zeros(n)
    for i in range(n):
        rng = high[i] - low[i]
        if rng > 0:
            lower_wick_frac[i] = (min(open_[i], close[i]) - low[i]) / rng
            upper_wick_frac[i] = (high[i] - max(open_[i], close[i])) / rng

    return {
        "rsi": rsi,
        "atr": atr,
        "vol_ratio": vol_ratio,
        "lower_wick_frac": lower_wick_frac,
        "upper_wick_frac": upper_wick_frac,
    }


# ═══════════════════════════════════════════════════════════════
#  Numba Signal Functions
# ═══════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def compute_signals_multi_confirm(
    close: np.ndarray,
    btc_close: np.ndarray,
    vol_ratio: np.ndarray,
    rsi: np.ndarray,
    atr: np.ndarray,
    lower_wick_frac: np.ndarray,
    upper_wick_frac: np.ndarray,
    lookback: int,
    threshold: float,
    confirmed_target: float,
    confirmed_stop: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    min_confirmations: int,
    vol_thresh: float,
    rsi_lo: float,
    rsi_hi: float,
    atr_mult: float,
    wick_thresh: float,
    leverage: float,
) -> np.ndarray:
    """Compute signals with multi-indicator confirmation gate.

    Entry requires:
      1. Basic trigger: |8-bar return| > threshold
      2. BTC directional alignment (HARD gate)
      3. confirmation_score >= min_confirmations

    Confirmed entries use tighter stop + wider target → better R:R.
    Signal magnitude encodes leverage (e.g., ±5.0 = 5x).

    Returns:
        Signal array: 0.0 (flat), ±leverage (position).
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    pos_target = 0.0
    pos_stop = 0.0

    start_bar = max(lookback * 2, btc_mom_period, 30)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                # BTC directional (HARD gate)
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                direction = 0
                if ret < -threshold and btc_mom < 0:
                    direction = 1   # Long: fade crash when BTC also falling
                elif ret > threshold and btc_mom > 0:
                    direction = -1  # Short: fade spike when BTC also rising

                if direction != 0:
                    # ── Count confirmation score ──
                    score = 0

                    # 1. Volume spike
                    if vol_ratio[i] >= vol_thresh:
                        score += 1

                    # 2. RSI extreme (direction-dependent)
                    if direction == 1 and rsi[i] < rsi_lo:
                        score += 1
                    elif direction == -1 and rsi[i] > rsi_hi:
                        score += 1

                    # 3. ATR extension: move >> ATR
                    if atr[i] > 0:
                        move_in_price = abs(ret) * close[i]
                        if move_in_price / atr[i] >= atr_mult:
                            score += 1

                    # 4. Wick rejection (direction-dependent)
                    if direction == 1 and lower_wick_frac[i] >= wick_thresh:
                        score += 1  # Hammer: buyers stepping in
                    elif direction == -1 and upper_wick_frac[i] >= wick_thresh:
                        score += 1  # Shooting star: sellers stepping in

                    # 5. Multi-lookback confluence
                    # Extreme on half-lookback AND double-lookback
                    if i >= lookback * 2:
                        half_ret = abs(close[i] / close[i - lookback // 2] - 1.0)
                        dbl_ret = abs(close[i] / close[i - lookback * 2] - 1.0)
                        if half_ret > threshold * 0.6 and dbl_ret > threshold * 1.2:
                            score += 1

                    # ── Gate: only enter if score >= min_confirmations ──
                    if score >= min_confirmations:
                        position = direction
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                        pos_target = move_size * confirmed_target
                        pos_stop = move_size * confirmed_stop

        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            eff_stop = pos_stop * time_factor

            if gain >= pos_target:
                position = 0
            elif gain < -eff_stop:
                position = 0
            elif best_pnl > pos_target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

        if position != 0:
            sig[i] = float(position) * leverage
        else:
            sig[i] = 0.0

    return sig


@numba.jit(nopython=True, cache=True)
def compute_signals_multi_confirm_partial(
    close: np.ndarray,
    btc_close: np.ndarray,
    vol_ratio: np.ndarray,
    rsi: np.ndarray,
    atr: np.ndarray,
    lower_wick_frac: np.ndarray,
    upper_wick_frac: np.ndarray,
    lookback: int,
    threshold: float,
    confirmed_target: float,
    confirmed_stop: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    min_confirmations: int,
    vol_thresh: float,
    rsi_lo: float,
    rsi_hi: float,
    atr_mult: float,
    wick_thresh: float,
    leverage: float,
) -> np.ndarray:
    """Multi-confirm entry + partial exit (50% on profit, BE stop on rest).

    Same entry logic as compute_signals_multi_confirm.
    Exit: Phase 1 (full) → partial close on profit → Phase 2 (half) → BE stop.
    Signal magnitude: ±leverage (full), ±leverage*0.5 (half), 0 (flat).
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    phase = 0  # 0=flat, 1=full, 2=half
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    pos_target = 0.0
    pos_stop = 0.0

    start_bar = max(lookback * 2, btc_mom_period, 30)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if phase == 0:
            if abs(ret) > threshold:
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                direction = 0
                if ret < -threshold and btc_mom < 0:
                    direction = 1
                elif ret > threshold and btc_mom > 0:
                    direction = -1

                if direction != 0:
                    score = 0
                    if vol_ratio[i] >= vol_thresh:
                        score += 1
                    if direction == 1 and rsi[i] < rsi_lo:
                        score += 1
                    elif direction == -1 and rsi[i] > rsi_hi:
                        score += 1
                    if atr[i] > 0:
                        move_in_price = abs(ret) * close[i]
                        if move_in_price / atr[i] >= atr_mult:
                            score += 1
                    if direction == 1 and lower_wick_frac[i] >= wick_thresh:
                        score += 1
                    elif direction == -1 and upper_wick_frac[i] >= wick_thresh:
                        score += 1
                    if i >= lookback * 2:
                        half_ret = abs(close[i] / close[i - lookback // 2] - 1.0)
                        dbl_ret = abs(close[i] / close[i - lookback * 2] - 1.0)
                        if half_ret > threshold * 0.6 and dbl_ret > threshold * 1.2:
                            score += 1

                    if score >= min_confirmations:
                        position = direction
                        phase = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                        pos_target = move_size * confirmed_target
                        pos_stop = move_size * confirmed_stop

        elif phase == 1:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            eff_stop = pos_stop * time_factor

            if gain >= pos_target:
                position = 0
                phase = 0
            elif gain < -eff_stop:
                position = 0
                phase = 0
            elif best_pnl > pos_target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0
            elif gain > 0:
                phase = 2
                best_pnl = gain

        elif phase == 2:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain

            if gain >= pos_target:
                position = 0
                phase = 0
            elif gain <= 0:
                position = 0
                phase = 0
            elif best_pnl > pos_target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0

        if phase == 0:
            sig[i] = 0.0
        elif phase == 1:
            sig[i] = float(position) * leverage
        elif phase == 2:
            sig[i] = float(position) * leverage * 0.5

    return sig


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def load_all_data():
    """Load OHLCV data for all symbols + pre-compute indicators."""
    cache_dir = Path("data/cache_extended")
    all_data = {}
    btc_data = None

    for f in sorted(cache_dir.glob("*_15m_365d.pkl.gz")):
        symbol = f.name.split("_15m_")[0]
        with gzip.open(f, "rb") as fh:
            df = pickle.load(fh)
        if "Close" not in df.columns or len(df) < 200:
            continue

        ohlcv = {
            "open": df["Open"].values.astype(np.float64),
            "high": df["High"].values.astype(np.float64),
            "low": df["Low"].values.astype(np.float64),
            "close": df["Close"].values.astype(np.float64),
            "volume": df["Volume"].values.astype(np.float64),
        }

        if symbol == "BTCUSDT":
            btc_data = ohlcv
        else:
            indicators = precompute_indicators(ohlcv)
            all_data[symbol] = {**ohlcv, **indicators}

    return all_data, btc_data


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def fast_metrics(eq, start=0):
    """Return (ret%, dd%, calmar, sharpe)."""
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


def trade_stats_from_core(close, signals, cost_per_side, funding_per_bar, use_partial=False):
    """Run backtest and extract detailed trade statistics."""
    if use_partial:
        result = _backtest_core_partial(close, signals, cost_per_side, funding_per_bar)
    else:
        result = _backtest_core(close, signals, cost_per_side, funding_per_bar)
    equity, total_ret, sharpe, max_dd, win_rate, n_trades, bars_in, pf, avg_ret, trade_rets = result
    return {
        "equity": equity,
        "total_ret": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "pf": pf,
        "avg_ret": avg_ret,
        "trade_rets": trade_rets,
    }


def portfolio_metrics(symbol_results, symbols, n_bars):
    """Aggregate per-symbol results into portfolio metrics."""
    n = len(symbols)
    if n == 0:
        return {"ret": 0, "dd": 0, "cal": 0, "sh": 0, "wr": 0, "trades": 0, "equity": np.ones(1)}

    port_eq = np.ones(n_bars)
    total_trades = 0
    total_wins = 0
    all_trade_rets = []

    for sym in symbols:
        r = symbol_results.get(sym)
        if r is None:
            continue
        eq = r["equity"][:n_bars]
        port_eq *= eq ** (1.0 / n)
        total_trades += r["n_trades"]
        if r["n_trades"] > 0:
            total_wins += int(r["win_rate"] / 100 * r["n_trades"])
        all_trade_rets.extend(r["trade_rets"].tolist())

    ret, dd, cal, sh = fast_metrics(port_eq)
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    # Realized R:R
    winners = [x for x in all_trade_rets if x > 0]
    losers = [x for x in all_trade_rets if x < 0]
    avg_win = np.mean(winners) if winners else 0
    avg_loss = abs(np.mean(losers)) if losers else 1e-10
    realized_rr = avg_win / avg_loss if avg_loss > 0 else 999

    return {
        "ret": ret, "dd": dd, "cal": cal, "sh": sh,
        "wr": wr, "trades": total_trades,
        "realized_rr": realized_rr,
        "avg_win": avg_win * 100,
        "avg_loss": -abs(np.mean(losers)) * 100 if losers else 0,
        "equity": port_eq,
    }


def rolling_90d_positive(equity, bar_per_90d=8640):
    """Count how many 90-day rolling windows are profitable."""
    n = len(equity)
    if n < bar_per_90d:
        return 0, 0
    windows = 0
    positive = 0
    step = bar_per_90d // 4  # 22.5-day step
    for start in range(0, n - bar_per_90d + 1, step):
        end = start + bar_per_90d
        ret = equity[end - 1] / equity[start] - 1
        windows += 1
        if ret > 0:
            positive += 1
    return positive, windows


# ═══════════════════════════════════════════════════════════════
#  Test Runner
# ═══════════════════════════════════════════════════════════════

def run_variant(all_data, btc_close, symbols, n_bars, label, signal_func, use_partial=False):
    """Run one A/B test variant and return portfolio metrics."""
    sym_results = {}
    for sym in symbols:
        d = all_data.get(sym)
        if d is None:
            continue
        close = d["close"][:n_bars]
        sigs = signal_func(sym, d, btc_close[:n_bars], n_bars)

        stats = trade_stats_from_core(close, sigs, COST_PER_SIDE, FUNDING_PER_BAR, use_partial)
        sym_results[sym] = stats

    return portfolio_metrics(sym_results, symbols, n_bars)


def print_result(label, m, top10_m=None):
    """Print formatted result line."""
    pos, total = rolling_90d_positive(m["equity"])
    line = (
        f"  {label:<45s}  "
        f"Ret {m['ret']:+8.1f}%  DD {m['dd']:5.1f}%  "
        f"Cal {m['cal']:6.2f}  Sh {m['sh']:5.2f}  "
        f"WR {m['wr']:5.1f}%  Trades {m['trades']:5d}  "
        f"R:R {m['realized_rr']:.2f}  "
        f"AvgW {m['avg_win']:+.2f}%  AvgL {m['avg_loss']:.2f}%  "
        f"Roll {pos}/{total}"
    )
    log(line)
    if top10_m:
        pos10, total10 = rolling_90d_positive(top10_m["equity"])
        line10 = (
            f"    └─ Top-10:                                   "
            f"Ret {top10_m['ret']:+8.1f}%  DD {top10_m['dd']:5.1f}%  "
            f"Cal {top10_m['cal']:6.2f}  Sh {top10_m['sh']:5.2f}  "
            f"WR {top10_m['wr']:5.1f}%  Trades {top10_m['trades']:5d}  "
            f"R:R {top10_m['realized_rr']:.2f}  "
            f"Roll {pos10}/{total10}"
        )
        log(line10)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    log("=" * 120)
    log(f"Multi-Confirmation Entry A/B Test — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Data: 30 symbols × 35,040 bars (15m × 365d)")
    log(f"Insight: 손익비가 높아야 레버리지를 쓴다 → 다중 확인이 손익비를 높인다")
    log("=" * 120)

    t0 = time.time()
    all_data, btc_data = load_all_data()
    btc_close = btc_data["close"]
    symbols = sorted(all_data.keys())
    n_bars = min(len(d["close"]) for d in all_data.values())
    n_bars = min(n_bars, len(btc_close))
    top10_syms = [s for s in TOP10 if s in all_data]

    log(f"\nLoaded: {len(symbols)} symbols, {n_bars} bars ({n_bars / 96:.0f} days)")
    log(f"Top-10: {top10_syms}")
    log(f"Load time: {time.time() - t0:.1f}s")

    # ── Warmup Numba (first call compiles) ──
    log("\nWarming up Numba...")
    _dummy = all_data[symbols[0]]
    _dc = _dummy["close"][:100].copy()
    _db = btc_close[:100].copy()
    compute_signals_multi_confirm(
        _dc, _db,
        _dummy["vol_ratio"][:100], _dummy["rsi"][:100],
        _dummy["atr"][:100], _dummy["lower_wick_frac"][:100],
        _dummy["upper_wick_frac"][:100],
        LOOKBACK, THRESHOLD, 0.5, 0.5, MAX_HOLD, STOP_DECAY,
        BTC_MOM_PERIOD, 1, 1.5, 25.0, 75.0, 2.0, 0.3, 1.0,
    )
    compute_signals_multi_confirm_partial(
        _dc, _db,
        _dummy["vol_ratio"][:100], _dummy["rsi"][:100],
        _dummy["atr"][:100], _dummy["lower_wick_frac"][:100],
        _dummy["upper_wick_frac"][:100],
        LOOKBACK, THRESHOLD, 0.5, 0.5, MAX_HOLD, STOP_DECAY,
        BTC_MOM_PERIOD, 1, 1.5, 25.0, 75.0, 2.0, 0.3, 1.0,
    )
    log("Numba compiled.\n")

    # ═══════════════════════════════════════════════════════
    #  VARIANT A: Baseline v4.3
    # ═══════════════════════════════════════════════════════
    log("=" * 120)
    log("VARIANT A: Baseline v4.3 (partial exit + R:R gate = 1x leverage)")
    log("=" * 120)

    def baseline_func(sym, d, btc_c, nb):
        close = d["close"][:nb]
        sigs = compute_signals_partial_exit_btc_directional(
            close, btc_c, LOOKBACK, THRESHOLD, BASE_TARGET, BASE_STOP,
            MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
        )
        lev_sigs = apply_rr_gated_leverage(
            close, sigs, LOOKBACK, THRESHOLD, BASE_TARGET, BASE_STOP,
            COST_PER_SIDE, 3.0, 10.0, 7.0, 0.30,
        )
        return lev_sigs

    m_a = run_variant(all_data, btc_close, symbols, n_bars, "A", baseline_func, use_partial=True)
    m_a10 = run_variant(all_data, btc_close, top10_syms, n_bars, "A-Top10", baseline_func, use_partial=True)
    print_result("A: Baseline v4.3 (partial+RR-gate=1x)", m_a, m_a10)

    # ═══════════════════════════════════════════════════════
    #  PHASE 1: Individual filter impact (same R:R, 1x leverage)
    # ═══════════════════════════════════════════════════════
    log("\n" + "=" * 120)
    log("PHASE 1: Individual filter impact (same R:R as baseline, 1x, no partial)")
    log("Which confirmations improve realized R:R and win rate?")
    log("=" * 120)

    indicator_configs = [
        ("No filter (BTC dir only)",     0, 99.0, 0.0, 100.0, 99.0, 0.0),
        ("Vol spike only (>1.5×SMA)",    1, 1.5,  0.0, 100.0, 99.0, 0.0),
        ("Vol spike only (>2.0×SMA)",    1, 2.0,  0.0, 100.0, 99.0, 0.0),
        ("RSI extreme only (<25/>75)",   1, 99.0, 25.0, 75.0,  99.0, 0.0),
        ("RSI extreme only (<30/>70)",   1, 99.0, 30.0, 70.0,  99.0, 0.0),
        ("ATR extension only (>2×)",     1, 99.0, 0.0, 100.0, 2.0,  0.0),
        ("ATR extension only (>3×)",     1, 99.0, 0.0, 100.0, 3.0,  0.0),
        ("Wick rejection only (>0.3)",   1, 99.0, 0.0, 100.0, 99.0, 0.3),
        ("Wick rejection only (>0.2)",   1, 99.0, 0.0, 100.0, 99.0, 0.2),
    ]

    for lbl, min_conf, vol_t, rsi_l, rsi_h, atr_m, wick_t in indicator_configs:
        def mk_func(mc, vt, rl, rh, am, wt):
            def f(sym, d, btc_c, nb):
                return compute_signals_multi_confirm(
                    d["close"][:nb], btc_c,
                    d["vol_ratio"][:nb], d["rsi"][:nb],
                    d["atr"][:nb], d["lower_wick_frac"][:nb],
                    d["upper_wick_frac"][:nb],
                    LOOKBACK, THRESHOLD, BASE_TARGET, BASE_STOP,
                    MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
                    mc, vt, rl, rh, am, wt, 1.0,
                )
            return f

        func = mk_func(min_conf, vol_t, rsi_l, rsi_h, atr_m, wick_t)
        m = run_variant(all_data, btc_close, symbols, n_bars, lbl, func)
        m10 = run_variant(all_data, btc_close, top10_syms, n_bars, lbl, func)
        print_result(lbl, m, m10)

    # ═══════════════════════════════════════════════════════
    #  PHASE 2: Combined confirmations + R:R improvement
    # ═══════════════════════════════════════════════════════
    log("\n" + "=" * 120)
    log("PHASE 2: Combined confirmations, sweep R:R and min_confirmations")
    log("Question: what confirmed_stop × confirmed_target × min_conf maximizes Calmar?")
    log("=" * 120)

    # Fixed indicator params (reasonable defaults)
    VOL_T = 1.5
    RSI_LO = 30.0
    RSI_HI = 70.0
    ATR_M = 2.0
    WICK_T = 0.2

    best_cal = -999
    best_cfg = None

    for min_conf in [1, 2, 3, 4]:
        for c_stop in [0.3, 0.5, 0.7, 1.0]:
            for c_target in [0.3, 0.5, 0.7]:
                theoretical_rr = c_target / c_stop if c_stop > 0 else 999

                def mk_func2(mc, cs, ct):
                    def f(sym, d, btc_c, nb):
                        return compute_signals_multi_confirm(
                            d["close"][:nb], btc_c,
                            d["vol_ratio"][:nb], d["rsi"][:nb],
                            d["atr"][:nb], d["lower_wick_frac"][:nb],
                            d["upper_wick_frac"][:nb],
                            LOOKBACK, THRESHOLD, ct, cs,
                            MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
                            mc, VOL_T, RSI_LO, RSI_HI, ATR_M, WICK_T, 1.0,
                        )
                    return f

                func = mk_func2(min_conf, c_stop, c_target)
                m10 = run_variant(all_data, btc_close, top10_syms, n_bars, "", func)
                pos, total = rolling_90d_positive(m10["equity"])

                tag = "★" if m10["cal"] > best_cal and m10["trades"] >= 30 else " "
                if m10["cal"] > best_cal and m10["trades"] >= 30:
                    best_cal = m10["cal"]
                    best_cfg = (min_conf, c_stop, c_target)

                log(
                    f" {tag} min_conf={min_conf}  stop={c_stop:.1f}  "
                    f"target={c_target:.1f}  "
                    f"theoR:R={theoretical_rr:.2f}  | "
                    f"Ret {m10['ret']:+7.1f}%  DD {m10['dd']:5.1f}%  "
                    f"Cal {m10['cal']:6.2f}  "
                    f"WR {m10['wr']:5.1f}%  Trades {m10['trades']:4d}  "
                    f"R:R {m10['realized_rr']:.2f}  "
                    f"Roll {pos}/{total}"
                )

    log(f"\n  ★ Best Phase 2: min_conf={best_cfg[0]}, stop={best_cfg[1]:.1f}, "
        f"target={best_cfg[2]:.1f}, Calmar={best_cal:.2f}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 3: Leverage on best confirmed setup
    # ═══════════════════════════════════════════════════════
    log("\n" + "=" * 120)
    log("PHASE 3: Leverage scaling on best confirmed setup")
    log(f"Using: min_conf={best_cfg[0]}, stop={best_cfg[1]}, target={best_cfg[2]}")
    log("Question: at what leverage does the confirmed setup maximize Calmar?")
    log("=" * 120)

    bc_min, bc_stop, bc_target = best_cfg

    for lev in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        # Without partial exit
        def mk_func3(leverage):
            def f(sym, d, btc_c, nb):
                return compute_signals_multi_confirm(
                    d["close"][:nb], btc_c,
                    d["vol_ratio"][:nb], d["rsi"][:nb],
                    d["atr"][:nb], d["lower_wick_frac"][:nb],
                    d["upper_wick_frac"][:nb],
                    LOOKBACK, THRESHOLD, bc_target, bc_stop,
                    MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
                    bc_min, VOL_T, RSI_LO, RSI_HI, ATR_M, WICK_T, leverage,
                )
            return f

        func = mk_func3(lev)
        m_all = run_variant(all_data, btc_close, symbols, n_bars, "", func)
        m10 = run_variant(all_data, btc_close, top10_syms, n_bars, "", func)
        print_result(f"Lev={lev:.0f}x (no partial)", m_all, m10)

    log("")

    # With partial exit
    log("  ── Same setup + partial exit (50% close on profit, BE stop) ──")
    for lev in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        def mk_func4(leverage):
            def f(sym, d, btc_c, nb):
                return compute_signals_multi_confirm_partial(
                    d["close"][:nb], btc_c,
                    d["vol_ratio"][:nb], d["rsi"][:nb],
                    d["atr"][:nb], d["lower_wick_frac"][:nb],
                    d["upper_wick_frac"][:nb],
                    LOOKBACK, THRESHOLD, bc_target, bc_stop,
                    MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
                    bc_min, VOL_T, RSI_LO, RSI_HI, ATR_M, WICK_T, leverage,
                )
            return f

        func = mk_func4(lev)
        m_all = run_variant(all_data, btc_close, symbols, n_bars, "", func, use_partial=True)
        m10 = run_variant(all_data, btc_close, top10_syms, n_bars, "", func, use_partial=True)
        print_result(f"Lev={lev:.0f}x + partial exit", m_all, m10)

    # ═══════════════════════════════════════════════════════
    #  PHASE 4: Indicator threshold sensitivity on best config
    # ═══════════════════════════════════════════════════════
    log("\n" + "=" * 120)
    log("PHASE 4: Indicator threshold sensitivity (best R:R + 3x leverage)")
    log("Sweep individual indicator thresholds to find sweet spots")
    log("=" * 120)

    for param_name, values in [
        ("vol_thresh", [1.0, 1.5, 2.0, 2.5, 3.0]),
        ("rsi_extreme", [20, 25, 30, 35, 40]),
        ("atr_mult", [1.5, 2.0, 2.5, 3.0, 4.0]),
        ("wick_thresh", [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]),
    ]:
        log(f"\n  ── {param_name} sweep ──")
        for v in values:
            vt = VOL_T if param_name != "vol_thresh" else v
            rl = RSI_LO if param_name != "rsi_extreme" else v
            rh = RSI_HI if param_name != "rsi_extreme" else (100 - v)
            am = ATR_M if param_name != "atr_mult" else v
            wt = WICK_T if param_name != "wick_thresh" else v

            def mk_func5(vt_, rl_, rh_, am_, wt_):
                def f(sym, d, btc_c, nb):
                    return compute_signals_multi_confirm(
                        d["close"][:nb], btc_c,
                        d["vol_ratio"][:nb], d["rsi"][:nb],
                        d["atr"][:nb], d["lower_wick_frac"][:nb],
                        d["upper_wick_frac"][:nb],
                        LOOKBACK, THRESHOLD, bc_target, bc_stop,
                        MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
                        bc_min, vt_, rl_, rh_, am_, wt_, 3.0,
                    )
                return f

            func = mk_func5(vt, rl, rh, am, wt)
            m10 = run_variant(all_data, btc_close, top10_syms, n_bars, "", func)
            pos, total = rolling_90d_positive(m10["equity"])
            log(
                f"    {param_name}={v:<5}  "
                f"Ret {m10['ret']:+7.1f}%  DD {m10['dd']:5.1f}%  "
                f"Cal {m10['cal']:6.2f}  "
                f"WR {m10['wr']:5.1f}%  Trades {m10['trades']:4d}  "
                f"R:R {m10['realized_rr']:.2f}  "
                f"Roll {pos}/{total}"
            )

    # ═══════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════
    elapsed = time.time() - t0
    log(f"\n{'=' * 120}")
    log(f"Total runtime: {elapsed:.1f}s")
    log(f"Output saved to: {OUT_FILE}")
    log("=" * 120)

    out.close()


if __name__ == "__main__":
    main()
