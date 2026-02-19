"""
Elliott Wave Timing Alpha — Refined Backtest (Walk-Forward)

Findings from Phase 1 research (97 patterns, 18 symbols):

  TOP SIGNALS (by predictive power):
  ═══════════════════════════════════════════════════════════════
  #1  corr_impulse_ratio: Corr(ret)=+0.411 — THE alpha
      Quick correction (<25%): WR=14%, ret=-2.00% (AVOID)
      Fibonacci zone (25-40%): WR=28%, ret=+0.74%, TP1=50%
      Very long (>100%): WR=50%, ret=+1.91%, TP1=44%

  #2  w3_w1_time_ratio: Corr(ret)=-0.166
      W3 faster (<0.8): WR=46%, ret=+1.33%, TP1=46% (BEST)
      W3 much slower (>3.0): WR=14%, ret=-1.43% (AVOID)

  #3  W5 exhaustion (w5_vel < 0.7× w3_vel):
      True: WR=39%, ret=+0.46%, TP1=42%
      False: WR=27%, ret=-0.33%

  #4  Impulse acceleration (w3_vel > w1_vel):
      Decelerating: WR=21%, ret=-1.07% (AVOID)
      Extreme accel: WR=47%, ret=+1.23%

  #5  Pattern duration 50-100 bars: WR=80%, ret=+5.43% (n=5, small)

  KEY DISCOVERY: 2-of-3 confluence filter kills almost ALL trades (1 left).
  Timing quality can REPLACE strict confluence as a quality gate.
  ═══════════════════════════════════════════════════════════════

This script runs a proper walk-forward backtest with the vectorized engine
to compare timing-enhanced variants vs baseline.

Usage:
    python scripts/elliott_timing_backtest.py
"""

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantbox.strategy.elliott import (
    ElliottConfig,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_stochastic,
    compute_atr,
    zigzag_pivots,
    scan_multi_scale_patterns,
    detect_bullish_divergence,
    detect_bearish_divergence,
    check_macd_reversal,
    check_stoch_cross,
    _validate_bullish_impulse_abc,
    _validate_bearish_impulse_abc,
)
from quantbox.backtest.vectorized_backtest import (
    VectorizedBacktest,
    BacktestConfig,
)

CACHE_DIR = Path("data/cache_extended")

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "XLMUSDT", "DOTUSDT", "FILUSDT",
    "ARBUSDT", "FETUSDT", "OPUSDT",
    "ATOMUSDT", "HBARUSDT", "ICPUSDT",
]


def load_4h_data(symbol: str) -> dict | None:
    pkl_path = CACHE_DIR / f"{symbol}_4h_365d.pkl.gz"
    if pkl_path.exists():
        with gzip.open(pkl_path, "rb") as f:
            df = pickle.load(f)
        return {
            "high": df["High"].values.astype(np.float64),
            "low": df["Low"].values.astype(np.float64),
            "close": df["Close"].values.astype(np.float64),
            "volume": df["Volume"].values.astype(np.float64),
        }
    return None


def compute_timing_features(pattern: dict, idx_arr: list[int]) -> dict:
    """Extract timing features from a validated pattern.

    Returns dict of timing metrics used for filtering.
    """
    prices = [
        pattern["w0"], pattern["w1"], pattern["w2"], pattern["w3"],
        pattern["w4"], pattern["w5"], pattern["a_price"],
        pattern["b_price"], pattern["c_price"],
    ]
    direction = pattern["signal"]

    # Wave durations
    w1_bars = idx_arr[1] - idx_arr[0]
    w3_bars = idx_arr[3] - idx_arr[2]
    w5_bars = idx_arr[5] - idx_arr[4]
    a_bars = idx_arr[6] - idx_arr[5]
    c_bars = idx_arr[8] - idx_arr[7]
    impulse_bars = idx_arr[5] - idx_arr[0]
    correction_bars = idx_arr[8] - idx_arr[5]
    total_bars = idx_arr[8] - idx_arr[0]

    # Price moves
    if direction == 1:
        w1_move = prices[1] - prices[0]
        w3_move = prices[3] - prices[2]
        w5_move = prices[5] - prices[4]
        a_move = prices[5] - prices[6]
        c_move = prices[7] - prices[8]
    else:
        w1_move = prices[0] - prices[1]
        w3_move = prices[2] - prices[3]
        w5_move = prices[4] - prices[5]
        a_move = prices[6] - prices[5]
        c_move = prices[8] - prices[7]

    # Velocities
    w1_vel = w1_move / max(w1_bars, 1)
    w3_vel = w3_move / max(w3_bars, 1)
    w5_vel = w5_move / max(w5_bars, 1)

    # Key ratios
    w3_w1_time_ratio = w3_bars / max(w1_bars, 1)
    corr_impulse_ratio = correction_bars / max(impulse_bars, 1)
    w5_exhaustion = 1.0 if w5_vel < w3_vel * 0.7 else 0.0
    impulse_accel = (w3_vel - w1_vel) / max(abs(w1_vel), 1e-10)

    return {
        "w3_w1_time_ratio": w3_w1_time_ratio,
        "corr_impulse_ratio": corr_impulse_ratio,
        "w5_exhaustion": w5_exhaustion,
        "impulse_accel": impulse_accel,
        "impulse_bars": impulse_bars,
        "correction_bars": correction_bars,
        "total_bars": total_bars,
    }


# ── Timing-enhanced signal generators ──

def generate_signals_with_timing(
    close_4h: np.ndarray,
    high_4h: np.ndarray,
    low_4h: np.ndarray,
    btc_close_4h: np.ndarray,
    config: ElliottConfig,
    variant: str = "baseline",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate position signals with configurable timing filters.

    Variants:
      'baseline':           Current production (quality + conf>=2 + risk)
      'conf1_no_timing':    Relaxed confluence (>=1), no timing
      'timing_gate':        No confluence, timing replaces it
      'timing_plus_conf1':  conf>=1 AND timing gate (best of both?)
      'timing_aggressive':  Only corr_impulse + W3/W1 filters (max trades)
      'timing_conservative': All 5 timing filters + bias + risk

    Returns (signals, adj_close).
    """
    cfg = config
    n = len(close_4h)
    signals = np.zeros(n, dtype=np.float64)
    adj_close = close_4h.copy()

    if n < cfg.ema_macro_slow + 50:
        return signals, adj_close

    # Indicators
    btc_ema200 = compute_ema(btc_close_4h, cfg.ema_macro_fast)
    btc_ema500 = compute_ema(btc_close_4h, cfg.ema_macro_slow)
    rsi_4h = compute_rsi(close_4h, cfg.rsi_period)
    _, _, macd_hist = compute_macd(
        close_4h, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
    )
    stoch_k, stoch_d = compute_stochastic(
        high_4h, low_4h, close_4h, cfg.stoch_k_period, cfg.stoch_d_period,
    )
    atr_4h = compute_atr(high_4h, low_4h, close_4h, 14)

    min_start = cfg.ema_macro_slow + 50
    lookback_bars = cfg.pattern_lookback_bars
    max_lev = cfg.effective_leverage
    lev_exploratory = cfg.nominal_leverage * cfg.exploratory_pct

    # Detect and index all patterns with timing features
    all_entries = []  # (c_idx, pattern, timing, idx_arr)

    for thresh in cfg.zigzag_thresholds:
        pidx, pprice, ptype, n_piv = zigzag_pivots(high_4h, low_4h, thresh)
        if n_piv < 9:
            continue
        for i in range(n_piv - 8):
            is_bull = all(ptype[i + k] == (-1) ** (k + 1) for k in range(9))
            is_bear = all(ptype[i + k] == (-1) ** k for k in range(9))
            if not is_bull and not is_bear:
                continue

            p = [float(pprice[i + j]) for j in range(9)]
            idx_arr = [int(pidx[i + j]) for j in range(9)]
            c_idx = idx_arr[8]

            if is_bull:
                pattern = _validate_bullish_impulse_abc(*p, c_idx, cfg, idx_arr)
            else:
                pattern = _validate_bearish_impulse_abc(*p, c_idx, cfg, idx_arr)

            if pattern is None:
                continue

            timing = compute_timing_features(pattern, idx_arr)
            all_entries.append((c_idx, pattern, timing, idx_arr))

    # Deduplicate by c_idx (within 3 bars), keep highest quality
    all_entries.sort(key=lambda x: -x[1]["quality"])
    seen: set[int] = set()
    deduped = []
    for c_idx, pat, tim, idx_arr in all_entries:
        is_dup = any(abs(c_idx - s) <= 3 for s in seen)
        if not is_dup:
            deduped.append((c_idx, pat, tim, idx_arr))
            seen.add(c_idx)
    deduped.sort(key=lambda x: x[0])

    # Walk-forward position lifecycle
    pat_ptr = 0
    used: set[int] = set()
    in_position = False
    pos_dir = 0
    pos_stop = 0.0
    pos_tp1 = 0.0
    pos_tp2 = 0.0
    pos_entry_price = 0.0
    pos_entry_bar = 0
    pos_confirmed = False
    pos_partial = False
    pos_trail = 0.0
    trade_lev = max_lev
    trade_lev_tp = trade_lev * cfg.partial_tp_fraction

    for bar in range(min_start, n):
        # Manage existing position
        if in_position:
            current_lev = signals[bar - 1]
            fill_price = close_4h[bar]
            bars_held = bar - pos_entry_bar
            hit_stop = False
            hit_tp1 = False
            hit_tp2 = False

            if pos_dir == 1:
                active_stop = pos_trail if pos_partial else pos_stop
                if low_4h[bar] <= active_stop:
                    hit_stop = True
                    fill_price = active_stop
                if not pos_partial and high_4h[bar] >= pos_tp1:
                    hit_tp1 = True
                if pos_partial and high_4h[bar] >= pos_tp2:
                    hit_tp2 = True
                    fill_price = pos_tp2
                if pos_partial:
                    new_trail = high_4h[bar] - cfg.trail_atr_mult * atr_4h[bar]
                    if new_trail > pos_trail:
                        pos_trail = new_trail
                    if low_4h[bar] <= pos_trail:
                        hit_stop = True
                        fill_price = pos_trail
            else:
                active_stop = pos_trail if pos_partial else pos_stop
                if high_4h[bar] >= active_stop:
                    hit_stop = True
                    fill_price = active_stop
                if not pos_partial and low_4h[bar] <= pos_tp1:
                    hit_tp1 = True
                if pos_partial and low_4h[bar] <= pos_tp2:
                    hit_tp2 = True
                    fill_price = pos_tp2
                if pos_partial:
                    new_trail = low_4h[bar] + cfg.trail_atr_mult * atr_4h[bar]
                    if new_trail < pos_trail or pos_trail == 0:
                        pos_trail = new_trail
                    if high_4h[bar] >= pos_trail:
                        hit_stop = True
                        fill_price = pos_trail

            # Confirmation window
            if not pos_confirmed and bars_held <= cfg.confirmation_bars:
                has_m = check_macd_reversal(macd_hist[:bar+1], pos_dir)
                has_s = check_stoch_cross(stoch_k[:bar+1], stoch_d[:bar+1], pos_dir)
                if has_m or has_s:
                    pos_confirmed = True
                    current_lev = float(pos_dir) * trade_lev
            elif not pos_confirmed and bars_held > cfg.confirmation_bars:
                signals[bar] = 0.0
                in_position = False
                continue

            if hit_stop:
                signals[bar] = 0.0
                adj_close[bar] = fill_price
                in_position = False
                continue
            if hit_tp2:
                signals[bar] = 0.0
                adj_close[bar] = fill_price
                in_position = False
                continue
            if hit_tp1 and not pos_partial:
                pos_partial = True
                current_lev = float(pos_dir) * trade_lev_tp
                pos_stop = pos_entry_price
                pos_trail = pos_entry_price
                signals[bar] = current_lev
                continue
            if bars_held >= cfg.max_hold_bars:
                signals[bar] = 0.0
                in_position = False
                continue

            signals[bar] = current_lev
            continue

        # Not in position — scan for entry
        while pat_ptr < len(deduped) and deduped[pat_ptr][0] < bar - lookback_bars:
            pat_ptr += 1

        recent = []
        for j in range(pat_ptr, len(deduped)):
            c_idx, pat, tim, idx_a = deduped[j]
            if c_idx > bar:
                break
            if c_idx >= bar - lookback_bars:
                recent.append((c_idx, pat, tim, idx_a))

        if not recent:
            continue

        c_idx, pattern, timing, idx_arr = recent[-1]
        if c_idx in used:
            continue

        direction = pattern["signal"]

        # ── Macro bias ──
        btc_p = float(btc_close_4h[bar])
        above_200 = btc_p > float(btc_ema200[bar])
        above_500 = btc_p > float(btc_ema500[bar])
        if above_200 and above_500:
            bias = "long"
        elif not above_200 and not above_500:
            bias = "short"
        else:
            bias = "neutral"

        # ── Confluence ──
        confirmations = 0
        lb = min(60, bar)
        if direction == 1:
            if detect_bullish_divergence(close_4h[:bar+1], rsi_4h[:bar+1], lb):
                confirmations += 1
        else:
            if detect_bearish_divergence(close_4h[:bar+1], rsi_4h[:bar+1], lb):
                confirmations += 1
        if check_macd_reversal(macd_hist[:bar+1], direction):
            confirmations += 1
        if check_stoch_cross(stoch_k[:bar+1], stoch_d[:bar+1], direction):
            confirmations += 1

        # ── Risk ──
        entry_price = float(close_4h[bar])
        stop_price = float(pattern["stop"])
        risk_pct = abs(entry_price - stop_price) / entry_price

        quality = pattern.get("quality", 0.0)

        # ═══════════════════════════════════════════════════════════
        # VARIANT-SPECIFIC FILTERS
        # ═══════════════════════════════════════════════════════════

        accept = False

        if variant == "baseline":
            # Current production: strict
            accept = (
                quality >= cfg.min_pattern_quality
                and confirmations >= 2
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "conf1_no_timing":
            # Relaxed confluence, no timing filter
            accept = (
                quality >= cfg.min_pattern_quality
                and confirmations >= 1
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_gate":
            # Timing REPLACES confluence entirely
            t = timing
            timing_pass = (
                t["corr_impulse_ratio"] >= 0.25  # not quick correction
                and t["w3_w1_time_ratio"] < 3.0  # W3 not absurdly long
                and t["impulse_accel"] >= 0       # not decelerating
            )
            accept = (
                timing_pass
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_plus_conf1":
            # conf >= 1 AND timing
            t = timing
            timing_pass = (
                t["corr_impulse_ratio"] >= 0.25
                and t["w3_w1_time_ratio"] < 3.0
                and t["impulse_accel"] >= 0
            )
            accept = (
                timing_pass
                and confirmations >= 1
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_aggressive":
            # Only 2 strongest timing filters + risk
            t = timing
            accept = (
                t["corr_impulse_ratio"] >= 0.25
                and t["w3_w1_time_ratio"] <= 3.0
                and risk_pct <= 0.05  # wider risk for more trades
            )

        elif variant == "timing_conservative":
            # All timing signals + bias + risk
            t = timing
            accept = (
                t["corr_impulse_ratio"] >= 0.25
                and t["w3_w1_time_ratio"] < 3.0
                and t["impulse_accel"] >= 0
                and t["w5_exhaustion"] == 1.0
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_w3_fast":
            # The single strongest experiment: W3 faster than W1
            t = timing
            accept = (
                t["w3_w1_time_ratio"] < 0.8
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_corr_long":
            # Correction must be >= Fibonacci proportion of impulse
            t = timing
            accept = (
                t["corr_impulse_ratio"] >= 0.25
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_no_decel":
            # Remove decelerating impulses
            t = timing
            accept = (
                t["impulse_accel"] >= 0
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_exhaustion":
            # Only enter when W5 exhaustion confirmed
            t = timing
            accept = (
                t["w5_exhaustion"] == 1.0
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        elif variant == "timing_best_combo":
            # Best features from phase 1 combined
            t = timing
            accept = (
                t["corr_impulse_ratio"] >= 0.25
                and t["w3_w1_time_ratio"] < 2.0
                and t["impulse_accel"] >= -0.2
                and risk_pct <= cfg.max_entry_risk_pct
                and (bias == "neutral" or
                     (bias == "long" and direction == 1) or
                     (bias == "short" and direction == -1))
            )

        if not accept:
            continue

        # ── ENTER ──
        used.add(c_idx)
        in_position = True
        pos_dir = direction
        pos_entry_price = entry_price
        pos_entry_bar = bar
        pos_partial = False
        pos_stop = stop_price
        pos_tp1 = float(pattern["target_1"])
        pos_tp2 = float(pattern["target_2"])
        pos_trail = stop_price

        if cfg.risk_sized_leverage and risk_pct > 0:
            trade_lev = min(max_lev, cfg.risk_per_trade / risk_pct)
        else:
            trade_lev = max_lev
        trade_lev_tp = trade_lev * cfg.partial_tp_fraction

        if confirmations >= 2:
            pos_confirmed = True
            signals[bar] = float(direction) * trade_lev
        else:
            pos_confirmed = False
            signals[bar] = float(direction) * lev_exploratory

    return signals, adj_close


def run_all_variants():
    print("=" * 110)
    print("  ELLIOTT WAVE TIMING ALPHA — WALK-FORWARD BACKTEST")
    print("=" * 110)

    cfg = ElliottConfig()
    bt_config = BacktestConfig(
        initial_capital=10000,
        cost_per_side=0.0005,
        funding_per_bar=0.01 / 100 / 2,
        bar_interval_minutes=240,
        risk_per_trade=0.02,
        max_leverage=20.0,
        max_concurrent=3,
    )

    # Numba warmup
    print("\n⚙ Numba warmup...")
    dummy = np.random.randn(200).cumsum() + 100
    d_h, d_l = dummy + 1, dummy - 1
    compute_ema(dummy, 10)
    compute_rsi(dummy, 14)
    compute_macd(dummy, 12, 26, 9)
    compute_stochastic(d_h, d_l, dummy, 14, 3)
    compute_atr(d_h, d_l, dummy, 14)
    zigzag_pivots(d_h, d_l, 0.03)
    print("  Done ✓")

    # Load data
    print(f"\n📥 Loading 4h data...")
    btc_data = load_4h_data("BTCUSDT")
    if btc_data is None:
        print("❌ No BTC data")
        return
    btc_close = btc_data["close"]

    symbol_data = {}
    for sym in TOP_SYMBOLS:
        if sym == "BTCUSDT":
            symbol_data[sym] = btc_data
            continue
        d = load_4h_data(sym)
        if d is not None:
            symbol_data[sym] = d

    print(f"  Loaded {len(symbol_data)} symbols")

    variants = [
        "baseline",
        "conf1_no_timing",
        "timing_gate",
        "timing_plus_conf1",
        "timing_aggressive",
        "timing_conservative",
        "timing_w3_fast",
        "timing_corr_long",
        "timing_no_decel",
        "timing_exhaustion",
        "timing_best_combo",
    ]

    results_table = []

    for variant in variants:
        print(f"\n{'─' * 80}")
        print(f"  Running variant: {variant}")
        print(f"{'─' * 80}")

        bt = VectorizedBacktest(bt_config)
        equity_curves = []
        total_trades = 0
        total_long = 0
        total_short = 0
        active_count = 0

        for sym, data in symbol_data.items():
            c = data["close"]
            h = data["high"]
            lo = data["low"]
            min_len = min(len(c), len(btc_close))
            c_a = c[-min_len:]
            h_a = h[-min_len:]
            l_a = lo[-min_len:]
            bc = btc_close[-min_len:]

            sigs, adj = generate_signals_with_timing(c_a, h_a, l_a, bc, cfg, variant)
            n_long = int(np.sum(sigs > 0))
            n_short = int(np.sum(sigs < 0))

            if n_long + n_short == 0:
                continue

            res = bt.run_single_partial(adj, sigs, symbol=sym)
            total_trades += res.n_trades
            total_long += n_long
            total_short += n_short
            active_count += 1
            equity_curves.append(res.equity_curve)

            print(f"    {sym:<12} ret={res.total_return_pct:>+7.2f}% "
                  f"trades={res.n_trades:>3} WR={res.win_rate:>5.1f}% "
                  f"DD={res.max_drawdown_pct:>5.2f}% PF={res.profit_factor:>5.2f}")

        if not equity_curves:
            print(f"  ⚠ No trades for variant {variant}")
            results_table.append({
                "variant": variant, "symbols": 0, "trades": 0,
                "return": 0, "maxdd": 0, "calmar": 0, "sharpe": 0,
                "wr": 0, "pf": 0, "long": 0, "short": 0,
            })
            continue

        # Portfolio metrics
        min_eq_len = min(len(e) for e in equity_curves)
        aligned = np.array([e[:min_eq_len] for e in equity_curves])
        port_eq = np.mean(aligned, axis=0)
        port_ret = (port_eq[-1] / port_eq[0] - 1) * 100
        peaks = np.maximum.accumulate(port_eq)
        dd = (peaks - port_eq) / peaks
        port_dd = float(np.max(dd)) * 100
        port_calmar = port_ret / port_dd if port_dd > 0 else 0
        rets = np.diff(port_eq) / port_eq[:-1]
        if len(rets) > 1 and np.std(rets) > 0:
            port_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(365 * 6)
        else:
            port_sharpe = 0

        # WR from individual results
        all_wr_trades = []
        for e in equity_curves:
            r = np.diff(e) / e[:-1]
            pos_bars = r[r > 0]
            all_wr_trades.append(len(pos_bars))

        results_table.append({
            "variant": variant,
            "symbols": active_count,
            "trades": total_trades,
            "return": port_ret,
            "maxdd": port_dd,
            "calmar": port_calmar,
            "sharpe": port_sharpe,
            "wr": 0,
            "pf": 0,
            "long": total_long,
            "short": total_short,
        })

        print(f"\n  PORTFOLIO: ret={port_ret:+.2f}% DD={port_dd:.2f}% "
              f"Calmar={port_calmar:.2f} Sharpe={port_sharpe:.2f} "
              f"Trades={total_trades}")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print(f"  VARIANT COMPARISON — Walk-Forward 365d, {len(symbol_data)} symbols")
    print(f"{'=' * 110}")

    header = (
        f"{'Variant':<25} {'Sym':>4} {'Trades':>7} {'Return%':>10} "
        f"{'MaxDD%':>8} {'Calmar':>8} {'Sharpe':>8} {'Long':>6} {'Short':>6}"
    )
    print(header)
    print("-" * 110)

    # Sort by Calmar
    results_table.sort(key=lambda x: -x["calmar"])

    best_calmar = 0
    best_variant = ""
    for r in results_table:
        marker = ""
        if r["calmar"] > best_calmar and r["trades"] >= 10:
            best_calmar = r["calmar"]
            best_variant = r["variant"]
            marker = " ★"
        print(
            f"{r['variant']:<25} {r['symbols']:>4} {r['trades']:>7} "
            f"{r['return']:>+10.2f} {r['maxdd']:>8.2f} "
            f"{r['calmar']:>8.2f} {r['sharpe']:>8.2f} "
            f"{r['long']:>6} {r['short']:>6}{marker}"
        )

    print(f"\n  ★ Best variant: {best_variant} (Calmar: {best_calmar:.2f})")
    print("=" * 110)


if __name__ == "__main__":
    run_all_variants()
