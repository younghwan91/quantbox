"""
Elliott Wave Timing Alpha — Round 2: Refine the winner (timing_no_decel)

Round 1 winner: timing_no_decel (impulse_accel >= 0)
  → Calmar 5.60, Sharpe 1.91, Return +6.11%, MaxDD 1.09%, 25 trades

This script tests combinations BUILDING ON that base:
  1. no_decel (base)             — already proven
  2. no_decel + corr_long        — + correction >= 25% of impulse
  3. no_decel + w3_fast          — + W3/W1 time < 2.0
  4. no_decel + exhaustion       — + W5 exhaustion
  5. no_decel + corr + w3        — triple combo
  6. no_decel + quality_bump     — loosen from 40 to 30 quality
  7. no_decel + conf1            — + at least 1 confluence
  8. no_decel_mild               — accel >= -0.2 (allow mild decel)
  9. no_decel + duration_filter  — 20-120 bar impulse
  10. best_combo_refined         — tuned from round 1 data

Also sweeps impulse_accel threshold from -0.5 to +0.5.

Usage:
    python scripts/elliott_timing_backtest_r2.py
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
    """Extract timing features from a validated pattern."""
    prices = [
        pattern["w0"], pattern["w1"], pattern["w2"], pattern["w3"],
        pattern["w4"], pattern["w5"], pattern["a_price"],
        pattern["b_price"], pattern["c_price"],
    ]
    direction = pattern["signal"]

    w1_bars = idx_arr[1] - idx_arr[0]
    w3_bars = idx_arr[3] - idx_arr[2]
    w5_bars = idx_arr[5] - idx_arr[4]
    a_bars = idx_arr[6] - idx_arr[5]
    c_bars = idx_arr[8] - idx_arr[7]
    impulse_bars = idx_arr[5] - idx_arr[0]
    correction_bars = idx_arr[8] - idx_arr[5]

    if direction == 1:
        w1_move = prices[1] - prices[0]
        w3_move = prices[3] - prices[2]
        w5_move = prices[5] - prices[4]
    else:
        w1_move = prices[0] - prices[1]
        w3_move = prices[2] - prices[3]
        w5_move = prices[4] - prices[5]

    w1_vel = w1_move / max(w1_bars, 1)
    w3_vel = w3_move / max(w3_bars, 1)
    w5_vel = w5_move / max(w5_bars, 1)

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
        "w1_bars": w1_bars,
        "w3_bars": w3_bars,
        "w5_bars": w5_bars,
    }


def generate_signals(
    close_4h: np.ndarray,
    high_4h: np.ndarray,
    low_4h: np.ndarray,
    btc_close_4h: np.ndarray,
    config: ElliottConfig,
    variant: str = "no_decel",
    accel_thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate signals for a given variant. Returns (signals, adj_close, n_patterns_found)."""
    cfg = config
    n = len(close_4h)
    signals = np.zeros(n, dtype=np.float64)
    adj_close = close_4h.copy()

    if n < cfg.ema_macro_slow + 50:
        return signals, adj_close, 0

    btc_ema200 = compute_ema(btc_close_4h, cfg.ema_macro_fast)
    btc_ema500 = compute_ema(btc_close_4h, cfg.ema_macro_slow)
    rsi_4h = compute_rsi(close_4h, cfg.rsi_period)
    _, _, macd_hist = compute_macd(close_4h, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period)
    stoch_k, stoch_d = compute_stochastic(high_4h, low_4h, close_4h, cfg.stoch_k_period, cfg.stoch_d_period)
    atr_4h = compute_atr(high_4h, low_4h, close_4h, 14)

    min_start = cfg.ema_macro_slow + 50
    lookback_bars = cfg.pattern_lookback_bars
    max_lev = cfg.effective_leverage
    lev_exploratory = cfg.nominal_leverage * cfg.exploratory_pct

    # Detect all validated patterns
    all_entries = []
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

    # Dedup
    all_entries.sort(key=lambda x: -x[1]["quality"])
    seen: set[int] = set()
    deduped = []
    for c_idx, pat, tim, idx_arr in all_entries:
        if not any(abs(c_idx - s) <= 3 for s in seen):
            deduped.append((c_idx, pat, tim, idx_arr))
            seen.add(c_idx)
    deduped.sort(key=lambda x: x[0])
    n_patterns = len(deduped)

    # Walk-forward
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
        if in_position:
            current_lev = signals[bar - 1]
            fill_price = close_4h[bar]
            bars_held = bar - pos_entry_bar

            if pos_dir == 1:
                active_stop = pos_trail if pos_partial else pos_stop
                if low_4h[bar] <= active_stop:
                    signals[bar] = 0.0; adj_close[bar] = active_stop; in_position = False; continue
                if not pos_partial and high_4h[bar] >= pos_tp1:
                    pos_partial = True
                    current_lev = float(pos_dir) * trade_lev_tp
                    pos_stop = pos_entry_price; pos_trail = pos_entry_price
                    signals[bar] = current_lev; continue
                if pos_partial and high_4h[bar] >= pos_tp2:
                    signals[bar] = 0.0; adj_close[bar] = pos_tp2; in_position = False; continue
                if pos_partial:
                    new_trail = high_4h[bar] - cfg.trail_atr_mult * atr_4h[bar]
                    if new_trail > pos_trail:
                        pos_trail = new_trail
                    if low_4h[bar] <= pos_trail:
                        signals[bar] = 0.0; adj_close[bar] = pos_trail; in_position = False; continue
            else:
                active_stop = pos_trail if pos_partial else pos_stop
                if high_4h[bar] >= active_stop:
                    signals[bar] = 0.0; adj_close[bar] = active_stop; in_position = False; continue
                if not pos_partial and low_4h[bar] <= pos_tp1:
                    pos_partial = True
                    current_lev = float(pos_dir) * trade_lev_tp
                    pos_stop = pos_entry_price; pos_trail = pos_entry_price
                    signals[bar] = current_lev; continue
                if pos_partial and low_4h[bar] <= pos_tp2:
                    signals[bar] = 0.0; adj_close[bar] = pos_tp2; in_position = False; continue
                if pos_partial:
                    new_trail = low_4h[bar] + cfg.trail_atr_mult * atr_4h[bar]
                    if new_trail < pos_trail or pos_trail == 0:
                        pos_trail = new_trail
                    if high_4h[bar] >= pos_trail:
                        signals[bar] = 0.0; adj_close[bar] = pos_trail; in_position = False; continue

            if not pos_confirmed and bars_held <= cfg.confirmation_bars:
                has_m = check_macd_reversal(macd_hist[:bar+1], pos_dir)
                has_s = check_stoch_cross(stoch_k[:bar+1], stoch_d[:bar+1], pos_dir)
                if has_m or has_s:
                    pos_confirmed = True
                    current_lev = float(pos_dir) * trade_lev
            elif not pos_confirmed and bars_held > cfg.confirmation_bars:
                signals[bar] = 0.0; in_position = False; continue

            if bars_held >= cfg.max_hold_bars:
                signals[bar] = 0.0; in_position = False; continue

            signals[bar] = current_lev
            continue

        # Not in position
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
        quality = pattern.get("quality", 0.0)
        entry_price = float(close_4h[bar])
        stop_price = float(pattern["stop"])
        risk_pct = abs(entry_price - stop_price) / entry_price

        # Macro bias
        btc_p = float(btc_close_4h[bar])
        if btc_p > float(btc_ema200[bar]) and btc_p > float(btc_ema500[bar]):
            bias = "long"
        elif btc_p < float(btc_ema200[bar]) and btc_p < float(btc_ema500[bar]):
            bias = "short"
        else:
            bias = "neutral"
        bias_ok = (bias == "neutral" or
                   (bias == "long" and direction == 1) or
                   (bias == "short" and direction == -1))

        # Confluence
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

        t = timing

        # ═══════ VARIANT FILTERS ═══════
        accept = False
        min_q = cfg.min_pattern_quality  # 40

        if variant == "no_decel":
            # Round 1 winner: just filter out decelerating impulses
            accept = (
                t["impulse_accel"] >= accel_thresh
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_corr_long":
            accept = (
                t["impulse_accel"] >= accel_thresh
                and t["corr_impulse_ratio"] >= 0.25
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_w3_fast":
            accept = (
                t["impulse_accel"] >= accel_thresh
                and t["w3_w1_time_ratio"] < 2.0
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_exhaustion":
            accept = (
                t["impulse_accel"] >= accel_thresh
                and t["w5_exhaustion"] == 1.0
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_triple":
            accept = (
                t["impulse_accel"] >= accel_thresh
                and t["corr_impulse_ratio"] >= 0.25
                and t["w3_w1_time_ratio"] < 2.0
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_quality30":
            accept = (
                t["impulse_accel"] >= accel_thresh
                and quality >= 30
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_conf1":
            accept = (
                t["impulse_accel"] >= accel_thresh
                and confirmations >= 1
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_mild":
            # Allow mild deceleration (threshold sweep)
            accept = (
                t["impulse_accel"] >= accel_thresh
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "no_decel_duration":
            accept = (
                t["impulse_accel"] >= accel_thresh
                and 20 <= t["impulse_bars"] <= 120
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )
        elif variant == "best_refined":
            # Fine-tuned combo of top features
            accept = (
                t["impulse_accel"] >= 0
                and t["corr_impulse_ratio"] >= 0.20
                and t["w3_w1_time_ratio"] < 2.5
                and risk_pct <= cfg.max_entry_risk_pct
                and bias_ok
            )

        if not accept:
            continue

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

    return signals, adj_close, n_patterns


def run_variant(
    variant: str,
    symbol_data: dict,
    btc_close: np.ndarray,
    cfg: ElliottConfig,
    bt_config: BacktestConfig,
    accel_thresh: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Run a single variant and return metrics."""
    bt = VectorizedBacktest(bt_config)
    equity_curves = []
    total_trades = 0
    active_count = 0
    total_patterns = 0
    per_symbol_results = []

    for sym, data in symbol_data.items():
        c = data["close"]
        h = data["high"]
        lo = data["low"]
        min_len = min(len(c), len(btc_close))
        c_a = c[-min_len:]
        h_a = h[-min_len:]
        l_a = lo[-min_len:]
        bc = btc_close[-min_len:]

        sigs, adj, n_pat = generate_signals(c_a, h_a, l_a, bc, cfg, variant, accel_thresh)
        total_patterns += n_pat
        n_entries = int(np.sum(np.diff(np.sign(sigs)) != 0) // 2) + (1 if sigs[np.argmax(np.abs(sigs))] != 0 else 0)

        if np.any(sigs != 0):
            res = bt.run_single_partial(adj, sigs, symbol=sym)
            total_trades += res.n_trades
            active_count += 1
            equity_curves.append(res.equity_curve)
            per_symbol_results.append((sym, res))

            if verbose:
                print(f"    {sym:<12} ret={res.total_return_pct:>+7.2f}% "
                      f"trades={res.n_trades:>3} WR={res.win_rate:>5.1f}% "
                      f"DD={res.max_drawdown_pct:>5.2f}% PF={res.profit_factor:>5.2f} "
                      f"pats={n_pat}")

    if not equity_curves:
        return {"variant": variant, "accel": accel_thresh, "symbols": 0,
                "trades": 0, "return": 0, "maxdd": 0, "calmar": 0,
                "sharpe": 0, "patterns": total_patterns}

    min_eq_len = min(len(e) for e in equity_curves)
    aligned = np.array([e[:min_eq_len] for e in equity_curves])
    port_eq = np.mean(aligned, axis=0)
    port_ret = (port_eq[-1] / port_eq[0] - 1) * 100
    peaks = np.maximum.accumulate(port_eq)
    dd = (peaks - port_eq) / peaks
    port_dd = float(np.max(dd)) * 100
    port_calmar = port_ret / port_dd if port_dd > 0 else 0
    rets = np.diff(port_eq) / port_eq[:-1]
    port_sharpe = (
        np.mean(rets) / np.std(rets) * np.sqrt(365 * 6)
        if len(rets) > 1 and np.std(rets) > 0 else 0
    )

    return {
        "variant": variant,
        "accel": accel_thresh,
        "symbols": active_count,
        "trades": total_trades,
        "return": port_ret,
        "maxdd": port_dd,
        "calmar": port_calmar,
        "sharpe": port_sharpe,
        "patterns": total_patterns,
    }


def main():
    print("=" * 110)
    print("  ELLIOTT TIMING ALPHA — ROUND 2: REFINE THE WINNER")
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
        print("❌ No BTC data"); return
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

    # ═══════════════════════════════════════════════════════════
    # PART 1: Variant grid
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print("  PART 1: VARIANT GRID (accel_thresh=0.0)")
    print(f"{'=' * 110}")

    variant_names = [
        "no_decel",
        "no_decel_corr_long",
        "no_decel_w3_fast",
        "no_decel_exhaustion",
        "no_decel_triple",
        "no_decel_quality30",
        "no_decel_conf1",
        "no_decel_duration",
        "best_refined",
    ]

    results = []
    for v in variant_names:
        print(f"\n{'─' * 80}")
        print(f"  {v}")
        print(f"{'─' * 80}")
        r = run_variant(v, symbol_data, btc_close, cfg, bt_config, accel_thresh=0.0)
        results.append(r)
        print(f"  → ret={r['return']:+.2f}% DD={r['maxdd']:.2f}% "
              f"Calmar={r['calmar']:.2f} Sharpe={r['sharpe']:.2f} "
              f"trades={r['trades']}")

    # ═══════════════════════════════════════════════════════════
    # PART 2: Accel threshold sweep for best variant
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print("  PART 2: ACCELERATION THRESHOLD SWEEP (no_decel base)")
    print(f"{'=' * 110}")

    sweep_results = []
    for at in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        r = run_variant("no_decel", symbol_data, btc_close, cfg, bt_config,
                        accel_thresh=at, verbose=False)
        sweep_results.append(r)
        print(f"  accel>={at:>+5.1f}  trades={r['trades']:>3}  "
              f"ret={r['return']:>+7.2f}%  DD={r['maxdd']:>5.2f}%  "
              f"Calmar={r['calmar']:>6.2f}  Sharpe={r['sharpe']:>5.2f}")

    # Also sweep the best combo variants
    print(f"\n  ── Sweep for best_refined ──")
    for at in [-0.3, -0.1, 0.0, 0.1, 0.3]:
        r = run_variant("best_refined", symbol_data, btc_close, cfg, bt_config,
                        accel_thresh=at, verbose=False)
        sweep_results.append(r)
        print(f"  accel>={at:>+5.1f}  trades={r['trades']:>3}  "
              f"ret={r['return']:>+7.2f}%  DD={r['maxdd']:>5.2f}%  "
              f"Calmar={r['calmar']:>6.2f}  Sharpe={r['sharpe']:>5.2f}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    all_results = results + sweep_results
    print(f"\n{'=' * 110}")
    print(f"  ROUND 2 — FULL SUMMARY (sorted by Calmar)")
    print(f"{'=' * 110}")

    header = (
        f"{'Variant':<25} {'Accel':>6} {'Sym':>4} {'Trades':>7} "
        f"{'Return%':>10} {'MaxDD%':>8} {'Calmar':>8} {'Sharpe':>8}"
    )
    print(header)
    print("-" * 110)

    all_results.sort(key=lambda x: -x["calmar"])
    for r in all_results:
        marker = " ★" if r == all_results[0] and r["trades"] >= 8 else ""
        print(
            f"{r['variant']:<25} {r['accel']:>+6.1f} {r['symbols']:>4} "
            f"{r['trades']:>7} {r['return']:>+10.2f} {r['maxdd']:>8.2f} "
            f"{r['calmar']:>8.2f} {r['sharpe']:>8.2f}{marker}"
        )

    # Find best with minimum trade count
    valid = [r for r in all_results if r["trades"] >= 10]
    if valid:
        best = max(valid, key=lambda x: x["calmar"])
        print(f"\n  ★ Best (≥10 trades): {best['variant']} "
              f"accel>={best['accel']:.1f} → Calmar={best['calmar']:.2f} "
              f"Sharpe={best['sharpe']:.2f} Return={best['return']:+.2f}% "
              f"DD={best['maxdd']:.2f}% Trades={best['trades']}")

    valid5 = [r for r in all_results if r["trades"] >= 5]
    if valid5:
        best5 = max(valid5, key=lambda x: x["calmar"])
        print(f"  ★ Best (≥5 trades):  {best5['variant']} "
              f"accel>={best5['accel']:.1f} → Calmar={best5['calmar']:.2f} "
              f"Sharpe={best5['sharpe']:.2f} Return={best5['return']:+.2f}% "
              f"DD={best5['maxdd']:.2f}% Trades={best5['trades']}")

    print("=" * 110)


if __name__ == "__main__":
    main()
