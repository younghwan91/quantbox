#!/usr/bin/env python3
"""
Robust Configuration Search — Find parameters that work across 365 days.

Key findings from regime diagnostic:
  1. threshold=0.035 is the clear winner (+34.71%, Calmar 1.24)
  2. Vol-Top-3 selection HURTS — concentrates into trending losers
  3. Only 6/30 symbols are consistently profitable 
  4. Equal weight is safer than vol-selection

This script:
  A. Fine-grained sweep around threshold=0.035 (the discovered sweet spot)
  B. Tests portfolio construction methods (EW, vol-inverse, curated basket)
  C. Rolling 90-day stability across ALL windows
  D. Monthly consistency check
  E. Final recommendation with realistic projections
"""

import sys
import os
import gzip
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quantbox.strategy.rubberband import compute_signals, compute_signals_mtf
from quantbox.strategy.vol_selector import rank_symbols_by_vol
from quantbox.backtest.vectorized_backtest import _backtest_core

COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32
BARS_PER_DAY = 96
VOL_LOOKBACK = 2880   # 30 days
REBALANCE = 960        # 10 days


def load_data():
    cache_dir = Path("data/cache_extended")
    all_data = {}
    for f in sorted(cache_dir.glob("*_15m_365d.pkl.gz")):
        symbol = f.name.split("_15m_")[0]
        with gzip.open(f, "rb") as fh:
            df = pickle.load(fh)
        if "Close" in df.columns and len(df) >= 200:
            all_data[symbol] = df
    return all_data


def compute_metrics(eq, start_idx=0):
    sub = eq[start_idx:]
    if len(sub) < 2:
        return {'ret': 0, 'dd': 0, 'calmar': 0, 'sharpe': 0}
    ret = (sub[-1] / sub[0] - 1.0) * 100
    peak = np.maximum.accumulate(sub)
    dd = np.max((peak - sub) / peak) * 100
    calmar = ret / dd if dd > 0 else 999.0
    rets = np.diff(sub) / sub[:-1]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25) if np.std(rets) > 0 else 0.0
    return {'ret': ret, 'dd': dd, 'calmar': calmar, 'sharpe': sharpe}


def backtest_symbol(close, sig_func):
    """Backtest single symbol, return equity and trade returns."""
    sigs = sig_func(close)
    eq, ret, sharpe, dd, wr, n_trades, bars_in, pf, avg_ret, trade_rets = (
        _backtest_core(close, sigs, COST_PER_SIDE, FUNDING_PER_BAR)
    )
    return eq, trade_rets, n_trades, wr, pf


def portfolio_equal_weight(close_dict, symbols, n_bars, sig_func):
    """Equal-weight portfolio across all symbols."""
    bar_rets = {}
    for sym in symbols:
        eq, _, _, _, _ = backtest_symbol(close_dict[sym], sig_func)
        r = np.zeros(n_bars)
        for j in range(1, n_bars):
            r[j] = eq[j] / eq[j-1] - 1.0
        bar_rets[sym] = r
    
    portfolio_eq = np.ones(n_bars)
    n_sym = len(symbols)
    for j in range(1, n_bars):
        avg_r = sum(bar_rets[s][j] for s in symbols) / n_sym
        portfolio_eq[j] = portfolio_eq[j-1] * (1.0 + avg_r)
    return portfolio_eq


def portfolio_vol_topn(close_dict, symbols, n_bars, sig_func, top_n):
    """Vol-selection top-N portfolio."""
    bar_rets = {}
    for sym in symbols:
        eq, _, _, _, _ = backtest_symbol(close_dict[sym], sig_func)
        r = np.zeros(n_bars)
        for j in range(1, n_bars):
            r[j] = eq[j] / eq[j-1] - 1.0
        bar_rets[sym] = r
    
    portfolio_eq = np.ones(n_bars)
    rebalance_pts = list(range(VOL_LOOKBACK, n_bars, REBALANCE))
    if not rebalance_pts:
        rebalance_pts = [VOL_LOOKBACK]
    if rebalance_pts[-1] < n_bars:
        rebalance_pts.append(n_bars)

    for ri in range(len(rebalance_pts) - 1):
        rp = rebalance_pts[ri]
        next_rp = rebalance_pts[ri + 1]
        ws = max(0, rp - VOL_LOOKBACK)
        ranked = rank_symbols_by_vol(
            {s: close_dict[s][ws:rp] for s in symbols},
            top_n=top_n, lookback_bars=VOL_LOOKBACK,
            annualize_factor=np.sqrt(96 * 365.25),
        )
        selected = [sym for sym, _ in ranked]
        for j in range(rp, next_rp):
            avg_r = sum(bar_rets[s][j] for s in selected) / top_n
            portfolio_eq[j] = portfolio_eq[j-1] * (1.0 + avg_r)

    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        avg_r = sum(bar_rets[s][j] for s in symbols) / len(symbols)
        portfolio_eq[j] = portfolio_eq[j-1] * (1.0 + avg_r)

    return portfolio_eq


def portfolio_vol_inverse(close_dict, symbols, n_bars, sig_func, top_n):
    """INVERSE vol-selection: pick LEAST volatile symbols.
    Hypothesis: low-vol symbols are more mean-reverting, less trending."""
    bar_rets = {}
    for sym in symbols:
        eq, _, _, _, _ = backtest_symbol(close_dict[sym], sig_func)
        r = np.zeros(n_bars)
        for j in range(1, n_bars):
            r[j] = eq[j] / eq[j-1] - 1.0
        bar_rets[sym] = r
    
    portfolio_eq = np.ones(n_bars)
    rebalance_pts = list(range(VOL_LOOKBACK, n_bars, REBALANCE))
    if not rebalance_pts:
        rebalance_pts = [VOL_LOOKBACK]
    if rebalance_pts[-1] < n_bars:
        rebalance_pts.append(n_bars)

    for ri in range(len(rebalance_pts) - 1):
        rp = rebalance_pts[ri]
        next_rp = rebalance_pts[ri + 1]
        ws = max(0, rp - VOL_LOOKBACK)
        ranked = rank_symbols_by_vol(
            {s: close_dict[s][ws:rp] for s in symbols},
            top_n=len(symbols), lookback_bars=VOL_LOOKBACK,
            annualize_factor=np.sqrt(96 * 365.25),
        )
        # Invert: pick the BOTTOM (least volatile)
        selected = [sym for sym, _ in ranked[-top_n:]]
        for j in range(rp, next_rp):
            avg_r = sum(bar_rets[s][j] for s in selected) / top_n
            portfolio_eq[j] = portfolio_eq[j-1] * (1.0 + avg_r)

    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        avg_r = sum(bar_rets[s][j] for s in symbols) / len(symbols)
        portfolio_eq[j] = portfolio_eq[j-1] * (1.0 + avg_r)

    return portfolio_eq


def portfolio_curated(close_dict, n_bars, sig_func, basket):
    """Equal-weight among curated symbol basket."""
    bar_rets = {}
    for sym in basket:
        eq, _, _, _, _ = backtest_symbol(close_dict[sym], sig_func)
        r = np.zeros(n_bars)
        for j in range(1, n_bars):
            r[j] = eq[j] / eq[j-1] - 1.0
        bar_rets[sym] = r
    
    portfolio_eq = np.ones(n_bars)
    n_sym = len(basket)
    for j in range(1, n_bars):
        avg_r = sum(bar_rets[s][j] for s in basket) / n_sym
        portfolio_eq[j] = portfolio_eq[j-1] * (1.0 + avg_r)
    return portfolio_eq


def rolling_window_analysis(close_dict, symbols, n_bars, sig_func, portfolio_func, 
                            window_bars, step_bars, label=""):
    """Run strategy on rolling windows and check consistency."""
    results = []
    n_windows = (n_bars - window_bars) // step_bars + 1
    for w in range(n_windows):
        start = w * step_bars
        end = start + window_bars
        if end > n_bars:
            break
        sub_close = {s: close_dict[s][start:end] for s in symbols}
        eq = portfolio_func(sub_close, symbols, window_bars, sig_func)
        m = compute_metrics(eq)
        results.append(m)
    return results


def main():
    print("=" * 70)
    print("  ROBUST CONFIG SEARCH — Finding stable parameters for 365 days")
    print("=" * 70)

    all_data = load_data()
    symbols = sorted(all_data.keys())
    n_bars = min(len(all_data[s]) for s in symbols)
    close_dict = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64) for s in symbols}
    days = n_bars / BARS_PER_DAY
    print(f"Loaded {len(symbols)} symbols, {n_bars:,} bars ({days:.0f} days)\n")

    # ══════════════════════════════════════════════════════════════
    #  A. FINE-GRAINED PARAMETER SWEEP AROUND th=0.035
    # ══════════════════════════════════════════════════════════════
    print(f"{'█' * 70}")
    print(f"  A. FOCUSED SWEEP — Threshold neighborhood + key params")
    print(f"{'█' * 70}")

    sweep_results = []
    
    # Focused: threshold x target x max_hold x stop_decay (reasonable ranges)
    thresholds = [0.028, 0.030, 0.032, 0.035, 0.038, 0.040, 0.045]
    targets = [0.3, 0.4, 0.5, 0.6, 0.7]
    max_holds = [6, 8, 10, 12]
    stop_decays = [0.0, 0.2, 0.3, 0.5]
    lookbacks = [6, 8]

    total = len(thresholds) * len(targets) * len(max_holds) * len(stop_decays) * len(lookbacks)
    print(f"  Testing {total} configs (EW portfolio)...")
    
    count = 0
    for lb in lookbacks:
        for th in thresholds:
            for tgt in targets:
                for mh in max_holds:
                    for sd in stop_decays:
                        count += 1
                        if count % 100 == 0:
                            print(f"    [{count}/{total}]...", flush=True)
                        
                        def sig_func(close, _lb=lb, _th=th, _tgt=tgt, _mh=mh, _sd=sd):
                            return compute_signals(close, _lb, _th, _tgt, 1.0, _mh, _sd)
                        
                        eq = portfolio_equal_weight(close_dict, symbols, n_bars, sig_func)
                        m = compute_metrics(eq)
                        sweep_results.append((lb, th, tgt, mh, sd, m))
    
    sweep_results.sort(key=lambda x: x[5]['calmar'], reverse=True)
    
    print(f"\n  Top 30 configs (by Calmar, Equal Weight):")
    print(f"  {'LB':>3} {'TH':>6} {'TGT':>5} {'MH':>3} {'SD':>4} "
          f"{'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 65)
    for lb, th, tgt, mh, sd, m in sweep_results[:30]:
        print(f"  {lb:>3} {th:>6.3f} {tgt:>5.2f} {mh:>3} {sd:>4.1f} "
              f"{m['ret']:>+8.2f} {m['dd']:>7.2f} {m['calmar']:>7.2f} {m['sharpe']:>7.2f}")
    
    pos_configs = sum(1 for r in sweep_results if r[5]['ret'] > 0)
    print(f"\n  Profitable: {pos_configs}/{total} ({pos_configs/total*100:.1f}%)")
    
    # Identify the best candidate
    best = sweep_results[0]
    best_lb, best_th, best_tgt, best_mh, best_sd = best[0], best[1], best[2], best[3], best[4]
    print(f"\n  ★ BEST: lb={best_lb}, th={best_th}, tgt={best_tgt}, mh={best_mh}, sd={best_sd}")
    print(f"    Ret: {best[5]['ret']:+.2f}%, DD: {best[5]['dd']:.2f}%, "
          f"Calmar: {best[5]['calmar']:.2f}, Sharpe: {best[5]['sharpe']:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  B. PORTFOLIO CONSTRUCTION — EW vs Vol vs Inverse-Vol vs Curated
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  B. PORTFOLIO CONSTRUCTION — Which method is best?")
    print(f"{'█' * 70}")

    def best_sig(close):
        return compute_signals(close, best_lb, best_th, best_tgt, 1.0, best_mh, best_sd)
    
    # Per-symbol analysis with the best config
    sym_results = {}
    for sym in symbols:
        eq, tr, n_tr, wr, pf = backtest_symbol(close_dict[sym], best_sig)
        m = compute_metrics(eq)
        sym_results[sym] = (m, n_tr, wr, pf)
    
    # Sort by return
    sorted_syms = sorted(sym_results.items(), key=lambda x: x[1][0]['ret'], reverse=True)
    
    print(f"\n  Per-symbol results (best config: lb={best_lb}, th={best_th}, tgt={best_tgt}):")
    print(f"  {'Symbol':>12} {'Ret%':>8} {'DD%':>7} {'Sharpe':>7} {'WR%':>6} {'Trades':>7}")
    print("  " + "-" * 55)
    profitable_syms = []
    for sym, (m, n_tr, wr, pf) in sorted_syms:
        marker = "★" if m['ret'] > 0 else " "
        print(f"  {sym:>12} {m['ret']:>+8.2f} {m['dd']:>7.2f} {m['sharpe']:>7.2f} "
              f"{wr:>5.1f}% {n_tr:>7} {marker}")
        if m['ret'] > 0:
            profitable_syms.append(sym)
    
    print(f"\n  Profitable symbols: {len(profitable_syms)}/{len(symbols)}")

    # Test different portfolio methods
    methods = {}
    
    # 1. Equal Weight (all 30)
    eq_ew = portfolio_equal_weight(close_dict, symbols, n_bars, best_sig)
    methods['EW All-30'] = compute_metrics(eq_ew)
    
    # 2. Vol Top-3 (current production)
    eq_vt3 = portfolio_vol_topn(close_dict, symbols, n_bars, best_sig, 3)
    methods['Vol Top-3'] = compute_metrics(eq_vt3, VOL_LOOKBACK)
    
    # 3. Vol Top-5
    eq_vt5 = portfolio_vol_topn(close_dict, symbols, n_bars, best_sig, 5)
    methods['Vol Top-5'] = compute_metrics(eq_vt5, VOL_LOOKBACK)
    
    # 4. Vol Top-10 
    eq_vt10 = portfolio_vol_topn(close_dict, symbols, n_bars, best_sig, 10)
    methods['Vol Top-10'] = compute_metrics(eq_vt10, VOL_LOOKBACK)
    
    # 5. Vol-INVERSE Top-5 (least volatile)
    eq_vi5 = portfolio_vol_inverse(close_dict, symbols, n_bars, best_sig, 5)
    methods['Vol-Inv Top-5'] = compute_metrics(eq_vi5, VOL_LOOKBACK)
    
    # 6. Vol-INVERSE Top-10
    eq_vi10 = portfolio_vol_inverse(close_dict, symbols, n_bars, best_sig, 10)
    methods['Vol-Inv Top-10'] = compute_metrics(eq_vi10, VOL_LOOKBACK)
    
    # 7. Curated basket (only profitable symbols from full-year analysis)
    if len(profitable_syms) >= 3:
        eq_cur = portfolio_curated(close_dict, n_bars, best_sig, profitable_syms)
        methods[f'Curated ({len(profitable_syms)})'] = compute_metrics(eq_cur)
    
    # 8. Top-10 profitable only (if enough)
    top10_syms = [s for s, _ in sorted_syms[:10]]
    eq_t10 = portfolio_curated(close_dict, n_bars, best_sig, top10_syms)
    methods['Top-10 syms'] = compute_metrics(eq_t10)

    # 9. EW Top-15
    top15_syms = [s for s, _ in sorted_syms[:15]]
    eq_t15 = portfolio_curated(close_dict, n_bars, best_sig, top15_syms)
    methods['Top-15 syms'] = compute_metrics(eq_t15)

    print(f"\n  Portfolio method comparison:")
    print(f"  {'Method':>18} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 50)
    for method, m in sorted(methods.items(), key=lambda x: x[1]['calmar'], reverse=True):
        print(f"  {method:>18} {m['ret']:>+8.2f} {m['dd']:>7.2f} "
              f"{m['calmar']:>7.2f} {m['sharpe']:>7.2f}")
    
    # Find best portfolio method
    best_method = max(methods.items(), key=lambda x: x[1]['calmar'])
    print(f"\n  ★ BEST METHOD: {best_method[0]}")

    # ══════════════════════════════════════════════════════════════
    #  C. MTF FILTER — Does it help or hurt with the new threshold?
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  C. MTF FILTER — Testing with best config")
    print(f"{'█' * 70}")

    mtf_configs = [
        ("No MTF", 0, 0, 1.0, 1.0),
        ("MTF p=48 b=0.01", 48, 0.01, 0.85, 0.95),
        ("MTF p=64 b=0.01", 64, 0.01, 0.85, 0.95),
        ("MTF p=64 b=0.005", 64, 0.005, 0.90, 0.95),
        ("MTF p=64 b=0.015", 64, 0.015, 0.85, 0.95),
        ("MTF p=96 b=0.01", 96, 0.01, 0.85, 0.95),
        ("MTF p=96 b=0.015", 96, 0.015, 0.90, 0.95),
        ("MTF block only p=64", 64, 0.01, 1.0, 1.0),
        ("MTF block only p=96", 96, 0.01, 1.0, 1.0),
        ("MTF p=64 b=0.02", 64, 0.02, 0.85, 0.95),
    ]

    print(f"\n  {'Config':>25} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 60)
    
    best_mtf = None
    best_mtf_calmar = -999
    
    for label, period, block, disc, stop_m in mtf_configs:
        if period == 0:
            def sf(close, _lb=best_lb, _th=best_th, _tgt=best_tgt, _mh=best_mh, _sd=best_sd):
                return compute_signals(close, _lb, _th, _tgt, 1.0, _mh, _sd)
        else:
            def sf(close, _lb=best_lb, _th=best_th, _tgt=best_tgt, _mh=best_mh, _sd=best_sd,
                   _p=period, _b=block, _d=disc, _sm=stop_m):
                return compute_signals_mtf(close, _lb, _th, _tgt, 1.0, _mh, _sd,
                                          _p, _b, _d, _sm)
        
        eq = portfolio_equal_weight(close_dict, symbols, n_bars, sf)
        m = compute_metrics(eq)
        print(f"  {label:>25} {m['ret']:>+8.2f} {m['dd']:>7.2f} "
              f"{m['calmar']:>7.2f} {m['sharpe']:>7.2f}")
        
        if m['calmar'] > best_mtf_calmar:
            best_mtf_calmar = m['calmar']
            best_mtf = (label, period, block, disc, stop_m)

    print(f"\n  ★ BEST MTF: {best_mtf[0]}")

    # ══════════════════════════════════════════════════════════════
    #  D. ROLLING 90-DAY WINDOWS — Stability test
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  D. ROLLING 90-DAY WINDOW STABILITY")
    print(f"{'█' * 70}")
    
    # Build the FINAL signal function with best params + best MTF
    final_period, final_block, final_disc, final_stop_m = best_mtf[1], best_mtf[2], best_mtf[3], best_mtf[4]
    
    if final_period == 0:
        def final_sig(close):
            return compute_signals(close, best_lb, best_th, best_tgt, 1.0, best_mh, best_sd)
    else:
        def final_sig(close):
            return compute_signals_mtf(close, best_lb, best_th, best_tgt, 1.0, best_mh, best_sd,
                                       final_period, final_block, final_disc, final_stop_m)

    window = 90 * BARS_PER_DAY  # 8640 bars
    step = 30 * BARS_PER_DAY    # 30-day step

    print(f"\n  Window: 90 days, Step: 30 days")
    print(f"  {'Window':>8} {'Days':>10} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 55)
    
    w_results = []
    w = 0
    while True:
        start = w * step
        end = start + window
        if end > n_bars:
            break
        sub_close = {s: close_dict[s][start:end] for s in symbols}
        eq = portfolio_equal_weight(sub_close, symbols, window, final_sig)
        m = compute_metrics(eq)
        day_start = start // BARS_PER_DAY
        day_end = end // BARS_PER_DAY
        print(f"  W{w+1:>5}  D{day_start:>3}-D{day_end:<3} {m['ret']:>+8.2f} {m['dd']:>7.2f} "
              f"{m['calmar']:>7.2f} {m['sharpe']:>7.2f}")
        w_results.append(m)
        w += 1

    profitable_windows = sum(1 for m in w_results if m['ret'] > 0)
    print(f"\n  Profitable windows: {profitable_windows}/{len(w_results)} "
          f"({profitable_windows/len(w_results)*100:.0f}%)")
    avg_ret = np.mean([m['ret'] for m in w_results])
    avg_dd = np.mean([m['dd'] for m in w_results])
    print(f"  Avg window return: {avg_ret:+.2f}%")
    print(f"  Avg window DD: {avg_dd:.2f}%")
    
    worst_window = min(w_results, key=lambda m: m['ret'])
    print(f"  Worst window: {worst_window['ret']:+.2f}% (DD: {worst_window['dd']:.2f}%)")

    # ══════════════════════════════════════════════════════════════
    #  E. MONTHLY BREAKDOWN — Full period
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  E. MONTHLY BREAKDOWN — Full 365-day period")
    print(f"{'█' * 70}")
    
    eq_final = portfolio_equal_weight(close_dict, symbols, n_bars, final_sig)
    bars_per_month = BARS_PER_DAY * 30
    
    print(f"\n  {'Month':>8} {'Ret%':>8} {'DD%':>7} {'Cumul%':>8}")
    print("  " + "-" * 40)
    
    monthly_rets = []
    for mo in range(n_bars // bars_per_month):
        ms = mo * bars_per_month
        me = min((mo + 1) * bars_per_month, n_bars)
        sub = eq_final[ms:me]
        if len(sub) < 2:
            continue
        mret = (sub[-1] / sub[0] - 1.0) * 100
        mpeak = np.maximum.accumulate(sub)
        mdd = np.max((mpeak - sub) / mpeak) * 100
        cumul = (eq_final[me - 1] / eq_final[0] - 1.0) * 100
        monthly_rets.append(mret)
        marker = "★" if mret > 0 else "✗"
        print(f"  M{mo+1:>5}  {mret:>+8.2f} {mdd:>7.2f} {cumul:>+8.2f} {marker}")
    
    pos_months = sum(1 for r in monthly_rets if r > 0)
    print(f"\n  Profitable months: {pos_months}/{len(monthly_rets)} "
          f"({pos_months/len(monthly_rets)*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════
    #  F. PARAMETER NEIGHBORHOOD STABILITY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  F. PARAMETER NEIGHBORHOOD — Is the best config stable?")
    print(f"{'█' * 70}")

    # Perturb each param slightly from best
    neighborhood = []
    for dlb in [-2, 0, 2]:
        for dth in [-0.003, 0, 0.003]:
            for dtgt in [-0.1, 0, 0.1]:
                lb = best_lb + dlb
                th = best_th + dth
                tgt = best_tgt + dtgt
                if lb < 4 or th < 0.015 or tgt < 0.1 or tgt > 1.0:
                    continue
                def sf(close, _lb=lb, _th=th, _tgt=tgt, _mh=best_mh, _sd=best_sd):
                    return compute_signals(close, _lb, _th, _tgt, 1.0, _mh, _sd)
                eq = portfolio_equal_weight(close_dict, symbols, n_bars, sf)
                m = compute_metrics(eq)
                neighborhood.append((lb, th, tgt, m))
    
    neighborhood.sort(key=lambda x: x[3]['calmar'], reverse=True)
    
    print(f"\n  {'LB':>3} {'TH':>6} {'TGT':>5} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 50)
    for lb, th, tgt, m in neighborhood:
        marker = "★" if m['calmar'] > 0.5 else " "
        print(f"  {lb:>3} {th:>6.3f} {tgt:>5.2f} {m['ret']:>+8.2f} {m['dd']:>7.2f} "
              f"{m['calmar']:>7.2f} {m['sharpe']:>7.2f} {marker}")
    
    stable_count = sum(1 for _, _, _, m in neighborhood if m['ret'] > 0)
    print(f"\n  Profitable neighbors: {stable_count}/{len(neighborhood)} "
          f"({stable_count/len(neighborhood)*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════
    #  G. FINAL VERDICT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  G. FINAL VERDICT & RECOMMENDATION")
    print(f"{'█' * 70}")
    
    m_final = compute_metrics(eq_final)
    
    print(f"\n  BEST CONFIG FOUND:")
    print(f"    Signal: lb={best_lb}, th={best_th}, tgt={best_tgt}, mh={best_mh}, sd={best_sd}")
    print(f"    MTF: {best_mtf[0]}")
    print(f"    Portfolio: Equal Weight All-30")
    print(f"    Full-year: Ret={m_final['ret']:+.2f}%, DD={m_final['dd']:.2f}%, "
          f"Calmar={m_final['calmar']:.2f}, Sharpe={m_final['sharpe']:.2f}")
    
    # Robustness scoring
    score = 0
    checks = []
    
    # 1. Full-year return > 0
    c1 = m_final['ret'] > 0
    checks.append(("Full-year return > 0%", c1))
    if c1: score += 1
    
    # 2. Calmar > 0.5
    c2 = m_final['calmar'] > 0.5
    checks.append(("Calmar > 0.5", c2))
    if c2: score += 1
    
    # 3. Max DD < 35%
    c3 = m_final['dd'] < 35
    checks.append(("Max DD < 35%", c3))
    if c3: score += 1
    
    # 4. >50% profitable monthly
    c4 = pos_months / len(monthly_rets) > 0.5 if monthly_rets else False
    checks.append(("Profitable months > 50%", c4))
    if c4: score += 1
    
    # 5. >60% profitable rolling windows
    c5 = profitable_windows / len(w_results) > 0.6 if w_results else False
    checks.append(("Profitable windows > 60%", c5))
    if c5: score += 1
    
    # 6. >50% parameter neighbors profitable
    c6 = stable_count / len(neighborhood) > 0.5 if neighborhood else False
    checks.append(("Param neighbors > 50% profitable", c6))
    if c6: score += 1
    
    # 7. Sharpe > 0.5
    c7 = m_final['sharpe'] > 0.5
    checks.append(("Sharpe > 0.5", c7))
    if c7: score += 1
    
    print(f"\n  ROBUSTNESS CHECKS:")
    for label, passed in checks:
        marker = "✓" if passed else "✗"
        print(f"    {marker} {label}")
    print(f"\n  SCORE: {score}/7")
    
    if score >= 5:
        print(f"\n  ═══ VERDICT: READY for live trading ═══")
        print(f"  Expected monthly return: {m_final['ret']/12:+.2f}%")
        print(f"  Expected max drawdown: {m_final['dd']:.1f}%")
        print(f"  Parameter retuning: Every 3-6 months")
    elif score >= 3:
        print(f"\n  ═══ VERDICT: MARGINAL — deploy with REDUCED size ═══")
        print(f"  Use 50% of intended capital")
        print(f"  Monitor weekly, retune monthly")
    else:
        print(f"\n  ═══ VERDICT: NOT READY — strategy needs redesign ═══")
        print(f"  Do NOT deploy with real capital")

    print(f"\n{'=' * 70}")
    print(f"  SEARCH COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
