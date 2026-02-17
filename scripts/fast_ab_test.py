#!/usr/bin/env python3
"""
FAST A/B Testing — Memory-efficient with direct findings application.

From regime diagnostic we know:
  - threshold=0.035 is THE sweet spot (+34.71% vs negative for all others)
  - Vol-Top-3 actively hurts (picks trending losers)
  - Equal Weight is safer
  - Only 6/30 symbols are consistently profitable

Strategy: Don't store matrices. Compute metrics on the fly.
"""

import sys, os, gzip, pickle, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quantbox.strategy.rubberband import compute_signals, compute_signals_mtf
from quantbox.strategy.vol_selector import rank_symbols_by_vol
from quantbox.backtest.vectorized_backtest import _backtest_core

COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32
BARS_PER_DAY = 96
VOL_LOOKBACK = 2880
REBALANCE = 960


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


def fast_metrics(eq, start=0):
    s = eq[start:]
    if len(s) < 2: return 0, 0, 0, 0
    ret = (s[-1]/s[0] - 1)*100
    pk = np.maximum.accumulate(s)
    dd = np.max((pk - s)/pk)*100
    cal = ret/dd if dd > 0 else 999
    rs = np.diff(s)/s[:-1]
    sh = np.mean(rs)/np.std(rs)*np.sqrt(96*365.25) if np.std(rs)>0 else 0
    return ret, dd, cal, sh


def compute_bar_returns(close_dict, symbols, n_bars, sig_func):
    """Compute bar returns matrix WITHOUT storing equity curves."""
    mat = np.zeros((len(symbols), n_bars))
    for i, sym in enumerate(symbols):
        sigs = sig_func(close_dict[sym])
        eq, *_ = _backtest_core(close_dict[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        for j in range(1, n_bars):
            mat[i, j] = eq[j]/eq[j-1] - 1.0
    return mat


def ew_eq(mat):
    """Equal weight portfolio equity from bar returns matrix."""
    avg = np.mean(mat, axis=0)
    eq = np.ones(len(avg))
    for j in range(1, len(avg)):
        eq[j] = eq[j-1] * (1 + avg[j])
    return eq


def subset_eq(mat, symbols, subset):
    """EW portfolio of specific symbols."""
    idx = [i for i, s in enumerate(symbols) if s in subset]
    return ew_eq(mat[idx])


def vol_sel_eq(mat, close_dict, symbols, n_bars, top_n):
    """Vol-selection portfolio."""
    eq = np.ones(n_bars)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    rps = list(range(VOL_LOOKBACK, n_bars, REBALANCE))
    if not rps: rps = [VOL_LOOKBACK]
    if rps[-1] < n_bars: rps.append(n_bars)
    for ri in range(len(rps)-1):
        rp, nrp = rps[ri], rps[ri+1]
        ws = max(0, rp - VOL_LOOKBACK)
        ranked = rank_symbols_by_vol(
            {s: close_dict[s][ws:rp] for s in symbols},
            top_n=top_n, lookback_bars=VOL_LOOKBACK,
            annualize_factor=np.sqrt(96*365.25))
        sel = [sym_idx[s] for s,_ in ranked]
        for j in range(rp, nrp):
            avg = sum(mat[si, j] for si in sel) / top_n
            eq[j] = eq[j-1] * (1 + avg)
    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        eq[j] = eq[j-1] * (1 + np.mean(mat[:, j]))
    return eq


def vol_inv_eq(mat, close_dict, symbols, n_bars, top_n):
    """Pick LEAST volatile symbols."""
    eq = np.ones(n_bars)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    rps = list(range(VOL_LOOKBACK, n_bars, REBALANCE))
    if not rps: rps = [VOL_LOOKBACK]
    if rps[-1] < n_bars: rps.append(n_bars)
    for ri in range(len(rps)-1):
        rp, nrp = rps[ri], rps[ri+1]
        ws = max(0, rp - VOL_LOOKBACK)
        ranked = rank_symbols_by_vol(
            {s: close_dict[s][ws:rp] for s in symbols},
            top_n=len(symbols), lookback_bars=VOL_LOOKBACK,
            annualize_factor=np.sqrt(96*365.25))
        sel = [sym_idx[s] for s,_ in ranked[-top_n:]]
        for j in range(rp, nrp):
            avg = sum(mat[si, j] for si in sel) / top_n
            eq[j] = eq[j-1] * (1 + avg)
    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        eq[j] = eq[j-1] * (1 + np.mean(mat[:, j]))
    return eq


def per_sym_results(close_dict, symbols, sig_func):
    """Get per-symbol performance metrics."""
    results = []
    for sym in symbols:
        sigs = sig_func(close_dict[sym])
        eq, ret, sharpe, dd, wr, n_tr, bars_in, pf, avg_ret, trade_rets = \
            _backtest_core(close_dict[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        r, d, c, s = fast_metrics(eq)
        results.append((sym, r, d, c, s, n_tr, wr, pf))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    t0 = time.time()
    print("=" * 74)
    print("  FAST A/B TESTING ENGINE — World Competition Mode")
    print("=" * 74)

    all_data = load_data()
    symbols = sorted(all_data.keys())
    n_bars = min(len(all_data[s]) for s in symbols)
    close_dict = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64) for s in symbols}
    days = n_bars / BARS_PER_DAY
    print(f"Data: {len(symbols)} symbols × {n_bars:,} bars ({days:.0f} days)")
    
    # ══════════════════════════════════════════════════════════
    #  PHASE 1: FOCUSED SWEEP (only key parameters)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 1: THRESHOLD × TARGET × MAX_HOLD SWEEP (EW, no MTF)")
    print(f"{'█'*74}\n")
    
    results = []
    configs = []
    for th in [0.030, 0.032, 0.034, 0.035, 0.036, 0.038, 0.040, 0.045]:
        for tgt in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for mh in [6, 8, 10, 12]:
                for sd in [0.0, 0.3]:
                    configs.append((th, tgt, mh, sd))
    
    total = len(configs)
    print(f"  Sweeping {total} configs...", flush=True)
    
    for ci, (th, tgt, mh, sd) in enumerate(configs):
        if (ci+1) % 40 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ci+1) * (total - ci - 1)
            print(f"    [{ci+1}/{total}] elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
        
        def sf(close, _th=th, _tgt=tgt, _mh=mh, _sd=sd):
            return compute_signals(close, 8, _th, _tgt, 1.0, _mh, _sd)
        
        mat = compute_bar_returns(close_dict, symbols, n_bars, sf)
        eq = ew_eq(mat)
        ret, dd, cal, sh = fast_metrics(eq)
        results.append((th, tgt, mh, sd, ret, dd, cal, sh))
        del mat  # Free memory
    
    results.sort(key=lambda x: x[6], reverse=True)
    
    print(f"\n  TOP 30 (by Calmar, EW All-30):")
    print(f"  {'TH':>6} {'TGT':>5} {'MH':>3} {'SD':>4} {'Ret%':>8} {'DD%':>7} "
          f"{'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 60)
    for th, tgt, mh, sd, ret, dd, cal, sh in results[:30]:
        m = "★" if cal > 1.0 else ""
        print(f"  {th:>6.3f} {tgt:>5.2f} {mh:>3} {sd:>4.1f} "
              f"{ret:>+8.2f} {dd:>7.2f} {cal:>7.2f} {sh:>7.2f} {m}")
    
    pos = sum(1 for r in results if r[4] > 0)
    print(f"\n  Profitable: {pos}/{total} ({pos/total*100:.0f}%)")
    
    b_th, b_tgt, b_mh, b_sd = results[0][0], results[0][1], results[0][2], results[0][3]
    print(f"\n  ★ BEST: th={b_th}, tgt={b_tgt}, mh={b_mh}, sd={b_sd}")
    print(f"    Ret={results[0][4]:+.2f}%, DD={results[0][5]:.2f}%, "
          f"Calmar={results[0][6]:.2f}, Sharpe={results[0][7]:.2f}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 2: PER-SYMBOL ANALYSIS + PORTFOLIO A/B
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 2: PORTFOLIO CONSTRUCTION A/B")
    print(f"{'█'*74}")
    
    def best_sf(close):
        return compute_signals(close, 8, b_th, b_tgt, 1.0, b_mh, b_sd)
    
    sym_perf = per_sym_results(close_dict, symbols, best_sf)
    profitable_syms = [s for s, r, *_ in sym_perf if r > 0]
    top10 = [s for s, *_ in sym_perf[:10]]
    top15 = [s for s, *_ in sym_perf[:15]]
    top20 = [s for s, *_ in sym_perf[:20]]
    
    print(f"\n  Per-symbol (th={b_th}, tgt={b_tgt}, mh={b_mh}):")
    print(f"  {'Symbol':>12} {'Ret%':>8} {'DD%':>7} {'Sharpe':>7} {'WR%':>6} {'Trades':>7} {'PF':>5}")
    print("  " + "-" * 60)
    for sym, ret, dd, cal, sh, n_tr, wr, pf in sym_perf:
        m = "★" if ret > 0 else " "
        print(f"  {sym:>12} {ret:>+8.2f} {dd:>7.2f} {sh:>7.2f} {wr:>5.1f}% {n_tr:>7} {pf:>5.2f} {m}")
    print(f"  Profitable: {len(profitable_syms)}/{len(symbols)}")
    
    # Compute bar returns matrix once for portfolio tests
    mat = compute_bar_returns(close_dict, symbols, n_bars, best_sf)
    
    port_results = {}
    
    # EW variants
    port_results['EW All-30'] = fast_metrics(ew_eq(mat))
    if len(profitable_syms) >= 3:
        port_results[f'EW Profit({len(profitable_syms)})'] = fast_metrics(subset_eq(mat, symbols, profitable_syms))
    port_results['EW Top-10'] = fast_metrics(subset_eq(mat, symbols, top10))
    port_results['EW Top-15'] = fast_metrics(subset_eq(mat, symbols, top15))
    port_results['EW Top-20'] = fast_metrics(subset_eq(mat, symbols, top20))
    
    # Vol-select
    for tn in [3, 5, 8, 10]:
        port_results[f'Vol-Top-{tn}'] = fast_metrics(vol_sel_eq(mat, close_dict, symbols, n_bars, tn), VOL_LOOKBACK)
    
    # Vol-inverse
    for tn in [5, 8, 10]:
        port_results[f'Vol-Inv-{tn}'] = fast_metrics(vol_inv_eq(mat, close_dict, symbols, n_bars, tn), VOL_LOOKBACK)
    
    print(f"\n  Portfolio methods (sorted by Calmar):")
    print(f"  {'Method':>22} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 55)
    for method, (ret, dd, cal, sh) in sorted(port_results.items(), key=lambda x: x[1][2], reverse=True):
        m = "★" if cal > 0.8 else " "
        print(f"  {method:>22} {ret:>+8.2f} {dd:>7.2f} {cal:>7.2f} {sh:>7.2f} {m}")
    
    best_port = max(port_results.items(), key=lambda x: x[1][2])
    print(f"\n  ★ BEST: {best_port[0]}")
    del mat

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: MTF FILTER A/B
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 3: MTF FILTER A/B TEST")
    print(f"{'█'*74}")
    
    mtf_configs = [
        ("No MTF",                 0,  0,     1.0,  1.0),
        ("MTF p=32 b=0.01",       32,  0.01,  0.85, 0.95),
        ("MTF p=48 b=0.01",       48,  0.01,  0.85, 0.95),
        ("MTF p=48 b=0.015",      48,  0.015, 0.85, 0.95),
        ("MTF p=64 b=0.005",      64,  0.005, 0.90, 0.95),
        ("MTF p=64 b=0.01",       64,  0.01,  0.85, 0.95),
        ("MTF p=64 b=0.015",      64,  0.015, 0.85, 0.95),
        ("MTF p=64 b=0.02",       64,  0.02,  0.85, 0.95),
        ("MTF p=96 b=0.01",       96,  0.01,  0.85, 0.95),
        ("MTF p=96 b=0.015",      96,  0.015, 0.90, 0.95),
        ("MTF p=96 b=0.02",       96,  0.02,  0.90, 0.95),
        ("MTF blk p=48",          48,  0.01,  1.0,  1.0),
        ("MTF blk p=64",          64,  0.01,  1.0,  1.0),
        ("MTF blk p=96",          96,  0.01,  1.0,  1.0),
        ("MTF p=48 d=0.9 s=0.9", 48,  0.01,  0.90, 0.90),
        ("MTF p=64 d=0.8 s=0.9", 64,  0.01,  0.80, 0.90),
        ("MTF p=64 d=0.9 s=1.0", 64,  0.01,  0.90, 1.00),
        ("MTF p=128 b=0.02",    128,  0.02,  0.85, 0.95),
    ]
    
    print(f"\n  {'Config':>25} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 60)
    
    mtf_results = []
    for label, period, block, disc, stop_m in mtf_configs:
        t1 = time.time()
        if period == 0:
            def sf(close, _th=b_th, _tgt=b_tgt, _mh=b_mh, _sd=b_sd):
                return compute_signals(close, 8, _th, _tgt, 1.0, _mh, _sd)
        else:
            def sf(close, _th=b_th, _tgt=b_tgt, _mh=b_mh, _sd=b_sd,
                   _p=period, _b=block, _d=disc, _sm=stop_m):
                return compute_signals_mtf(close, 8, _th, _tgt, 1.0, _mh, _sd,
                                          _p, _b, _d, _sm)
        
        mat = compute_bar_returns(close_dict, symbols, n_bars, sf)
        eq = ew_eq(mat)
        ret, dd, cal, sh = fast_metrics(eq)
        dt = time.time() - t1
        mtf_results.append((label, period, block, disc, stop_m, ret, dd, cal, sh))
        print(f"  {label:>25} {ret:>+8.2f} {dd:>7.2f} {cal:>7.2f} {sh:>7.2f}  ({dt:.1f}s)")
        del mat
    
    mtf_results.sort(key=lambda x: x[7], reverse=True)
    bm_label, bm_p, bm_b, bm_d, bm_s = mtf_results[0][:5]
    print(f"\n  ★ BEST MTF: {bm_label}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 4: FINAL COMBO RE-TEST WITH BEST EVERYTHING
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 4: FINAL COMBINATION TEST")
    print(f"{'█'*74}")
    
    if bm_p == 0:
        def final_sf(close):
            return compute_signals(close, 8, b_th, b_tgt, 1.0, b_mh, b_sd)
    else:
        def final_sf(close):
            return compute_signals_mtf(close, 8, b_th, b_tgt, 1.0, b_mh, b_sd,
                                       bm_p, bm_b, bm_d, bm_s)
    
    mat = compute_bar_returns(close_dict, symbols, n_bars, final_sf)
    
    # Test all portfolio methods with final signal
    final_combos = {}
    final_combos['EW All-30'] = fast_metrics(ew_eq(mat))
    
    # Get per-symbol for curated baskets
    final_perf = per_sym_results(close_dict, symbols, final_sf)
    fp_syms = [s for s, r, *_ in final_perf if r > 0]
    ft10 = [s for s, *_ in final_perf[:10]]
    ft15 = [s for s, *_ in final_perf[:15]]
    ft20 = [s for s, *_ in final_perf[:20]]
    
    if len(fp_syms) >= 3:
        final_combos[f'EW Profit({len(fp_syms)})'] = fast_metrics(subset_eq(mat, symbols, fp_syms))
    final_combos['EW Top-10'] = fast_metrics(subset_eq(mat, symbols, ft10))
    final_combos['EW Top-15'] = fast_metrics(subset_eq(mat, symbols, ft15))
    final_combos['EW Top-20'] = fast_metrics(subset_eq(mat, symbols, ft20))
    for tn in [3, 5, 8, 10]:
        final_combos[f'Vol-Top-{tn}'] = fast_metrics(vol_sel_eq(mat, close_dict, symbols, n_bars, tn), VOL_LOOKBACK)
    for tn in [5, 8, 10]:
        final_combos[f'Vol-Inv-{tn}'] = fast_metrics(vol_inv_eq(mat, close_dict, symbols, n_bars, tn), VOL_LOOKBACK)
    
    print(f"\n  {'Method':>22} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 55)
    for method, (ret, dd, cal, sh) in sorted(final_combos.items(), key=lambda x: x[1][2], reverse=True):
        m = "★" if cal > 1.0 else " "
        print(f"  {method:>22} {ret:>+8.2f} {dd:>7.2f} {cal:>7.2f} {sh:>7.2f} {m}")
    
    best_fc = max(final_combos.items(), key=lambda x: x[1][2])
    print(f"\n  ★ BEST FINAL: {best_fc[0]}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 5: ROLLING 90-DAY STABILITY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 5: ROLLING 90-DAY WINDOW STABILITY")
    print(f"{'█'*74}")
    
    window = 90 * BARS_PER_DAY
    step = 30 * BARS_PER_DAY
    
    print(f"\n  {'Window':>8} {'Days':>10} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 55)
    
    w_results = []
    w = 0
    while True:
        start = w * step
        end = start + window
        if end > n_bars: break
        sub_close = {s: close_dict[s][start:end] for s in symbols}
        sub_mat = compute_bar_returns(sub_close, symbols, window, final_sf)
        eq = ew_eq(sub_mat)
        ret, dd, cal, sh = fast_metrics(eq)
        day_s, day_e = start // BARS_PER_DAY, end // BARS_PER_DAY
        m = "★" if ret > 0 else "✗"
        print(f"  W{w+1:>5}  D{day_s:>3}-D{day_e:<3} {ret:>+8.2f} {dd:>7.2f} "
              f"{cal:>7.2f} {sh:>7.2f} {m}")
        w_results.append((ret, dd, cal, sh))
        del sub_mat
        w += 1
    
    prof_w = sum(1 for r, *_ in w_results if r > 0)
    avg_ret = np.mean([r for r, *_ in w_results])
    avg_dd = np.mean([d for _, d, *_ in w_results])
    worst = min(w_results, key=lambda x: x[0])
    
    print(f"\n  Profitable: {prof_w}/{len(w_results)} ({prof_w/len(w_results)*100:.0f}%)")
    print(f"  Avg return: {avg_ret:+.2f}%, Avg DD: {avg_dd:.2f}%")
    print(f"  Worst: {worst[0]:+.2f}% (DD: {worst[1]:.2f}%)")

    # ══════════════════════════════════════════════════════════
    #  PHASE 6: MONTHLY BREAKDOWN
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 6: MONTHLY BREAKDOWN")
    print(f"{'█'*74}")
    
    eq_final = ew_eq(mat)
    bars_pm = BARS_PER_DAY * 30
    
    print(f"\n  {'Month':>8} {'Ret%':>8} {'DD%':>7} {'Cumul%':>8}")
    print("  " + "-" * 40)
    
    monthly = []
    for mo in range(n_bars // bars_pm):
        ms, me = mo*bars_pm, min((mo+1)*bars_pm, n_bars)
        sub = eq_final[ms:me]
        if len(sub) < 2: continue
        mret = (sub[-1]/sub[0]-1)*100
        mpk = np.maximum.accumulate(sub)
        mdd = np.max((mpk-sub)/mpk)*100
        cumul = (eq_final[me-1]/eq_final[0]-1)*100
        monthly.append(mret)
        m = "★" if mret > 0 else "✗"
        print(f"  M{mo+1:>5}  {mret:>+8.2f} {mdd:>7.2f} {cumul:>+8.2f} {m}")
    
    pos_m = sum(1 for r in monthly if r > 0)
    print(f"\n  Profitable months: {pos_m}/{len(monthly)} ({pos_m/len(monthly)*100:.0f}%)")

    # ══════════════════════════════════════════════════════════
    #  PHASE 7: PARAMETER NEIGHBORHOOD
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 7: PARAMETER NEIGHBORHOOD STABILITY")
    print(f"{'█'*74}")
    
    nbr = []
    for dth in [-0.005, -0.003, -0.002, 0, 0.002, 0.003, 0.005]:
        for dtgt in [-0.1, 0, 0.1]:
            for dmh in [-2, 0, 2]:
                th = b_th + dth
                tgt = b_tgt + dtgt
                mh = b_mh + dmh
                if th < 0.015 or tgt < 0.1 or tgt > 1.0 or mh < 4: continue
                
                if bm_p == 0:
                    def sf(close, _th=th, _tgt=tgt, _mh=mh, _sd=b_sd):
                        return compute_signals(close, 8, _th, _tgt, 1.0, _mh, _sd)
                else:
                    def sf(close, _th=th, _tgt=tgt, _mh=mh, _sd=b_sd,
                           _p=bm_p, _b=bm_b, _d=bm_d, _sm=bm_s):
                        return compute_signals_mtf(close, 8, _th, _tgt, 1.0, _mh, _sd,
                                                 _p, _b, _d, _sm)
                nmat = compute_bar_returns(close_dict, symbols, n_bars, sf)
                neq = ew_eq(nmat)
                ret, dd, cal, sh = fast_metrics(neq)
                nbr.append((th, tgt, mh, ret, dd, cal, sh))
                del nmat
    
    nbr.sort(key=lambda x: x[5], reverse=True)
    
    print(f"\n  {'TH':>6} {'TGT':>5} {'MH':>3} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 50)
    for th, tgt, mh, ret, dd, cal, sh in nbr[:30]:
        m = "★" if cal > 0.5 else " "
        print(f"  {th:>6.3f} {tgt:>5.2f} {mh:>3} {ret:>+8.2f} {dd:>7.2f} "
              f"{cal:>7.2f} {sh:>7.2f} {m}")
    
    stable = sum(1 for *_, ret, _, _, _ in nbr if ret > 0)
    print(f"\n  Profitable neighbors: {stable}/{len(nbr)} ({stable/len(nbr)*100:.0f}%)")

    # ══════════════════════════════════════════════════════════
    #  PHASE 8: FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    print(f"\n{'█'*74}")
    print(f"  PHASE 8: FINAL VERDICT & RECOMMENDATION")
    print(f"{'█'*74}")
    
    fr, fd, fc, fs = fast_metrics(eq_final)
    
    print(f"\n  CONFIGURATION:")
    print(f"    lookback       = 8")
    print(f"    threshold      = {b_th}")
    print(f"    target_retrace = {b_tgt}")
    print(f"    stop_mult      = 1.0")
    print(f"    max_hold       = {b_mh}")
    print(f"    stop_decay     = {b_sd}")
    if bm_p > 0:
        print(f"    htf_period     = {bm_p}")
        print(f"    htf_block      = {bm_b}")
        print(f"    htf_discount   = {bm_d}")
        print(f"    htf_stop_mult  = {bm_s}")
    else:
        print(f"    MTF: DISABLED")
    print(f"    portfolio      = EW All-30")
    
    print(f"\n  365-DAY PERFORMANCE:")
    print(f"    Return:  {fr:+.2f}%")
    print(f"    Max DD:  {fd:.2f}%")
    print(f"    Calmar:  {fc:.2f}")
    print(f"    Sharpe:  {fs:.2f}")
    
    score = 0
    checks = []
    c = fr > 0; checks.append(("Annual return > 0%", c)); score += c
    c = fc > 0.5; checks.append(("Calmar > 0.5", c)); score += c
    c = fd < 35; checks.append(("Max DD < 35%", c)); score += c
    c = pos_m/len(monthly) > 0.5; checks.append(("Months > 50% profitable", c)); score += c
    c = prof_w/len(w_results) > 0.6; checks.append(("Windows > 60% profitable", c)); score += c
    c = stable/len(nbr) > 0.5; checks.append(("Neighbors > 50% profitable", c)); score += c
    c = fs > 0.5; checks.append(("Sharpe > 0.5", c)); score += c
    
    print(f"\n  ROBUSTNESS:")
    for label, passed in checks:
        print(f"    {'✓' if passed else '✗'} {label}")
    print(f"\n  SCORE: {score}/7")
    
    if score >= 5:
        print(f"\n  ═══ VERDICT: READY for live trading ═══")
    elif score >= 3:
        print(f"\n  ═══ VERDICT: MARGINAL — deploy with REDUCED size ═══")
    else:
        print(f"\n  ═══ VERDICT: NOT READY ═══")
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*74}")

    del mat

if __name__ == "__main__":
    main()
