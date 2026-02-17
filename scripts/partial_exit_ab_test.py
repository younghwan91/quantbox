#!/usr/bin/env python3
"""
PARTIAL EXIT + BREAK-EVEN STOP + R:R-GATED LEVERAGE — A/B Test

v4.3 Capital Preservation Framework:
  "Execute a partial 50% liquidation upon any profit to secure capital,
   followed by an immediate transition to a break-even stop-loss to
   eliminate downside risk... 손익비가 높아야 레버리지를 쓴다."

Variants tested:
  A: Baseline v4.2 (BTC directional + confidence leverage)
  B: Partial exit + BE stop only (same leverage)
  C: R:R-gated leverage only (no partial exit)
  D: Partial exit + R:R-gated leverage (COMBINED)

For each variant: All-30 EW, Top-10 EW, per-symbol breakdown,
rolling 90-day stability, drawdown analysis.
"""

import sys, os, gzip, pickle, time, warnings
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
from quantbox.strategy.rubberband import (
    compute_signals,
    compute_signals_with_btc_directional,
    compute_signals_with_partial_exit,
    compute_signals_partial_exit_btc_directional,
    apply_confidence_leverage,
    apply_rr_gated_leverage,
)
from quantbox.backtest.vectorized_backtest import (
    _backtest_core,
    _backtest_core_partial,
)

# ── Constants ──
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32
BARS_PER_DAY = 96

# ── v4.2 parameters ──
LOOKBACK = 8
THRESHOLD = 0.045
TARGET = 0.3
STOP = 1.0
MAX_HOLD = 10
STOP_DECAY = 0.0
BTC_MOM_PERIOD = 32

# ── Leverage parameters ──
BASE_LEV = 3.0
LEV_SCALE = 10.0
MAX_LEV = 7.0

# ── R:R gate parameter sweep ──
MIN_RR_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

# ── Curated Top-10 ──
TOP_10 = [
    'FETUSDT', 'FILUSDT', 'ICPUSDT', 'HBARUSDT', 'DOTUSDT',
    'OPUSDT', 'SHIBUSDT', 'DOGEUSDT', 'LTCUSDT', 'ATOMUSDT',
]

OUTPUT = "logs/partial_exit_ab.txt"
out = open(OUTPUT, "w")


def log(msg=""):
    print(msg)
    out.write(msg + "\n")
    out.flush()


def load_data():
    cache_dir = Path("data/cache_extended")
    all_data = {}
    for f in sorted(cache_dir.glob("*_15m_365d.pkl.gz")):
        symbol = f.name.split("_15m_")[0]
        with gzip.open(f, "rb") as fh:
            df = pickle.load(fh)
        if "Close" in df.columns and len(df) >= 200:
            all_data[symbol] = df["Close"].values.astype(np.float64)
    return all_data


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


def ew_portfolio(equities: list[np.ndarray], n_bars: int) -> np.ndarray:
    """Equal-weight portfolio from per-asset equity curves."""
    n = len(equities)
    eq = np.ones(n_bars)
    for j in range(1, n_bars):
        avg_ret = 0.0
        for e in equities:
            avg_ret += e[j] / e[j - 1] - 1.0
        avg_ret /= n
        eq[j] = eq[j - 1] * (1.0 + avg_ret)
    return eq


def rolling_90d(eq, bars_per_window=96 * 90):
    """Rolling 90-day windows: number profitable / total."""
    n = len(eq)
    wins = 0
    total = 0
    start = 0
    while start + bars_per_window <= n:
        end = start + bars_per_window
        ret = eq[end - 1] / eq[start] - 1.0
        total += 1
        if ret > 0:
            wins += 1
        start += bars_per_window // 3  # overlap by 2/3
    return wins, total


# ══════════════════════════════════════════════════════════════
#  VARIANT GENERATORS
# ══════════════════════════════════════════════════════════════

def variant_a_baseline(close, btc_close):
    """Baseline v4.2: BTC directional + confidence leverage."""
    sigs = compute_signals_with_btc_directional(
        close, btc_close, LOOKBACK, THRESHOLD, TARGET, STOP,
        MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
    )
    lev_sigs = apply_confidence_leverage(
        close, sigs, LOOKBACK, THRESHOLD, BASE_LEV, LEV_SCALE, MAX_LEV,
    )
    # Use original backtest core (same as v4.2)
    return _backtest_core(close, lev_sigs, COST_PER_SIDE, FUNDING_PER_BAR)


def variant_b_partial_exit(close, btc_close):
    """Partial exit + BE stop + BTC directional + confidence leverage."""
    sigs = compute_signals_partial_exit_btc_directional(
        close, btc_close, LOOKBACK, THRESHOLD, TARGET, STOP,
        MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
    )
    lev_sigs = apply_confidence_leverage(
        close, sigs, LOOKBACK, THRESHOLD, BASE_LEV, LEV_SCALE, MAX_LEV,
    )
    # Must use partial backtest core (delta-based costs)
    return _backtest_core_partial(close, lev_sigs, COST_PER_SIDE, FUNDING_PER_BAR)


def variant_c_rr_leverage(close, btc_close, min_rr=0.30):
    """R:R-gated leverage + BTC directional (no partial exit)."""
    sigs = compute_signals_with_btc_directional(
        close, btc_close, LOOKBACK, THRESHOLD, TARGET, STOP,
        MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
    )
    lev_sigs = apply_rr_gated_leverage(
        close, sigs, LOOKBACK, THRESHOLD, TARGET, STOP,
        COST_PER_SIDE, BASE_LEV, LEV_SCALE, MAX_LEV, min_rr,
    )
    return _backtest_core(close, lev_sigs, COST_PER_SIDE, FUNDING_PER_BAR)


def variant_d_combined(close, btc_close, min_rr=0.30):
    """COMBINED: Partial exit + R:R-gated leverage + BTC directional."""
    sigs = compute_signals_partial_exit_btc_directional(
        close, btc_close, LOOKBACK, THRESHOLD, TARGET, STOP,
        MAX_HOLD, STOP_DECAY, BTC_MOM_PERIOD,
    )
    lev_sigs = apply_rr_gated_leverage(
        close, sigs, LOOKBACK, THRESHOLD, TARGET, STOP,
        COST_PER_SIDE, BASE_LEV, LEV_SCALE, MAX_LEV, min_rr,
    )
    return _backtest_core_partial(close, lev_sigs, COST_PER_SIDE, FUNDING_PER_BAR)


def run_variant(close_dict, btc_close, symbols, variant_func, label, **kwargs):
    """Run a variant across symbols and return per-symbol + portfolio metrics."""
    n_bars = min(len(close_dict[s]) for s in symbols)
    equities = []
    per_sym = []

    for sym in symbols:
        c = close_dict[sym][:n_bars]
        b = btc_close[:n_bars]
        eq, ret, sharpe, dd, wr, n_trades, bars_in, pf, avg_ret, trade_rets = \
            variant_func(c, b, **kwargs)
        equities.append(eq)
        r, d, cal, sh = fast_metrics(eq)
        per_sym.append({
            'symbol': sym, 'ret': r, 'dd': d, 'calmar': cal,
            'sharpe': sh, 'n_trades': n_trades, 'win_rate': wr,
            'profit_factor': pf,
        })

    port_eq = ew_portfolio(equities, n_bars)
    p_ret, p_dd, p_cal, p_sh = fast_metrics(port_eq)
    r90_wins, r90_total = rolling_90d(port_eq)

    return {
        'label': label,
        'port_ret': p_ret,
        'port_dd': p_dd,
        'calmar': p_cal,
        'sharpe': p_sh,
        'rolling_90d': f"{r90_wins}/{r90_total}",
        'r90_pct': r90_wins / r90_total * 100 if r90_total > 0 else 0,
        'total_trades': sum(s['n_trades'] for s in per_sym),
        'avg_win_rate': np.mean([s['win_rate'] for s in per_sym]),
        'pct_sym_pos': sum(1 for s in per_sym if s['ret'] > 0) / len(per_sym) * 100,
        'per_sym': per_sym,
        'equity': port_eq,
    }


def print_results_table(results: list[dict], title: str):
    """Print comparison table."""
    log(f"\n{'═' * 80}")
    log(f"  {title}")
    log(f"{'═' * 80}")
    log(f"  {'Variant':<40} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7} "
        f"{'WR%':>6} {'90d':>7} {'Trades':>7}")
    log(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")
    for r in results:
        star = " ★" if r is max(results, key=lambda x: x['calmar']) else ""
        log(f"  {r['label']:<40} {r['port_ret']:>+8.2f} {r['port_dd']:>7.2f} "
            f"{r['calmar']:>7.2f} {r['sharpe']:>7.2f} "
            f"{r['avg_win_rate']:>6.1f} {r['rolling_90d']:>7} "
            f"{r['total_trades']:>7}{star}")
    log("")


def print_per_symbol(results: list[dict]):
    """Print per-symbol comparison across variants."""
    if not results:
        return
    symbols = [s['symbol'] for s in results[0]['per_sym']]
    labels = [r['label'][:20] for r in results]

    log(f"\n{'─' * 80}")
    log(f"  PER-SYMBOL RETURNS (% 365d)")
    log(f"{'─' * 80}")
    header = f"  {'Symbol':<12}" + "".join(f" {l:>16}" for l in labels)
    log(header)
    log(f"  {'-'*12}" + " ".join(f"{'-'*16}" for _ in labels))

    for i, sym in enumerate(symbols):
        row = f"  {sym:<12}"
        for r in results:
            ret = r['per_sym'][i]['ret']
            row += f" {ret:>+16.2f}"
        log(row)


def main():
    t0 = time.time()
    log("=" * 80)
    log("  PARTIAL EXIT + BE STOP + R:R-GATED LEVERAGE — A/B TEST")
    log("  v4.3 Capital Preservation Framework")
    log("=" * 80)

    all_data = load_data()
    symbols = sorted(all_data.keys())
    n_bars = min(len(all_data[s]) for s in symbols)
    btc_close = all_data.get('BTCUSDT')
    if btc_close is None:
        log("ERROR: BTCUSDT data not found!")
        return
    btc_close = btc_close[:n_bars]
    close_dict = {s: all_data[s][:n_bars] for s in symbols}
    days = n_bars / BARS_PER_DAY
    log(f"\nData: {len(symbols)} symbols × {n_bars:,} bars ({days:.0f} days)")
    top10_present = [s for s in TOP_10 if s in symbols]
    log(f"Top-10 present: {len(top10_present)}/{len(TOP_10)}: {top10_present}")

    # Numba warmup
    log("\nWarming up Numba...")
    _warmup = np.linspace(100, 105, 200)
    compute_signals_with_partial_exit(_warmup, 8, 0.045, 0.3, 1.0, 10, 0.0)
    compute_signals_partial_exit_btc_directional(
        _warmup, _warmup, 8, 0.045, 0.3, 1.0, 10, 0.0, 32,
    )
    apply_rr_gated_leverage(
        _warmup, np.ones(200), 8, 0.045, 0.3, 1.0, 0.0004, 3.0, 10.0, 7.0, 0.3,
    )
    _backtest_core_partial(_warmup, np.ones(200), 0.0004, 0.0001 / 32)
    log("Warmup complete.")

    # ══════════════════════════════════════════════════════════
    #  PHASE 1: CORE A/B — All-30
    # ══════════════════════════════════════════════════════════
    log(f"\n{'█' * 80}")
    log(f"  PHASE 1: CORE A/B TEST — All {len(symbols)} Symbols (EW)")
    log(f"{'█' * 80}")

    results_all = []

    # A: Baseline
    log("\n  Running A: Baseline v4.2...")
    ra = run_variant(close_dict, btc_close, symbols, variant_a_baseline,
                     "A: Baseline v4.2")
    results_all.append(ra)

    # B: Partial exit only
    log("  Running B: Partial exit + BE stop...")
    rb = run_variant(close_dict, btc_close, symbols, variant_b_partial_exit,
                     "B: Partial exit + BE stop")
    results_all.append(rb)

    # C: R:R-gated leverage (sweep min_rr)
    log("  Running C: R:R-gated leverage sweep...")
    best_c = None
    for min_rr in MIN_RR_VALUES:
        rc = run_variant(close_dict, btc_close, symbols, variant_c_rr_leverage,
                         f"C: R:R-gate (min={min_rr:.2f})", min_rr=min_rr)
        if best_c is None or rc['calmar'] > best_c['calmar']:
            best_c = rc
        log(f"    min_rr={min_rr:.2f}: ret={rc['port_ret']:+.2f}% "
            f"dd={rc['port_dd']:.2f}% cal={rc['calmar']:.2f}")
    results_all.append(best_c)

    # D: Combined (sweep min_rr)
    log("  Running D: Combined (partial + R:R) sweep...")
    best_d = None
    for min_rr in MIN_RR_VALUES:
        rd = run_variant(close_dict, btc_close, symbols, variant_d_combined,
                         f"D: Combined (min={min_rr:.2f})", min_rr=min_rr)
        if best_d is None or rd['calmar'] > best_d['calmar']:
            best_d = rd
        log(f"    min_rr={min_rr:.2f}: ret={rd['port_ret']:+.2f}% "
            f"dd={rd['port_dd']:.2f}% cal={rd['calmar']:.2f}")
    results_all.append(best_d)

    print_results_table(results_all, "ALL SYMBOLS — Equal Weight")

    # ══════════════════════════════════════════════════════════
    #  PHASE 2: TOP-10 ONLY
    # ══════════════════════════════════════════════════════════
    log(f"\n{'█' * 80}")
    log(f"  PHASE 2: TOP-10 A/B TEST")
    log(f"{'█' * 80}")

    if len(top10_present) < 5:
        log("  SKIP: Not enough Top-10 symbols present.")
    else:
        t10_close = {s: close_dict[s] for s in top10_present}
        results_t10 = []

        log("\n  Running A: Baseline v4.2 (Top-10)...")
        ra10 = run_variant(t10_close, btc_close, top10_present,
                           variant_a_baseline, "A: Baseline v4.2")
        results_t10.append(ra10)

        log("  Running B: Partial exit + BE stop (Top-10)...")
        rb10 = run_variant(t10_close, btc_close, top10_present,
                           variant_b_partial_exit, "B: Partial exit + BE stop")
        results_t10.append(rb10)

        # C: Best min_rr from All-30
        best_min_rr_c = float(best_c['label'].split('=')[1].rstrip(')'))
        log(f"  Running C: R:R-gate (min={best_min_rr_c:.2f}) (Top-10)...")
        rc10 = run_variant(t10_close, btc_close, top10_present,
                           variant_c_rr_leverage,
                           f"C: R:R-gate (min={best_min_rr_c:.2f})",
                           min_rr=best_min_rr_c)
        results_t10.append(rc10)

        best_min_rr_d = float(best_d['label'].split('=')[1].rstrip(')'))
        log(f"  Running D: Combined (min={best_min_rr_d:.2f}) (Top-10)...")
        rd10 = run_variant(t10_close, btc_close, top10_present,
                           variant_d_combined,
                           f"D: Combined (min={best_min_rr_d:.2f})",
                           min_rr=best_min_rr_d)
        results_t10.append(rd10)

        print_results_table(results_t10, "TOP-10 — Equal Weight")
        print_per_symbol(results_t10)

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: DRAWDOWN ANALYSIS
    # ══════════════════════════════════════════════════════════
    log(f"\n{'█' * 80}")
    log(f"  PHASE 3: DRAWDOWN & CAPITAL PRESERVATION ANALYSIS")
    log(f"{'█' * 80}")

    for result_set, title in [(results_all, "All Symbols"), (results_t10 if len(top10_present) >= 5 else [], "Top-10")]:
        if not result_set:
            continue
        log(f"\n  {title}:")
        log(f"  {'Variant':<40} {'MaxDD%':>7} {'AvgDD%':>7} {'DDDuration':>10} {'Recoveries':>10}")
        log(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*10} {'-'*10}")

        for r in result_set:
            eq = r['equity']
            pk = np.maximum.accumulate(eq)
            dds = (pk - eq) / pk * 100
            max_dd = np.max(dds)
            avg_dd = np.mean(dds[dds > 0]) if np.any(dds > 0) else 0

            # Max DD duration
            in_dd = False
            dd_start = 0
            max_dur = 0
            n_recoveries = 0
            for j in range(len(eq)):
                if dds[j] > 0.5:  # In drawdown (>0.5%)
                    if not in_dd:
                        dd_start = j
                        in_dd = True
                else:
                    if in_dd:
                        dur = j - dd_start
                        if dur > max_dur:
                            max_dur = dur
                        n_recoveries += 1
                        in_dd = False
            max_dur_hours = max_dur * 15 / 60

            log(f"  {r['label']:<40} {max_dd:>7.2f} {avg_dd:>7.2f} "
                f"{max_dur_hours:>9.1f}h {n_recoveries:>10}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 4: ROLLING STABILITY
    # ══════════════════════════════════════════════════════════
    log(f"\n{'█' * 80}")
    log(f"  PHASE 4: ROLLING 90-DAY STABILITY")
    log(f"{'█' * 80}")

    for result_set, title in [(results_all, "All Symbols"), (results_t10 if len(top10_present) >= 5 else [], "Top-10")]:
        if not result_set:
            continue
        log(f"\n  {title}:")
        for r in result_set:
            eq = r['equity']
            window = 96 * 90
            periods = []
            start = 0
            while start + window <= len(eq):
                end = start + window
                ret = (eq[end - 1] / eq[start] - 1.0) * 100
                periods.append(ret)
                start += window // 3
            pos = sum(1 for p in periods if p > 0)
            neg = len(periods) - pos
            if periods:
                min_ret = min(periods)
                max_ret = max(periods)
                log(f"  {r['label']:<40} {pos}/{len(periods)} profitable "
                    f"({pos/len(periods)*100:.0f}%) "
                    f"min={min_ret:+.1f}% max={max_ret:+.1f}%")
            else:
                log(f"  {r['label']:<40} No 90-day windows")

    # ══════════════════════════════════════════════════════════
    #  FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    log(f"\n{'█' * 80}")
    log(f"  FINAL VERDICT")
    log(f"{'█' * 80}")

    # Compare on Top-10 if available, else All
    cmp = results_t10 if (len(top10_present) >= 5 and results_t10) else results_all
    baseline = cmp[0]

    for r in cmp[1:]:
        ret_delta = r['port_ret'] - baseline['port_ret']
        dd_delta = r['port_dd'] - baseline['port_dd']
        cal_delta = r['calmar'] - baseline['calmar']
        sh_delta = r['sharpe'] - baseline['sharpe']

        verdict = "BETTER" if cal_delta > 0 and dd_delta <= 0 else \
                  "MIXED" if cal_delta > 0 else "WORSE"

        log(f"\n  {r['label']} vs Baseline:")
        log(f"    Return: {ret_delta:+.2f}%  DD: {dd_delta:+.2f}%  "
            f"Calmar: {cal_delta:+.2f}  Sharpe: {sh_delta:+.2f}")
        log(f"    Verdict: {verdict}")

    # Recommend
    best = max(cmp, key=lambda x: x['calmar'])
    log(f"\n  ★★★ WINNER: {best['label']}")
    log(f"      Return: {best['port_ret']:+.2f}% | DD: {best['port_dd']:.2f}% | "
        f"Calmar: {best['calmar']:.2f} | Sharpe: {best['sharpe']:.2f}")
    log(f"      Rolling 90d: {best['rolling_90d']} | "
        f"Win Rate: {best['avg_win_rate']:.1f}%")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.1f}s")
    log(f"\n  Results saved to {OUTPUT}")
    out.close()


if __name__ == "__main__":
    main()
