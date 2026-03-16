"""
Edge Decay Research Script — Minimal Working Example

Shows exactly how to:
1. Load data once
2. Collect trades across 2 strategies
3. Run yearly decomposition
4. Generate edge decay report
"""

import numpy as np
from pathlib import Path
from quantbox.backtest.trade_sim import (
    build_resampled, precompute_all_vr, precompute,
    collect_trades_trailing, portfolio_sim, compute_metrics, bootstrap_pf,
    G_3TF, G_4TF,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

ROOT = Path(__file__).resolve().parents[2]


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data(cfg):
    """Load 15m base, resample to 30m/1h/4h, precompute VR."""
    cache_dir = ROOT / "data" / "cache_extended"
    
    # Load 15m
    data_15m = {}
    max_bars = 0
    for f in sorted(cache_dir.glob("*_15m_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in cfg.symbol_whitelist:
            continue
        
        npz = np.load(f, allow_pickle=False)
        if "close" not in npz:
            continue
        
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        
        data_15m[sym] = {k: npz[k].astype(np.float64) for k in npz.files}
        if len(c) > max_bars:
            max_bars = len(c)
    
    # Align all symbols to same length
    for sym, d in data_15m.items():
        d["_offset"] = max_bars - len(d["close"])
    
    print(f"  Loaded {len(data_15m)} symbols, avg {max_bars} bars")
    
    # Resample
    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    
    # Precompute
    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg.vr_short, cfg.vr_long)])
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)
    
    # Calculate years
    bars_list = [len(d["close"]) for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)
    
    return data_15m, data_30m, data_1h, data_4h, data_years


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Collect Trades
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_trades(cfg, data_15m, data_1h, data_4h, data_30m):
    """Collect D_mom8 and S_d8 trades using trail-aware sims."""
    
    # D_mom8: directional momentum
    vr_trades = collect_trades_trailing(
        cfg, data_15m, data_1h, data_4h, "D_mom8",
        tf_group=list(G_3TF), data_30m=data_30m
    )
    
    # S_d8: straddle (long+short simultaneous)
    strd_trades = collect_trades_trailing(
        cfg, data_15m, data_1h, data_4h, "S_d8",
        tf_group=list(G_4TF), data_30m=data_30m
    )
    
    strd_syms = frozenset(sym for sym, *_ in strd_trades)
    combined = vr_trades + strd_trades
    
    return combined, strd_syms, len(vr_trades), len(strd_trades)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: Yearly Decomposition (Core for Edge Decay Study)
# ══════════════════════════════════════════════════════════════════════════════

def yearly_breakdown(trades, data_years, max_concurrent, straddle_syms=None):
    """Decompose trades by calendar year to detect edge decay."""
    
    # Estimate bars per year
    all_entries = [t[1] for t in trades]
    if not all_entries:
        return {}
    
    min_bar = min(all_entries)
    max_bar = max(all_entries)
    total_range = max_bar - min_bar + 1
    bars_per_year = total_range / data_years if data_years > 0 else 35040
    
    results = {}
    
    for year_offset in range(int(data_years) + 1):
        yr_start = min_bar + int(year_offset * bars_per_year)
        yr_end = min_bar + int((year_offset + 1) * bars_per_year)
        
        # Filter trades in this year
        yr_trades = [(s, e, x, p) for s, e, x, p in trades 
                     if yr_start <= e < yr_end]
        
        if not yr_trades:
            continue
        
        # Portfolio sim this year
        yr_pnl = portfolio_sim(yr_trades, max_concurrent, straddle_syms)
        
        if len(yr_pnl) == 0:
            continue
        
        # Metrics for this year
        m = compute_metrics(yr_pnl, 1.0)
        year_label = 2020 + year_offset
        results[year_label] = m
        
        # Print year summary
        status = "✅" if m["PF"] > 1.0 else "❌"
        print(f"  {year_label}: N={m['N']:4d}  PF={m['PF']:.3f}  "
              f"WR={m['WR']:.1f}%  MaxDD={m['max_dd_pct']:.1f}%  {status}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4: Edge Decay Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_edge_decay(yearly_results):
    """Report edge decay metrics."""
    
    if not yearly_results:
        print("  ERROR: No yearly results to analyze")
        return
    
    years = sorted(yearly_results.keys())
    pf_values = [yearly_results[yr]["PF"] for yr in years]
    wr_values = [yearly_results[yr]["WR"] for yr in years]
    calmar_values = [yearly_results[yr]["calmar"] for yr in years]
    
    # Decay calculations
    if len(years) >= 2:
        pf_decay = (pf_values[-1] / pf_values[0] - 1) * 100
        wr_decay = wr_values[-1] - wr_values[0]
        calmar_decay = (calmar_values[-1] / calmar_values[0] - 1) * 100
        
        print("\n  ┌─ EDGE DECAY SUMMARY ─────────────────────────┐")
        print(f"  │ Period: {years[0]} → {years[-1]} ({len(years)} years)")
        print(f"  │ PF decay:     {pf_values[0]:.3f} → {pf_values[-1]:.3f}  ({pf_decay:+.1f}%)")
        print(f"  │ WR decay:     {wr_values[0]:.1f}% → {wr_values[-1]:.1f}%  ({wr_decay:+.1f}pp)")
        print(f"  │ Calmar decay: {calmar_values[0]:.2f} → {calmar_values[-1]:.2f}  ({calmar_decay:+.1f}%)")
        
        # Estimate annual decay rate (exponential)
        years_elapsed = years[-1] - years[0]
        if years_elapsed > 0:
            annual_pf_decay = (pf_values[-1] / pf_values[0]) ** (1 / years_elapsed) - 1
            print(f"  │ Annual PF decay rate: {annual_pf_decay*100:+.1f}%/year")
        
        print("  └──────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("EDGE DECAY RESEARCH — Minimal Working Example")
    print("=" * 80)
    
    # Step 1: Load config
    cfg = VolStraddleConfig()
    print(f"\n✓ Loaded config: {len(cfg.symbol_whitelist)} symbols")
    print(f"  VR params: sp={cfg.vr_short}, lp={cfg.vr_long}")
    print(f"  Entry thresholds: d_vr={cfg.d_vr_threshold}, s_vr={cfg.s_vr_threshold}")
    
    # Step 2: Load data
    print("\n✓ Loading data...")
    data_15m, data_30m, data_1h, data_4h, data_years = load_all_data(cfg)
    print(f"  Data spans {data_years:.2f} years")
    
    # Step 3: Collect trades
    print("\n✓ Collecting trades...")
    combined, strd_syms, n_vr, n_strd = collect_all_trades(
        cfg, data_15m, data_1h, data_4h, data_30m
    )
    print(f"  VR trades: {n_vr}, Straddle trades: {n_strd}, Total: {len(combined)}")
    
    # Step 4: Full backtest
    print("\n✓ Running full backtest...")
    pnl = portfolio_sim(combined, cfg.max_concurrent, straddle_syms=strd_syms)
    metrics = compute_metrics(pnl, data_years)
    boot = bootstrap_pf(pnl, n_boot=5000)
    
    print(f"  N={metrics['N']:,d}")
    print(f"  PF={metrics['PF']:.3f} (95% CI: {boot['pf_ci_lower']:.3f}, {boot['pf_ci_upper']:.3f})")
    print(f"  WR={metrics['WR']:.1f}%  RR={metrics['RR']:.2f}")
    print(f"  MaxDD={metrics['max_dd_pct']:.1f}%  Calmar={metrics['calmar']:.2f}")
    print(f"  AnnRet={metrics['ann_ret_pct']:.1f}%  P(PF≥1)={boot['P_PF_ge_1']:.1%}")
    
    # Step 5: Yearly breakdown (EDGE DECAY)
    print("\n✓ Yearly breakdown:")
    yearly = yearly_breakdown(combined, data_years, cfg.max_concurrent, strd_syms)
    
    # Step 6: Decay analysis
    analyze_edge_decay(yearly)
    
    print("\n" + "=" * 80)
    print("COMPLETE ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
