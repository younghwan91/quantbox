#!/usr/bin/env python3
"""r77: Slippage sensitivity analysis — flat vs tiered vs conservative.

Current backtest uses flat 0.08% round-trip for all 60 symbols.
Reality:
  - Thin alts (ICX, STORJ, ZRX, ...) can have 0.15-0.30% slippage during
    squeeze signals (vol already elevated → spreads wider).
  - Large caps (BTC, ETH, SOL, ...) fill near mid-price: 0.08% is fair.
  - Mid-cap alts (AAVE, RUNE, SAND, ...): 0.10-0.15% is more realistic.

Three slippage models compared:
  flat         = 0.08% all symbols (current production assumption)
  tiered       = per-symbol tier (HIGH=0.08%, MED=0.12%, LOW=0.16%)
  conservative = tiered × 1.5   (HIGH=0.12%, MED=0.18%, LOW=0.24%)

Phase 0: Config + symbol universe validation
Phase 1: Baseline sanity-check vs canonical (N≈6207, PF≈1.563)
Phase 2: Three-model sweep for D_mom8 and S_d8
Phase 3: Print EV sensitivity table

Usage:
    uv run python research/active/r76_r80/r77_slippage_sensitivity.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    _COST_HIGH_LIQ,
    _COST_LOW_LIQ,
    _COST_MED_LIQ,
    _HIGH_LIQ_SYMS,
    _LOW_LIQ_SYMS,
    bootstrap_pf,
    build_resampled,
    collect_trades,
    compute_metrics,
    get_symbol_slippage,
    portfolio_sim,
    precompute_all_vr,
    simulate_with_tiered_slippage,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Phase 0: Config ───────────────────────────────────────────────────────────
cfg_obj = VolStraddleConfig()
SYMBOLS = cfg_obj.symbol_whitelist  # 60 live symbols — DO NOT hardcode

cfg = {
    "sp":    cfg_obj.vr_short,
    "lp":    cfg_obj.vr_long,
    "d_vr":  cfg_obj.d_vr_threshold,
    "s_vr":  cfg_obj.s_vr_threshold,
    "s_dur": cfg_obj.s_squeeze_dur,
    "d_sl":  cfg_obj.d_sl,
    "s_sl":  cfg_obj.s_sl,
    "tp":    cfg_obj.d_tp,
}
VR_LN = cfg_obj.base_leverage / cfg_obj.max_concurrent  # 3.0
MA_LN = cfg_obj.ma_cluster_leverage / cfg_obj.max_concurrent  # 2.1

CACHE_DIR = ROOT / "data" / "cache"
DAYS = 2190  # 6 years of 15m data


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_data() -> tuple[dict, dict, dict, float]:
    """Load 15m/1h/4h data for symbol whitelist from cache."""
    whitelist = set(SYMBOLS)
    data_15m: dict = {}
    for f in sorted(CACHE_DIR.glob(f"*_15m_{DAYS}d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        d = np.load(f)
        data_15m[sym] = {k: d[k] for k in d.files}

    if not data_15m:
        raise RuntimeError(
            f"No cache files found in {CACHE_DIR}. "
            "Run the data downloader first."
        )

    data_1h = build_resampled(data_15m, ratio=4)
    data_4h = build_resampled(data_15m, ratio=16)

    n_bars = max(len(d["close"]) for d in data_15m.values())
    data_years = n_bars * 15 / 60 / 24 / 365.25
    return data_15m, data_1h, data_4h, data_years


def run_model(
    mode: str,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_years: float,
) -> tuple[dict, dict]:
    """Run D_mom8 + S_d8 under a given slippage mode, return metrics dicts."""
    t0 = time.time()
    trades_d = simulate_with_tiered_slippage(cfg, data_15m, data_1h, data_4h, "D_mom8", mode)
    trades_s = simulate_with_tiered_slippage(cfg, data_15m, data_1h, data_4h, "S_d8", mode)

    # Merge and portfolio-simulate (slot competition across both strats)
    all_trades = trades_d + trades_s
    pnl_combined = portfolio_sim(all_trades, max_concurrent=cfg_obj.max_concurrent)
    m_comb = compute_metrics(pnl_combined, data_years)

    # Also get per-strategy metrics for EV breakdown
    pnl_d = portfolio_sim(trades_d, max_concurrent=cfg_obj.max_concurrent)
    m_d = compute_metrics(pnl_d, data_years)

    pnl_s = portfolio_sim(trades_s, max_concurrent=cfg_obj.max_concurrent)
    m_s = compute_metrics(pnl_s, data_years)

    elapsed = time.time() - t0
    print(f"  [{mode:12s}] D: N={m_d['N']:4d} PF={m_d['PF']:.3f} EV={m_d['EV']:+.1f}bp | "
          f"S: N={m_s['N']:4d} PF={m_s['PF']:.3f} EV={m_s['EV']:+.1f}bp | "
          f"Combined: N={m_comb['N']:4d} PF={m_comb['PF']:.3f} ({elapsed:.1f}s)")
    return {"d": m_d, "s": m_s, "combined": m_comb}


def print_symbol_tier_summary() -> None:
    """Print how the 60 symbols split across liquidity tiers."""
    whitelist = set(SYMBOLS)
    high = sorted(whitelist & _HIGH_LIQ_SYMS)
    low  = sorted(whitelist & _LOW_LIQ_SYMS)
    med  = sorted(whitelist - _HIGH_LIQ_SYMS - _LOW_LIQ_SYMS)

    print("\n══════════════════════════════════════════════════════════")
    print("  Symbol Liquidity Tier Classification")
    print("══════════════════════════════════════════════════════════")
    print(f"  HIGH (0.08%): {len(high):2d} symbols — {high}")
    print(f"  MED  (0.12%): {len(med):2d} symbols — {med}")
    print(f"  LOW  (0.16%): {len(low):2d} symbols — {low}")
    total_syms = len(whitelist)
    w_avg = (len(high) * _COST_HIGH_LIQ + len(med) * _COST_MED_LIQ + len(low) * _COST_LOW_LIQ) / total_syms
    print(f"  Weighted avg tiered cost: {w_avg:.4f} ({w_avg*100:.3f}%)")
    print(f"  vs flat cost:             {COST_DIRECTIONAL:.4f} ({COST_DIRECTIONAL*100:.3f}%)")
    print(f"  Delta (tiered − flat):    {w_avg - COST_DIRECTIONAL:+.4f} ({(w_avg - COST_DIRECTIONAL)*100:+.3f}%)")
    print()


def main() -> None:
    commit = get_commit()
    print(f"\nr77 Slippage Sensitivity — commit={commit}")
    print(f"Config: VR_LN={VR_LN:.1f}, MA_LN={MA_LN:.1f}, max_concurrent={cfg_obj.max_concurrent}")
    print(f"Flat cost: dir={COST_DIRECTIONAL:.4f} str={COST_STRADDLE:.4f}")

    print_symbol_tier_summary()

    print("Loading data...")
    data_15m, data_1h, data_4h, data_years = load_data()
    print(f"  {len(data_15m)} symbols, {data_years:.2f} years of 15m data\n")

    # Precompute VR for all symbols
    sp, lp = cfg_obj.vr_short, cfg_obj.vr_long
    precompute_all_vr([data_15m, data_1h, data_4h], [(sp, lp)])

    # ── Phase 1: Baseline check ─────────────────────────────────────────────
    print("Phase 1: Baseline sanity check (flat mode)...")
    trades_d_flat = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
    trades_s_flat = collect_trades(cfg, data_15m, data_1h, data_4h, "S_d8")
    pnl_d_flat = portfolio_sim(trades_d_flat, max_concurrent=cfg_obj.max_concurrent)
    pnl_s_flat = portfolio_sim(trades_s_flat, max_concurrent=cfg_obj.max_concurrent)
    pnl_all_flat = portfolio_sim(trades_d_flat + trades_s_flat, max_concurrent=cfg_obj.max_concurrent)

    m_d_bl = compute_metrics(pnl_d_flat, data_years)
    m_s_bl = compute_metrics(pnl_s_flat, data_years)
    m_all_bl = compute_metrics(pnl_all_flat, data_years)

    # Canonical (v14.5.0, D_mom8 only): N=6207, PF=1.563
    # Combined (D+S) r73 trail canonical: N=9773, PF=2.002
    n_canon_d = 6207
    pf_canon_d = 1.563
    n_ok = abs(m_d_bl["N"] - n_canon_d) / n_canon_d < 0.10
    pf_ok = abs(m_d_bl["PF"] - pf_canon_d) / pf_canon_d < 0.10
    print(f"  D_mom8 baseline: N={m_d_bl['N']} (canon≈{n_canon_d}, {'✅' if n_ok else '❌'}), "
          f"PF={m_d_bl['PF']:.3f} (canon≈{pf_canon_d:.3f}, {'✅' if pf_ok else '❌'})")
    print(f"  S_d8   baseline: N={m_s_bl['N']}, PF={m_s_bl['PF']:.3f}")
    print(f"  Combined (D+S):  N={m_all_bl['N']}, PF={m_all_bl['PF']:.3f}")
    if not n_ok or not pf_ok:
        print("  ⚠️  Baseline drift >10% from canonical — check data range / config")

    # ── Phase 2: Three-model sweep ───────────────────────────────────────────
    print("\nPhase 2: Slippage model comparison...")
    results: dict[str, dict] = {}
    for mode in ("flat", "tiered", "conservative"):
        results[mode] = run_model(mode, data_15m, data_1h, data_4h, data_years)

    # ── Phase 3: EV sensitivity table ───────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════")
    print("  EV Sensitivity to Slippage Model — D_mom8 + S_d8 Combined")
    print("══════════════════════════════════════════════════════════")
    header = f"  {'Model':13s} {'N':>5s} {'PF':>6s} {'WR%':>6s} {'EV(bp)':>8s} {'MaxDD%':>8s}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for mode in ("flat", "tiered", "conservative"):
        m = results[mode]["combined"]
        flag = " ◀ current" if mode == "flat" else ""
        print(f"  {mode:13s} {m['N']:5d} {m['PF']:6.3f} {m['WR']*100:6.1f} "
              f"{m['EV']*10000:8.1f} {m['MaxDD']*100:8.1f}{flag}")

    print("\n  Per-strategy breakdown:")
    for strat_key, strat_label in (("d", "D_mom8"), ("s", "S_d8")):
        print(f"\n  {strat_label}:")
        sub_hdr = f"    {'Model':13s} {'N':>5s} {'PF':>6s} {'EV(bp)':>8s}  cost_delta"
        print(sub_hdr)
        print("    " + "─" * (len(sub_hdr) - 4))
        for mode in ("flat", "tiered", "conservative"):
            m = results[mode][strat_key]
            flat_cost = COST_DIRECTIONAL if strat_key == "d" else COST_STRADDLE
            # Representative delta for a mid-tier symbol
            mid_cost = get_symbol_slippage("SANDUSDT", conservative=(mode == "conservative"))
            delta = (mid_cost * (2 if strat_key == "s" else 1)) - flat_cost
            print(f"    {mode:13s} {m['N']:5d} {m['PF']:6.3f} {m['EV']*10000:8.1f}  "
                  f"Δ={delta*100:+.3f}% (mid-tier sym example)")

    # ── CAGR under each model ────────────────────────────────────────────────
    print("\n  1x CAGR under each model (combined D+S):")
    print(f"  {'Model':13s} {'CAGR_1x':>10s} {'CAGR_LN3.0':>12s}")
    for mode in ("flat", "tiered", "conservative"):
        m = results[mode]["combined"]
        cagr_1x = m.get("CAGR", float("nan"))
        cagr_ln = m.get("CAGR_LN", float("nan"))
        print(f"  {mode:13s} {cagr_1x*100:10.1f}% {cagr_ln*100:12.1f}%")

    # ── Conclusion ───────────────────────────────────────────────────────────
    flat_pf = results["flat"]["combined"]["PF"]
    tiered_pf = results["tiered"]["combined"]["PF"]
    cons_pf = results["conservative"]["combined"]["PF"]
    print(f"\n══════════════════════════════════════════════════════════")
    print(f"  PF impact:")
    print(f"    flat → tiered:       {tiered_pf - flat_pf:+.3f} ({(tiered_pf/flat_pf-1)*100:+.1f}%)")
    print(f"    flat → conservative: {cons_pf - flat_pf:+.3f} ({(cons_pf/flat_pf-1)*100:+.1f}%)")

    white = set(SYMBOLS)
    high_cnt = len(white & _HIGH_LIQ_SYMS)
    med_cnt  = len(white - _HIGH_LIQ_SYMS - _LOW_LIQ_SYMS)
    low_cnt  = len(white & _LOW_LIQ_SYMS)
    print(f"\n  Slippage tier breakdown: HIGH={high_cnt}, MED={med_cnt}, LOW={low_cnt}")
    print(f"  Tiered model adds {med_cnt*_COST_MED_LIQ + low_cnt*_COST_LOW_LIQ + high_cnt*_COST_HIGH_LIQ:.4f} "
          f"avg cost vs {len(white)*COST_DIRECTIONAL:.4f} flat")
    print(f"\n  Recommendation: use tiered for realistic cost modeling.")
    print(f"  Use conservative for worst-case stress testing.")
    print(f"  Production slippage_mode='{cfg_obj.slippage_mode}' (from config).")


if __name__ == "__main__":
    main()
