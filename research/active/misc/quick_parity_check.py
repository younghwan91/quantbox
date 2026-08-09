#!/usr/bin/env python3
"""Quick backtest check: D_mom8 vs S_d8 profitability on available symbols.

Loads all available symbols from cache_extended, runs both entry modes,
reports per-mode PF, WR, EV, cost impact, and per-mode trade breakdown.

Usage:
    uv run python research/active/misc/quick_parity_check.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    build_resampled,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

CACHE_DIR = ROOT / "data" / "cache_extended"


def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_15m(whitelist: list[str]) -> dict:
    data: dict = {}
    for sym in whitelist:
        candidates = [
            f for f in CACHE_DIR.glob(f"{sym}_15m_*.npz")
            if "numtrades" not in f.name
        ]
        if not candidates:
            continue
        best = max(candidates, key=_extract_days)
        npz = np.load(best)
        if not {"close", "high", "low", "volume"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {
            "close": c,
            "high": npz["high"].astype(np.float64),
            "low": npz["low"].astype(np.float64),
            "volume": npz["volume"].astype(np.float64),
            "bars": len(c),
            "days": _extract_days(best),
        }
    return data


def print_mode(label: str, m: dict, n_trades_raw: int, cost_bp: float) -> None:
    print(f"\n  [{label}]")
    print(f"    Raw trades (pre-portfolio): {n_trades_raw:,}")
    print(f"    Accepted trades (portfolio): {m['N']:,}")
    print(f"    PF={m['PF']:.3f}  WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp/trade")
    print(f"    AnnRet={m['ann_ret_pct']:+.2f}% (1x lev)  MaxDD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.2f}")
    print(f"    TPD={m['tpd']:.3f} trades/day")
    print(f"    Round-trip cost applied: {cost_bp:.0f}bp per trade")


def main() -> None:
    t0 = time.time()
    print(f"\n{'='*70}")
    print("  Quick Parity Check: D_mom8 vs S_d8 Backtest")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*70}")

    cfg_obj = VolStraddleConfig()
    cfg = {
        "sp":    cfg_obj.vr_short,
        "lp":    cfg_obj.vr_long,
        "d_vr":  cfg_obj.d_vr_threshold,
        "s_vr":  cfg_obj.s_vr_threshold,
        "s_dur": cfg_obj.s_squeeze_dur,
        "d_sl":  cfg_obj.d_sl,
        "s_sl":  cfg_obj.s_sl,
        "tp":    cfg_obj.s_tp,
    }
    max_concurrent = cfg_obj.max_concurrent
    fee_rt = cfg_obj.fee_rate * 2  # round-trip fee
    slippage = 0.0004              # one-way slippage (from CLAUDE.md: round-trip=0.0008)
    cost_rt = fee_rt + slippage * 2
    cost_bp = cost_rt * 10000

    print(f"\n  Config source: VolStraddleConfig()")
    print(f"    vr_short={cfg['sp']}  vr_long={cfg['lp']}")
    print(f"    d_vr={cfg['d_vr']}  s_vr={cfg['s_vr']}  s_dur={cfg['s_dur']}")
    print(f"    d_sl={cfg['d_sl']*100:.1f}%  s_sl={cfg['s_sl']*100:.1f}%  tp={cfg['tp']*100:.1f}%")
    print(f"    max_concurrent={max_concurrent}  fee_rate={cfg_obj.fee_rate*10000:.0f}bp")
    print(f"    Round-trip cost: {cost_bp:.0f}bp (fees+slippage)")

    # Load data
    print(f"\n  Loading data from {CACHE_DIR.name}/ ...")
    data_15m = load_15m(cfg_obj.symbol_whitelist)
    if not data_15m:
        print("  ERROR: No data loaded. Check data/cache_extended/")
        return

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    precompute_all_vr([data_15m, data_1h, data_4h], [(cfg["sp"], cfg["lp"])])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=cfg["sp"], vr_lp=cfg["lp"])

    bars_list = [d["bars"] for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  {data_years:.2f} years")
    print(f"  Load + precompute: {time.time()-t0:.1f}s")

    # ── D_mom8 ──────────────────────────────────────────────────────────────
    t1 = time.time()
    d_trades = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
    d_pnl = portfolio_sim(d_trades, max_concurrent)
    d_m = compute_metrics(d_pnl, data_years)
    print_mode("D_mom8 — Directional VR Squeeze", d_m, len(d_trades), cost_bp)
    print(f"    compute: {time.time()-t1:.1f}s")

    # Per-symbol breakdown (top 5 by trade count)
    from collections import Counter
    d_sym_counts = Counter(t[0] for t in d_trades)
    print(f"    Top symbols by raw trades: " +
          ", ".join(f"{s}:{n}" for s, n in d_sym_counts.most_common(5)))

    # Win/loss breakdown
    if len(d_pnl) > 0:
        wins = d_pnl[d_pnl > 0]
        losses = d_pnl[d_pnl <= 0]
        print(f"    Wins: {len(wins)} avg={wins.mean()*100:.2f}% | "
              f"Losses: {len(losses)} avg={losses.mean()*100:.2f}%")

    # ── S_d8 ────────────────────────────────────────────────────────────────
    t2 = time.time()
    s_trades = collect_trades(cfg, data_15m, data_1h, data_4h, "S_d8")
    # Straddle: each trade consumes 2 slots, use max_concurrent=5 for standalone
    s_pnl = portfolio_sim(s_trades, max_concurrent // 2)
    s_m = compute_metrics(s_pnl, data_years)
    print_mode("S_d8 — Straddle VR Squeeze (5-slot standalone)", s_m, len(s_trades), cost_bp)
    print(f"    compute: {time.time()-t2:.1f}s")

    s_sym_counts = Counter(t[0] for t in s_trades)
    print(f"    Top symbols by raw trades: " +
          ", ".join(f"{s}:{n}" for s, n in s_sym_counts.most_common(5)))

    if len(s_pnl) > 0:
        wins = s_pnl[s_pnl > 0]
        losses = s_pnl[s_pnl <= 0]
        print(f"    Wins: {len(wins)} avg={wins.mean()*100:.2f}% | "
              f"Losses: {len(losses)} avg={losses.mean()*100:.2f}%")

    # ── Combined (10-slot, straddle uses 2 slots each) ──────────────────────
    print(f"\n  [COMBINED — 10 slots: D_mom8 + S_d8 competing]")
    strd_syms = frozenset(t[0] for t in s_trades)
    comb_pnl = portfolio_sim(d_trades + s_trades, max_concurrent, straddle_syms=strd_syms)
    comb_m = compute_metrics(comb_pnl, data_years)
    print(f"    Accepted trades: {comb_m['N']:,}")
    print(f"    PF={comb_m['PF']:.3f}  WR={comb_m['WR']:.1f}%  EV={comb_m['EV_bp']:+.1f}bp")
    print(f"    AnnRet={comb_m['ann_ret_pct']:+.2f}% (1x)  MaxDD={comb_m['max_dd_pct']:.1f}%")

    # ── Cost sensitivity ─────────────────────────────────────────────────────
    print(f"\n  [Cost Impact — D_mom8 EV breakdown]")
    if len(d_pnl) > 0:
        gross_ev = d_pnl.mean()
        net_ev = gross_ev - cost_rt
        print(f"    Gross EV: {gross_ev*10000:+.1f}bp  Cost: -{cost_bp:.0f}bp  Net EV: {net_ev*10000:+.1f}bp")
        print(f"    PF gross={d_m['PF']:.3f} -> estimated net PF considering {cost_bp:.0f}bp cost/trade")

    print(f"\n  [Cost Impact — S_d8 EV breakdown]")
    if len(s_pnl) > 0:
        gross_ev = s_pnl.mean()
        net_ev = gross_ev - cost_rt
        print(f"    Gross EV: {gross_ev*10000:+.1f}bp  Cost: -{cost_bp:.0f}bp  Net EV: {net_ev*10000:+.1f}bp")

    print(f"\n{'='*70}")
    print(f"  Total runtime: {time.time()-t0:.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
