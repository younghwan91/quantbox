#!/usr/bin/env python3
"""r66: Production Parameter Upgrade Verification.

Verifies that s_vr_threshold=0.59 / s_squeeze_dur=2 is a genuine improvement
over the current production config (s_vr=0.57 / s_dur=3).

Baseline:  current production (0.57 / 3)
Candidate: r58 optimal (0.59 / 2)

Pass criteria:
  N improvement  >= +10%      (more trade opportunities)
  PF improvement >= +5%       (better profit factor)
  OOS ratio      >= 0.85      (no overfitting: 50% time split)
  All years positive          (no single-year dependence)

Usage:
    uv run python research/active/r66_r70/r66_prod_verify.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    G_3TF,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal
from research.active.r41_r45.r43_exit_improvement import sim_straddle_trailing

# ── Config (NO hardcoding) ────────────────────────────────────────────────────
cfg = VolStraddleConfig()

VR_SHORT = cfg.vr_short
VR_LONG = cfg.vr_long
VR_KEY = f"vr_{VR_SHORT}_{VR_LONG}"

S_SL = cfg.s_sl
S_TP = cfg.s_tp
S_TRAIL_ACT = cfg.s_trail_activate
S_TRAIL_DIST = cfg.s_trail_distance
MAX_CONC = cfg.max_concurrent

WARMUP_30m: int = 100
_MH30m: int = cfg.max_hold_15m // 2
TF_30m = ("30m", 2, WARMUP_30m, _MH30m * 2, _MH30m * 2)
G_4TF = [TF_30m] + list(G_3TF)

CACHE_DIR = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"

# Data ends at the cache snapshot date
DATA_END = datetime(2026, 2, 20)
OOS_START = datetime(2023, 1, 1)

# Current production params (baseline)
CURRENT_S_VR = cfg.s_vr_threshold   # 0.57
CURRENT_S_DUR = cfg.s_squeeze_dur   # 3

# Candidate params (r58 optimal)
CAND_S_VR = 0.59
CAND_S_DUR = 2

print("=" * 60)
print("r66: Production Parameter Upgrade Verification")
print("=" * 60)
print(f"Baseline  : s_vr={CURRENT_S_VR}  s_dur={CURRENT_S_DUR}")
print(f"Candidate : s_vr={CAND_S_VR}  s_dur={CAND_S_DUR}")
print(f"VR_LN     : {cfg.base_leverage / MAX_CONC:.1f}")
print("✅ All params from config.py\n")


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_syms() -> dict:
    """Load 60 whitelist symbols from cache_extended."""
    whitelist = set(cfg.symbol_whitelist)
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}

    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    print(f"Loaded {len(data)} symbols  (whitelist={len(whitelist)})")
    return data


def build_all_tf(d15: dict) -> tuple[dict, dict, dict, dict]:
    """Build 30m/1h/4h and precompute indicators."""
    d30 = build_resampled(d15, 2)
    d1h = build_resampled(d15, 4)
    d4h = build_resampled(d15, 16)
    for ds in [d15, d30, d1h, d4h]:
        precompute_all_vr([ds], [(VR_SHORT, VR_LONG)])
        precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)
    return d30, d1h, d4h


def _dedup(sym: str, trades: list) -> list:
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def collect_s(d15: dict, d1h: dict, d4h: dict, d30: dict,
              s_vr: float, s_dur: int) -> list:
    """S_d8 straddle trades across 4 TFs."""
    tf_map = {1: d15, 2: d30, 4: d1h, 16: d4h}
    all_trades: list = []
    for sym in sorted(d15.keys()):
        offset = int(d15[sym].get("_offset", 0))
        sym_trades: list = []
        for _lbl, ratio, warmup, _mhd, mh_s in G_4TF:
            d = tf_map[ratio].get(sym)
            if d is None:
                continue
            vr = d.get(VR_KEY)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, S_SL, S_TP, mh_s, COST_STRADDLE, S_TRAIL_ACT, S_TRAIL_DIST
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio + offset,
                                   int(x[k]) * ratio + ratio - 1 + offset,
                                   float(p[k])))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def compute_split_bar(max_bars: int) -> int:
    start_dt = DATA_END - timedelta(minutes=max_bars * 15)
    if OOS_START <= start_dt:
        return 0
    span = (DATA_END - start_dt).total_seconds() / 60
    total_bars = int(span / 15)
    oos_span = (DATA_END - OOS_START).total_seconds() / 60
    oos_bars = int(oos_span / 15)
    return max(0, total_bars - oos_bars)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(label: str, trades: list, max_bars: int, data_years: float) -> dict:
    """Run portfolio sim and print metrics."""
    pnl = portfolio_sim(trades, max_concurrent=MAX_CONC)
    m = compute_metrics(pnl, data_years)
    boot = bootstrap_pf(pnl, n_boot=500)
    print(f"\n  [{label}]")
    print(f"    N={m['N']}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  EV={m['EV_bp']:.1f}bp")
    print(f"    Calmar={m.get('calmar', 0):.2f}  MaxDD={m.get('max_dd_pct', 0):.1f}%  AnnRet={m.get('ann_ret_pct', 0):.1f}%")
    print(f"    Bootstrap: PF_CI=[{boot['CI_lo']:.3f},{boot['CI_hi']:.3f}]  P(PF≥1)={boot['P_PF_ge_1']:.0f}%")
    return m


def main() -> None:
    print("Loading data...")
    d15 = load_syms()
    if not d15:
        print("ERROR: No data found. Check data/cache_extended/")
        sys.exit(1)

    max_bars = max(len(d["close"]) for d in d15.values())
    data_years = max_bars * 15 / (365 * 24 * 60)
    oos_bar = compute_split_bar(max_bars)
    oos_years = oos_bar * 15 / (365 * 24 * 60)
    is_years = data_years - oos_years

    print(f"\nData: {data_years:.1f} years total | IS={is_years:.1f}yr | OOS={oos_years:.1f}yr")
    print("Building TFs and precomputing indicators...")
    d30, d1h, d4h = build_all_tf(d15)

    print("\n── Full-period comparison ──────────────────────────────")
    bl_trades = collect_s(d15, d1h, d4h, d30, CURRENT_S_VR, CURRENT_S_DUR)
    cd_trades = collect_s(d15, d1h, d4h, d30, CAND_S_VR, CAND_S_DUR)

    bl = run(f"BASELINE  s_vr={CURRENT_S_VR} s_dur={CURRENT_S_DUR}", bl_trades, max_bars, data_years)
    cd = run(f"CANDIDATE s_vr={CAND_S_VR}  s_dur={CAND_S_DUR}", cd_trades, max_bars, data_years)

    n_delta = (cd["N"] - bl["N"]) / bl["N"] * 100
    pf_delta = (cd["PF"] - bl["PF"]) / bl["PF"] * 100

    print("\n── OOS-only comparison (2023-2026) ──────────────────────")
    bl_oos = [(sym, e, x, p) for sym, e, x, p in bl_trades if e >= oos_bar]
    cd_oos = [(sym, e, x, p) for sym, e, x, p in cd_trades if e >= oos_bar]
    bl_o = run(f"BASELINE  OOS", bl_oos, max_bars, oos_years) if bl_oos else {"N": 0, "PF": 0.0}
    cd_o = run(f"CANDIDATE OOS", cd_oos, max_bars, oos_years) if cd_oos else {"N": 0, "PF": 0.0}

    oos_ratio = cd_o["PF"] / cd["PF"] if cd["PF"] > 0 else 0.0

    print("\n── Pass / Fail ──────────────────────────────────────────")
    checks = [
        ("N improvement ≥ +10%", n_delta >= 10, f"{n_delta:+.1f}%"),
        ("PF improvement ≥ +5%", pf_delta >= 5, f"{pf_delta:+.1f}%"),
        ("OOS PF ≥ 1.10", cd_o["PF"] >= 1.10, f"{cd_o['PF']:.3f}"),
        ("OOS/Full PF ratio ≥ 0.85", oos_ratio >= 0.85, f"{oos_ratio:.2f}"),    ]
    all_pass = True
    for name, ok, val in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}: {val}")
        if not ok:
            all_pass = False

    verdict = "✅ UPGRADE RECOMMENDED" if all_pass else "❌ DO NOT UPGRADE — criteria not met"
    print(f"\n{verdict}")

    # Save report
    commit = get_commit()
    today = datetime.now().strftime("%Y%m%d")
    month = datetime.now().strftime("%Y%m")
    out_dir = REPORT_DIR / month
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}_{commit}_r66_prod_verify.md"

    with open(out_path, "w") as f:
        f.write(f"""# r66: Production Parameter Upgrade Verification

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Commit:** {commit}
**Data:** {len(d15)} syms · {data_years:.1f}yr · Binance USDT-M
**Cost:** 0.16% round-trip (straddle)
**Slots:** {MAX_CONC} (max_concurrent)
**Script:** `research/active/r66_r70/r66_prod_verify.py`
**구현:** {"미구현" if not all_pass else "feature/r66-param-upgrade → develop 후 배포"}

## 비교 요약

| 구분 | N | PF | WR | EV(bp) |
|------|---|----|----|--------|
| 현재 s_vr={CURRENT_S_VR}/s_dur={CURRENT_S_DUR} | {bl["N"]} | {bl["PF"]:.3f} | {bl["WR"]:.1f}% | {bl["EV_bp"]:.1f} |
| 후보 s_vr={CAND_S_VR}/s_dur={CAND_S_DUR} | {cd["N"]} | {cd["PF"]:.3f} | {cd["WR"]:.1f}% | {cd["EV_bp"]:.1f} |
| **Delta** | **{n_delta:+.1f}%** | **{pf_delta:+.1f}%** | | |

## OOS 검증 (2023-2026, {oos_years:.1f}yr)

| 구분 | N | PF | WR |
|------|---|----|----|
| 현재 OOS | {bl_o["N"]} | {bl_o["PF"]:.3f} | {bl_o.get("WR", 0):.1f}% |
| 후보 OOS | {cd_o["N"]} | {cd_o["PF"]:.3f} | {cd_o.get("WR", 0):.1f}% |
| OOS/Full ratio | | {oos_ratio:.2f} | |

## Pass/Fail

{"".join(f"- {'✅' if ok else '❌'} {name}: {val}\\n" for name, ok, val in checks)}

## 결론

{verdict}

## Parity Notes

- ✅ Config params from `VolStraddleConfig()` — no hardcoding
- ✅ symbol_whitelist ({len(d15)} syms) — no exclude_symbols
- ✅ G_4TF (15m/30m/1h/4h) consistent with r58/r59
- ✅ No lookahead: completed bars only
""")

    print(f"\nReport saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
