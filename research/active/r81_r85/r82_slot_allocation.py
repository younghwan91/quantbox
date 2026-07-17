"""r82: straddle_max_slots 최적 비율 스윕.

**배경:**
  straddle_max_slots=6 (3 pairs) 직관적 설정. 최적 비율 미검증.
  MA 비활성화 후 VR+STRD만 남았으므로, 10슬롯 내 최적 STRD:VR 비율 찾기.

**핵심 질문:**
  1. straddle_max_slots=[2,4,6,8,10] 별 혼합 Calmar/PF
  2. STRD 슬롯이 너무 많으면 VR 진입 기회 상실 — 어디가 sweet spot?

사용:
    uv run python research/active/r81_r85/r82_slot_allocation.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    build_resampled,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal

CACHE_DIR = ROOT / "data" / "cache_extended"
WARMUP_30m = 100
cfg = VolStraddleConfig()


def load_all_data():
    """Load and precompute all timeframe data."""
    whitelist = set(cfg.symbol_whitelist)
    raw: dict = {}
    max_bars = 0
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        npz = np.load(str(f), allow_pickle=False)
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        raw[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
        if len(c) > max_bars:
            max_bars = len(c)

    data_15m: dict = {}
    for sym, d in raw.items():
        d["_offset"] = max_bars - len(d["close"])
        data_15m[sym] = d

    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg.vr_short, cfg.vr_long)])
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)

    bars_list = [len(d["close"]) for d in data_15m.values()]
    data_years = int(np.mean(bars_list)) / (4 * 24 * 365)

    return data_15m, data_30m, data_1h, data_4h, data_years


def _dedup(sym: str, trades: list) -> list:
    result = []
    last_exit = -1
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


def collect_strd_trades(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
) -> list:
    """Collect STRD trades with production params."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    mh_30m = cfg.max_hold_15m // 2
    tf_30m = ("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)
    g4tf = [tf_30m] + list(G_3TF)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    mh_15m = cfg.max_hold_15m

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list = []
        for _lbl, ratio, warmup, _mhd, _mhs in g4tf:
            d = tf_map[ratio].get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            mh_d = int(mh_15m // ratio)
            mh_s = max(1, int(mh_d * 2))
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, cfg.s_sl, cfg.s_tp, mh_s, COST_STRADDLE,
                cfg.s_trail_activate, cfg.s_trail_distance,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def collect_vr_trades(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
) -> list:
    """Collect VR D_mom8 trades with production params."""
    vr_cfg = {
        "sp": cfg.vr_short,
        "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold,
        "s_vr": cfg.s_vr_threshold,
        "s_dur": cfg.s_squeeze_dur,
        "d_sl": cfg.d_sl,
        "s_sl": cfg.s_sl,
        "tp": cfg.d_tp,
    }
    return collect_trades(vr_cfg, data_15m, data_1h, data_4h, strat="D_mom8")


def mixed_portfolio_sim(
    vr_trades: list,
    strd_trades: list,
    strd_max_slots: int,
    total_slots: int = 10,
) -> np.ndarray:
    """Simulate mixed portfolio with STRD slot cap.

    STRD trades consume 2 slots each (LONG+SHORT).
    VR trades consume 1 slot each.
    Total slot budget = total_slots.

    Strategy:
      - Merge all trades by entry bar, sorted chronologically
      - Track active slots: strd_active (count pairs×2), vr_active
      - Accept STRD if strd_active + 2 <= strd_max_slots AND total_active + 2 <= total_slots
      - Accept VR if total_active + 1 <= total_slots
    """
    # Tag trades: (entry_bar, exit_bar, pnl, type, sym)
    tagged = []
    for t in vr_trades:
        if len(t) == 4:
            sym, eb, xb, pnl = t
        else:
            eb, xb, pnl = t
            sym = "?"
        tagged.append((eb, xb, pnl, "vr", sym))

    for t in strd_trades:
        if len(t) == 4:
            sym, eb, xb, pnl = t
        else:
            eb, xb, pnl = t
            sym = "?"
        tagged.append((eb, xb, pnl, "strd", sym))

    tagged.sort(key=lambda x: x[0])

    # Simulate with slot constraints
    active: list = []  # (exit_bar, type, pnl)
    pnl_out = []

    for eb, xb, pnl, ttype, sym in tagged:
        # Remove expired positions
        active = [(ax, at, ap) for ax, at, ap in active if ax > eb]

        strd_slots = sum(2 for _, t, _ in active if t == "strd")
        vr_slots = sum(1 for _, t, _ in active if t == "vr")
        total_active = strd_slots + vr_slots

        if ttype == "strd":
            if strd_slots + 2 <= strd_max_slots and total_active + 2 <= total_slots:
                active.append((xb, "strd", pnl))
                pnl_out.append(pnl)
        else:  # vr
            if total_active + 1 <= total_slots:
                active.append((xb, "vr", pnl))
                pnl_out.append(pnl)

    return np.array(pnl_out) if pnl_out else np.array([0.0])


def main() -> None:
    t0 = time.time()
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    print(f"r82: straddle_max_slots sweep — commit {sha}")
    print(f"Universe: {len(cfg.symbol_whitelist)} syms")
    print()

    print("Loading & precomputing data...")
    data_15m, data_30m, data_1h, data_4h, data_years = load_all_data()
    n_syms = len(data_15m)
    print(f"Loaded {n_syms} symbols, {data_years:.2f} years\n")

    # Collect all trades
    print("Collecting VR trades...")
    vr_trades = collect_vr_trades(data_15m, data_1h, data_4h)
    print(f"  VR: {len(vr_trades)} trades")

    print("Collecting STRD trades...")
    strd_trades = collect_strd_trades(data_15m, data_30m, data_1h, data_4h)
    print(f"  STRD: {len(strd_trades)} trades")

    # Phase 1: Individual baselines
    print(f"\n{'='*70}")
    print("Phase 1: Individual strategy baselines")
    print(f"{'='*70}")

    vr_pnl = portfolio_sim(vr_trades, 10)
    vr_m = compute_metrics(vr_pnl, data_years)
    print(f"  VR only (10 slots): N={vr_m['N']} PF={vr_m['PF']:.3f} "
          f"Calmar={vr_m['calmar']:.0f} MaxDD={vr_m['max_dd_pct']:.1f}%")

    strd_pnl = portfolio_sim(strd_trades, 5)  # STRD parity: max 5 pairs
    strd_m = compute_metrics(strd_pnl, data_years)
    print(f"  STRD only (5 pairs): N={strd_m['N']} PF={strd_m['PF']:.3f} "
          f"Calmar={strd_m['calmar']:.0f} MaxDD={strd_m['max_dd_pct']:.1f}%")

    # Phase 2: Mixed sweep
    print(f"\n{'='*70}")
    print("Phase 2: straddle_max_slots sweep (total=10)")
    print(f"{'='*70}")
    print(f"{'strd_max':>10} {'strd_pairs':>10} {'vr_avail':>10} "
          f"{'N':>6} {'PF':>8} {'WR':>6} {'EV':>8} {'MaxDD':>8} {'Calmar':>10} ★")
    print("-" * 80)

    best_calmar = 0
    best_config = 0

    for strd_max in [0, 2, 4, 6, 8, 10]:
        pnl = mixed_portfolio_sim(vr_trades, strd_trades, strd_max, 10)
        m = compute_metrics(pnl, data_years)
        star = ""
        if m["calmar"] > best_calmar:
            best_calmar = m["calmar"]
            best_config = strd_max
            star = " ★"
        print(f"{strd_max:>10} {strd_max // 2:>10} {10 - strd_max:>10} "
              f"{m['N']:>6} {m['PF']:>8.3f} {m['WR']:>5.1f}% "
              f"{m['EV_bp']:>7.1f}bp {m['max_dd_pct']:>7.1f}% "
              f"{m['calmar']:>10.0f}{star}")

    print(f"\nBest: straddle_max_slots={best_config} (Calmar={best_calmar:.0f})")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
