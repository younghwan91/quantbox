"""R38b — Fast Grid Search: 132 variants × IS/OOS + Slot×TPD optimization.

Phase 1: Fast PnL array scan (no trade metadata = 10× faster)
Phase 2: Only top variants get full trade collection for slot analysis
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from research.active.r31_r40.r31_mtf_vol_straddle import (
    load_15m, build_resampled, precompute, compute_vr, compute_momentum,
    compute_squeeze_dur, vol_percentile, gen_vr_signal, gen_vr_dur_signal,
    make_direction_momentum,
)
from research.active.r31_r40.scaling_r38 import (
    collect_trades, portfolio_sim, compute_metrics, precompute_all_vr,
    sim_straddle_trades, sim_directional_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF,
)


def fast_pnl(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    strat: str,
) -> np.ndarray:
    """Fast PnL array — no trade metadata, simpler dedup."""
    sp, lp = cfg["sp"], cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    d_sl, s_sl, tp = cfg["d_sl"], cfg["s_sl"], cfg["tp"]
    d_vr, s_vr, s_dur = cfg["d_vr"], cfg["s_vr"], cfg["s_dur"]
    all_pnl: list[float] = []

    for sym in sorted(data_15m.keys()):
        sym_entries: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sl = s_sl if is_straddle else d_sl
            mh = mh_s if is_straddle else mh_d

            if is_straddle:
                sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
            else:
                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_entries.append((entry_15m, exit_15m, float(p[k])))

        # Fast dedup: sort by entry, keep non-overlapping
        sym_entries.sort()
        last_exit = -1
        for entry, exit_b, pnl_val in sym_entries:
            if entry > last_exit:
                all_pnl.append(pnl_val)
                last_exit = exit_b

    return np.array(all_pnl) if all_pnl else np.empty(0)


def main() -> None:
    t0 = time.time()
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)

    all_pairs = [
        (4,28),(4,30),(4,35),(4,40),
        (5,30),(5,35),(5,40),(5,45),
        (6,30),(6,35),(6,40),(6,42),(6,45),(6,50),
        (7,35),(7,40),(7,42),(7,45),(7,50),
        (8,40),(8,50),(8,56),
    ]
    precompute_all_vr([data_15m, data_1h, data_4h], all_pairs)

    # OOS split
    is_15m: dict = {}; oos_15m: dict = {}
    is_1h: dict = {}; oos_1h: dict = {}
    is_4h: dict = {}; oos_4h: dict = {}
    for sym in data_15m:
        n = len(data_15m[sym]["close"])
        cut = int(n * 0.67)
        for src, is_d, oos_d in [
            (data_15m, is_15m, oos_15m),
            (data_1h, is_1h, oos_1h),
            (data_4h, is_4h, oos_4h),
        ]:
            if sym not in src:
                continue
            d = src[sym]; n_tf = len(d["close"])
            ratio = n // n_tf if n_tf > 0 else 1
            cut_tf = cut // ratio
            for start, end_b, dst in [(0, cut_tf, is_d), (cut_tf, n_tf, oos_d)]:
                seg_len = end_b - start
                if seg_len < 100:
                    continue
                d_seg: dict = {}
                for key in ("close", "high", "low", "volume"):
                    d_seg[key] = d[key][start:end_b].copy()
                d_seg["bars"] = seg_len
                d_seg["days"] = d.get("days", 0)
                c, h, lo, v = d_seg["close"], d_seg["high"], d_seg["low"], d_seg["volume"]
                d_seg["vr"] = compute_vr(c, h, lo, 7, 50)
                d_seg["mom_8"] = compute_momentum(c, 8)
                d_seg["vol_pct"] = vol_percentile(v, min(200, seg_len // 2))
                d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)
                for sp, lp in all_pairs:
                    d_seg[f"vr_{sp}_{lp}"] = compute_vr(c, h, lo, sp, lp)
                dst[sym] = d_seg

    is_years = data_years * 0.67
    oos_years = data_years * 0.33

    # ═════════════════════════════════════════════════════════
    #  Phase 1: Fast scan all configs
    # ═════════════════════════════════════════════════════════
    configs: list[tuple[str, dict]] = []
    for sp, lp in all_pairs:
        for d_vr in [0.45, 0.50, 0.55]:
            s_vr = d_vr + 0.05
            for s_dur in [4, 8]:
                name = f"VR{sp}_{lp}_T{int(d_vr*100)}_D{s_dur}"
                configs.append((name, {
                    "sp": sp, "lp": lp, "d_vr": d_vr, "s_vr": s_vr,
                    "s_dur": s_dur, "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
                }))

    print(f"Phase 1: Fast scan {len(configs)} variants × 3 splits...")
    t1 = time.time()

    results: list[dict] = []
    for idx, (name, cfg) in enumerate(configs):
        if (idx + 1) % 30 == 0:
            elapsed = time.time() - t1
            print(f"  ... {idx+1}/{len(configs)} ({elapsed:.0f}s)")

        full_d = fast_pnl(cfg, data_15m, data_1h, data_4h, "D_mom8")
        full_s = fast_pnl(cfg, data_15m, data_1h, data_4h, "S_d8")
        full_pnl = np.concatenate([full_d, full_s]) if len(full_d)+len(full_s) > 0 else np.empty(0)

        is_d = fast_pnl(cfg, is_15m, is_1h, is_4h, "D_mom8")
        is_s = fast_pnl(cfg, is_15m, is_1h, is_4h, "S_d8")
        is_pnl = np.concatenate([is_d, is_s]) if len(is_d)+len(is_s) > 0 else np.empty(0)

        oos_d = fast_pnl(cfg, oos_15m, oos_1h, oos_4h, "D_mom8")
        oos_s = fast_pnl(cfg, oos_15m, oos_1h, oos_4h, "S_d8")
        oos_pnl = np.concatenate([oos_d, oos_s]) if len(oos_d)+len(oos_s) > 0 else np.empty(0)

        m_full = compute_metrics(full_pnl, data_years)
        m_is = compute_metrics(is_pnl, is_years)
        m_oos = compute_metrics(oos_pnl, oos_years)

        results.append({
            "name": name, "cfg": cfg,
            "full": m_full, "is": m_is, "oos": m_oos,
        })

    phase1_time = time.time() - t1
    print(f"  Phase 1 done in {phase1_time:.1f}s")

    # Filter: IS PF > 1.0 AND OOS PF > 1.0 AND OOS t > 2.0 AND full N > 2000
    valid = [r for r in results
             if r["is"]["PF"] > 1.0 and r["oos"]["PF"] > 1.0
             and r["oos"]["t"] > 2.0 and r["full"]["N"] > 2000]
    valid.sort(key=lambda r: r["oos"]["calmar"], reverse=True)

    print(f"\n{'='*120}")
    print(f"  Phase 1 Results: {len(valid)} / {len(results)} passed filter")
    print(f"  Filter: IS_PF>1.0 AND OOS_PF>1.0 AND OOS_t>2.0 AND N>2000")
    print(f"{'='*120}")

    hdr = (f"  {'Variant':<25} {'N':>5} {'PF':>5} {'EV':>6} {'t':>5} "
           f"{'WR':>4} {'R:R':>4} {'TPD':>4} | "
           f"{'IS_PF':>5} {'IS_EV':>6} | "
           f"{'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>5} {'OOS_Cal':>7}")
    print(hdr)
    print("  " + "─" * 115)

    for r in valid[:30]:
        f, i, o = r["full"], r["is"], r["oos"]
        print(f"  {r['name']:<25} {f['N']:>5} {f['PF']:>5.3f} {f['EV_bp']:>+6.1f} "
              f"{f['t']:>5.2f} {f['WR']:>4.1f} {f['RR']:>4.2f} {f['tpd']:>4.2f} | "
              f"{i['PF']:>5.3f} {i['EV_bp']:>+6.1f} | "
              f"{o['PF']:>6.3f} {o['EV_bp']:>+7.1f} {o['t']:>5.2f} {o['calmar']:>7.1f}")

    # References - always show
    print(f"\n  ── References ──")
    for ref in ["VR5_35_T50_D8", "VR7_50_T50_D8"]:
        for r in results:
            if r["name"] == ref:
                f, i, o = r["full"], r["is"], r["oos"]
                print(f"  {r['name']:<25} {f['N']:>5} {f['PF']:>5.3f} {f['EV_bp']:>+6.1f} "
                      f"{f['t']:>5.2f} {f['WR']:>4.1f} {f['RR']:>4.2f} {f['tpd']:>4.2f} | "
                      f"{i['PF']:>5.3f} {i['EV_bp']:>+6.1f} | "
                      f"{o['PF']:>6.3f} {o['EV_bp']:>+7.1f} {o['t']:>5.2f} {o['calmar']:>7.1f}")

    # ═════════════════════════════════════════════════════════
    #  Phase 2: Slot × TPD for top 5 + references
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  Phase 2: Slot × TPD Optimization (top finalists)")
    print(f"{'='*120}")

    targets = list(valid[:5])
    for ref_name in ["VR5_35_T50_D8", "VR7_50_T50_D8"]:
        for r in results:
            if r["name"] == ref_name and r not in targets:
                targets.append(r)

    t2 = time.time()
    for r in targets:
        name = r["name"]
        cfg = r["cfg"]
        # Full trade collection (with metadata for portfolio sim)
        d_t = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
        s_t = collect_trades(cfg, data_15m, data_1h, data_4h, "S_d8")
        trades = d_t + s_t
        unlimited_n = len(trades)

        print(f"\n  {name} (unlimited N={unlimited_n}):")
        print(f"  {'Slots':>5} {'N':>5} {'PF':>5} {'EV':>6} {'TPD':>5} "
              f"{'AnnRet%':>8} {'DD%':>5} {'Calmar':>7} {'Fill%':>6}")
        print(f"  {'─'*65}")
        for slots in [1, 2, 3, 4, 5, 6, 8]:
            pnl = portfolio_sim(trades, slots)
            m = compute_metrics(pnl, data_years)
            fill = m["N"] / unlimited_n * 100 if unlimited_n else 0
            print(f"  {slots:>5} {m['N']:>5} {m['PF']:>5.3f} {m['EV_bp']:>+6.1f} "
                  f"{m['tpd']:>5.2f} {m['ann_ret_pct']:>+8.1f} "
                  f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f} {fill:>5.1f}%")

    phase2_time = time.time() - t2
    total = time.time() - t0
    print(f"\n  Phase 2 done in {phase2_time:.1f}s")
    print(f"  Total R38b completed in {total:.1f}s\n")


if __name__ == "__main__":
    main()
