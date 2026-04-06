"""r62: VR Momentum Filter — does VR trajectory at entry predict WR?

r61 finding: SMA distance is the key WR driver for S_d8, but hard filters
reduce N by ~34% without achieving WR+N target simultaneously.

r62 hypothesis: When VR is RISING from its local minimum at entry (compression
releasing), the breakout is more imminent → higher WR with less N sacrifice.

Two metrics (both lookahead-free):
  dVR[i]         = vr[i] - vr[i-1]     positive = VR rising = decompressing
  bars_since_min = i - argmin(vr[i-W:i+1]) + start  (W-bar lookback window)

Research phases:
  A: S_d8 WR by dVR bucket (strongly falling / falling / rising / strongly rising)
  B: S_d8 WR by bars_since_min (0 / 1 / 2 / 3 / 4+)
  C: Filter sweep on dVR threshold and bars_since_min >= K
  D: Portfolio-level evaluation (combined S_d8 + D_mom8), yearly breakdown, bootstrap CI

Baseline (60 syms, 6yr, Combo2): Combined N≈7,802, WR=48.2%, PF=4.570, CI=[4.294,4.879]

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/r61_r65/r62_vr_momentum.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    gen_vr_signal,
    make_direction_momentum,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Straddle stubs (S_d8 removed from production) ────────────────────────────


def gen_vr_dur_signal(vr, threshold, min_dur, warmup):  # type: ignore[override]
    """Stub: S_d8 straddle signal removed from production."""
    import numpy as np
    return np.zeros(len(vr), dtype=np.int8)


def sim_straddle_trailing(*args, **kwargs):  # type: ignore[override]
    """Stub: straddle simulator removed from production."""
    import numpy as np
    return np.array([]), np.array([]), np.array([])

# ── Config (all from production config.py — no hardcoding) ────────────────────
cfg = VolStraddleConfig()

SP = cfg.vr_short   # 8
LP = cfg.vr_long    # 56
VR_KEY = f"vr_{SP}_{LP}"

S_VR = 0.57   # s_vr_threshold removed from production config
S_DUR = 3     # s_squeeze_dur removed from production config
D_VR = cfg.d_vr_threshold    # 0.5

D_SL = cfg.d_sl    # 0.01
S_SL = 0.015       # s_sl removed from production config
D_TP = cfg.d_tp    # 0.12
S_TP = 0.05        # s_tp removed from production config

D_TRAIL_ACT = cfg.d_trail_activate    # 0.005
D_TRAIL_DIST = cfg.d_trail_distance   # 0.002
S_TRAIL_ACT = 0.020   # s_trail_activate removed from production config
S_TRAIL_DIST = 0.002  # s_trail_distance removed from production config

MAX_CONCURRENT = cfg.max_concurrent  # 10
MOM_PERIOD = cfg.momentum_period     # 8

COST_DIRECTIONAL = 0.0008
COST_STRADDLE = 0.0016
WARMUP = 200

BSM_WINDOW = 10  # bars_since_min lookback window

CACHE_DIR = ROOT / "data" / "cache_extended"

print("=" * 70)
print("r62: VR Momentum Filter Analysis")
print("=" * 70)
print(f"  Config: S_VR={S_VR}, S_DUR={S_DUR}, D_VR={D_VR}")
print(f"  S_trail: act={S_TRAIL_ACT*100:.1f}%  dist={S_TRAIL_DIST*100:.1f}%")
print(f"  D_trail: act={D_TRAIL_ACT*100:.1f}%  dist={D_TRAIL_DIST*100:.1f}%")
assert S_VR == 0.57, f"S_VR mismatch: {S_VR}"
assert D_VR == 0.5, f"D_VR mismatch: {D_VR}"
assert S_TRAIL_ACT == 0.020
assert D_TRAIL_ACT == 0.040


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data_for_symbols(cache_dir: Path) -> tuple[dict, dict, dict, list[str], float]:
    """Load 60-symbol market data from cache_extended.

    Returns:
        (data_15m, data_1h, data_4h, syms, data_years)
    """
    whitelist = set(cfg.symbol_whitelist)
    min_bars = int(5.0 * 365 * 24 * 4)
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if nb >= min_bars and (sym not in seen or nb > seen[sym][1]):
            seen[sym] = (f, nb)

    data_15m: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        data_15m[sym] = {k: npz[k] for k in npz.files}

    if not data_15m:
        return {}, {}, {}, [], 0.0

    max_bars = max(len(d["close"]) for d in data_15m.values())
    for d in data_15m.values():
        d["_offset"] = max_bars - len(d["close"])
        if "mom_8" not in d:
            c = d["close"]
            mom = np.zeros(len(c))
            mom[MOM_PERIOD:] = c[MOM_PERIOD:] / c[:-MOM_PERIOD] - 1.0
            d["mom_8"] = mom

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_15m, data_1h, data_4h], [(SP, LP)])
    precompute(data_15m, vr_sp=SP, vr_lp=LP)
    precompute(data_1h, vr_sp=SP, vr_lp=LP)
    precompute(data_4h, vr_sp=SP, vr_lp=LP)

    syms = list(data_15m.keys())
    data_years = max_bars / (4 * 24 * 365)
    return data_15m, data_1h, data_4h, syms, data_years


# ── VR momentum metrics (no lookahead) ───────────────────────────────────────

def compute_vr_momentum(vr: np.ndarray, bar_i: int, window: int = BSM_WINDOW) -> tuple[float, int]:
    """Compute VR momentum metrics at entry bar.

    Both metrics use only vr[0:bar_i+1] — no future data.

    Args:
        vr: VR array for this TF.
        bar_i: Entry bar index.
        window: Lookback window for bars_since_min.

    Returns:
        (dVR, bars_since_min):
            dVR = vr[bar_i] - vr[bar_i-1], or 0.0 if bar_i < 1 or NaN.
            bars_since_min = how many bars ago VR reached its minimum
                within vr[max(0,bar_i-window):bar_i+1].
                0 = current bar is the minimum.
    """
    # dVR: first difference — VR rising (>0) or falling (<0)?
    if bar_i < 1 or np.isnan(vr[bar_i]) or np.isnan(vr[bar_i - 1]):
        dvr = 0.0
    else:
        dvr = float(vr[bar_i]) - float(vr[bar_i - 1])

    # bars_since_min: how many bars since VR was at its local minimum
    start = max(0, bar_i - window)
    segment = vr[start : bar_i + 1]
    # Replace NaN with inf so argmin ignores them
    safe_seg = np.where(np.isnan(segment), np.inf, segment)
    if np.all(np.isinf(safe_seg)):
        bsm = 0
    else:
        min_pos = int(np.argmin(safe_seg))
        bsm = len(segment) - 1 - min_pos  # bars since minimum

    return dvr, bsm


# ── Tagged trade collection ───────────────────────────────────────────────────

def collect_s_d8_tagged(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    syms: list[str],
) -> list[tuple]:
    """Collect S_d8 trades tagged with VR momentum metrics.

    Returns:
        List of (sym, entry_15m, exit_15m, pnl, sq_dur, dvr, bars_since_min)
    """
    all_trades: list[tuple] = []
    mh = 192  # max hold in 15m bars

    for sym in syms:
        d15 = data_15m.get(sym)
        d1h = data_1h.get(sym)
        d4h = data_4h.get(sym)
        if d15 is None or d1h is None or d4h is None:
            continue

        for ratio, d in [(1, d15), (4, d1h), (16, d4h)]:
            if len(d.get("close", [])) < WARMUP + 10:
                continue
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(VR_KEY, d.get("vr", np.zeros(len(c))))

            warmup = WARMUP // ratio
            sig = gen_vr_dur_signal(vr, S_VR, S_DUR, warmup)

            pnl_arr, ent_arr, ext_arr = sim_straddle_trailing(
                c, h, lo, sig, S_SL, S_TP, mh // ratio, COST_STRADDLE,
                S_TRAIL_ACT, S_TRAIL_DIST,
            )

            for k in range(len(pnl_arr)):
                bar_i = int(ent_arr[k])
                exit_bar = int(ext_arr[k])

                # Squeeze duration at entry (no lookahead)
                sq_dur = 0
                for back in range(bar_i, max(0, bar_i - 60), -1):
                    if vr[back] < S_VR:
                        sq_dur += 1
                    else:
                        break

                # VR momentum metrics
                dvr, bsm = compute_vr_momentum(vr, bar_i)

                entry_15m = bar_i * ratio
                exit_15m = exit_bar * ratio + (ratio - 1)
                all_trades.append((
                    sym, entry_15m, exit_15m, float(pnl_arr[k]),
                    sq_dur, dvr, bsm,
                ))

    return all_trades


def collect_d_mom8_tagged(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    syms: list[str],
) -> list[tuple]:
    """Collect D_mom8 trades tagged with VR momentum metrics.

    Returns:
        List of (sym, entry_15m, exit_15m, pnl, sq_dur=0, dvr, bars_since_min, direction)
    """
    all_trades: list[tuple] = []
    mh = 192

    for sym in syms:
        d15 = data_15m.get(sym)
        d1h = data_1h.get(sym)
        d4h = data_4h.get(sym)
        if d15 is None or d1h is None or d4h is None:
            continue

        for ratio, d in [(1, d15), (4, d1h), (16, d4h)]:
            if len(d.get("close", [])) < WARMUP + 10:
                continue
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(VR_KEY, d.get("vr", np.zeros(len(c))))

            warmup = WARMUP // ratio
            sig = gen_vr_signal(vr, D_VR, warmup)
            mom_key = "mom_8"
            if mom_key not in d:
                mom = np.zeros(len(c))
                mom[MOM_PERIOD:] = c[MOM_PERIOD:] / c[:-MOM_PERIOD] - 1.0
                d[mom_key] = mom
            dire = make_direction_momentum(sig, d[mom_key])

            pnl_arr, ent_arr, ext_arr = sim_directional_trailing(
                c, h, lo, sig, dire, D_SL, D_TP, mh // ratio, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )

            for k in range(len(pnl_arr)):
                bar_i = int(ent_arr[k])
                exit_bar = int(ext_arr[k])

                # VR momentum metrics
                dvr, bsm = compute_vr_momentum(vr, bar_i)
                direction = "LONG" if (bar_i < len(dire) and dire[bar_i] > 0) else "SHORT"

                entry_15m = bar_i * ratio
                exit_15m = exit_bar * ratio + (ratio - 1)
                all_trades.append((
                    sym, entry_15m, exit_15m, float(pnl_arr[k]),
                    0, dvr, bsm, direction,
                ))

    return all_trades


# ── Analysis helpers ──────────────────────────────────────────────────────────

def wr_stats(pnls: list[float]) -> tuple[float, float, float]:
    """Return (WR fraction, PF, EV_bp)."""
    if not pnls:
        return 0.0, 0.0, 0.0
    arr = np.array(pnls)
    wr = float(np.mean(arr > 0))
    wins = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")
    ev_bp = float(np.mean(arr)) * 10000
    return wr, pf, ev_bp


def dvr_bucket_analysis(trades: list[tuple], label: str) -> list[dict]:
    """Bucket S_d8 trades by dVR direction at entry (index 5 = dvr).

    Buckets: strongly falling / falling / rising / strongly rising
    """
    buckets = [
        ("≤-0.020 (strong↓)",  -9.9, -0.020),
        ("-0.020~0 (weak↓)",   -0.020,  0.0),
        ("0~0.020 (weak↑)",     0.0,  0.020),
        (">0.020 (strong↑)",   0.020,  9.9),
    ]
    print(f"\n  {label}")
    print(f"  {'Bucket':>22} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'EV_bp':>8}")
    print("  " + "-" * 58)
    results = []
    for bname, lo, hi in buckets:
        group = [t[3] for t in trades if lo <= t[5] < hi]
        wr, pf, ev = wr_stats(group)
        print(f"  {bname:>22} | {len(group):>6,} | {wr*100:>5.1f}% | {pf:>6.3f} | {ev:>+8.1f}")
        results.append({"name": bname, "N": len(group), "WR": wr, "PF": pf, "EV_bp": ev})
    total_wr, total_pf, total_ev = wr_stats([t[3] for t in trades])
    print(f"  {'TOTAL':>22} | {len(trades):>6,} | {total_wr*100:>5.1f}% | {total_pf:>6.3f} | {total_ev:>+8.1f}")
    return results


def bsm_bucket_analysis(trades: list[tuple], label: str) -> list[dict]:
    """Bucket trades by bars_since_min (index 6 = bsm)."""
    buckets = [
        ("bsm=0 (at bottom)", 0, 1),
        ("bsm=1 (1 bar ago)",  1, 2),
        ("bsm=2",              2, 3),
        ("bsm=3",              3, 4),
        ("bsm=4+",             4, 9999),
    ]
    print(f"\n  {label}")
    print(f"  {'Bucket':>22} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'EV_bp':>8}")
    print("  " + "-" * 58)
    results = []
    for bname, lo, hi in buckets:
        group = [t[3] for t in trades if lo <= t[6] < hi]
        wr, pf, ev = wr_stats(group)
        print(f"  {bname:>22} | {len(group):>6,} | {wr*100:>5.1f}% | {pf:>6.3f} | {ev:>+8.1f}")
        results.append({"name": bname, "N": len(group), "WR": wr, "PF": pf, "EV_bp": ev, "lo": lo})
    total_wr, total_pf, total_ev = wr_stats([t[3] for t in trades])
    print(f"  {'TOTAL':>22} | {len(trades):>6,} | {total_wr*100:>5.1f}% | {total_pf:>6.3f} | {total_ev:>+8.1f}")
    return results


def dvr_threshold_sweep(trades: list[tuple], label: str, data_years: float) -> list[dict]:
    """Sweep dVR threshold — keep only trades where dVR > threshold."""
    thresholds = [-0.030, -0.010, 0.0, 0.003, 0.005, 0.010, 0.015, 0.020]
    total_n = len(trades)
    print(f"\n  {label}  [filter: dVR > threshold]")
    print(f"  {'Threshold':>10} | {'N':>6} | {'%kept':>6} | {'WR':>6} | {'PF':>6} | {'EV_bp':>8}")
    print("  " + "-" * 58)
    results = []
    for thr in thresholds:
        group = [t[3] for t in trades if t[5] > thr]
        n = len(group)
        pct = n / total_n * 100 if total_n else 0
        wr, pf, ev = wr_stats(group)
        print(f"  {thr:>10.3f} | {n:>6,} | {pct:>5.1f}% | {wr*100:>5.1f}% | {pf:>6.3f} | {ev:>+8.1f}")
        results.append({"thr": thr, "N": n, "pct_kept": pct, "WR": wr, "PF": pf, "EV_bp": ev})
    return results


def bsm_threshold_sweep(trades: list[tuple], label: str, data_years: float) -> list[dict]:
    """Sweep bars_since_min threshold — keep trades where bsm >= K."""
    thresholds = [0, 1, 2, 3, 4, 5]
    total_n = len(trades)
    print(f"\n  {label}  [filter: bars_since_min >= K]")
    print(f"  {'K':>6} | {'N':>6} | {'%kept':>6} | {'WR':>6} | {'PF':>6} | {'EV_bp':>8}")
    print("  " + "-" * 55)
    results = []
    for k in thresholds:
        group = [t[3] for t in trades if t[6] >= k]
        n = len(group)
        pct = n / total_n * 100 if total_n else 0
        wr, pf, ev = wr_stats(group)
        print(f"  {k:>6} | {n:>6,} | {pct:>5.1f}% | {wr*100:>5.1f}% | {pf:>6.3f} | {ev:>+8.1f}")
        results.append({"k": k, "N": n, "pct_kept": pct, "WR": wr, "PF": pf, "EV_bp": ev})
    return results


def evaluate_portfolio(
    s_trades: list[tuple],
    d_trades: list[tuple],
    s_filter_fn,
    d_filter_fn,
    data_years: float,
    label: str,
) -> dict:
    """Apply filters, run portfolio_sim on combined, return metrics."""
    s_filtered = [(t[0], t[1], t[2], t[3]) for t in s_trades if s_filter_fn(t)]
    d_filtered = [(t[0], t[1], t[2], t[3]) for t in d_trades if d_filter_fn(t)]
    combined = s_filtered + d_filtered
    pnl = portfolio_sim(combined, MAX_CONCURRENT)
    if len(pnl) == 0:
        return {"label": label, "N": 0, "WR": 0.0, "PF": 0.0, "CI_lo": 0.0, "CI_hi": 0.0}
    m = compute_metrics(pnl, data_years)
    bt = bootstrap_pf(pnl, n_boot=300)
    return {
        "label": label,
        "N": m["N"],
        "WR": m["WR"],
        "PF": m["PF"],
        "EV_bp": m["EV_bp"],
        "calmar": m["calmar"],
        "max_dd_pct": m["max_dd_pct"],
        "ann_ret_pct": m["ann_ret_pct"],
        "CI_lo": bt["CI_lo"],
        "CI_hi": bt["CI_hi"],
    }


def yearly_breakdown(trades: list[tuple], data_years: float, label: str) -> None:
    """Print per-year WR/N for a filtered trade set."""
    bars_per_year = 4 * 24 * 365  # 15m bars
    years = int(np.ceil(data_years))
    print(f"\n  Yearly breakdown — {label}")
    print(f"  {'Year':>6} | {'N':>6} | {'WR':>6} | {'PF':>6}")
    print("  " + "-" * 32)
    for yr in range(years):
        lo_bar = yr * bars_per_year
        hi_bar = (yr + 1) * bars_per_year
        group = [t[3] for t in trades if lo_bar <= t[1] < hi_bar]
        if len(group) < 5:
            continue
        wr, pf, _ = wr_stats(group)
        yr_label = 2020 + yr
        print(f"  {yr_label:>6} | {len(group):>6,} | {wr*100:>5.1f}% | {pf:>6.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\nLoading 60 symbols...")
    d15, d1h, d4h, syms, data_years = load_data_for_symbols(CACHE_DIR)
    print(f"  {len(syms)} syms | {data_years:.2f}yr")

    print("\nCollecting S_d8 tagged trades (with VR momentum)...")
    s_trades = collect_s_d8_tagged(d15, d1h, d4h, syms)
    print(f"  S_d8 raw trades: {len(s_trades):,}")

    print("\nCollecting D_mom8 tagged trades (with VR momentum)...")
    d_trades = collect_d_mom8_tagged(d15, d1h, d4h, syms)
    print(f"  D_mom8 raw trades: {len(d_trades):,}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Phase A: S_d8 WR by dVR bucket ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase A: S_d8 WR by VR Momentum (dVR) at Entry")
    print("=" * 70)
    dvr_a = dvr_bucket_analysis(s_trades, "S_d8 dVR buckets")
    dvr_d = dvr_bucket_analysis(d_trades, "D_mom8 dVR buckets")

    # ── Phase B: S_d8 WR by bars_since_min ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase B: S_d8 WR by Bars Since VR Minimum (bars_since_min)")
    print("=" * 70)
    bsm_a = bsm_bucket_analysis(s_trades, "S_d8 bars_since_min buckets")
    bsm_d = bsm_bucket_analysis(d_trades, "D_mom8 bars_since_min buckets")

    # ── Phase C: Threshold sweeps ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase C: Threshold Sweep (raw, before portfolio_sim)")
    print("=" * 70)
    dvr_sweep_s = dvr_threshold_sweep(s_trades, "C1: S_d8 dVR sweep", data_years)
    dvr_sweep_d = dvr_threshold_sweep(d_trades, "C2: D_mom8 dVR sweep", data_years)
    bsm_sweep_s = bsm_threshold_sweep(s_trades, "C3: S_d8 bars_since_min sweep", data_years)
    bsm_sweep_d = bsm_threshold_sweep(d_trades, "C4: D_mom8 bars_since_min sweep", data_years)

    # ── Phase D: Portfolio-level evaluation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase D: Portfolio-level Evaluation (combined S_d8 + D_mom8)")
    print("=" * 70)
    print(f"  {'Label':<38} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'CI':>14} | {'Calmar':>8} | {'DD':>6}")
    print("  " + "-" * 105)

    no_filter = lambda t: True  # noqa: E731

    candidates = [
        ("Baseline (no filter)", no_filter, no_filter),
        # dVR > 0 (rising at all)
        ("S_d8: dVR>0", lambda t: t[5] > 0.0, no_filter),
        ("S+D: dVR>0", lambda t: t[5] > 0.0, lambda t: t[5] > 0.0),
        # dVR > 0.005
        ("S_d8: dVR>0.005", lambda t: t[5] > 0.005, no_filter),
        ("S+D: dVR>0.005", lambda t: t[5] > 0.005, lambda t: t[5] > 0.005),
        # dVR > 0.010
        ("S_d8: dVR>0.010", lambda t: t[5] > 0.010, no_filter),
        ("S+D: dVR>0.010", lambda t: t[5] > 0.010, lambda t: t[5] > 0.010),
        # bars_since_min >= 1 (VR bounced at least 1 bar)
        ("S_d8: bsm>=1", lambda t: t[6] >= 1, no_filter),
        ("S+D: bsm>=1", lambda t: t[6] >= 1, lambda t: t[6] >= 1),
        # bars_since_min >= 2
        ("S_d8: bsm>=2", lambda t: t[6] >= 2, no_filter),
        ("S+D: bsm>=2", lambda t: t[6] >= 2, lambda t: t[6] >= 2),
        # Combo: dVR > 0 AND bsm >= 1
        ("S_d8: dVR>0 + bsm>=1", lambda t: t[5] > 0.0 and t[6] >= 1, no_filter),
        ("S+D: dVR>0 + bsm>=1", lambda t: t[5] > 0.0 and t[6] >= 1,
                                 lambda t: t[5] > 0.0 and t[6] >= 1),
        # Combo: dVR > 0.005 AND bsm >= 1
        ("S_d8: dVR>0.005 + bsm>=1", lambda t: t[5] > 0.005 and t[6] >= 1, no_filter),
        # Rising filter on S only (D keeps all)
        ("S_d8: dVR>0 + bsm>=1 (D=all)", lambda t: t[5] > 0.0 and t[6] >= 1, no_filter),
    ]

    results = []
    best = None
    for label, sfn, dfn in candidates:
        r = evaluate_portfolio(s_trades, d_trades, sfn, dfn, data_years, label)
        ci = f"[{r['CI_lo']:.3f},{r['CI_hi']:.3f}]"
        calmar = r.get("calmar", 0.0) or 0.0
        dd = r.get("max_dd_pct", 0.0) or 0.0
        wr = r.get("WR", 0.0)
        print(f"  {label:<38} | {r['N']:>6,} | {wr:>5.1f}% | {r['PF']:>6.3f} | {ci:>14} | {calmar:>8.1f} | {dd:>5.1f}%")
        results.append(r)
        if best is None or (r["N"] >= 6200 and r["PF"] > best["PF"] and r["WR"] >= 43.0):
            best = r

    # Yearly breakdown for baseline and best candidate
    print("\n" + "=" * 70)
    print("Phase D.2: Yearly Breakdown")
    print("=" * 70)
    baseline_combined = [(t[0], t[1], t[2], t[3]) for t in s_trades + d_trades]
    yearly_breakdown(baseline_combined, data_years, "Baseline (all S+D trades, raw)")

    if best and best["label"] != "Baseline (no filter)":
        # Find the best filter functions
        for label, sfn, dfn in candidates:
            if label == best["label"]:
                best_s = [(t[0], t[1], t[2], t[3]) for t in s_trades if sfn(t)]
                best_d = [(t[0], t[1], t[2], t[3]) for t in d_trades if dfn(t)]
                yearly_breakdown(best_s + best_d, data_years, best["label"])
                break

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("r62 SUMMARY")
    print("=" * 70)
    baseline_r = results[0]
    print(f"  Baseline: N={baseline_r['N']:,} WR={baseline_r['WR']:.1f}% PF={baseline_r['PF']:.3f}")
    if best:
        print(f"  Best:     N={best['N']:,} WR={best['WR']:.1f}% PF={best['PF']:.3f} — {best['label']}")

    # Print top 3 by PF (with N constraint)
    valid = sorted([r for r in results if r["N"] >= 6200], key=lambda x: -x["PF"])
    print(f"\n  Top candidates (N≥6200, sorted by PF):")
    for r in valid[:5]:
        print(f"    {r['label']:<42} N={r['N']:,} WR={r['WR']:.1f}% PF={r['PF']:.3f} CI=[{r['CI_lo']:.3f},{r['CI_hi']:.3f}]")


if __name__ == "__main__":
    main()
