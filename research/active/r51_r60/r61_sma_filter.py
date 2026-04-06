"""r61: S_d8 entry-condition deep analysis — squeeze duration & SMA distance.

r60 finding: S_d8 score=0 (all conviction conditions False) has WR=68.1%.
This means trades where:
  - E2=False: price FAR from 20-bar SMA (>0.5%)
  - A3=False: squeeze duration SHORT (<16 bars)
...outperform trades where price is near SMA or squeeze has run long.

r61 Goal: Identify the single best entry filter that improves S_d8 WR
toward 50%+ while maintaining N≥8,000 total (S_d8+D_mom8 combined).

Research vectors:
  A: S_d8 WR by squeeze duration bucket [1-3, 4-7, 8-15, 16+]
  B: S_d8 WR by SMA-distance bucket [<0.5%, 0.5-1%, 1-2%, >2%]
  C: S_d8 WR by VR depth bucket (how far VR below s_vr_threshold)
  D: Sweep filter thresholds → find optimal single filter for WR/N balance
  E: Yearly breakdown for best candidate (no regression)

Baseline (60 syms, 6yr, Combo2): S_d8 N≈7,706 WR≈48.5%, Combined N≈9,934
Target: S_d8 WR≥50%, or Combined WR≥49% with N≥8,000

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/r51_r60/r61_sma_filter.py
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

# ── Config (all from production config.py) ────────────────────────────────────
cfg = VolStraddleConfig()

SP = cfg.vr_short  # 8
LP = cfg.vr_long  # 56
VR_KEY = f"vr_{SP}_{LP}"

S_VR = 0.57   # s_vr_threshold removed from production config
S_DUR = 3     # s_squeeze_dur removed from production config
D_VR = cfg.d_vr_threshold  # 0.5

D_SL = cfg.d_sl  # 0.01
S_SL = 0.015     # s_sl removed from production config
D_TP = cfg.d_tp  # 0.12
S_TP = 0.05      # s_tp removed from production config

D_TRAIL_ACT = cfg.d_trail_activate  # 0.005
D_TRAIL_DIST = cfg.d_trail_distance  # 0.002
S_TRAIL_ACT = 0.020   # s_trail_activate removed from production config
S_TRAIL_DIST = 0.002  # s_trail_distance removed from production config

MAX_CONCURRENT = cfg.max_concurrent  # 10
COST_DIRECTIONAL = 0.0008
COST_STRADDLE = 0.0016
WARMUP = 200

E2_SMA_PERIOD = cfg.e2_sma_period  # 20
MOM_PERIOD = cfg.momentum_period  # 8

CACHE_DIR = ROOT / "data" / "cache_extended"

print("=" * 70)
print("r61: S_d8 Entry-Condition Deep Analysis")
print("=" * 70)
print(f"  Config: S_VR={S_VR}, S_DUR={S_DUR}, D_VR={D_VR}")
print(f"  S_trail: act={S_TRAIL_ACT*100:.1f}%  dist={S_TRAIL_DIST*100:.1f}%")
print(f"  D_trail: act={D_TRAIL_ACT*100:.1f}%  dist={D_TRAIL_DIST*100:.1f}%")
print(f"  SMA period={E2_SMA_PERIOD}  MOM period={MOM_PERIOD}")
assert S_VR == 0.57, f"S_VR mismatch: {S_VR}"
assert D_VR == 0.5, f"D_VR mismatch: {D_VR}"
assert S_TRAIL_ACT == 0.020
assert D_TRAIL_ACT == 0.040


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data_for_symbols(cache_dir: Path) -> tuple[dict, dict, dict, list[str], float]:
    """Load 60-symbol market data using cache_extended pattern."""
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
        # mom_8 for D_mom8 signal
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


# ── Extended trade collection ─────────────────────────────────────────────────

def compute_entry_attrs(
    close: np.ndarray,
    vr: np.ndarray,
    bar_idx: int,
    is_straddle: bool,
    squeeze_dur: int,
    sma_period: int = 20,
) -> tuple[float, float]:
    """Compute SMA distance and VR depth at entry bar.

    Returns:
        (sma_dist_pct, vr_depth): SMA distance as fraction (e.g. 0.015=1.5%),
            VR depth = how far VR is below the threshold (s_vr or d_vr).
    """
    i = bar_idx
    sma_dist = 0.0
    if i >= sma_period - 1:
        sma = float(np.mean(close[i - sma_period + 1 : i + 1]))
        if sma > 0:
            sma_dist = abs(float(close[i]) - sma) / sma

    vr_val = float(vr[i]) if not np.isnan(vr[i]) else 0.0
    threshold = S_VR if is_straddle else D_VR
    vr_depth = max(0.0, threshold - vr_val)  # how far below threshold

    return sma_dist, vr_depth


def collect_s_d8_tagged(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    syms: list[str],
) -> list[tuple]:
    """Collect S_d8 trades tagged with entry attributes.

    Returns:
        List of (sym, entry_15m, exit_15m, pnl, squeeze_dur, sma_dist, vr_depth)
    """
    all_trades: list[tuple] = []
    mh = 192  # max hold bars (15m)

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

            # Precompute SMA for this TF
            sma_arr = np.full(len(c), np.nan)
            sp = E2_SMA_PERIOD
            for idx in range(sp - 1, len(c)):
                sma_arr[idx] = float(np.mean(c[idx - sp + 1 : idx + 1]))

            for k in range(len(pnl_arr)):
                bar_i = int(ent_arr[k])
                exit_bar = int(ext_arr[k])

                # Squeeze duration at entry
                sq_dur = 0
                for back in range(bar_i, max(0, bar_i - 60), -1):
                    if vr[back] < S_VR:
                        sq_dur += 1
                    else:
                        break

                # SMA distance at entry
                sma_dist = 0.0
                if not np.isnan(sma_arr[bar_i]) and sma_arr[bar_i] > 0:
                    sma_dist = abs(float(c[bar_i]) - sma_arr[bar_i]) / sma_arr[bar_i]

                # VR depth at entry
                vr_val = float(vr[bar_i]) if not np.isnan(vr[bar_i]) else 0.0
                vr_depth = max(0.0, S_VR - vr_val)

                entry_15m = bar_i * ratio
                exit_15m = exit_bar * ratio + (ratio - 1)
                all_trades.append((
                    sym, entry_15m, exit_15m, float(pnl_arr[k]),
                    sq_dur, sma_dist, vr_depth,
                ))

    return all_trades


def collect_d_mom8_tagged(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    syms: list[str],
) -> list[tuple]:
    """Collect D_mom8 trades tagged with entry attributes.

    Returns:
        List of (sym, entry_15m, exit_15m, pnl, squeeze_dur=0, sma_dist, vr_depth, direction)
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

            # Precompute SMA
            sma_arr = np.full(len(c), np.nan)
            sp = E2_SMA_PERIOD
            for idx in range(sp - 1, len(c)):
                sma_arr[idx] = float(np.mean(c[idx - sp + 1 : idx + 1]))

            for k in range(len(pnl_arr)):
                bar_i = int(ent_arr[k])
                exit_bar = int(ext_arr[k])

                sma_dist = 0.0
                if not np.isnan(sma_arr[bar_i]) and sma_arr[bar_i] > 0:
                    sma_dist = abs(float(c[bar_i]) - sma_arr[bar_i]) / sma_arr[bar_i]

                vr_val = float(vr[bar_i]) if not np.isnan(vr[bar_i]) else 0.0
                vr_depth = max(0.0, D_VR - vr_val)

                direction = "LONG" if (bar_i < len(dire) and dire[bar_i] > 0) else "SHORT"
                entry_15m = bar_i * ratio
                exit_15m = exit_bar * ratio + (ratio - 1)
                all_trades.append((
                    sym, entry_15m, exit_15m, float(pnl_arr[k]),
                    0, sma_dist, vr_depth, direction,
                ))

    return all_trades


# ── Analysis helpers ──────────────────────────────────────────────────────────

def wr_stats(pnls: list[float]) -> tuple[float, float, float]:
    """Return (WR, PF, EV_bp)."""
    if not pnls:
        return 0.0, 0.0, 0.0
    arr = np.array(pnls)
    wr = float(np.mean(arr > 0))
    wins = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")
    ev_bp = float(np.mean(arr)) * 10000
    return wr, pf, ev_bp


def bucket_analysis(
    trades: list[tuple],
    attr_idx: int,
    buckets: list[tuple[str, float, float]],
    label: str,
) -> list[dict]:
    """Group trades by attribute bucket and print WR breakdown.

    Args:
        trades: List of trade tuples.
        attr_idx: Index of the attribute in each tuple (e.g. sq_dur=4, sma_dist=5).
        buckets: List of (name, lo, hi) — hi is exclusive.
        label: Section label for printing.

    Returns:
        List of dicts with bucket stats.
    """
    print(f"\n  {label}")
    print(f"  {'Bucket':>18} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'EV_bp':>8}")
    print("  " + "-" * 50)
    results = []
    for bname, lo, hi in buckets:
        group = [t[3] for t in trades if lo <= t[attr_idx] < hi]
        wr, pf, ev = wr_stats(group)
        print(f"  {bname:>18} | {len(group):>6,} | {wr*100:>5.1f}% | {pf:>6.3f} | {ev:>+8.1f}")
        results.append({"name": bname, "N": len(group), "WR": wr, "PF": pf, "EV_bp": ev,
                         "lo": lo, "hi": hi})
    total_wr, total_pf, total_ev = wr_stats([t[3] for t in trades])
    print(f"  {'TOTAL':>18} | {len(trades):>6,} | {total_wr*100:>5.1f}% | {total_pf:>6.3f} | {total_ev:>+8.1f}")
    return results


def threshold_sweep(
    trades: list[tuple],
    attr_idx: int,
    thresholds: list[float],
    mode: str,  # "above" or "below"
    label: str,
    data_years: float,
) -> list[dict]:
    """Sweep filter threshold on an attribute, show WR/N/PF progression.

    Args:
        mode: "above" keeps trades where attr > threshold (far from SMA, short squeeze).
              "below" keeps trades where attr < threshold.
    """
    print(f"\n  {label}  [mode={mode}]")
    print(f"  {'Threshold':>10} | {'N':>6} | {'%kept':>6} | {'WR':>6} | {'PF':>6} | {'EV_bp':>8}")
    print("  " + "-" * 55)
    total_n = len(trades)
    results = []
    for thr in thresholds:
        if mode == "above":
            group = [t[3] for t in trades if t[attr_idx] > thr]
        else:
            group = [t[3] for t in trades if t[attr_idx] < thr]
        n = len(group)
        pct_kept = n / total_n * 100 if total_n else 0
        wr, pf, ev = wr_stats(group)
        print(f"  {thr:>10.3f} | {n:>6,} | {pct_kept:>5.1f}% | {wr*100:>5.1f}% | {pf:>6.3f} | {ev:>+8.1f}")
        results.append({"thr": thr, "N": n, "pct_kept": pct_kept, "WR": wr, "PF": pf, "EV_bp": ev})
    return results


def evaluate_with_portfolio_sim(
    s_trades: list[tuple],
    d_trades: list[tuple],
    s_filter_fn,
    d_filter_fn,
    data_years: float,
    label: str,
) -> dict:
    """Apply filters, run portfolio_sim on combined, return compute_metrics result."""
    # Filter S_d8
    s_filtered = [(t[0], t[1], t[2], t[3]) for t in s_trades if s_filter_fn(t)]
    # Filter D_mom8 (keep all by default or apply d_filter)
    d_filtered = [(t[0], t[1], t[2], t[3]) for t in d_trades if d_filter_fn(t)]

    combined = s_filtered + d_filtered
    pnl = portfolio_sim(combined, MAX_CONCURRENT)
    if len(pnl) == 0:
        return {"label": label, "N": 0, "WR": 0.0, "PF": 0.0}

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


def yearly_breakdown(trades: list[tuple], max_bars: int, data_years: float) -> None:
    """Print per-year WR/N/PF for a filtered trade set."""
    # Compute year boundaries (approx: 15m bars per year = 4*24*365=35040)
    bars_per_year = 4 * 24 * 365  # 15m bars
    years = int(np.ceil(data_years))
    print(f"  {'Year':>6} | {'N':>6} | {'WR':>6} | {'PF':>6}")
    print("  " + "-" * 32)
    for yr in range(years):
        lo_bar = yr * bars_per_year
        hi_bar = (yr + 1) * bars_per_year
        # entry_15m is index 1
        group = [t[3] for t in trades if lo_bar <= t[1] < hi_bar]
        if len(group) < 5:
            continue
        wr, pf, _ = wr_stats(group)
        # Approximate year label: 2020 + yr
        yr_label = 2020 + yr
        print(f"  {yr_label:>6} | {len(group):>6,} | {wr*100:>5.1f}% | {pf:>6.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\nLoading 60 symbols...")
    d15, d1h, d4h, syms, data_years = load_data_for_symbols(CACHE_DIR)
    print(f"  {len(syms)} syms | {data_years:.2f}yr")

    print("\nCollecting S_d8 tagged trades...")
    s_trades = collect_s_d8_tagged(d15, d1h, d4h, syms)
    print(f"  S_d8 raw trades: {len(s_trades):,}")

    print("\nCollecting D_mom8 tagged trades...")
    d_trades = collect_d_mom8_tagged(d15, d1h, d4h, syms)
    print(f"  D_mom8 raw trades: {len(d_trades):,}")

    print(f"\n  Elapsed: {time.time()-t0:.1f}s")

    # ── Phase A: S_d8 WR by squeeze duration ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase A: S_d8 WR by Squeeze Duration at Entry")
    print("=" * 70)
    sq_buckets = [
        ("1-3 bars", 1, 4),
        ("4-7 bars", 4, 8),
        ("8-15 bars", 8, 16),
        ("16-31 bars", 16, 32),
        ("32+ bars", 32, 9999),
    ]
    sq_results = bucket_analysis(s_trades, 4, sq_buckets, "Squeeze Duration → WR")

    # ── Phase B: S_d8 WR by SMA distance ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase B: S_d8 WR by SMA Distance at Entry")
    print("=" * 70)
    sma_buckets = [
        ("<0.5% (E2=T)", 0.0, 0.005),
        ("0.5-1.0%", 0.005, 0.010),
        ("1.0-2.0%", 0.010, 0.020),
        ("2.0-3.0%", 0.020, 0.030),
        (">3.0%", 0.030, 9.999),
    ]
    sma_results = bucket_analysis(s_trades, 5, sma_buckets, "SMA Distance → WR")

    # ── Phase C: S_d8 WR by VR depth ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase C: S_d8 WR by VR Depth (how far below S_VR threshold)")
    print("=" * 70)
    vr_buckets = [
        ("<0.05 shallow", 0.0, 0.05),
        ("0.05-0.10", 0.05, 0.10),
        ("0.10-0.20", 0.10, 0.20),
        (">0.20 deep", 0.20, 9.99),
    ]
    vr_results = bucket_analysis(s_trades, 6, vr_buckets, "VR Depth → WR")

    # ── Phase D: Threshold sweeps ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase D: Threshold Sweeps (S_d8 only, raw — before portfolio_sim)")
    print("=" * 70)

    # D1: SMA distance above threshold (keep trades where price is FAR from SMA)
    sma_above_thrs = [0.0, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]
    sma_above = threshold_sweep(
        s_trades, 5, sma_above_thrs, "above",
        "D1: S_d8 with SMA-distance > threshold (price FAR from SMA)", data_years
    )

    # D2: Squeeze duration below threshold (keep SHORT squeezes)
    sq_below_thrs = [99, 31, 16, 12, 8, 6, 4]
    sq_below = threshold_sweep(
        s_trades, 4, sq_below_thrs, "below",
        "D2: S_d8 with squeeze_dur < threshold (SHORT squeezes only)", data_years
    )

    # D3: VR depth above threshold (keep DEEP squeezes by VR)
    vr_above_thrs = [0.0, 0.05, 0.10, 0.15, 0.20]
    vr_above = threshold_sweep(
        s_trades, 6, vr_above_thrs, "above",
        "D3: S_d8 with VR_depth > threshold (deeper squeeze)", data_years
    )

    # D4: SMA distance below threshold (keep price CLOSE to SMA — opposite hypothesis)
    sma_below_thrs = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
    sma_below = threshold_sweep(
        s_trades, 5, sma_below_thrs, "below",
        "D4: S_d8 with SMA-distance < threshold (price CLOSE to SMA)", data_years
    )

    # ── Phase E: Portfolio-level evaluation of top candidates ─────────────────
    print("\n" + "=" * 70)
    print("Phase E: Portfolio-level Evaluation (combined S_d8 + D_mom8)")
    print("=" * 70)
    print(f"  {'Label':<38} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'CI':>14} | {'Calmar':>8} | {'DD':>6}")
    print("  " + "-" * 100)

    # Baseline: no filter
    no_filter = lambda t: True  # noqa: E731

    candidates = [
        # (label, s_filter, d_filter)
        ("Baseline (no filter)", no_filter, no_filter),
    ]

    # Best SMA-far candidates from sweep
    for thr, label_suffix in [(0.005, "SMA>0.5%"), (0.010, "SMA>1.0%"), (0.020, "SMA>2.0%")]:
        candidates.append((
            f"S_d8 {label_suffix}",
            lambda t, _thr=thr: t[5] > _thr,
            no_filter,
        ))

    # Best squeeze-short candidates
    for thr, label_suffix in [(16, "sq_dur<16"), (8, "sq_dur<8"), (4, "sq_dur<4")]:
        candidates.append((
            f"S_d8 {label_suffix}",
            lambda t, _thr=thr: t[4] < _thr,
            no_filter,
        ))

    # VR depth
    for thr, label_suffix in [(0.10, "VR_depth>0.10"), (0.15, "VR_depth>0.15")]:
        candidates.append((
            f"S_d8 {label_suffix}",
            lambda t, _thr=thr: t[6] > _thr,
            no_filter,
        ))

    # Combined filters (SMA + squeeze_dur)
    candidates.append((
        "S_d8 SMA>0.5% + sq_dur<16",
        lambda t: t[5] > 0.005 and t[4] < 16,
        no_filter,
    ))
    candidates.append((
        "S_d8 SMA>1.0% + sq_dur<16",
        lambda t: t[5] > 0.010 and t[4] < 16,
        no_filter,
    ))
    candidates.append((
        "S_d8 SMA>0.5% + sq_dur<8",
        lambda t: t[5] > 0.005 and t[4] < 8,
        no_filter,
    ))

    best = None
    for label, s_fn, d_fn in candidates:
        r = evaluate_with_portfolio_sim(
            s_trades, d_trades, s_fn, d_fn, data_years, label
        )
        ci_str = f"[{r.get('CI_lo', 0):.3f},{r.get('CI_hi', 0):.3f}]"
        print(
            f"  {r['label']:<38} | {r['N']:>6,} | {r['WR']:>5.1f}% | {r['PF']:>6.3f} | "
            f"{ci_str:>14} | {r.get('calmar', 0):>8.1f} | {r.get('max_dd_pct', 0):>5.1f}%"
        )
        if best is None or (r["N"] >= 8000 and r["WR"] > best.get("WR", 0)):
            best = r

    # ── Phase F: Yearly breakdown of best candidate ───────────────────────────
    if best and best["label"] != "Baseline (no filter)":
        print("\n" + "=" * 70)
        print(f"Phase F: Yearly Breakdown — {best['label']}")
        print("=" * 70)
        # Reconstruct filtered trades for best
        label = best["label"]
        # Find matching candidate
        for cand_label, s_fn, d_fn in candidates:
            if cand_label == label:
                s_filtered = [t for t in s_trades if s_fn(t)]
                d_filtered = [t for t in d_trades if d_fn(t)]
                print(f"\n  S_d8 filtered: {len(s_filtered):,} trades")
                yearly_breakdown(s_filtered, 0, data_years)
                print(f"\n  D_mom8 filtered: {len(d_filtered):,} trades")
                yearly_breakdown(d_filtered, 0, data_years)
                break
    else:
        print("\n  (Baseline is already best — no filter improves WR + N simultaneously)")
        # Still show baseline yearly breakdown
        print("\n" + "=" * 70)
        print("Phase F: Yearly Breakdown — S_d8 Baseline")
        print("=" * 70)
        yearly_breakdown(s_trades, 0, data_years)

    # ── D_mom8 direction analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase G: D_mom8 Direction Analysis (for completeness)")
    print("=" * 70)
    long_trades = [t for t in d_trades if t[7] == "LONG"]
    short_trades = [t for t in d_trades if t[7] == "SHORT"]
    print(f"\n  Direction breakdown (raw, before portfolio_sim):")
    print(f"  {'Direction':>10} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'EV_bp':>8}")
    print("  " + "-" * 45)
    for label_d, group in [("LONG", long_trades), ("SHORT", short_trades), ("ALL", d_trades)]:
        wr, pf, ev = wr_stats([t[3] for t in group])
        print(f"  {label_d:>10} | {len(group):>6,} | {wr*100:>5.1f}% | {pf:>6.3f} | {ev:>+8.1f}")

    print("\n" + "=" * 70)
    print("r61 완료.")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
