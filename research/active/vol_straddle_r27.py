#!/usr/bin/env python3
"""
R27: Bayesian RR Distribution Analysis of VolRatio Squeeze
============================================================
Pure research — NO backtesting. Understanding the NATURE of moves after VR squeeze.

Key questions:
1. After VR squeeze, how BIG is the explosive move? (MFE distribution)
2. Is the move unidirectional? (dominant >> subdominant → high RR)
3. Bayesian: P(explosion > X% | VR level, duration)
4. Does lower VR systematically → bigger + more asymmetric explosion?
5. Histograms + Matplotlib PNGs
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Constants ────────────────────────────────────────────────────────────────
MAX_HOLD = 192          # 48h in 15m bars
SAMPLE_INTERVAL = 48    # sample every 12h for baseline
WARMUP = 200
SHORT_P, LONG_P = 7, 50
VR_THRESHOLD = 0.5
MIN_EPISODE_GAP = 96    # 24h minimum between squeeze episodes

VR_BIN_EDGES = np.array([0.0, 0.35, 0.45, 0.55, 0.70, 1.00, 99.0])
VR_BIN_LABELS = [
    "VR<0.35", "0.35≤VR<0.45", "0.45≤VR<0.55",
    "0.55≤VR<0.70", "0.70≤VR<1.0", "VR≥1.0",
]
N_BINS = len(VR_BIN_LABELS)

EXPLOSION_PCT = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
ASYM_THRESHOLDS = [1.5, 2.0, 3.0, 5.0]  # dom/sub > k


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Indicators
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def _atr_pct(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = (high[0] - low[0]) / close[0] if close[0] > 0 else 0.0
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        mx = max(hl, max(hc, lc))
        tr[i] = mx / close[i] if close[i] > 0 else 0.0
    if n >= period:
        atr = 0.0
        for k in range(period):
            atr += tr[k]
        atr /= period
        out[period - 1] = atr
        for i in range(period, n):
            atr = (atr * (period - 1) + tr[i]) / period
            out[i] = atr
    return out


@numba.jit(nopython=True, cache=True)
def compute_vr_array(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sp: int, lp: int,
) -> np.ndarray:
    """VR = ATR(short) / ATR(long) for every bar."""
    atr_s = _atr_pct(close, high, low, sp)
    atr_l = _atr_pct(close, high, low, lp)
    n = len(close)
    vr = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_s[i]) and not np.isnan(atr_l[i]) and atr_l[i] > 0:
            vr[i] = atr_s[i] / atr_l[i]
    return vr


@numba.jit(nopython=True, cache=True)
def compute_squeeze_dur(vr: np.ndarray, threshold: float) -> np.ndarray:
    """Consecutive bars where VR < threshold."""
    n = len(vr)
    dur = np.zeros(n, dtype=np.int64)
    cnt = 0
    for i in range(n):
        if not np.isnan(vr[i]) and vr[i] < threshold:
            cnt += 1
        else:
            cnt = 0
        dur[i] = cnt
    return dur


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Forward MFE collector
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_forward_mfe(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    indices: np.ndarray, max_hold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For given bar indices, compute max forward excursion up & down (% of entry)."""
    n_idx = len(indices)
    n = len(close)
    mfe_up = np.empty(n_idx, dtype=np.float64)
    mfe_down = np.empty(n_idx, dtype=np.float64)
    for k in range(n_idx):
        i = indices[k]
        entry = close[i]
        mx_up = 0.0
        mx_dn = 0.0
        if entry > 0.0:
            end = min(i + 1 + max_hold, n)
            for j in range(i + 1, end):
                up = (high[j] - entry) / entry
                dn = (entry - low[j]) / entry
                if up > mx_up:
                    mx_up = up
                if dn > mx_dn:
                    mx_dn = dn
        mfe_up[k] = mx_up
        mfe_down[k] = mx_dn
    return mfe_up, mfe_down


@numba.jit(nopython=True, cache=True)
def compute_time_to_move(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    indices: np.ndarray, target_pct: float, max_hold: int,
) -> np.ndarray:
    """For each event, how many bars until price moves target_pct in EITHER direction?
    Returns -1 if never reached within max_hold."""
    n_idx = len(indices)
    n = len(close)
    ttm = np.full(n_idx, -1, dtype=np.int64)
    for k in range(n_idx):
        i = indices[k]
        entry = close[i]
        if entry <= 0.0:
            continue
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            up = (high[j] - entry) / entry
            dn = (entry - low[j]) / entry
            if up >= target_pct or dn >= target_pct:
                ttm[k] = j - i
                break
    return ttm


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════


def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_data() -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    from quantbox.core.settings import QuantBoxSettings

    settings = QuantBoxSettings()
    whitelist = settings.rb_symbol_whitelist.split(",")
    data: dict = {}
    for sym in whitelist:
        candidates = list(cache_dir.glob(f"{sym}_15m_*.npz"))
        if not candidates:
            continue
        best = max(candidates, key=_extract_days)
        npz = np.load(best)
        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        h = npz["high"].astype(np.float64)
        lo = npz["low"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {"close": c, "high": h, "low": lo, "bars": len(c)}
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  ASCII histogram
# ══════════════════════════════════════════════════════════════════════════════


def ascii_hist(
    values: np.ndarray, n_bins: int = 15, width: int = 50,
    title: str = "", pct_format: bool = True,
    lo_pct: float = 2.0, hi_pct: float = 98.0,
) -> None:
    """Print a compact ASCII histogram to terminal."""
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        print(f"  {title}: (empty)")
        return
    lo = np.percentile(vals, lo_pct)
    hi = np.percentile(vals, hi_pct)
    if lo >= hi:
        hi = lo + 0.01
    counts, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
    mx = max(counts) if max(counts) > 0 else 1
    med = np.median(vals)
    mean = np.mean(vals)

    if pct_format:
        print(f"\n  {title}  (N={len(vals)}, mean={mean:.2%}, med={med:.2%})")
    else:
        print(f"\n  {title}  (N={len(vals)}, mean={mean:.2f}, med={med:.2f})")

    for i in range(n_bins):
        bar_len = int(counts[i] / mx * width)
        pct = counts[i] / len(vals) * 100
        if pct_format:
            lb = f"{edges[i]:.1%}"
            ub = f"{edges[i + 1]:.1%}"
        else:
            lb = f"{edges[i]:.1f}"
            ub = f"{edges[i + 1]:.1f}"
        bar = "█" * bar_len
        print(f"  {lb:>7}–{ub:>6} |{bar:<{width}} {counts[i]:>5} ({pct:>4.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    now = datetime.now(timezone.utc)

    print("=" * 90)
    print("  R27: BAYESIAN RR DISTRIBUTION ANALYSIS OF VOLRATIO SQUEEZE")
    print(f"  Commit: {commit}")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Focus: Move distribution, RR asymmetry, Bayesian conditionals")
    print("  NO backtesting — pure distribution research")
    print("=" * 90)
    print()

    data = load_data()
    if not data:
        print("  ❌ No data!")
        return
    total_bars = sum(d["bars"] for d in data.values())
    print(f"  Loaded: {len(data)} symbols, {total_bars:,} total bars\n")
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    #  EVENT COLLECTION
    # ══════════════════════════════════════════════════════════════════════════

    # Per VR bin: lists of (dom, sub, rr, asymmetry, vr_exact, dur)
    bin_dom: list[list] = [[] for _ in range(N_BINS)]
    bin_sub: list[list] = [[] for _ in range(N_BINS)]
    bin_rr:  list[list] = [[] for _ in range(N_BINS)]
    bin_asym: list[list] = [[] for _ in range(N_BINS)]  # (dom-sub)/(dom+sub)

    # Squeeze episode events (VR < 0.5, non-overlapping)
    sq_dom_l: list[float] = []
    sq_sub_l: list[float] = []
    sq_rr_l:  list[float] = []
    sq_vr_l:  list[float] = []
    sq_dur_l: list[int] = []

    # Time-to-move for squeeze episodes (bars until move > 2%)
    sq_ttm2_l: list[int] = []
    sq_ttm4_l: list[int] = []

    # Duration-binned events
    dur_dom: dict[str, list] = {}
    dur_sub: dict[str, list] = {}

    print("  Collecting forward move stats...", end=" ", flush=True)

    for sym, d in data.items():
        c, h, lo = d["close"], d["high"], d["low"]
        n = len(c)

        # Compute indicators for full symbol
        vr = compute_vr_array(c, h, lo, SHORT_P, LONG_P)
        dur = compute_squeeze_dur(vr, VR_THRESHOLD)

        # ── Uniform sample (every SAMPLE_INTERVAL bars) ──
        idx_all = np.arange(WARMUP, n - MAX_HOLD, SAMPLE_INTERVAL, dtype=np.int64)
        valid_mask = np.array([not np.isnan(vr[i]) for i in idx_all])
        idx_valid = idx_all[valid_mask]

        if len(idx_valid) > 0:
            mfe_up, mfe_down = compute_forward_mfe(c, h, lo, idx_valid, MAX_HOLD)
            dom = np.maximum(mfe_up, mfe_down)
            sub = np.minimum(mfe_up, mfe_down)
            rr = dom / np.maximum(sub, 1e-6)
            rr = np.minimum(rr, 50.0)  # clip extreme RR
            asym = (dom - sub) / np.maximum(dom + sub, 1e-6)

            for k in range(len(idx_valid)):
                v = vr[idx_valid[k]]
                for b in range(N_BINS):
                    if VR_BIN_EDGES[b] <= v < VR_BIN_EDGES[b + 1]:
                        bin_dom[b].append(dom[k])
                        bin_sub[b].append(sub[k])
                        bin_rr[b].append(rr[k])
                        bin_asym[b].append(asym[k])
                        break

        # ── Squeeze episodes (VR < 0.5, non-overlapping, onset) ──
        episode_starts: list[int] = []
        in_squeeze = False
        last_recorded = -MIN_EPISODE_GAP * 2
        for i in range(WARMUP, n - MAX_HOLD):
            below = not np.isnan(vr[i]) and vr[i] < VR_THRESHOLD
            if below and not in_squeeze:
                if i - last_recorded >= MIN_EPISODE_GAP:
                    episode_starts.append(i)
                    last_recorded = i
                in_squeeze = True
            elif not below:
                in_squeeze = False

        if episode_starts:
            ep_idx = np.array(episode_starts, dtype=np.int64)
            ep_up, ep_down = compute_forward_mfe(c, h, lo, ep_idx, MAX_HOLD)
            ep_dom = np.maximum(ep_up, ep_down)
            ep_sub = np.minimum(ep_up, ep_down)
            ep_rr = ep_dom / np.maximum(ep_sub, 1e-6)
            ep_rr = np.minimum(ep_rr, 50.0)

            # Time-to-move
            ttm2 = compute_time_to_move(c, h, lo, ep_idx, 0.02, MAX_HOLD)
            ttm4 = compute_time_to_move(c, h, lo, ep_idx, 0.04, MAX_HOLD)

            for k in range(len(ep_idx)):
                sq_dom_l.append(ep_dom[k])
                sq_sub_l.append(ep_sub[k])
                sq_rr_l.append(ep_rr[k])
                sq_vr_l.append(vr[ep_idx[k]])
                d_val = int(dur[ep_idx[k]])
                sq_dur_l.append(d_val)
                sq_ttm2_l.append(int(ttm2[k]))
                sq_ttm4_l.append(int(ttm4[k]))

                # Duration bins
                if d_val <= 1:
                    dbin = "dur=1"
                elif d_val <= 5:
                    dbin = "dur 2-5"
                elif d_val <= 10:
                    dbin = "dur 6-10"
                elif d_val <= 20:
                    dbin = "dur 11-20"
                else:
                    dbin = "dur 21+"
                dur_dom.setdefault(dbin, []).append(ep_dom[k])
                dur_sub.setdefault(dbin, []).append(ep_sub[k])

    # Convert to numpy
    for b in range(N_BINS):
        bin_dom[b] = np.array(bin_dom[b])
        bin_sub[b] = np.array(bin_sub[b])
        bin_rr[b] = np.array(bin_rr[b])
        bin_asym[b] = np.array(bin_asym[b])
    sq_dom = np.array(sq_dom_l)
    sq_sub = np.array(sq_sub_l)
    sq_rr = np.array(sq_rr_l)
    sq_vr = np.array(sq_vr_l)
    sq_dur = np.array(sq_dur_l)
    sq_ttm2 = np.array(sq_ttm2_l)
    sq_ttm4 = np.array(sq_ttm4_l)

    total_uniform = sum(len(b) for b in bin_dom)
    print(f"done ({time.time()-t0:.1f}s)")
    print(f"  Uniform sample: {total_uniform:,} events across {N_BINS} VR bins")
    print(f"  Squeeze episodes (VR<{VR_THRESHOLD}, non-overlap): {len(sq_dom)}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1: MOVE STATISTICS BY VR BIN
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ SECTION 1: FORWARD MOVE STATISTICS BY VR BIN (48h window) ══")
    print()
    print(f"  {'VR Bin':<14} {'N':>7}  {'MeanDom':>8} {'MedDom':>8} {'P75Dom':>8}"
          f" {'P90Dom':>8}  {'MeanSub':>8} {'MedSub':>8}  {'MedRR':>6} {'MeanAs':>7}")
    print("  " + "─" * 96)

    for b in range(N_BINS):
        if len(bin_dom[b]) < 10:
            continue
        d_arr = bin_dom[b]
        s_arr = bin_sub[b]
        r_arr = bin_rr[b]
        a_arr = bin_asym[b]
        print(
            f"  {VR_BIN_LABELS[b]:<14} {len(d_arr):>7}  "
            f"{np.mean(d_arr):>8.2%} {np.median(d_arr):>8.2%} "
            f"{np.percentile(d_arr, 75):>8.2%} {np.percentile(d_arr, 90):>8.2%}  "
            f"{np.mean(s_arr):>8.2%} {np.median(s_arr):>8.2%}  "
            f"{np.median(r_arr):>6.1f} {np.mean(a_arr):>7.2f}"
        )
    print()
    print("  (MedRR = median(dominant/subdominant), MeanAs = mean((dom-sub)/(dom+sub)))")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2: DOMINANT MFE HISTOGRAMS
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ SECTION 2: DOMINANT MFE HISTOGRAMS ══")
    key_bins = [0, 1, 2, 4]  # VR<0.35, 0.35-0.45, 0.45-0.55, 0.70-1.0 (baseline)
    for b in key_bins:
        if len(bin_dom[b]) >= 20:
            ascii_hist(
                bin_dom[b], n_bins=15, title=f"Dominant MFE: {VR_BIN_LABELS[b]}",
                hi_pct=97,
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3: RR DISTRIBUTION (dominant / subdominant)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n  ══ SECTION 3: RR DISTRIBUTION (dominant / subdominant) ══")
    for b in key_bins:
        if len(bin_rr[b]) >= 20:
            ascii_hist(
                bin_rr[b], n_bins=12, width=45,
                title=f"RR: {VR_BIN_LABELS[b]}",
                pct_format=False, hi_pct=95,
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 4: BAYESIAN CONDITIONAL PROBABILITIES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n  ══ SECTION 4: BAYESIAN P(dominant > X% | VR bin) ══")
    all_dom = np.concatenate([b for b in bin_dom if len(b) > 0])

    # 4a: P(explosion > X%)
    hdr = f"  {'VR Bin':<14} {'N':>7}"
    for x in EXPLOSION_PCT:
        hdr += f"  P(>{x:.0%})"
    print(hdr)
    print("  " + "─" * (23 + 8 * len(EXPLOSION_PCT)))

    prior_probs: dict[float, float] = {}
    row = f"  {'ALL (prior)':<14} {len(all_dom):>7}"
    for x in EXPLOSION_PCT:
        p = float(np.mean(all_dom > x))
        prior_probs[x] = p
        row += f"  {p:>5.1%}"
    print(row)
    print("  " + "─" * (23 + 8 * len(EXPLOSION_PCT)))

    for b in range(N_BINS):
        if len(bin_dom[b]) < 20:
            continue
        d_arr = bin_dom[b]
        row = f"  {VR_BIN_LABELS[b]:<14} {len(d_arr):>7}"
        for x in EXPLOSION_PCT:
            p = float(np.mean(d_arr > x))
            row += f"  {p:>5.1%}"
        print(row)

    # 4b: Bayesian Lift
    print(f"\n  Bayesian Lift = P(dom>X | VR bin) / P(dom>X | all)")
    hdr = f"  {'VR Bin':<14} {'N':>7}"
    for x in EXPLOSION_PCT:
        hdr += f"  L>{x:.0%}  "
    print(hdr)
    print("  " + "─" * (23 + 8 * len(EXPLOSION_PCT)))

    for b in range(N_BINS):
        if len(bin_dom[b]) < 20:
            continue
        d_arr = bin_dom[b]
        row = f"  {VR_BIN_LABELS[b]:<14} {len(d_arr):>7}"
        for x in EXPLOSION_PCT:
            p = float(np.mean(d_arr > x))
            prior = prior_probs[x]
            lift = p / prior if prior > 0 else 0
            marker = "▲" if lift > 1.2 else ("▼" if lift < 0.8 else " ")
            row += f"  {marker}{lift:>5.2f}x"
        print(row)

    # 4c: Asymmetry — P(dom > k × sub)
    print(f"\n  Asymmetry: P(dom > k × sub | VR bin)")
    hdr = f"  {'VR Bin':<14} {'N':>7}"
    for k in ASYM_THRESHOLDS:
        hdr += f"  P(>{k:.1f}×)"
    print(hdr)
    print("  " + "─" * (23 + 10 * len(ASYM_THRESHOLDS)))

    row = f"  {'ALL (prior)':<14} {len(all_dom):>7}"
    all_sub = np.concatenate([b for b in bin_sub if len(b) > 0])
    all_rr = np.concatenate([b for b in bin_rr if len(b) > 0])
    asym_prior: dict[float, float] = {}
    for k in ASYM_THRESHOLDS:
        p = float(np.mean(all_rr > k))
        asym_prior[k] = p
        row += f"  {p:>7.1%}"
    print(row)
    print("  " + "─" * (23 + 10 * len(ASYM_THRESHOLDS)))

    for b in range(N_BINS):
        if len(bin_rr[b]) < 20:
            continue
        r_arr = bin_rr[b]
        row = f"  {VR_BIN_LABELS[b]:<14} {len(r_arr):>7}"
        for k in ASYM_THRESHOLDS:
            p = float(np.mean(r_arr > k))
            lift = p / asym_prior[k] if asym_prior[k] > 0 else 0
            marker = "▲" if lift > 1.2 else ("▼" if lift < 0.8 else " ")
            row += f"  {marker}{p:>5.1%} "
        print(row)
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 5: SQUEEZE EPISODES (non-overlapping)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ SECTION 5: SQUEEZE EPISODES (VR<0.5, non-overlapping, onset) ══")
    if len(sq_dom) >= 10:
        print(f"  Episodes: {len(sq_dom)}")
        print(f"  Dominant MFE:  mean={np.mean(sq_dom):.2%}  med={np.median(sq_dom):.2%}  "
              f"P75={np.percentile(sq_dom, 75):.2%}  P90={np.percentile(sq_dom, 90):.2%}")
        print(f"  Subdominant:   mean={np.mean(sq_sub):.2%}  med={np.median(sq_sub):.2%}  "
              f"P75={np.percentile(sq_sub, 75):.2%}")
        print(f"  RR (dom/sub):  mean={np.mean(sq_rr):.1f}  med={np.median(sq_rr):.1f}  "
              f"P75={np.percentile(sq_rr, 75):.1f}  P90={np.percentile(sq_rr, 90):.1f}")

        # Time to move
        ttm2_valid = sq_ttm2[sq_ttm2 > 0]
        ttm4_valid = sq_ttm4[sq_ttm4 > 0]
        print(f"\n  Time to ±2% move: {len(ttm2_valid)}/{len(sq_dom)} episodes reached "
              f"({len(ttm2_valid)/len(sq_dom):.0%})")
        if len(ttm2_valid) > 0:
            print(f"    med={np.median(ttm2_valid):.0f} bars ({np.median(ttm2_valid)*15/60:.1f}h)  "
                  f"P25={np.percentile(ttm2_valid, 25):.0f}  P75={np.percentile(ttm2_valid, 75):.0f}")
        print(f"  Time to ±4% move: {len(ttm4_valid)}/{len(sq_dom)} episodes reached "
              f"({len(ttm4_valid)/len(sq_dom):.0%})")
        if len(ttm4_valid) > 0:
            print(f"    med={np.median(ttm4_valid):.0f} bars ({np.median(ttm4_valid)*15/60:.1f}h)  "
                  f"P25={np.percentile(ttm4_valid, 25):.0f}  P75={np.percentile(ttm4_valid, 75):.0f}")

        ascii_hist(sq_dom, n_bins=15, title="Squeeze Episode: Dominant MFE", hi_pct=97)
        ascii_hist(sq_rr, n_bins=12, width=45,
                   title="Squeeze Episode: RR (dom/sub)", pct_format=False, hi_pct=95)

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 6: DURATION EFFECT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n  ══ SECTION 6: DURATION EFFECT ON EXPLOSION (VR<0.5 episodes) ══")
    dur_order = ["dur=1", "dur 2-5", "dur 6-10", "dur 11-20", "dur 21+"]
    print(f"  {'Duration':<12} {'N':>5}  {'MeanDom':>8} {'MedDom':>8} {'P90Dom':>8}"
          f"  {'MeanSub':>8} {'MedRR':>6}  P(>4%) P(>5%)")
    print("  " + "─" * 80)
    for dbin in dur_order:
        if dbin in dur_dom and len(dur_dom[dbin]) >= 5:
            d_arr = np.array(dur_dom[dbin])
            s_arr = np.array(dur_sub[dbin])
            rr_arr = d_arr / np.maximum(s_arr, 1e-6)
            rr_arr = np.minimum(rr_arr, 50.0)
            p4 = np.mean(d_arr > 0.04)
            p5 = np.mean(d_arr > 0.05)
            print(
                f"  {dbin:<12} {len(d_arr):>5}  "
                f"{np.mean(d_arr):>8.2%} {np.median(d_arr):>8.2%} "
                f"{np.percentile(d_arr, 90):>8.2%}  "
                f"{np.mean(s_arr):>8.2%} {np.median(rr_arr):>6.1f}  "
                f"{p4:>5.1%}  {p5:>5.1%}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 7: VR-AS-CONTINUOUS Bayesian (finer bins)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ══ SECTION 7: CONTINUOUS VR vs EXPLOSION PROBABILITY ══")
    print(f"  (Using uniform sample, binned by VR value)")
    fine_edges = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                  0.70, 0.80, 0.90, 1.00, 1.20, 99.0]
    fine_labels = [f"{fine_edges[i]:.2f}-{fine_edges[i+1]:.2f}"
                   for i in range(len(fine_edges)-1)]

    # Rebuild from uniform sample using all_dom/all_sub which are concatenated
    # We need per-event VR values → reconstruct from bins
    # Actually, let me re-collect with VR values stored
    # Better: use the squeeze episode data + uniform data we already have

    # For proper fine binning, re-collect with VR values
    all_dom_vr: list[tuple[float, float]] = []  # (vr, dom)
    for sym, d in data.items():
        c, h, lo = d["close"], d["high"], d["low"]
        n = len(c)
        vr = compute_vr_array(c, h, lo, SHORT_P, LONG_P)
        idx_all = np.arange(WARMUP, n - MAX_HOLD, SAMPLE_INTERVAL, dtype=np.int64)
        valid_mask = np.array([not np.isnan(vr[i]) for i in idx_all])
        idx_valid = idx_all[valid_mask]
        if len(idx_valid) == 0:
            continue
        mfe_up, mfe_down = compute_forward_mfe(c, h, lo, idx_valid, MAX_HOLD)
        dom = np.maximum(mfe_up, mfe_down)
        for k in range(len(idx_valid)):
            all_dom_vr.append((vr[idx_valid[k]], dom[k]))

    dom_vr_arr = np.array(all_dom_vr)  # shape (N, 2), col0=vr, col1=dom

    print(f"  {'VR Range':<12} {'N':>7}  P(>2%)  P(>3%)  P(>4%)  P(>5%)  "
          f"P(>8%) MedDom  Lift>4%")
    print("  " + "─" * 80)

    base_p4 = float(np.mean(dom_vr_arr[:, 1] > 0.04))
    for i in range(len(fine_edges) - 1):
        lo_e, hi_e = fine_edges[i], fine_edges[i + 1]
        mask = (dom_vr_arr[:, 0] >= lo_e) & (dom_vr_arr[:, 0] < hi_e)
        subset = dom_vr_arr[mask, 1]
        if len(subset) < 20:
            continue
        p2 = float(np.mean(subset > 0.02))
        p3 = float(np.mean(subset > 0.03))
        p4 = float(np.mean(subset > 0.04))
        p5 = float(np.mean(subset > 0.05))
        p8 = float(np.mean(subset > 0.08))
        med = float(np.median(subset))
        lift4 = p4 / base_p4 if base_p4 > 0 else 0
        marker = "▲" if lift4 > 1.2 else ("▼" if lift4 < 0.8 else " ")
        print(
            f"  {fine_labels[i]:<12} {len(subset):>7}  "
            f"{p2:>5.1%}  {p3:>5.1%}  {p4:>5.1%}  {p5:>5.1%}  "
            f"{p8:>5.1%} {med:>6.2%}  {marker}{lift4:>5.2f}x"
        )
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 8: MATPLOTLIB VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        out_dir = ROOT / "logs" / "research" / "r27_histograms"
        out_dir.mkdir(parents=True, exist_ok=True)
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

        # ── Fig 1: Dominant MFE by VR bin (overlaid histogram) ──
        fig, ax = plt.subplots(figsize=(12, 7))
        for b in range(min(5, N_BINS)):
            if len(bin_dom[b]) >= 20:
                ax.hist(
                    bin_dom[b] * 100, bins=50, range=(0, 15), alpha=0.45,
                    label=f"{VR_BIN_LABELS[b]} (N={len(bin_dom[b]):,})",
                    color=colors[b], density=True,
                )
        ax.set_xlabel("Dominant MFE (%)", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title("Distribution of Dominant Move by VR Bin (48h forward)", fontsize=15)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=4, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="TP=4%")
        fig.savefig(out_dir / "1_dominant_mfe_by_vr.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Fig 2: RR distribution by VR bin ──
        fig, ax = plt.subplots(figsize=(12, 7))
        for b in range(min(5, N_BINS)):
            if len(bin_rr[b]) >= 20:
                rr_c = np.clip(bin_rr[b], 0, 20)
                ax.hist(
                    rr_c, bins=40, range=(1, 15), alpha=0.45,
                    label=f"{VR_BIN_LABELS[b]} (N={len(bin_rr[b]):,})",
                    color=colors[b], density=True,
                )
        ax.set_xlabel("RR = dominant / subdominant", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title("Risk-Reward Distribution by VR Bin", fontsize=15)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "2_rr_by_vr.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Fig 3: Survival function P(dom > x) ──
        fig, ax = plt.subplots(figsize=(12, 7))
        for b in range(min(5, N_BINS)):
            if len(bin_dom[b]) >= 20:
                sorted_d = np.sort(bin_dom[b])
                surv = 1 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
                ax.plot(
                    sorted_d * 100, surv,
                    label=f"{VR_BIN_LABELS[b]} (N={len(bin_dom[b]):,})",
                    color=colors[b], linewidth=2,
                )
        ax.set_xlabel("Dominant MFE (%)", fontsize=13)
        ax.set_ylabel("P(dominant > x)", fontsize=13)
        ax.set_title("Survival Function: P(dominant MFE > x%) by VR Bin", fontsize=15)
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1)
        for tp_lvl in [2, 3, 4, 5]:
            ax.axvline(x=tp_lvl, color="gray", linestyle=":", alpha=0.5)
            ax.text(tp_lvl + 0.1, 0.95, f"TP={tp_lvl}%", fontsize=8, color="gray")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "3_survival_dominant_mfe.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Fig 4: Bayesian lift heatmap ──
        fig, ax = plt.subplots(figsize=(11, 6))
        lift_matrix = np.ones((N_BINS, len(EXPLOSION_PCT)))
        for b in range(N_BINS):
            if len(bin_dom[b]) >= 20:
                for j, x in enumerate(EXPLOSION_PCT):
                    p = float(np.mean(bin_dom[b] > x))
                    prior = prior_probs[x]
                    lift_matrix[b, j] = p / prior if prior > 0 else 1.0
        im = ax.imshow(lift_matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.5)
        ax.set_yticks(range(N_BINS))
        ax.set_yticklabels(VR_BIN_LABELS)
        ax.set_xticks(range(len(EXPLOSION_PCT)))
        ax.set_xticklabels([f">{x:.0%}" for x in EXPLOSION_PCT])
        ax.set_xlabel("Explosion Threshold", fontsize=12)
        ax.set_ylabel("VR Bin", fontsize=12)
        ax.set_title("Bayesian Lift: P(dom>X | VR bin) / P(dom>X | all)", fontsize=14)
        for bi in range(N_BINS):
            for j in range(len(EXPLOSION_PCT)):
                v = lift_matrix[bi, j]
                c = "white" if v > 2.0 or v < 0.7 else "black"
                ax.text(j, bi, f"{v:.2f}", ha="center", va="center", fontsize=9, color=c)
        fig.colorbar(im, label="Lift (green = higher probability)")
        fig.savefig(out_dir / "4_bayesian_lift_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Fig 5: Duration vs Explosion scatter ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        if len(sq_dur) > 0 and len(sq_dom) > 0:
            # Left: scatter
            sc = axes[0].scatter(
                sq_dur, sq_dom * 100, alpha=0.4, s=20, c=sq_vr,
                cmap="coolwarm_r", vmin=0.2, vmax=0.55,
            )
            axes[0].set_xlabel("Squeeze Duration (bars in VR<0.5)", fontsize=12)
            axes[0].set_ylabel("Dominant MFE (%)", fontsize=12)
            axes[0].set_title("Duration vs Explosion Size", fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=4, color="red", linestyle="--", alpha=0.5)
            fig.colorbar(sc, ax=axes[0], label="VR value at entry")

            # Right: box plot by duration bin
            dur_bins_data = []
            dur_bins_labels = []
            for dbin in dur_order:
                if dbin in dur_dom and len(dur_dom[dbin]) >= 5:
                    dur_bins_data.append(np.array(dur_dom[dbin]) * 100)
                    dur_bins_labels.append(dbin)
            if dur_bins_data:
                bp = axes[1].boxplot(dur_bins_data, labels=dur_bins_labels, patch_artist=True)
                for patch, c in zip(bp["boxes"], colors[: len(dur_bins_data)]):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.5)
                axes[1].set_xlabel("Squeeze Duration", fontsize=12)
                axes[1].set_ylabel("Dominant MFE (%)", fontsize=12)
                axes[1].set_title("Explosion Size by Duration", fontsize=14)
                axes[1].grid(True, alpha=0.3, axis="y")
                axes[1].axhline(y=4, color="red", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "5_duration_vs_explosion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Fig 6: Asymmetry comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        # Left: Asymmetry by VR bin
        asym_bins_data = []
        asym_bins_labels = []
        for b in range(N_BINS):
            if len(bin_asym[b]) >= 20:
                asym_bins_data.append(bin_asym[b])
                asym_bins_labels.append(VR_BIN_LABELS[b])
        if asym_bins_data:
            bp = axes[0].boxplot(asym_bins_data, labels=asym_bins_labels, patch_artist=True)
            for patch, c in zip(bp["boxes"], colors[: len(asym_bins_data)]):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            axes[0].set_ylabel("Asymmetry (dom-sub)/(dom+sub)", fontsize=11)
            axes[0].set_title("Move Asymmetry by VR Bin", fontsize=14)
            axes[0].grid(True, alpha=0.3, axis="y")
            axes[0].tick_params(axis="x", rotation=30)

        # Right: P(asymmetric) by VR bin — bar chart
        vr_labels_short = []
        p_asym_1p5 = []
        p_asym_3 = []
        for b in range(N_BINS):
            if len(bin_rr[b]) >= 20:
                vr_labels_short.append(VR_BIN_LABELS[b])
                p_asym_1p5.append(float(np.mean(bin_rr[b] > 1.5)))
                p_asym_3.append(float(np.mean(bin_rr[b] > 3.0)))
        if vr_labels_short:
            x = np.arange(len(vr_labels_short))
            w = 0.35
            axes[1].bar(x - w / 2, p_asym_1p5, w, label="P(RR > 1.5)", color="#2ecc71", alpha=0.7)
            axes[1].bar(x + w / 2, p_asym_3, w, label="P(RR > 3.0)", color="#e74c3c", alpha=0.7)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(vr_labels_short, rotation=30, ha="right")
            axes[1].set_ylabel("Probability", fontsize=12)
            axes[1].set_title("P(asymmetric move) by VR Bin", fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "6_asymmetry_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"\n  📊 Saved 6 PNGs to {out_dir}/")

    except ImportError:
        print("\n  ⚠️  matplotlib not installed, skipping PNG output")
    except Exception as e:
        print(f"\n  ⚠️  matplotlib error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"  RUNTIME: {elapsed:.1f}s")
    print(f"{'=' * 90}")

    # Save JSON
    out_path = ROOT / "logs" / "research" / "vol_straddle_r27_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data: dict = {
        "commit": commit,
        "date": now.isoformat(),
        "runtime_s": round(elapsed, 1),
        "n_symbols": len(data),
        "total_bars": total_bars,
        "uniform_events": total_uniform,
        "squeeze_episodes": len(sq_dom),
    }
    for b in range(N_BINS):
        if len(bin_dom[b]) >= 10:
            save_data[f"bin_{VR_BIN_LABELS[b]}_n"] = len(bin_dom[b])
            save_data[f"bin_{VR_BIN_LABELS[b]}_mean_dom"] = round(float(np.mean(bin_dom[b])), 5)
            save_data[f"bin_{VR_BIN_LABELS[b]}_med_rr"] = round(float(np.median(bin_rr[b])), 2)
            save_data[f"bin_{VR_BIN_LABELS[b]}_mean_asym"] = round(float(np.mean(bin_asym[b])), 3)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  💾 {out_path}")


if __name__ == "__main__":
    main()
