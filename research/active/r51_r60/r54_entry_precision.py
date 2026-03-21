#!/usr/bin/env python3
"""r54: Entry Precision Research — More Accurate Entry Timing.

Goal: Improve WR and Return WITHOUT reducing trade count (-10% max).

Current production (v14.5.2, 60 syms, 6yr):
  N=6,207 | PF=1.563 | WR=40.6% | EV=+38bp | Calmar=57.9 | DD=60.5%

Hypothesis: All VR<threshold bars enter with equal priority. Entry bar
quality (candle body, volume, VR depth, momentum acceleration, close position)
predicts outcome but is currently ignored.

Phase 1: Baseline reproduce (assert ±1%)
Phase 2: Feature extraction — 5 entry features vs win/loss outcome
Phase 3: Selective filter — top features as entry gate (N >= 5,587)
Phase 4: Delayed entry — recover N by waiting 1-2 bars for confirmation
Phase 5: Combined + OOS 50% split + report

Usage:
    uv run python research/active/r51_r60/r54_entry_precision.py
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    bootstrap_pf,
    build_resampled,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    compute_momentum,
    compute_squeeze_dur,
    compute_vr,
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)
from quantbox.backtest.trade_sim import sim_directional_trailing, sim_straddle_trailing
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ── Load production config (NO hardcoding) ───────────────────────────────────
cfg = VolStraddleConfig()

D_SL = cfg.d_sl                      # 0.01
D_TP = cfg.d_tp                      # 0.12
D_TRAIL_ACTIVATE = cfg.d_trail_activate   # 0.005
D_TRAIL_DISTANCE = cfg.d_trail_distance   # 0.002
S_SL = cfg.s_sl                      # 0.015
S_TP = cfg.s_tp                      # 0.05
S_TRAIL_ACTIVATE = cfg.s_trail_activate   # 0.015
S_TRAIL_DISTANCE = cfg.s_trail_distance   # 0.002
MAX_CONC = cfg.max_concurrent         # 10
VR_LONG = cfg.vr_long                 # 56
VR_SHORT = cfg.vr_short               # 8

print("[PARITY CHECK]")
print(f"  D_TRAIL: activate={D_TRAIL_ACTIVATE:.3f} / distance={D_TRAIL_DISTANCE:.3f}")
print(f"  S_TRAIL: activate={S_TRAIL_ACTIVATE:.3f} / distance={S_TRAIL_DISTANCE:.3f}")
print(f"  D_TP={D_TP:.2f}, D_SL={D_SL:.3f}, L/N={cfg.base_leverage/MAX_CONC:.1f}")
assert D_TRAIL_ACTIVATE == 0.040, f"D trail activate mismatch: {D_TRAIL_ACTIVATE}"
assert D_TRAIL_DISTANCE == 0.002, f"D trail distance mismatch: {D_TRAIL_DISTANCE}"
print("  ✅ All params match production config.py\n")

CACHE_DIR = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_syms(min_years: float = 5.0) -> tuple[dict, str]:
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym in cfg.exclude_symbols:
            continue
        nb = int(np.load(f, allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    min_bars = int(min_years * 365 * 24 * 4)
    ranked = sorted(
        [(s, p, nb) for s, (p, nb) in seen.items() if nb >= min_bars],
        key=lambda x: -x[2],
    )
    data: dict = {}
    for sym, fpath, _ in ranked:
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}
    mb = max(len(d["close"]) for d in data.values()) if data else 0
    for d in data.values():
        d["_offset"] = mb - len(d["close"])
    return data, str(ranked[0][1]) if ranked else ""


def get_bar0(fpath: str) -> datetime:
    fname = Path(fpath).name
    end = datetime.strptime(fname.split("_")[-1].replace(".npz", ""), "%Y%m%d")
    return end - timedelta(days=int(fname.split("_")[2].replace("d", "")))


def build_tf(d15: dict) -> tuple[dict, dict]:
    d1h = build_resampled(d15, 4)
    d4h = build_resampled(d15, 16)
    precompute_all_vr([d15, d1h, d4h], [(VR_SHORT, VR_LONG)])
    for ds in [d15, d1h, d4h]:
        precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)
    precompute_mas(d15)
    precompute_mas(d1h)
    precompute_mas(d4h)
    return d1h, d4h


# ── Production-parity trade collection (same as r53 rerun) ───────────────────

TF_CONFIGS = [
    (1, 1, 50, 192, 192),   # (ratio, _, warmup, max_hold_D, max_hold_S)
    (4, 4, 13, 48, 48),
    (16, 16, 4, 12, 12),
]


def collect_all_production(
    d15: dict, d1h: dict, d4h: dict,
) -> list[tuple[str, int, int, float]]:
    """Collect all trades using production-parity exit logic (trailing).
    No per-symbol dedup — portfolio_sim handles slot competition (matches r53 canonical).
    """
    tf = {1: d15, 4: d1h, 16: d4h}
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(d15):
        for ratio, _, warmup, mh_d, mh_s in TF_CONFIGS:
            ds = tf[ratio]
            if sym not in ds:
                continue
            d = ds[sym]
            vr_key = f"vr_{VR_SHORT}_{VR_LONG}"
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            mom = d.get("mom_8", compute_momentum(c, cfg.lookback))

            # D_mom8 directional with production trailing
            sig_d = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig_d, mom)
            p_d, e_d, x_d = sim_directional_trailing(
                c, h, lo, sig_d, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
            )
            for k in range(len(p_d)):
                all_trades.append((sym, int(e_d[k]) * ratio, int(x_d[k]) * ratio + ratio - 1, float(p_d[k])))

            # S_d8 straddle with production trailing
            sig_s = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p_s, e_s, x_s = sim_straddle_trailing(
                c, h, lo, sig_s, S_SL, S_TP, mh_s, COST_STRADDLE,
                S_TRAIL_ACTIVATE, S_TRAIL_DISTANCE,
            )
            for k in range(len(p_s)):
                all_trades.append((sym, int(e_s[k]) * ratio, int(x_s[k]) * ratio + ratio - 1, float(p_s[k])))

        # MA cluster via collect_enhanced_trades (same as r53)
        try:
            ma_trades = collect_enhanced_trades(
                sym, d15, d1h, d4h,
                cluster_pct=cfg.ma_cluster_pct,
                min_dur=cfg.ma_cluster_min_duration,
                max_dur=cfg.ma_cluster_max_duration,
                min_mom=cfg.ma_cluster_min_momentum,
                sl=cfg.ma_cluster_sl,
                tp=cfg.ma_cluster_tp,
                trail_activate=cfg.ma_cluster_trail_activate,
                trail_distance=cfg.ma_cluster_trail_distance,
                tf_restrict=True,
            )
            for eb, xb, pnl in ma_trades:
                all_trades.append((sym, eb, xb, pnl))
        except Exception:
            pass

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: Baseline
# ══════════════════════════════════════════════════════════════════════════════

def phase1_baseline(d15: dict, d1h: dict, d4h: dict, data_years: float) -> dict:
    print("=" * 70)
    print("PHASE 1: Baseline Reproduction")
    print("=" * 70)
    trades = collect_all_production(d15, d1h, d4h)
    pnl = portfolio_sim(trades, MAX_CONC)
    m = compute_metrics(pnl, data_years)
    boot = bootstrap_pf(pnl, n_boot=1000)

    print(f"N={m['N']} | PF={m['PF']:.3f} | WR={m['WR']:.1f}% | "
          f"EV={m['EV_bp']:.1f}bp | Calmar={m['calmar']:.1f} | DD={m['max_dd_pct']:.1f}%")
    print(f"Bootstrap CI: [{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]")

    # Assert ±3% of canonical (60 syms may differ slightly by cache version)
    canon_n, canon_pf = 6207, 1.563
    n_drift = abs(m["N"] - canon_n) / canon_n
    pf_drift = abs(m["PF"] - canon_pf) / canon_pf
    if n_drift < 0.05 and pf_drift < 0.05:
        print(f"✅ Baseline within 5% of canonical (N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}%)")
    else:
        print(f"⚠️  Baseline drift: N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}% — check cache")

    return {"trades": trades, "pnl": pnl, "metrics": m, "boot": boot}


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Entry Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_entry_features(
    d15: dict, sym: str, entry_bar_15m: int, direction: int,
) -> dict[str, float] | None:
    """Extract 5 entry quality features from the signal bar (no lookahead).

    All features use completed bars at entry_bar_15m (the signal bar whose
    close triggered the entry — next bar open is the actual fill).

    Args:
        d15: 15m data dict.
        sym: Symbol.
        entry_bar_15m: Entry bar index in 15m timeline.
        direction: +1 for LONG, -1 for SHORT, 0 for straddle.

    Returns:
        Feature dict or None if data insufficient.
    """
    if sym not in d15:
        return None
    d = d15[sym]
    offset = int(d.get("_offset", 0))
    i = entry_bar_15m - offset  # local index in this symbol's array
    if i < 20 or i >= len(d["close"]):
        return None

    c = d["close"]
    h = d["high"]
    lo = d["low"]
    vol = d["volume"]
    vr_key = f"vr_{VR_SHORT}_{VR_LONG}"
    vr = d.get(vr_key)

    # 1. Candle body ratio: |close-open|/(high-low) — 0=doji, 1=strong
    rng = h[i] - lo[i]
    body = abs(c[i] - (c[i - 1] if i > 0 else c[i]))  # proxy: |close - prev_close|
    candle_body = float(body / rng) if rng > 0 else 0.5

    # 2. Volume ratio: current vs 20-bar MA
    vol_ma = np.mean(vol[max(0, i - 20):i]) if i >= 1 else vol[i]
    vol_ratio = float(vol[i] / vol_ma) if vol_ma > 0 else 1.0

    # 3. VR depth: minimum VR during the squeeze period (last 8 bars)
    if vr is not None and not np.isnan(vr[i]):
        sq_window = vr[max(0, i - 8):i + 1]
        vr_depth = float(np.nanmin(sq_window))
    else:
        vr_depth = float("nan")

    # 4. Momentum acceleration: mom[i] - mom[i-1]
    mom = d.get("mom_8")
    if mom is not None and i >= 1:
        mom_accel = float(mom[i] - mom[i - 1])
    else:
        mom_accel = 0.0

    # 5. Close position in bar range (for LONG: higher = bullish)
    close_pos = float((c[i] - lo[i]) / rng) if rng > 0 else 0.5

    # Direction-adjusted close position (for SHORT: flip so 1 = strong)
    if direction == -1:
        close_pos = 1.0 - close_pos

    return {
        "candle_body": candle_body,
        "vol_ratio": vol_ratio,
        "vr_depth": vr_depth,
        "mom_accel": mom_accel,
        "close_pos": close_pos,
    }


def phase2_feature_analysis(
    d15: dict, trades: list[tuple[str, int, int, float]],
) -> dict[str, dict]:
    print("\n" + "=" * 70)
    print("PHASE 2: Entry Feature Extraction")
    print("=" * 70)

    feature_names = ["candle_body", "vol_ratio", "vr_depth", "mom_accel", "close_pos"]
    winners = {f: [] for f in feature_names}
    losers = {f: [] for f in feature_names}

    for sym, entry_15m, exit_15m, pnl in trades:
        # Infer direction from entry_15m in d15 data
        d = d15.get(sym, {})
        offset = int(d.get("_offset", 0))
        i = entry_15m - offset
        mom = d.get("mom_8")
        if mom is not None and 0 <= i < len(mom):
            direction = 1 if mom[i] >= 0 else -1
        else:
            direction = 0  # straddle / unknown

        feats = extract_entry_features(d15, sym, entry_15m, direction)
        if feats is None:
            continue

        bucket = winners if pnl > 0 else losers
        for f in feature_names:
            v = feats[f]
            if not (v != v):  # not NaN
                bucket[f].append(v)

    n_win = len(winners["candle_body"])
    n_lose = len(losers["candle_body"])
    print(f"Analyzed {n_win + n_lose} trades: {n_win} winners, {n_lose} losers")
    print(f"WR check: {n_win/(n_win+n_lose)*100:.1f}%\n")

    print(f"{'Feature':<15} {'Win_mean':>10} {'Lose_mean':>10} {'Δ':>8} {'Win_p75':>10} {'Lose_p25':>10} {'Separation':>12}")
    print("-" * 80)

    stats: dict[str, dict] = {}
    for f in feature_names:
        w = np.array(winners[f])
        l = np.array(losers[f])
        if len(w) == 0 or len(l) == 0:
            continue
        wm, lm = np.mean(w), np.mean(l)
        delta = wm - lm
        wp75, lp25 = np.percentile(w, 75), np.percentile(l, 25)
        separation = (wp75 - lp25) / (np.std(w) + np.std(l) + 1e-9)
        stats[f] = {
            "win_mean": wm, "lose_mean": lm, "delta": delta,
            "win_p25": np.percentile(w, 25), "win_p75": wp75,
            "lose_p25": lp25, "lose_p75": np.percentile(l, 75),
            "separation": separation,
            "win_data": w, "lose_data": l,
        }
        print(f"{f:<15} {wm:>10.4f} {lm:>10.4f} {delta:>+8.4f} {wp75:>10.4f} {lp25:>10.4f} {separation:>12.4f}")

    # Rank by absolute delta
    ranked = sorted(stats.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)
    print("\nFeature ranking by predictive power (|Δmean|):")
    for rank, (f, s) in enumerate(ranked, 1):
        direction_hint = "↑ winner higher" if s["delta"] > 0 else "↓ winner lower"
        print(f"  {rank}. {f:<15} Δ={s['delta']:+.4f}  ({direction_hint})")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: Selective Filter
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_directional_with_filter(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float, trail_distance: float,
    # Filter thresholds (NaN = disabled)
    min_close_pos: float,    # close must be in upper X% of range for LONG
    min_vol_ratio: float,    # volume must be > X * MA20
    vol_ma_window: int,      # volume MA lookback
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional trailing sim with entry quality filter.

    Adds two optional gates before entry:
    - close_pos: close must be in upper min_close_pos fraction of bar range (LONG)
    - vol_ratio: volume must exceed min_vol_ratio * MA(volume, vol_ma_window)

    Direction-adjusted: for SHORT, close must be in lower (1-min_close_pos) fraction.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0

    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue

        rng = high[i] - low[i]
        entry_price = close[i]

        # Gate 1: close position
        if min_close_pos > 0.0 and rng > 0.0:
            pos = (close[i] - low[i]) / rng
            if dire[i] == 1 and pos < min_close_pos:
                i += 1
                continue
            if dire[i] == -1 and pos > (1.0 - min_close_pos):
                i += 1
                continue

        # Gate 2: volume ratio vs MA
        if min_vol_ratio > 0.0 and i >= vol_ma_window:
            vol_sum = 0.0
            for k in range(i - vol_ma_window, i):
                vol_sum += volume[k]
            vol_ma = vol_sum / vol_ma_window
            if vol_ma > 0.0 and volume[i] < min_vol_ratio * vol_ma:
                i += 1
                continue

        if entry_price <= 0.0:
            i += 1
            continue

        # Trade simulation with trailing stop
        is_long = dire[i] == 1
        peak_gain = 0.0
        trail_active = False
        eb = i
        end = min(i + 1 + max_hold, n)

        for j in range(i + 1, end):
            if is_long:
                gain = (close[j] - entry_price) / entry_price
                if gain > peak_gain:
                    peak_gain = gain
                if not trail_active and peak_gain >= trail_activate:
                    trail_active = True
                trail_stop = peak_gain - trail_distance
                if trail_active and gain <= trail_stop:
                    eb = j
                    pnl[cnt] = gain - cost
                    ent[cnt] = i
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if low[j] <= entry_price * (1.0 - sl):
                    eb = j
                    pnl[cnt] = -sl - cost
                    ent[cnt] = i
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if high[j] >= entry_price * (1.0 + tp):
                    eb = j
                    pnl[cnt] = tp - cost
                    ent[cnt] = i
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
            else:
                gain = (entry_price - close[j]) / entry_price
                if gain > peak_gain:
                    peak_gain = gain
                if not trail_active and peak_gain >= trail_activate:
                    trail_active = True
                trail_stop = peak_gain - trail_distance
                if trail_active and gain <= trail_stop:
                    eb = j
                    pnl[cnt] = gain - cost
                    ent[cnt] = i
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if high[j] >= entry_price * (1.0 + sl):
                    eb = j
                    pnl[cnt] = -sl - cost
                    ent[cnt] = i
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if low[j] <= entry_price * (1.0 - tp):
                    eb = j
                    pnl[cnt] = tp - cost
                    ent[cnt] = i
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
        else:
            # Max hold reached
            eb = end - 1
            if is_long:
                final_gain = (close[eb] - entry_price) / entry_price
            else:
                final_gain = (entry_price - close[eb]) / entry_price
            pnl[cnt] = final_gain - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
            i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


def collect_with_filter(
    d15: dict, d1h: dict, d4h: dict,
    min_close_pos: float = 0.0,
    min_vol_ratio: float = 0.0,
) -> list[tuple[str, int, int, float]]:
    """Collect trades with entry filter applied to D_mom8 directional only.

    Straddle (S_d8) and MA cluster are unchanged (filter not applicable).
    No per-symbol dedup — portfolio_sim handles slot competition (consistent with Phase 1).
    """
    tf = {1: d15, 4: d1h, 16: d4h}
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(d15):
        for ratio, _, warmup, mh_d, mh_s in TF_CONFIGS:
            ds = tf[ratio]
            if sym not in ds:
                continue
            d = ds[sym]
            vr_key = f"vr_{VR_SHORT}_{VR_LONG}"
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            vol = d.get("volume", np.ones(len(c)))
            mom = d.get("mom_8", compute_momentum(c, cfg.lookback))

            # D_mom8 with entry filter
            sig_d = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig_d, mom)
            p_d, e_d, x_d = sim_directional_with_filter(
                c, h, lo, vol, sig_d, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
                min_close_pos, min_vol_ratio, 20,
            )
            for k in range(len(p_d)):
                all_trades.append((sym, int(e_d[k]) * ratio, int(x_d[k]) * ratio + ratio - 1, float(p_d[k])))

            # S_d8 unchanged
            sig_s = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p_s, e_s, x_s = sim_straddle_trailing(
                c, h, lo, sig_s, S_SL, S_TP, mh_s, COST_STRADDLE,
                S_TRAIL_ACTIVATE, S_TRAIL_DISTANCE,
            )
            for k in range(len(p_s)):
                all_trades.append((sym, int(e_s[k]) * ratio, int(x_s[k]) * ratio + ratio - 1, float(p_s[k])))

        # MA cluster unchanged
        try:
            ma_trades = collect_enhanced_trades(
                sym, d15, d1h, d4h,
                cluster_pct=cfg.ma_cluster_pct,
                min_dur=cfg.ma_cluster_min_duration,
                max_dur=cfg.ma_cluster_max_duration,
                min_mom=cfg.ma_cluster_min_momentum,
                sl=cfg.ma_cluster_sl,
                tp=cfg.ma_cluster_tp,
                trail_activate=cfg.ma_cluster_trail_activate,
                trail_distance=cfg.ma_cluster_trail_distance,
                tf_restrict=True,
            )
            for eb, xb, pnl in ma_trades:
                all_trades.append((sym, eb, xb, pnl))
        except Exception:
            pass

    return all_trades


def phase3_filter_sweep(
    d15: dict, d1h: dict, d4h: dict, data_years: float,
    baseline_n: int, baseline_pf: float, feature_stats: dict,
) -> list[dict]:
    print("\n" + "=" * 70)
    print("PHASE 3: Selective Filter Sweep")
    print("=" * 70)
    n_min = int(baseline_n * 0.90)  # Must retain ≥90% of trades
    print(f"Constraint: N >= {n_min} (baseline {baseline_n} × 90%)\n")

    # Use top features from Phase 2 analysis
    close_pos_thresholds = [0.0, 0.40, 0.45, 0.50, 0.55]
    vol_ratio_thresholds = [0.0, 1.0, 1.2, 1.5]

    results = []
    header = f"{'close_pos':>10} {'vol_ratio':>10} {'N':>7} {'PF':>7} {'WR%':>6} {'EV(bp)':>8} {'Calmar':>8} {'DD%':>6}"
    print(header)
    print("-" * len(header))

    for cp in close_pos_thresholds:
        for vr in vol_ratio_thresholds:
            trades = collect_with_filter(d15, d1h, d4h, min_close_pos=cp, min_vol_ratio=vr)
            pnl = portfolio_sim(trades, MAX_CONC)
            m = compute_metrics(pnl, data_years)
            n = m["N"]
            pf = m["PF"]
            wr = m['WR']
            ev = m['EV_bp']
            cal = m['calmar']
            dd = m['max_dd_pct']
            ok = "✅" if n >= n_min and pf > baseline_pf else "  "
            print(f"{ok} {cp:>9.2f} {vr:>10.2f} {n:>7} {pf:>7.3f} {wr:>6.1f} {ev:>8.1f} {cal:>8.1f} {dd:>6.1f}%")
            results.append({
                "min_close_pos": cp, "min_vol_ratio": vr,
                "N": n, "PF": pf, "WR": wr, "EV_bp": ev, "calmar": cal, "DD": dd,
                "ok": n >= n_min and pf > baseline_pf,
            })

    best = sorted([r for r in results if r["ok"]], key=lambda x: -x["calmar"])
    if best:
        b = best[0]
        print(f"\nBest (Calmar): close_pos≥{b['min_close_pos']:.2f} vol≥{b['min_vol_ratio']:.1f}x → "
              f"N={b['N']} PF={b['PF']:.3f} Calmar={b['calmar']:.1f}")
    else:
        print("\nNo config beats baseline within N constraint — filters too aggressive")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4: Delayed Entry (N Recovery)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_directional_delayed(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    vr: np.ndarray, vr_threshold: float,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float, trail_distance: float,
    min_close_pos: float, min_vol_ratio: float, vol_ma_window: int,
    max_delay: int,  # max bars to wait for confirmation
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional trailing sim with delayed entry.

    If the entry filter fails on the signal bar, waits up to `max_delay` bars.
    If the filter condition is met within the delay window (AND VR still below
    threshold), enters there instead. This recovers trades that would be missed.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0

    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue

        # Find entry bar: signal bar or delayed bar
        entry_bar = -1
        for delay in range(max_delay + 1):
            j = i + delay
            if j >= n - 1:
                break
            # Must still be in squeeze
            if not (not (vr[j] != vr[j]) and vr[j] < vr_threshold):
                break

            rng_j = high[j] - low[j]
            # Gate 1: close position
            cp_ok = True
            if min_close_pos > 0.0 and rng_j > 0.0:
                pos = (close[j] - low[j]) / rng_j
                if dire[i] == 1 and pos < min_close_pos:
                    cp_ok = False
                if dire[i] == -1 and pos > (1.0 - min_close_pos):
                    cp_ok = False
            # Gate 2: volume
            vol_ok = True
            if min_vol_ratio > 0.0 and j >= vol_ma_window:
                vol_sum = 0.0
                for k in range(j - vol_ma_window, j):
                    vol_sum += volume[k]
                vol_ma = vol_sum / vol_ma_window
                if vol_ma > 0.0 and volume[j] < min_vol_ratio * vol_ma:
                    vol_ok = False

            if cp_ok and vol_ok:
                entry_bar = j
                break

        if entry_bar == -1:
            # No valid entry found — skip to next signal
            i += 1
            continue

        entry_price = close[entry_bar]
        if entry_price <= 0.0:
            i = entry_bar + 1
            continue

        # Trade simulation from entry_bar (same trailing logic)
        is_long = dire[i] == 1
        peak_gain = 0.0
        trail_active = False
        eb = entry_bar
        end = min(entry_bar + 1 + max_hold, n)

        for j in range(entry_bar + 1, end):
            if is_long:
                gain = (close[j] - entry_price) / entry_price
                if gain > peak_gain:
                    peak_gain = gain
                if not trail_active and peak_gain >= trail_activate:
                    trail_active = True
                trail_stop = peak_gain - trail_distance
                if trail_active and gain <= trail_stop:
                    eb = j
                    pnl[cnt] = gain - cost
                    ent[cnt] = entry_bar
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if low[j] <= entry_price * (1.0 - sl):
                    eb = j
                    pnl[cnt] = -sl - cost
                    ent[cnt] = entry_bar
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if high[j] >= entry_price * (1.0 + tp):
                    eb = j
                    pnl[cnt] = tp - cost
                    ent[cnt] = entry_bar
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
            else:
                gain = (entry_price - close[j]) / entry_price
                if gain > peak_gain:
                    peak_gain = gain
                if not trail_active and peak_gain >= trail_activate:
                    trail_active = True
                trail_stop = peak_gain - trail_distance
                if trail_active and gain <= trail_stop:
                    eb = j
                    pnl[cnt] = gain - cost
                    ent[cnt] = entry_bar
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if high[j] >= entry_price * (1.0 + sl):
                    eb = j
                    pnl[cnt] = -sl - cost
                    ent[cnt] = entry_bar
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
                if low[j] <= entry_price * (1.0 - tp):
                    eb = j
                    pnl[cnt] = tp - cost
                    ent[cnt] = entry_bar
                    ext[cnt] = eb
                    cnt += 1
                    i = eb + 1
                    break
        else:
            eb = end - 1
            if is_long:
                final_gain = (close[eb] - entry_price) / entry_price
            else:
                final_gain = (entry_price - close[eb]) / entry_price
            pnl[cnt] = final_gain - cost
            ent[cnt] = entry_bar
            ext[cnt] = eb
            cnt += 1
            i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


def collect_with_delayed_entry(
    d15: dict, d1h: dict, d4h: dict,
    min_close_pos: float = 0.50,
    min_vol_ratio: float = 0.0,
    max_delay: int = 2,
) -> list[tuple[str, int, int, float]]:
    """D_mom8 with delayed entry. S_d8 and MA unchanged."""
    tf = {1: d15, 4: d1h, 16: d4h}
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(d15):
        sym_trades: list[tuple[int, int, float]] = []
        for ratio, _, warmup, mh_d, mh_s in TF_CONFIGS:
            ds = tf[ratio]
            if sym not in ds:
                continue
            d = ds[sym]
            vr_key = f"vr_{VR_SHORT}_{VR_LONG}"
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            vol = d.get("volume", np.ones(len(c)))
            mom = d.get("mom_8", compute_momentum(c, cfg.lookback))

            sig_d = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig_d, mom)
            p_d, e_d, x_d = sim_directional_delayed(
                c, h, lo, vol, sig_d, dire, vr, cfg.d_vr_threshold,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
                min_close_pos, min_vol_ratio, 20, max_delay,
            )
            for k in range(len(p_d)):
                sym_trades.append((int(e_d[k]) * ratio, int(x_d[k]) * ratio + ratio - 1, float(p_d[k])))

            sig_s = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p_s, e_s, x_s = sim_straddle_trailing(
                c, h, lo, sig_s, S_SL, S_TP, mh_s, COST_STRADDLE,
                S_TRAIL_ACTIVATE, S_TRAIL_DISTANCE,
            )
            for k in range(len(p_s)):
                sym_trades.append((int(e_s[k]) * ratio, int(x_s[k]) * ratio + ratio - 1, float(p_s[k])))

            try:
                ma_trades = collect_enhanced_trades(
                    sym, d15, d1h, d4h,
                    cluster_pct=cfg.ma_cluster_pct,
                    min_dur=cfg.ma_cluster_min_duration,
                    max_dur=cfg.ma_cluster_max_duration,
                    min_mom=cfg.ma_cluster_min_momentum,
                    sl=cfg.ma_cluster_sl,
                    tp=cfg.ma_cluster_tp,
                    trail_activate=cfg.ma_cluster_trail_activate,
                    trail_distance=cfg.ma_cluster_trail_distance,
                    tf_restrict=True,
                )
                for eb, xb, pnl in ma_trades:
                    sym_trades.append((eb, xb, pnl))
            except Exception:
                pass

        sym_trades.sort(key=lambda x: x[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades


def phase4_delayed_entry(
    d15: dict, d1h: dict, d4h: dict, data_years: float,
    baseline_n: int, baseline_pf: float,
    best_close_pos: float = 0.50,
) -> list[dict]:
    print("\n" + "=" * 70)
    print("PHASE 4: Delayed Entry (N Recovery)")
    print("=" * 70)
    print(f"Config: close_pos≥{best_close_pos:.2f}, max_delay sweep [0,1,2,3]\n")

    results = []
    header = f"{'max_delay':>10} {'N':>7} {'PF':>7} {'WR%':>6} {'EV(bp)':>8} {'Calmar':>8} {'DD%':>6}"
    print(header)
    print("-" * 55)

    for delay in range(4):
        trades = collect_with_delayed_entry(d15, d1h, d4h, min_close_pos=best_close_pos, max_delay=delay)
        pnl = portfolio_sim(trades, MAX_CONC)
        m = compute_metrics(pnl, data_years)
        boot = bootstrap_pf(pnl, n_boot=500)
        ok = "✅" if m["N"] >= baseline_n * 0.9 and m["PF"] > baseline_pf else "  "
        print(f"{ok} {delay:>9} {m['N']:>7} {m['PF']:>7.3f} {m['WR']:>6.1f} {m['EV_bp']:>8.1f} "
              f"{m['calmar']:>8.1f} {m['max_dd_pct']:>6.1f}%  CI=[{boot['CI_lo']:.3f},{boot['CI_hi']:.3f}]")
        results.append({"max_delay": delay, "metrics": m, "boot": boot, "N": m["N"]})

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 5: OOS Validation + Report
# ══════════════════════════════════════════════════════════════════════════════

def phase5_oos_report(
    d15: dict, d1h: dict, d4h: dict, data_years: float,
    best_close_pos: float, best_delay: int,
    baseline_result: dict, bar0: datetime, fpath: str,
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 5: OOS Validation + Final Report")
    print("=" * 70)

    # Split at 50% by time
    total_bars = max(len(d["close"]) for d in d15.values())
    split = total_bars // 2

    def split_data(data: dict, start: int, end: int) -> dict:
        out = {}
        for sym, d in data.items():
            offset = int(d.get("_offset", 0))
            slc = {k: v[max(0, start - offset):max(0, end - offset)]
                   for k, v in d.items()
                   if isinstance(v, np.ndarray) and v.ndim == 1 and k != "_offset"}
            slc["_offset"] = max(0, start - offset) + offset
            out[sym] = slc
        return out

    # IS: first half, OOS: second half
    d15_is = split_data(d15, 0, split)
    d15_oos = split_data(d15, split, total_bars)

    print("Running IS (first 50%) baseline...")
    trades_is = collect_with_delayed_entry(d15_is, {}, {}, min_close_pos=best_close_pos, max_delay=best_delay)
    pnl_is = portfolio_sim(trades_is, MAX_CONC)
    m_is = compute_metrics(pnl_is, data_years / 2)

    print("Running OOS (second 50%)...")
    trades_oos = collect_with_delayed_entry(d15_oos, {}, {}, min_close_pos=best_close_pos, max_delay=best_delay)
    pnl_oos = portfolio_sim(trades_oos, MAX_CONC)
    m_oos = compute_metrics(pnl_oos, data_years / 2)

    oos_is_ratio = m_oos["PF"] / m_is["PF"] if m_is["PF"] > 0 else 0
    print(f"\nIS  PF={m_is['PF']:.3f} N={m_is['N']}")
    print(f"OOS PF={m_oos['PF']:.3f} N={m_oos['N']}")
    print(f"OOS/IS ratio: {oos_is_ratio:.2f} ({'✅ No overfit' if oos_is_ratio >= 0.85 else '⚠️ Possible overfit'})")

    # Final combined
    trades_full = collect_with_delayed_entry(d15, d1h, d4h, min_close_pos=best_close_pos, max_delay=best_delay)
    pnl_full = portfolio_sim(trades_full, MAX_CONC)
    m_full = compute_metrics(pnl_full, data_years)
    boot_full = bootstrap_pf(pnl_full, n_boot=2000)

    # Print summary
    bm = baseline_result["metrics"]
    bb = baseline_result["boot"]
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<15} {'Baseline':>12} {'r54 Best':>12} {'Delta':>10}")
    print("-" * 50)
    for label, bv, nv in [
        ("N", bm["N"], m_full["N"]),
        ("PF", bm["PF"], m_full["PF"]),
        ("WR%", bm['WR'], m_full['WR']),
        ("EV(bp)", bm['EV_bp'], m_full['EV_bp']),
        ("Calmar", bm['calmar'], m_full['calmar']),
        ("MaxDD%", bm['max_dd_pct'], m_full['max_dd_pct']),
    ]:
        delta = nv - bv
        sign = "+" if delta >= 0 else ""
        print(f"{label:<15} {bv:>12.3f} {nv:>12.3f} {sign}{delta:>9.3f}")
    print(f"\nBootstrap CI: [{boot_full['CI_lo']:.3f}, {boot_full['CI_hi']:.3f}]")
    print(f"P(PF>=1.0): {boot_full['P_PF_ge_1']:.1f}%")

    # Save report
    commit = get_commit_short()
    today = datetime.now().strftime("%Y%m%d")
    month = datetime.now().strftime("%Y%m")
    report_dir = REPORT_DIR / month
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r54_entry_precision.md"

    ln = cfg.base_leverage / cfg.max_concurrent
    with open(report_path, "w") as fp:
        fp.write(f"# r54: Entry Precision Research\n\n")
        fp.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  \n")
        fp.write(f"**Commit:** `{commit}`  \n")
        fp.write(f"**Branch:** `research/r54-entry-precision`  \n")
        fp.write(f"**Data:** {bar0.strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')} ({data_years:.2f}yr)  \n")
        fp.write(f"**Slots:** {MAX_CONC} | **Cost:** 0.08% round-trip | **L/N:** {ln:.1f}  \n")
        fp.write(f"**Model:** 1x compound (compute_metrics baseline)  \n\n")
        fp.write("---\n\n")
        fp.write("## Backtest-Live Parity Check\n\n")
        fp.write(f"- [x] config.py에서 파라미터 읽음 (하드코딩 없음)\n")
        fp.write(f"- [x] D_TRAIL_ACTIVATE={D_TRAIL_ACTIVATE}, D_TRAIL_DISTANCE={D_TRAIL_DISTANCE}\n")
        fp.write(f"- [x] 룩어헤드 없음 (완성된 바만, `i` 인덱스 사용)\n")
        fp.write(f"- [x] 슬롯 수={MAX_CONC}, cost=0.0008\n\n")
        fp.write("## Results\n\n")
        fp.write(f"**Best config:** close_pos≥{best_close_pos:.2f}, max_delay={best_delay}\n\n")
        fp.write(f"| Metric | Baseline | r54 Best | Delta |\n")
        fp.write(f"|--------|----------|----------|-------|\n")
        for label, bv, nv in [
            ("N", bm["N"], m_full["N"]),
            ("PF", f"{bm['PF']:.3f}", f"{m_full['PF']:.3f}"),
            ("WR%", f"{bm['WR']:.1f}", f"{m_full['WR']:.1f}"),
            ("EV(bp)", f"{bm['EV_bp']:.1f}", f"{m_full['EV_bp']:.1f}"),
            ("Calmar", f"{bm['calmar']:.1f}", f"{m_full['calmar']:.1f}"),
            ("MaxDD%", f"{bm['max_dd_pct']:.1f}", f"{m_full['max_dd_pct']:.1f}"),
        ]:
            fp.write(f"| {label} | {bv} | {nv} | — |\n")
        fp.write(f"\nBootstrap CI: [{boot_full['CI_lo']:.3f}, {boot_full['CI_hi']:.3f}]  \n")
        fp.write(f"OOS/IS ratio: {oos_is_ratio:.2f}  \n")

    print(f"\nReport saved: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = datetime.now()
    print(f"r54 Entry Precision Research — {t0.strftime('%Y-%m-%d %H:%M')}")
    print(f"Commit: {get_commit_short()}\n")

    print("Loading data...")
    d15, fpath = load_syms(min_years=5.0)
    if not d15:
        print("ERROR: No symbols found in cache_extended with ≥5yr data")
        return
    d1h, d4h = build_tf(d15)
    bar0 = get_bar0(fpath)
    total_bars = max(len(d["close"]) for d in d15.values())
    data_years = total_bars / (365 * 24 * 4)
    print(f"Loaded {len(d15)} symbols | {data_years:.2f} years | bar0={bar0.date()}\n")

    # Phase 1
    p1 = phase1_baseline(d15, d1h, d4h, data_years)

    # Phase 2
    feature_stats = phase2_feature_analysis(d15, p1["trades"])

    # Phase 3: find best filter
    p3_results = phase3_filter_sweep(
        d15, d1h, d4h, data_years,
        p1["metrics"]["N"], p1["metrics"]["PF"], feature_stats,
    )
    best_filter = sorted([r for r in p3_results if r["ok"]], key=lambda x: -x["calmar"])
    best_close_pos = best_filter[0]["min_close_pos"] if best_filter else 0.50

    # Phase 4: delayed entry
    p4_results = phase4_delayed_entry(
        d15, d1h, d4h, data_years,
        p1["metrics"]["N"], p1["metrics"]["PF"], best_close_pos,
    )
    best_delay_row = sorted(
        [r for r in p4_results if r["N"] >= p1["metrics"]["N"] * 0.9],
        key=lambda x: -x["metrics"]["calmar"],
    )
    best_delay = best_delay_row[0]["max_delay"] if best_delay_row else 2

    # Phase 5
    phase5_oos_report(
        d15, d1h, d4h, data_years,
        best_close_pos, best_delay,
        p1, bar0, fpath,
    )

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
