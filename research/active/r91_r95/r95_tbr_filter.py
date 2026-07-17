"""r95: Taker Buy Ratio (TBR) filter for S_d8 Straddle + MA Cluster.

VR Squeeze (D_mom8) is excluded — r71 (2026-03-13) already tested TBR z-score gate:
  TBR ON vs OFF: PF delta=-0.004, 2/3 OOS years negative. Conclusion: FAIL, keep disabled.

Research questions:
  A) S_d8 Straddle (direction-neutral): Does entry-bar tbr extremeness (|tbr-0.5|)
     correlate with better straddle outcomes?
     H_extreme: strong imbalance → real squeeze → better post-squeeze move
     H_neutral:  balanced tbr → market uncertainty → both legs have a chance

  B) MA Cluster (directional): Does TBR alignment filter improve EV?
     LONG only when mean(tbr[-window:]) > threshold (buyers dominating)
     SHORT only when mean(tbr[-window:]) < (1-threshold) (sellers dominating)
     Why promising vs VR: MA cluster trail_activate=2% (VR=1%) → bad entries reach SL
     → entry filter has larger impact

Data:
  data/cache_extended/*_15m_2190d_*.npz — taker_buy column present (confirmed).
  TBR distribution: mean=0.494, std=0.061 (P75=0.533, P90=0.570).

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r91_r95/r95_tbr_filter.py
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    build_resampled,
    compute_metrics,
    portfolio_sim,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import compute_vr, gen_vr_dur_signal

# ── Config (always read from VolStraddleConfig — no hardcoding) ──────────────
cfg = VolStraddleConfig()

STRD_PARAMS = dict(
    sl=cfg.s_sl,  # 0.015
    tp=cfg.s_tp,  # 0.05
    max_hold=cfg.max_hold_15m,  # 192 bars at 15m
    trail_activate=cfg.s_trail_activate,  # 0.015
    trail_distance=cfg.s_trail_distance,  # 0.001
    s_vr=cfg.s_vr_threshold,  # 0.57
    s_dur=cfg.s_squeeze_dur,  # 3
)
MA_PARAMS = dict(
    cluster_pct=cfg.ma_cluster_pct,  # 0.02
    min_duration=cfg.ma_cluster_min_duration,  # 6
    max_duration=cfg.ma_cluster_max_duration,  # 24
    min_momentum=cfg.ma_cluster_min_momentum,  # 0.01
    sl=cfg.ma_cluster_sl,  # 0.02
    tp=0.20,  # dead cap — trail exits first
    trail_activate=cfg.ma_cluster_trail_activate,  # 0.02
    trail_distance=cfg.ma_cluster_trail_distance,  # 0.003
    max_hold=cfg.ma_cluster_max_hold_4h,  # 12 (4h bars)
    cost=0.0008,
    warmup=200,
)
BTC_GATE_PERIOD = cfg.ma_cluster_btc_ma_period_4h  # 50
MA_LN = cfg.ma_cluster_leverage / cfg.max_concurrent  # 21/10 = 2.1
COST_STRDL = 0.0016  # straddle: 2 legs × 0.08%
COST_DIR = 0.0008
RATIO = 16  # 15m → 4h
DATA_YEARS = 6.0
WARMUP = 200

# ── Data loading ─────────────────────────────────────────────────────────────


def load_data() -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    whitelist = set(cfg.symbol_whitelist) | {"BTCUSDT"}
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
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
        keys = set(npz.files)
        if not {"close", "high", "low", "taker_buy", "volume"}.issubset(keys):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


# ── Straddle: TBR bucket sim ─────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _sim_straddle_tbr(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    tbr: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,
    trail_distance: float,
    tbr_min: float,
    tbr_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim with TBR extremeness filter.

    Only enters when tbr_min <= tbr_extremeness[i] < tbr_max,
    where tbr_extremeness = abs(tbr[i] - 0.5).

    tbr_min=0.0, tbr_max=1.0 → no filter (baseline).
    tbr_min=0.10, tbr_max=1.0 → only extreme TBR entries.
    tbr_min=0.0, tbr_max=0.06 → only neutral TBR entries.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, np.float64)
    ent = np.empty(cap, np.int64)
    ext = np.empty(cap, np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        tbr_ext = abs(tbr[i] - 0.5)
        if not (tbr_min <= tbr_ext < tbr_max):
            i += 1
            continue

        la = True
        sa = True
        le = 0.0
        se = 0.0
        l_peak = 0.0
        s_peak = 0.0
        l_trailing = False
        s_trailing = False
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= entry * (1.0 - sl):
                    le = -sl
                    la = False
                else:
                    l_gain = (high[j] - entry) / entry
                    if l_gain > l_peak:
                        l_peak = l_gain
                    if not l_trailing and l_peak >= trail_activate:
                        l_trailing = True
                    if l_trailing:
                        trail_lvl = entry * (1.0 + l_peak - trail_distance)
                        if low[j] <= trail_lvl:
                            le = max(l_peak - trail_distance, -sl)
                            la = False
                    if la and high[j] >= entry * (1.0 + tp):
                        le = tp
                        la = False
            if sa:
                if high[j] >= entry * (1.0 + sl):
                    se = -sl
                    sa = False
                else:
                    s_gain = (entry - low[j]) / entry
                    if s_gain > s_peak:
                        s_peak = s_gain
                    if not s_trailing and s_peak >= trail_activate:
                        s_trailing = True
                    if s_trailing:
                        trail_lvl = entry * (1.0 - s_peak + trail_distance)
                        if high[j] >= trail_lvl:
                            se = max(s_peak - trail_distance, -sl)
                            sa = False
                    if sa and low[j] <= entry * (1.0 - tp):
                        se = tp
                        sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ── MA Cluster: TBR directional filter sim ───────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _sim_ma_tbr(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    tbr: np.ndarray,
    btc_close_4h: np.ndarray,
    btc_ma50_4h: np.ndarray,
    tbr_window: int,
    tbr_thresh: float,
    cluster_pct: float,
    min_duration: int,
    max_duration: int,
    min_momentum: float,
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MA cluster sim with BTC gate (always on) + optional TBR directional filter.

    TBR filter:
      LONG enters only when mean(tbr[i-w:i]) > tbr_thresh
      SHORT enters only when mean(tbr[i-w:i]) < (1 - tbr_thresh)
      tbr_thresh=0.0 → no TBR filter (baseline with BTC gate only).
    """
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    cnt = 0
    i = warmup
    cluster_count = 0
    in_cooldown = 0

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            in_cooldown = 0
            i += 1
            continue
        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0.0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            in_cooldown = 0
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue
        if max_duration > 0 and cluster_count > max_duration:
            i += 1
            continue
        if in_cooldown == 1:
            i += 1
            continue
        if ma_min <= close[i] <= ma_max:
            i += 1
            continue
        if abs(momentum[i]) < min_momentum:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

        # BTC 4h gate (always applied — gate_enabled hardcoded True here)
        if i < len(btc_ma50_4h) and not np.isnan(btc_ma50_4h[i]):
            if d == 1 and btc_close_4h[i] < btc_ma50_4h[i]:
                in_cooldown = 1
                i += 1
                continue

        # TBR directional filter (tbr_thresh=0.0 → skip filter)
        if tbr_thresh > 0.0:
            w = min(tbr_window, i)
            tbr_mean = 0.0
            for k in range(i - w, i):
                tbr_mean += tbr[k]
            tbr_mean /= w
            if d == 1 and tbr_mean <= tbr_thresh:
                in_cooldown = 1
                i += 1
                continue
            if d == -1 and tbr_mean >= (1.0 - tbr_thresh):
                in_cooldown = 1
                i += 1
                continue

        # Enter trade
        in_cooldown = 1
        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        peak_gain = 0.0
        trailing_active = False
        for j in range(i + 1, end):
            cur_gain = d * (close[j] - entry) / entry
            if cur_gain > peak_gain:
                peak_gain = cur_gain
            if not trailing_active and peak_gain >= trail_activate:
                trailing_active = True
            # SL first (server SL fires intra-bar)
            bar_loss = d * (entry - (high[j] if d == -1 else low[j])) / entry
            if bar_loss >= sl:
                trade_pnl = -sl
                eb = j
                break
            if trailing_active:
                trail_stop = peak_gain - trail_distance
                if cur_gain < trail_stop:
                    trade_pnl = max(trail_stop, -sl)
                    eb = j
                    break
            if d == 1 and high[j] >= entry * (1.0 + tp):
                trade_pnl = tp
                eb = j
                break
            if d == -1 and low[j] <= entry * (1.0 - tp):
                trade_pnl = tp
                eb = j
                break
        else:
            trade_pnl = d * (close[eb] - entry) / entry
        net = trade_pnl - cost
        if cnt < cap:
            pnl_arr[cnt] = net
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _compute_sma_vec(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    out[period - 1 :] = (cs[period - 1 :] - np.concatenate(([0.0], cs[: -period]))) / period
    return out


def _momentum8(close: np.ndarray) -> np.ndarray:
    mom = np.full(len(close), np.nan)
    mom[8:] = (close[8:] - close[:-8]) / close[:-8]
    return mom


def _bar_to_year(offset: int, total_bars: int, base_year: int = 2020) -> np.ndarray:
    """Map bar indices to year integers (approx, for bucketing)."""
    bars_per_year = 365.25 * 24 * 4  # 15m bars
    years = np.arange(total_bars, dtype=np.float64)
    years = base_year + (years + offset) / bars_per_year
    return years.astype(np.int64)


# ── Section A: Straddle TBR buckets ──────────────────────────────────────────


def run_straddle_tbr_buckets(data: dict) -> None:
    print("\n" + "═" * 70)
    print("SECTION A — S_d8 Straddle TBR Bucket Analysis")
    print("═" * 70)
    print(f"Config: s_sl={STRD_PARAMS['sl']} s_vr={STRD_PARAMS['s_vr']} s_dur={STRD_PARAMS['s_dur']}")
    print(f"        trail_activate={STRD_PARAMS['trail_activate']} trail_distance={STRD_PARAMS['trail_distance']}")
    print()

    syms = [s for s in data if s in set(cfg.symbol_whitelist)]

    # TBR extremeness buckets: (min_ext, max_ext, label)
    buckets = [
        (0.00, 0.03, "Neutral   [0.47–0.53]"),
        (0.03, 0.06, "Mild      [0.44–0.47 / 0.53–0.56]"),
        (0.06, 0.10, "Moderate  [0.40–0.44 / 0.56–0.60]"),
        (0.10, 1.00, "Extreme   [<0.40 / >0.60]"),
        (0.00, 1.00, "ALL (baseline)"),
    ]

    sl = STRD_PARAMS["sl"]
    tp = STRD_PARAMS["tp"]
    mh = STRD_PARAMS["max_hold"]
    trail_act = STRD_PARAMS["trail_activate"]
    trail_dist = STRD_PARAMS["trail_distance"]
    s_vr = STRD_PARAMS["s_vr"]
    s_dur = STRD_PARAMS["s_dur"]

    bucket_results: list[dict] = []

    for tbr_min, tbr_max, label in buckets:
        all_pnl: list[np.ndarray] = []

        for sym in syms:
            d = data[sym]
            c, h, l = d["close"], d["high"], d["low"]
            vol = d["volume"]
            tb = d["taker_buy"]
            tbr = np.where(vol > 0, tb / vol, 0.5)

            # VR signal (ATR-based, matching canonical trade_sim)
            vr_arr = compute_vr(c, h, l, cfg.vr_short, cfg.vr_long)
            sig = gen_vr_dur_signal(vr_arr, s_vr, s_dur, WARMUP)

            pnl, _, _ = _sim_straddle_tbr(
                c, h, l, sig, tbr, sl, tp, mh, COST_STRDL,
                trail_act, trail_dist, tbr_min, tbr_max,
            )
            if len(pnl) > 0:
                all_pnl.append(pnl)

        if not all_pnl:
            continue
        combined = np.concatenate(all_pnl)
        m = compute_metrics(combined, DATA_YEARS)

        bucket_results.append({
            "label": label,
            "tbr_min": tbr_min,
            "tbr_max": tbr_max,
            **m,
        })

    print(f"{'Bucket':<38} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} {'MaxDD':>7}")
    print("-" * 75)
    for r in bucket_results:
        marker = " ← baseline" if r["tbr_min"] == 0.0 and r["tbr_max"] == 1.0 else ""
        print(
            f"{r['label']:<38} {r['N']:>5} {r['PF']:>6.3f} {r['WR']:>6.1f}% "
            f"{r['EV_bp']:>+8.1f} {r['max_dd_pct']:>6.1f}%{marker}"
        )

    # Determine H_extreme vs H_neutral
    base = next(r for r in bucket_results if r["tbr_min"] == 0.0 and r["tbr_max"] == 1.0)
    extreme = next(r for r in bucket_results if r["tbr_min"] == 0.10)
    neutral = next(r for r in bucket_results if r["tbr_min"] == 0.00 and r["tbr_max"] == 0.03)

    print()
    print("Hypothesis verdict:")
    if extreme["EV_bp"] > base["EV_bp"] + 5 and extreme["PF"] > base["PF"]:
        print(f"  ✅ H_EXTREME confirmed: extreme TBR → PF={extreme['PF']:.3f} vs base={base['PF']:.3f}")
        print(f"     → Filter: keep only extreme TBR entries (|tbr-0.5| > 0.10)")
    elif neutral["EV_bp"] > base["EV_bp"] + 5 and neutral["PF"] > base["PF"]:
        print(f"  ✅ H_NEUTRAL confirmed: neutral TBR → PF={neutral['PF']:.3f} vs base={base['PF']:.3f}")
        print(f"     → Filter: keep only neutral TBR entries (|tbr-0.5| < 0.03)")
    else:
        print(f"  ❌ Neither hypothesis confirmed. No consistent TBR signal for straddle.")
        print(f"     → Straddle TBR filter: FAIL (do not implement)")

    return bucket_results


# ── Section B: MA Cluster TBR sweep ─────────────────────────────────────────


def _build_btc_4h_arrays(data: dict, n_15m: int) -> tuple[np.ndarray, np.ndarray]:
    """Build BTC 4h close and MA50 arrays aligned to 15m bar space."""
    btc = data.get("BTCUSDT")
    if btc is None:
        return np.zeros(n_15m // RATIO + 1), np.full(n_15m // RATIO + 1, np.nan)
    btc_4h_data = build_resampled({"BTCUSDT": btc}, RATIO)
    if "BTCUSDT" not in btc_4h_data:
        return np.zeros(n_15m // RATIO + 1), np.full(n_15m // RATIO + 1, np.nan)
    c4 = btc_4h_data["BTCUSDT"]["close"]
    sma50_4h = _compute_sma_vec(c4, BTC_GATE_PERIOD)
    return c4, sma50_4h


def _build_ma_arrays(d: dict, n_15m: int) -> tuple:
    """Resample to 4h and compute SMAs, momentum, TBR."""
    resampled = build_resampled({"_sym": d}, RATIO)
    if "_sym" not in resampled:
        # Fallback: empty arrays
        n4 = len(d["close"]) // RATIO
        empty = np.full(n4, np.nan)
        return empty, empty, empty, empty, empty, empty, empty, empty, np.full(n4, 0.5)
    r = resampled["_sym"]
    c4, h4, l4 = r["close"], r["high"], r["low"]

    sma20 = _compute_sma_vec(c4, 20)
    sma50 = _compute_sma_vec(c4, 50)
    sma100 = _compute_sma_vec(c4, 100)
    sma200 = _compute_sma_vec(c4, 200)
    mom = _momentum8(c4)

    # TBR: aggregate 15m taker_buy/volume into 4h bars
    vol_15 = d["volume"]
    tb_15 = d["taker_buy"]
    n4 = len(c4)
    tbr_4h = np.full(n4, 0.5)
    for i in range(n4):
        s = i * RATIO
        e = s + RATIO
        v = vol_15[s:e].sum()
        tb = tb_15[s:e].sum()
        if v > 0:
            tbr_4h[i] = tb / v

    return c4, h4, l4, sma20, sma50, sma100, sma200, mom, tbr_4h


def run_ma_tbr_sweep(data: dict) -> None:
    print("\n" + "═" * 70)
    print("SECTION B — MA Cluster TBR Directional Filter Sweep")
    print("═" * 70)
    print(f"Config: sl={MA_PARAMS['sl']} trail_activate={MA_PARAMS['trail_activate']}")
    print(f"        trail_distance={MA_PARAMS['trail_distance']} max_hold={MA_PARAMS['max_hold']}×4h")
    print(f"BTC gate: always ON (MA50_4h, period={BTC_GATE_PERIOD})")
    print()

    syms = [s for s in data if s in set(cfg.symbol_whitelist)]
    max_bars = max(len(data[s]["close"]) for s in syms)

    # Pre-build BTC 4h arrays
    btc_c4, btc_ma50 = _build_btc_4h_arrays(data, max_bars)

    # TBR sweep grid
    tbr_windows = [1, 4, 16]  # 15m → 1bar(15m), 4bars(1h), 16bars(4h)
    tbr_thresholds = [0.00, 0.50, 0.52, 0.55, 0.57, 0.60]  # 0.0 = gate-only baseline

    p = MA_PARAMS
    results: list[dict] = []

    for window in tbr_windows:
        for thresh in tbr_thresholds:
            all_pnl: list[np.ndarray] = []
            all_ent: list[tuple[str, int]] = []

            for sym in syms:
                d = data[sym]
                offset = int(d.get("_offset", 0))

                c4, h4, l4, sma20, sma50, sma100, sma200, mom, tbr_4h = _build_ma_arrays(d, max_bars)
                btc_offset_4h = offset // RATIO
                btc_start = btc_offset_4h
                btc_end = btc_start + len(c4)
                btc_c = btc_c4[btc_start:btc_end] if btc_end <= len(btc_c4) else np.zeros(len(c4))
                btc_ma = btc_ma50[btc_start:btc_end] if btc_end <= len(btc_ma50) else np.zeros(len(c4))

                # Pad/trim to same length
                n4 = len(c4)
                btc_c = btc_c[:n4] if len(btc_c) >= n4 else np.concatenate([np.zeros(n4 - len(btc_c)), btc_c])
                btc_ma = btc_ma[:n4] if len(btc_ma) >= n4 else np.concatenate([np.full(n4 - len(btc_ma), np.nan), btc_ma])

                pnl, ent, _ = _sim_ma_tbr(
                    c4, h4, l4, sma20, sma50, sma100, sma200, mom,
                    tbr_4h, btc_c, btc_ma,
                    window, thresh,
                    p["cluster_pct"], p["min_duration"], p["max_duration"],
                    p["min_momentum"], p["sl"], p["tp"],
                    p["trail_activate"], p["trail_distance"],
                    p["max_hold"], p["cost"], p["warmup"],
                )
                if len(pnl) > 0:
                    all_pnl.append(pnl)
                    all_ent.extend([(sym, int(e) + offset) for e in ent])

            if not all_pnl:
                continue
            combined = np.concatenate(all_pnl)
            m = compute_metrics(combined, DATA_YEARS)
            results.append({
                "window": window,
                "thresh": thresh,
                "label": f"w={window:2d} t={thresh:.2f}",
                **m,
            })

    # Baseline = window=1, thresh=0.00 (gate-only)
    baseline = next(r for r in results if r["thresh"] == 0.00 and r["window"] == 1)
    b_pf = baseline["PF"]
    b_ev = baseline["EV_bp"]
    b_n = baseline["N"]

    print(f"{'Config':<14} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} {'MaxDD':>7}  Status")
    print("-" * 70)
    for r in results:
        if r["thresh"] == 0.00:
            status = "← baseline (gate-only)"
        elif r["PF"] >= b_pf and r["EV_bp"] >= b_ev and r["N"] >= b_n * 0.62:
            status = "✅ PASS"
        elif r["PF"] >= b_pf - 0.05 or r["EV_bp"] >= b_ev - 5:
            status = "~marginal"
        else:
            status = "✗"
        print(
            f"{r['label']:<14} {r['N']:>5} {r['PF']:>6.3f} {r['WR']:>6.1f}% "
            f"{r['EV_bp']:>+8.1f} {r['max_dd_pct']:>6.1f}%  {status}"
        )

    # Find best config
    candidates = [r for r in results if r["thresh"] > 0 and r["PF"] >= b_pf and r["EV_bp"] >= b_ev and r["N"] >= b_n * 0.62]
    if candidates:
        best = max(candidates, key=lambda r: r["EV_bp"])
        print(f"\n🏆 Best config: window={best['window']}, threshold={best['thresh']:.2f}")
        print(f"   PF={best['PF']:.3f} (+{best['PF']-b_pf:+.3f}), EV={best['EV_bp']:+.1f}bp (+{best['EV_bp']-b_ev:+.1f}), N={best['N']}")
        return best, results, baseline
    else:
        print(f"\n❌ No TBR config improves on gate-only baseline (PF={b_pf:.3f}, EV={b_ev:+.1f}bp)")
        print(f"   MA cluster TBR filter: FAIL — do not implement")
        return None, results, baseline


# ── Section C: OOS year-by-year validation for best MA config ────────────────


def run_ma_oos_validation(data: dict, best_window: int, best_thresh: float, baseline_thresh: float = 0.00) -> None:
    print("\n" + "═" * 70)
    print(f"SECTION C — MA Cluster OOS Year Validation (window={best_window}, thresh={best_thresh:.2f})")
    print("═" * 70)

    syms = [s for s in data if s in set(cfg.symbol_whitelist)]
    max_bars = max(len(data[s]["close"]) for s in syms)
    btc_c4, btc_ma50 = _build_btc_4h_arrays(data, max_bars)

    p = MA_PARAMS
    bars_per_year = int(365.25 * 24 / 4)  # 4h bars per year

    def _run_year(year: int, thresh: float) -> dict | None:
        all_pnl: list[np.ndarray] = []
        for sym in syms:
            d = data[sym]
            offset = int(d.get("_offset", 0))
            c4, h4, l4, sma20, sma50, sma100, sma200, mom, tbr_4h = _build_ma_arrays(d, max_bars)

            # Year slice in 4h bar space
            total_4h = len(c4)
            # Approximate: 2020-01-01 is bar 0, each year = bars_per_year bars
            year_start_4h = max(0, (year - 2020) * bars_per_year - int(d.get("_offset", 0)) // RATIO)
            year_end_4h = min(total_4h, year_start_4h + bars_per_year)
            if year_end_4h - year_start_4h < 50:
                continue

            c_yr = c4[year_start_4h:year_end_4h]
            h_yr = h4[year_start_4h:year_end_4h]
            l_yr = l4[year_start_4h:year_end_4h]
            sma20_yr = sma20[year_start_4h:year_end_4h]
            sma50_yr = sma50[year_start_4h:year_end_4h]
            sma100_yr = sma100[year_start_4h:year_end_4h]
            sma200_yr = sma200[year_start_4h:year_end_4h]
            mom_yr = mom[year_start_4h:year_end_4h]
            tbr_yr = tbr_4h[year_start_4h:year_end_4h]

            btc_offset_4h = int(d.get("_offset", 0)) // RATIO
            b_start = btc_offset_4h + year_start_4h
            b_end = b_start + len(c_yr)
            btc_c = btc_c4[b_start:b_end] if b_end <= len(btc_c4) else np.zeros(len(c_yr))
            btc_ma = btc_ma50[b_start:b_end] if b_end <= len(btc_ma50) else np.zeros(len(c_yr))

            n4 = len(c_yr)
            btc_c = btc_c[:n4] if len(btc_c) >= n4 else np.concatenate([np.zeros(n4 - len(btc_c)), btc_c])
            btc_ma = btc_ma[:n4] if len(btc_ma) >= n4 else np.concatenate([np.full(n4 - len(btc_ma), np.nan), btc_ma])

            pnl, _, _ = _sim_ma_tbr(
                c_yr, h_yr, l_yr, sma20_yr, sma50_yr, sma100_yr, sma200_yr, mom_yr,
                tbr_yr, btc_c, btc_ma,
                best_window, thresh,
                p["cluster_pct"], p["min_duration"], p["max_duration"],
                p["min_momentum"], p["sl"], p["tp"],
                p["trail_activate"], p["trail_distance"],
                p["max_hold"], p["cost"], p["warmup"],
            )
            if len(pnl) > 0:
                all_pnl.append(pnl)

        if not all_pnl:
            return None
        combined = np.concatenate(all_pnl)
        return compute_metrics(combined, 1.0)

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    print(f"\n{'Year':<6} {'N_base':>7} {'PF_base':>8} {'N_tbr':>7} {'PF_tbr':>8} {'ΔPF':>7} {'Pass?'}")
    print("-" * 60)
    passes = 0
    for yr in years:
        base_m = _run_year(yr, 0.00)
        tbr_m = _run_year(yr, best_thresh)
        if base_m is None or tbr_m is None:
            print(f"{yr:<6} {'N/A':>7}")
            continue
        delta = tbr_m["PF"] - base_m["PF"]
        ok = tbr_m["PF"] >= base_m["PF"] and tbr_m["N"] >= base_m["N"] * 0.62
        if ok:
            passes += 1
        print(
            f"{yr:<6} {base_m['N']:>7} {base_m['PF']:>8.3f} {tbr_m['N']:>7} {tbr_m['PF']:>8.3f} "
            f"{delta:>+7.3f}  {'✅' if ok else '❌'}"
        )

    print(f"\n{passes}/{len(years)} years passed OOS criterion (TBR PF > baseline PF, N ≥ 62%)")
    if passes >= 4:
        print("✅ OOS PASS — TBR filter is robust across years")
    else:
        print("❌ OOS FAIL — TBR filter not consistent across years → do not implement")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("r95: TBR Buy Pressure Filter — S_d8 Straddle + MA Cluster")
    print(f"Config: {cfg.__class__.__name__}()")
    print(f"  VR Squeeze: EXCLUDED (r71 FAIL: PF delta=-0.004)")
    print()

    print("Loading data...")
    data = load_data()
    syms_loaded = [s for s in data if s in set(cfg.symbol_whitelist)]
    print(f"Loaded {len(syms_loaded)}/{len(cfg.symbol_whitelist)} whitelist symbols + BTC")
    print(f"taker_buy present: {all('taker_buy' in data[s] for s in syms_loaded)}")

    # TBR distribution check
    all_tbr: list[float] = []
    for sym in syms_loaded[:10]:
        d = data[sym]
        vol = d["volume"]
        tb = d["taker_buy"]
        tbr = np.where(vol > 0, tb / vol, 0.5)
        all_tbr.extend(tbr[200:].tolist())
    tbr_arr = np.array(all_tbr)
    print(f"\nTBR sample distribution (10 syms): mean={tbr_arr.mean():.3f} std={tbr_arr.std():.3f}")
    print(f"  P10={np.percentile(tbr_arr,10):.3f} P25={np.percentile(tbr_arr,25):.3f} "
          f"P50={np.percentile(tbr_arr,50):.3f} P75={np.percentile(tbr_arr,75):.3f} "
          f"P90={np.percentile(tbr_arr,90):.3f}")

    # ── Section A: Straddle ───────────────────────────────────────────────────
    strd_results = run_straddle_tbr_buckets(data)

    # ── Section B: MA Cluster sweep ──────────────────────────────────────────
    best_ma, ma_results, ma_baseline = run_ma_tbr_sweep(data)

    # ── Section C: OOS validation (only if PASS) ─────────────────────────────
    if best_ma is not None:
        run_ma_oos_validation(data, best_ma["window"], best_ma["thresh"])

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("FINAL VERDICT")
    print("═" * 70)
    print(f"  VR Squeeze TBR:    EXCLUDED (r71 FAIL)")
    print(f"  S_d8 Straddle TBR: {'→ see Section A above'}")
    print(f"  MA Cluster TBR:    {'→ PASS (implement as disabled-default)' if best_ma else '→ FAIL (do not implement)'}")
    if best_ma:
        print(f"    Recommended: window={best_ma['window']}, threshold={best_ma['thresh']:.2f}")
        print(f"    Add to config.py: ma_cluster_tbr_enabled=False, ma_cluster_tbr_window={best_ma['window']}, ma_cluster_tbr_threshold={best_ma['thresh']:.2f}")


if __name__ == "__main__":
    main()
