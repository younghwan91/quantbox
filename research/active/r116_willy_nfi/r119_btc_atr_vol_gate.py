"""r119: BTC ATR Volatility Gate for MA Cluster.

Hypothesis:
  When BTC 4h ATR is elevated vs its rolling average, MA cluster breakouts
  are noise-driven — they break in random directions (both LONG and SHORT
  fail). The existing BTC MA50 trend gate only blocks direction-mismatched
  LONGs; it cannot catch high-volatility whipsaw periods.

  Evidence: Apr 21-22 2026 — 4 consecutive SL hits (GRTUSDT SHORT,
  IOSTUSDT SHORT, AVAXUSDT LONG, XMRUSDT LONG). Mixed directions → not a
  trend issue, a volatility issue. MA50 gate was active but useless here.

Method:
  ATR ratio = BTC_4h_ATR[i] / rolling_mean(BTC_4h_ATR, window)
  Block ALL entries (LONG + SHORT) when ATR ratio > threshold.

  Sweep:
    atr_window  ∈ [14, 20, 50]
    atr_thresh  ∈ [1.3, 1.5, 2.0, 2.5]

  Also: trail_activate 0.010 (current live) vs 0.015 (r115 best) comparison,
  since r115 sweep grid started at 0.015 and never tested 0.010 directly.

Baseline for comparison:
  r115 best: trail_activate=0.015, BTC MA50 gate ON, no ATR gate → PF=2.681

Pass criteria (deploy to live):
  - PF >= 2.681 (beat r115 best)
  - N >= r115_best_N × 0.80  (don't over-filter, keep ≥ 80% of trades)
  - MaxDD <= 14.1%  (r115 best MaxDD)
  - Every year with N≥20: PF >= 1.0

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r116_willy_nfi/r119_btc_atr_vol_gate.py
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import bootstrap_pf, compute_metrics, portfolio_sim
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config ───────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

MA_PARAMS = {
    "cluster_pct":    cfg.ma_cluster_pct,
    "min_duration":   cfg.ma_cluster_min_duration,
    "max_duration":   cfg.ma_cluster_max_duration,
    "min_momentum":   cfg.ma_cluster_min_momentum,
    "sl":             cfg.ma_cluster_sl,
    "tp":             0.20,
    "max_hold":       cfg.ma_cluster_max_hold_4h,
    "cost":           0.0008,
    "warmup":         200,
}
BTC_MA_PERIOD   = cfg.ma_cluster_btc_ma_period_4h   # 50
MA_LN           = cfg.ma_cluster_leverage / cfg.max_concurrent  # 2.1
DATA_YEARS      = 6.0
RATIO           = 16  # 15m → 4h

# r115 reference (beat this)
R115_BEST_PF    = 2.681
R115_BEST_N     = 636
R115_BEST_DD    = 14.1

# Sweep grids
TRAIL_GRID      = [0.010, 0.015]          # current live vs r115 best
ATR_WINDOW_GRID = [14, 20, 50]
ATR_THRESH_GRID = [1.3, 1.5, 2.0, 2.5]
TRAIL_DISTANCE  = 0.005                    # Binance floor (Rule #8)


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    all_syms = set(whitelist) | {"BTCUSDT"}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in all_syms:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
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


def resample_4h(c15: np.ndarray, h15: np.ndarray, l15: np.ndarray) -> tuple:
    n4 = len(c15) // RATIO
    close = c15[RATIO - 1 : n4 * RATIO : RATIO].copy()
    high  = np.array([np.max(h15[i : i + RATIO]) for i in range(0, n4 * RATIO, RATIO)])
    low   = np.array([np.min(l15[i : i + RATIO]) for i in range(0, n4 * RATIO, RATIO)])
    return close, high, low


from quantbox.research.harness import compute_sma  # 이주(P2): 재구현 제거 (라이브 SMA 정렬)


def momentum8(close: np.ndarray) -> np.ndarray:
    mom = np.full(len(close), np.nan)
    for i in range(8, len(close)):
        mom[i] = (close[i] - close[i - 8]) / close[i - 8]
    return mom


def compute_atr(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """True Average Range over `period` bars."""
    n = len(close)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        hl   = high[i]  - low[i]
        hpc  = abs(high[i]  - close[i - 1])
        lpc  = abs(low[i]   - close[i - 1])
        tr[i] = max(hl, hpc, lpc)
    atr = np.full(n, np.nan)
    for i in range(period, n):
        atr[i] = np.mean(tr[i - period + 1 : i + 1])
    return atr


def compute_atr_ratio(atr: np.ndarray, window: int) -> np.ndarray:
    """ATR ratio = ATR[i] / rolling_mean(ATR, window).
    Values > 1.0 mean current vol is above the rolling average.
    """
    ratio = np.full(len(atr), np.nan)
    for i in range(window, len(atr)):
        base = np.nanmean(atr[i - window : i])
        if base > 0 and not np.isnan(atr[i]):
            ratio[i] = atr[i] / base
    return ratio


# ── Numba sim ─────────────────────────────────────────────────────────────────

@numba.jit(nopython=True, cache=True)
def sim_ma_cluster(
    close, high, low, sma20, sma50, sma100, sma200, momentum,
    btc_close_4h, btc_ma50_4h, btc_ma_gate,      # BTC trend gate
    btc_atr_ratio_4h, atr_thresh, atr_gate,       # BTC ATR vol gate (NEW)
    cluster_pct, min_duration, max_duration, min_momentum,
    sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
):
    """MA cluster sim with optional BTC trend gate AND ATR volatility gate.

    btc_ma_gate=1  : block LONG when BTC close < BTC MA50 (trend filter)
    atr_gate=1     : block ALL (LONG+SHORT) when ATR ratio > atr_thresh (vol filter)

    Both gates set in_cooldown=1 to match live strategy behaviour.
    SL-first fix: SL checked before trail activation.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    dir_arr = np.empty(cap, np.int64)
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

        # ── BTC MA50 trend gate (LONG only) ──────────────────────────────────
        if btc_ma_gate == 1 and d == 1 and i < len(btc_ma50_4h) and not np.isnan(btc_ma50_4h[i]):
            if btc_close_4h[i] < btc_ma50_4h[i]:
                in_cooldown = 1
                i += 1
                continue

        # ── BTC ATR volatility gate (LONG + SHORT) ───────────────────────────
        if atr_gate == 1 and i < len(btc_atr_ratio_4h) and not np.isnan(btc_atr_ratio_4h[i]):
            if btc_atr_ratio_4h[i] > atr_thresh:
                in_cooldown = 1
                i += 1
                continue

        # ── Enter trade ───────────────────────────────────────────────────────
        in_cooldown = 1
        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        peak_gain = 0.0
        trailing_active = False

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_active and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_active and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
        else:
            trade_pnl = ((close[eb] - entry) / entry) * d

        if cnt < cap:
            pnl_arr[cnt] = trade_pnl - cost
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            dir_arr[cnt] = d
            cnt += 1

        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], dir_arr[:cnt]


# ── Per-symbol runner ─────────────────────────────────────────────────────────

def run_symbol(
    sym: str,
    d15: dict,
    btc_close_4h: np.ndarray,
    btc_ma50_4h: np.ndarray,
    btc_atr_ratio_4h: np.ndarray,
    btc_ma_gate: int,
    atr_gate: int,
    atr_thresh: float,
    trail_activate: float,
) -> tuple[list, list, list]:
    """Run sim for one symbol. Returns (pnl_list, ent15_list, ext15_list)."""
    c15, h15, l15 = d15["close"], d15["high"], d15["low"]
    c4, h4, l4    = resample_4h(c15, h15, l15)
    sma20  = compute_sma(c4, 20)
    sma50  = compute_sma(c4, 50)
    sma100 = compute_sma(c4, 100)
    sma200 = compute_sma(c4, 200)
    mom    = momentum8(c4)
    offset = d15.get("_offset", 0)

    # Align BTC 4h arrays to this symbol's offset
    btc_off = offset // RATIO
    n4 = len(c4)

    def _align(arr: np.ndarray) -> np.ndarray:
        if len(arr) >= btc_off + n4:
            return arr[btc_off : btc_off + n4]
        return np.full(n4, np.nan)

    btc_close_al     = _align(btc_close_4h)
    btc_ma50_al      = _align(btc_ma50_4h)
    btc_atr_ratio_al = _align(btc_atr_ratio_4h)

    pnl, ent4, ext4, _ = sim_ma_cluster(
        c4, h4, l4, sma20, sma50, sma100, sma200, mom,
        btc_close_al, btc_ma50_al, btc_ma_gate,
        btc_atr_ratio_al, atr_thresh, atr_gate,
        MA_PARAMS["cluster_pct"], MA_PARAMS["min_duration"], MA_PARAMS["max_duration"],
        MA_PARAMS["min_momentum"], MA_PARAMS["sl"], MA_PARAMS["tp"],
        trail_activate, TRAIL_DISTANCE, MA_PARAMS["max_hold"],
        MA_PARAMS["cost"], MA_PARAMS["warmup"],
    )

    ent15 = (ent4 * RATIO + offset).astype(np.int64)
    ext15 = ((ext4 + 1) * RATIO - 1 + offset).astype(np.int64)
    return pnl.tolist(), ent15.tolist(), ext15.tolist()


def build_trades(
    data: dict,
    syms: list[str],
    btc_close_4h: np.ndarray,
    btc_ma50_4h: np.ndarray,
    btc_atr_ratio_4h: np.ndarray,
    btc_ma_gate: int,
    atr_gate: int,
    atr_thresh: float,
    trail_activate: float,
) -> list[tuple[str, int, int, float]]:
    trades = []
    for sym in syms:
        try:
            pnl_list, ent_list, ext_list = run_symbol(
                sym, data[sym],
                btc_close_4h, btc_ma50_4h, btc_atr_ratio_4h,
                btc_ma_gate, atr_gate, atr_thresh, trail_activate,
            )
            for p, e, x in zip(pnl_list, ent_list, ext_list):
                trades.append((sym, int(e), int(x), float(p)))
        except Exception as err:
            print(f"  SKIP {sym}: {err}")
    return trades


def metrics(trades: list) -> dict:
    if len(trades) < 10:
        return {}
    pnl = portfolio_sim(trades, max_concurrent=cfg.max_concurrent)
    return compute_metrics(pnl, DATA_YEARS)


def yearly(trades: list) -> dict[int, dict]:
    bars_per_year = 35064
    buckets: dict[int, list[float]] = {}
    for _, ent, _, pnl in trades:
        yr = 2020 + int(ent) // bars_per_year
        buckets.setdefault(yr, []).append(float(pnl))
    out = {}
    for yr, pnls in sorted(buckets.items()):
        arr = np.array(pnls)
        w = arr[arr > 0]; l = arr[arr < 0]
        pf = (w.sum() / abs(l.sum())) if len(l) > 0 and abs(l.sum()) > 0 else float("inf")
        out[yr] = {"N": len(pnls), "PF": pf, "WR": 100 * len(w) / max(len(pnls), 1)}
    return out


# ── Warmup ───────────────────────────────────────────────────────────────────

def warmup_numba() -> None:
    c = np.ones(500); nan = np.full(500, np.nan); nan[200:] = 1.0
    btc = np.ones(500) * 90000.0; bma = np.full(500, np.nan); bma[50:] = 88000.0
    atr_r = np.full(500, 1.0)
    for mg in (0, 1):
        for ag in (0, 1):
            sim_ma_cluster(
                c, c, c, nan, nan, nan, nan, nan,
                btc, bma, mg, atr_r, 1.5, ag,
                0.02, 6, 24, 0.01, 0.02, 0.20, 0.015, 0.005, 6, 0.0, 200,
            )


# ── Report helpers ────────────────────────────────────────────────────────────

HDR = f"  {'Variant':<42} {'N':>5}  {'PF':>6}  {'WR':>5}  {'MaxDD':>6}  {'EV(bp)':>7}"


def print_row(label: str, m: dict, mark: str = "") -> None:
    if not m:
        print(f"  {label}: no data")
        return
    ev = m.get("EV_bp", float("nan"))
    flag = " ✅" if mark == "pass" else (" 🔴" if mark == "fail" else "")
    print(f"  {label:<42} {m['N']:>5}  {m['PF']:>6.3f}  {m['WR']:>5.1f}%  "
          f"{m['max_dd_pct']:>6.1f}%  {ev:>7.1f}{flag}")


def passes(m: dict, yr_m: dict) -> bool:
    if not m:
        return False
    if m["PF"] < R115_BEST_PF:
        return False
    if m["N"] < R115_BEST_N * 0.80:
        return False
    if m["max_dd_pct"] > R115_BEST_DD:
        return False
    for yr, yd in yr_m.items():
        if yd["N"] >= 20 and yd["PF"] < 1.0:
            return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    commit = get_commit()
    today  = datetime.now().strftime("%Y%m%d")

    print("r119: BTC ATR Volatility Gate for MA Cluster")
    print("=" * 70)
    print(f"Config: cluster_pct={MA_PARAMS['cluster_pct']}  min_dur={MA_PARAMS['min_duration']}  "
          f"sl={MA_PARAMS['sl']}  trail_dist={TRAIL_DISTANCE}")
    print(f"Reference (r115 best): PF={R115_BEST_PF}  N={R115_BEST_N}  MaxDD={R115_BEST_DD}%")
    print(f"ATR windows: {ATR_WINDOW_GRID}  thresholds: {ATR_THRESH_GRID}")
    print(f"Trail grid: {TRAIL_GRID}")
    print()

    # Load data
    print("Loading data...")
    data = load_data(cfg.symbol_whitelist)
    print(f"  {len(data)} symbols loaded")
    if "BTCUSDT" not in data:
        print("ERROR: BTCUSDT missing"); return

    syms = [s for s in cfg.symbol_whitelist if s in data and s != "BTCUSDT"]

    # BTC 4h arrays
    btc = data["BTCUSDT"]
    btc_c4, btc_h4, btc_l4 = resample_4h(btc["close"], btc["high"], btc["low"])
    btc_ma50_4h = compute_sma(btc_c4, BTC_MA_PERIOD)
    btc_atr_14  = compute_atr(btc_c4, btc_h4, btc_l4, 14)
    btc_atr_20  = compute_atr(btc_c4, btc_h4, btc_l4, 20)
    btc_atr_50  = compute_atr(btc_c4, btc_h4, btc_l4, 50)
    print(f"  BTC 4h bars: {len(btc_c4)}\n")

    # Precompute ATR ratio arrays for all window/threshold combos
    atr_ratio_cache: dict[int, np.ndarray] = {}
    for w in ATR_WINDOW_GRID:
        atr = btc_atr_14 if w == 14 else (btc_atr_20 if w == 20 else btc_atr_50)
        atr_ratio_cache[w] = compute_atr_ratio(atr, w)

    # Warmup numba
    print("Numba warmup... ", end="", flush=True)
    warmup_numba()
    print("done\n")

    # ════════════════════════════════════════════════════════════════════════
    #  Section 1: trail_activate 0.010 vs 0.015 (gate ON, no ATR gate)
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("SECTION 1: trail_activate comparison (BTC MA50 gate ON, no ATR gate)")
    print(f"  r115 did NOT test trail_activate=0.010 (grid started at 0.015)")
    print("=" * 70)
    print(HDR)
    print("  " + "-" * 68)

    trail_results: dict[float, dict] = {}
    trail_trades:  dict[float, list] = {}
    for ta in TRAIL_GRID:
        t = build_trades(data, syms, btc_c4, btc_ma50_4h,
                         np.full(len(btc_c4), np.nan),  # ATR gate OFF
                         btc_ma_gate=1, atr_gate=0, atr_thresh=9.9, trail_activate=ta)
        m = metrics(t)
        trail_results[ta] = m
        trail_trades[ta]  = t
        mark = "pass" if m and m["PF"] >= R115_BEST_PF else ""
        print_row(f"trail={ta:.3f} + MA50 gate (no ATR gate)", m, mark)

    # ════════════════════════════════════════════════════════════════════════
    #  Section 2: ATR Volatility Gate sweep
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 2: ATR Volatility Gate sweep (best trail from Sec1 + MA50 gate)")
    print("=" * 70)

    # Use trail that gave better PF in Section 1
    best_trail_sec1 = max(TRAIL_GRID, key=lambda ta: trail_results[ta].get("PF", 0))
    print(f"  Using trail_activate={best_trail_sec1} (best from Section 1)\n")
    print(HDR)
    print("  " + "-" * 68)

    # Reference: best trail + MA50 gate, no ATR gate
    ref_trades = trail_trades[best_trail_sec1]
    ref_m      = trail_results[best_trail_sec1]
    print_row(f"REF: trail={best_trail_sec1:.3f} + MA50 gate (no ATR)", ref_m)
    print()

    best_atr: dict = {"pf": 0.0, "w": 0, "t": 0.0, "m": {}, "trades": [], "yr": {}}

    for w in ATR_WINDOW_GRID:
        print(f"  ATR window={w} bars ({w * 4}h = {w * 4 / 24:.1f} days):")
        for thresh in ATR_THRESH_GRID:
            t = build_trades(data, syms, btc_c4, btc_ma50_4h,
                             atr_ratio_cache[w],
                             btc_ma_gate=1, atr_gate=1, atr_thresh=thresh,
                             trail_activate=best_trail_sec1)
            m = metrics(t)
            yr_m = yearly(t)
            ok = passes(m, yr_m)
            n_filtered = len(ref_trades) - len(t) if ref_trades else 0
            n_pct = 100 * len(t) / max(len(ref_trades), 1)
            label = f"  ATR w={w} thresh={thresh:.1f}x  (N%={n_pct:.0f}%)"
            print_row(label, m, "pass" if ok else "")

            if m and m.get("PF", 0) > best_atr["pf"]:
                best_atr = {"pf": m["PF"], "w": w, "t": thresh, "m": m, "trades": t, "yr": yr_m}
        print()

    # ════════════════════════════════════════════════════════════════════════
    #  Section 3: Best combo — trail × ATR gate
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("SECTION 3: Best combo — both trail values × best ATR gate")
    print("=" * 70)
    print(HDR)
    print("  " + "-" * 68)

    combo_results: dict[str, dict] = {}
    for ta in TRAIL_GRID:
        for w in ATR_WINDOW_GRID:
            for thresh in ATR_THRESH_GRID:
                t = build_trades(data, syms, btc_c4, btc_ma50_4h,
                                 atr_ratio_cache[w],
                                 btc_ma_gate=1, atr_gate=1, atr_thresh=thresh,
                                 trail_activate=ta)
                m = metrics(t)
                yr_m = yearly(t)
                ok = passes(m, yr_m)
                key = f"trail={ta:.3f}_w={w}_t={thresh:.1f}"
                combo_results[key] = m
                if ok:
                    print_row(f"{key}", m, "pass")

    # Overall best
    best_key = max(combo_results, key=lambda k: combo_results[k].get("PF", 0))
    best_m   = combo_results[best_key]
    print(f"\n  >>> Overall best: {best_key} → PF={best_m.get('PF', 0):.3f}")

    # Yearly for best combo
    parts = best_key.split("_")
    best_ta_final    = float(parts[0].split("=")[1])
    best_w_final     = int(parts[1].split("=")[1])
    best_t_final     = float(parts[2].split("=")[1])
    best_trades_final = build_trades(
        data, syms, btc_c4, btc_ma50_4h, atr_ratio_cache[best_w_final],
        btc_ma_gate=1, atr_gate=1, atr_thresh=best_t_final,
        trail_activate=best_ta_final,
    )
    yr_best = yearly(best_trades_final)

    print(f"\n  Yearly breakdown ({best_key}):")
    print(f"  {'Year':<6} {'N':>5}  {'PF':>6}  {'WR':>5}")
    print("  " + "-" * 28)
    all_pass = True
    for yr, yd in sorted(yr_best.items()):
        pf_s = f"{yd['PF']:.2f}" if yd["PF"] < 99 else "inf"
        flag = "✅" if yd["N"] < 20 or yd["PF"] >= 1.0 else "❌"
        if yd["N"] >= 20 and yd["PF"] < 1.0:
            all_pass = False
        print(f"  {yr:<6} {yd['N']:>5}  {pf_s:>6}  {yd['WR']:>5.0f}%  {flag}")

    # Bootstrap CI
    if best_trades_final:
        pnl_arr = portfolio_sim(best_trades_final, max_concurrent=cfg.max_concurrent)
        boot = bootstrap_pf(pnl_arr)
        print(f"\n  Bootstrap PF: median={boot['median']:.3f}  "
              f"95% CI=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]  "
              f"P(PF≥1)={boot['P_PF_ge_1']:.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    #  Decision
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    ref_pf = ref_m.get("PF", 0) if ref_m else 0
    ref_dd = ref_m.get("max_dd_pct", 99) if ref_m else 99
    best_pf_final = best_m.get("PF", 0)
    best_n_final  = best_m.get("N", 0)
    best_dd_final = best_m.get("max_dd_pct", 99)

    p_pf = best_pf_final >= R115_BEST_PF
    p_n  = best_n_final  >= R115_BEST_N * 0.80
    p_dd = best_dd_final <= R115_BEST_DD
    p_yr = all_pass

    print(f"  PF≥{R115_BEST_PF}:       {'✅' if p_pf else '❌'} ({best_pf_final:.3f})")
    print(f"  N≥{int(R115_BEST_N*0.8)}: {'✅' if p_n  else '❌'} ({best_n_final})")
    print(f"  MaxDD≤{R115_BEST_DD}%:    {'✅' if p_dd else '❌'} ({best_dd_final:.1f}%)")
    print(f"  No loss year:  {'✅' if p_yr else '❌'}")

    if all([p_pf, p_n, p_dd, p_yr]):
        verdict = f"✅ DEPLOY — add ATR gate to config: atr_window={best_w_final}, atr_thresh={best_t_final}, trail_activate={best_ta_final}"
    else:
        verdict = f"❌ NO DEPLOY — ATR gate does not beat r115 best on all criteria"
    print(f"\n  → {verdict}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.0f}s")

    # ── Save report ──────────────────────────────────────────────────────────
    report_dir = ROOT / "docs" / "backtest_results" / "202604"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r119_btc_atr_vol_gate.md"

    lines = [
        "# r119: BTC ATR Volatility Gate for MA Cluster",
        "",
        f"Date: {today}, Commit: {commit}",
        f"Reference (r115 best): PF={R115_BEST_PF}, N={R115_BEST_N}, MaxDD={R115_BEST_DD}%",
        "",
        "## Section 1: trail_activate 0.010 vs 0.015",
        "",
        "| Variant | N | PF | WR% | MaxDD% | EV(bp) |",
        "|---------|---|----|----|--------|--------|",
    ]
    for ta in TRAIL_GRID:
        m = trail_results[ta]
        if m:
            lines.append(f"| trail={ta:.3f} + MA50 gate | {m['N']} | {m['PF']:.3f} | "
                         f"{m['WR']:.1f} | {m['max_dd_pct']:.1f} | {m.get('EV_bp',0):.1f} |")

    lines += [
        "",
        "## Section 2: ATR Gate Sweep",
        "",
        f"Using trail_activate={best_trail_sec1} + BTC MA50 gate",
        "",
        "| ATR window | Threshold | N | PF | WR% | MaxDD% | EV(bp) | Pass? |",
        "|------------|-----------|---|----|----|--------|--------|-------|",
    ]
    for w in ATR_WINDOW_GRID:
        for thresh in ATR_THRESH_GRID:
            key = f"trail={best_trail_sec1:.3f}_w={w}_t={thresh:.1f}"
            m = combo_results.get(key, {})
            if m:
                ok = "✅" if passes(m, yearly(build_trades(
                    data, syms, btc_c4, btc_ma50_4h, atr_ratio_cache[w],
                    1, 1, thresh, best_trail_sec1))) else "❌"
                lines.append(f"| {w} | {thresh:.1f}x | {m['N']} | {m['PF']:.3f} | "
                             f"{m['WR']:.1f} | {m['max_dd_pct']:.1f} | {m.get('EV_bp',0):.1f} | {ok} |")

    lines += [
        "",
        "## Section 3: Best Combo",
        "",
        f"Best: {best_key} → PF={best_pf_final:.3f}",
        "",
        "### Yearly",
        "",
        "| Year | N | PF | WR% |",
        "|------|---|----|----|",
    ]
    for yr, yd in sorted(yr_best.items()):
        pf_s = f"{yd['PF']:.2f}" if yd["PF"] < 99 else "inf"
        lines.append(f"| {yr} | {yd['N']} | {pf_s} | {yd['WR']:.0f}% |")

    lines += [
        "",
        f"## Verdict",
        "",
        f"{verdict}",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
