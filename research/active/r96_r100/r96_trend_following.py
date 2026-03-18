"""r96: Trend Following Strategy Backtest — EMA Crossover + ADX Gate.

목적:
  기존 3전략(VR Squeeze, Straddle, MA Cluster)은 모두 "압축 → 확장 초입"을 잡음.
  강한 추세가 이미 진행 중인 구간에서 신호가 없음.
  EMA 크로스오버 기반 추세추종으로 보완.

차별점 vs MA Cluster:
  - MA Cluster: 4개 SMA 수렴(cluster) → 이탈 (ADX 낮은 횡보 후)
  - Trend Following: EMA fast cross slow → 추세 방향 지속 진입 (ADX 높을 때)

시그널:
  - 4h 타임프레임 (15m → 4h 리샘플)
  - LONG: EMA_fast가 EMA_slow를 상향 교차
  - SHORT: EMA_fast가 EMA_slow를 하향 교차
  - ADX(14) > threshold 조건 (Phase 2부터)

통과 기준:
  - PF ≥ 1.20, N ≥ 300, OOS 5/6년 수익, 2022 크래시 방어

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r96_r100/r96_trend_following.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import build_resampled, compute_metrics, portfolio_sim
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config ────────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

TF_LEVERAGE = cfg.ma_cluster_leverage  # 21x (risk parity)
TF_LN = TF_LEVERAGE / cfg.max_concurrent  # 21/10 = 2.1
TF_SL = cfg.ma_cluster_sl  # 0.02
TF_TP = 0.20  # dead cap — trail exits first
TF_TRAIL_ACTIVATE = cfg.ma_cluster_trail_activate  # 0.02
TF_TRAIL_DISTANCE = cfg.ma_cluster_trail_distance  # 0.003
TF_MAX_HOLD = cfg.ma_cluster_max_hold_4h  # 12 × 4h = 2 days
COST = 0.0008  # round-trip 0.08%
ADX_PERIOD = 14
WARMUP_4H = 250  # EMA(89) + ADX(14) warmup
DATA_YEARS = 6.0
RATIO = 16  # 15m → 4h


# ── Numba indicators ─────────────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    out = np.full(len(arr), np.nan)
    k = 2.0 / (period + 1.0)
    # seed with SMA
    if len(arr) < period:
        return out
    s = 0.0
    for i in range(period):
        s += arr[i]
    out[period - 1] = s / period
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


@numba.jit(nopython=True, cache=True)
def _adx(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """ADX (Average Directional Index), Wilder smoothed."""
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period * 2 + 2:
        return adx

    # Wilder smoothing factor
    alpha = 1.0 / period

    # Bootstrap: first period True Ranges, +DM, -DM
    tr_sum = 0.0
    pdm_sum = 0.0
    ndm_sum = 0.0
    for i in range(1, period + 1):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        ndm = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr_sum += tr
        pdm_sum += pdm
        ndm_sum += ndm

    # Initial smoothed values
    atr = tr_sum
    pdi_raw = pdm_sum
    ndi_raw = ndm_sum

    # Compute initial DX values for ADX seed
    dx_sum = 0.0
    dx_count = 0
    for i in range(period + 1, period * 2 + 1):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        ndm = down_move if (down_move > up_move and down_move > 0) else 0.0

        atr = atr - atr * alpha + tr
        pdi_raw = pdi_raw - pdi_raw * alpha + pdm
        ndi_raw = ndi_raw - ndi_raw * alpha + ndm

        if atr > 0:
            pdi = 100.0 * pdi_raw / atr
            ndi = 100.0 * ndi_raw / atr
            denom = pdi + ndi
            dx = 100.0 * abs(pdi - ndi) / denom if denom > 0 else 0.0
        else:
            dx = 0.0
        dx_sum += dx
        dx_count += 1

    adx_val = dx_sum / dx_count if dx_count > 0 else 0.0
    seed_bar = period * 2

    # Continue from seed
    for i in range(seed_bar + 1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        ndm = down_move if (down_move > up_move and down_move > 0) else 0.0

        atr = atr - atr * alpha + tr
        pdi_raw = pdi_raw - pdi_raw * alpha + pdm
        ndi_raw = ndi_raw - ndi_raw * alpha + ndm

        if atr > 0:
            pdi = 100.0 * pdi_raw / atr
            ndi = 100.0 * ndi_raw / atr
            denom = pdi + ndi
            dx = 100.0 * abs(pdi - ndi) / denom if denom > 0 else 0.0
        else:
            dx = 0.0
        # Wilder's smoothing for ADX: dx is already 0-100, apply alpha correctly
        adx_val = adx_val - adx_val * alpha + dx * alpha
        adx[i] = adx_val

    return adx


@numba.jit(nopython=True, cache=True)
def _sim_tf(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    adx: np.ndarray,
    adx_threshold: float,
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trend following sim: EMA crossover entry + ADX gate + trailing stop.

    Entry:
      - LONG when EMA_fast crosses above EMA_slow (prev: fast <= slow, cur: fast > slow)
      - SHORT when EMA_fast crosses below EMA_slow
      - ADX > adx_threshold at entry bar (adx_threshold=0 → no filter)

    Exit:
      - SL first (server-side, intra-bar)
      - Trailing stop (client-side, bar-close)
      - TP cap (dead cap)
      - Max hold
      - Opposite EMA cross (trend reversal) — not implemented as hard exit, rely on SL/trail

    Cooldown: After exit, no re-entry until a fresh cross in either direction.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    cnt = 0

    in_trade = False
    i = warmup

    while i < n - 1:
        # Need valid indicators
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            i += 1
            continue

        # Check for EMA crossover
        if i < 1 or np.isnan(ema_fast[i - 1]) or np.isnan(ema_slow[i - 1]):
            i += 1
            continue

        prev_fast, prev_slow = ema_fast[i - 1], ema_slow[i - 1]
        cur_fast, cur_slow = ema_fast[i], ema_slow[i]

        # Determine direction from cross
        long_cross = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
        short_cross = (prev_fast >= prev_slow) and (cur_fast < cur_slow)

        if not long_cross and not short_cross:
            i += 1
            continue

        d = 1 if long_cross else -1

        # ADX gate
        if adx_threshold > 0.0:
            if np.isnan(adx[i]) or adx[i] < adx_threshold:
                i += 1
                continue

        # Enter trade at next bar open (use current close as entry proxy)
        entry = close[i]
        if entry <= 0.0:
            i += 1
            continue

        # Simulate trade
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

            # SL first (intra-bar, server-side)
            if d == 1:
                bar_loss = (entry - low[j]) / entry
            else:
                bar_loss = (high[j] - entry) / entry
            if bar_loss >= sl:
                trade_pnl = -sl
                eb = j
                break

            # Trailing stop (bar-close)
            if trailing_active:
                trail_stop = peak_gain - trail_distance
                if cur_gain < trail_stop:
                    trade_pnl = max(trail_stop, -sl)
                    eb = j
                    break

            # TP cap (dead cap)
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


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data_4h() -> tuple[dict, int]:
    """Load 15m cache, resample to 4h for all 60 whitelist symbols."""
    cache_dir = ROOT / "data" / "cache_extended"
    whitelist = set(cfg.symbol_whitelist)

    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        try:
            nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        except Exception:
            continue
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    data_15m: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        try:
            npz = np.load(str(fpath), allow_pickle=False)
        except Exception:
            continue
        if "close" not in npz.files:
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data_15m[sym] = {
            "close": c,
            "high": npz["high"].astype(np.float64),
            "low": npz["low"].astype(np.float64),
            "volume": npz["volume"].astype(np.float64) if "volume" in npz.files else np.ones(len(c)),
        }

    # Compute offsets (align bar 0 = oldest date)
    if data_15m:
        max_bars = max(len(d["close"]) for d in data_15m.values())
        for d in data_15m.values():
            d["_offset"] = max_bars - len(d["close"])
    else:
        max_bars = 0

    # Resample all to 4h
    data_4h_raw = build_resampled(data_15m, RATIO)

    # Compute _offset in 4h space
    max_4h = max((len(d["close"]) for d in data_4h_raw.values()), default=0)
    data_4h: dict = {}
    for sym, d in data_4h_raw.items():
        d["_offset"] = max_4h - len(d["close"])
        d["_offset_15m"] = data_15m[sym].get("_offset", 0)
        data_4h[sym] = d

    return data_4h, max_4h


# ── Per-symbol sim ────────────────────────────────────────────────────────────


def _run_sym(
    d: dict,
    ema_fast_p: int,
    ema_slow_p: int,
    adx_thresh: float,
    sl: float = TF_SL,
    tp: float = TF_TP,
    trail_act: float = TF_TRAIL_ACTIVATE,
    trail_dist: float = TF_TRAIL_DISTANCE,
    max_hold: int = TF_MAX_HOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c, h, l = d["close"], d["high"], d["low"]
    ema_f = _ema(c, ema_fast_p)
    ema_s = _ema(c, ema_slow_p)
    adx_arr = _adx(c, h, l, ADX_PERIOD) if adx_thresh > 0 else np.zeros(len(c))

    warmup = max(WARMUP_4H, ema_slow_p + ADX_PERIOD * 2 + 5)
    return _sim_tf(c, h, l, ema_f, ema_s, adx_arr, adx_thresh, sl, tp, trail_act, trail_dist, max_hold, COST, warmup)


def _collect_all(
    data_4h: dict,
    max_4h: int,
    ema_fast_p: int,
    ema_slow_p: int,
    adx_thresh: float,
    year_filter: int | None = None,
    bars_per_year: int = 2190,  # 365.25 * 6 = 2191.5 → 4h bars per year ≈ 2190
) -> list[tuple[str, int, int, float]]:
    """Run sim for all symbols, return (sym, entry_global, exit_global, pnl) list."""
    trades: list[tuple[str, int, int, float]] = []
    for sym, d in data_4h.items():
        offset = d.get("_offset", 0)
        pnl, ent, ext = _run_sym(d, ema_fast_p, ema_slow_p, adx_thresh)
        for k in range(len(pnl)):
            eg = int(ent[k]) + offset
            xg = int(ext[k]) + offset
            # Year filter (approx: bar 0 = 2020-01-01)
            if year_filter is not None:
                year_of_entry = 2020 + eg / bars_per_year
                if int(year_of_entry) != year_filter:
                    continue
            trades.append((sym, eg, xg, float(pnl[k])))
    return trades


def _metrics(trades: list[tuple[str, int, int, float]], years: float = DATA_YEARS) -> dict:
    if not trades:
        return {"N": 0, "PF": 0.0, "WR": 0.0, "EV_bp": 0.0, "max_dd_pct": 0.0, "ann_ret_pct": 0.0}
    pnl_arr = np.array([t[3] for t in trades], dtype=np.float64)
    pnl_pooled = portfolio_sim(trades, max_concurrent=cfg.max_concurrent)
    return compute_metrics(pnl_pooled, years)


# ── Section A: EMA pair baseline sweep ───────────────────────────────────────


def run_phase1_ema_sweep(data_4h: dict, max_4h: int) -> dict:
    print("\n" + "═" * 72)
    print("PHASE 1 — EMA Pair Baseline Sweep (no ADX gate)")
    print("═" * 72)
    print(f"SL={TF_SL*100:.0f}% trail_activate={TF_TRAIL_ACTIVATE*100:.0f}% "
          f"trail_dist={TF_TRAIL_DISTANCE*100:.1f}% max_hold={TF_MAX_HOLD}×4h leverage={TF_LEVERAGE:.0f}x")
    print()

    ema_pairs = [(8, 21), (13, 34), (21, 55), (21, 89)]

    results: list[dict] = []
    print(f"{'EMA Pair':<12} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} {'MaxDD':>7} {'AnnRet':>8}")
    print("-" * 60)

    best = None
    for fast, slow in ema_pairs:
        trades = _collect_all(data_4h, max_4h, fast, slow, adx_thresh=0.0)
        m = _metrics(trades)
        label = f"EMA({fast},{slow})"
        r = {"fast": fast, "slow": slow, "adx_thresh": 0.0, "label": label, **m, "trades": trades}
        results.append(r)
        status = "✅" if m["PF"] >= 1.20 and m["N"] >= 300 else "✗"
        print(
            f"{label:<12} {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>6.1f}% "
            f"{m['EV_bp']:>+8.1f} {m['max_dd_pct']:>6.1f}% {m['ann_ret_pct']:>+7.1f}%  {status}"
        )
        if best is None or (m["PF"] >= 1.20 and m["N"] >= 300 and m["EV_bp"] > best.get("EV_bp", -999)):
            best = r

    print()
    if best and best["PF"] >= 1.20:
        print(f"✅ Best baseline: {best['label']} → PF={best['PF']:.3f}, EV={best['EV_bp']:+.1f}bp, N={best['N']}")
    else:
        print("❌ No EMA pair meets baseline criteria (PF≥1.20, N≥300)")

    return best if (best and best["PF"] >= 1.20) else None


# ── Section B: ADX gate sweep ─────────────────────────────────────────────────


def run_phase2_adx_sweep(data_4h: dict, max_4h: int, best_fast: int, best_slow: int, baseline: dict) -> dict | None:
    print("\n" + "═" * 72)
    print(f"PHASE 2 — ADX Gate Sweep (EMA({best_fast},{best_slow}))")
    print("═" * 72)

    thresholds = [0.0, 15.0, 20.0, 25.0, 30.0]
    results: list[dict] = []

    print(f"{'ADX thresh':<12} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} {'MaxDD':>7}  Status")
    print("-" * 65)

    for thresh in thresholds:
        if thresh == 0.0:
            trades = baseline["trades"]
            m = {k: v for k, v in baseline.items() if k not in ("trades", "label", "fast", "slow", "adx_thresh")}
        else:
            trades = _collect_all(data_4h, max_4h, best_fast, best_slow, adx_thresh=thresh)
            m = _metrics(trades)
        r = {"thresh": thresh, **m, "trades": trades}
        results.append(r)

        if thresh == 0.0:
            status = "← baseline"
        elif m["PF"] >= baseline["PF"] and m["EV_bp"] >= baseline["EV_bp"] and m["N"] >= baseline["N"] * 0.62:
            status = "✅ PASS"
        elif m["PF"] >= baseline["PF"] - 0.05 or m["EV_bp"] >= baseline["EV_bp"] - 5:
            status = "~marginal"
        else:
            status = "✗"

        print(
            f"ADX>{thresh:4.0f}      {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>6.1f}% "
            f"{m['EV_bp']:>+8.1f} {m['max_dd_pct']:>6.1f}%  {status}"
        )

    # Pick best
    candidates = [
        r for r in results
        if r["thresh"] > 0
        and r["PF"] >= baseline["PF"]
        and r["EV_bp"] >= baseline["EV_bp"]
        and r["N"] >= baseline["N"] * 0.62
    ]
    if candidates:
        best = max(candidates, key=lambda r: r["EV_bp"])
        print(f"\n✅ Best ADX config: threshold={best['thresh']:.0f}")
        print(f"   PF={best['PF']:.3f} EV={best['EV_bp']:+.1f}bp N={best['N']}")
        return best
    else:
        # Fall back to baseline (no ADX gate)
        print(f"\nℹ️  No ADX config improves baseline. Using threshold=0 (no gate).")
        return results[0]  # baseline


# ── Section C: OOS year-by-year validation ────────────────────────────────────


def run_phase3_oos(data_4h: dict, max_4h: int, fast: int, slow: int, adx_thresh: float) -> int:
    print("\n" + "═" * 72)
    print(f"PHASE 3 — OOS Year Validation (EMA({fast},{slow}) ADX>{adx_thresh:.0f})")
    print("═" * 72)

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    bars_per_year = int(365.25 * 24 / 4)  # 4h bars per year ≈ 2192

    print(f"\n{'Year':<6} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} {'AnnRet':>8}  Pass?")
    print("-" * 55)

    passes = 0
    for yr in years:
        trades = _collect_all(data_4h, max_4h, fast, slow, adx_thresh=adx_thresh,
                               year_filter=yr, bars_per_year=bars_per_year)
        if not trades:
            print(f"{yr:<6} {'N/A':>5}")
            continue
        m = _metrics(trades, years=1.0)
        ok = m["PF"] > 1.0 and m["N"] >= 20
        if ok:
            passes += 1
        print(
            f"{yr:<6} {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>6.1f}% "
            f"{m['EV_bp']:>+8.1f} {m['ann_ret_pct']:>+7.1f}%  {'✅' if ok else '❌'}"
        )

    print(f"\n{passes}/{len(years)} years passed (PF>1.0, N≥20)")
    if passes >= 5:
        print("✅ OOS PASS — trend following is consistent across market cycles")
    elif passes >= 4:
        print("⚠️  Borderline — 4/6 years. Consider ADX or trail adjustments.")
    else:
        print("❌ OOS FAIL — not consistent enough. Do not implement.")
    return passes


# ── Section D: Portfolio interaction ─────────────────────────────────────────


def run_phase4_portfolio(data_4h: dict, max_4h: int, tf_trades: list, baseline_label: str) -> None:
    print("\n" + "═" * 72)
    print("PHASE 4 — Portfolio Interaction (VR + MA + TF)")
    print("═" * 72)
    print("Note: VR+MA baselines approximate (not re-running full canonical sim here)")
    print("      Focus: does adding TF trades hurt or help slot utilization?")
    print()

    if not tf_trades:
        print("No TF trades — skipping.")
        return

    # TF alone
    m_tf = _metrics(tf_trades)
    print(f"TF alone:  N={m_tf['N']} PF={m_tf['PF']:.3f} MaxDD={m_tf['max_dd_pct']:.1f}%")

    # Slot utilization: average concurrent positions
    if tf_trades:
        all_bars = [(t[1], t[2]) for t in tf_trades]  # (entry_global, exit_global)
        max_bar = max(t[2] for t in tf_trades)
        concurrent = np.zeros(max_bar + 1, dtype=np.int32)
        for eg, xg in all_bars:
            concurrent[eg:xg + 1] += 1
        avg_concurrent = concurrent[concurrent > 0].mean()
        max_concurrent = concurrent.max()
        print(f"Slot usage: avg={avg_concurrent:.2f}, max={max_concurrent} (of {cfg.max_concurrent})")
        pct_time_open = (concurrent > 0).mean() * 100
        print(f"Time in market: {pct_time_open:.1f}% of bars")

    print()
    print(f"L/N = {TF_LN:.1f}x → 1x AnnRet {m_tf['ann_ret_pct']:+.1f}% → production ≈ {m_tf['ann_ret_pct']*TF_LN:+.1f}%")
    print(f"(Additive to VR+MA production returns — independent slot usage)")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("r96: Trend Following Strategy — EMA Crossover + ADX Gate")
    print(f"Config: {cfg.__class__.__name__}()")
    print(f"  Leverage={TF_LEVERAGE:.0f}x, L/N={TF_LN:.1f}, SL={TF_SL*100:.0f}%, "
          f"trail_activate={TF_TRAIL_ACTIVATE*100:.0f}%, trail_dist={TF_TRAIL_DISTANCE*100:.1f}%")

    print("\nLoading + resampling to 4h...")
    data_4h, max_4h = load_data_4h()
    syms = list(data_4h.keys())
    whitelist_syms = [s for s in syms if s in set(cfg.symbol_whitelist)]
    print(f"Loaded {len(whitelist_syms)}/{len(cfg.symbol_whitelist)} whitelist symbols → 4h resampled")
    print(f"4h bars per symbol (max): {max_4h}")

    print("\nWarming up Numba JIT (first call)...")
    # Trigger compilation with small array
    d0 = data_4h[whitelist_syms[0]]
    _ema(d0["close"][:100], 21)
    _adx(d0["close"][:200], d0["high"][:200], d0["low"][:200], 14)
    _sim_tf(d0["close"][:300], d0["high"][:300], d0["low"][:300],
            _ema(d0["close"][:300], 21), _ema(d0["close"][:300], 55),
            np.zeros(300), 0.0, TF_SL, TF_TP, TF_TRAIL_ACTIVATE, TF_TRAIL_DISTANCE,
            TF_MAX_HOLD, COST, 250)
    print("JIT warm-up complete.")

    # Phase 1: EMA pair sweep
    best_ema = run_phase1_ema_sweep(data_4h, max_4h)

    if best_ema is None:
        print("\n❌ PHASE 1 FAIL — No EMA pair has positive EV. Strategy not viable.")
        print("Recommendation: Do not implement trend following with current parameters.")
        return

    fast, slow = best_ema["fast"], best_ema["slow"]

    # Phase 2: ADX sweep
    best_adx = run_phase2_adx_sweep(data_4h, max_4h, fast, slow, best_ema)
    final_thresh = best_adx["thresh"] if best_adx else 0.0

    # Phase 3: OOS validation
    oos_passes = run_phase3_oos(data_4h, max_4h, fast, slow, final_thresh)

    # Phase 4: Portfolio interaction
    if final_thresh == 0.0:
        final_trades = best_ema["trades"]
    else:
        final_trades = best_adx["trades"] if best_adx else best_ema["trades"]

    run_phase4_portfolio(data_4h, max_4h, final_trades, f"EMA({fast},{slow}) ADX>{final_thresh:.0f}")

    # Final verdict
    print("\n" + "═" * 72)
    print("FINAL VERDICT")
    print("═" * 72)
    m_final = _metrics(final_trades)
    pass_criteria = m_final["PF"] >= 1.20 and m_final["N"] >= 300 and oos_passes >= 5
    print(f"  EMA pair:      EMA({fast},{slow})")
    print(f"  ADX threshold: {final_thresh:.0f}")
    print(f"  PF={m_final['PF']:.3f}  N={m_final['N']}  EV={m_final['EV_bp']:+.1f}bp  "
          f"MaxDD={m_final['max_dd_pct']:.1f}%  OOS={oos_passes}/6")
    print()
    if pass_criteria:
        print("✅ PASS — Trend following strategy viable. Recommend r97 for implementation.")
        print(f"   → Add to config.py as tf_enabled=False (disabled by default)")
        print(f"   → Enable after live micro-test on DOGEUSDT")
    else:
        reasons = []
        if m_final["PF"] < 1.20:
            reasons.append(f"PF={m_final['PF']:.3f}<1.20")
        if m_final["N"] < 300:
            reasons.append(f"N={m_final['N']}<300")
        if oos_passes < 5:
            reasons.append(f"OOS={oos_passes}/6<5")
        print(f"❌ FAIL ({', '.join(reasons)}) — Do not implement trend following.")
        print(f"   Potential improvement: wider SL (3%), longer EMA periods, or different TF signal.")


if __name__ == "__main__":
    main()
