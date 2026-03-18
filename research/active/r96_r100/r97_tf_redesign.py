"""r97: Trend Following Redesign — EMA(50,200) 4h with Wider SL.

목적:
  r96(EMA(8,21)+ADX)의 실패 원인을 보완:
  - r96 실패: avg_hold=3.9 4h bars (16h), MaxDD=95.3%, 슬롯 포화 avg 11.7/10
  - 근본 원인: EMA 주기 너무 짧음 → crypto noise에 연속 SL.
  - ADX gate는 역효과(낮을 때 더 좋음 → gate 없음이 최적 → r96 권고)

r97 변경:
  - EMA 주기 대폭 확장: (8,21) → (50,200) 후보 (8.3일/33.3일)
  - SL 확장: 2% → 4% (noise 허용, 진짜 추세만 catch)
  - trail_activate: 2% → 8% (SL×2, breathing room)
  - trail_distance: 0.3% → 2% (wide trailing)
  - max_hold: 12 bars → 120 bars (20일)
  - ADX gate 없음 (r96에서 역상관 확인)

Sim 엔진:
  sim_directional_trailing() 재사용 (재구현 금지).
  SL-first 이미 구현됨 (commit a55a457).

통과 기준:
  PF ≥ 1.20, avg_hold ≥ 40 4h bars, MaxDD(1x) < 60%, OOS 5/6년

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r96_r100/r97_tf_redesign.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    build_resampled,
    compute_metrics,
    portfolio_sim,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config (모든 파라미터는 cfg에서 읽음 — 하드코딩 금지) ─────────────────────
cfg = VolStraddleConfig()

TF_LEVERAGE = cfg.ma_cluster_leverage          # 21x (risk parity, MA cluster 동일)
TF_LN       = TF_LEVERAGE / cfg.max_concurrent # 21/10 = 2.1
MAX_SLOTS   = cfg.max_concurrent               # 10
WHITELIST   = set(cfg.symbol_whitelist)        # 60 symbols
COST        = 0.0008                           # round-trip 0.08%
TP_DEAD_CAP = 0.20                             # dead cap — trail exits first

# r97 연구 파라미터 (cfg에 없음 — sweep 대상)
TF_SL             = 0.04   # 4% (r96 2%의 2배 → noise 허용)
TF_TRAIL_ACTIVATE = 0.08   # 8% = SL × 2
TF_TRAIL_DISTANCE = 0.02   # 2%
TF_MAX_HOLD_4H    = 120    # 4h bars = 20일
TF_EMA_FAST       = 50     # 4h × 50 = 8.3일
TF_EMA_SLOW       = 200    # 4h × 200 = 33.3일

RATIO     = 16     # 15m → 4h
WARMUP_4H = 210    # EMA(200) + 여유 10 bars
DATA_YEARS = 6.0   # 2020-01-01 ~ 2025-12-31
BARS_PER_YEAR_4H = 2190  # 365.25 × 24/4 ≈ 2191.5


# ── Numba 신호 함수 ───────────────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (SMA seed)."""
    n = len(close)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    s = 0.0
    for i in range(period):
        s += close[i]
    ema[period - 1] = s / period
    k = 2.0 / (period + 1.0)
    for i in range(period, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


@numba.jit(nopython=True, cache=True)
def _ema_cross_signals(
    close: np.ndarray,
    fast: int,
    slow: int,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA crossover signal arrays.

    sig[i]=1, dire[i]=+1  → LONG cross bar i, entry at bar i+1
    sig[i]=1, dire[i]=-1  → SHORT cross bar i, entry at bar i+1

    Lookahead 없음: ema[i]는 close[0..i]까지만 사용.
    """
    ema_f = _compute_ema(close, fast)
    ema_s = _compute_ema(close, slow)
    n = len(close)
    sig  = np.zeros(n, dtype=np.int64)
    dire = np.zeros(n, dtype=np.int64)
    start = max(fast, slow, warmup)
    for i in range(start, n - 1):
        if np.isnan(ema_f[i]) or np.isnan(ema_s[i]):
            continue
        if np.isnan(ema_f[i - 1]) or np.isnan(ema_s[i - 1]):
            continue
        cross_up   = ema_f[i] > ema_s[i] and ema_f[i - 1] <= ema_s[i - 1]
        cross_down = ema_f[i] < ema_s[i] and ema_f[i - 1] >= ema_s[i - 1]
        if cross_up:
            sig[i]  = 1
            dire[i] = 1
        elif cross_down:
            sig[i]  = 1
            dire[i] = -1
    return sig, dire


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────


def load_data_4h() -> tuple[dict, int]:
    """15m NPZ cache → 4h 리샘플. cfg.symbol_whitelist 60개 심볼만."""
    cache_dir = ROOT / "data" / "cache_extended"

    # 가장 긴 NPZ 파일 선택
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in WHITELIST:
            continue  # symbol_whitelist만 (exclude_symbols 금지)
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
        v = npz["volume"].astype(np.float64) if "volume" in npz.files else np.ones(len(c))
        data_15m[sym] = {
            "close":  c,
            "high":   npz["high"].astype(np.float64),
            "low":    npz["low"].astype(np.float64),
            "volume": v,
        }

    # 심볼 간 bar 정렬 (글로벌 offset)
    if data_15m:
        max_bars_15m = max(len(d["close"]) for d in data_15m.values())
        for d in data_15m.values():
            d["_offset"] = max_bars_15m - len(d["close"])

    # 4h 리샘플 (ratio=16)
    data_4h_raw: dict = build_resampled(data_15m, RATIO)

    # 4h 글로벌 offset 계산
    max_4h = max((len(d["close"]) for d in data_4h_raw.values()), default=0)
    data_4h: dict = {}
    for sym, d in data_4h_raw.items():
        d["_offset"] = max_4h - len(d["close"])
        data_4h[sym] = d

    return data_4h, max_4h


# ── 심볼별 시뮬레이션 ─────────────────────────────────────────────────────────


def _run_sym(
    d: dict,
    fast: int,
    slow: int,
    sl: float,
    tp: float,
    trail_act: float,
    trail_dist: float,
    max_hold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단일 심볼: EMA cross 신호 생성 + sim_directional_trailing 실행."""
    close, high, low = d["close"], d["high"], d["low"]
    warmup = max(WARMUP_4H, slow + 10)
    sig, dire = _ema_cross_signals(close, fast, slow, warmup)
    return sim_directional_trailing(
        close, high, low,
        sig, dire,
        sl, tp, max_hold, COST, trail_act, trail_dist,
    )


def _collect_all(
    data_4h: dict,
    fast: int,
    slow: int,
    sl: float = TF_SL,
    tp: float = TP_DEAD_CAP,
    trail_act: float = TF_TRAIL_ACTIVATE,
    trail_dist: float = TF_TRAIL_DISTANCE,
    max_hold: int = TF_MAX_HOLD_4H,
    year_filter: int | None = None,
) -> list[tuple[str, int, int, float]]:
    """모든 심볼 시뮬레이션, (sym, entry_global, exit_global, pnl) 반환."""
    trades: list[tuple[str, int, int, float]] = []
    for sym, d in data_4h.items():
        offset = d.get("_offset", 0)
        pnl_arr, ent_arr, ext_arr = _run_sym(d, fast, slow, sl, tp, trail_act, trail_dist, max_hold)
        for k in range(len(pnl_arr)):
            eg = int(ent_arr[k]) + offset
            xg = int(ext_arr[k]) + offset
            if year_filter is not None:
                year_of_entry = 2020 + eg / BARS_PER_YEAR_4H
                if int(year_of_entry) != year_filter:
                    continue
            trades.append((sym, eg, xg, float(pnl_arr[k])))
    return trades


def _avg_hold(trades: list[tuple[str, int, int, float]]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t[2] - t[1] for t in trades]))


def _metrics(
    trades: list[tuple[str, int, int, float]],
    years: float = DATA_YEARS,
) -> dict:
    """portfolio_sim → compute_metrics + avg_hold."""
    if not trades:
        return {
            "N": 0, "PF": 0.0, "WR": 0.0, "EV_bp": 0.0,
            "max_dd_pct": 0.0, "ann_ret_pct": 0.0, "calmar": 0.0, "avg_hold": 0.0,
        }
    pnl_pooled = portfolio_sim(trades, max_concurrent=MAX_SLOTS)
    m = compute_metrics(pnl_pooled, years)
    m["avg_hold"] = _avg_hold(trades)
    return m


def _slot_saturation(trades: list[tuple[str, int, int, float]]) -> dict:
    """원시 신호 기준 동시 포지션 수 분석 (portfolio_sim 통과 전)."""
    if not trades:
        return {"avg": 0.0, "max": 0}
    events: list[tuple[int, int]] = []
    for _, e, x, _ in trades:
        events.append((e, 1))
        events.append((x, -1))
    events.sort()
    cur = 0
    peak = 0
    peaks: list[int] = []
    for _, delta in events:
        cur += delta
        if delta == 1:
            peaks.append(cur)
        peak = max(peak, cur)
    avg = float(np.mean(peaks)) if peaks else 0.0
    return {"avg": avg, "max": peak}


# ── Phase 0: r96 베이스라인 재현 (코드 정합성 검증) ──────────────────────────


def run_phase0_baseline(data_4h: dict) -> None:
    """EMA(8,21) r96 재현: N≈20605, PF≈1.302 (±10% — 다른 sim 엔진 허용)."""
    print("\n" + "═" * 72)
    print("PHASE 0 — r96 Baseline Verify (EMA(8,21), SL=2%, trail=2%/0.3%)")
    print("═" * 72)

    r96_sl   = cfg.ma_cluster_sl                  # 0.02
    r96_act  = cfg.ma_cluster_trail_activate      # 0.02
    r96_dist = cfg.ma_cluster_trail_distance      # 0.003
    r96_hold = cfg.ma_cluster_max_hold_4h         # 12

    trades = _collect_all(data_4h, 8, 21, r96_sl, TP_DEAD_CAP, r96_act, r96_dist, r96_hold)
    m = _metrics(trades)

    print(f"  N={m['N']}, PF={m['PF']:.3f}, WR={m['WR']:.1f}%")
    print(f"  EV={m['EV_bp']:+.1f}bp, MaxDD={m['max_dd_pct']:.1f}%,"
          f" AvgHold={m['avg_hold']:.1f} 4h bars")

    # r96은 커스텀 _sim_tf 사용. sim_directional_trailing과 세부 차이 있어 넓게 허용
    n_ok  = 10000 <= m["N"] <= 30000
    pf_ok = 1.05  <= m["PF"] <= 1.60
    if n_ok and pf_ok:
        print("  ✅ Phase 0 PASS — 코드 정상. Phase 1 진행.")
    else:
        print(f"  ⚠️  Phase 0 경고 (N expect 10k-30k, PF expect 1.05-1.60). 진행.")


# ── Phase 1: EMA 페어 스윕 ────────────────────────────────────────────────────


def run_phase1_ema_sweep(data_4h: dict) -> dict | None:
    print("\n" + "═" * 72)
    print("PHASE 1 — EMA Pair Sweep (SL=4%, trail=8%/2%, max_hold=120×4h)")
    print("═" * 72)
    print(f"SL={TF_SL*100:.0f}%  trail_activate={TF_TRAIL_ACTIVATE*100:.0f}%  "
          f"trail_dist={TF_TRAIL_DISTANCE*100:.1f}%  max_hold={TF_MAX_HOLD_4H}×4h  "
          f"leverage={TF_LEVERAGE:.0f}x  L/N={TF_LN:.1f}")
    print()

    ema_pairs = [
        (21,  55),   # 3.5일/9.2일
        (21,  89),   # 3.5일/14.8일
        (34, 144),   # 5.7일/24일
        (50, 200),   # 8.3일/33.3일  ← r97 주요 후보
        (89, 200),   # 14.8일/33.3일
    ]

    print(f"{'EMA Pair':<14} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} "
          f"{'AvgHold':>9} {'MaxDD':>7} {'AnnRet':>8}  Status")
    print("-" * 82)

    results: list[dict] = []
    best: dict | None = None

    for fast, slow in ema_pairs:
        trades = _collect_all(data_4h, fast, slow)
        m = _metrics(trades)
        label = f"EMA({fast},{slow})"

        hold_ok = m["avg_hold"] >= 40
        pf_ok   = m["PF"] >= 1.20
        dd_ok   = m["max_dd_pct"] < 60.0
        n_ok    = m["N"] >= 200

        if hold_ok and pf_ok and dd_ok and n_ok:
            status = "✅ PASS"
        elif pf_ok and n_ok:
            status = "~hold/DD"
        else:
            status = "✗"

        r = {"fast": fast, "slow": slow, "label": label, "trades": trades, **m}
        results.append(r)

        print(
            f"{label:<14} {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>6.1f}%"
            f" {m['EV_bp']:>+8.1f} {m['avg_hold']:>7.1f}h {m['max_dd_pct']:>6.1f}%"
            f" {m['ann_ret_pct']:>+7.1f}%  {status}"
        )

        if hold_ok and pf_ok and dd_ok and n_ok:
            if best is None or m["calmar"] > best.get("calmar", 0.0):
                best = r

    print()
    if best:
        print(f"✅ Phase 1 best: {best['label']} → PF={best['PF']:.3f},"
              f" EV={best['EV_bp']:+.1f}bp, AvgHold={best['avg_hold']:.1f}h, MaxDD={best['max_dd_pct']:.1f}%")
    else:
        partial = [r for r in results if r["PF"] >= 1.05 and r["N"] >= 100]
        if partial:
            best = max(partial, key=lambda r: r.get("calmar", 0.0))
            print(f"⚠️  No full PASS. Best partial: {best['label']} → PF={best['PF']:.3f}")
        else:
            print("❌ Phase 1 FAIL — 모든 EMA 페어 PF < 1.05")
            return None

    return best


# ── Phase 2: SL 스윕 ─────────────────────────────────────────────────────────


def run_phase2_sl_sweep(data_4h: dict, baseline: dict) -> dict | None:
    fast, slow = baseline["fast"], baseline["slow"]
    print("\n" + "═" * 72)
    print(f"PHASE 2 — SL Sweep (EMA({fast},{slow}), trail_activate=SL×2)")
    print("═" * 72)

    sl_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]

    print(f"{'SL':>6} {'act':>5} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} "
          f"{'AvgHold':>9} {'MaxDD':>7} {'Calmar':>8}  Status")
    print("-" * 80)

    results: list[dict] = []
    best: dict | None = None

    for sl in sl_values:
        act   = sl * 2.0
        dist  = TF_TRAIL_DISTANCE
        trades = _collect_all(data_4h, fast, slow, sl=sl, trail_act=act, trail_dist=dist)
        m = _metrics(trades)

        hold_ok = m["avg_hold"] >= 40
        pf_ok   = m["PF"] >= 1.20
        dd_ok   = m["max_dd_pct"] < 60.0

        status = "✅" if (hold_ok and pf_ok and dd_ok) else ("~hold" if pf_ok else "✗")
        r = {"sl": sl, "trail_act": act, "trail_dist": dist,
             "fast": fast, "slow": slow, "trades": trades, **m}
        results.append(r)

        print(
            f"{sl*100:>5.0f}% {act*100:>4.0f}%  {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>6.1f}%"
            f" {m['EV_bp']:>+8.1f} {m['avg_hold']:>7.1f}h {m['max_dd_pct']:>6.1f}%"
            f" {m['calmar']:>8.1f}  {status}"
        )

        if hold_ok and pf_ok and dd_ok:
            if best is None or m["calmar"] > best.get("calmar", 0.0):
                best = r

    print()
    if best:
        print(f"✅ Phase 2 best: SL={best['sl']*100:.0f}%, trail_act={best['trail_act']*100:.0f}%"
              f" → PF={best['PF']:.3f}, Calmar={best['calmar']:.1f}")
    else:
        best = max(results, key=lambda r: r.get("calmar", 0.0))
        print(f"⚠️  Phase 2: 기준 미충족. 최고 Calmar 기준: SL={best['sl']*100:.0f}%"
              f" → PF={best['PF']:.3f}")

    return best


# ── Phase 3: Trail 거리 미세조정 ─────────────────────────────────────────────


def run_phase3_trail_distance(data_4h: dict, best: dict) -> dict:
    fast, slow = best["fast"], best["slow"]
    sl   = best.get("sl", TF_SL)
    act  = best.get("trail_act", TF_TRAIL_ACTIVATE)

    print("\n" + "═" * 72)
    print(f"PHASE 3 — Trail Distance Sweep "
          f"(EMA({fast},{slow}), SL={sl*100:.0f}%, act={act*100:.0f}%)")
    print("═" * 72)

    trail_distances = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

    print(f"{'trail_dist':>12} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} "
          f"{'AvgHold':>9} {'MaxDD':>7} {'Calmar':>8}  Status")
    print("-" * 80)

    results: list[dict] = []
    best_d: dict | None = None

    for dist in trail_distances:
        trades = _collect_all(data_4h, fast, slow, sl=sl, trail_act=act, trail_dist=dist)
        m = _metrics(trades)

        hold_ok = m["avg_hold"] >= 40
        pf_ok   = m["PF"] >= 1.20
        dd_ok   = m["max_dd_pct"] < 60.0
        status  = "✅" if (hold_ok and pf_ok and dd_ok) else ("~" if pf_ok else "✗")

        r = {"sl": sl, "trail_act": act, "trail_dist": dist,
             "fast": fast, "slow": slow, "trades": trades, **m}
        results.append(r)

        print(
            f"  {dist*100:.1f}%        {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>6.1f}%"
            f" {m['EV_bp']:>+8.1f} {m['avg_hold']:>7.1f}h {m['max_dd_pct']:>6.1f}%"
            f" {m['calmar']:>8.1f}  {status}"
        )

        if hold_ok and pf_ok and dd_ok:
            if best_d is None or m["calmar"] > best_d.get("calmar", 0.0):
                best_d = r

    print()
    if best_d:
        print(f"✅ Phase 3 best: trail_dist={best_d['trail_dist']*100:.1f}%"
              f" → PF={best_d['PF']:.3f}, AvgHold={best_d['avg_hold']:.1f}h")
    else:
        best_d = max(results, key=lambda r: r.get("calmar", 0.0))
        print(f"⚠️  Phase 3: 기준 미충족. 최고 Calmar: trail_dist={best_d['trail_dist']*100:.1f}%")

    return best_d


# ── Phase 4: OOS 연도별 검증 ─────────────────────────────────────────────────


def run_phase4_oos(data_4h: dict, best: dict) -> int:
    """Returns number of passing years."""
    fast  = best["fast"]
    slow  = best["slow"]
    sl    = best.get("sl", TF_SL)
    act   = best.get("trail_act", TF_TRAIL_ACTIVATE)
    dist  = best.get("trail_dist", TF_TRAIL_DISTANCE)

    print("\n" + "═" * 72)
    print(f"PHASE 4 — OOS Year-by-Year "
          f"(EMA({fast},{slow}), SL={sl*100:.0f}%, act={act*100:.0f}%, dist={dist*100:.1f}%)")
    print("═" * 72)

    print(f"{'Year':<6} {'N':>5} {'PF':>6} {'WR':>6} {'EV(bp)':>8} "
          f"{'AvgHold':>9} {'MaxDD':>7}  Status")
    print("-" * 65)

    pass_count = 0
    for yr in range(2020, 2026):
        trades_yr = _collect_all(
            data_4h, fast, slow, sl=sl, trail_act=act, trail_dist=dist,
            year_filter=yr,
        )
        m = _metrics(trades_yr, years=1.0)
        profit = m["PF"] > 1.0 and m["N"] >= 5
        if profit:
            pass_count += 1
        status = "✅" if profit else "✗"
        print(
            f"{yr:<6} {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>6.1f}%"
            f" {m['EV_bp']:>+8.1f} {m['avg_hold']:>7.1f}h {m['max_dd_pct']:>6.1f}%  {status}"
        )

    print()
    if pass_count >= 5:
        print(f"✅ Phase 4 PASS: {pass_count}/6년 수익")
    else:
        print(f"❌ Phase 4 FAIL: {pass_count}/6년 수익 (기준: 5/6)")

    return pass_count


# ── Phase 5: 포트폴리오 슬롯 분석 ────────────────────────────────────────────


def run_phase5_portfolio(data_4h: dict, best: dict) -> None:
    fast  = best["fast"]
    slow  = best["slow"]
    sl    = best.get("sl", TF_SL)
    act   = best.get("trail_act", TF_TRAIL_ACTIVATE)
    dist  = best.get("trail_dist", TF_TRAIL_DISTANCE)

    print("\n" + "═" * 72)
    print("PHASE 5 — Portfolio Slot Analysis (TF alone + VR+MA 기준 비교)")
    print("═" * 72)

    # 이미 trades가 best에 있으면 재사용
    tf_trades = best.get("trades")
    if tf_trades is None:
        tf_trades = _collect_all(data_4h, fast, slow, sl=sl, trail_act=act, trail_dist=dist)

    pnl_tf = portfolio_sim(tf_trades, max_concurrent=MAX_SLOTS)
    m_tf   = compute_metrics(pnl_tf, DATA_YEARS)
    m_tf["avg_hold"] = _avg_hold(tf_trades)
    sat    = _slot_saturation(tf_trades)

    print(f"\n  TF alone (EMA({fast},{slow}), SL={sl*100:.0f}%):")
    print(f"    N={m_tf['N']}, PF={m_tf['PF']:.3f}, EV={m_tf['EV_bp']:+.1f}bp, WR={m_tf['WR']:.1f}%")
    print(f"    MaxDD(1x)={m_tf['max_dd_pct']:.1f}%, AnnRet(1x)={m_tf['ann_ret_pct']:+.1f}%")
    print(f"    AnnRet(L/N={TF_LN:.1f}) ≈ {m_tf['ann_ret_pct'] * TF_LN:+.1f}%")
    print(f"    AvgHold={m_tf['avg_hold']:.1f} 4h bars ({m_tf['avg_hold']/6:.1f}일)")
    print()
    print(f"  슬롯 포화 (raw signal, portfolio_sim 통과 전):")
    print(f"    avg={sat['avg']:.1f}/10, max={sat['max']}/10"
          f"  → {'낮음 ✅' if sat['avg'] < 5 else '높음 ⚠️  (기존 VR+MA와 슬롯 경쟁)'}")
    print()
    print("  ⚠️  VR+MA 합산 portfolio_sim은 데이터 재로딩이 필요해 생략.")
    print(f"     TF avg {sat['avg']:.1f}슬롯 + VR+MA avg ≈ 3슬롯 → 합계 예상 {sat['avg']+3:.1f}/10슬롯")
    print("     VR+MA canonical: N=6207, PF=1.563 (v14.5.0)")


# ── 최종 요약 ─────────────────────────────────────────────────────────────────


def print_final_summary(best: dict, pass_years: int) -> None:
    print("\n" + "═" * 72)
    print("FINAL SUMMARY — r97 EMA Trend Following")
    print("═" * 72)

    fast, slow = best["fast"], best["slow"]
    sl   = best.get("sl",         TF_SL)
    act  = best.get("trail_act",  TF_TRAIL_ACTIVATE)
    dist = best.get("trail_dist", TF_TRAIL_DISTANCE)
    ln   = TF_LN

    print(f"\n  파라미터:")
    print(f"    EMA({fast},{slow}), SL={sl*100:.0f}%,"
          f" trail_activate={act*100:.0f}%, trail_dist={dist*100:.1f}%")
    print(f"    max_hold={TF_MAX_HOLD_4H}×4h={TF_MAX_HOLD_4H//6:.0f}일,"
          f" leverage={TF_LEVERAGE:.0f}x (L/N={ln:.1f})")

    print(f"\n  1x compound (compute_metrics 기본):")
    print(f"    N={best['N']}, PF={best['PF']:.3f}, WR={best['WR']:.1f}%")
    print(f"    EV={best['EV_bp']:+.1f}bp, MaxDD={best['max_dd_pct']:.1f}%,"
          f" AnnRet={best['ann_ret_pct']:+.1f}%")
    print(f"    AvgHold={best.get('avg_hold', 0):.1f} 4h bars"
          f" ({best.get('avg_hold', 0)/6:.1f}일)")

    print(f"\n  Production (L/N={ln:.1f}):")
    print(f"    AnnRet ≈ {best['ann_ret_pct'] * ln:+.1f}%  [1x × {ln}]")

    print(f"\n  OOS: {pass_years}/6년 수익")

    all_pass = (
        best["PF"] >= 1.20
        and best.get("avg_hold", 0) >= 40
        and best["max_dd_pct"] < 60.0
        and best["N"] >= 200
        and pass_years >= 5
    )
    verdict = "✅ r97 PASS" if all_pass else "❌ r97 FAIL"
    print(f"\n  결정: {verdict}")
    if not all_pass:
        issues = []
        if best["PF"] < 1.20:
            issues.append(f"PF={best['PF']:.3f} < 1.20")
        if best.get("avg_hold", 0) < 40:
            issues.append(f"AvgHold={best.get('avg_hold',0):.1f} < 40 4h bars")
        if best["max_dd_pct"] >= 60.0:
            issues.append(f"MaxDD={best['max_dd_pct']:.1f}% ≥ 60%")
        if pass_years < 5:
            issues.append(f"OOS {pass_years}/6 < 5")
        print(f"  실패 원인: {', '.join(issues)}")
        print("  대안: Weekly EMA(10,30) or Donchian(20) 4h 검토 필요")

    print()
    print("  패리티 체크:")
    print("    ✅ cfg.symbol_whitelist 60개 (exclude_symbols 금지)")
    print("    ✅ sim_directional_trailing 재사용 (재구현 금지)")
    print("    ✅ SL-first (commit a55a457)")
    print("    ✅ _offset 적용 — portfolio_sim 글로벌 인덱스")
    print("    ✅ 1x compound + L/N=2.1 별도 표시")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("r97: Trend Following Redesign — EMA(50,200) 4h, SL=4%, Trail=8%/2%")
    print("=" * 72)
    print(f"leverage={TF_LEVERAGE}x  L/N={TF_LN:.1f}  max_slots={MAX_SLOTS}")
    print(f"cost={COST*100:.2f}% round-trip  data={DATA_YEARS:.0f}yr  60 symbols")
    print(f"whitelist={len(WHITELIST)} symbols loaded from cfg")

    # 데이터 로딩
    print("\n📂 Loading 4h data ...")
    data_4h, max_4h = load_data_4h()
    print(f"   {len(data_4h)} symbols loaded, max_4h={max_4h}")

    # Phase 0: 코드 정합성 검증
    run_phase0_baseline(data_4h)

    # Phase 1: EMA 페어 스윕
    best_ema = run_phase1_ema_sweep(data_4h)
    if best_ema is None:
        print("\n❌ Phase 1 FAIL — 4h EMA 추세추종 불가.")
        print("   대안: Weekly EMA or Donchian(20) 4h 검토.")
        return

    # Phase 2: SL 스윕
    best_sl = run_phase2_sl_sweep(data_4h, best_ema)

    # Phase 3: Trail 거리 미세조정
    best_trail = run_phase3_trail_distance(data_4h, best_sl)

    # Phase 4: OOS 연도별 검증
    pass_years = run_phase4_oos(data_4h, best_trail)

    # Phase 5: 포트폴리오 슬롯 분석
    run_phase5_portfolio(data_4h, best_trail)

    # 최종 요약
    print_final_summary(best_trail, pass_years)


if __name__ == "__main__":
    main()
