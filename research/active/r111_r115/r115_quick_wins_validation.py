#!/usr/bin/env python3
"""r115: Non-ML Quick Wins Validation for MA Cluster.

Experiments:
  1. trail_activate sweep with BTC gate ON combo
     (r75: trail_activate=0.02 → +21% PF, r92: BTC gate → +7.6% PF — combo 미검증)
  2. CW conviction sizing heuristic for MA Cluster
     (현재 고정 21x leverage → squeeze quality 기반 0.5x-1.5x 가변)
  3. Combined: best trail + CW sizing

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r111_r115/r115_quick_wins_validation.py

(ref: r75 trail sweep, r92 BTC gate)
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    bootstrap_pf,
    compute_metrics,
    portfolio_sim,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config ──────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

MA_PARAMS = {
    "cluster_pct": cfg.ma_cluster_pct,
    "min_duration": cfg.ma_cluster_min_duration,
    "max_duration": cfg.ma_cluster_max_duration,
    "min_momentum": cfg.ma_cluster_min_momentum,
    "sl": cfg.ma_cluster_sl,
    "trail_activate": cfg.ma_cluster_trail_activate,
    "trail_distance": cfg.ma_cluster_trail_distance,
    "tp": 0.20,  # dead cap — trail exits first
    "max_hold": cfg.ma_cluster_max_hold_4h,
    "cost": 0.0008,
    "warmup": 200,
}
BTC_GATE_PERIOD = cfg.ma_cluster_btc_ma_period_4h  # 50 × 4h = 200h ≈ 8.3일
MA_LN = cfg.ma_cluster_leverage / cfg.max_concurrent  # 21/10 = 2.1
DATA_YEARS = 6.0
RATIO = 16  # 15m → 4h
COST_DIRECTIONAL = 0.0008

# trail_activate sweep 값 (Binance floor: trail_distance >= 0.005, Rule #8)
TRAIL_ACTIVATE_GRID = [0.015, 0.020, 0.025, 0.030, 0.040]
TRAIL_DISTANCE_FIXED = 0.005  # Binance callbackRate floor (Rule #8)

# CW conviction sizing 배수
CW_MULTIPLIERS = {0: 0.5, 1: 0.75, 2: 1.0, 3: 1.5}


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ── Data loading (r92 패턴) ─────────────────────────────────────────────────

def load_data(whitelist: list[str]) -> dict:
    """cache_extended에서 60 whitelist 심볼 + BTCUSDT 로드."""
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    all_syms = set(whitelist) | {"BTCUSDT"}  # BTC gate용 항상 로드
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


def build_resampled_arrays(
    c15: np.ndarray, h15: np.ndarray, l15: np.ndarray, ratio: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """15m → 4h OHLC 리샘플링."""
    n4 = len(c15) // ratio
    close = c15[ratio - 1 : n4 * ratio : ratio].copy()
    high = np.array([np.max(h15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    low = np.array([np.min(l15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    return close, high, low


def compute_sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.mean(arr[i - period + 1 : i + 1])
    return out


def momentum8(close: np.ndarray) -> np.ndarray:
    mom = np.full(len(close), np.nan)
    for i in range(8, len(close)):
        mom[i] = (close[i] - close[i - 8]) / close[i - 8]
    return mom


# ── Numba sim: r92 패턴 (BTC gate + configurable trail) ────────────────────

@numba.jit(nopython=True, cache=True)
def sim_live_style_gated(
    close, high, low, sma20, sma50, sma100, sma200, momentum,
    btc_close_4h, btc_ma50_4h, gate_enabled,
    cluster_pct, min_duration, max_duration, min_momentum,
    sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
):
    """Live-style MA cluster sim with optional 4h BTC gate.

    Gate: gate_enabled=1이고 signal=LONG이고 btc_close_4h < btc_ma50_4h이면 진입 차단.
    SL-first fix 적용: trail activation 전에 SL 먼저 확인.
    Cooldown: cluster spread > cluster_pct일 때만 해제 (live 방식).
    """
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    dir_arr = np.empty(cap, np.int64)   # +1 LONG, -1 SHORT
    # spread at entry bar (conviction scoring용)
    spread_arr = np.empty(cap, np.float64)
    # volume placeholder — numba에서 별도 배열 필요 시 추가
    mom_arr = np.empty(cap, np.float64)
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

        # 4h BTC gate (LONG만 차단, live gate 로직 매칭)
        if gate_enabled == 1 and d == 1 and i < len(btc_ma50_4h) and not np.isnan(btc_ma50_4h[i]):
            if btc_close_4h[i] < btc_ma50_4h[i]:
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
                # SL before trail activation (SL-first fix)
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
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl_arr[cnt] = trade_pnl - cost
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            dir_arr[cnt] = d
            spread_arr[cnt] = spread
            mom_arr[cnt] = abs(momentum[i])
            cnt += 1

        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], dir_arr[:cnt], spread_arr[:cnt], mom_arr[:cnt]


# ── Per-symbol runner ─────────────────────────────────────────────────────────

def run_symbol(
    sym: str, d15: dict,
    btc_4h: np.ndarray, btc_ma50: np.ndarray,
    p: dict, gate_enabled: int = 1,
    trail_activate_override: float | None = None,
    trail_distance_override: float | None = None,
) -> dict:
    """심볼별 sim 실행. trail/gate 오버라이드 지원."""
    c15, h15, l15 = d15["close"], d15["high"], d15["low"]
    c4, h4, l4 = build_resampled_arrays(c15, h15, l15, RATIO)
    sma20 = compute_sma(c4, 20)
    sma50 = compute_sma(c4, 50)
    sma100 = compute_sma(c4, 100)
    sma200 = compute_sma(c4, 200)
    mom = momentum8(c4)
    offset = d15.get("_offset", 0)

    # BTC 4h 정렬
    btc_offset_4h = offset // RATIO
    n4 = len(c4)
    if len(btc_4h) >= btc_offset_4h + n4:
        btc_close_aligned = btc_4h[btc_offset_4h : btc_offset_4h + n4]
        btc_ma50_aligned = btc_ma50[btc_offset_4h : btc_offset_4h + n4]
    else:
        btc_close_aligned = np.full(n4, np.nan)
        btc_ma50_aligned = np.full(n4, np.nan)

    ta = trail_activate_override if trail_activate_override is not None else p["trail_activate"]
    td = trail_distance_override if trail_distance_override is not None else p["trail_distance"]

    base_args = (c4, h4, l4, sma20, sma50, sma100, sma200, mom,
                 btc_close_aligned, btc_ma50_aligned)
    sim_args = (p["cluster_pct"], p["min_duration"], p["max_duration"], p["min_momentum"],
                p["sl"], p["tp"], ta, td, p["max_hold"], p["cost"], p["warmup"])

    pnl, ent, ext, dirs, spreads, moms = sim_live_style_gated(*base_args, gate_enabled, *sim_args)

    # 4h bar → 15m bar 변환
    ent15 = (ent * RATIO + offset).astype(np.int64)
    ext15 = ((ext + 1) * RATIO - 1 + offset).astype(np.int64)

    # Volume 데이터 (4h bar 기준) — conviction scoring용
    vol_4h = None
    if "volume" in d15:
        v15 = d15["volume"]
        n4v = len(v15) // RATIO
        vol_4h = np.array([np.sum(v15[i : i + RATIO]) for i in range(0, n4v * RATIO, RATIO)])

    return {
        "sym": sym,
        "pnl": pnl, "ent": ent15, "ext": ext15, "dirs": dirs,
        "spreads": spreads, "moms": moms,
        "ent_4h": ent, "vol_4h": vol_4h, "n4h": n4,
        "offset_4h": offset // RATIO,
    }


# ── Trade stacking + metrics ────────────────────────────────────────────────

def stack_trades(results: list[dict]) -> list[tuple[str, int, int, float]]:
    """결과를 portfolio_sim 호환 형식으로 스택."""
    trades = []
    for r in results:
        for p, e, x in zip(r["pnl"], r["ent"], r["ext"]):
            trades.append((r["sym"], int(e), int(x), float(p)))
    return trades


def get_portfolio_metrics(trades: list[tuple[str, int, int, float]]) -> dict:
    """portfolio_sim → compute_metrics 파이프라인."""
    if len(trades) < 10:
        return {}
    pnl = portfolio_sim(trades, max_concurrent=cfg.max_concurrent)
    return compute_metrics(pnl, DATA_YEARS)


# ── Yearly breakdown (r92 패턴) ─────────────────────────────────────────────

def yearly_metrics(trades: list[tuple[str, int, int, float]]) -> dict[int, dict]:
    """15m bar index 기준 연도별 분해."""
    years: dict[int, list[float]] = {}
    bars_per_year_15m = 35064  # 365.25 * 24 * 4
    for _sym, ent, _ext, pnl in trades:
        year = 2020 + int(ent) // bars_per_year_15m
        if year not in years:
            years[year] = []
        years[year].append(float(pnl))
    result_by_year: dict[int, dict] = {}
    for yr in sorted(years):
        pnls = np.array(years[yr])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf")
        wr = 100.0 * len(wins) / max(len(pnls), 1)
        ev = float(np.mean(pnls)) * 10000  # bp
        result_by_year[yr] = {"N": len(pnls), "PF": pf, "WR": wr, "EV": ev}
    return result_by_year


# ── CW Conviction Scoring ───────────────────────────────────────────────────

def apply_cw_sizing(results: list[dict]) -> list[tuple[str, int, int, float]]:
    """Conviction-weighted sizing으로 trade PnL 조정.

    Heuristic scoring (0-3점):
      - tight_spread_score: entry spread < 전체 entry spread 중앙값
      - volume_spike_score: entry bar volume > 1.5x 평균 volume
      - strong_momentum_score: |momentum| > 2x min_momentum

    Score → multiplier: {0: 0.5, 1: 0.75, 2: 1.0, 3: 1.5}
    PnL에 multiplier를 적용하여 sizing 효과 근사.
    """
    # 전체 entry spread 중앙값 계산
    all_spreads = []
    for r in results:
        all_spreads.extend(r["spreads"].tolist())
    if not all_spreads:
        return []
    median_spread = float(np.median(all_spreads))

    # 전체 entry momentum 평균 (min_momentum 기준)
    min_mom = cfg.ma_cluster_min_momentum

    trades = []
    for r in results:
        # 심볼별 평균 volume 계산
        avg_vol = None
        if r["vol_4h"] is not None and len(r["vol_4h"]) > 0:
            avg_vol = float(np.nanmean(r["vol_4h"]))

        for k in range(len(r["pnl"])):
            score = 0

            # 1) tight spread: entry spread < median
            if r["spreads"][k] < median_spread:
                score += 1

            # 2) volume spike: entry bar volume > 1.5x average
            if avg_vol is not None and avg_vol > 0 and r["vol_4h"] is not None:
                ent_bar_4h = int(r["ent_4h"][k])
                if ent_bar_4h < len(r["vol_4h"]):
                    entry_vol = r["vol_4h"][ent_bar_4h]
                    if entry_vol > 1.5 * avg_vol:
                        score += 1

            # 3) strong momentum: |momentum| > 2x min_momentum
            if r["moms"][k] > 2.0 * min_mom:
                score += 1

            mult = CW_MULTIPLIERS[score]
            # PnL scaling: cost 부분은 이미 포함되어 있으므로
            # (pnl + cost) * mult - cost 로 sizing 효과만 조정
            raw_pnl = float(r["pnl"][k])
            adjusted = (raw_pnl + COST_DIRECTIONAL) * mult - COST_DIRECTIONAL
            trades.append((r["sym"], int(r["ent"][k]), int(r["ext"][k]), adjusted))

    return trades


# ── Numba warmup ────────────────────────────────────────────────────────────

def warmup_numba() -> None:
    """JIT 컴파일 예열."""
    _c = np.ones(500)
    _nan = np.full(500, np.nan)
    _nan[200:] = 1.0
    _btc = np.ones(500) * 90000.0
    _bma = np.full(500, np.nan)
    _bma[50:] = 88000.0
    for gate in (0, 1):
        sim_live_style_gated(
            _c, _c, _c, _nan, _nan, _nan, _nan, _nan,
            _btc, _bma, gate,
            0.02, 6, 24, 0.01, 0.02, 0.20, 0.02, 0.005, 48, 0.0, 200,
        )


# ── Print helpers ───────────────────────────────────────────────────────────

def print_metrics_row(name: str, m: dict) -> None:
    """메트릭 한 줄 출력."""
    if not m:
        print(f"  {name}: no data")
        return
    ev_bp = m.get("EV_bp", float("nan"))
    cagr_ln = ((1 + m.get("ann_ret_pct", 0) / 100 * MA_LN) ** (1 / DATA_YEARS) - 1) * 100
    print(f"  {name:<28} {m['N']:>5}  {m['PF']:>6.3f}  {m['WR']:>5.1f}%  "
          f"{m['max_dd_pct']:>6.1f}%  {ev_bp:>7.1f}  {cagr_ln:>7.1f}%")


def print_yearly_table(yearly: dict[int, dict], label: str) -> None:
    """연도별 테이블 출력."""
    print(f"\n  [{label}]")
    print(f"  {'Year':<6} {'N':>5}  {'PF':>6}  {'WR':>5}  {'EV(bp)':>8}")
    print(f"  {'-'*36}")
    for yr in sorted(yearly):
        y = yearly[yr]
        pf_str = f"{y['PF']:.2f}" if y["PF"] < 100 else "inf"
        print(f"  {yr:<6} {y['N']:>5}  {pf_str:>6}  {y['WR']:>5.0f}%  {y['EV']:>8.1f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    commit = get_commit()
    today = datetime.now().strftime("%Y%m%d")

    print("r115: Non-ML Quick Wins Validation for MA Cluster")
    print("=" * 70)
    print(f"Config: cluster_pct={MA_PARAMS['cluster_pct']}  min_dur={MA_PARAMS['min_duration']}  "
          f"sl={MA_PARAMS['sl']}  trail_act={MA_PARAMS['trail_activate']}  "
          f"trail_dist={MA_PARAMS['trail_distance']}")
    print(f"BTC Gate: MA{BTC_GATE_PERIOD} (={BTC_GATE_PERIOD * 4}h = {BTC_GATE_PERIOD * 4 / 24:.1f}일)")
    print(f"MA_LN = {cfg.ma_cluster_leverage}/{cfg.max_concurrent} = {MA_LN:.1f}")
    print()

    # ── Data load ────────────────────────────────────────────────────────────
    data = load_data(cfg.symbol_whitelist)
    print(f"Loaded {len(data)} symbols from cache_extended")
    if not data:
        print("ERROR: cache_extended에 데이터 없음")
        return
    if "BTCUSDT" not in data:
        print("ERROR: BTCUSDT 미발견 — gate 계산 불가")
        return

    # BTC 4h 배열 생성
    btc = data["BTCUSDT"]
    btc_close_4h, _, _ = build_resampled_arrays(btc["close"], btc["high"], btc["low"], RATIO)
    btc_ma50_4h = compute_sma(btc_close_4h, BTC_GATE_PERIOD)
    print(f"BTC 4h bars: {len(btc_close_4h)}")

    # Numba warmup
    print("Numba JIT 예열... ", end="", flush=True)
    warmup_numba()
    print("done\n")

    syms = [s for s in cfg.symbol_whitelist if s in data and s != "BTCUSDT"]

    # ════════════════════════════════════════════════════════════════════════
    #  Experiment 1: Trail Activate Sweep + BTC Gate
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("EXPERIMENT 1: Trail Activate Sweep + BTC Gate ON")
    print("=" * 70)
    print(f"trail_activate grid: {TRAIL_ACTIVATE_GRID}")
    print(f"trail_distance fixed: {TRAIL_DISTANCE_FIXED} (Binance floor, Rule #8)")
    print()

    # 각 trail_activate 값에 대해 sim 실행 (gate ON)
    exp1_results: dict[float, dict] = {}
    exp1_trades: dict[float, list] = {}
    exp1_sym_results: dict[float, list[dict]] = {}

    for ta in TRAIL_ACTIVATE_GRID:
        results = []
        for sym in syms:
            try:
                r = run_symbol(
                    sym, data[sym], btc_close_4h, btc_ma50_4h, MA_PARAMS,
                    gate_enabled=1,
                    trail_activate_override=ta,
                    trail_distance_override=TRAIL_DISTANCE_FIXED,
                )
                results.append(r)
            except Exception as e:
                print(f"  SKIP {sym}: {e}")

        trades = stack_trades(results)
        metrics = get_portfolio_metrics(trades)
        exp1_results[ta] = metrics
        exp1_trades[ta] = trades
        exp1_sym_results[ta] = results

    # Baseline: gate OFF, current config trail
    baseline_results = []
    for sym in syms:
        try:
            r = run_symbol(
                sym, data[sym], btc_close_4h, btc_ma50_4h, MA_PARAMS,
                gate_enabled=0,
            )
            baseline_results.append(r)
        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    baseline_trades = stack_trades(baseline_results)
    baseline_metrics = get_portfolio_metrics(baseline_trades)

    # 결과 테이블
    hdr = f"  {'Variant':<28} {'N':>5}  {'PF':>6}  {'WR':>5}   {'MaxDD':>6}  {'EV(bp)':>7}  {'CAGR_LN':>7}"
    print(hdr)
    print(f"  {'-'*70}")
    print_metrics_row("Baseline (gate OFF, cfg)", baseline_metrics)
    for ta in TRAIL_ACTIVATE_GRID:
        label = f"gate ON + trail_act={ta}"
        print_metrics_row(label, exp1_results[ta])

    # Best trail_activate 선택
    best_ta = max(
        (ta for ta in TRAIL_ACTIVATE_GRID if exp1_results[ta]),
        key=lambda ta: exp1_results[ta].get("PF", 0),
        default=TRAIL_ACTIVATE_GRID[1],
    )
    best_m = exp1_results.get(best_ta, {})
    print(f"\n  >>> Best: trail_activate={best_ta} (PF={best_m.get('PF', 0):.3f})")

    # Best variant yearly breakdown
    best_yearly = yearly_metrics(exp1_trades[best_ta])
    baseline_yearly = yearly_metrics(baseline_trades)
    print_yearly_table(baseline_yearly, "Baseline (gate OFF)")
    print_yearly_table(best_yearly, f"Best: gate ON + trail_act={best_ta}")

    # Bootstrap PF CI for best
    if exp1_trades[best_ta]:
        best_pnl = portfolio_sim(exp1_trades[best_ta], max_concurrent=cfg.max_concurrent)
        boot = bootstrap_pf(best_pnl)
        print(f"\n  Bootstrap PF CI (best trail): median={boot['median']:.3f}  "
              f"95% CI=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]  "
              f"P(PF>=1)={boot['P_PF_ge_1']:.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    #  Experiment 2: CW Conviction Sizing
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CW Conviction Sizing Heuristic")
    print("=" * 70)
    print("Scoring: tight_spread(<median) + volume_spike(>1.5x) + strong_mom(>2x min)")
    print(f"Multipliers: {CW_MULTIPLIERS}")
    print()

    # CW sizing 적용 (baseline gate OFF 결과에 적용하여 순수 CW 효과 측정)
    cw_trades_baseline = apply_cw_sizing(baseline_results)
    cw_metrics_baseline = get_portfolio_metrics(cw_trades_baseline)

    # Score 분포 출력
    all_scores = []
    min_mom = cfg.ma_cluster_min_momentum
    all_spreads = []
    for r in baseline_results:
        all_spreads.extend(r["spreads"].tolist())
    median_spread = float(np.median(all_spreads)) if all_spreads else 0

    for r in baseline_results:
        avg_vol = None
        if r["vol_4h"] is not None and len(r["vol_4h"]) > 0:
            avg_vol = float(np.nanmean(r["vol_4h"]))
        for k in range(len(r["pnl"])):
            score = 0
            if r["spreads"][k] < median_spread:
                score += 1
            if avg_vol is not None and avg_vol > 0 and r["vol_4h"] is not None:
                ent_bar_4h = int(r["ent_4h"][k])
                if ent_bar_4h < len(r["vol_4h"]):
                    if r["vol_4h"][ent_bar_4h] > 1.5 * avg_vol:
                        score += 1
            if r["moms"][k] > 2.0 * min_mom:
                score += 1
            all_scores.append(score)

    if all_scores:
        scores_arr = np.array(all_scores)
        print("  Score distribution:")
        for s in range(4):
            cnt = int(np.sum(scores_arr == s))
            pct = 100.0 * cnt / len(scores_arr)
            print(f"    Score {s} (mult={CW_MULTIPLIERS[s]:.2f}x): {cnt:>5} trades ({pct:.1f}%)")
        print()

    print(hdr)
    print(f"  {'-'*70}")
    print_metrics_row("Baseline (no CW)", baseline_metrics)
    print_metrics_row("CW sized (gate OFF)", cw_metrics_baseline)

    if baseline_metrics and cw_metrics_baseline:
        dpf = cw_metrics_baseline.get("PF", 0) - baseline_metrics.get("PF", 0)
        dev = cw_metrics_baseline.get("EV_bp", 0) - baseline_metrics.get("EV_bp", 0)
        print(f"\n  ΔPF={dpf:+.3f}  ΔEV={dev:+.1f}bp")

    cw_yearly = yearly_metrics(cw_trades_baseline)
    print_yearly_table(cw_yearly, "CW sized (gate OFF)")

    # ════════════════════════════════════════════════════════════════════════
    #  Experiment 3: Combined (best trail + BTC gate + CW)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Combined (trail_act={best_ta} + gate ON + CW)")
    print("=" * 70)
    print()

    # Best trail + gate 결과에 CW sizing 적용
    best_sym_results = exp1_sym_results[best_ta]
    combined_trades = apply_cw_sizing(best_sym_results)
    combined_metrics = get_portfolio_metrics(combined_trades)

    print(hdr)
    print(f"  {'-'*70}")
    print_metrics_row("Baseline (gate OFF, cfg)", baseline_metrics)
    print_metrics_row(f"Trail={best_ta}+gate", exp1_results[best_ta])
    print_metrics_row("CW only (gate OFF)", cw_metrics_baseline)
    print_metrics_row("Combined (all 3)", combined_metrics)

    if baseline_metrics and combined_metrics:
        dpf = combined_metrics.get("PF", 0) - baseline_metrics.get("PF", 0)
        dev = combined_metrics.get("EV_bp", 0) - baseline_metrics.get("EV_bp", 0)
        ddd = combined_metrics.get("max_dd_pct", 0) - baseline_metrics.get("max_dd_pct", 0)
        print(f"\n  vs Baseline: ΔPF={dpf:+.3f}  ΔEV={dev:+.1f}bp  ΔMaxDD={ddd:+.1f}%p")

    combined_yearly = yearly_metrics(combined_trades)
    print_yearly_table(combined_yearly, "Combined")

    # Bootstrap PF CI for combined
    if combined_trades:
        combined_pnl = portfolio_sim(combined_trades, max_concurrent=cfg.max_concurrent)
        boot_c = bootstrap_pf(combined_pnl)
        print(f"\n  Bootstrap PF CI (combined): median={boot_c['median']:.3f}  "
              f"95% CI=[{boot_c['CI_lo']:.3f}, {boot_c['CI_hi']:.3f}]  "
              f"P(PF>=1)={boot_c['P_PF_ge_1']:.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    #  Summary & Report
    # ════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SUMMARY (elapsed: {elapsed:.0f}s)")
    print(f"{'='*70}")

    print(hdr)
    print(f"  {'-'*70}")
    print_metrics_row("1. Baseline (gate OFF)", baseline_metrics)
    if best_m:
        print_metrics_row(f"2. Trail={best_ta}+gate ON", best_m)
    print_metrics_row("3. CW sizing (gate OFF)", cw_metrics_baseline)
    print_metrics_row("4. Combined (all)", combined_metrics)

    # ── Report 저장 ──────────────────────────────────────────────────────
    report_dir = ROOT / "docs" / "backtest_results" / "202604"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r115_quick_wins.md"

    lines = [
        "# r115: Non-ML Quick Wins Validation for MA Cluster",
        "",
        f"Date: {today}, Commit: {commit}",
        "",
        "## Config",
        f"- cluster_pct={MA_PARAMS['cluster_pct']}, min_dur={MA_PARAMS['min_duration']}, "
        f"sl={MA_PARAMS['sl']}",
        f"- BTC Gate: MA{BTC_GATE_PERIOD} ({BTC_GATE_PERIOD * 4}h = {BTC_GATE_PERIOD * 4 / 24:.1f} days)",
        f"- MA_LN = {MA_LN:.1f}, max_concurrent = {cfg.max_concurrent}",
        f"- Symbols: {len(syms)}",
        "",
        "## Experiment 1: Trail Activate Sweep + BTC Gate",
        "",
        "| Variant | N | PF | WR% | MaxDD% | EV(bp) |",
        "|---------|---|----|----|--------|--------|",
    ]

    def _row(name: str, m: dict) -> str:
        if not m:
            return f"| {name} | - | - | - | - | - |"
        return (f"| {name} | {m['N']} | {m['PF']:.3f} | {m['WR']:.1f} | "
                f"{m['max_dd_pct']:.1f} | {m.get('EV_bp', 0):.1f} |")

    lines.append(_row("Baseline (gate OFF)", baseline_metrics))
    for ta in TRAIL_ACTIVATE_GRID:
        lines.append(_row(f"gate ON + trail_act={ta}", exp1_results[ta]))
    lines.append("")
    lines.append(f"Best: trail_activate={best_ta}")
    lines.append("")

    # Exp 1 yearly
    lines.append("### Yearly (best trail + gate)")
    lines.append("")
    lines.append("| Year | N | PF | WR% | EV(bp) |")
    lines.append("|------|---|----|----|--------|")
    for yr in sorted(best_yearly):
        y = best_yearly[yr]
        pf_str = f"{y['PF']:.2f}" if y["PF"] < 100 else "inf"
        lines.append(f"| {yr} | {y['N']} | {pf_str} | {y['WR']:.0f} | {y['EV']:.1f} |")

    if exp1_trades[best_ta]:
        lines.append("")
        lines.append(f"Bootstrap PF: median={boot['median']:.3f}, "
                      f"95% CI=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}], "
                      f"P(PF>=1)={boot['P_PF_ge_1']:.1f}%")

    lines.append("")
    lines.append("## Experiment 2: CW Conviction Sizing")
    lines.append("")
    lines.append("Scoring: tight_spread + volume_spike + strong_momentum (0-3)")
    lines.append(f"Multipliers: {CW_MULTIPLIERS}")
    lines.append("")
    lines.append("| Variant | N | PF | WR% | MaxDD% | EV(bp) |")
    lines.append("|---------|---|----|----|--------|--------|")
    lines.append(_row("No CW", baseline_metrics))
    lines.append(_row("CW sized", cw_metrics_baseline))

    if all_scores:
        lines.append("")
        lines.append("Score distribution:")
        scores_arr = np.array(all_scores)
        for s in range(4):
            cnt = int(np.sum(scores_arr == s))
            pct = 100.0 * cnt / len(scores_arr)
            lines.append(f"- Score {s} (x{CW_MULTIPLIERS[s]:.2f}): {cnt} trades ({pct:.1f}%)")

    lines.append("")
    lines.append("## Experiment 3: Combined")
    lines.append("")
    lines.append("| Variant | N | PF | WR% | MaxDD% | EV(bp) |")
    lines.append("|---------|---|----|----|--------|--------|")
    lines.append(_row("Baseline", baseline_metrics))
    lines.append(_row(f"Trail={best_ta}+gate", exp1_results.get(best_ta, {})))
    lines.append(_row("CW only", cw_metrics_baseline))
    lines.append(_row("Combined", combined_metrics))

    if combined_trades:
        lines.append("")
        lines.append(f"Bootstrap PF (combined): median={boot_c['median']:.3f}, "
                      f"95% CI=[{boot_c['CI_lo']:.3f}, {boot_c['CI_hi']:.3f}], "
                      f"P(PF>=1)={boot_c['P_PF_ge_1']:.1f}%")

    # Combined yearly
    lines.append("")
    lines.append("### Yearly (combined)")
    lines.append("")
    lines.append("| Year | N | PF | WR% | EV(bp) |")
    lines.append("|------|---|----|----|--------|")
    for yr in sorted(combined_yearly):
        y = combined_yearly[yr]
        pf_str = f"{y['PF']:.2f}" if y["PF"] < 100 else "inf"
        lines.append(f"| {yr} | {y['N']} | {pf_str} | {y['WR']:.0f} | {y['EV']:.1f} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
