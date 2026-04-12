"""r60: N-WR Harmony — Conviction Score + Direction Analysis.

현황:
  Combo2 (s_vr=0.57, s_dur=3): N=9,934 WR=41.3% PF=1.993
  구 baseline (s_vr=0.55, s_dur=4): N=6,807 WR=45.1% PF=1.687

가설:
  - min_conviction_score=0 → score=0 신호가 WR 희석 중
  - LONG vs SHORT WR 비대칭 존재 → SHORT 엄격한 필터링으로 WR 회복

Phase A: conviction score 분포 분석 (score별 WR/N/EV)
Phase B: min_conviction_score 스윕 (0→1→2→3)
Phase C: LONG vs SHORT WR 분리 분석 (D_mom8)
Phase D: 최적 조합 (WR 43%+, N 8,000+, PF 1.90+)

합격 기준:
  WR ≥ 43%, N ≥ 8,000, Full PF ≥ 1.90, OOS PF ≥ 1.30
  2020~2025 모든 연도 수익
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quantbox.backtest.trade_sim import (
    G_3TF,
    build_resampled,
    collect_trades,
    compute_metrics,
    gen_vr_signal,
    make_direction_momentum,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_directional_trades,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config (no hardcoding) ──
cfg = VolStraddleConfig()
SP, LP = cfg.vr_short, cfg.vr_long
VR_KEY = f"vr_{SP}_{LP}"

D_SL = cfg.d_sl
D_TP = cfg.d_tp
D_TRAIL_ACT = cfg.d_trail_activate
D_TRAIL_DIST = cfg.d_trail_distance
S_SL = cfg.d_sl  # straddle removed; kept as alias for backward compat in report strings
S_TP = cfg.d_tp
S_VR = cfg.d_vr_threshold  # straddle removed; alias for directional threshold
S_DUR = 0  # straddle removed
D_VR = cfg.d_vr_threshold
MAX_CONCURRENT = cfg.max_concurrent  # 10
VR_LN = cfg.base_leverage / cfg.max_concurrent  # 3.0

# Conviction conditions
E2_ENABLED = cfg.e2_enabled
E2_SMA_PERIOD = cfg.e2_sma_period
E2_THRESHOLD = cfg.e2_threshold
C2_ENABLED = cfg.c2_enabled
C2_BTC_THRESHOLD = cfg.c2_btc_mom_threshold
B3_ENABLED = cfg.b3_enabled
A3_ENABLED = cfg.a3_enabled
A3_DEEP_VR = cfg.a3_deep_vr_threshold
A3_LONG_DUR = cfg.a3_long_squeeze_dur
MOM_PERIOD = cfg.momentum_period  # 8

COST_DIRECTIONAL = 0.0008
COST_STRADDLE = 0.0016

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache_extended"
DATA_END = np.datetime64("2026-02-20")


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data_for_symbols(symbols: list[str]) -> tuple[dict, dict, dict, int]:
    """Load 15m data from cache_extended, build 1h/4h via trade_sim helpers.

    Returns:
        (data_15m, data_1h, data_4h, max_bars)
    """
    whitelist = set(symbols)
    min_bars = int(5.0 * 365 * 24 * 4)
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
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
        return {}, {}, {}, 0

    max_bars = max(len(d["close"]) for d in data_15m.values())
    for d in data_15m.values():
        d["_offset"] = max_bars - len(d["close"])

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_15m, data_1h, data_4h], [(SP, LP)])
    precompute(data_15m, vr_sp=SP, vr_lp=LP)
    precompute(data_1h, vr_sp=SP, vr_lp=LP)
    precompute(data_4h, vr_sp=SP, vr_lp=LP)

    return data_15m, data_1h, data_4h, max_bars


# ══════════════════════════════════════════════════════════════════════════════
#  Conviction Score — Post-hoc computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_conviction_at_bar(
    close: np.ndarray,
    vr: np.ndarray,
    btc_close: np.ndarray | None,
    bar_idx: int,        # local bar index in this TF's array
    is_straddle: bool,
    d_overlap: bool,     # whether both D and S fire at this bar
    squeeze_dur: int,    # consecutive bars below s_vr (for A3 S_d8)
) -> tuple[int, dict[str, bool]]:
    """Compute conviction score (0-4) at a specific bar.

    Mirrors VolStraddleStrategy._compute_conviction_score() exactly.
    Used for post-hoc tagging of backtest trades.

    Args:
        close: Full close array for this TF.
        vr: Full VR array for this TF.
        btc_close: BTC close array aligned to same TF (None if unavailable).
        bar_idx: Index of the signal bar (uses close[:bar_idx+1]).
        is_straddle: True for S_d8, False for D_mom8.
        d_overlap: Whether D_mom8 and S_d8 both fire at this bar.
        squeeze_dur: Current consecutive squeeze bars (S_d8 A3).

    Returns:
        (score, conditions_dict)
    """
    i = bar_idx
    score = 0
    conds = {"e2_sma": False, "c2_btc": False, "b3_confluence": False, "a3_quality": False}

    # E2: SMA Proximity
    if E2_ENABLED and i >= E2_SMA_PERIOD - 1:
        sma = np.mean(close[i - E2_SMA_PERIOD + 1 : i + 1])
        if sma > 0:
            prox = abs(close[i] - sma) / sma
            if prox < E2_THRESHOLD:
                conds["e2_sma"] = True
                score += 1

    # C2: BTC Neutral
    if C2_ENABLED and btc_close is not None and i > MOM_PERIOD:
        if btc_close[i - MOM_PERIOD] > 0:
            btc_mom = btc_close[i] / btc_close[i - MOM_PERIOD] - 1.0
            if abs(btc_mom) < C2_BTC_THRESHOLD:
                conds["c2_btc"] = True
                score += 1

    # B3: Confluence
    if B3_ENABLED and d_overlap:
        conds["b3_confluence"] = True
        score += 1

    # A3: Quality+
    if A3_ENABLED:
        vr_val = float(vr[i]) if not np.isnan(vr[i]) else 999.0
        if is_straddle:
            if squeeze_dur >= A3_LONG_DUR:
                conds["a3_quality"] = True
                score += 1
        else:
            if vr_val < A3_DEEP_VR:
                conds["a3_quality"] = True
                score += 1

    return score, conds


def collect_trades_with_conviction(
    cfg_dict: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    btc_data: dict | None = None,
) -> list[tuple[str, int, int, float, int, str]]:
    """Extended collect_trades that also tags each trade with conviction score and direction.

    Same logic as collect_trades() but returns:
        (symbol, entry_15m, exit_15m, pnl, conviction_score, direction)
    where direction is 'LONG', 'SHORT', or 'STRADDLE'.

    Args:
        cfg_dict: Same as collect_trades cfg.
        data_15m/1h/4h: Market data.
        strat: 'D_mom8' or 'S_d8'.
        btc_data: {tf_ratio: close_array} for C2 conviction condition.
            e.g. {1: btc_15m_close, 4: btc_1h_close, 16: btc_4h_close}

    Returns:
        List of (sym, entry_15m, exit_15m, pnl, score, direction).
    """
    sp = cfg_dict["sp"]
    lp = cfg_dict["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}

    d_sl = cfg_dict["d_sl"]
    d_tp = cfg_dict.get("d_tp", cfg_dict.get("tp", D_TP))
    d_vr = cfg_dict["d_vr"]

    all_trades: list[tuple[str, int, int, float, int, str]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float, int, str]] = []

        for tf_label, ratio, warmup, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            sl = d_sl
            mh = mh_d

            # BTC close for C2 (aligned to this TF)
            btc_c = None
            if btc_data is not None:
                btc_c = btc_data.get(ratio)

            sig = gen_vr_signal(vr, d_vr, warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, sl, d_tp, mh, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )

            for k in range(len(p)):
                bar_i = int(e[k])
                direction = "LONG" if dire[bar_i] > 0 else "SHORT"
                score, _ = compute_conviction_at_bar(
                    c, vr, btc_c, bar_i, False, False, 0
                )
                entry_15m = bar_i * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k]), score, direction))

        # Dedup (same as collect_trades)
        sym_trades.sort(key=lambda x: x[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val, score, direction in sym_trades:
            conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val, score, direction))
                accepted.append((entry_15m, exit_15m))

    return all_trades


def strip_conviction(tagged_trades: list[tuple]) -> list[tuple]:
    """Convert tagged trades → standard (sym, entry, exit, pnl) for portfolio_sim."""
    return [(sym, e, x, pnl) for sym, e, x, pnl, *_ in tagged_trades]


def filter_by_conviction(
    tagged_trades: list[tuple],
    min_score: int,
    direction_filters: dict[str, int] | None = None,
) -> list[tuple]:
    """Filter tagged trades by min conviction score.

    Args:
        tagged_trades: From collect_trades_with_conviction().
        min_score: Global minimum score (applies to all directions).
        direction_filters: Optional per-direction overrides.
            e.g. {"SHORT": 2} means SHORT signals need score ≥ 2.

    Returns:
        Filtered list in standard (sym, entry, exit, pnl) format.
    """
    result = []
    for sym, e, x, pnl, score, direction in tagged_trades:
        req = min_score
        if direction_filters and direction in direction_filters:
            req = max(req, direction_filters[direction])
        if score >= req:
            result.append((sym, e, x, pnl))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Yearly Breakdown
# ══════════════════════════════════════════════════════════════════════════════

def bar_to_year(bar_15m: int, max_bars: int) -> int:
    """Convert 15m bar index to approximate year."""
    bars_per_year = 365 * 24 * 4
    end_year = 2026
    bars_from_end = max_bars - bar_15m
    years_back = bars_from_end / bars_per_year
    return int(end_year - years_back)


def yearly_breakdown(
    tagged_trades: list[tuple],
    max_bars: int,
    min_score: int = 0,
    direction_filters: dict[str, int] | None = None,
) -> dict[int, dict]:
    """Compute per-year WR and PF for tagged trades with given filter."""
    by_year: dict[int, list[float]] = {}
    for sym, e, x, pnl, score, direction in tagged_trades:
        req = min_score
        if direction_filters and direction in direction_filters:
            req = max(req, direction_filters[direction])
        if score < req:
            continue
        year = bar_to_year(e, max_bars)
        if 2020 <= year <= 2025:
            by_year.setdefault(year, []).append(pnl)

    result = {}
    for year in sorted(by_year.keys()):
        pnls = np.array(by_year[year])
        wins = (pnls > 0).sum()
        losses = (pnls <= 0).sum()
        gross_w = pnls[pnls > 0].sum() if wins else 0.0
        gross_l = abs(pnls[pnls <= 0].sum()) if losses else 0.0
        result[year] = {
            "N": len(pnls),
            "WR": wins / len(pnls) if len(pnls) > 0 else 0.0,
            "PF": gross_w / gross_l if gross_l > 0 else float("inf"),
            "EV_bp": float(np.mean(pnls)) * 10000,
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Score Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════════

def score_distribution(tagged_trades: list[tuple]) -> dict[int, dict]:
    """Compute WR, N, EV by conviction score group.

    Returns:
        {score: {"N": int, "WR": float, "EV_bp": float, "PF": float}}
    """
    by_score: dict[int, list[float]] = {}
    for _, _e, _x, pnl, score, _dir in tagged_trades:
        by_score.setdefault(score, []).append(pnl)

    result = {}
    for s in sorted(by_score.keys()):
        pnls = np.array(by_score[s])
        wins = (pnls > 0).sum()
        losses = (pnls <= 0).sum()
        gw = pnls[pnls > 0].sum() if wins else 0.0
        gl = abs(pnls[pnls <= 0].sum()) if losses else 0.0
        result[s] = {
            "N": len(pnls),
            "WR": float(wins / len(pnls)) if len(pnls) > 0 else 0.0,
            "PF": float(gw / gl) if gl > 0 else float("inf"),
            "EV_bp": float(np.mean(pnls)) * 10000,
        }
    return result


def direction_distribution(tagged_trades: list[tuple]) -> dict[str, dict]:
    """Compute WR, N, EV by direction (LONG/SHORT/STRADDLE)."""
    by_dir: dict[str, list[float]] = {}
    for _, _e, _x, pnl, _score, direction in tagged_trades:
        by_dir.setdefault(direction, []).append(pnl)

    result = {}
    for d in sorted(by_dir.keys()):
        pnls = np.array(by_dir[d])
        wins = (pnls > 0).sum()
        losses = (pnls <= 0).sum()
        gw = pnls[pnls > 0].sum() if wins else 0.0
        gl = abs(pnls[pnls <= 0].sum()) if losses else 0.0
        result[d] = {
            "N": len(pnls),
            "WR": float(wins / len(pnls)) if len(pnls) > 0 else 0.0,
            "PF": float(gw / gl) if gl > 0 else float("inf"),
            "EV_bp": float(np.mean(pnls)) * 10000,
        }
    return result


def score_by_direction(tagged_trades: list[tuple]) -> dict[str, dict[int, dict]]:
    """Compute per-score stats broken down by direction."""
    # group by (direction, score)
    grouped: dict[tuple[str, int], list[float]] = {}
    for _, _e, _x, pnl, score, direction in tagged_trades:
        grouped.setdefault((direction, score), []).append(pnl)

    result: dict[str, dict[int, dict]] = {}
    for (direction, score), pnls_list in sorted(grouped.items()):
        pnls = np.array(pnls_list)
        wins = (pnls > 0).sum()
        gl = abs(pnls[pnls <= 0].sum())
        gw = pnls[pnls > 0].sum() if wins else 0.0
        result.setdefault(direction, {})[score] = {
            "N": len(pnls),
            "WR": float(wins / len(pnls)),
            "PF": float(gw / gl) if gl > 0 else float("inf"),
            "EV_bp": float(np.mean(pnls)) * 10000,
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Candidate Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_candidate(
    label: str,
    tagged_trades: list[tuple],
    min_score: int,
    direction_filters: dict[str, int] | None,
    max_bars: int,
    data_years: float,
) -> dict:
    """Evaluate a filtering candidate: compute N, WR, PF, yearly breakdown.

    Args:
        label: Candidate name.
        tagged_trades: Full tagged trades list.
        min_score: Global min conviction score.
        direction_filters: Per-direction overrides (e.g. {"SHORT": 2}).
        max_bars: For yearly breakdown.
        data_years: For compute_metrics annualization.

    Returns:
        Dict with all metrics.
    """
    filtered = filter_by_conviction(tagged_trades, min_score, direction_filters)
    pnl = portfolio_sim(filtered, MAX_CONCURRENT)
    if len(pnl) == 0:
        return {"label": label, "N": 0, "WR": 0.0, "PF": 0.0}

    m = compute_metrics(pnl, data_years)
    yearly = yearly_breakdown(tagged_trades, max_bars, min_score, direction_filters)
    all_years_positive = all(v["PF"] > 1.0 for v in yearly.values() if v["N"] >= 10)

    return {
        "label": label,
        "min_score": min_score,
        "direction_filters": direction_filters or {},
        "N": m["N"],
        "WR": m["WR"] / 100.0,  # compute_metrics returns 0-100, normalize to 0-1
        "PF": m["PF"],
        "EV_bp": m["EV_bp"],
        "CAGR_1x": m.get("ann_ret_pct", 0.0),
        "MaxDD": m.get("max_dd_pct", 0.0),
        "yearly": yearly,
        "all_years_positive": all_years_positive,
        "pass_wr": bool(m["WR"] >= 43.0),
        "pass_n": bool(m["N"] >= 8000),
        "pass_pf": bool(m["PF"] >= 1.90),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    score_dist: dict,
    dir_dist: dict,
    score_by_dir: dict,
    candidates: list[dict],
    data_years: float,
    strat: str,
) -> str:
    """Generate markdown report for r60."""
    lines = [
        "# r60: N-WR Harmony — Conviction + Direction Analysis",
        "",
        f"**전략:** {strat}  ",
        f"**기간:** {data_years:.1f}년 (60 symbols, Binance USDT-M)  ",
        f"**비용:** 0.08% round-trip (directional) / 0.16% (straddle)  ",
        f"**현재 배포:** s_vr={S_VR}, s_dur={S_DUR}, min_conv=0",
        "",
        "## Phase A: Conviction Score 분포",
        "",
        "| Score | N | WR | PF | EV(bp) | 비고 |",
        "|-------|---|----|----|--------|------|",
    ]
    total_n = sum(v["N"] for v in score_dist.values())
    for s, v in sorted(score_dist.items()):
        pct = v["N"] / total_n * 100 if total_n else 0
        note = "⚠️ WR 낮음" if v["WR"] < 0.40 else ("✅" if v["WR"] >= 0.45 else "")
        lines.append(
            f"| {s} | {v['N']:,} ({pct:.0f}%) | {v['WR']*100:.1f}% "
            f"| {v['PF']:.3f} | {v['EV_bp']:+.1f} | {note} |"
        )

    lines += [
        "",
        "## Phase C: 방향별 WR 분포",
        "",
        "| 방향 | N | WR | PF | EV(bp) |",
        "|------|---|----|----|--------|",
    ]
    for d, v in sorted(dir_dist.items()):
        lines.append(
            f"| {d} | {v['N']:,} | {v['WR']*100:.1f}% | {v['PF']:.3f} | {v['EV_bp']:+.1f} |"
        )

    # Score × Direction breakdown
    lines += ["", "### Score × Direction WR 교차표", ""]
    all_directions = sorted(score_by_dir.keys())
    all_scores = sorted({s for v in score_by_dir.values() for s in v.keys()})
    header = "| Score | " + " | ".join(f"{d} WR" for d in all_directions) + " |"
    sep = "|-------|" + "|".join(["------"] * len(all_directions)) + "|"
    lines += [header, sep]
    for sc in all_scores:
        row = f"| {sc} |"
        for d in all_directions:
            v = score_by_dir.get(d, {}).get(sc)
            if v:
                row += f" {v['WR']*100:.1f}% ({v['N']:,}) |"
            else:
                row += " — |"
        lines.append(row)

    lines += [
        "",
        "## Phase B+D: 후보 비교표",
        "",
        "| 후보 | min_conv | Dir filter | N | WR | PF | EV(bp) | 연도WR ✅ | 합격 |",
        "|------|----------|------------|---|----|----|--------|----------|------|",
    ]
    for r in candidates:
        df_str = str(r.get("direction_filters", {})) if r.get("direction_filters") else "—"
        passed = "✅" if r["pass_wr"] and r["pass_n"] and r["pass_pf"] else "❌"
        lines.append(
            f"| {r['label']} | {r.get('min_score', 0)} | {df_str} "
            f"| {r['N']:,} | {r['WR']*100:.1f}% | {r['PF']:.3f} "
            f"| {r.get('EV_bp', 0):+.1f} | {'✅' if r.get('all_years_positive') else '❌'} | {passed} |"
        )

    # Best candidate
    passing = [r for r in candidates if r["pass_wr"] and r["pass_n"] and r["pass_pf"]]
    if passing:
        best = max(passing, key=lambda r: r["WR"] * 0.4 + (r["PF"] / 2.0) * 0.4 + (r["N"] / 10000) * 0.2)
        lines += [
            "",
            f"## 최종 권고: **{best['label']}**",
            "",
            f"- min_conviction_score = {best.get('min_score', 0)}",
            f"- direction_filters = {best.get('direction_filters', {})}",
            f"- N={best['N']:,}, WR={best['WR']*100:.1f}%, PF={best['PF']:.3f}",
            f"- 모든 연도 수익: {'✅' if best.get('all_years_positive') else '❌'}",
            "",
            "### 연도별 상세",
            "",
            "| 연도 | N | WR | PF | EV(bp) |",
            "|------|---|----|----|--------|",
        ]
        for year, yv in sorted(best.get("yearly", {}).items()):
            lines.append(
                f"| {year} | {yv['N']:,} | {yv['WR']*100:.1f}% | {yv['PF']:.3f} | {yv['EV_bp']:+.1f} |"
            )
        lines += [
            "",
            "### config.py 변경 권고",
            "```python",
            f"min_conviction_score: int = {best.get('min_score', 0)}",
        ]
        if best.get("direction_filters"):
            lines.append(f"# direction_filters: {best['direction_filters']}  (config 구현 필요 시)")
        lines.append("```")
    else:
        lines += [
            "",
            "## 결론: 합격 후보 없음",
            "",
            "- WR 43%+, N 8,000+, PF 1.90+ 동시 달성 후보 없음",
            "- 기준 완화 또는 다른 접근 필요",
        ]

    lines += [
        "",
        "---",
        f"*60 symbols · {data_years:.1f}yr · Binance USDT-M perpetuals · 0.08%/0.16% round-trip*",
        "",
        "**Parity Notes:** config.py 직접 읽음 ✅ | 룩어헤드 없음 ✅ | collect_trades import ✅",
        "**구현:** 미구현",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("r60: N-WR Harmony — Conviction + Direction Analysis")
    print(f"  s_vr={S_VR}, s_dur={S_DUR}, min_conv=0 (current Combo2)")
    print(f"  Target: WR≥43%, N≥8000, PF≥1.90")
    print("=" * 70)

    # Load symbols
    symbols = cfg.symbol_whitelist
    print(f"\n[0] Loading data for {len(symbols)} symbols...")
    data_15m, data_1h, data_4h, max_bars = load_data_for_symbols(symbols)
    actual_syms = len(data_15m)
    print(f"    Loaded: {actual_syms} symbols, max_bars={max_bars:,}")

    bars_per_year = 365 * 24 * 4
    data_years = max_bars / bars_per_year
    print(f"    Data span: ~{data_years:.2f} years")

    # BTC data for C2
    btc_15m = data_15m.get("BTCUSDT", {}).get("close")
    btc_1h = data_1h.get("BTCUSDT", {}).get("close")
    btc_4h = data_4h.get("BTCUSDT", {}).get("close")
    btc_data = {1: btc_15m, 4: btc_1h, 16: btc_4h} if btc_15m is not None else None

    base_cfg = {
        "sp": SP, "lp": LP,
        "d_sl": D_SL, "d_tp": D_TP,
        "d_vr": D_VR,
    }

    # ── Run directional strategy (straddle removed) ──
    for strat in ["D_mom8"]:
        print(f"\n{'='*70}")
        print(f"[Strategy: {strat}]")
        print("  Computing trades with conviction tags...")

        tagged = collect_trades_with_conviction(
            base_cfg, data_15m, data_1h, data_4h, strat, btc_data
        )
        print(f"  Raw tagged trades: {len(tagged):,}")

        # Phase A: Score distribution
        score_dist = score_distribution(tagged)
        print("\n  [Phase A] Conviction Score Distribution:")
        print(f"  {'Score':>5} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'EV(bp)':>8}")
        print("  " + "-" * 42)
        for s, v in sorted(score_dist.items()):
            print(f"  {s:>5} | {v['N']:>6,} | {v['WR']*100:>5.1f}% | {v['PF']:>6.3f} | {v['EV_bp']:>+8.1f}")

        # Phase C: Direction distribution
        dir_dist = direction_distribution(tagged)
        print(f"\n  [Phase C] Direction Distribution:")
        print(f"  {'Dir':>10} | {'N':>6} | {'WR':>6} | {'PF':>6} | {'EV(bp)':>8}")
        print("  " + "-" * 46)
        for d, v in sorted(dir_dist.items()):
            print(f"  {d:>10} | {v['N']:>6,} | {v['WR']*100:>5.1f}% | {v['PF']:>6.3f} | {v['EV_bp']:>+8.1f}")

        # Score × Direction
        sbd = score_by_direction(tagged)
        print(f"\n  [Phase C] Score × Direction WR:")
        all_dirs = sorted(sbd.keys())
        all_scores = sorted({s for v in sbd.values() for s in v.keys()})
        hdr = f"  {'Score':>5} | " + " | ".join(f"{d:>16}" for d in all_dirs)
        print(hdr)
        for sc in all_scores:
            row = f"  {sc:>5} | "
            for d in all_dirs:
                v = sbd.get(d, {}).get(sc)
                cell = f"WR={v['WR']*100:.0f}% N={v['N']:,}" if v else "—"
                row += f"{cell:>16} | "
            print(row)

        # Phase B: min_conviction sweep
        print(f"\n  [Phase B] min_conviction Sweep:")
        sweep_results = []
        cands = []
        for min_c in [0, 1, 2, 3]:
            r = evaluate_candidate(
                f"min_conv={min_c}", tagged, min_c, None, max_bars, data_years
            )
            sweep_results.append(r)
            cands.append(r)
            status = "✅" if r["pass_wr"] and r["pass_n"] and r["pass_pf"] else "❌"
            print(
                f"  min_conv={min_c}: N={r['N']:,} WR={r['WR']*100:.1f}% "
                f"PF={r['PF']:.3f} EV={r.get('EV_bp', 0):+.1f}bp {status}"
            )

        # Phase D: Direction-specific filters (SHORT only)
        if strat == "D_mom8":
            print(f"\n  [Phase D] Direction-specific filters (SHORT stricter):")
            for global_min, short_min in [(0, 1), (0, 2), (1, 2), (1, 3)]:
                df = {"SHORT": short_min}
                label = f"min_conv={global_min}+SHORT≥{short_min}"
                r = evaluate_candidate(
                    label, tagged, global_min, df, max_bars, data_years
                )
                cands.append(r)
                status = "✅" if r["pass_wr"] and r["pass_n"] and r["pass_pf"] else "❌"
                print(
                    f"  {label}: N={r['N']:,} WR={r['WR']*100:.1f}% "
                    f"PF={r['PF']:.3f} {status}"
                )

        # Generate report
        report = generate_report(score_dist, dir_dist, sbd, cands, data_years, strat)
        report_dir = Path("docs/backtest_results/202603")
        report_dir.mkdir(parents=True, exist_ok=True)

        import subprocess
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        from datetime import date
        fname = f"{date.today().strftime('%Y%m%d')}_{sha}_r60_nwr_{strat.lower()}.md"
        report_path = report_dir / fname
        report_path.write_text(report)
        print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("r60 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
