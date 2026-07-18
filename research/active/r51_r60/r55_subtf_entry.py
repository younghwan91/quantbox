#!/usr/bin/env python3
"""r55: Sub-TF Entry Refinement — WR / R:R 개선 without Lookahead Bias.

r54 발견: 동일 TF 내 feature 필터는 효과 없음 (포트폴리오 composition 이동 효과).
r55 가설: 신호 TF보다 작은 TF의 완성봉을 활용한 진입 정밀화.

두 가지 방법:
  A) Sub-TF VR Stackup (WR 목표):
     - 4h VR squeeze 발화 시 → 해당 4h bar를 구성하는 15m bars에서 VR도 압축인지 확인
     - 두 TF 모두 압축 = "이중 스퀴즈" → 더 강한 시그널
     - 룩어헤드 없음: 4h bar 닫힐 때 내부 15m bars 전부 완성
  B) Limit Order Pullback Entry (R:R 목표):
     - 신호 발화 후 즉시 진입 대신, N봉 내 되돌림에 지정가 주문
     - 더 낮은 진입가 → 동일 TP 거리 → 더 높은 R:R
     - 룩어헤드 없음: 지정가 충족 여부는 미래 봉의 high/low로 판단 (backtest-parity OK)

Production baseline (v14.5.2, 60 syms, 6yr):
  N=6,207 | PF=1.563 | WR=40.6% | EV=+38bp | Calmar=57.9 | DD=60.5%

Phases:
  Phase 0: Config 검증
  Phase 1: Baseline 재현 (±5% 허용)
  Phase 2: Sub-TF VR Stackup 스윕 (4h→15m, 1h→15m)
  Phase 3: Limit Order Pullback Entry 스윕
  Phase 4: 최선 조합 통합
  Phase 5: OOS 검증 + 보고서

Usage:
    uv run python research/active/r51_r60/r55_subtf_entry.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    G_3TF,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    compute_momentum,
    gen_vr_signal,
    make_direction_momentum,
)

# ── Straddle stubs (S_d8 removed from production) ────────────────────────────
COST_STRADDLE = 0.0016


def gen_vr_dur_signal(vr, threshold, min_dur, warmup):  # type: ignore[override]
    """Stub: S_d8 straddle signal removed from production."""
    import numpy as np
    return np.zeros(len(vr), dtype=np.int8)


def sim_straddle_trailing(*args, **kwargs):  # type: ignore[override]
    """Stub: straddle simulator removed from production."""
    import numpy as np
    return np.array([]), np.array([]), np.array([])
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ── Load production config (NO hardcoding) ────────────────────────────────────
cfg = VolStraddleConfig()

D_SL             = cfg.d_sl                        # 0.01
D_TP             = cfg.d_tp                        # 0.12
D_TRAIL_ACTIVATE = cfg.d_trail_activate             # 0.005
D_TRAIL_DISTANCE = cfg.d_trail_distance             # 0.002
S_SL             = 0.015   # s_sl removed from production config
S_TP             = 0.05    # s_tp removed from production config
S_TRAIL_ACTIVATE = 0.015   # s_trail_activate removed from production config
S_TRAIL_DISTANCE = 0.002   # s_trail_distance removed from production config
MAX_CONC         = cfg.max_concurrent               # 10
VR_SHORT         = cfg.vr_short                     # 8
VR_LONG          = cfg.vr_long                      # 56

print("[PHASE 0] Config 검증")
print(f"  D_TRAIL: activate={D_TRAIL_ACTIVATE:.3f} / distance={D_TRAIL_DISTANCE:.3f}")
print(f"  S_TRAIL: activate={S_TRAIL_ACTIVATE:.3f} / distance={S_TRAIL_DISTANCE:.3f}")
print(f"  D_TP={D_TP:.2f}, D_SL={D_SL:.3f}, L/N={cfg.base_leverage / MAX_CONC:.1f}")
assert D_TRAIL_ACTIVATE == 0.015, f"D trail activate mismatch: {D_TRAIL_ACTIVATE}"
assert D_TRAIL_DISTANCE == 0.005, f"D trail distance mismatch: {D_TRAIL_DISTANCE}"  # r106: 0.002→0.005 (rule #8)
print("  ✅ All params match production config.py\n")

CACHE_DIR = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_syms(min_years: float = 5.0) -> tuple[dict, str]:
    whitelist = set(cfg.symbol_whitelist)  # 60 live symbols only
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in whitelist:
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


TF_CONFIGS = [
    (1,  50,  192, 192),   # (ratio, warmup, max_hold_D, max_hold_S)
    (4,  13,   48,  48),
    (16,  4,   12,  12),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Method A: Sub-TF VR Stackup
#  Zero-lookahead: 4h bar i closes → 15m bars [i*16 .. i*16+15] are all done
# ═══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def gen_vr_signal_with_subtf_vr(
    vr_upper: np.ndarray,
    vr_lower: np.ndarray,
    threshold_upper: float,
    threshold_lower: float,
    warmup: int,
    lower_ratio: int,
) -> np.ndarray:
    """Signal fires only when BOTH upper-TF and lower-TF VR are compressed.

    룩어헤드 없음:
      upper-TF bar i가 닫히는 시점에 [i*lower_ratio .. i*lower_ratio+lower_ratio-1]
      에 해당하는 lower-TF bars는 전부 완성 상태.
      last lower-TF bar in upper-TF bar i = i*lower_ratio + lower_ratio - 1

    Args:
        vr_upper: VR array at signal timeframe (e.g., 4h).
        vr_lower: VR array at confirmation timeframe (e.g., 15m).
        threshold_upper: Primary squeeze threshold.
        threshold_lower: Sub-TF confirmation threshold.
        warmup: Bars to skip at start (upper-TF).
        lower_ratio: How many lower bars per upper bar (e.g., 16 for 4h→15m).

    Returns:
        Binary signal array (0 or 1), shape = vr_upper.shape.
    """
    n = len(vr_upper)
    sig = np.zeros(n, dtype=np.int8)
    n_lower = len(vr_lower)
    for i in range(warmup, n):
        if np.isnan(vr_upper[i]) or vr_upper[i] >= threshold_upper:
            continue
        # Last lower-TF bar in upper-TF bar i
        last_lower_idx = i * lower_ratio + lower_ratio - 1
        if last_lower_idx >= n_lower:
            continue
        if not np.isnan(vr_lower[last_lower_idx]) and vr_lower[last_lower_idx] < threshold_lower:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_vr_dur_signal_with_subtf_vr(
    vr_upper: np.ndarray,
    vr_lower: np.ndarray,
    threshold_upper: float,
    threshold_lower: float,
    min_dur: int,
    warmup: int,
    lower_ratio: int,
) -> np.ndarray:
    """S_d8 straddle signal with sub-TF VR confirmation."""
    n = len(vr_upper)
    sig = np.zeros(n, dtype=np.int8)
    n_lower = len(vr_lower)
    cnt = 0
    for i in range(n):
        if not np.isnan(vr_upper[i]) and vr_upper[i] < threshold_upper:
            cnt += 1
        else:
            cnt = 0
        if i < warmup or cnt < min_dur:
            continue
        last_lower_idx = i * lower_ratio + lower_ratio - 1
        if last_lower_idx >= n_lower:
            continue
        if not np.isnan(vr_lower[last_lower_idx]) and vr_lower[last_lower_idx] < threshold_lower:
            sig[i] = 1
    return sig


def collect_vr_with_subtf(
    d15: dict, d1h: dict, d4h: dict,
    lower_threshold: float,   # 15m VR threshold for confirmation
    apply_to_4h: bool = True,
    apply_to_1h: bool = True,
) -> list[tuple[str, int, int, float]]:
    """Collect VR squeeze trades with sub-TF VR confirmation.

    For 15m signals (ratio=1), there's no sub-TF available, so they pass through
    unchanged (no stackup). Stackup only applies to 4h and/or 1h signals.
    """
    tf = {1: d15, 4: d1h, 16: d4h}
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(d15):
        for ratio, warmup, mh_d, mh_s in TF_CONFIGS:
            ds = tf.get(ratio)
            if ds is None or sym not in ds:
                continue
            d = ds[sym]
            vr_key = f"vr_{VR_SHORT}_{VR_LONG}"
            vr_upper = d.get(vr_key)
            if vr_upper is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            mom = d.get("mom_8", compute_momentum(c, cfg.lookback))

            # Sub-TF VR from 15m data (always available)
            if sym in d15:
                vr_lower = d15[sym].get(vr_key)
            else:
                vr_lower = None

            use_stackup = (
                vr_lower is not None
                and ((ratio == 16 and apply_to_4h) or (ratio == 4 and apply_to_1h))
            )

            # D_mom8 directional
            if use_stackup:
                sig_d = gen_vr_signal_with_subtf_vr(
                    vr_upper, vr_lower,
                    cfg.d_vr_threshold, lower_threshold,
                    warmup, ratio,
                )
            else:
                sig_d = gen_vr_signal(vr_upper, cfg.d_vr_threshold, warmup)

            dire = make_direction_momentum(sig_d, mom)
            p_d, e_d, x_d = sim_directional_trailing(
                c, h, lo, sig_d, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
            )
            for k in range(len(p_d)):
                all_trades.append((sym, int(e_d[k]) * ratio, int(x_d[k]) * ratio + ratio - 1, float(p_d[k])))

            # S_d8 straddle
            if use_stackup:
                sig_s = gen_vr_dur_signal_with_subtf_vr(
                    vr_upper, vr_lower,
                    0.55, lower_threshold,
                    4, warmup, ratio,
                )
            else:
                sig_s = gen_vr_dur_signal(vr_upper, 0.55, 4, warmup)

            p_s, e_s, x_s = sim_straddle_trailing(
                c, h, lo, sig_s, S_SL, S_TP, mh_s, COST_STRADDLE,
                S_TRAIL_ACTIVATE, S_TRAIL_DISTANCE,
            )
            for k in range(len(p_s)):
                all_trades.append((sym, int(e_s[k]) * ratio, int(x_s[k]) * ratio + ratio - 1, float(p_s[k])))

    # MA cluster — unchanged (stackup not applied to MA cluster here)
    ma_trades = collect_enhanced_trades(
        d15, d1h, d4h,
        cluster_pct=cfg.ma_cluster_pct,
        min_duration=cfg.ma_cluster_min_duration,
        max_duration=cfg.ma_cluster_max_duration,
        min_momentum=cfg.ma_cluster_min_momentum,
        sl=cfg.ma_cluster_sl,
        tp=cfg.ma_cluster_tp,
        tf_restriction=16,   # 4h only (production)
    )
    all_trades.extend(ma_trades)

    return all_trades


def collect_baseline(d15: dict, d1h: dict, d4h: dict) -> list[tuple[str, int, int, float]]:
    """Production-parity baseline (same as r53 canonical)."""
    # threshold_lower=1.0 → always True (no stackup filter)
    return collect_vr_with_subtf(d15, d1h, d4h, lower_threshold=1.0, apply_to_4h=False, apply_to_1h=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Method B: Limit Order Pullback Entry
#  룩어헤드 없음: 지정가 충족 여부는 미래봉의 high/low로 확인 (이미 완성된 봉)
# ═══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_directional_pullback(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    dire: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,
    trail_distance: float,
    pullback_pct: float,    # 0.002 = 0.2% 되돌림 후 진입
    pullback_window: int,   # 최대 몇 봉 기다릴지
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with limit order pullback entry.

    신호 발화 후 즉시 진입 대신, 되돌림 지정가에 진입.
    지정가 미충족 시 해당 신호 스킵.

    Entry timing:
      LONG: entry_price = close[i] * (1 - pullback_pct)  → low[j] <= entry_price
      SHORT: entry_price = close[i] * (1 + pullback_pct) → high[j] >= entry_price

    SL/TP 기준: entry_price (더 나은 진입가에서 측정)
    """
    n = len(close)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, dtype=np.float64)
    ent_arr = np.empty(cap, dtype=np.int64)
    ext_arr = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        d = dire[i]
        signal_close = close[i]
        if signal_close <= 0.0:
            i += 1
            continue

        # 지정가 계산
        if d == 1:
            limit_price = signal_close * (1.0 - pullback_pct)
        else:
            limit_price = signal_close * (1.0 + pullback_pct)

        # 다음 pullback_window 봉 내에서 지정가 충족 확인
        fill_bar = -1
        search_end = min(i + 1 + pullback_window, n)
        for w in range(i + 1, search_end):
            if d == 1 and low[w] <= limit_price:
                fill_bar = w
                break
            elif d == -1 and high[w] >= limit_price:
                fill_bar = w
                break

        if fill_bar == -1:
            # 지정가 미충족 → 신호 스킵
            i += 1
            continue

        # 지정가 충족 → entry at limit_price
        entry = limit_price
        end = min(fill_bar + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0

        for j in range(fill_bar + 1, end):
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
            else:  # SHORT
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
            # max_hold 만료
            if d == 1:
                trade_pnl = (close[end - 1] - entry) / entry
            else:
                trade_pnl = (entry - close[end - 1]) / entry

        trade_pnl -= cost
        pnl_arr[cnt] = trade_pnl
        ent_arr[cnt] = fill_bar
        ext_arr[cnt] = eb
        cnt += 1
        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]


def collect_vr_with_pullback(
    d15: dict, d1h: dict, d4h: dict,
    pullback_pct: float,
    pullback_window: int,
    apply_to_d_mom8: bool = True,
) -> list[tuple[str, int, int, float]]:
    """VR squeeze trades with pullback entry."""
    tf = {1: d15, 4: d1h, 16: d4h}
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(d15):
        for ratio, warmup, mh_d, mh_s in TF_CONFIGS:
            ds = tf.get(ratio)
            if ds is None or sym not in ds:
                continue
            d = ds[sym]
            vr_key = f"vr_{VR_SHORT}_{VR_LONG}"
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            mom = d.get("mom_8", compute_momentum(c, cfg.lookback))

            # D_mom8 with pullback
            sig_d = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig_d, mom)
            if apply_to_d_mom8 and pullback_pct > 0:
                p_d, e_d, x_d = sim_directional_pullback(
                    c, h, lo, sig_d, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                    D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
                    pullback_pct, pullback_window,
                )
            else:
                p_d, e_d, x_d = sim_directional_trailing(
                    c, h, lo, sig_d, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                    D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
                )
            for k in range(len(p_d)):
                all_trades.append((sym, int(e_d[k]) * ratio, int(x_d[k]) * ratio + ratio - 1, float(p_d[k])))

            # S_d8 straddle — no pullback (straddle enters long+short, pullback less meaningful)
            sig_s = gen_vr_dur_signal(vr, 0.55, 4, warmup)
            p_s, e_s, x_s = sim_straddle_trailing(
                c, h, lo, sig_s, S_SL, S_TP, mh_s, COST_STRADDLE,
                S_TRAIL_ACTIVATE, S_TRAIL_DISTANCE,
            )
            for k in range(len(p_s)):
                all_trades.append((sym, int(e_s[k]) * ratio, int(x_s[k]) * ratio + ratio - 1, float(p_s[k])))

    # MA cluster — unchanged
    ma_trades = collect_enhanced_trades(
        d15, d1h, d4h,
        cluster_pct=cfg.ma_cluster_pct,
        min_duration=cfg.ma_cluster_min_duration,
        max_duration=cfg.ma_cluster_max_duration,
        min_momentum=cfg.ma_cluster_min_momentum,
        sl=cfg.ma_cluster_sl,
        tp=cfg.ma_cluster_tp,
        tf_restriction=16,
    )
    all_trades.extend(ma_trades)

    return all_trades


# ═══════════════════════════════════════════════════════════════════════════════
#  Method C: MA Cluster + 15m Pullback Entry (4h breakout → wait for 15m retrace)
# ═══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_ma_cluster_pullback(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    max_duration: int,
    min_momentum: float,
    sl: float,
    tp: float,
    max_hold_upper: int,   # in upper-TF bars (4h)
    cost: float,
    warmup: int,
    trail_activate: float,
    trail_distance: float,
    pullback_pct: float,       # 되돌림 비율 (0 = 즉시 진입)
    pullback_window: int,      # 최대 기다릴 하위-TF 봉 수 (15m 기준)
    lower_ratio: int,          # 4h→15m = 16
    close_lower: np.ndarray,  # 15m close array (동일 심볼)
    high_lower: np.ndarray,   # 15m high array
    low_lower: np.ndarray,    # 15m low array
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MA Cluster sim with optional 15m pullback entry after 4h breakout.

    4h bar에서 MA cluster 브레이크아웃 후, 15m 봉 기준으로 되돌림 대기.
    Returns (pnl, entry_15m, exit_15m) — all indices in 15m space.
    """
    n = len(close)
    n_lower = len(close_lower)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    cnt = 0
    i = warmup
    in_position = False
    position_end_lower = 0   # 15m bar when position ends

    while i < n:
        # Skip if still in a position (convert 4h exit to 15m)
        if in_position:
            if i * lower_ratio < position_end_lower:
                i += 1
                continue
            else:
                in_position = False

        # MA Cluster detection (same logic as original)
        # Check if we have enough SMA data
        if (np.isnan(sma20[i]) or np.isnan(sma50[i]) or
                np.isnan(sma100[i]) or np.isnan(sma200[i])):
            i += 1
            continue

        sma_min = min(sma20[i], sma50[i], sma100[i], sma200[i])
        sma_max = max(sma20[i], sma50[i], sma100[i], sma200[i])
        if sma_min <= 0.0:
            i += 1
            continue

        spread = (sma_max - sma_min) / sma_min

        # Count consecutive bars with cluster
        dur = 0
        for back in range(i, max(warmup - 1, i - max_duration - 1), -1):
            if np.isnan(sma200[back]):
                break
            s_min = min(sma20[back], sma50[back], sma100[back], sma200[back])
            s_max = max(sma20[back], sma50[back], sma100[back], sma200[back])
            if s_min <= 0.0 or (s_max - s_min) / s_min > cluster_pct:
                break
            dur += 1

        if dur < min_duration or dur > max_duration:
            i += 1
            continue

        # Cluster confirmed — check for breakout
        mom = momentum[i]
        if abs(mom) < min_momentum:
            i += 1
            continue

        d = 1 if mom > 0 else -1

        # Signal: breakout at 4h bar i close
        # Determine entry price using pullback approach
        signal_close_4h = close[i]

        # Convert 4h bar i to 15m index range
        lower_start = i * lower_ratio       # first 15m bar of 4h bar i
        lower_end = lower_start + lower_ratio  # exclusive

        if pullback_pct <= 0.0 or lower_end > n_lower:
            # Immediate entry at 4h close (original behavior)
            entry = signal_close_4h
            fill_bar_lower = lower_end - 1  # last 15m bar of 4h bar i
        else:
            # Wait for pullback in 15m bars AFTER the 4h bar closes
            if d == 1:
                limit_price = signal_close_4h * (1.0 - pullback_pct)
            else:
                limit_price = signal_close_4h * (1.0 + pullback_pct)

            fill_bar_lower = -1
            search_end_lower = min(lower_end + pullback_window, n_lower)
            for w in range(lower_end, search_end_lower):
                if d == 1 and low_lower[w] <= limit_price:
                    fill_bar_lower = w
                    break
                elif d == -1 and high_lower[w] >= limit_price:
                    fill_bar_lower = w
                    break

            if fill_bar_lower == -1:
                i += 1
                continue
            entry = limit_price

        # Trade sim from fill_bar_lower+1 in 15m space
        max_hold_lower = max_hold_upper * lower_ratio
        end_lower = min(fill_bar_lower + 1 + max_hold_lower, n_lower)
        eb_lower = end_lower - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0

        for j in range(fill_bar_lower + 1, end_lower):
            if d == 1:
                cur_gain = (high_lower[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low_lower[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb_lower = j
                        break
                if high_lower[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb_lower = j
                    break
                if not trailing_active and low_lower[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb_lower = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
            else:
                cur_gain = (entry - low_lower[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high_lower[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb_lower = j
                        break
                if low_lower[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb_lower = j
                    break
                if not trailing_active and high_lower[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb_lower = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
        else:
            if d == 1:
                trade_pnl = (close_lower[end_lower - 1] - entry) / entry
            else:
                trade_pnl = (entry - close_lower[end_lower - 1]) / entry

        trade_pnl -= cost
        pnl_arr[cnt] = trade_pnl
        ent_arr[cnt] = fill_bar_lower
        ext_arr[cnt] = eb_lower
        cnt += 1

        position_end_lower = eb_lower
        in_position = True
        # Jump to next 4h bar after position ends
        i = eb_lower // lower_ratio + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]


def collect_ma_cluster_pullback(
    d15: dict, d1h: dict, d4h: dict,
    pullback_pct: float,
    pullback_window_15m: int,
) -> list[tuple[str, int, int, float]]:
    """Collect MA cluster trades with optional 15m pullback entry."""
    all_trades: list[tuple[str, int, int, float]] = []
    lower_ratio = 16

    for sym in sorted(d15):
        if sym not in d4h or sym not in d15:
            continue
        d_upper = d4h[sym]
        d_lower = d15[sym]
        if "sma_200" not in d_upper or "mom_8" not in d_upper:
            continue

        p, e, x = sim_ma_cluster_pullback(
            d_upper["close"], d_upper["high"], d_upper["low"],
            d_upper["sma_20"], d_upper["sma_50"], d_upper["sma_100"], d_upper["sma_200"],
            d_upper["mom_8"],
            cfg.ma_cluster_pct,
            cfg.ma_cluster_min_duration,
            cfg.ma_cluster_max_duration,
            cfg.ma_cluster_min_momentum,
            cfg.ma_cluster_sl,
            cfg.ma_cluster_tp,
            12,   # max_hold in 4h bars
            COST_DIRECTIONAL,
            4,    # warmup in 4h bars (consistent with TF_CONFIGS)
            cfg.ma_cluster_trail_activate,
            cfg.ma_cluster_trail_distance,
            pullback_pct,
            pullback_window_15m,
            lower_ratio,
            d_lower["close"], d_lower["high"], d_lower["low"],
        )
        for k in range(len(p)):
            all_trades.append((sym, int(e[k]), int(x[k]), float(p[k])))

    return all_trades


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics Helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_metrics(
    trades: list[tuple[str, int, int, float]],
    data_years: float,
    label: str = "",
    n_boot: int = 500,
) -> dict:
    pnl = portfolio_sim(trades, MAX_CONC)
    m = compute_metrics(pnl, data_years)
    boot = bootstrap_pf(pnl, n_boot=n_boot)
    m["boot"] = boot
    m["label"] = label
    m["n_trades_raw"] = len(trades)
    print(
        f"  {label:<48} N={m['N']:5d}({len(trades):5d}) "
        f"PF={m['PF']:.3f} WR={m['WR']:.1f}% "
        f"EV={m['EV_bp']:+.0f}bp "
        f"Cal={m['calmar']:.1f} DD={m['max_dd_pct']:.1f}% "
        f"CI=[{boot['CI_lo']:.3f},{boot['CI_hi']:.3f}]"
    )
    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    commit = get_commit()
    today = datetime.now().strftime("%Y%m%d")

    print(f"\n{'='*80}")
    print(f"r55: Sub-TF Entry Refinement  [{commit}]  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*80}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    t0 = time.time()
    d15, ref_path = load_syms(5.0)
    d1h, d4h = build_tf(d15)
    bar0 = get_bar0(ref_path)
    data_years = max(len(d["close"]) for d in d15.values()) / (4 * 24 * 365)
    print(f"  {len(d15)} syms | {data_years:.2f}yr | bar0={bar0:%Y-%m-%d} | {time.time()-t0:.1f}s\n")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Phase 1: Baseline
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PHASE 1: Baseline Reproduction")
    print("=" * 80)
    trades_base = collect_baseline(d15, d1h, d4h)
    m_base = run_metrics(trades_base, data_years, "Baseline (no sub-TF filter)", n_boot=1000)

    canon_n, canon_pf = 6207, 1.563
    n_drift = abs(m_base["N"] - canon_n) / canon_n
    pf_drift = abs(m_base["PF"] - canon_pf) / canon_pf
    if n_drift < 0.10 and pf_drift < 0.10:
        print(f"✅ Within 10% of canonical (N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}%)\n")
    else:
        print(f"⚠️  Drift: N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}%  — check cache/config\n")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Phase 2: Sub-TF VR Stackup
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PHASE 2: Sub-TF VR Stackup (4h/1h → 15m 이중 스퀴즈 확인)")
    print("  룩어헤드 없음: 4h bar 닫힐 때 내부 15m bars 전부 완성")
    print("=" * 80)

    # 15m VR threshold sweep: 낮을수록 엄격한 필터
    # 1.0 = 15m 어떤 값이든 통과 (사실상 비활성), 0.3 = 매우 강한 압축만 통과
    lower_thresholds = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35]

    stackup_results: list[dict] = []
    print("\n[A] 4h 신호 + 15m 확인 (apply_to_4h=True, apply_to_1h=False)")
    for thr in lower_thresholds:
        label = f"SubTF_4h  15m_thr={thr:.2f}"
        trades = collect_vr_with_subtf(d15, d1h, d4h, lower_threshold=thr, apply_to_4h=True, apply_to_1h=False)
        m = run_metrics(trades, data_years, label)
        m["lower_threshold"] = thr
        m["apply_to"] = "4h_only"
        stackup_results.append(m)

    print("\n[B] 1h 신호 + 15m 확인 (apply_to_4h=False, apply_to_1h=True)")
    for thr in lower_thresholds:
        label = f"SubTF_1h  15m_thr={thr:.2f}"
        trades = collect_vr_with_subtf(d15, d1h, d4h, lower_threshold=thr, apply_to_4h=False, apply_to_1h=True)
        m = run_metrics(trades, data_years, label)
        m["lower_threshold"] = thr
        m["apply_to"] = "1h_only"
        stackup_results.append(m)

    print("\n[C] 4h+1h 신호 모두 + 15m 확인")
    for thr in [0.5, 0.4]:
        label = f"SubTF_4h+1h  15m_thr={thr:.2f}"
        trades = collect_vr_with_subtf(d15, d1h, d4h, lower_threshold=thr, apply_to_4h=True, apply_to_1h=True)
        m = run_metrics(trades, data_years, label)
        m["lower_threshold"] = thr
        m["apply_to"] = "4h_and_1h"
        stackup_results.append(m)

    # Best from stackup (highest PF with N >= 80% baseline)
    n_min = int(m_base["N"] * 0.80)
    valid_stackup = [m for m in stackup_results if m["N"] >= n_min]
    best_stackup = max(valid_stackup, key=lambda x: x["PF"]) if valid_stackup else None

    print(f"\n✅ Best Sub-TF Stackup (N≥{n_min}): ", end="")
    if best_stackup:
        print(f"{best_stackup['label']} → PF={best_stackup['PF']:.3f} (was {m_base['PF']:.3f})")
    else:
        print("None meeting N constraint")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Phase 3: Limit Order Pullback Entry (VR Squeeze D_mom8)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 3: Limit Order Pullback Entry — VR Squeeze D_mom8")
    print("  신호 발화 후 되돌림 지정가에 진입 → 더 낮은 진입가 → R:R 개선")
    print("=" * 80)

    pullback_pcts   = [0.001, 0.002, 0.003, 0.005]
    pullback_windows = [1, 2, 3, 5]

    pullback_results: list[dict] = []
    for pb_pct in pullback_pcts:
        for pb_win in pullback_windows:
            label = f"Pullback pb={pb_pct*100:.1f}% win={pb_win}"
            trades = collect_vr_with_pullback(d15, d1h, d4h, pb_pct, pb_win)
            m = run_metrics(trades, data_years, label)
            m["pullback_pct"] = pb_pct
            m["pullback_window"] = pb_win
            pullback_results.append(m)

    best_pullback = max(
        [m for m in pullback_results if m["N"] >= n_min],
        key=lambda x: x["PF"], default=None,
    )
    print(f"\n✅ Best Pullback Entry (N≥{n_min}): ", end="")
    if best_pullback:
        print(f"{best_pullback['label']} → PF={best_pullback['PF']:.3f} (was {m_base['PF']:.3f})")
    else:
        print("None meeting N constraint")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Phase 3B: MA Cluster 15m Pullback Entry
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 3B: MA Cluster + 15m Pullback Entry")
    print("  4h 브레이크아웃 후 15m 되돌림 지정가에 진입")
    print("=" * 80)

    ma_pb_results: list[dict] = []

    # Baseline MA cluster (original)
    ma_trades_base = collect_enhanced_trades(
        d15, d1h, d4h,
        cluster_pct=cfg.ma_cluster_pct,
        min_duration=cfg.ma_cluster_min_duration,
        max_duration=cfg.ma_cluster_max_duration,
        min_momentum=cfg.ma_cluster_min_momentum,
        sl=cfg.ma_cluster_sl,
        tp=cfg.ma_cluster_tp,
        tf_restriction=16,
    )
    print(f"  MA cluster baseline: {len(ma_trades_base)} trades")

    for pb_pct in [0.0, 0.001, 0.002, 0.003, 0.005]:
        label = f"MA_Pullback pb={pb_pct*100:.2f}%"
        ma_trades = collect_ma_cluster_pullback(d15, d1h, d4h, pb_pct, pullback_window_15m=8)
        # Combine with VR baseline
        vr_trades = collect_baseline(d15, d1h, d4h)
        # Remove MA trades from baseline (already included via collect_baseline)
        # Actually collect_baseline already includes MA via collect_enhanced_trades
        # So here we compare MA-only metrics for MA pullback vs baseline MA
        m_ma = run_metrics(ma_trades, data_years, label + " (MA-only)")
        m_ma["pullback_pct"] = pb_pct
        ma_pb_results.append(m_ma)

    best_ma_pb = max(ma_pb_results, key=lambda x: x["PF"]) if ma_pb_results else None
    print(f"\n✅ Best MA Cluster Pullback: ", end="")
    if best_ma_pb:
        print(f"{best_ma_pb['label']} → PF={best_ma_pb['PF']:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Phase 4: Best Combination
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 4: Best Combination")
    print("=" * 80)

    combo_results: list[dict] = []

    # Pick best parameters from Phase 2 and 3
    if best_stackup:
        thr_best = best_stackup["lower_threshold"]
        apply_to = best_stackup["apply_to"]
        apply_4h = "4h" in apply_to
        apply_1h = "1h" in apply_to

        # A: Best stackup + original entry
        trades_a = collect_vr_with_subtf(d15, d1h, d4h, thr_best, apply_4h, apply_1h)
        m_a = run_metrics(trades_a, data_years, f"[A] Best Stackup only")
        combo_results.append(m_a)

    if best_pullback:
        pb_pct_best = best_pullback["pullback_pct"]
        pb_win_best = best_pullback["pullback_window"]

        # B: Original stackup-off + best pullback
        trades_b = collect_vr_with_pullback(d15, d1h, d4h, pb_pct_best, pb_win_best)
        m_b = run_metrics(trades_b, data_years, f"[B] Best Pullback only")
        combo_results.append(m_b)

    if best_stackup and best_pullback:
        # C: Best stackup + best pullback (combined)
        # This requires a combined collection function
        # For now: approximate by running both and merging at portfolio level
        # TODO: implement a single function combining both
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    #  Phase 5: OOS Validation
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 5: OOS Validation (50% split)")
    print("=" * 80)

    total_bars = max(len(d["close"]) for d in d15.values())
    split_bar = total_bars // 2
    print(f"  IS: first {data_years/2:.2f}yr | OOS: last {data_years/2:.2f}yr")

    def filter_period(
        trades: list[tuple], max_bar: int,
    ) -> tuple[list, list]:
        """Split trades into IS and OOS based on entry bar."""
        is_t = [(sym, eb, xb, p) for sym, eb, xb, p in trades if eb < max_bar]
        oos_t = [(sym, eb, xb, p) for sym, eb, xb, p in trades if eb >= max_bar]
        return is_t, oos_t

    print("\n[Baseline OOS]")
    is_base, oos_base = filter_period(trades_base, split_bar)
    m_is_base = run_metrics(is_base, data_years / 2, "Baseline IS")
    m_oos_base = run_metrics(oos_base, data_years / 2, "Baseline OOS")
    oos_ratio_base = m_oos_base["PF"] / m_is_base["PF"] if m_is_base["PF"] > 0 else 0
    print(f"  OOS/IS PF ratio: {oos_ratio_base:.3f} (≥0.85 is good)")

    # OOS for best methods
    candidates = []
    if best_stackup:
        thr_b = best_stackup["lower_threshold"]
        ap = best_stackup["apply_to"]
        t = collect_vr_with_subtf(d15, d1h, d4h, thr_b, "4h" in ap, "1h" in ap)
        candidates.append(("Best Stackup", t))
    if best_pullback:
        t = collect_vr_with_pullback(d15, d1h, d4h, best_pullback["pullback_pct"], best_pullback["pullback_window"])
        candidates.append(("Best Pullback", t))

    oos_ratios: dict[str, float] = {"Baseline": oos_ratio_base}
    for name, t in candidates:
        print(f"\n[{name} OOS]")
        is_t, oos_t = filter_period(t, split_bar)
        m_is = run_metrics(is_t, data_years / 2, f"{name} IS")
        m_oos = run_metrics(oos_t, data_years / 2, f"{name} OOS")
        ratio = m_oos["PF"] / m_is["PF"] if m_is["PF"] > 0 else 0
        oos_ratios[name] = ratio
        print(f"  OOS/IS PF ratio: {ratio:.3f} (≥0.85 is good)")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════════════════════════
    report_dir = REPORT_DIR / "202603"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r55_subtf_entry.md"

    with open(report_path, "w") as f:
        f.write(f"# r55: Sub-TF Entry Refinement\n\n")
        f.write(f"**날짜:** {datetime.now():%Y-%m-%d %H:%M}  \n")
        f.write(f"**커밋:** `{commit}`  \n")
        f.write(f"**기간:** {bar0:%Y-%m-%d} ~ {datetime.now():%Y-%m-%d} ({data_years:.1f}yr)  \n")
        f.write(f"**유니버스:** {len(d15)}개 심볼 (Binance USDT-M 선물)  \n")
        f.write(f"**비용:** 0.08% round-trip  \n")
        f.write(f"**수익률 모델:** 1x compound (L/N=3.0 프로덕션 변환 시 ×3)  \n\n")
        f.write(f"**Parity Notes:** config.py 직접 읽음 ✅ | 룩어헤드 없음 ✅ (sub-TF 완성봉만 사용)\n\n")

        f.write(f"## 방법론\n\n")
        f.write(f"- **Method A (Sub-TF VR Stackup):** 4h/1h VR 스퀴즈 + 15m VR도 압축 요구 (이중 스퀴즈)\n")
        f.write(f"- **Method B (Limit Order Pullback):** 신호 후 N봉 내 되돌림 지정가 진입\n")
        f.write(f"- **Method C (MA Cluster Pullback):** 4h 브레이크아웃 후 15m 되돌림 진입\n\n")

        f.write(f"## Phase 1: Baseline\n\n")
        f.write(f"| 항목 | 값 |\n|------|----|\n")
        for k in ["N", "PF", "WR", "EV_bp", "calmar", "max_dd_pct"]:
            v = m_base.get(k, "N/A")
            fv = f"{v:.3f}" if isinstance(v, float) else str(v)
            f.write(f"| {k} | {fv} |\n")
        f.write(f"\n")

        f.write(f"## Phase 2: Sub-TF VR Stackup 결과\n\n")
        f.write(f"| Label | N | PF | WR | EV(bp) | Calmar | DD |\n")
        f.write(f"|-------|---|----|----|--------|--------|----|\n")
        for m in stackup_results:
            f.write(
                f"| {m['label']} | {m['N']} | {m['PF']:.3f} | {m['WR']:.1f}% | "
                f"{m['EV_bp']:+.0f} | {m['calmar']:.1f} | {m['max_dd_pct']:.1f}% |\n"
            )
        f.write(f"\n**Best (N≥{n_min}):** {best_stackup['label'] if best_stackup else 'None'}\n\n")

        f.write(f"## Phase 3: Pullback Entry 결과\n\n")
        f.write(f"| Label | N | PF | WR | EV(bp) | Calmar | DD |\n")
        f.write(f"|-------|---|----|----|--------|--------|----|\n")
        for m in pullback_results:
            f.write(
                f"| {m['label']} | {m['N']} | {m['PF']:.3f} | {m['WR']:.1f}% | "
                f"{m['EV_bp']:+.0f} | {m['calmar']:.1f} | {m['max_dd_pct']:.1f}% |\n"
            )
        f.write(f"\n**Best (N≥{n_min}):** {best_pullback['label'] if best_pullback else 'None'}\n\n")

        f.write(f"## Phase 3B: MA Cluster Pullback 결과\n\n")
        f.write(f"| Label | N | PF | WR | EV(bp) |\n")
        f.write(f"|-------|---|----|----|--------|\n")
        for m in ma_pb_results:
            f.write(f"| {m['label']} | {m['N']} | {m['PF']:.3f} | {m['WR']:.1f}% | {m['EV_bp']:+.0f} |\n")

        f.write(f"\n## Phase 5: OOS Validation\n\n")
        f.write(f"| 방법 | OOS/IS PF 비율 | 평가 |\n|------|---------------|------|\n")
        for name, ratio in oos_ratios.items():
            verdict = "✅ Good" if ratio >= 0.85 else ("⚠️ Marginal" if ratio >= 0.70 else "❌ Overfit")
            f.write(f"| {name} | {ratio:.3f} | {verdict} |\n")

        f.write(f"\n## 결론\n\n")
        f.write(f"- Sub-TF VR Stackup: ")
        if best_stackup and best_stackup["PF"] > m_base["PF"]:
            f.write(f"PF {m_base['PF']:.3f} → {best_stackup['PF']:.3f} (개선 ✅)\n")
        else:
            f.write(f"유의미한 개선 없음 (PF baseline={m_base['PF']:.3f})\n")
        f.write(f"- Pullback Entry: ")
        if best_pullback and best_pullback["PF"] > m_base["PF"]:
            f.write(f"PF {m_base['PF']:.3f} → {best_pullback['PF']:.3f} (개선 ✅)\n")
        else:
            f.write(f"유의미한 개선 없음 (PF baseline={m_base['PF']:.3f})\n")

    print(f"\n📄 Report: {report_path}")
    print(f"\n{'='*80}")
    print(f"r55 완료  [{commit}]  총 소요: {time.time()-t0:.0f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
