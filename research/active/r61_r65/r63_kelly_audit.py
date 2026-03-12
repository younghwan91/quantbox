"""r63: Half-Kelly Leverage Audit — VR vs MA 레버리지 정합성 검증.

목적: config의 base_leverage(30x), ma_cluster_leverage(21x)가
현재 실제 WR/RR에서 Half-Kelly 공식으로 역산됐을 때 여전히 유효한지 확인.

공식: f* = WR - (1-WR)/RR
      L  = (f*/2) × max_concurrent / SL
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

cfg = VolStraddleConfig()

# ── Import r57 data/trade functions ──────────────────────────────────────────
from research.active.r51_r60.r57_combo_30m_pullback import (
    build_all_tf,
    collect_d_pullback,
    collect_ma,
    collect_s_trailing,
    load_syms,
)
from quantbox.backtest.trade_sim import compute_metrics, portfolio_sim

# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: Config 확인
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 0: 현재 config 레버리지 파라미터")
print("=" * 70)
VR_LEV = cfg.base_leverage        # 30x
MA_LEV = cfg.ma_cluster_leverage  # 21x
VR_SL  = cfg.d_sl                 # 1%
MA_SL  = cfg.ma_cluster_sl        # 2%
N_SLOTS = cfg.max_concurrent      # 10

# 역산: 내재 f*/2
vr_fstar2 = VR_LEV * VR_SL / N_SLOTS
ma_fstar2 = MA_LEV * MA_SL / N_SLOTS

print(f"  VR  leverage={VR_LEV}x  SL={VR_SL*100:.1f}%  → implicit f*/2 = {vr_fstar2*100:.2f}%")
print(f"  MA  leverage={MA_LEV}x  SL={MA_SL*100:.1f}%  → implicit f*/2 = {ma_fstar2*100:.2f}%")
print(f"  N_slots = {N_SLOTS}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: 데이터 로드 & 거래 수집
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 1: 데이터 로드 (60 syms, 6yr)")
print("=" * 70)
d15 = load_syms(min_years=5.0)
d30, d1h, d4h = build_all_tf(d15)
DATA_YEARS = max(len(d["close"]) for d in d15.values()) / (365 * 24 * 4)
print(f"  Data years: {DATA_YEARS:.2f}")
print()

# VR trades (directional D_mom8 + straddle S_d8, with 30m)
vr_trades = (
    collect_d_pullback(d15, d1h, d4h, 0.0, 5, d30)
    + collect_s_trailing(d15, d1h, d4h, d30)
)
# MA cluster trades
ma_trades = collect_ma(d15, d1h, d4h, tf_restriction=16)

print(f"  VR trades collected: {len(vr_trades):,}")
print(f"  MA trades collected: {len(ma_trades):,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: 각 전략별 WR/RR 측정
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 2: 전략별 WR / RR 측정 (portfolio_sim은 별도 적용)")
print("=" * 70)

def pnl_from_trades(trades: list) -> np.ndarray:
    """거래 목록에서 pnl 배열 추출."""
    return np.array([t[3] for t in trades], dtype=np.float64)

vr_pnl_raw  = pnl_from_trades(vr_trades)
ma_pnl_raw  = pnl_from_trades(ma_trades)

m_vr = compute_metrics(vr_pnl_raw, DATA_YEARS)
m_ma = compute_metrics(ma_pnl_raw, DATA_YEARS)

print(f"  [VR] N={m_vr['N']:,}  WR={m_vr['WR']:.1f}%  RR={m_vr['RR']:.3f}")
print(f"       avg_win={m_vr['avg_win_pct']:.2f}%  avg_loss={m_vr['avg_loss_pct']:.2f}%  EV={m_vr['EV_bp']:.1f}bp")
print(f"  [MA] N={m_ma['N']:,}  WR={m_ma['WR']:.1f}%  RR={m_ma['RR']:.3f}")
print(f"       avg_win={m_ma['avg_win_pct']:.2f}%  avg_loss={m_ma['avg_loss_pct']:.2f}%  EV={m_ma['EV_bp']:.1f}bp")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Half-Kelly 재계산
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 3: Half-Kelly 재계산")
print("=" * 70)

def half_kelly(wr: float, rr: float) -> float:
    """f* = WR - (1-WR)/RR, half-Kelly = f*/2."""
    wr_f = wr / 100.0
    fstar = wr_f - (1 - wr_f) / rr
    return max(fstar / 2, 0.0)

def kelly_leverage(fstar2: float, sl: float, n_slots: int) -> float:
    """L = (f*/2) × N / SL."""
    return fstar2 * n_slots / sl

# Original values (from config comment)
vr_orig_fstar2 = half_kelly(25.9, 3.75)
ma_orig_fstar2 = half_kelly(38.6, 2.05)
vr_orig_lev = kelly_leverage(vr_orig_fstar2, VR_SL, N_SLOTS)
ma_orig_lev = kelly_leverage(ma_orig_fstar2, MA_SL, N_SLOTS)

# Current values (from actual backtest)
vr_curr_fstar2 = half_kelly(m_vr['WR'], m_vr['RR'])
ma_curr_fstar2 = half_kelly(m_ma['WR'], m_ma['RR'])
vr_curr_lev = kelly_leverage(vr_curr_fstar2, VR_SL, N_SLOTS)
ma_curr_lev = kelly_leverage(ma_curr_fstar2, MA_SL, N_SLOTS)

print("  ┌─────────────────────────────────────────────────────────────────")
print("  │ 원본 Kelly (config 주석 기준, 20260304)")
print(f"  │  VR: WR=25.9%  RR=3.75  → f*={vr_orig_fstar2*2*100:.2f}%  f*/2={vr_orig_fstar2*100:.2f}%  → L={vr_orig_lev:.1f}x (config: {VR_LEV}x)")
print(f"  │  MA: WR=38.6%  RR=2.05  → f*={ma_orig_fstar2*2*100:.2f}%  f*/2={ma_orig_fstar2*100:.2f}%  → L={ma_orig_lev:.1f}x (config: {MA_LEV}x)")
print("  ├─────────────────────────────────────────────────────────────────")
print("  │ 현재 Kelly (v14.16.0 실제 WR/RR)")
print(f"  │  VR: WR={m_vr['WR']:.1f}%  RR={m_vr['RR']:.3f}  → f*={vr_curr_fstar2*2*100:.2f}%  f*/2={vr_curr_fstar2*100:.2f}%  → L={vr_curr_lev:.1f}x (config: {VR_LEV}x)")
print(f"  │  MA: WR={m_ma['WR']:.1f}%  RR={m_ma['RR']:.3f}  → f*={ma_curr_fstar2*2*100:.2f}%  f*/2={ma_curr_fstar2*100:.2f}%  → L={ma_curr_lev:.1f}x (config: {MA_LEV}x)")
print("  └─────────────────────────────────────────────────────────────────")
print()

vr_gap = abs(vr_curr_lev - VR_LEV) / VR_LEV * 100
ma_gap = abs(ma_curr_lev - MA_LEV) / MA_LEV * 100
print(f"  VR 괴리: {VR_LEV}x(config) vs {vr_curr_lev:.1f}x(Kelly) → {vr_gap:.1f}%")
print(f"  MA 괴리: {MA_LEV}x(config) vs {ma_curr_lev:.1f}x(Kelly) → {ma_gap:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: CW-Tier 멀티플라이어 정합성
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 4: CW-Tier 멀티플라이어 정합성")
print("=" * 70)
print("  CW-Tier는 백테스트에 conviction 점수가 없어 per-score WR 분리 불가.")
print("  → config 멀티플라이어가 'risk ∝ Kelly fraction' 원칙을 따르는지 분석.")
print()
print(f"  cw_multipliers = {cfg.cw_multipliers}")
print(f"  min_conviction_score = {cfg.min_conviction_score}")
print()

# 멀티플라이어가 선형이면 Kelly 비례 원칙 충족 (같은 SL, 다른 conviction)
mults = cfg.cw_multipliers
# 기준: score=1 → 1.0x
# score=0 → 0.5x (절반), score=2 → 1.5x, score=3 → 2.0x, score=4 → 2.5x
base_mult = mults[1] if len(mults) > 1 else 1.0
ratios = [m / base_mult for m in mults]
print(f"  score  mult    ratio vs score-1")
for i, (m, r) in enumerate(zip(mults, ratios)):
    flag = "← baseline" if i == 1 else ""
    print(f"    {i}    {m:.1f}x   {r:.2f}x  {flag}")
print()
is_linear = all(abs(ratios[i+1] - ratios[i] - (ratios[1] - ratios[0])) < 0.01
                for i in range(len(ratios)-1))
print(f"  선형 비례 여부: {'✅ YES (Kelly 일관)' if is_linear else '⚠️ 비선형'}")
print(f"  범위: {mults[0]}x ~ {mults[-1]}x (max/min = {mults[-1]/mults[0]:.1f}x)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: 결론
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 5: 결론 및 권장 조치")
print("=" * 70)

THRESHOLD = 20.0  # 괴리 20% 이상이면 조정 필요

def verdict(gap: float, label: str, config_lev: float, kelly_lev: float):
    if gap < 10.0:
        status = "✅ OK"
        action = "현행 유지"
    elif gap < THRESHOLD:
        status = "⚠️ 경고"
        action = f"모니터링 권장 (Kelly 제안: {kelly_lev:.0f}x)"
    else:
        status = "❌ 조정 필요"
        action = f"config 수정 권장: {config_lev}x → {kelly_lev:.0f}x"
    print(f"  [{label}] 괴리={gap:.1f}% {status} → {action}")

verdict(vr_gap, "VR ", VR_LEV, vr_curr_lev)
verdict(ma_gap, "MA ", MA_LEV, ma_curr_lev)

print()
print("  ※ WR 변화 설명:")
print(f"     원본 VR WR=25.9% vs 현재 {m_vr['WR']:.1f}%: Combo2(s_vr=0.57, dur=3)로")
print(f"     S_d8 straddle이 추가돼 WR이 올라간 것 (straddle은 long+short 동시)")
print(f"     원본 MA WR=38.6% vs 현재 {m_ma['WR']:.1f}%: SL/TP/trailing 파라미터 변경")
print()
if vr_gap < THRESHOLD and ma_gap < THRESHOLD:
    print("  → 전략적 결론: 두 전략 모두 Half-Kelly 범위 내 (±20% 이내).")
    print("     현행 레버리지 유지. 다음 대규모 파라미터 변경 시 재검토.")
else:
    print("  → 전략적 결론: 조정 필요한 항목 있음 (위 ❌ 참조).")
