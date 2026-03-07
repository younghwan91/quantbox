#!/usr/bin/env python3
"""r48: DD Halt Threshold Analysis — v14.4.x (Option D: r47 exits + TBR OFF).

목적: 프로덕션 배포 버전의 자연 낙폭 분포를 측정하고
      max_drawdown_pct 적정값을 결정한다.

Phase 0: Config pre-flight
Phase 1: Baseline 검증 (Option D 재현)
Phase 2: Production equity curve MaxDD 분포
Phase 3: 세션 단위 DD 분포 (rolling window)
Phase 4: 권고 임계값 결정
"""
from __future__ import annotations

import subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import (
    bootstrap_pf,
    portfolio_sim_with_details,
    precompute_all_vr,
    precompute,
    build_resampled,
)
from research.active.r31_r40.scaling_r38 import CFG_V13
from research.active.r41_r45.backtest_v14_3_0 import (
    load_15m,
    collect_enhanced_trades,
    extended_metrics,
    VR_LN, MA_LN,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION,
    MA_MIN_MOMENTUM, MA_SL, MA_TP, MA_TF_RESTRICTION,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r47_r50.r47_exit_ab_test import (
    collect_d_trades_custom_trail,
    collect_ma_cluster_trailing,
)

# ── Phase 0: Config ───────────────────────────────────────────────────────────
cfg = VolStraddleConfig()
print("=" * 60)
print("[Phase 0] Config (v14.4.x, Option D deployed)")
print(f"  d_tp={cfg.d_tp}  d_sl={cfg.d_sl}")
print(f"  d_trail_activate={cfg.d_trail_activate}  d_trail_distance={cfg.d_trail_distance}")
print(f"  base_leverage={cfg.base_leverage}  max_concurrent={cfg.max_concurrent}")
print(f"  VR_LN={VR_LN:.1f}  MA_LN={MA_LN:.1f}")
print(f"  f5_tbr_enabled={cfg.f5_tbr_enabled}  (must be False for Option D)")
assert not cfg.f5_tbr_enabled, "TBR must be OFF for deployed config"
print(f"  max_drawdown_pct (current halt)={cfg.max_drawdown_pct:.0%}")

def get_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[Data] Loading 15m OHLCV...")
t0 = time.time()
data_15m = load_15m()
DATA_YEARS = 5.47  # 2020-01-01 to 2025-06-30 approx
print(f"  {len(data_15m)} symbols  ({time.time()-t0:.1f}s)")

data_1h = build_resampled(data_15m, 4)
data_4h = build_resampled(data_15m, 16)
precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
for ds in [data_15m, data_1h, data_4h]:
    precompute(ds, vr_sp=8, vr_lp=56)
precompute_mas(data_15m)
precompute_mas(data_1h)
precompute_mas(data_4h)

avg_bars = int(np.mean([len(v["close"]) for v in data_15m.values()]))
DATA_YEARS = avg_bars / (4 * 24 * 365)

# ── Phase 1: Option D baseline ────────────────────────────────────────────────
print("\n[Phase 1] Option D baseline (r47 exits + TBR OFF)...")
R47_TA = 0.005; R47_TD = 0.002
R47_MA_TA = 0.04; R47_MA_TD = 0.005

d_trades_D = collect_d_trades_custom_trail(
    CFG_V13, data_15m, data_1h, data_4h,
    trail_activate=R47_TA, trail_distance=R47_TD,
)
ma_trades_D = collect_ma_cluster_trailing(
    data_15m, data_1h, data_4h,
    cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
    max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
    sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    trail_activate=R47_MA_TA, trail_distance=R47_MA_TD,
)
comb_D = d_trades_D + ma_trades_D
pnl_D, _ = portfolio_sim_with_details(comb_D, 10)
blended_ln = (len(d_trades_D) * VR_LN + len(ma_trades_D) * MA_LN) / max(len(comb_D), 1)
mD = extended_metrics(pnl_D, DATA_YEARS, ln_leverage=blended_ln)

print(f"  N={mD['N']}  PF={mD['PF']:.3f}  WR={mD['WR']:.1f}%  EV={mD['EV_bp']:+.1f}bp")
print(f"  MaxDD(1x)={mD['max_dd_pct']:.1f}%  MaxDD(prod L/N={blended_ln:.2f})={mD['prod_max_dd_pct']:.1f}%")

# Phase 1 validation (±2% vs r47b_comparison baseline)
BASELINE_N_D = 4097; BASELINE_PF_D = 1.521
n_drift = abs(mD['N'] - BASELINE_N_D) / BASELINE_N_D
pf_drift = abs(mD['PF'] - BASELINE_PF_D) / BASELINE_PF_D
if n_drift > 0.02 or pf_drift > 0.02:
    print(f"  ⚠️  Drift: N={mD['N']} vs {BASELINE_N_D} ({n_drift:.1%}), PF={mD['PF']:.3f} vs {BASELINE_PF_D:.3f} ({pf_drift:.1%})")
else:
    print(f"  ✅ Baseline verified (N drift={n_drift:.1%}, PF drift={pf_drift:.1%})")

# ── Phase 2: Production equity curve — full DD distribution ───────────────────
print("\n[Phase 2] Production DD distribution")
prod_pnl = pnl_D * blended_ln
prod_equity = np.cumprod(1.0 + prod_pnl)
prod_peaks  = np.maximum.accumulate(prod_equity)
prod_dd_arr = (prod_peaks - prod_equity) / prod_peaks * 100  # % at each trade

# Distribution of drawdown values across all trades
pcts = [50, 75, 90, 95, 99, 100]
print(f"  Drawdown at each trade-step (percentiles):")
for p in pcts:
    print(f"    P{p:3d}: {np.percentile(prod_dd_arr, p):5.1f}%")
print(f"  → Global MaxDD (prod): {prod_dd_arr.max():.1f}%")

# ── Phase 3: Session-level DD (rolling 192-bar ≈ 48h windows) ─────────────────
print("\n[Phase 3] Session-level DD distribution")
print("  (Simulating 'new session start' every N trades ≈ 48h window)")

# Approximate trades per 48h: strategy averages ~tpd × 2 trades/session
TPD = mD['tpd']  # trades per day
TPW = int(round(TPD * 2))  # trades in 48h session window
TPW = max(TPW, 10)

session_max_dds = []
for start in range(0, len(prod_pnl) - TPW, TPW // 2):  # 50% overlap
    window = prod_pnl[start:start + TPW]
    eq = np.cumprod(1.0 + window)
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / pk * 100
    session_max_dds.append(float(dd.max()))

session_max_dds = np.array(session_max_dds)
print(f"  Window size: {TPW} trades (~48h)  |  Sessions simulated: {len(session_max_dds)}")
print(f"  Session MaxDD distribution:")
for p in [50, 75, 90, 95, 99]:
    print(f"    P{p:3d}: {np.percentile(session_max_dds, p):5.1f}%")
print(f"  Worst session: {session_max_dds.max():.1f}%")

# Halt interference: % of sessions where current 10% halt would trigger
triggers_10 = np.sum(session_max_dds >= 10.0)
triggers_20 = np.sum(session_max_dds >= 20.0)
triggers_30 = np.sum(session_max_dds >= 30.0)
triggers_40 = np.sum(session_max_dds >= 40.0)
n_sess = len(session_max_dds)
print(f"\n  Halt trigger rate per threshold:")
print(f"    10%: {triggers_10}/{n_sess} sessions ({triggers_10/n_sess:.1%}) ← CURRENT")
print(f"    20%: {triggers_20}/{n_sess} sessions ({triggers_20/n_sess:.1%})")
print(f"    30%: {triggers_30}/{n_sess} sessions ({triggers_30/n_sess:.1%})")
print(f"    40%: {triggers_40}/{n_sess} sessions ({triggers_40/n_sess:.1%})")

# ── Phase 4: Recommendation ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Phase 4] RECOMMENDATION")
p90 = float(np.percentile(session_max_dds, 90))
p95 = float(np.percentile(session_max_dds, 95))
p99 = float(np.percentile(session_max_dds, 99))

# Halt should only trigger in catastrophic scenarios (P99+)
# Round up to nearest 5%
def round_up5(x): return int(np.ceil(x / 5.0) * 5)

recommended = round_up5(p99)
print(f"""
  전략 자연 낙폭:
    Global MaxDD (prod): {prod_dd_arr.max():.1f}%
    Session P90:         {p90:.1f}%  → 10%에 1번은 이 정도 DD 발생
    Session P95:         {p95:.1f}%  → 20%에 1번
    Session P99:         {p99:.1f}%  → 100%에 1번 (이상현상 경계)

  현재 halt=10%: {triggers_10/n_sess:.0%} 세션에서 발동 → 정상 운영 중 반복 차단
  권고 halt={recommended}%: 실제 비정상 구간에서만 발동 (P99 기준)

  → max_drawdown_pct = {recommended/100:.2f}  (config.py or .env)
""")

commit = get_commit()
print(f"Commit: {commit}  |  Data: 2020~2025 ({DATA_YEARS:.2f}yr)  |  LN={blended_ln:.2f}")
