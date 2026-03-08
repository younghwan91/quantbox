#!/usr/bin/env python3
"""r49: 6년 신호 타임라인 시각화.

VR Squeeze + MA Cluster 진입 시점을 시간순으로 heatmap으로 표시.
매매 공백 구간 vs 밀집 구간을 한눈에 확인.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    build_resampled,
    precompute_all_vr,
    precompute,
    collect_trades,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from research.active.r41_r45.backtest_v14_3_0 import (
    load_15m,
    collect_d_trades_trailing,
    collect_enhanced_trades,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION,
    MA_MIN_MOMENTUM, MA_SL, MA_TP, MA_TF_RESTRICTION,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r31_r40.scaling_r38 import CFG_V13

# ── Config ────────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()
print(f"[Config] d_tp={cfg.d_tp}, d_sl={cfg.d_sl}, "
      f"trail_act={cfg.d_trail_activate}, trail_dist={cfg.d_trail_distance}")
print(f"         VR_LN={cfg.base_leverage/cfg.max_concurrent:.1f}, "
      f"MA_LN={cfg.ma_cluster_leverage/cfg.max_concurrent:.1f}")

# ── Data ──────────────────────────────────────────────────────────────────────
t0 = time.time()
print("\nLoading data...")
data_15m = load_15m()
if not data_15m:
    print("No data found. Exiting.")
    sys.exit(1)

max_bars = max(len(v["close"]) for v in data_15m.values())
# 기준 심볼 (bars 가장 많은 것)
ref_sym = max(data_15m, key=lambda s: len(data_15m[s]["close"]))
n_bars = len(data_15m[ref_sym]["close"])

# 타임스탬프 역산 (데이터 끝 = 2026-03-07 00:00 UTC 근사)
END_DT = datetime(2026, 3, 7, 0, 0)
bar_times = [END_DT - timedelta(minutes=15 * (n_bars - 1 - i)) for i in range(n_bars)]

print(f"  {len(data_15m)} symbols | {n_bars:,} bars | "
      f"{bar_times[0].date()} ~ {bar_times[-1].date()}")

print("Resampling & precomputing...")
data_1h = build_resampled(data_15m, 4)
data_4h = build_resampled(data_15m, 16)
precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
for ds in [data_15m, data_1h, data_4h]:
    precompute(ds)
    precompute_mas(ds)
print(f"  Done in {time.time()-t0:.1f}s")

# ── Collect trades ────────────────────────────────────────────────────────────
print("\nCollecting VR Squeeze trades...")
vr_trades = collect_d_trades_trailing(CFG_V13, data_15m, data_1h, data_4h)
print(f"  VR: {len(vr_trades)} trades")

print("Collecting MA Cluster trades...")
ma_only = collect_enhanced_trades(
    data_15m, data_1h, data_4h,
    cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
    max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
    sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
)
print(f"  MA: {len(ma_only)} trades")

# ── Convert bar index → datetime ───────────────────────────────────────────────
def bar_to_dt(sym: str, bar_idx: int) -> datetime:
    """bar_idx (15m bar, 0=oldest) → datetime."""
    sym_bars = len(data_15m[sym]["close"])
    # 심볼마다 길이 다름: 짧은 심볼은 뒤쪽 정렬
    offset = max_bars - sym_bars
    abs_bar = offset + bar_idx
    return END_DT - timedelta(minutes=15 * (max_bars - 1 - abs_bar))


vr_entries: list[datetime] = []
for sym, entry_bar, exit_bar, pnl in vr_trades:
    if sym in data_15m:
        vr_entries.append(bar_to_dt(sym, entry_bar))

ma_entries: list[datetime] = []
for sym, entry_bar, exit_bar, pnl in ma_only:
    if sym in data_15m:
        ma_entries.append(bar_to_dt(sym, entry_bar))

print(f"\nVR entries: {len(vr_entries)}, MA entries: {len(ma_entries)}")

# ── Weekly bucketing ──────────────────────────────────────────────────────────
START_DT = datetime(2020, 3, 9)  # 워밍업 이후

# 주 단위로 버킷
def to_week(dt: datetime) -> datetime:
    # 시간 컴포넌트 제거 후 해당 주 월요일 반환 (key 일관성 필수)
    return (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

vr_by_week: dict[datetime, int] = {}
for dt in vr_entries:
    if dt >= START_DT:
        w = to_week(dt)
        vr_by_week[w] = vr_by_week.get(w, 0) + 1

ma_by_week: dict[datetime, int] = {}
for dt in ma_entries:
    if dt >= START_DT:
        w = to_week(dt)
        ma_by_week[w] = ma_by_week.get(w, 0) + 1

# 전체 주 목록
all_weeks: list[datetime] = []
cur = to_week(START_DT)
end_week = to_week(END_DT)
while cur <= end_week:
    all_weeks.append(cur)
    cur += timedelta(weeks=1)

weeks = np.array(all_weeks)
vr_counts = np.array([vr_by_week.get(w, 0) for w in all_weeks])
ma_counts = np.array([ma_by_week.get(w, 0) for w in all_weeks])
total_counts = vr_counts + ma_counts

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(18, 14),
                          gridspec_kw={"height_ratios": [3, 2, 2, 1.5]})
fig.patch.set_facecolor("#0e1117")

colors = {"vr": "#00d4ff", "ma": "#ff6b35", "total": "#a8ff78", "zero": "#1a1a2e"}

# ── Panel 1: 누적 스택 바 (VR + MA by week) ──────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor("#0e1117")
ax1.bar(weeks, vr_counts, color=colors["vr"], alpha=0.85, label="VR Squeeze", width=5)
ax1.bar(weeks, ma_counts, bottom=vr_counts, color=colors["ma"], alpha=0.85,
        label="MA Cluster", width=5)

# 무신호 구간 음영
zero_mask = total_counts == 0
if zero_mask.any():
    in_gap = False
    gap_start = None
    for i, (w, z) in enumerate(zip(weeks, zero_mask)):
        if z and not in_gap:
            gap_start = w
            in_gap = True
        elif not z and in_gap:
            ax1.axvspan(gap_start, w, alpha=0.15, color="red", label="_nolegend_")
            in_gap = False
    if in_gap:
        ax1.axvspan(gap_start, weeks[-1], alpha=0.15, color="red")

ax1.set_xlim(weeks[0], weeks[-1])
ax1.set_ylabel("Entries / Week", color="white", fontsize=11)
ax1.set_title("6-Year Signal Timeline (2020-03 ~ 2026-03)  |  VR Squeeze + MA Cluster",
              color="white", fontsize=14, fontweight="bold", pad=12)
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#333")
ax1.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white", fontsize=10)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.yaxis.set_tick_params(labelcolor="white")

# ── Panel 2: 4주 롤링 평균 (추세) ────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#0e1117")
window = 4
rolling_total = np.convolve(total_counts, np.ones(window)/window, mode="same")
rolling_vr    = np.convolve(vr_counts,    np.ones(window)/window, mode="same")
rolling_ma    = np.convolve(ma_counts,    np.ones(window)/window, mode="same")

ax2.fill_between(weeks, rolling_vr, alpha=0.4, color=colors["vr"])
ax2.fill_between(weeks, rolling_vr, rolling_vr + rolling_ma, alpha=0.4, color=colors["ma"])
ax2.plot(weeks, rolling_total, color=colors["total"], linewidth=1.5, label="4-Week Rolling Sum")
ax2.axhline(np.mean(total_counts), color="white", linestyle="--", linewidth=0.8,
            alpha=0.6, label=f"Weekly Avg {np.mean(total_counts):.1f}")

ax2.set_xlim(weeks[0], weeks[-1])
ax2.set_ylabel("Rolling Avg Entries/Week", color="white", fontsize=10)
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#333")
ax2.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ── Panel 3: 연도별 월 heatmap ────────────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor("#0e1117")

# 월별 집계
from collections import defaultdict
monthly: dict[tuple[int,int], int] = defaultdict(int)
for dt in vr_entries + ma_entries:
    if dt >= START_DT:
        monthly[(dt.year, dt.month)] += 1

years = list(range(2020, 2027))
months = list(range(1, 13))
grid = np.zeros((len(years), 12))
for (y, m), cnt in monthly.items():
    if y in years:
        grid[years.index(y), m-1] = cnt

im = ax3.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax3.set_yticks(range(len(years)))
ax3.set_yticklabels(years, color="white", fontsize=9)
ax3.set_xticks(range(12))
ax3.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"],
                    color="white", fontsize=9)
ax3.set_title("Monthly Entry Count Heatmap", color="white", fontsize=11, pad=6)
ax3.tick_params(colors="white")
ax3.spines[:].set_color("#333")
plt.colorbar(im, ax=ax3, orientation="vertical", pad=0.02).ax.tick_params(colors="white")

# 셀에 숫자 표시
for i in range(len(years)):
    for j in range(12):
        v = int(grid[i, j])
        if v > 0:
            ax3.text(j, i, str(v), ha="center", va="center",
                     color="black" if v > grid.max()*0.5 else "white",
                     fontsize=7, fontweight="bold")

# ── Panel 4: 무신호 연속 기간 분포 ──────────────────────────────────────────
ax4 = axes[3]
ax4.set_facecolor("#0e1117")

# 연속 0-count 주 계산
gaps = []
gap_len = 0
for c in total_counts:
    if c == 0:
        gap_len += 1
    else:
        if gap_len > 0:
            gaps.append(gap_len)
        gap_len = 0
if gap_len > 0:
    gaps.append(gap_len)

if gaps:
    bins = np.arange(0.5, max(gaps)+1.5, 1)
    ax4.hist(gaps, bins=bins, color=colors["vr"], alpha=0.8, edgecolor="#0e1117")
    ax4.axvline(np.mean(gaps), color="yellow", linestyle="--", linewidth=1.2,
                label=f"Avg Gap {np.mean(gaps):.1f}w")
    ax4.axvline(np.percentile(gaps, 90), color="red", linestyle="--", linewidth=1.0,
                label=f"P90 {np.percentile(gaps, 90):.0f}w Gap")
    ax4.set_xlabel("Consecutive No-Signal Weeks", color="white", fontsize=10)
    ax4.set_ylabel("Frequency", color="white", fontsize=10)
    ax4.set_title("No-Signal Gap Distribution", color="white", fontsize=11, pad=6)
    ax4.tick_params(colors="white")
    ax4.spines[:].set_color("#333")
    ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
else:
    ax4.text(0.5, 0.5, "No Signal Gaps Found", transform=ax4.transAxes,
             ha="center", va="center", color="white", fontsize=12)

# ── 통계 출력 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  총 진입: VR={len(vr_entries)}, MA={len(ma_entries)}, 합계={len(vr_entries)+len(ma_entries)}")
print(f"  주간 평균: {np.mean(total_counts):.1f}건  중앙값: {np.median(total_counts):.1f}건")
print(f"  무신호 주: {(total_counts==0).sum()}주 / {len(total_counts)}주 총 "
      f"({(total_counts==0).mean()*100:.1f}%)")
if gaps:
    print(f"  최장 연속 무신호: {max(gaps)}주 ({max(gaps)*7/30:.1f}개월)")
    print(f"  평균 공백: {np.mean(gaps):.1f}주")

# 연도별 통계
print(f"\n  연도별 진입:")
for y in range(2020, 2027):
    cnt = sum(monthly.get((y,m),0) for m in range(1,13))
    bar = "█" * (cnt // 5)
    print(f"    {y}: {cnt:4d}건  {bar}")
print("="*60)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 1])
out_dir = ROOT / "docs" / "backtest_results" / "202603"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "20260308_signal_timeline_6yr.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out_path}")
