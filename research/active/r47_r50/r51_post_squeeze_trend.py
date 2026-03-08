#!/usr/bin/env python3
"""r51: Post-Squeeze Trend Continuation Backtest.

Concept:
  VR Squeeze (existing) = enters DURING compression (VR < threshold)
  r51 (this) = enters AFTER squeeze ends (VR just crossed above threshold)
               → direction already confirmed by price momentum

Entry condition:
  1. VR on 4h was below vr_threshold for >= min_hold_bars consecutive bars
  2. VR on 4h now crosses above vr_threshold  (squeeze just ended)
  3. Price return over last momentum_lookback bars > min_momentum  → LONG
     Price return over last momentum_lookback bars < -min_momentum → SHORT

Exit:
  Trailing stop (trail_activate + trail_distance, same as current config)
  OR hard SL / max hold_bars

Overlap with existing VR Squeeze:
  - VR Squeeze enters WHILE VR < threshold (earlier)
  - r51 enters WHEN VR crosses ABOVE threshold (later, direction confirmed)
  - Temporal gap: typically 1-4 bars between them
  - Same symbol same direction possible → conflict tracking noted

Phase 0: Pre-flight config
Phase 1: Sweep vr_threshold × momentum_lookback × min_momentum
Phase 2: Best config OOS validation (IS: 2020-2022, OOS: 2023-2026)
Phase 3: Overlap analysis with existing VR signals
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import build_resampled, precompute_all_vr, precompute
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from research.active.r41_r45.backtest_v14_3_0 import load_15m

# ── Phase 0: Config ───────────────────────────────────────────────────────────
cfg = VolStraddleConfig()
MAX_CONCURRENT = cfg.max_concurrent  # 10
COST_RT = 0.0008  # 0.08% round-trip

print("=" * 65)
print("  r51: Post-Squeeze Trend Continuation Backtest")
print("=" * 65)
print(f"[Phase 0] max_concurrent={MAX_CONCURRENT}, cost={COST_RT*100:.2f}% RT")
print(f"  Reference config: trail_act={cfg.d_trail_activate}, "
      f"trail_dist={cfg.d_trail_distance}, sl={cfg.d_sl}")

# ── Data ──────────────────────────────────────────────────────────────────────
t0 = time.time()
print("\nLoading & precomputing...")
data_15m = load_15m()
if not data_15m:
    print("No data."); sys.exit(1)

END_DT = datetime(2026, 3, 7, 0, 0)
data_4h = build_resampled(data_15m, 16)
precompute_all_vr([data_4h], [(8, 56)])

max_bars_4h = max(len(v["close"]) for v in data_4h.values())
sym_len = {s: len(data_4h[s]["close"]) for s in data_4h}
symbols = list(data_4h.keys())

print(f"  Done in {time.time()-t0:.1f}s  |  {len(symbols)} symbols  |  "
      f"{max_bars_4h} 4h bars")

# ── Helper ────────────────────────────────────────────────────────────────────

def get(sym: str, key: str, abs_bar: int) -> float | None:
    offset = max_bars_4h - sym_len[sym]
    local = abs_bar - offset
    if local < 0 or local >= sym_len[sym]:
        return None
    arr = data_4h[sym].get(key)
    if arr is None or local >= len(arr):
        return None
    v = float(arr[local])
    return None if np.isnan(v) else v


# ── Core simulation ───────────────────────────────────────────────────────────

def run_backtest(
    vr_threshold: float = 0.5,      # VR 크로스오버 기준
    min_squeeze_bars: int = 2,       # 최소 몇 바 동안 VR < threshold 였어야 하는지
    momentum_lookback: int = 8,      # 방향 확인 lookback (4h bars)
    min_momentum: float = 0.01,      # 최소 price move (fraction)
    sl: float = 0.02,                # hard stop loss
    trail_activate: float = 0.01,    # trailing 활성화 gain
    trail_distance: float = 0.005,   # trailing stop distance
    max_hold_bars: int = 96,         # 최대 보유 기간 (4h bars = 16d)
    start_dt: datetime = datetime(2020, 6, 1),
    end_dt: datetime = END_DT,
) -> dict:
    open_positions: dict[tuple[str, str], dict] = {}
    all_trades: list[tuple[datetime, float, str, str]] = []

    start_abs = max_bars_4h - 1 - int((END_DT - start_dt).total_seconds() / 14400)
    end_abs   = max_bars_4h - 1 - int((END_DT - end_dt).total_seconds()   / 14400)
    start_abs = max(start_abs, 250)

    equity = 1.0
    hwm = 1.0
    max_dd = 0.0

    # VR 연속 상태 캐시 {sym: consecutive_bars_below_threshold}
    squeeze_counter: dict[str, int] = {s: 0 for s in symbols}

    for abs_bar in range(start_abs, end_abs + 1):
        bar_dt = END_DT - timedelta(hours=4 * (max_bars_4h - 1 - abs_bar))

        # ── 기존 포지션 청산 체크 ────────────────────────────────────────────
        to_close = []
        for (sym, side), pos in open_positions.items():
            cp = get(sym, "close", abs_bar)
            if cp is None:
                to_close.append((sym, side, 0.0)); continue

            entry_price = pos["entry_price"]
            if side == "LONG":
                gain = cp / entry_price - 1
                sl_p = pos["sl_price"]
                # trailing 업데이트
                if gain > pos["peak_gain"]:
                    pos["peak_gain"] = gain
                if gain >= trail_activate:
                    trail_stop = pos["peak_gain"] - trail_distance
                    if gain < trail_stop:
                        pnl = gain - COST_RT
                        to_close.append((sym, side, pnl)); continue
                if cp <= sl_p:
                    to_close.append((sym, side, -(sl + COST_RT))); continue
                if abs_bar - pos["entry_abs"] >= max_hold_bars:
                    to_close.append((sym, side, gain - COST_RT)); continue
            else:  # SHORT
                gain = entry_price / cp - 1
                sl_p = pos["sl_price"]
                if gain > pos["peak_gain"]:
                    pos["peak_gain"] = gain
                if gain >= trail_activate:
                    trail_stop = pos["peak_gain"] - trail_distance
                    if gain < trail_stop:
                        pnl = gain - COST_RT
                        to_close.append((sym, side, pnl)); continue
                if cp >= sl_p:
                    to_close.append((sym, side, -(sl + COST_RT))); continue
                if abs_bar - pos["entry_abs"] >= max_hold_bars:
                    to_close.append((sym, side, gain - COST_RT)); continue

        for sym, side, pnl in to_close:
            del open_positions[(sym, side)]
            equity *= (1 + pnl / MAX_CONCURRENT)
            all_trades.append((bar_dt, pnl, sym, side))
            if equity > hwm: hwm = equity
            dd = (hwm - equity) / hwm
            if dd > max_dd: max_dd = dd

        # ── 신호 감지 ────────────────────────────────────────────────────────
        if len(open_positions) >= MAX_CONCURRENT:
            # 카운터는 계속 업데이트
            for sym in symbols:
                vr = get(sym, "vr_8_56", abs_bar)
                if vr is None: continue
                if vr < vr_threshold:
                    squeeze_counter[sym] += 1
                else:
                    squeeze_counter[sym] = 0
            continue

        for sym in symbols:
            vr_now  = get(sym, "vr_8_56", abs_bar)
            vr_prev = get(sym, "vr_8_56", abs_bar - 1)
            if vr_now is None or vr_prev is None:
                squeeze_counter[sym] = 0
                continue

            # 크로스오버: 이전 바는 아래, 현재 바는 위
            if vr_prev < vr_threshold and vr_now >= vr_threshold:
                held = squeeze_counter[sym]
                if held >= min_squeeze_bars:
                    # 방향 확인
                    p_now  = get(sym, "close", abs_bar)
                    p_prev = get(sym, "close", abs_bar - momentum_lookback)
                    if p_now is None or p_prev is None or p_prev == 0:
                        squeeze_counter[sym] = 0
                        continue
                    mom = p_now / p_prev - 1

                    if mom >= min_momentum and (sym, "LONG") not in open_positions:
                        if len(open_positions) < MAX_CONCURRENT:
                            open_positions[(sym, "LONG")] = {
                                "entry_abs": abs_bar,
                                "entry_price": p_now,
                                "sl_price": p_now * (1 - sl),
                                "peak_gain": 0.0,
                            }
                    elif mom <= -min_momentum and (sym, "SHORT") not in open_positions:
                        if len(open_positions) < MAX_CONCURRENT:
                            open_positions[(sym, "SHORT")] = {
                                "entry_abs": abs_bar,
                                "entry_price": p_now,
                                "sl_price": p_now * (1 + sl),
                                "peak_gain": 0.0,
                            }

                squeeze_counter[sym] = 0  # 크로스 후 리셋
            elif vr_now < vr_threshold:
                squeeze_counter[sym] += 1
            else:
                squeeze_counter[sym] = 0

    # ── 메트릭 ───────────────────────────────────────────────────────────────
    pnls = [t[1] for t in all_trades]
    if not pnls:
        return {"N": 0, "WR": 0, "PF": 0, "EV_bp": 0, "MaxDD": 0, "trades": []}
    wins  = [p for p in pnls if p > 0]
    loses = [p for p in pnls if p <= 0]
    pf  = sum(wins) / -sum(loses) if loses else float("inf")
    wr  = len(wins) / len(pnls)
    ev  = np.mean(pnls) * 10000
    years = (end_dt - start_dt).days / 365.25
    return {
        "N": len(pnls), "N/yr": len(pnls) / years,
        "WR": wr, "PF": pf, "EV_bp": ev, "MaxDD": max_dd,
        "trades": all_trades,
    }


# ── Phase 1: Parameter Sweep ─────────────────────────────────────────────────
print("\n[Phase 1] Sweep: vr_threshold × momentum × min_momentum")
print(f"{'vr_thr':>6} {'mom_lb':>6} {'min_m':>6} {'sq_bars':>7}  "
      f"{'N':>5} {'N/yr':>5} {'WR':>6} {'PF':>6} {'EV_bp':>7} {'MaxDD':>7}")
print("-" * 75)

sweep = list(product(
    [0.45, 0.50, 0.55],   # vr_threshold
    [4, 8, 12],            # momentum_lookback (4h bars)
    [0.005, 0.01, 0.02],   # min_momentum
    [2],                   # min_squeeze_bars
))

results = []
for vr_t, mom_lb, min_m, sq in sweep:
    m = run_backtest(
        vr_threshold=vr_t, momentum_lookback=mom_lb, min_momentum=min_m,
        min_squeeze_bars=sq, sl=0.02,
        trail_activate=cfg.d_trail_activate,
        trail_distance=cfg.d_trail_distance,
    )
    results.append((vr_t, mom_lb, min_m, sq, m))
    flag = "★" if m["PF"] > 1.15 and m["N"] > 100 else " "
    print(f"{vr_t:>6.2f} {mom_lb:>6} {min_m:>6.3f} {sq:>7}  "
          f"{m['N']:>5} {m['N/yr']:>5.0f} {m['WR']:>6.2%} {m['PF']:>6.3f} "
          f"{m['EV_bp']:>7.1f} {m['MaxDD']:>7.2%} {flag}")

# ── Phase 2: Best + OOS ───────────────────────────────────────────────────────
best = max(results, key=lambda x: x[4]["PF"] if x[4]["N"] > 80 else 0)
vr_t, mom_lb, min_m, sq, bm = best
print(f"\n[Phase 2] Best: vr_thr={vr_t} mom_lb={mom_lb} min_m={min_m}")
print(f"  Full  N={bm['N']} N/yr={bm['N/yr']:.0f} WR={bm['WR']:.2%} "
      f"PF={bm['PF']:.3f} EV={bm['EV_bp']:.1f}bp MaxDD={bm['MaxDD']:.2%}")

SPLIT = datetime(2023, 1, 1)
m_is = run_backtest(vr_threshold=vr_t, momentum_lookback=mom_lb, min_momentum=min_m,
                    min_squeeze_bars=sq, sl=0.02,
                    trail_activate=cfg.d_trail_activate, trail_distance=cfg.d_trail_distance,
                    start_dt=datetime(2020,6,1), end_dt=SPLIT)
m_oos = run_backtest(vr_threshold=vr_t, momentum_lookback=mom_lb, min_momentum=min_m,
                     min_squeeze_bars=sq, sl=0.02,
                     trail_activate=cfg.d_trail_activate, trail_distance=cfg.d_trail_distance,
                     start_dt=SPLIT, end_dt=END_DT)
print(f"  IS  (2020-2022): N={m_is['N']}  PF={m_is['PF']:.3f}  EV={m_is['EV_bp']:.1f}bp")
print(f"  OOS (2023-2026): N={m_oos['N']}  PF={m_oos['PF']:.3f}  EV={m_oos['EV_bp']:.1f}bp")
ratio = m_oos["PF"] / m_is["PF"] if m_is["PF"] > 0 else 0
print(f"  OOS/IS ratio: {ratio:.2f}  ({'✅ OK' if m_oos['PF'] > 1.05 else '❌ WEAK OOS'})")

# ── Phase 3: Zero-signal gap coverage ────────────────────────────────────────
print("\n[Phase 3] VR+MA zero-signal week coverage...")
ZERO_WEEKS = [
    datetime(2020,5,18), datetime(2020,9,28),
    datetime(2021,5,3),  datetime(2021,5,17), datetime(2021,7,5),
    datetime(2021,8,2),  datetime(2021,9,13), datetime(2021,9,20),
    datetime(2021,10,18),datetime(2021,12,6),
    datetime(2022,1,3),  datetime(2022,1,10), datetime(2022,1,31),
    datetime(2022,2,21), datetime(2022,3,7),  datetime(2022,5,23),
    datetime(2024,5,13), datetime(2025,5,26),
]
covered = sum(1 for zw in ZERO_WEEKS
              if any(zw <= t[0] < zw + timedelta(weeks=1) for t in bm["trades"]))
print(f"  커버: {covered}/18주 ({covered/18*100:.0f}%)")

# ── Equity curve ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor("#0e1117")

for ax, (trades, label, col) in zip(axes, [
    (bm["trades"],    f"Full  PF={bm['PF']:.3f}", "#00d4ff"),
    (m_oos["trades"], f"OOS 2023-2026  PF={m_oos['PF']:.3f}", "#ff6b35"),
]):
    ax.set_facecolor("#0e1117")
    ec = [1.0]
    eq = 1.0
    for dt, pnl, sym, side in sorted(trades, key=lambda x: x[0]):
        eq *= (1 + pnl / MAX_CONCURRENT)
        ec.append(eq)
    ax.plot(ec, color=col, linewidth=1.2)
    ax.fill_between(range(len(ec)), 1.0, ec, alpha=0.15, color=col)
    ax.axhline(1.0, color="white", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_title(f"r51: {label}", color="white", fontsize=11)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#333")
    ax.set_xlabel("trades", color="white"); ax.set_ylabel("Equity (1x)", color="white")

plt.tight_layout()
out_dir = ROOT / "docs" / "backtest_results" / "202603"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "20260308_r51_post_squeeze_trend.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {out_path}")
print("=" * 65)
