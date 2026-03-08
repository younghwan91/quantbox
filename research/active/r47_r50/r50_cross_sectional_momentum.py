#!/usr/bin/env python3
"""r50: Cross-Sectional Momentum Backtest — v2.

Improvements over v1:
- Sharpe-based ranking (return/vol) instead of raw return
- BTC trend filter (long-only above MA, short-only below)
- VR expansion filter (only symbols with VR > threshold = post-squeeze)
- Conflict-awareness flag (skip symbols active in VR/MA — estimated)

Phase 0: Pre-flight config
Phase 1: Sweep ranking_mode × filter combos
Phase 2: Best config detail + OOS 50% split
Phase 3: Zero-signal gap coverage
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
from research.active.r41_r45.r44c_ma_confluence import precompute_mas

# ── Phase 0: Config ───────────────────────────────────────────────────────────
cfg = VolStraddleConfig()
MAX_CONCURRENT = cfg.max_concurrent
COST_RT = 0.0008

print("=" * 65)
print("  r50 v2: Cross-Sectional Momentum  (Sharpe + Filters)")
print("=" * 65)
print(f"[Phase 0] max_concurrent={MAX_CONCURRENT}, cost={COST_RT*100:.2f}% RT")

# ── Data ──────────────────────────────────────────────────────────────────────
t0 = time.time()
print("\nLoading & precomputing...")
data_15m = load_15m()
if not data_15m:
    print("No data. Exiting."); sys.exit(1)

END_DT = datetime(2026, 3, 7, 0, 0)
data_4h = build_resampled(data_15m, 16)

# VR 계산 (expansion filter용)
precompute_all_vr([data_4h], [(8, 56)])

max_bars_4h = max(len(v["close"]) for v in data_4h.values())
sym_len: dict[str, int] = {s: len(data_4h[s]["close"]) for s in data_4h}
print(f"  Done in {time.time()-t0:.1f}s  |  {len(data_4h)} symbols")

# ── Helper: bar index → absolute index ───────────────────────────────────────

def get_arr(sym: str, key: str, abs_bar: int):
    offset = max_bars_4h - sym_len[sym]
    local = abs_bar - offset
    if local < 0 or local >= sym_len[sym]:
        return None
    arr = data_4h[sym].get(key)
    if arr is None or local >= len(arr):
        return None
    v = arr[local]
    return None if (np.isnan(v) if isinstance(v, float) else False) else float(v)

def get_vr(sym: str, abs_bar: int) -> float | None:
    return get_arr(sym, "vr_8_56", abs_bar)

# BTC MA200 (4h bars) 계산
btc_close = data_4h.get("BTCUSDT", {}).get("close", np.array([]))
btc_ma200 = np.full(len(btc_close), np.nan)
for i in range(200, len(btc_close)):
    btc_ma200[i] = btc_close[i - 200:i].mean()

def btc_trend(abs_bar: int) -> int:
    """+1 = BTC above MA200, -1 = below, 0 = unknown."""
    offset = max_bars_4h - sym_len.get("BTCUSDT", max_bars_4h)
    local = abs_bar - offset
    if local < 200 or local >= len(btc_close):
        return 0
    return 1 if btc_close[local] > btc_ma200[local] else -1

# ── Core simulation ───────────────────────────────────────────────────────────

def run_backtest(
    lookback_bars: int = 168,
    top_k: int = 5,
    hold_bars: int = 48,
    sl: float = 0.05,
    tp: float = 0.10,
    rebalance_bars: int = 24,
    ranking_mode: str = "sharpe",   # "return" | "sharpe"
    btc_filter: bool = False,       # long only when BTC>MA200, short only when below
    vr_filter: float = 0.0,         # skip symbols with VR < threshold (not yet expanded)
    start_dt: datetime = datetime(2020, 6, 1),
    end_dt: datetime = END_DT,
) -> dict:
    symbols = list(data_4h.keys())
    open_positions: dict[tuple[str, str], tuple[int, float, float, float]] = {}
    all_trades: list[tuple[datetime, float, str, str]] = []

    ref_sym = max(data_4h, key=lambda s: sym_len[s])
    start_abs = max_bars_4h - 1 - int((END_DT - start_dt).total_seconds() / 14400)
    end_abs   = max_bars_4h - 1 - int((END_DT - end_dt).total_seconds()   / 14400)
    start_abs = max(start_abs, 200 + lookback_bars)

    equity = 1.0
    hwm = 1.0
    max_dd = 0.0
    last_rebalance = start_abs

    for abs_bar in range(start_abs, end_abs + 1):
        bar_dt = END_DT - timedelta(hours=4 * (max_bars_4h - 1 - abs_bar))

        # ── 포지션 청산 체크 ─────────────────────────────────────────────────
        to_close = []
        for (sym, side), (entry_abs, ep, sl_p, tp_p) in open_positions.items():
            cp = get_arr(sym, "close", abs_bar)
            if cp is None:
                to_close.append((sym, side, 0.0)); continue
            if side == "LONG":
                if   cp <= sl_p: pnl = sl_p/ep - 1 - COST_RT
                elif cp >= tp_p: pnl = tp_p/ep - 1 - COST_RT
                elif abs_bar - entry_abs >= hold_bars: pnl = cp/ep - 1 - COST_RT
                else: continue
            else:
                if   cp >= sl_p: pnl = -(cp/ep - 1) - COST_RT
                elif cp <= tp_p: pnl = -(cp/ep - 1) - COST_RT
                elif abs_bar - entry_abs >= hold_bars: pnl = -(cp/ep - 1) - COST_RT
                else: continue
            to_close.append((sym, side, pnl))

        for sym, side, pnl in to_close:
            del open_positions[(sym, side)]
            equity *= (1 + pnl / MAX_CONCURRENT)
            all_trades.append((bar_dt, pnl, sym, side))
            if equity > hwm: hwm = equity
            dd = (hwm - equity) / hwm
            if dd > max_dd: max_dd = dd

        # ── 리밸런싱 ────────────────────────────────────────────────────────
        if abs_bar - last_rebalance < rebalance_bars:
            continue
        last_rebalance = abs_bar

        trend = btc_trend(abs_bar)

        # 랭킹 계산
        scores: dict[str, float] = {}
        for sym in symbols:
            prices = [get_arr(sym, "close", abs_bar - i) for i in range(lookback_bars)]
            prices = [p for p in prices if p is not None]
            if len(prices) < lookback_bars * 0.8:
                continue
            prices = np.array(prices[::-1])  # oldest first
            rets = np.diff(prices) / prices[:-1]

            # VR filter: skip if VR below threshold (still compressed)
            if vr_filter > 0:
                vr = get_vr(sym, abs_bar)
                if vr is None or vr < vr_filter:
                    continue

            if ranking_mode == "sharpe":
                if rets.std() < 1e-8: continue
                scores[sym] = rets.mean() / rets.std()
            else:
                scores[sym] = prices[-1] / prices[0] - 1

        if len(scores) < top_k * 2:
            continue

        ranked = sorted(scores, key=scores.get, reverse=True)
        long_syms  = ranked[:top_k]
        short_syms = ranked[-top_k:]

        # BTC filter: 방향 제한
        allow_long  = True if not btc_filter else (trend >= 0)
        allow_short = True if not btc_filter else (trend <= 0)

        for sym in long_syms:
            if not allow_long: break
            if (sym, "LONG") in open_positions: continue
            if len(open_positions) >= MAX_CONCURRENT: break
            price = get_arr(sym, "close", abs_bar)
            if price is None: continue
            open_positions[(sym, "LONG")] = (abs_bar, price, price*(1-sl), price*(1+tp))

        for sym in short_syms:
            if not allow_short: break
            if (sym, "SHORT") in open_positions: continue
            if len(open_positions) >= MAX_CONCURRENT: break
            price = get_arr(sym, "close", abs_bar)
            if price is None: continue
            open_positions[(sym, "SHORT")] = (abs_bar, price, price*(1+sl), price*(1-tp))

    # ── 메트릭 ──────────────────────────────────────────────────────────────
    pnls = [t[1] for t in all_trades]
    if not pnls:
        return {"N": 0, "WR": 0, "PF": 0, "EV_bp": 0, "MaxDD": 0, "trades": [], "equity": [1.0]}
    wins  = [p for p in pnls if p > 0]
    loses = [p for p in pnls if p <= 0]
    pf  = sum(wins) / -sum(loses) if loses else float("inf")
    wr  = len(wins) / len(pnls)
    ev  = np.mean(pnls) * 10000
    years = (end_dt - start_dt).days / 365.25
    return {
        "N": len(pnls), "N/yr": len(pnls)/years,
        "WR": wr, "PF": pf, "EV_bp": ev, "MaxDD": max_dd,
        "trades": all_trades,
    }


# ── Phase 1: Sweep ranking × filter combos ───────────────────────────────────
print("\n[Phase 1] Ranking mode × Filter sweep (lookback=168bars, top_k=5, hold=8d)")
print(f"{'ranking':>8} {'btc_f':>6} {'vr_f':>5}  "
      f"{'N':>5} {'N/yr':>5} {'WR':>6} {'PF':>6} {'EV_bp':>7} {'MaxDD':>7}")
print("-" * 65)

combos = list(product(
    ["return", "sharpe"],   # ranking_mode
    [False, True],          # btc_filter
    [0.0, 0.5, 0.65],       # vr_filter
))

results = []
for rm, bf, vf in combos:
    m = run_backtest(
        lookback_bars=168, top_k=5, hold_bars=48,
        sl=0.05, tp=0.10, rebalance_bars=24,
        ranking_mode=rm, btc_filter=bf, vr_filter=vf,
    )
    results.append((rm, bf, vf, m))
    flag = "★" if m["PF"] > 1.10 and m["N"] > 200 else " "
    print(f"{rm:>8} {str(bf):>6} {vf:>5.2f}  "
          f"{m['N']:>5} {m['N/yr']:>5.0f} {m['WR']:>6.2%} {m['PF']:>6.3f} "
          f"{m['EV_bp']:>7.1f} {m['MaxDD']:>7.2%} {flag}")

# ── Phase 2: Best + OOS ───────────────────────────────────────────────────────
best = max(results, key=lambda x: x[3]["PF"] if x[3]["N"] > 100 else 0)
rm, bf, vf, bm = best
print(f"\n[Phase 2] Best: ranking={rm} btc_filter={bf} vr_filter={vf}")
print(f"  IS  N={bm['N']} N/yr={bm['N/yr']:.0f} WR={bm['WR']:.2%} PF={bm['PF']:.3f} "
      f"EV={bm['EV_bp']:.1f}bp MaxDD={bm['MaxDD']:.2%}")

SPLIT = datetime(2023, 1, 1)  # IS: 2020-2022, OOS: 2023-2026
m_is  = run_backtest(lookback_bars=168, top_k=5, hold_bars=48, sl=0.05, tp=0.10,
                     rebalance_bars=24, ranking_mode=rm, btc_filter=bf, vr_filter=vf,
                     start_dt=datetime(2020,6,1), end_dt=SPLIT)
m_oos = run_backtest(lookback_bars=168, top_k=5, hold_bars=48, sl=0.05, tp=0.10,
                     rebalance_bars=24, ranking_mode=rm, btc_filter=bf, vr_filter=vf,
                     start_dt=SPLIT, end_dt=END_DT)
print(f"  IS  (2020-2022): N={m_is['N']}  PF={m_is['PF']:.3f}  EV={m_is['EV_bp']:.1f}bp")
print(f"  OOS (2023-2026): N={m_oos['N']}  PF={m_oos['PF']:.3f}  EV={m_oos['EV_bp']:.1f}bp")
oos_is_ratio = m_oos["PF"] / m_is["PF"] if m_is["PF"] > 0 else 0
print(f"  OOS/IS ratio: {oos_is_ratio:.2f}  ({'OK' if oos_is_ratio > 0.85 else 'OVERFIT?'})")

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

# ── Equity curve plot ─────────────────────────────────────────────────────────
# IS/OOS 별도로 equity 재계산
def equity_curve(trades, start_dt, end_dt):
    ec = [1.0]
    eq = 1.0
    for dt, pnl, sym, side in sorted(trades, key=lambda x: x[0]):
        if start_dt <= dt <= end_dt:
            eq *= (1 + pnl / MAX_CONCURRENT)
            ec.append(eq)
    return ec

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor("#0e1117")
for ax, (m_plot, label, col) in zip(axes, [
    (bm,    f"Full ({rm} btcf={bf} vrf={vf})", "#00d4ff"),
    (m_oos, "OOS 2023-2026",                   "#ff6b35"),
]):
    ax.set_facecolor("#0e1117")
    ec = equity_curve(m_plot["trades"], datetime(2020,1,1), END_DT)
    ax.plot(ec, color=col, linewidth=1.2)
    ax.fill_between(range(len(ec)), 1.0, ec, alpha=0.15, color=col)
    ax.axhline(1.0, color="white", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_title(f"r50v2: {label}  PF={m_plot['PF']:.3f}", color="white", fontsize=11)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#333")

out_dir = ROOT / "docs" / "backtest_results" / "202603"
out_path = out_dir / "20260308_r50v2_cs_momentum.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {out_path}")
print("=" * 65)
