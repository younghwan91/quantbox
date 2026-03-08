#!/usr/bin/env python3
"""r52: Asymmetric Small-Cap Strategies — 비대칭 수익 추구.

3가지 전략 백테스트:
  A. VSB: Volume Shock Breakout — 과거 ATR 압축 + 현재 거래량 폭발 + 돌파
  B. RCB: Range Compression Breakout — 횡보 압축 후 방향성 돌파
  C. BSB: BTC Beta Surge — BTC 신고점 시 고베타 알트 추종

공통 Exit: SL=1%, Trailing Stop 전용 (flat TP 없음 → 30-100%+ 목표)
"""
from __future__ import annotations
import sys, time, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import portfolio_sim, compute_metrics, bootstrap_pf

# ── Phase 0: Config ───────────────────────────────────────────────────────────
cfg = VolStraddleConfig()
SL_PCT   = cfg.d_sl
COST     = 0.0008
MAX_CONC = cfg.max_concurrent
LN       = cfg.base_leverage / MAX_CONC
MAX_HOLD = 192

try:
    COMMIT = subprocess.check_output(["git","rev-parse","--short","HEAD"],
                                     cwd=ROOT, text=True).strip()
except Exception:
    COMMIT = "unknown"

print("=" * 72)
print("r52: Asymmetric Small-Cap Strategies")
print(f"  commit={COMMIT}  SL={SL_PCT*100:.1f}%  cost={COST*100:.2f}%  "
      f"LN={LN:.1f}  max_hold={MAX_HOLD}bars")
print(f"  [Phase 0] d_sl={cfg.d_sl}  LN={LN:.1f}  ✅ from VolStraddleConfig")
print("=" * 72)

# ── Data Loading ──────────────────────────────────────────────────────────────
CACHE_DIR = ROOT / "data" / "cache_extended"

def load_all_symbols(min_years: float = 3.0) -> dict[str, dict]:
    seen: dict[str, tuple[str, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        npz = np.load(f, allow_pickle=False)
        if "close" not in npz.files or "volume" not in npz.files:
            continue
        n = len(npz["close"])
        if sym not in seen or n > seen[sym][1]:
            seen[sym] = (str(f), n)
    min_bars = int(min_years * 365 * 4 * 24)
    data: dict[str, dict] = {}
    for sym, (fpath, n) in seen.items():
        if n < min_bars:
            continue
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}
    max_bars = max(len(d["close"]) for d in data.values())
    for sym, d in data.items():
        d["_offset"] = max_bars - len(d["close"])
    return data

t0 = time.time()
print("\nLoading data...")
data_15m = load_all_symbols(min_years=3.0)
MAX_BARS = max(len(d["close"]) for d in data_15m.values())
n_syms = len(data_15m)
avg_bars = int(np.mean([len(d["close"]) for d in data_15m.values()]))
data_years = avg_bars / (4 * 24 * 365)
print(f"  {n_syms} symbols | avg {avg_bars:,} bars | {data_years:.1f} years | {time.time()-t0:.1f}s")

# ── Rolling Helpers ────────────────────────────────────────────────────────────
def rmean(arr: np.ndarray, w: int) -> np.ndarray:
    a = arr.astype(np.float64)
    cs = np.nancumsum(a)
    out = np.full(len(a), np.nan)
    out[w-1:] = cs[w-1:]
    out[w:] -= cs[:-w]
    out[w-1:] /= w
    return out

def rmax_excl(arr: np.ndarray, w: int) -> np.ndarray:
    """max of previous w bars, excluding current bar."""
    r = pd.Series(arr.astype(float)).rolling(w, min_periods=w).max().values
    out = np.empty_like(r); out[0] = np.nan; out[1:] = r[:-1]
    return out

def rmin_excl(arr: np.ndarray, w: int) -> np.ndarray:
    """min of previous w bars, excluding current bar."""
    r = pd.Series(arr.astype(float)).rolling(w, min_periods=w).min().values
    out = np.empty_like(r); out[0] = np.nan; out[1:] = r[:-1]
    return out

def rmin_rolling(arr: np.ndarray, w: int) -> np.ndarray:
    """min of past w bars (inclusive of current bar)."""
    return pd.Series(arr.astype(float)).rolling(w, min_periods=w).min().values

# ── Trade Simulation ──────────────────────────────────────────────────────────
def sim_trade(ep, closes, highs, lows, direction, sl_pct, trail_act, trail_dist, max_hold):
    """Returns (raw_pnl, hold_bars)."""
    sl = ep * (1.0 - sl_pct * direction)
    peak = 0.0; trail_active = False
    n = min(max_hold, len(closes))
    for i in range(n):
        lo, hi, cl = lows[i], highs[i], closes[i]
        if direction == 1:
            if lo <= sl: return -sl_pct, i + 1
            pnl = (cl - ep) / ep
        else:
            if hi >= sl: return -sl_pct, i + 1
            pnl = (ep - cl) / ep
        if pnl > peak: peak = pnl
        if peak >= trail_act: trail_active = True
        if trail_active and pnl <= peak - trail_dist: return pnl, i + 1
    if n == 0: return 0.0, 0
    cl = closes[n-1]
    return (cl - ep) / ep * direction, n

# ── Strategy A: VSB ───────────────────────────────────────────────────────────
def collect_vsb(
    data, atr_s_w=5, atr_l_w=20, compress=0.80,
    compress_lookback=48,  # check compression in past 2h
    vol_w=30, vol_mult=3.0, break_w=20,
    trail_act=0.05, trail_dist=0.025, warmup=250,
):
    """VSB: Recent ATR compression + current volume surge + N-bar breakout."""
    trades = []
    for sym, d in data.items():
        close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]
        offset = d["_offset"]; n = len(close)

        tr = np.empty(n); tr[0] = high[0] - low[0]
        tr[1:] = np.maximum(high[1:]-low[1:], np.maximum(
            np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))

        atr_s = rmean(tr, atr_s_w)
        atr_l = rmean(tr, atr_l_w)
        vol_ma = rmean(vol, vol_w)

        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = atr_s / np.where(atr_l > 0, atr_l, np.nan)

        # Was compressed recently? min(ratio) over past compress_lookback bars
        was_compressed = rmin_rolling(ratio, compress_lookback) < compress

        prev_high = rmax_excl(high, break_w)
        prev_low  = rmin_excl(low,  break_w)

        with np.errstate(invalid='ignore', divide='ignore'):
            vol_ok = vol > vol_mult * np.where(vol_ma > 0, vol_ma, np.inf)

        long_ok  = close > prev_high
        short_ok = close < prev_low
        valid = (~np.isnan(atr_l) & ~np.isnan(vol_ma) &
                 ~np.isnan(prev_high) & ~np.isnan(prev_low) & was_compressed)
        valid[:warmup] = False; valid[-1] = False

        for i in np.where(valid & vol_ok & (long_ok | short_ok))[0]:
            direction = 1 if long_ok[i] else -1
            ep = close[i]
            pnl_raw, hold = sim_trade(ep, close[i+1:], high[i+1:], low[i+1:],
                                       direction, SL_PCT, trail_act, trail_dist, MAX_HOLD)
            entry_abs = i + 1 + offset
            trades.append((sym, entry_abs, entry_abs + hold, pnl_raw - COST))
    return trades

# ── Strategy B: RCB ───────────────────────────────────────────────────────────
def collect_rcb(
    data, range_w=96, range_thresh=0.04,
    trail_act=0.05, trail_dist=0.025, warmup=250,
):
    """RCB: Daily range compressed + close breaks outside range."""
    trades = []
    for sym, d in data.items():
        close, high, low = d["close"], d["high"], d["low"]
        offset = d["_offset"]; n = len(close)

        rng_high = rmax_excl(high, range_w)
        rng_low  = rmin_excl(low,  range_w)

        with np.errstate(invalid='ignore', divide='ignore'):
            rng_pct = np.where(close > 0, (rng_high - rng_low) / close, np.inf)

        compressed = rng_pct < range_thresh
        long_ok    = close > rng_high
        short_ok   = close < rng_low
        valid = (~np.isnan(rng_high) & ~np.isnan(rng_low) & (close > 0))
        valid[:warmup] = False; valid[-1] = False

        for i in np.where(valid & compressed & (long_ok | short_ok))[0]:
            direction = 1 if long_ok[i] else -1
            ep = close[i]
            pnl_raw, hold = sim_trade(ep, close[i+1:], high[i+1:], low[i+1:],
                                       direction, SL_PCT, trail_act, trail_dist, MAX_HOLD)
            entry_abs = i + 1 + offset
            trades.append((sym, entry_abs, entry_abs + hold, pnl_raw - COST))
    return trades

# ── Strategy C: BSB ───────────────────────────────────────────────────────────
def collect_bsb(
    data, btc_break_w=480, beta_w=96, top_k=3,
    trail_act=0.06, trail_dist=0.03, warmup=300,
):
    """BSB: BTC N-day breakout → top-K beta alts in BTC direction."""
    if "BTCUSDT" not in data: return []
    btc = data["BTCUSDT"]
    btc_c, btc_h, btc_l = btc["close"], btc["high"], btc["low"]
    N_btc = len(btc_c)

    btc_prev_high = rmax_excl(btc_h, btc_break_w)
    btc_prev_low  = rmin_excl(btc_l, btc_break_w)

    btc_ret = np.zeros(N_btc)
    btc_ret[1:] = np.diff(btc_c) / np.maximum(btc_c[:-1], 1e-10)

    alts: dict[str, dict] = {}
    alt_ret_aligned: dict[str, np.ndarray] = {}
    for sym, d in data.items():
        if sym == "BTCUSDT": continue
        n_alt = len(d["close"])
        start = N_btc - n_alt
        ar = np.zeros(N_btc)
        if n_alt > 1:
            r = np.zeros(n_alt); r[1:] = np.diff(d["close"]) / np.maximum(d["close"][:-1], 1e-10)
            ar[start:] = r
        alts[sym] = {"close": d["close"], "high": d["high"], "low": d["low"],
                     "start": start, "offset": d["_offset"]}
        alt_ret_aligned[sym] = ar

    trades = []
    cooldown: dict[str, int] = {}

    for i in range(max(warmup, btc_break_w, beta_w), N_btc - 1):
        if np.isnan(btc_prev_high[i]) or np.isnan(btc_prev_low[i]): continue
        if btc_c[i] > btc_prev_high[i]: direction = 1
        elif btc_c[i] < btc_prev_low[i]: direction = -1
        else: continue

        btc_slice = btc_ret[i - beta_w + 1: i + 1]
        btc_var = np.var(btc_slice)
        if btc_var < 1e-12: continue

        scored: list[tuple[float, str]] = []
        for sym, ad in alts.items():
            if cooldown.get(sym, 0) > i: continue
            if ad["start"] > i - beta_w: continue
            alt_slice = alt_ret_aligned[sym][i - beta_w + 1: i + 1]
            cov = np.cov(btc_slice, alt_slice)[0, 1]
            beta = cov / btc_var
            if direction * beta > 0:
                scored.append((abs(beta), sym))

        if not scored: continue
        scored.sort(reverse=True)
        for _, sym in scored[:top_k]:
            ad = alts[sym]
            local_i = i - ad["start"]
            if local_i < 0 or local_i + 1 >= len(ad["close"]): continue
            ep = ad["close"][local_i]
            pnl_raw, hold = sim_trade(ep,
                ad["close"][local_i+1:], ad["high"][local_i+1:], ad["low"][local_i+1:],
                direction, SL_PCT, trail_act, trail_dist, MAX_HOLD)
            entry_abs = i + 1 + ad["offset"]
            trades.append((sym, entry_abs, entry_abs + hold, pnl_raw - COST))
            cooldown[sym] = i + btc_break_w // 4
    return trades

# ── Run Helper ────────────────────────────────────────────────────────────────
def run(trades, label):
    if not trades:
        print(f"  [{label}] No trades"); return None
    pnl = portfolio_sim(trades, MAX_CONC)
    if len(pnl) == 0:
        print(f"  [{label}] All filtered"); return None
    m = compute_metrics(pnl, data_years)
    boot = bootstrap_pf(pnl, n_boot=500)
    wr = float(m["WR"])
    print(f"  [{label:30s}] N={m['N']:5d}  PF={m['PF']:.3f}  WR={wr:.1f}%  "
          f"EV={m['EV_bp']:+.0f}bp  DD={m['max_dd_pct']:.1f}%  "
          f"AnnRet={m['ann_ret_pct']:.1f}%  "
          f"CI=[{boot['CI_lo']:.3f},{boot['CI_hi']:.3f}]")
    return {"label": label, **m, "boot_lo": boot["CI_lo"], "boot_hi": boot["CI_hi"]}

# ── Phase 1: Baseline ─────────────────────────────────────────────────────────
print("\n── Phase 1: Baseline ────────────────────────────────────────────────────")
t1 = time.time()
vsb_t = collect_vsb(data_15m)
print(f"  VSB raw: {len(vsb_t)}")
r_vsb = run(vsb_t, "VSB baseline")

rcb_t = collect_rcb(data_15m)
print(f"  RCB raw: {len(rcb_t)}")
r_rcb = run(rcb_t, "RCB baseline")

bsb_t = collect_bsb(data_15m)
print(f"  BSB raw: {len(bsb_t)}")
r_bsb = run(bsb_t, "BSB baseline")

r_all = run(vsb_t + rcb_t + bsb_t, "VSB+RCB+BSB combined")
print(f"  Phase 1 elapsed: {time.time()-t1:.1f}s")

# ── Phase 2: Grid Search ──────────────────────────────────────────────────────
print("\n── Phase 2: Grid Search ─────────────────────────────────────────────────")
results = []

# VSB grid
for compress in [0.70, 0.80, 0.90]:
    for vol_mult in [2.0, 3.0, 5.0]:
        for trail_act in [0.03, 0.05, 0.08]:
            t = collect_vsb(data_15m, compress=compress, vol_mult=vol_mult,
                            trail_act=trail_act, trail_dist=trail_act*0.5)
            r = run(t, f"VSB c={compress:.2f} vm={vol_mult:.0f} ta={trail_act:.2f}")
            if r and r["N"] >= 50: results.append(r)

# RCB grid
for range_w in [48, 96, 192]:
    for thresh in [0.03, 0.05, 0.07]:
        for trail_act in [0.04, 0.06, 0.10]:
            t = collect_rcb(data_15m, range_w=range_w, range_thresh=thresh,
                            trail_act=trail_act, trail_dist=trail_act*0.5)
            r = run(t, f"RCB rw={range_w} th={thresh:.2f} ta={trail_act:.2f}")
            if r and r["N"] >= 50: results.append(r)

# BSB grid
for btc_w in [240, 480, 960]:
    for k in [2, 3, 5]:
        for trail_act in [0.05, 0.08, 0.12]:
            t = collect_bsb(data_15m, btc_break_w=btc_w, top_k=k,
                            trail_act=trail_act, trail_dist=trail_act*0.5)
            r = run(t, f"BSB bw={btc_w} k={k} ta={trail_act:.2f}")
            if r and r["N"] >= 30: results.append(r)

# Best per strategy
if results:
    best = sorted(results, key=lambda x: (x["PF"], x["N"]), reverse=True)
    print("\n── Top 10 configs by PF ────────────────────────────────────────────────")
    for r in best[:10]:
        print(f"  {r['label']:50s} PF={r['PF']:.3f}  N={r['N']}  WR={r['WR']:.1f}%")

# ── Phase 3: OOS ─────────────────────────────────────────────────────────────
print("\n── Phase 3: OOS (50% holdout) ───────────────────────────────────────────")
SPLIT = MAX_BARS // 2

def split_run(trades, label):
    is_t  = [(s,e,x,p) for s,e,x,p in trades if e < SPLIT]
    oos_t = [(s,e,x,p) for s,e,x,p in trades if e >= SPLIT]
    is_pnl  = portfolio_sim(is_t,  MAX_CONC)
    oos_pnl = portfolio_sim(oos_t, MAX_CONC)
    if len(is_pnl) == 0 or len(oos_pnl) == 0:
        print(f"  [{label}] Insufficient for OOS"); return
    m_is  = compute_metrics(is_pnl,  data_years/2)
    m_oos = compute_metrics(oos_pnl, data_years/2)
    ratio = m_oos["PF"] / m_is["PF"] if m_is["PF"] > 0 else 0
    print(f"  [{label:15s}] IS PF={m_is['PF']:.3f} N={m_is['N']}  "
          f"OOS PF={m_oos['PF']:.3f} N={m_oos['N']}  ratio={ratio:.2f}")

split_run(vsb_t, "VSB")
split_run(rcb_t, "RCB")
split_run(bsb_t, "BSB")
split_run(vsb_t + rcb_t + bsb_t, "ALL")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("r52 SUMMARY")
print(f"  Data: {data_years:.1f}yr  Syms: {n_syms}  SL={SL_PCT*100:.1f}%  "
      f"Trail-only  Cost={COST*100:.2f}%  LN={LN:.1f}")
print(f"  Elapsed: {time.time()-t0:.0f}s")
for label, res in [("VSB", r_vsb), ("RCB", r_rcb), ("BSB", r_bsb), ("ALL", r_all)]:
    if res:
        print(f"  Baseline {label}: PF={res['PF']:.3f}  N={res['N']}  "
              f"WR={res['WR']:.1f}%  EV={res['EV_bp']:+.0f}bp  "
              f"AnnRet={res['ann_ret_pct']:.1f}%")
print("=" * 72)
