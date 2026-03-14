"""r83: Live (Mar 4-14) vs Backtest comparison.

Downloads gap data to extend cached 15m NPZ through 2026-03-15,
builds 1h/4h from 15m, runs VR/STRD strategies, compares with journal.
"""
import sys
import numpy as np
import time
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from binance.client import Client
from dotenv import load_dotenv
load_dotenv()

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import (
    collect_trades, portfolio_sim, compute_metrics,
    precompute, precompute_all_vr, build_resampled,
)

cfg = VolStraddleConfig()
CACHE_DIR = Path("data/cache_extended")
LIVE_END_DT = datetime(2026, 3, 16, tzinfo=timezone.utc)
client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_API_SECRET"])
whitelist = sorted(cfg.symbol_whitelist)


def download_gap(symbol, interval, start_ms, end_ms):
    all_kl = []
    cur = start_ms
    while cur < end_ms:
        kl = client.futures_klines(symbol=symbol, interval=interval,
                                   startTime=cur, endTime=end_ms, limit=1500)
        if not kl:
            break
        all_kl.extend(kl)
        cur = kl[-1][0] + 1
        time.sleep(0.08)
    return all_kl


def load_extend_15m(symbol):
    """Load 15m NPZ + download gap → {close, high, low, volume, taker_buy}."""
    for days in ("2190d", "1095d", "365d"):
        files = sorted(CACHE_DIR.glob(f"{symbol}_15m_{days}_*.npz"))
        files = [f for f in files if "numtrades" not in f.name]
        if files:
            break
    else:
        return None

    npz = np.load(files[-1])
    tb_key = "taker_buy_volume" if "taker_buy_volume" in npz else "taker_buy"
    arrs = {
        "close": npz["close"].copy(), "high": npz["high"].copy(),
        "low": npz["low"].copy(), "volume": npz["volume"].copy(),
        "taker_buy": npz[tb_key].copy(),
    }

    date_str = files[-1].stem.split("_")[-1]
    file_end = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)

    gap_start_ms = int(file_end.timestamp() * 1000)
    gap_end_ms = int(LIVE_END_DT.timestamp() * 1000)
    n_orig = len(arrs["close"])

    if gap_start_ms < gap_end_ms:
        klines = download_gap(symbol, "15m", gap_start_ms, gap_end_ms)
        if klines:
            new_c = np.array([float(k[4]) for k in klines], dtype=np.float64)
            new_h = np.array([float(k[2]) for k in klines], dtype=np.float64)
            new_l = np.array([float(k[3]) for k in klines], dtype=np.float64)
            new_v = np.array([float(k[5]) for k in klines], dtype=np.float64)
            new_t = np.array([float(k[9]) for k in klines], dtype=np.float64)
            # Deduplicate overlap
            if len(new_c) > 0 and abs(new_c[0] - arrs["close"][-1]) < 1e-8:
                new_c, new_h, new_l, new_v, new_t = (
                    new_c[1:], new_h[1:], new_l[1:], new_v[1:], new_t[1:])
            arrs["close"] = np.concatenate([arrs["close"], new_c])
            arrs["high"] = np.concatenate([arrs["high"], new_h])
            arrs["low"] = np.concatenate([arrs["low"], new_l])
            arrs["volume"] = np.concatenate([arrs["volume"], new_v])
            arrs["taker_buy"] = np.concatenate([arrs["taker_buy"], new_t])

    arrs["_file_end"] = file_end
    arrs["_n_orig"] = n_orig
    arrs["_n_new"] = len(arrs["close"]) - n_orig
    return arrs


# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("r83: Live (Mar 4–14) vs Backtest Comparison")
print("=" * 70)

# ── Step 1: Load 15m data ──
print("\n[1/5] Loading 15m data + downloading gap...")
data_15m_raw = {}
skipped = []

for i, sym in enumerate(whitelist):
    print(f"  [{i+1:2d}/60] {sym:<14s}", end=" ", flush=True)
    d = load_extend_15m(sym)
    if d is None:
        skipped.append(sym)
        print("SKIP")
        continue
    data_15m_raw[sym] = d
    print(f"OK  {d['_n_orig']} + {d['_n_new']} = {len(d['close'])} bars")

print(f"\nLoaded: {len(data_15m_raw)} symbols  |  Skipped: {skipped}")

# Reference for bar→date
REF = "ETHUSDT"
ref_end = data_15m_raw[REF]["_file_end"]
n_orig = data_15m_raw[REF]["_n_orig"]
total_bars = len(data_15m_raw[REF]["close"])

# ── Step 2: Build data dicts for precompute ──
print("\n[2/5] Building data dicts...")
data_15m = {}
for sym, d in data_15m_raw.items():
    data_15m[sym] = {
        "close": d["close"], "high": d["high"],
        "low": d["low"], "volume": d["volume"],
    }

# Build 1h and 4h from 15m via build_resampled
data_1h = build_resampled(data_15m, 4)
data_4h = build_resampled(data_15m, 16)
print(f"  15m: {len(data_15m)} syms, 1h: {len(data_1h)} syms, 4h: {len(data_4h)} syms")

# ── Step 3: Precompute signals ──
print("\n[3/5] Precomputing VR + momentum + squeeze...")
sp, lp = cfg.vr_short, cfg.vr_long
for label, ds in [("15m", data_15m), ("1h", data_1h), ("4h", data_4h)]:
    precompute(ds, vr_sp=sp, vr_lp=lp, mom_lb=8,
               sq_thresh=cfg.s_vr_threshold, n_workers=1)
    print(f"  {label}: done")

precompute_all_vr([data_15m, data_1h, data_4h], [(sp, lp)])
print(f"  precompute_all_vr: vr_{sp}_{lp}")

# ── Step 4: Collect trades ──
print("\n[4/5] Collecting trades (full history)...")
static_cfg = {
    "sp": sp, "lp": lp,
    "d_vr": cfg.d_vr_threshold, "s_vr": cfg.s_vr_threshold,
    "s_dur": cfg.s_squeeze_dur,
    "d_sl": cfg.d_sl, "s_sl": cfg.s_sl, "tp": cfg.d_tp,
}

vr_trades = collect_trades(static_cfg, data_15m, data_1h, data_4h, strat="D_mom8")
strd_trades = collect_trades(static_cfg, data_15m, data_1h, data_4h, strat="S_d8")
print(f"  VR D_mom8: {len(vr_trades)} trades total")
print(f"  STRD S_d8: {len(strd_trades)} trades total")

# ── Step 5: Filter to live window (Mar 4-14) ──
print("\n[5/5] Filtering to Mar 4–14...")
live_start = datetime(2026, 3, 4, tzinfo=timezone.utc)
live_end = datetime(2026, 3, 15, tzinfo=timezone.utc)
bar_start = n_orig + int((live_start - ref_end).total_seconds() / 900)
bar_end = n_orig + int((live_end - ref_end).total_seconds() / 900)
print(f"  Bar window: [{bar_start}, {bar_end})")

def filter_window(trades, b0, b1):
    return [(s, eb, xb, p) for s, eb, xb, p in trades if b0 <= eb < b1]

vr_w = filter_window(vr_trades, bar_start, bar_end)
strd_w = filter_window(strd_trades, bar_start, bar_end)

def bar_to_date(bar_idx):
    """Convert 15m bar index to approximate datetime."""
    delta_bars = bar_idx - n_orig
    return ref_end + timedelta(minutes=15 * delta_bars)

def print_bt_trades(trades_w, label):
    if not trades_w:
        print(f"\n  {label}: 0 trades")
        return {}
    pnls = [t[3] for t in trades_w]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)
    wr = len(wins) / n * 100
    pf = sum(wins) / abs(sum(losses)) if losses else float('inf')
    avg_w = np.mean(wins) * 100 if wins else 0
    avg_l = np.mean(losses) * 100 if losses else 0

    print(f"\n{'─' * 60}")
    print(f"  {label} BACKTEST (Mar 4–14)")
    print(f"{'─' * 60}")
    print(f"  N={n}  WR={wr:.1f}%  PF={pf:.3f}  avg_win={avg_w:+.2f}%  avg_loss={avg_l:+.2f}%")
    print(f"  Individual trades:")
    for s, eb, xb, p in sorted(trades_w, key=lambda x: x[1]):
        entry_dt = bar_to_date(eb)
        exit_dt = bar_to_date(xb)
        bars = xb - eb
        print(f"    {entry_dt.strftime('%m/%d %H:%M')} → {exit_dt.strftime('%m/%d %H:%M')} "
              f" {s:<14s} {p*100:+7.2f}%  ({bars} bars)")
    return {"N": n, "WR": wr, "PF": pf, "avg_win": avg_w, "avg_loss": avg_l}

vr_bt = print_bt_trades(vr_w, "VR D_mom8")
strd_bt = print_bt_trades(strd_w, "STRD S_d8")


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LIVE JOURNAL (same period)")
print("=" * 70)

conn = sqlite3.connect("data/journal.sqlite")
conn.row_factory = sqlite3.Row

for strat_type, strat_label in [("vr_squeeze", "VR Squeeze"), ("straddle", "Straddle")]:
    rows = conn.execute(
        "SELECT symbol, direction, pnl_pct, pnl_usd, exit_reason, entry_time, exit_time "
        "FROM trades WHERE strategy_type=? ORDER BY entry_time", (strat_type,)
    ).fetchall()
    if not rows:
        continue

    raw_pnls = [r["pnl_pct"] for r in rows]
    # Detect scale: if >10 likely basis points
    if any(abs(p) > 10 for p in raw_pnls):
        pnls_frac = [p / 10000 for p in raw_pnls]
    else:
        pnls_frac = [p / 100 for p in raw_pnls]

    wins_l = [p for p in pnls_frac if p > 0]
    losses_l = [p for p in pnls_frac if p <= 0]
    n_l = len(rows)
    wr_l = len(wins_l) / n_l * 100
    pf_l = sum(wins_l) / abs(sum(losses_l)) if losses_l else float('inf')
    pnl_usd = sum(r["pnl_usd"] for r in rows)

    print(f"\n{'─' * 60}")
    print(f"  {strat_label} LIVE")
    print(f"{'─' * 60}")
    print(f"  N={n_l}  WR={wr_l:.1f}%  PF={pf_l:.3f}  PnL=${pnl_usd:+.0f}")
    for r in rows:
        entry_t = r["entry_time"][:16] if r["entry_time"] else "?"
        print(f"    {entry_t}  {r['symbol']:<14s} {r['direction']:<6s} "
              f"${r['pnl_usd']:+8.2f}  {r['exit_reason']}")

conn.close()


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GAP ANALYSIS: BACKTEST vs LIVE")
print("=" * 70)

for label, bt, live_type in [("VR", vr_bt, "vr_squeeze"), ("STRD", strd_bt, "straddle")]:
    if not bt:
        continue
    conn = sqlite3.connect("data/journal.sqlite")
    rows = conn.execute(
        "SELECT pnl_pct, pnl_usd FROM trades WHERE strategy_type=?", (live_type,)
    ).fetchall()
    conn.close()
    if not rows:
        continue
    raw = [r[0] for r in rows]
    pnls = [p/10000 for p in raw] if any(abs(p)>10 for p in raw) else [p/100 for p in raw]
    live_wins = [p for p in pnls if p > 0]
    live_losses = [p for p in pnls if p <= 0]
    n_l = len(rows)
    wr_l = len(live_wins) / n_l * 100
    pf_l = sum(live_wins) / abs(sum(live_losses)) if live_losses else float('inf')

    print(f"\n  {label}:")
    print(f"    {'':12s}   {'N':>5s}   {'WR':>6s}   {'PF':>7s}")
    print(f"    {'Backtest':12s}   {bt['N']:5d}   {bt['WR']:5.1f}%   {bt['PF']:7.3f}")
    print(f"    {'Live':12s}   {n_l:5d}   {wr_l:5.1f}%   {pf_l:7.3f}")
    print(f"    {'Gap':12s}   {bt['N']-n_l:+5d}   {bt['WR']-wr_l:+5.1f}pp  {bt['PF']-pf_l:+7.3f}")

print("""
■ 참고: 백테스트는 flat exit (SL/TP/max_hold), 라이브는 trailing stop.
  PnL 분포가 다를 수 있음. N과 WR 비교가 더 유의미.
■ 백테스트 N > 라이브 N → 라이브에서 슬롯 경쟁으로 일부 신호 미체결.
■ 백테스트 N < 라이브 N → 라이브에서 추가 진입 로직 존재 가능.
""")
