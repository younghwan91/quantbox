"""r104b: Trail Noise Floor — 1m vs 15m trail simulation.

Goal:
  Find the trail_distance where 15m backtest results converge with 1m
  (tick-proxy) results. Below this floor, 15m backtests are overfit.

Method:
  1. Download 1m data for 10 representative symbols (60 days)
  2. Run S_d8 trail simulation on 1m bars (ground truth)
  3. Aggregate to 15m, run same simulation
  4. Compare PnL per trail_distance — find convergence point

The noise floor = trail_distance where 1m PF / 15m PF >= 0.8
(live will be even noisier than 1m due to tick spread, but 1m is
much closer to reality than 15m)

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r101_r105/r104b_trail_noise_floor.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]

from binance.client import Client
from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    compute_metrics,
    gen_vr_dur_signal,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import compute_vr

cfg = VolStraddleConfig()

# Representative symbols: diverse market cap + liquidity
TEST_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "SUIUSDT",
    "AVAXUSDT", "LINKUSDT", "AAVEUSDT", "COMPUSDT", "CRVUSDT",
]
DAYS = 90  # 90 days of 1m data

# Trail sweep: realistic range for crypto (0.1% = noise, 2.0% = generous)
TRAIL_DIST_GRID = [0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020]
TRAIL_ACT = cfg.s_trail_activate  # 0.015
SL = cfg.s_sl  # 0.015
TP = cfg.s_tp  # 0.05
VR_SP = cfg.vr_short  # 8
VR_LP = cfg.vr_long  # 56
S_VR_THRESH = cfg.s_vr_threshold  # 0.57
S_DUR = cfg.s_squeeze_dur  # 3
WARMUP = 200


# ── Download ──────────────────────────────────────────────────────────────

def download_1m(client: Client, symbol: str, days: int) -> dict | None:
    """Download 1m OHLC from Binance."""
    cache_dir = ROOT / "data" / "cache_1m"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{symbol}_1m_{days}d.npz"

    if cache_file.exists():
        npz = np.load(str(cache_file))
        print(f"  {symbol}: cached {len(npz['close']):,} bars")
        return {k: npz[k].astype(np.float64) for k in ["close", "high", "low"]}

    print(f"  {symbol}: downloading {days}d of 1m...", end=" ", flush=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_klines, cur = [], start_ms
    while cur < end_ms:
        try:
            klines = client.futures_klines(
                symbol=symbol, interval="1m",
                startTime=cur, endTime=end_ms, limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            cur = klines[-1][0] + 1
            time.sleep(0.15)
        except Exception as e:
            print(f"error: {e}")
            time.sleep(5)

    if len(all_klines) < 1000:
        print(f"too few bars ({len(all_klines)})")
        return None

    all_klines = all_klines[:-1]  # remove incomplete
    close = np.array([float(k[4]) for k in all_klines], dtype=np.float64)
    high = np.array([float(k[2]) for k in all_klines], dtype=np.float64)
    low = np.array([float(k[3]) for k in all_klines], dtype=np.float64)

    np.savez_compressed(cache_file, close=close, high=high, low=low)
    print(f"{len(close):,} bars")
    return {"close": close, "high": high, "low": low}


def resample(data_1m: dict, ratio: int) -> dict:
    """Resample 1m OHLC to higher timeframe."""
    c, h, l = data_1m["close"], data_1m["high"], data_1m["low"]
    n = len(c) // ratio * ratio
    c, h, l = c[:n], h[:n], l[:n]

    c_r = c[ratio - 1::ratio].copy()
    h_r = h.reshape(-1, ratio).max(axis=1)
    l_r = l.reshape(-1, ratio).min(axis=1)
    return {"close": c_r, "high": h_r, "low": l_r}


def run_trail_sim(data: dict, trail_dist: float, warmup: int) -> tuple[np.ndarray, int]:
    """Run S_d8 straddle trail sim on given OHLC data.

    Returns (pnl_array, n_trades).
    """
    c, h, l = data["close"], data["high"], data["low"]
    vr = compute_vr(h, l, c, VR_SP, VR_LP)
    sig = gen_vr_dur_signal(vr, S_VR_THRESH, S_DUR, warmup)

    max_hold = len(c)  # no max_hold limit for this test
    pnl, entry, exit_ = sim_straddle_trailing(
        c, h, l, sig, SL, TP, max_hold, COST_STRADDLE, TRAIL_ACT, trail_dist,
    )
    return pnl, len(pnl)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("r104b: Trail Noise Floor — 1m vs 15m Comparison")
    print("=" * 70)
    print(f"Symbols: {len(TEST_SYMBOLS)}")
    print(f"Trail activate: {TRAIL_ACT} ({TRAIL_ACT*100:.1f}%)")
    print(f"Trail distance grid: {[f'{d*100:.1f}%' for d in TRAIL_DIST_GRID]}")
    print(f"SL: {SL*100:.1f}%  TP: {TP*100:.1f}%")
    print()

    # ── Download 1m data ──
    print("Downloading 1m data...")
    client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_API_SECRET"])
    data_1m: dict[str, dict] = {}
    for sym in TEST_SYMBOLS:
        d = download_1m(client, sym, DAYS)
        if d is not None:
            data_1m[sym] = d
    print(f"  Loaded {len(data_1m)} symbols\n")

    # ── Per-symbol, per-trail_distance: 1m vs 15m comparison ──
    print("─" * 70)
    print("SECTION 1: 1m vs 15m Trail Simulation Comparison")
    print("─" * 70)

    # Collect results: {trail_dist: {"1m": [pnl_list], "15m": [pnl_list]}}
    results_1m: dict[float, list[float]] = {td: [] for td in TRAIL_DIST_GRID}
    results_15m: dict[float, list[float]] = {td: [] for td in TRAIL_DIST_GRID}

    for sym, d1m in data_1m.items():
        d15m = resample(d1m, 15)
        n_1m = len(d1m["close"])
        n_15m = len(d15m["close"])

        for td in TRAIL_DIST_GRID:
            # 1m sim (ground truth): warmup scaled to 1m bars
            pnl_1m, n1 = run_trail_sim(d1m, td, warmup=WARMUP * 15)
            results_1m[td].extend(pnl_1m.tolist())

            # 15m sim (backtest standard)
            pnl_15m, n15 = run_trail_sim(d15m, td, warmup=WARMUP)
            results_15m[td].extend(pnl_15m.tolist())

    # ── Summary table ──
    print(f"\n  {'Trail%':>7} {'N_1m':>6} {'N_15m':>6} {'PF_1m':>7} {'PF_15m':>7} "
          f"{'EV_1m':>8} {'EV_15m':>8} {'Ratio':>7} {'Verdict':>10}")
    print("  " + "-" * 80)

    noise_floor = None
    for td in TRAIL_DIST_GRID:
        pnl_1m = np.array(results_1m[td])
        pnl_15m = np.array(results_15m[td])

        n1 = len(pnl_1m)
        n15 = len(pnl_15m)

        if n1 > 0:
            wins_1m = pnl_1m[pnl_1m > 0].sum()
            loss_1m = abs(pnl_1m[pnl_1m < 0].sum())
            pf_1m = wins_1m / loss_1m if loss_1m > 0 else float("inf")
            ev_1m = pnl_1m.mean() * 10000
        else:
            pf_1m, ev_1m = 0, 0

        if n15 > 0:
            wins_15m = pnl_15m[pnl_15m > 0].sum()
            loss_15m = abs(pnl_15m[pnl_15m < 0].sum())
            pf_15m = wins_15m / loss_15m if loss_15m > 0 else float("inf")
            ev_15m = pnl_15m.mean() * 10000
        else:
            pf_15m, ev_15m = 0, 0

        ratio = pf_1m / pf_15m if pf_15m > 0 else 0
        # Ratio > 0.8 means 1m and 15m converge (15m overestimates by <20%)
        if ratio >= 0.8 and noise_floor is None and n1 >= 10:
            verdict = "✓ CONVERGE"
            noise_floor = td
        elif ratio >= 0.8 and n1 >= 10:
            verdict = "✓ OK"
        elif n1 < 10:
            verdict = "— few"
        else:
            verdict = f"✗ {(1-ratio)*100:.0f}% bias"

        marker = " ◀ PROD" if abs(td - 0.001) < 1e-6 else ""
        print(f"  {td*100:>6.1f}% {n1:>6} {n15:>6} {pf_1m:>7.3f} {pf_15m:>7.3f} "
              f"{ev_1m:>+7.1f} {ev_15m:>+7.1f} {ratio:>6.2f}x {verdict}{marker}")

    # ── Per-trail exit quality ──
    print(f"\n" + "─" * 70)
    print("SECTION 2: Trail Exit Quality (1m ground truth)")
    print("─" * 70)
    print(f"  Trail activate = {TRAIL_ACT*100:.1f}% → trail fires only after +{TRAIL_ACT*100:.1f}% gain")
    print(f"  SL = {SL*100:.1f}% → max loss if trail never activates")
    print()
    print(f"  {'Trail%':>7} {'Trail exits':>12} {'Avg trail PnL':>14} {'SL exits':>10} {'Avg SL PnL':>12} {'Net EV':>8}")
    print("  " + "-" * 70)

    for td in TRAIL_DIST_GRID:
        pnl_arr = np.array(results_1m[td])
        if len(pnl_arr) == 0:
            continue
        # Approximate: trail exits are positive or small negative, SL exits are ~ -SL
        trail_mask = pnl_arr > -SL + 0.001  # trail or TP exits
        sl_mask = ~trail_mask
        n_trail = trail_mask.sum()
        n_sl = sl_mask.sum()
        avg_trail = pnl_arr[trail_mask].mean() * 10000 if n_trail > 0 else 0
        avg_sl = pnl_arr[sl_mask].mean() * 10000 if n_sl > 0 else 0
        ev = pnl_arr.mean() * 10000
        print(f"  {td*100:>6.1f}% {n_trail:>12} {avg_trail:>+13.1f}bp {n_sl:>10} {avg_sl:>+11.1f}bp {ev:>+7.1f}bp")

    # ── Decision ──
    print(f"\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    if noise_floor:
        print(f"  Noise floor: trail_distance >= {noise_floor*100:.1f}%")
        print(f"  Below this, 15m backtests overestimate by >20% vs 1m ground truth")
        print(f"  Current production: {0.001*100:.1f}% {'⚠️  BELOW NOISE FLOOR' if 0.001 < noise_floor else '✓ OK'}")
        if noise_floor > 0.001:
            print(f"\n  RECOMMENDATION: Widen trail_distance to {noise_floor*100:.1f}% (minimum)")
            print(f"  Consider {noise_floor*100:.1f}% - {min(noise_floor*2, 0.02)*100:.1f}% range for robustness")
    else:
        print("  No convergence found in tested range — all distances show significant 1m/15m divergence")
        print("  Consider widening trail_distance beyond tested range")

    print(f"\n  Runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
