#!/usr/bin/env python3
"""r101: 1-minute data validation for trailing stop accuracy.

Goal: Compare 15m-based trail simulation vs 1m-based simulation to quantify
intra-bar path ambiguity error.

Method:
1. Download 1m data for sample symbols (recent 60 days)
2. Run trail simulation on 1m data (ground truth)
3. Aggregate to 15m and run 15m simulation
4. Compare exit prices and PnL

The 15m sim assumes "high first, then low" within a bar.
The 1m sim shows actual price path.
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from binance.client import Client
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import compute_vr

cfg = VolStraddleConfig()

# Test symbols (diverse market cap)
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "SUIUSDT"]
DAYS = 60  # 60 days of 1m data = 86,400 bars per symbol


def download_1m_data(client: Client, symbol: str, days: int) -> dict | None:
    """Download 1m klines for validation."""
    print(f"  Downloading {symbol} 1m ({days}d)...", end=" ", flush=True)
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    all_klines = []
    current_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    while current_ms < end_ms:
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval="1m",
                startTime=current_ms,
                limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            current_ms = klines[-1][6] + 1  # close time + 1ms
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if len(all_klines) < 1000:
        print(f"SKIP (only {len(all_klines)} bars)")
        return None
    
    # Drop incomplete last bar
    all_klines = all_klines[:-1]
    
    close = np.array([float(k[4]) for k in all_klines], dtype=np.float64)
    high = np.array([float(k[2]) for k in all_klines], dtype=np.float64)
    low = np.array([float(k[3]) for k in all_klines], dtype=np.float64)
    
    print(f"OK ({len(close):,} bars)")
    return {"close": close, "high": high, "low": low}


def aggregate_to_15m(data_1m: dict) -> dict:
    """Aggregate 1m data to 15m OHLC."""
    c1 = data_1m["close"]
    h1 = data_1m["high"]
    l1 = data_1m["low"]
    
    n = len(c1)
    n_15m = n // 15
    
    c15 = np.empty(n_15m, dtype=np.float64)
    h15 = np.empty(n_15m, dtype=np.float64)
    l15 = np.empty(n_15m, dtype=np.float64)
    
    for i in range(n_15m):
        start = i * 15
        end = start + 15
        c15[i] = c1[end - 1]  # close = last 1m close
        h15[i] = np.max(h1[start:end])
        l15[i] = np.min(l1[start:end])
    
    return {"close": c15, "high": h15, "low": l15}


def sim_trail_1m(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_bar_1m: int,  # 1m bar index where entry happens
    direction: int,  # 1=LONG, -1=SHORT
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold_1m: int,
) -> tuple[float, int, str]:
    """Simulate single trade on 1m data (ground truth).
    
    Returns: (pnl, exit_bar_1m, exit_reason)
    """
    entry_price = close[entry_bar_1m]
    if entry_price <= 0:
        return 0.0, entry_bar_1m, "INVALID"
    
    n = len(close)
    end = min(entry_bar_1m + 1 + max_hold_1m, n)
    
    trailing_active = False
    peak_gain = 0.0
    
    for j in range(entry_bar_1m + 1, end):
        if direction == 1:  # LONG
            # SL check
            if low[j] <= entry_price * (1 - sl):
                return -sl, j, "SL"
            
            # Peak update
            cur_gain = (high[j] - entry_price) / entry_price
            if cur_gain > peak_gain:
                peak_gain = cur_gain
            
            # Trail activation
            if not trailing_active and peak_gain >= trail_activate:
                trailing_active = True
            
            # Trail exit
            if trailing_active:
                trail_level = entry_price * (1 + peak_gain - trail_distance)
                if low[j] <= trail_level:
                    return max(peak_gain - trail_distance, -sl), j, "TRAIL"
            
            # TP check
            if high[j] >= entry_price * (1 + tp):
                return tp, j, "TP"
        
        else:  # SHORT
            # SL check
            if high[j] >= entry_price * (1 + sl):
                return -sl, j, "SL"
            
            # Peak update
            cur_gain = (entry_price - low[j]) / entry_price
            if cur_gain > peak_gain:
                peak_gain = cur_gain
            
            # Trail activation
            if not trailing_active and peak_gain >= trail_activate:
                trailing_active = True
            
            # Trail exit
            if trailing_active:
                trail_level = entry_price * (1 - peak_gain + trail_distance)
                if high[j] >= trail_level:
                    return max(peak_gain - trail_distance, -sl), j, "TRAIL"
            
            # TP check
            if low[j] <= entry_price * (1 - tp):
                return tp, j, "TP"
    
    # Max hold exit
    final_close = close[end - 1]
    if direction == 1:
        pnl = (final_close - entry_price) / entry_price
    else:
        pnl = (entry_price - final_close) / entry_price
    return pnl, end - 1, "MAX_HOLD"


def sim_trail_15m(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_bar_15m: int,
    direction: int,
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold_15m: int,
) -> tuple[float, int, str]:
    """Simulate single trade on 15m data (current method)."""
    entry_price = close[entry_bar_15m]
    if entry_price <= 0:
        return 0.0, entry_bar_15m, "INVALID"
    
    n = len(close)
    end = min(entry_bar_15m + 1 + max_hold_15m, n)
    
    trailing_active = False
    peak_gain = 0.0
    
    for j in range(entry_bar_15m + 1, end):
        if direction == 1:  # LONG
            # SL check FIRST (server SL fires intra-bar)
            if low[j] <= entry_price * (1 - sl):
                return -sl, j, "SL"
            
            # Peak update (assumes high is reached)
            cur_gain = (high[j] - entry_price) / entry_price
            if cur_gain > peak_gain:
                peak_gain = cur_gain
            
            # Trail activation
            if not trailing_active and peak_gain >= trail_activate:
                trailing_active = True
            
            # Trail exit (assumes low is also reached in same bar)
            if trailing_active:
                trail_level = entry_price * (1 + peak_gain - trail_distance)
                if low[j] <= trail_level:
                    return max(peak_gain - trail_distance, -sl), j, "TRAIL"
            
            # TP check
            if high[j] >= entry_price * (1 + tp):
                return tp, j, "TP"
        
        else:  # SHORT
            if high[j] >= entry_price * (1 + sl):
                return -sl, j, "SL"
            
            cur_gain = (entry_price - low[j]) / entry_price
            if cur_gain > peak_gain:
                peak_gain = cur_gain
            
            if not trailing_active and peak_gain >= trail_activate:
                trailing_active = True
            
            if trailing_active:
                trail_level = entry_price * (1 - peak_gain + trail_distance)
                if high[j] >= trail_level:
                    return max(peak_gain - trail_distance, -sl), j, "TRAIL"
            
            if low[j] <= entry_price * (1 - tp):
                return tp, j, "TP"
    
    final_close = close[end - 1]
    if direction == 1:
        pnl = (final_close - entry_price) / entry_price
    else:
        pnl = (entry_price - final_close) / entry_price
    return pnl, end - 1, "MAX_HOLD"


def find_squeeze_signals(vr: np.ndarray, threshold: float, duration: int) -> list[int]:
    """Find squeeze signal bars (VR < threshold for duration bars)."""
    signals = []
    squeeze_count = 0
    
    for i in range(len(vr)):
        if np.isnan(vr[i]):
            squeeze_count = 0
            continue
        
        if vr[i] < threshold:
            squeeze_count += 1
        else:
            if squeeze_count >= duration:
                signals.append(i)
            squeeze_count = 0
    
    return signals


def main():
    print("=" * 60)
    print("r101: 1-Minute Trail Validation")
    print("=" * 60)
    
    # Connect to Binance
    client = Client(
        os.environ["BINANCE_API_KEY"],
        os.environ["BINANCE_API_SECRET"],
    )
    
    # Parameters
    s_vr = cfg.s_vr_threshold
    s_dur = cfg.s_squeeze_dur
    s_sl = cfg.s_sl
    s_tp = 0.20
    s_trail_act = cfg.s_trail_activate
    s_trail_dist = cfg.s_trail_distance
    max_hold_15m = 96  # 24h for faster test
    max_hold_1m = max_hold_15m * 15
    
    print(f"\nParameters:")
    print(f"  VR threshold: {s_vr}")
    print(f"  Squeeze duration: {s_dur}")
    print(f"  SL: {s_sl*100}%, TP: {s_tp*100}%")
    print(f"  Trail activate: {s_trail_act*100}%, Trail distance: {s_trail_dist*100}%")
    print(f"  Max hold: {max_hold_15m} bars (15m) = {max_hold_15m/4}h")
    
    print(f"\nDownloading 1m data ({DAYS} days)...")
    
    all_comparisons = []
    
    for sym in TEST_SYMBOLS:
        data_1m = download_1m_data(client, sym, DAYS)
        if data_1m is None:
            continue
        
        data_15m = aggregate_to_15m(data_1m)
        
        # Compute VR on 15m data
        vr_15m = compute_vr(
            data_15m["close"],
            data_15m["high"],
            data_15m["low"],
            cfg.vr_short,
            cfg.vr_long,
        )
        
        # Find squeeze signals
        signals_15m = find_squeeze_signals(vr_15m, s_vr, s_dur)
        
        print(f"  {sym}: {len(signals_15m)} squeeze signals in {DAYS}d")
        
        for sig_bar_15m in signals_15m:
            # Entry at signal bar close
            entry_bar_15m = sig_bar_15m
            entry_bar_1m = entry_bar_15m * 15 + 14  # last 1m of 15m bar
            
            if entry_bar_1m >= len(data_1m["close"]) - max_hold_1m:
                continue
            
            # Test both directions (straddle)
            for direction in [1, -1]:
                # 1m simulation (ground truth)
                pnl_1m, exit_1m, reason_1m = sim_trail_1m(
                    data_1m["close"], data_1m["high"], data_1m["low"],
                    entry_bar_1m, direction,
                    s_sl, s_tp, s_trail_act, s_trail_dist, max_hold_1m,
                )
                
                # 15m simulation
                pnl_15m, exit_15m, reason_15m = sim_trail_15m(
                    data_15m["close"], data_15m["high"], data_15m["low"],
                    entry_bar_15m, direction,
                    s_sl, s_tp, s_trail_act, s_trail_dist, max_hold_15m,
                )
                
                diff = (pnl_15m - pnl_1m) * 100  # in percentage points
                
                all_comparisons.append({
                    "sym": sym,
                    "dir": "LONG" if direction == 1 else "SHORT",
                    "pnl_1m": pnl_1m * 100,
                    "pnl_15m": pnl_15m * 100,
                    "diff": diff,
                    "reason_1m": reason_1m,
                    "reason_15m": reason_15m,
                    "reason_match": reason_1m == reason_15m,
                })
    
    if not all_comparisons:
        print("\nNo comparisons made!")
        return
    
    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    n = len(all_comparisons)
    diffs = [c["diff"] for c in all_comparisons]
    pnl_1m_arr = [c["pnl_1m"] for c in all_comparisons]
    pnl_15m_arr = [c["pnl_15m"] for c in all_comparisons]
    reason_matches = [c["reason_match"] for c in all_comparisons]
    
    print(f"\nTotal comparisons: {n}")
    print(f"\nExit reason match rate: {sum(reason_matches)/n*100:.1f}%")
    
    print(f"\nPnL Difference (15m - 1m):")
    print(f"  Mean:   {np.mean(diffs):+.4f}%")
    print(f"  Median: {np.median(diffs):+.4f}%")
    print(f"  Std:    {np.std(diffs):.4f}%")
    print(f"  Min:    {np.min(diffs):+.4f}%")
    print(f"  Max:    {np.max(diffs):+.4f}%")
    
    print(f"\nMean PnL:")
    print(f"  1m (ground truth): {np.mean(pnl_1m_arr):+.4f}%")
    print(f"  15m (current):     {np.mean(pnl_15m_arr):+.4f}%")
    print(f"  Bias:              {np.mean(pnl_15m_arr) - np.mean(pnl_1m_arr):+.4f}%")
    
    # Breakdown by exit reason
    print(f"\nBreakdown by exit reason mismatch:")
    mismatches = [c for c in all_comparisons if not c["reason_match"]]
    print(f"  Mismatched exits: {len(mismatches)} ({len(mismatches)/n*100:.1f}%)")
    if mismatches:
        for m in mismatches[:10]:
            print(f"    {m['sym']} {m['dir']}: 1m={m['reason_1m']} vs 15m={m['reason_15m']}, diff={m['diff']:+.3f}%")
    
    # Trail-specific analysis
    trail_1m = [c for c in all_comparisons if c["reason_1m"] == "TRAIL"]
    trail_15m = [c for c in all_comparisons if c["reason_15m"] == "TRAIL"]
    
    print(f"\nTrail exit analysis:")
    print(f"  1m trail exits: {len(trail_1m)}")
    print(f"  15m trail exits: {len(trail_15m)}")
    
    if trail_1m:
        trail_diffs = [c["diff"] for c in trail_1m]
        print(f"  Trail exit diff mean: {np.mean(trail_diffs):+.4f}%")
        print(f"  Trail exit diff std:  {np.std(trail_diffs):.4f}%")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    bias = np.mean(pnl_15m_arr) - np.mean(pnl_1m_arr)
    if abs(bias) < 0.05:  # < 0.05%
        print("✅ 15m simulation has MINIMAL bias vs 1m ground truth")
        print(f"   Bias magnitude: {abs(bias)*100:.1f}bp")
    elif bias > 0:
        print("⚠️ 15m simulation is OPTIMISTIC (overestimates PnL)")
        print(f"   Bias: +{bias:.4f}% per trade")
    else:
        print("⚠️ 15m simulation is PESSIMISTIC (underestimates PnL)")
        print(f"   Bias: {bias:.4f}% per trade")


if __name__ == "__main__":
    main()
