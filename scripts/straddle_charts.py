#!/usr/bin/env python3
"""Generate candlestick charts for straddle trades - winning and losing cases."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Handle both direct run and module run
if "__file__" in dir():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()

sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (  # noqa: E402
    build_resampled,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import (  # noqa: E402
    compute_momentum,
    compute_vr,
    gen_vr_dur_signal,
)

# Config
cfg = VolStraddleConfig()
CACHE_DIR = ROOT / "data" / "cache_extended"
OUTPUT_DIR = ROOT / "docs" / "charts" / "straddle_sanity"
WHITELIST = set(cfg.symbol_whitelist)

# Straddle params
S_VR_THRESH = cfg.s_vr_threshold  # 0.57
S_DUR = cfg.s_squeeze_dur  # 3
S_SL = cfg.s_sl  # 0.015 (1.5%)
S_TP = cfg.s_tp if hasattr(cfg, 's_tp') else 0.20
S_TRAIL_ACT = cfg.s_trail_activate  # 0.015
S_TRAIL_DIST = cfg.s_trail_distance  # 0.001
MAX_HOLD = cfg.max_hold_15m  # 192

COST = 0.0016  # straddle cost (both legs)

# Chart style (dark theme)
BG_DARK = '#0d1117'
BG_CARD = '#161b22'
TEXT_MAIN = '#f0f6fc'
TEXT_SUB = '#8b949e'
GREEN = '#3fb950'
RED = '#f85149'
ACCENT = '#58a6ff'


def load_data() -> tuple[dict, int]:
    """Load 15m data."""
    data = {}
    max_bars = 0

    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in WHITELIST:
            continue
        d = np.load(f)
        c = d["close"].astype(np.float64)
        h = d["high"].astype(np.float64)
        lo = d["low"].astype(np.float64)
        
        # Simulate open if missing: previous close (realistic approximation)
        if "open" in d:
            op = d["open"].astype(np.float64)
        else:
            op = np.empty_like(c)
            op[0] = c[0]
            op[1:] = c[:-1]  # open[i] = close[i-1]
        
        data[sym] = {
            "close": c,
            "high": h,
            "low": lo,
            "open": op,
            "volume": d["volume"].astype(np.float64),
        }
        max_bars = max(max_bars, len(d["close"]))

    # Add VR
    for sym, d in data.items():
        c, h, lo = d["close"], d["high"], d["low"]
        d["vr"] = compute_vr(c, h, lo, cfg.vr_short, cfg.vr_long)
        d["_offset"] = max_bars - len(c)

    return data, max_bars


def simulate_straddle_trades(data_15m: dict) -> list[dict]:
    """Simulate straddle trades and collect details."""
    all_trades = []
    warmup = max(cfg.vr_short, cfg.vr_long) + 10

    for sym, d in data_15m.items():
        c = d["close"]
        h = d["high"]
        lo = d["low"]
        op = d["open"]
        vr = d["vr"]
        n = len(c)

        if n < warmup + 50:
            continue

        # Generate straddle signal
        sig = gen_vr_dur_signal(vr, S_VR_THRESH, S_DUR, warmup)

        i = 0
        while i < n - 1:
            if sig[i] == 1:
                entry_bar = i + 1
                if entry_bar >= n:
                    break

                entry_price = c[i]  # entry at signal bar close

                # Straddle: LONG and SHORT at same price
                long_sl = entry_price * (1 - S_SL)
                short_sl = entry_price * (1 + S_SL)

                long_peak = 0.0
                short_peak = 0.0
                long_trail_active = False
                short_trail_active = False
                long_exit_bar = None
                short_exit_bar = None
                long_exit_price = None
                short_exit_price = None
                long_exit_reason = None
                short_exit_reason = None

                for j in range(entry_bar, min(entry_bar + MAX_HOLD, n)):
                    bar_high = h[j]
                    bar_low = lo[j]
                    bar_close = c[j]

                    # LONG leg
                    if long_exit_bar is None:
                        long_pnl = (bar_close - entry_price) / entry_price

                        # SL check (intra-bar)
                        if bar_low <= long_sl:
                            long_exit_bar = j
                            long_exit_price = long_sl
                            long_exit_reason = "SL"
                        else:
                            if long_pnl > long_peak:
                                long_peak = long_pnl
                            if not long_trail_active and long_peak >= S_TRAIL_ACT:
                                long_trail_active = True
                            if long_trail_active:
                                trail_stop = long_peak - S_TRAIL_DIST
                                if long_pnl < trail_stop:
                                    long_exit_bar = j
                                    long_exit_price = bar_close
                                    long_exit_reason = "TRAIL"

                    # SHORT leg
                    if short_exit_bar is None:
                        short_pnl = (entry_price - bar_close) / entry_price

                        # SL check (intra-bar)
                        if bar_high >= short_sl:
                            short_exit_bar = j
                            short_exit_price = short_sl
                            short_exit_reason = "SL"
                        else:
                            if short_pnl > short_peak:
                                short_peak = short_pnl
                            if not short_trail_active and short_peak >= S_TRAIL_ACT:
                                short_trail_active = True
                            if short_trail_active:
                                trail_stop = short_peak - S_TRAIL_DIST
                                if short_pnl < trail_stop:
                                    short_exit_bar = j
                                    short_exit_price = bar_close
                                    short_exit_reason = "TRAIL"

                    # Both exited?
                    if long_exit_bar is not None and short_exit_bar is not None:
                        break

                # Max hold exit
                if long_exit_bar is None:
                    long_exit_bar = min(entry_bar + MAX_HOLD - 1, n - 1)
                    long_exit_price = c[long_exit_bar]
                    long_exit_reason = "MAX_HOLD"
                if short_exit_bar is None:
                    short_exit_bar = min(entry_bar + MAX_HOLD - 1, n - 1)
                    short_exit_price = c[short_exit_bar]
                    short_exit_reason = "MAX_HOLD"

                # Calculate PnL
                long_pnl_final = (long_exit_price - entry_price) / entry_price
                short_pnl_final = (entry_price - short_exit_price) / entry_price
                total_pnl = long_pnl_final + short_pnl_final - COST

                final_exit_bar = max(long_exit_bar, short_exit_bar)

                all_trades.append({
                    "sym": sym,
                    "entry_bar": entry_bar,
                    "entry_price": entry_price,
                    "long_exit_bar": long_exit_bar,
                    "long_exit_price": long_exit_price,
                    "long_exit_reason": long_exit_reason,
                    "long_pnl": long_pnl_final,
                    "short_exit_bar": short_exit_bar,
                    "short_exit_price": short_exit_price,
                    "short_exit_reason": short_exit_reason,
                    "short_pnl": short_pnl_final,
                    "total_pnl": total_pnl,
                    "signal_bar": i,
                    "final_exit_bar": final_exit_bar,
                    "data": d,
                })

                i = final_exit_bar + 1
            else:
                i += 1

    return all_trades


def draw_candlestick(ax, opens, highs, lows, closes, start_idx=0):
    """Draw candlestick chart."""
    width = 0.6
    for i in range(len(closes)):
        x = start_idx + i
        o, h, lo, c = opens[i], highs[i], lows[i], closes[i]

        color = GREEN if c >= o else RED

        # Wick
        ax.plot([x, x], [lo, h], color=color, linewidth=1)

        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.0001:
            body_height = 0.0001
        rect = Rectangle((x - width / 2, body_bottom), width, body_height,
                          facecolor=color, edgecolor=color)
        ax.add_patch(rect)


def create_trade_chart(trade: dict, idx: int, is_winner: bool):
    """Create a candlestick chart for a single trade."""
    sym = trade["sym"]
    d = trade["data"]
    entry_bar = trade["entry_bar"]
    signal_bar = trade["signal_bar"]
    final_exit = trade["final_exit_bar"]
    entry_price = trade["entry_price"]

    # Context: 20 bars before signal, until exit + 10 bars
    start = max(0, signal_bar - 20)
    end = min(len(d["close"]), final_exit + 10)

    c = d["close"][start:end]
    h = d["high"][start:end]
    lo = d["low"][start:end]
    op = d["open"][start:end]

    # Adjust bar indices relative to start
    rel_signal = signal_bar - start
    rel_entry = entry_bar - start
    rel_long_exit = trade["long_exit_bar"] - start
    rel_short_exit = trade["short_exit_bar"] - start

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_DARK)
    ax.set_facecolor(BG_CARD)

    # Draw candlesticks
    draw_candlestick(ax, op, h, lo, c, start_idx=0)

    # Entry line
    ax.axhline(y=entry_price, color=ACCENT, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(x=rel_entry, color=ACCENT, linestyle=':', linewidth=1, alpha=0.5)

    # SL lines
    long_sl = entry_price * (1 - S_SL)
    short_sl = entry_price * (1 + S_SL)
    ax.axhline(y=long_sl, color=RED, linestyle=':', linewidth=1, alpha=0.5, label=f'Long SL ${long_sl:.4f}')
    ax.axhline(y=short_sl, color=RED, linestyle=':', linewidth=1, alpha=0.5, label=f'Short SL ${short_sl:.4f}')

    # Mark entry
    ax.scatter([rel_entry], [entry_price], color=ACCENT, s=150, marker='^', zorder=5,
               label=f'Entry ${entry_price:.4f}')

    # Mark LONG exit
    ax.scatter([rel_long_exit], [trade["long_exit_price"]], color=GREEN, s=150, marker='v', zorder=5,
               label=f'LONG Exit ${trade["long_exit_price"]:.4f} ({trade["long_exit_reason"]})')

    # Mark SHORT exit
    ax.scatter([rel_short_exit], [trade["short_exit_price"]], color='#ff6b6b', s=150, marker='v', zorder=5,
               label=f'SHORT Exit ${trade["short_exit_price"]:.4f} ({trade["short_exit_reason"]})')

    # Signal bar highlight
    ax.axvspan(rel_signal - 0.5, rel_signal + 0.5, alpha=0.2, color=ACCENT)

    # Title (suptitle for main, title for subtitle to avoid overlap)
    status = "WIN" if is_winner else "LOSS"
    color = GREEN if is_winner else RED
    title = f"{sym} Straddle — {status} (PnL: {trade['total_pnl']*100:+.2f}%)"
    fig.suptitle(title, fontsize=14, fontweight='bold', color=color, y=0.98)

    # Subtitle
    subtitle = (f"Entry: ${entry_price:.4f} | "
                f"LONG: {trade['long_pnl']*100:+.2f}% ({trade['long_exit_reason']}) | "
                f"SHORT: {trade['short_pnl']*100:+.2f}% ({trade['short_exit_reason']})")
    ax.set_title(subtitle, fontsize=10, color=TEXT_SUB, pad=5)

    # Axis styling
    ax.set_xlabel('Bar', color=TEXT_SUB)
    ax.set_ylabel('Price', color=TEXT_SUB)
    ax.tick_params(colors=TEXT_SUB)
    for spine in ax.spines.values():
        spine.set_color(TEXT_SUB)

    # Legend
    ax.legend(loc='upper left', facecolor=BG_CARD, edgecolor=TEXT_SUB,
              labelcolor=TEXT_MAIN, fontsize=9)

    # Grid
    ax.grid(True, alpha=0.2, color=TEXT_SUB)

    # Save
    prefix = "win" if is_winner else "loss"
    filename = OUTPUT_DIR / f"{prefix}_{idx+1:02d}_{sym}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor=BG_DARK, bbox_inches='tight')
    plt.close()

    return filename


def main():
    print("Loading data...")
    data_15m, max_bars = load_data()
    print(f"  Symbols: {len(data_15m)}")

    print("Simulating straddle trades...")
    trades = simulate_straddle_trades(data_15m)
    print(f"  Total trades: {len(trades)}")

    # Split winners and losers
    winners = [t for t in trades if t["total_pnl"] > 0]
    losers = [t for t in trades if t["total_pnl"] < 0]

    print(f"  Winners: {len(winners)}, Losers: {len(losers)}")

    # Sort by PnL magnitude for variety
    winners.sort(key=lambda x: x["total_pnl"], reverse=True)
    losers.sort(key=lambda x: x["total_pnl"])

    # Select diverse samples (spread across the list)
    def select_diverse(lst, n):
        if len(lst) <= n:
            return lst
        step = len(lst) // n
        return [lst[i * step] for i in range(n)]

    selected_winners = select_diverse(winners, 10)
    selected_losers = select_diverse(losers, 10)

    print(f"\nGenerating charts...")
    print(f"  Output: {OUTPUT_DIR}")

    # Generate winner charts
    print("\n  Winners:")
    for i, trade in enumerate(selected_winners):
        fname = create_trade_chart(trade, i, is_winner=True)
        print(f"    {i+1}. {trade['sym']}: PnL={trade['total_pnl']*100:+.2f}% → {fname.name}")

    # Generate loser charts
    print("\n  Losers:")
    for i, trade in enumerate(selected_losers):
        fname = create_trade_chart(trade, i, is_winner=False)
        print(f"    {i+1}. {trade['sym']}: PnL={trade['total_pnl']*100:+.2f}% → {fname.name}")

    print(f"\nDone! Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
