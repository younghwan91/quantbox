"""
Elliott Wave Timing Alpha Research

Hypothesis: Time intervals between wave impulse points carry strong predictive
information. Fibonacci ratios govern not just PRICE proportions but also TIME
proportions between waves. Patterns with "correct" timing are more likely to
produce the predicted follow-through move.

Features extracted per pattern:
  Wave-level durations (bars):
    w1_bars, w2_bars, w3_bars, w4_bars, w5_bars, a_bars, b_bars, c_bars
  Time ratios:
    w3_w1_time_ratio  — W3 should take >= W1 time (strong impulse)
    w5_w1_time_ratio  — W5 often shorter than W3 (exhaustion)
    c_a_time_ratio    — C vs A time symmetry
    corr_impulse_ratio — ABC time / Impulse time
  Velocity features:
    w1_velocity, w3_velocity, w5_velocity, a_velocity, c_velocity
    w3_w1_vel_ratio   — acceleration (W3 faster than W1 = strong)
    w5_w3_vel_ratio   — deceleration in W5 = exhaustion signal
    c_a_vel_ratio     — C velocity vs A velocity
  Fibonacci time levels:
    w3_time_fib_0618  — is W3 time near 1.618× W1 time?
    corr_time_fib     — is correction near 0.382× impulse time?

Outcome:
  For each pattern, measure the actual forward move after entry:
    - hit_tp1 (did it reach prior W5?)
    - hit_tp2 (did it reach equal-wave projection?)
    - max_fav_pct (max favorable excursion %)
    - max_adv_pct (max adverse excursion %)
    - trade_return_pct (at exit: TP1/TP2/stop/max_hold)

Then: statistical analysis → feature importance → build timing-enhanced scorer.

Usage:
    python scripts/elliott_wave_timing_research.py
"""

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantbox.strategy.elliott import (
    ElliottConfig,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_stochastic,
    compute_atr,
    zigzag_pivots,
    detect_impulse_abc_patterns,
    scan_multi_scale_patterns,
    detect_bullish_divergence,
    detect_bearish_divergence,
    check_macd_reversal,
    check_stoch_cross,
)

# ── Symbols ──
TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "XLMUSDT", "DOTUSDT", "1000PEPEUSDT", "FILUSDT",
    "ARBUSDT", "FETUSDT", "1000SHIBUSDT", "OPUSDT",
    "ATOMUSDT", "HBARUSDT", "ICPUSDT",
]

CACHE_DIR = Path("data/cache_extended")


def load_4h_data(symbol: str) -> dict | None:
    """Load 4h OHLCV data from .pkl.gz cache.

    Returns dict with 'open', 'high', 'low', 'close', 'volume' numpy arrays.
    """
    pkl_path = CACHE_DIR / f"{symbol}_4h_365d.pkl.gz"
    if pkl_path.exists():
        with gzip.open(pkl_path, "rb") as f:
            df = pickle.load(f)
        return {
            "open": df["Open"].values.astype(np.float64),
            "high": df["High"].values.astype(np.float64),
            "low": df["Low"].values.astype(np.float64),
            "close": df["Close"].values.astype(np.float64),
            "volume": df["Volume"].values.astype(np.float64),
        }

    # Fallback: npz (close only — insufficient for zigzag)
    npz_path = CACHE_DIR / f"{symbol}_4h_365d_20260218.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        if "high" in d and "low" in d:
            return {
                "close": d["close"],
                "high": d["high"],
                "low": d["low"],
                "volume": d.get("volume", np.ones(len(d["close"]))),
            }
    return None


def extract_pattern_timing(pattern: dict, pivot_indices: list[int]) -> dict:
    """Extract timing features from a detected impulse+ABC pattern.

    Args:
        pattern: Pattern dict from detect_impulse_abc_patterns.
        pivot_indices: 9 bar indices [W0, W1, W2, W3, W4, W5, A, B, C].

    Returns:
        Dict of timing features.
    """
    idx = pivot_indices
    prices = [
        pattern["w0"], pattern["w1"], pattern["w2"], pattern["w3"],
        pattern["w4"], pattern["w5"], pattern["a_price"],
        pattern["b_price"], pattern["c_price"],
    ]

    # ── Wave durations (bars) ──
    w1_bars = idx[1] - idx[0]
    w2_bars = idx[2] - idx[1]
    w3_bars = idx[3] - idx[2]
    w4_bars = idx[4] - idx[3]
    w5_bars = idx[5] - idx[4]
    a_bars = idx[6] - idx[5]
    b_bars = idx[7] - idx[6]
    c_bars = idx[8] - idx[7]

    impulse_bars = idx[5] - idx[0]
    correction_bars = idx[8] - idx[5]
    total_bars = idx[8] - idx[0]

    # ── Price moves (absolute) ──
    direction = pattern["signal"]
    if direction == 1:  # Bullish impulse UP
        w1_move = prices[1] - prices[0]
        w2_move = prices[1] - prices[2]
        w3_move = prices[3] - prices[2]
        w4_move = prices[3] - prices[4]
        w5_move = prices[5] - prices[4]
        a_move = prices[5] - prices[6]
        b_move = prices[7] - prices[6]
        c_move = prices[7] - prices[8]
    else:  # Bearish impulse DOWN
        w1_move = prices[0] - prices[1]
        w2_move = prices[2] - prices[1]
        w3_move = prices[2] - prices[3]
        w4_move = prices[4] - prices[3]
        w5_move = prices[4] - prices[5]
        a_move = prices[6] - prices[5]
        b_move = prices[6] - prices[7]
        c_move = prices[8] - prices[7]

    # ── Velocities (price_move / bars) ──
    w1_vel = w1_move / max(w1_bars, 1)
    w3_vel = w3_move / max(w3_bars, 1)
    w5_vel = w5_move / max(w5_bars, 1)
    a_vel = a_move / max(a_bars, 1)
    c_vel = c_move / max(c_bars, 1)

    # ── Time ratios ──
    w3_w1_time_ratio = w3_bars / max(w1_bars, 1)
    w5_w1_time_ratio = w5_bars / max(w1_bars, 1)
    w5_w3_time_ratio = w5_bars / max(w3_bars, 1)
    c_a_time_ratio = c_bars / max(a_bars, 1)
    b_a_time_ratio = b_bars / max(a_bars, 1)
    corr_impulse_ratio = correction_bars / max(impulse_bars, 1)

    # ── Velocity ratios ──
    w3_w1_vel_ratio = w3_vel / max(abs(w1_vel), 1e-10)
    w5_w3_vel_ratio = w5_vel / max(abs(w3_vel), 1e-10)
    w5_w1_vel_ratio = w5_vel / max(abs(w1_vel), 1e-10)
    c_a_vel_ratio = c_vel / max(abs(a_vel), 1e-10)

    # ── Fibonacci time proximity ──
    # W3 near 1.618× W1 time?
    w3_time_fib_dist = abs(w3_w1_time_ratio - 1.618) / 1.618
    # Correction near 0.382× impulse time?
    corr_time_fib_dist = abs(corr_impulse_ratio - 0.382) / 0.382
    # W5/W1 near 0.618 (W5 shorter = exhaustion)?
    w5_w1_fib_dist = abs(w5_w1_time_ratio - 0.618) / 0.618
    # C/A near 1.0 (time symmetry)?
    c_a_fib_dist = abs(c_a_time_ratio - 1.0) / 1.0

    # ── Impulse momentum profile ──
    # Is the impulse accelerating (w3_vel > w1_vel) or decelerating?
    impulse_accel = (w3_vel - w1_vel) / max(abs(w1_vel), 1e-10)
    # Is W5 exhausting (w5_vel < w3_vel)?
    w5_exhaustion = 1.0 if w5_vel < w3_vel * 0.7 else 0.0

    return {
        # Durations
        "w1_bars": w1_bars,
        "w2_bars": w2_bars,
        "w3_bars": w3_bars,
        "w4_bars": w4_bars,
        "w5_bars": w5_bars,
        "a_bars": a_bars,
        "b_bars": b_bars,
        "c_bars": c_bars,
        "impulse_bars": impulse_bars,
        "correction_bars": correction_bars,
        "total_bars": total_bars,
        # Time ratios
        "w3_w1_time_ratio": w3_w1_time_ratio,
        "w5_w1_time_ratio": w5_w1_time_ratio,
        "w5_w3_time_ratio": w5_w3_time_ratio,
        "c_a_time_ratio": c_a_time_ratio,
        "b_a_time_ratio": b_a_time_ratio,
        "corr_impulse_ratio": corr_impulse_ratio,
        # Velocities
        "w1_vel": w1_vel,
        "w3_vel": w3_vel,
        "w5_vel": w5_vel,
        "a_vel": a_vel,
        "c_vel": c_vel,
        # Velocity ratios
        "w3_w1_vel_ratio": w3_w1_vel_ratio,
        "w5_w3_vel_ratio": w5_w3_vel_ratio,
        "w5_w1_vel_ratio": w5_w1_vel_ratio,
        "c_a_vel_ratio": c_a_vel_ratio,
        # Fib time proximity (lower = closer to ideal)
        "w3_time_fib_dist": w3_time_fib_dist,
        "corr_time_fib_dist": corr_time_fib_dist,
        "w5_w1_fib_dist": w5_w1_fib_dist,
        "c_a_fib_dist": c_a_fib_dist,
        # Momentum profile
        "impulse_accel": impulse_accel,
        "w5_exhaustion": w5_exhaustion,
    }


def measure_outcome(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_bar: int,
    direction: int,
    stop: float,
    tp1: float,
    tp2: float,
    max_hold: int = 96,
    atr_trail_mult: float = 1.5,
    atr: np.ndarray | None = None,
    sl_buffer_pct: float = 0.005,
) -> dict:
    """Measure actual trade outcome after pattern detection.

    Simulates the full position lifecycle:
      Entry at close[entry_bar]
      50% partial TP at tp1
      Trail remainder to tp2
      Stop at pattern stop level
      Max hold exit

    Returns dict with outcome metrics.
    """
    n = len(close)
    entry_price = close[entry_bar]

    hit_tp1 = False
    hit_tp2 = False
    partial_done = False
    trail_stop = stop
    best_price = entry_price
    exit_bar = min(entry_bar + max_hold, n - 1)
    exit_price = close[exit_bar]
    exit_reason = "max_hold"

    for bar in range(entry_bar + 1, min(entry_bar + max_hold + 1, n)):
        h = high[bar]
        lo_bar = low[bar]

        if direction == 1:  # Long
            best_price = max(best_price, h)
            # Stop hit
            if lo_bar <= stop if not partial_done else lo_bar <= trail_stop:
                exit_bar = bar
                exit_price = stop if not partial_done else trail_stop
                exit_reason = "stop" if not partial_done else "trail_stop"
                break
            # TP1
            if not partial_done and h >= tp1:
                hit_tp1 = True
                partial_done = True
                trail_stop = entry_price  # move to breakeven
            # TP2
            if partial_done and h >= tp2:
                hit_tp2 = True
                exit_bar = bar
                exit_price = tp2
                exit_reason = "tp2"
                break
            # Update trailing stop
            if partial_done and atr is not None and bar < len(atr):
                new_trail = h - atr_trail_mult * atr[bar]
                if new_trail > trail_stop:
                    trail_stop = new_trail
                if lo_bar <= trail_stop:
                    exit_bar = bar
                    exit_price = trail_stop
                    exit_reason = "trail_stop"
                    break
        else:  # Short
            best_price = min(best_price, lo_bar)
            if h >= stop if not partial_done else h >= trail_stop:
                exit_bar = bar
                exit_price = stop if not partial_done else trail_stop
                exit_reason = "stop" if not partial_done else "trail_stop"
                break
            if not partial_done and lo_bar <= tp1:
                hit_tp1 = True
                partial_done = True
                trail_stop = entry_price
            if partial_done and lo_bar <= tp2:
                hit_tp2 = True
                exit_bar = bar
                exit_price = tp2
                exit_reason = "tp2"
                break
            if partial_done and atr is not None and bar < len(atr):
                new_trail = lo_bar + atr_trail_mult * atr[bar]
                if new_trail < trail_stop or trail_stop == 0:
                    trail_stop = new_trail
                if h >= trail_stop:
                    exit_bar = bar
                    exit_price = trail_stop
                    exit_reason = "trail_stop"
                    break

    # Max favorable/adverse excursion
    if direction == 1:
        max_fav_pct = (best_price - entry_price) / entry_price * 100
        max_adv_pct = (entry_price - min(low[entry_bar + 1:exit_bar + 1].min()
                       if exit_bar > entry_bar else entry_price,
                       entry_price)) / entry_price * 100
        trade_return_pct = (exit_price - entry_price) / entry_price * 100
    else:
        max_fav_pct = (entry_price - best_price) / entry_price * 100
        max_adv_pct = (max(high[entry_bar + 1:exit_bar + 1].max()
                       if exit_bar > entry_bar else entry_price,
                       entry_price) - entry_price) / entry_price * 100
        trade_return_pct = (entry_price - exit_price) / entry_price * 100

    bars_held = exit_bar - entry_bar

    return {
        "hit_tp1": hit_tp1,
        "hit_tp2": hit_tp2,
        "max_fav_pct": max_fav_pct,
        "max_adv_pct": max_adv_pct,
        "trade_return_pct": trade_return_pct,
        "bars_held": bars_held,
        "exit_reason": exit_reason,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


def main():
    print("=" * 100)
    print("  ELLIOTT WAVE TIMING ALPHA RESEARCH")
    print("  Hypothesis: Wave time intervals predict pattern reliability")
    print("=" * 100)

    cfg = ElliottConfig()

    # ── Numba warmup ──
    print("\n⚙ Numba JIT warmup...")
    dummy = np.random.randn(200).cumsum() + 100
    dummy_h, dummy_l = dummy + 1, dummy - 1
    compute_ema(dummy, 10)
    compute_rsi(dummy, 14)
    compute_macd(dummy, 12, 26, 9)
    compute_stochastic(dummy_h, dummy_l, dummy, 14, 3)
    compute_atr(dummy_h, dummy_l, dummy, 14)
    zigzag_pivots(dummy_h, dummy_l, 0.03)
    print("  Done ✓")

    # ── Load data & extract patterns ──
    print(f"\n📥 Loading 4h data for {len(TOP_SYMBOLS)} symbols...")
    all_records = []

    # Load BTC first for macro bias
    btc_data = load_4h_data("BTCUSDT")
    if btc_data is None:
        print("❌ No BTC data. Aborting.")
        return

    btc_close = btc_data["close"]
    btc_ema200 = compute_ema(btc_close, cfg.ema_macro_fast)
    btc_ema500 = compute_ema(btc_close, cfg.ema_macro_slow)

    for symbol in TOP_SYMBOLS:
        data = load_4h_data(symbol)
        if data is None:
            print(f"  {symbol}: ❌ no OHLCV data")
            continue

        close = data["close"]
        high = data["high"]
        low = data["low"]
        n = len(close)
        print(f"  {symbol}: {n} bars", end="")

        # Align with BTC
        min_len = min(n, len(btc_close))
        close_a = close[-min_len:]
        high_a = high[-min_len:]
        low_a = low[-min_len:]
        btc_c = btc_close[-min_len:]
        btc_e200 = btc_ema200[-min_len:]
        btc_e500 = btc_ema500[-min_len:]

        # Indicators
        rsi_4h = compute_rsi(close_a, cfg.rsi_period)
        _, _, macd_hist = compute_macd(
            close_a, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
        )
        stoch_k, stoch_d = compute_stochastic(
            high_a, low_a, close_a, cfg.stoch_k_period, cfg.stoch_d_period,
        )
        atr_4h = compute_atr(high_a, low_a, close_a, 14)

        # Multi-scale pattern detection
        patterns = scan_multi_scale_patterns(high_a, low_a, cfg)
        print(f"  → {len(patterns)} patterns")

        if not patterns:
            continue

        # Need to re-detect to get pivot_indices per pattern
        # Do per-threshold zigzag and match
        for thresh in cfg.zigzag_thresholds:
            pidx, pprice, ptype, n_piv = zigzag_pivots(high_a, low_a, thresh)
            if n_piv < 9:
                continue

            # Walk 9-pivot windows
            for i in range(n_piv - 8):
                # Check type sequence
                is_bull = True
                is_bear = True
                expected_bull = -1
                expected_bear = 1
                for k in range(9):
                    if ptype[i + k] != expected_bull:
                        is_bull = False
                    if ptype[i + k] != expected_bear:
                        is_bear = False
                    expected_bull = -expected_bull
                    expected_bear = -expected_bear

                if not is_bull and not is_bear:
                    continue

                p = [float(pprice[i + j]) for j in range(9)]
                idx_arr = [int(pidx[i + j]) for j in range(9)]
                c_idx = idx_arr[8]

                # Validate
                from quantbox.strategy.elliott import (
                    _validate_bullish_impulse_abc,
                    _validate_bearish_impulse_abc,
                )

                if is_bull:
                    pattern = _validate_bullish_impulse_abc(
                        *p, c_idx, cfg, idx_arr,
                    )
                elif is_bear:
                    pattern = _validate_bearish_impulse_abc(
                        *p, c_idx, cfg, idx_arr,
                    )
                else:
                    continue

                if pattern is None:
                    continue

                # ── Extract timing features ──
                timing = extract_pattern_timing(pattern, idx_arr)

                # ── Entry bar: c_idx + 1 (first bar after C is confirmed) ──
                entry_bar = c_idx + 1
                if entry_bar >= min_len - 10:
                    continue  # Not enough forward data

                # ── Macro bias at entry ──
                btc_p = btc_c[entry_bar] if entry_bar < min_len else btc_c[-1]
                above_200 = btc_p > btc_e200[entry_bar]
                above_500 = btc_p > btc_e500[entry_bar]
                if above_200 and above_500:
                    bias = "long"
                elif not above_200 and not above_500:
                    bias = "short"
                else:
                    bias = "neutral"

                # Bias alignment
                direction = pattern["signal"]
                if bias == "long" and direction != 1:
                    bias_aligned = False
                elif bias == "short" and direction != -1:
                    bias_aligned = False
                else:
                    bias_aligned = True

                # ── Confluence ──
                confirmations = 0
                lb = min(60, entry_bar)
                if direction == 1:
                    if detect_bullish_divergence(close_a[:entry_bar+1], rsi_4h[:entry_bar+1], lb):
                        confirmations += 1
                else:
                    if detect_bearish_divergence(close_a[:entry_bar+1], rsi_4h[:entry_bar+1], lb):
                        confirmations += 1
                if check_macd_reversal(macd_hist[:entry_bar+1], direction):
                    confirmations += 1
                if check_stoch_cross(stoch_k[:entry_bar+1], stoch_d[:entry_bar+1], direction):
                    confirmations += 1

                # ── Risk check ──
                entry_price = float(close_a[entry_bar])
                stop_price = float(pattern["stop"])
                risk_pct = abs(entry_price - stop_price) / entry_price

                # ── Measure outcome ──
                outcome = measure_outcome(
                    close_a, high_a, low_a,
                    entry_bar, direction,
                    stop_price,
                    float(pattern["target_1"]),
                    float(pattern["target_2"]),
                    max_hold=cfg.max_hold_bars,
                    atr_trail_mult=cfg.trail_atr_mult,
                    atr=atr_4h,
                )

                # ── Combine record ──
                record = {
                    "symbol": symbol,
                    "direction": direction,
                    "c_idx": c_idx,
                    "entry_bar": entry_bar,
                    "thresh": thresh,
                    "quality": pattern["quality"],
                    "w2_retrace": pattern["w2_retrace"],
                    "w3_extension": pattern["w3_extension"],
                    "b_retrace": pattern["b_retrace"],
                    "c_vs_a": pattern["c_vs_a"],
                    "c_fib_dist": pattern["c_fib_dist"],
                    "risk_pct": risk_pct,
                    "bias": bias,
                    "bias_aligned": bias_aligned,
                    "confirmations": confirmations,
                    **timing,
                    **outcome,
                }
                all_records.append(record)

    # ── Analysis ──
    if not all_records:
        print("\n❌ No patterns found. Check data.")
        return

    df = pd.DataFrame(all_records)

    # Deduplicate by (symbol, c_idx) — keep highest quality
    df = df.sort_values("quality", ascending=False).drop_duplicates(
        subset=["symbol", "c_idx"], keep="first",
    ).sort_values(["symbol", "c_idx"]).reset_index(drop=True)

    print(f"\n{'=' * 100}")
    print(f"  PATTERN STATISTICS")
    print(f"{'=' * 100}")
    print(f"  Total patterns detected: {len(df)}")
    print(f"  Symbols with patterns: {df['symbol'].nunique()}")
    print(f"  Bullish / Bearish: {(df['direction']==1).sum()} / {(df['direction']==-1).sum()}")
    print(f"  TP1 hit rate: {df['hit_tp1'].mean()*100:.1f}%")
    print(f"  TP2 hit rate: {df['hit_tp2'].mean()*100:.1f}%")
    print(f"  Avg return: {df['trade_return_pct'].mean():+.2f}%")
    print(f"  Win rate (return > 0): {(df['trade_return_pct']>0).mean()*100:.1f}%")
    print(f"  Avg max favorable: {df['max_fav_pct'].mean():.2f}%")
    print(f"  Avg max adverse: {df['max_adv_pct'].mean():.2f}%")

    # ── Exit reason distribution ──
    print(f"\n  Exit reasons:")
    for reason, count in df["exit_reason"].value_counts().items():
        avg_ret = df[df["exit_reason"] == reason]["trade_return_pct"].mean()
        print(f"    {reason:<15} {count:>5} trades  avg return {avg_ret:+.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    #  TIMING FEATURE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  TIMING FEATURE ANALYSIS — Correlation with Trade Return")
    print(f"{'=' * 100}")

    timing_features = [
        "w1_bars", "w2_bars", "w3_bars", "w4_bars", "w5_bars",
        "a_bars", "b_bars", "c_bars",
        "impulse_bars", "correction_bars", "total_bars",
        "w3_w1_time_ratio", "w5_w1_time_ratio", "w5_w3_time_ratio",
        "c_a_time_ratio", "b_a_time_ratio", "corr_impulse_ratio",
        "w3_w1_vel_ratio", "w5_w3_vel_ratio", "w5_w1_vel_ratio",
        "c_a_vel_ratio",
        "w3_time_fib_dist", "corr_time_fib_dist",
        "w5_w1_fib_dist", "c_a_fib_dist",
        "impulse_accel", "w5_exhaustion",
    ]

    # Correlation with trade return
    print(f"\n  {'Feature':<25} {'Corr(ret)':>10} {'Corr(TP1)':>10} {'Corr(WR)':>10} {'Mean(W)':>10} {'Mean(L)':>10} {'Diff':>10}")
    print(f"  {'-'*85}")

    df["winner"] = (df["trade_return_pct"] > 0).astype(float)
    winners = df[df["winner"] == 1]
    losers = df[df["winner"] == 0]

    feature_scores = {}
    for feat in timing_features:
        if feat not in df.columns or df[feat].std() == 0:
            continue
        corr_ret = df[feat].corr(df["trade_return_pct"])
        corr_tp1 = df[feat].corr(df["hit_tp1"].astype(float))
        corr_wr = df[feat].corr(df["winner"])
        mean_w = winners[feat].mean() if len(winners) > 0 else 0
        mean_l = losers[feat].mean() if len(losers) > 0 else 0
        diff = mean_w - mean_l

        # Score: sum of absolute correlations (higher = more predictive)
        score = abs(corr_ret) + abs(corr_tp1) + abs(corr_wr)
        feature_scores[feat] = score

        marker = "  ★" if score > 0.15 else ""
        print(f"  {feat:<25} {corr_ret:>+10.3f} {corr_tp1:>+10.3f} {corr_wr:>+10.3f} "
              f"{mean_w:>10.3f} {mean_l:>10.3f} {diff:>+10.3f}{marker}")

    # ── Top features ──
    sorted_features = sorted(feature_scores.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 predictive timing features:")
    for rank, (feat, score) in enumerate(sorted_features[:10], 1):
        print(f"    {rank}. {feat} (score: {score:.3f})")

    # ═══════════════════════════════════════════════════════════════════
    #  TIMING-BASED FILTER EXPERIMENTS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  TIMING FILTER EXPERIMENTS")
    print(f"{'=' * 100}")

    # Baseline: all patterns
    def eval_subset(label, mask):
        sub = df[mask]
        n = len(sub)
        if n == 0:
            print(f"  {label:<50} {'N/A':>8}")
            return
        wr = (sub["trade_return_pct"] > 0).mean() * 100
        avg_ret = sub["trade_return_pct"].mean()
        tp1_rate = sub["hit_tp1"].mean() * 100
        avg_fav = sub["max_fav_pct"].mean()
        avg_adv = sub["max_adv_pct"].mean()
        edge = avg_ret * n  # total return points
        print(f"  {label:<50} n={n:>4} WR={wr:>5.1f}% ret={avg_ret:>+6.2f}% "
              f"TP1={tp1_rate:>5.1f}% fav={avg_fav:>5.2f}% adv={avg_adv:>5.2f}% edge={edge:>+7.1f}")

    eval_subset("Baseline (all patterns)", pd.Series(True, index=df.index))
    eval_subset("Baseline + bias aligned", df["bias_aligned"])
    eval_subset("Baseline + confirmations>=2", df["confirmations"] >= 2)
    eval_subset("Baseline + quality>=40", df["quality"] >= 40)
    eval_subset("Baseline + risk<=3%", df["risk_pct"] <= 0.03)

    print()

    # ── Experiment 1: Wave 3 time vs Wave 1 time ──
    print("  ── Experiment 1: W3/W1 Time Ratio ──")
    for lo, hi, label in [
        (0, 0.8, "W3 faster than W1 (<0.8)"),
        (0.8, 1.5, "W3 similar to W1 (0.8-1.5)"),
        (1.5, 3.0, "W3 slower than W1 (1.5-3.0)"),
        (3.0, 100, "W3 much slower (>3.0)"),
    ]:
        mask = (df["w3_w1_time_ratio"] >= lo) & (df["w3_w1_time_ratio"] < hi)
        eval_subset(f"  {label}", mask)

    print()

    # ── Experiment 2: Correction/Impulse time ratio ──
    print("  ── Experiment 2: Correction/Impulse Time Ratio ──")
    for lo, hi, label in [
        (0, 0.25, "Quick correction (<25%)"),
        (0.25, 0.40, "Fibonacci correction (25-40%)"),
        (0.40, 0.62, "Standard correction (40-62%)"),
        (0.62, 1.0, "Extended correction (62-100%)"),
        (1.0, 100, "Very long correction (>100%)"),
    ]:
        mask = (df["corr_impulse_ratio"] >= lo) & (df["corr_impulse_ratio"] < hi)
        eval_subset(f"  {label}", mask)

    print()

    # ── Experiment 3: W3 velocity acceleration ──
    print("  ── Experiment 3: Impulse Acceleration (W3 vs W1 velocity) ──")
    for lo, hi, label in [
        (-100, 0, "Decelerating (W3 slower than W1)"),
        (0, 0.5, "Mild acceleration (0-50%)"),
        (0.5, 1.5, "Strong acceleration (50-150%)"),
        (1.5, 100, "Extreme acceleration (>150%)"),
    ]:
        mask = (df["impulse_accel"] >= lo) & (df["impulse_accel"] < hi)
        eval_subset(f"  {label}", mask)

    print()

    # ── Experiment 4: W5 exhaustion ──
    print("  ── Experiment 4: W5 Exhaustion (W5 vel < 70% of W3 vel) ──")
    eval_subset("  W5 exhaustion = True", df["w5_exhaustion"] == 1)
    eval_subset("  W5 exhaustion = False", df["w5_exhaustion"] == 0)

    print()

    # ── Experiment 5: C/A time symmetry ──
    print("  ── Experiment 5: C/A Time Symmetry ──")
    for lo, hi, label in [
        (0, 0.5, "C much faster than A (<0.5)"),
        (0.5, 0.8, "C slightly faster (0.5-0.8)"),
        (0.8, 1.2, "Symmetric C ~ A (0.8-1.2)"),
        (1.2, 2.0, "C slower than A (1.2-2.0)"),
        (2.0, 100, "C much slower (>2.0)"),
    ]:
        mask = (df["c_a_time_ratio"] >= lo) & (df["c_a_time_ratio"] < hi)
        eval_subset(f"  {label}", mask)

    print()

    # ── Experiment 6: Fibonacci time proximity ──
    print("  ── Experiment 6: W3/W1 Fib Time Distance (from 1.618) ──")
    for lo, hi, label in [
        (0, 0.3, "Close to 1.618× (dist < 0.3)"),
        (0.3, 0.6, "Moderate distance (0.3-0.6)"),
        (0.6, 100, "Far from 1.618× (dist > 0.6)"),
    ]:
        mask = (df["w3_time_fib_dist"] >= lo) & (df["w3_time_fib_dist"] < hi)
        eval_subset(f"  {label}", mask)

    print()

    # ── Experiment 7: Total pattern duration ──
    print("  ── Experiment 7: Total Pattern Duration (bars) ──")
    for lo, hi, label in [
        (0, 20, "Very short (<20 bars = <3.3d)"),
        (20, 50, "Short (20-50 bars = 3-8d)"),
        (50, 100, "Medium (50-100 bars = 8-17d)"),
        (100, 200, "Long (100-200 bars = 17-33d)"),
        (200, 10000, "Very long (>200 bars = >33d)"),
    ]:
        mask = (df["total_bars"] >= lo) & (df["total_bars"] < hi)
        eval_subset(f"  {label}", mask)

    # ═══════════════════════════════════════════════════════════════════
    #  COMPOSITE TIMING SCORER
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  COMPOSITE TIMING SCORE — A/B vs Baseline")
    print(f"{'=' * 100}")

    # Build a composite timing score based on top features
    # Higher score = features more consistent with winning patterns
    def compute_timing_score(row):
        score = 0.0

        # W3/W1 time ratio: 1.0-3.0 is ideal (strong W3, not micro-noise)
        if 1.0 <= row["w3_w1_time_ratio"] <= 3.0:
            score += 10
        elif 0.5 <= row["w3_w1_time_ratio"] < 1.0:
            score += 5

        # Correction/impulse time: 0.25-0.62 is classic (Fibonacci zone)
        if 0.25 <= row["corr_impulse_ratio"] <= 0.62:
            score += 10
        elif 0.15 <= row["corr_impulse_ratio"] < 0.25:
            score += 5

        # W5 exhaustion: W5 velocity << W3 = trend running out of steam
        # This means the corrective phase has a better chance of being the END
        if row["w5_exhaustion"] == 1:
            score += 8

        # Impulse acceleration: W3 faster than W1 = strong momentum
        if row["impulse_accel"] > 0:
            score += 5
            if row["impulse_accel"] > 0.5:
                score += 3

        # C/A time symmetry: 0.5-1.5 = well-formed correction
        if 0.5 <= row["c_a_time_ratio"] <= 1.5:
            score += 7

        # W3 time near 1.618× W1 (Fib time symmetry)
        if row["w3_time_fib_dist"] < 0.3:
            score += 8
        elif row["w3_time_fib_dist"] < 0.6:
            score += 4

        # Total duration: not too short (noise), not too long (stale)
        if 20 <= row["total_bars"] <= 200:
            score += 5

        return score

    df["timing_score"] = df.apply(compute_timing_score, axis=1)

    # Quantile splits
    q33 = df["timing_score"].quantile(0.33)
    q66 = df["timing_score"].quantile(0.66)

    print(f"\n  Timing score distribution: min={df['timing_score'].min():.0f}, "
          f"median={df['timing_score'].median():.0f}, max={df['timing_score'].max():.0f}, "
          f"q33={q33:.0f}, q66={q66:.0f}")

    print()
    eval_subset("A) LOW timing score (bottom 33%)", df["timing_score"] <= q33)
    eval_subset("B) MID timing score (33-66%)", (df["timing_score"] > q33) & (df["timing_score"] <= q66))
    eval_subset("C) HIGH timing score (top 33%)", df["timing_score"] > q66)
    print()

    # Combined: quality >= 40 AND high timing score AND bias aligned AND conf>=2
    combined_mask = (
        (df["quality"] >= 40) &
        (df["timing_score"] > q66) &
        (df["bias_aligned"]) &
        (df["confirmations"] >= 2) &
        (df["risk_pct"] <= 0.03)
    )
    eval_subset("COMBINED: quality+timing+bias+conf+risk", combined_mask)

    # Quality only (no timing)
    quality_only = (
        (df["quality"] >= 40) &
        (df["bias_aligned"]) &
        (df["confirmations"] >= 2) &
        (df["risk_pct"] <= 0.03)
    )
    eval_subset("NO-TIMING: quality+bias+conf+risk (control)", quality_only)

    # Timing-only (relax other filters)
    timing_only = df["timing_score"] > q66
    eval_subset("TIMING-ONLY: high timing score only", timing_only)

    # ═══════════════════════════════════════════════════════════════════
    #  SWEEP: Find optimal timing thresholds
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  TIMING THRESHOLD SWEEP")
    print(f"{'=' * 100}")

    timing_thresholds = list(range(0, int(df["timing_score"].max()) + 5, 5))
    print(f"\n  {'MinScore':>10} {'N':>5} {'WR%':>7} {'AvgRet%':>9} {'TP1%':>7} {'Edge':>8} {'AvgFav%':>9} {'AvgAdv%':>9}")
    print(f"  {'-'*70}")

    best_edge = -999
    best_threshold = 0
    for min_score in timing_thresholds:
        mask = df["timing_score"] >= min_score
        sub = df[mask]
        n = len(sub)
        if n < 5:
            continue
        wr = (sub["trade_return_pct"] > 0).mean() * 100
        avg_ret = sub["trade_return_pct"].mean()
        tp1_rate = sub["hit_tp1"].mean() * 100
        avg_fav = sub["max_fav_pct"].mean()
        avg_adv = sub["max_adv_pct"].mean()
        edge = avg_ret * n
        marker = " ★" if avg_ret > 0 and wr > 55 else ""
        print(f"  {min_score:>10} {n:>5} {wr:>7.1f} {avg_ret:>+9.2f} {tp1_rate:>7.1f} "
              f"{edge:>+8.1f} {avg_fav:>9.2f} {avg_adv:>9.2f}{marker}")
        # Best edge per trade with at least moderate sample
        if avg_ret > best_edge and n >= 10:
            best_edge = avg_ret
            best_threshold = min_score

    print(f"\n  Best threshold: timing_score >= {best_threshold} (edge: {best_edge:+.2f}%/trade)")

    # ═══════════════════════════════════════════════════════════════════
    #  SAVE RAW DATA
    # ═══════════════════════════════════════════════════════════════════
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "elliott_timing_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Raw data saved to {csv_path} ({len(df)} patterns)")

    print(f"\n{'=' * 100}")
    print(f"  DONE — Next steps:")
    print(f"  1. Review top features and their predictive power")
    print(f"  2. Build timing-enhanced quality scorer in elliott.py")
    print(f"  3. Run walk-forward backtest with timing filter")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
