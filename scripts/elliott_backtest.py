"""
Elliott Wave Strategy — 1-Year Backtest on Real Binance 4h Data

Downloads 365 days of 4h OHLCV klines from Binance Futures,
runs compute_elliott_signals(), then evaluates with VectorizedBacktest.

Usage:
    python scripts/elliott_backtest.py
    python scripts/elliott_backtest.py --days 180 --symbols ETHUSDT SOLUSDT
    python scripts/elliott_backtest.py --no-divergence  # relax divergence filter
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from binance.client import Client

from quantbox.strategy.elliott import (
    ElliottConfig,
    compute_elliott_signals,
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
    detect_triangle,
    check_macd_reversal,
    check_stoch_cross,
)
from quantbox.backtest.vectorized_backtest import (
    VectorizedBacktest,
    BacktestConfig,
)


# ── Top symbols from whitelist (same as live trading pool) ──
TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "XLMUSDT", "DOTUSDT", "1000PEPEUSDT", "FILUSDT",
    "ARBUSDT", "FETUSDT", "1000SHIBUSDT", "OPUSDT",
    "ATOMUSDT", "HBARUSDT", "ICPUSDT",
]


def download_4h_klines(
    client: Client,
    symbol: str,
    days: int = 365,
) -> dict | None:
    """Download 4h OHLCV data from Binance Futures.

    Returns dict with numpy arrays: close, high, low, volume, open_time.
    Returns None on failure.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_klines = []
    current_ms = start_ms

    while current_ms < end_ms:
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval="4h",
                startTime=current_ms,
                endTime=end_ms,
                limit=1000,
            )
            if not klines:
                break
            all_klines.extend(klines)
            # Next batch starts after the last kline close time
            current_ms = int(klines[-1][6]) + 1
            if len(klines) < 1000:
                break
            time.sleep(0.1)  # Rate limit
        except Exception as e:
            print(f"  ⚠ Error downloading {symbol}: {e}")
            return None

    if not all_klines:
        return None

    return {
        "open_time": np.array([int(k[0]) for k in all_klines]),
        "open": np.array([float(k[1]) for k in all_klines], dtype=np.float64),
        "high": np.array([float(k[2]) for k in all_klines], dtype=np.float64),
        "low": np.array([float(k[3]) for k in all_klines], dtype=np.float64),
        "close": np.array([float(k[4]) for k in all_klines], dtype=np.float64),
        "volume": np.array([float(k[5]) for k in all_klines], dtype=np.float64),
    }


def compute_signals_relaxed(
    close_4h: np.ndarray,
    high_4h: np.ndarray,
    low_4h: np.ndarray,
    btc_close_4h: np.ndarray,
    config: ElliottConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Position signals with full lifecycle — unified 2-of-3 confluence.

    Same position management as compute_elliott_signals (stops, partial TP,
    trailing, max hold). Requires ≥ 2 of {divergence, MACD, Stoch} for entry.
    3/3 → instant primary entry; 2/3 → exploratory with confirmation window.

    Position States (winning_strategy §4-§5):
      Entry:      exploratory 0.4x (2%) → primary 2.0x (10%) on confirmation
      Partial TP: 50% exit at TP1 → 1.0x remaining
      Trail:      ATR trailing stop from breakeven
      Exit:       stop (tight at C) / TP2 / trail hit / max hold

    Neutral BTC bias (between 200/500 EMA) allows both long and short.

    Returns:
        (signals, adj_close) — adj_close has fill prices on exit bars
        for accurate PnL computation in the backtest engine.
    """
    cfg = config or ElliottConfig()
    n = len(close_4h)
    signals = np.zeros(n, dtype=np.float64)
    adj_close = close_4h.copy()  # adjusted close for accurate stop/TP fills

    if n < cfg.ema_macro_slow + 50:
        return signals, adj_close

    # ── Pre-compute indicators ──
    btc_ema200 = compute_ema(btc_close_4h, cfg.ema_macro_fast)
    btc_ema500 = compute_ema(btc_close_4h, cfg.ema_macro_slow)
    rsi_4h = compute_rsi(close_4h, cfg.rsi_period)
    _, _, macd_hist = compute_macd(
        close_4h, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
    )
    stoch_k, stoch_d = compute_stochastic(
        high_4h, low_4h, close_4h, cfg.stoch_k_period, cfg.stoch_d_period,
    )
    atr_4h = compute_atr(high_4h, low_4h, close_4h, 14)

    min_start = cfg.ema_macro_slow + 50
    lookback_bars = cfg.pattern_lookback_bars

    # Position sizing (§4.1-§4.2)
    lev_exploratory = cfg.nominal_leverage * cfg.exploratory_pct  # 0.4
    lev_primary = cfg.effective_leverage                          # 2.0 (fallback)
    max_lev = cfg.effective_leverage                              # 2.0 cap
    # Per-trade vars initialized; overridden at each entry
    trade_lev = lev_primary
    trade_lev_after_tp = trade_lev * cfg.partial_tp_fraction      # 1.0
    max_hold = cfg.max_hold_bars

    # ── Pre-compute patterns ──
    all_patterns = scan_multi_scale_patterns(high_4h, low_4h, cfg)
    all_patterns.sort(key=lambda p: p["c_idx"])

    # ── Pre-compute divergence arrays for counter-divergence exits ──
    div_lookback = cfg.divergence_lookback
    bull_div = np.zeros(n, dtype=np.bool_)
    bear_div = np.zeros(n, dtype=np.bool_)
    for i in range(min_start, n):
        lb = min(div_lookback, i)
        bull_div[i] = detect_bullish_divergence(
            close_4h[:i + 1], rsi_4h[:i + 1], lb,
        )
        bear_div[i] = detect_bearish_divergence(
            close_4h[:i + 1], rsi_4h[:i + 1], lb,
        )

    # ── Walk-forward with position lifecycle ──
    pat_ptr = 0
    used_patterns: set[int] = set()

    # Position state
    in_position = False
    pos_direction = 0
    pos_stop = 0.0
    pos_tp1 = 0.0
    pos_tp2 = 0.0
    pos_entry_price = 0.0
    pos_entry_bar = 0
    pos_confirmed = False
    pos_partial_done = False
    pos_trail_stop = 0.0

    for bar in range(min_start, n):
        # ── Manage existing position ──
        if in_position:
            current_lev = signals[bar - 1]  # carry forward
            hit_stop = False
            hit_tp1 = False
            hit_tp2 = False
            bars_held = bar - pos_entry_bar
            fill_price = close_4h[bar]  # default fill at close

            if pos_direction == 1:  # Long
                if low_4h[bar] <= pos_stop:
                    hit_stop = True
                    fill_price = pos_stop
                if not pos_partial_done and high_4h[bar] >= pos_tp1:
                    hit_tp1 = True
                if pos_partial_done and high_4h[bar] >= pos_tp2:
                    hit_tp2 = True
                    fill_price = pos_tp2
                if pos_partial_done:
                    new_trail = high_4h[bar] - cfg.trail_atr_mult * atr_4h[bar]
                    if new_trail > pos_trail_stop:
                        pos_trail_stop = new_trail
                    if low_4h[bar] <= pos_trail_stop:
                        hit_stop = True
                        fill_price = pos_trail_stop
            else:  # Short
                if high_4h[bar] >= pos_stop:
                    hit_stop = True
                    fill_price = pos_stop
                if not pos_partial_done and low_4h[bar] <= pos_tp1:
                    hit_tp1 = True
                if pos_partial_done and low_4h[bar] <= pos_tp2:
                    hit_tp2 = True
                    fill_price = pos_tp2
                if pos_partial_done:
                    new_trail = low_4h[bar] + cfg.trail_atr_mult * atr_4h[bar]
                    if new_trail < pos_trail_stop or pos_trail_stop == 0.0:
                        pos_trail_stop = new_trail
                    if high_4h[bar] >= pos_trail_stop:
                        hit_stop = True
                        fill_price = pos_trail_stop

            # Exploratory → primary confirmation (§4.2)
            if not pos_confirmed and bars_held <= cfg.confirmation_bars:
                has_macd = check_macd_reversal(macd_hist[:bar + 1], pos_direction)
                has_stoch = check_stoch_cross(
                    stoch_k[:bar + 1], stoch_d[:bar + 1], pos_direction,
                )
                if has_macd or has_stoch:
                    pos_confirmed = True
                    current_lev = float(pos_direction) * trade_lev
            elif not pos_confirmed and bars_held > cfg.confirmation_bars:
                # No confirmation → exit failed pilot
                signals[bar] = 0.0
                in_position = False
                continue

            # Process exits (priority order)
            if hit_stop:
                signals[bar] = 0.0
                adj_close[bar] = fill_price  # accurate stop fill
                in_position = False
                continue
            if hit_tp2:
                signals[bar] = 0.0
                adj_close[bar] = fill_price  # accurate TP2 fill
                in_position = False
                continue
            if hit_tp1 and not pos_partial_done:
                pos_partial_done = True
                current_lev = float(pos_direction) * trade_lev_after_tp
                pos_stop = pos_entry_price  # move to breakeven
                pos_trail_stop = pos_entry_price
                signals[bar] = current_lev
                continue
            if bars_held >= max_hold:
                signals[bar] = 0.0
                in_position = False
                continue

            # Hold position
            signals[bar] = current_lev
            continue

        # ── Not in position: look for entry ──
        while pat_ptr < len(all_patterns) and all_patterns[pat_ptr]["c_idx"] < bar - lookback_bars:
            pat_ptr += 1

        recent = []
        for j in range(pat_ptr, len(all_patterns)):
            p = all_patterns[j]
            if p["c_idx"] > bar:
                break
            if p["c_idx"] >= bar - lookback_bars:
                recent.append(p)

        if not recent:
            continue

        pattern = recent[-1]
        if pattern["c_idx"] in used_patterns:
            continue

        # Macro bias (§2.1) — neutral allows both
        btc_price = float(btc_close_4h[bar])
        above_200 = btc_price > float(btc_ema200[bar])
        above_500 = btc_price > float(btc_ema500[bar])

        if above_200 and above_500:
            bias = "long"
        elif not above_200 and not above_500:
            bias = "short"
        else:
            bias = "neutral"

        if bias == "long" and pattern["signal"] != 1:
            continue
        if bias == "short" and pattern["signal"] != -1:
            continue

        direction = pattern["signal"]

        # Confluence: need at least 1 of {divergence, MACD, Stoch}
        confirmations = 0
        lookback = min(60, bar)

        if direction == 1:
            if detect_bullish_divergence(close_4h[:bar+1], rsi_4h[:bar+1], lookback):
                confirmations += 1
        else:
            if detect_bearish_divergence(close_4h[:bar+1], rsi_4h[:bar+1], lookback):
                confirmations += 1

        if check_macd_reversal(macd_hist[:bar+1], direction):
            confirmations += 1
        if check_stoch_cross(stoch_k[:bar+1], stoch_d[:bar+1], direction):
            confirmations += 1

        if confirmations < 2:
            continue

        # ── Risk filter: only enter if stop distance from entry is tight ──
        entry_price = float(close_4h[bar])
        stop_price = float(pattern["stop"])
        risk_pct = abs(entry_price - stop_price) / entry_price
        if risk_pct > cfg.max_entry_risk_pct:
            continue  # entry too far from C → poor RR → skip

        # ── Quality filter: only trade high-quality wave structures ──
        pattern_quality = pattern.get("quality", 0.0)
        if pattern_quality < cfg.min_pattern_quality:
            continue  # low quality wave reading → skip

        # ── ENTER: exploratory entry (§4.2: 2% pilot) ──
        used_patterns.add(pattern["c_idx"])
        in_position = True
        pos_direction = direction
        pos_entry_price = entry_price
        pos_entry_bar = bar
        pos_partial_done = False
        pos_stop = float(pattern["stop"])
        pos_tp1 = float(pattern["target_1"])
        pos_tp2 = float(pattern["target_2"])
        pos_trail_stop = pos_stop

        # Risk-sized leverage: ensure each trade risks exactly risk_per_trade
        if cfg.risk_sized_leverage and risk_pct > 0:
            trade_lev = min(max_lev, cfg.risk_per_trade / risk_pct)
        else:
            trade_lev = lev_primary
        trade_lev_after_tp = trade_lev * cfg.partial_tp_fraction

        # Enter exploratory (0.4x), confirm to primary (risk-sized) within N bars
        # If we already have 2+ confirmations → instant primary
        if confirmations >= 2:
            pos_confirmed = True
            signals[bar] = float(direction) * trade_lev
        else:
            pos_confirmed = False
            signals[bar] = float(direction) * lev_exploratory

    return signals, adj_close


def run_backtest(
    symbol_data: dict[str, dict],
    btc_data: dict,
    config: ElliottConfig,
    bt_config: BacktestConfig,
    use_relaxed: bool = False,
) -> dict[str, dict]:
    """Run backtest across all symbols.

    Returns dict[symbol] -> {result, signals, n_long, n_short}.
    """
    bt = VectorizedBacktest(bt_config)
    results = {}
    btc_close = btc_data["close"]

    for symbol, data in symbol_data.items():
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Align lengths (use shorter of symbol vs btc)
        min_len = min(len(close), len(btc_close))
        c = close[-min_len:]
        h = high[-min_len:]
        lo = low[-min_len:]
        bc = btc_close[-min_len:]

        if use_relaxed:
            signals, adj_close = compute_signals_relaxed(c, h, lo, bc, config)
        else:
            signals, adj_close = compute_elliott_signals(c, h, lo, bc, config)

        n_long = int(np.sum(signals > 0))
        n_short = int(np.sum(signals < 0))

        if n_long + n_short == 0:
            results[symbol] = {
                "result": None,
                "signals": signals,
                "n_long": 0,
                "n_short": 0,
            }
            continue

        # Use adj_close for accurate intra-bar fill PnL
        result = bt.run_single_partial(adj_close, signals, symbol=symbol)
        results[symbol] = {
            "result": result,
            "signals": signals,
            "n_long": n_long,
            "n_short": n_short,
        }

    return results


def print_report(
    results: dict[str, dict],
    days: int,
    mode: str,
):
    """Print formatted backtest report."""
    print("\n" + "=" * 90)
    print(f"  ELLIOTT WAVE BACKTEST — {days} Days — Mode: {mode}")
    print("=" * 90)

    header = (
        f"{'Symbol':<12} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} "
        f"{'Calmar':>8} {'WR%':>6} {'Trades':>7} {'PF':>6} "
        f"{'Long':>5} {'Short':>5}"
    )
    print(header)
    print("-" * 90)

    total_return = 0.0
    total_trades = 0
    total_wins = 0
    active_symbols = 0
    max_dd_worst = 0.0
    equity_curves = []

    for symbol in sorted(results.keys()):
        r = results[symbol]
        res = r["result"]
        if res is None:
            print(f"{symbol:<12} {'no signals':>10}")
            continue

        active_symbols += 1
        total_return += res.total_return_pct
        total_trades += res.n_trades
        total_wins += int(res.win_rate / 100.0 * res.n_trades)
        max_dd_worst = max(max_dd_worst, res.max_drawdown_pct)
        equity_curves.append(res.equity_curve)

        calmar = (
            res.total_return_pct / res.max_drawdown_pct
            if res.max_drawdown_pct > 0
            else 0.0
        )

        print(
            f"{symbol:<12} {res.total_return_pct:>+10.2f} {res.sharpe:>8.2f} "
            f"{res.max_drawdown_pct:>8.2f} {calmar:>8.2f} "
            f"{res.win_rate:>6.1f} {res.n_trades:>7d} "
            f"{res.profit_factor:>6.2f} {r['n_long']:>5d} {r['n_short']:>5d}"
        )

    print("-" * 90)

    # Portfolio metrics (equal weight)
    if active_symbols > 0 and equity_curves:
        # Align equity curves to shortest
        min_len = min(len(e) for e in equity_curves)
        aligned = np.array([e[:min_len] for e in equity_curves])
        portfolio_eq = np.mean(aligned, axis=0)

        # Portfolio return
        port_return = (portfolio_eq[-1] / portfolio_eq[0] - 1) * 100

        # Portfolio max drawdown
        peaks = np.maximum.accumulate(portfolio_eq)
        dd = (peaks - portfolio_eq) / peaks
        port_max_dd = float(np.max(dd)) * 100

        # Portfolio Sharpe
        port_rets = np.diff(portfolio_eq) / portfolio_eq[:-1]
        if len(port_rets) > 1 and np.std(port_rets) > 0:
            # Annualize: 4h bars → 6 bars/day → 2190 bars/year
            bars_per_year = 365 * 6
            port_sharpe = (
                np.mean(port_rets) / np.std(port_rets)
                * np.sqrt(bars_per_year)
            )
        else:
            port_sharpe = 0.0

        port_calmar = port_return / port_max_dd if port_max_dd > 0 else 0.0
        avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0.0

        print(f"\n{'PORTFOLIO':.<20}")
        print(f"  Active symbols:   {active_symbols}")
        print(f"  Total trades:     {total_trades}")
        print(f"  Avg return/sym:   {total_return / active_symbols:+.2f}%")
        print(f"  Portfolio return: {port_return:+.2f}%")
        print(f"  Portfolio MaxDD:  {port_max_dd:.2f}%")
        print(f"  Portfolio Sharpe: {port_sharpe:.2f}")
        print(f"  Portfolio Calmar: {port_calmar:.2f}")
        print(f"  Portfolio WR:     {avg_wr:.1f}%")
        print(f"  Worst single DD:  {max_dd_worst:.2f}%")
    else:
        print("\n  No signals generated across any symbol.")
        print("  This can happen if the divergence filter is too strict.")
        print("  Try: python scripts/elliott_backtest.py --no-divergence")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Elliott Wave 1-Year Backtest")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Specific symbols (default: top 20)",
    )
    parser.add_argument(
        "--no-divergence", action="store_true",
        help="Relax: use MACD/Stoch confluence instead of mandatory divergence",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=10000,
        help="Initial capital in USD",
    )
    args = parser.parse_args()

    # ── API connection ──
    from quantbox.core.settings import get_settings

    settings = get_settings()
    client = Client(settings.binance_api_key, settings.binance_api_secret)
    print("🔌 Connected to Binance")

    symbols = args.symbols or TOP_SYMBOLS

    # ── Strategy config ──
    config = ElliottConfig()

    # ── Backtest config (adjusted for 4h bars) ──
    bt_config = BacktestConfig(
        initial_capital=args.initial_capital,
        cost_per_side=0.0005,  # 0.05% taker (conservative)
        # Funding: 0.01%/8h → per 4h bar = 0.01% / 2
        funding_per_bar=0.01 / 100 / 2,
        bar_interval_minutes=240,  # 4h
        risk_per_trade=0.02,  # 2% risk for Elliott (exploratory)
        max_leverage=20.0,
        max_concurrent=3,
    )

    # ── Download data ──
    print(f"\n📥 Downloading {args.days}-day 4h data for {len(symbols)} symbols + BTC...")

    # Always need BTC for macro bias
    btc_data = download_4h_klines(client, "BTCUSDT", args.days)
    if btc_data is None:
        print("❌ Failed to download BTC data. Aborting.")
        return
    print(f"  BTC: {len(btc_data['close'])} bars")

    symbol_data = {}
    for sym in symbols:
        if sym == "BTCUSDT":
            symbol_data[sym] = btc_data
            continue
        data = download_4h_klines(client, sym, args.days)
        if data is not None:
            symbol_data[sym] = data
            print(f"  {sym}: {len(data['close'])} bars")
        else:
            print(f"  {sym}: ❌ failed")
        time.sleep(0.05)

    print(f"\n✅ Downloaded {len(symbol_data)} symbols")

    # ── Numba warmup ──
    print("\n⚙ Warming up Numba JIT...")
    dummy = np.random.randn(100).cumsum() + 100
    dummy_h = dummy + 1
    dummy_l = dummy - 1
    compute_ema(dummy, 10)
    compute_rsi(dummy, 14)
    compute_macd(dummy, 12, 26, 9)
    compute_stochastic(dummy_h, dummy_l, dummy, 14, 3)
    compute_atr(dummy_h, dummy_l, dummy, 14)
    zigzag_pivots(dummy_h, dummy_l, 0.03)
    print("  JIT compiled ✓")

    # ── Run backtest ──
    mode = "relaxed (MACD/Stoch confluence)" if args.no_divergence else "strict (divergence required)"
    print(f"\n🔄 Running backtest... Mode: {mode}")

    results = run_backtest(
        symbol_data, btc_data, config, bt_config,
        use_relaxed=args.no_divergence,
    )

    # ── Report ──
    print_report(results, args.days, mode)

    # ── Save equity curves ──
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym, r in results.items():
        if r["result"] is not None:
            np.save(out_dir / f"elliott_equity_{sym}.npy", r["result"].equity_curve)

    print(f"\n💾 Equity curves saved to {out_dir}/")


if __name__ == "__main__":
    main()
