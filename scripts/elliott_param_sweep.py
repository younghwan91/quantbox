"""
Elliott Wave — Fast Global Parameter Sweep + AB Test vs Rubber Band

Two-stage efficient search:
  Stage 1: Pattern detection (zigzag_threshold × min_pattern_quality × fib_c_tolerance)
  Stage 2: Risk/exit management (trail_atr_mult × effective_leverage) using best Stage 1
  Stage 3: AB test — best Elliott config vs Rubber Band on same 4h data window

Downloads 365-day 4h data ONCE, then iterates configs in-memory.

Usage:
    python scripts/elliott_param_sweep.py
    python scripts/elliott_param_sweep.py --days 180 --symbols ETHUSDT SOLUSDT
"""

import argparse
import itertools
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

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
)
from quantbox.backtest.vectorized_backtest import (
    VectorizedBacktest,
    BacktestConfig,
)

# Import compute_signals_relaxed from sibling script
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "elliott_bt", str(Path(__file__).parent / "elliott_backtest.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_signals_relaxed = _mod.compute_signals_relaxed


# ── Symbols: liquid large-caps for statistical significance ──
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT",
]

# ═══════════════════════════════════════════════════════════════════════
#  Data download (identical to elliott_backtest.py)
# ═══════════════════════════════════════════════════════════════════════


def download_4h_klines(client: Client, symbol: str, days: int = 365) -> dict | None:
    """Download 4h OHLCV from Binance Futures."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_klines = []
    current_ms = start_ms

    while current_ms < end_ms:
        try:
            klines = client.futures_klines(
                symbol=symbol, interval="4h",
                startTime=current_ms, endTime=end_ms, limit=1000,
            )
            if not klines:
                break
            all_klines.extend(klines)
            current_ms = int(klines[-1][6]) + 1
            if len(klines) < 1000:
                break
            time.sleep(0.1)
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


# ═══════════════════════════════════════════════════════════════════════
#  Backtest runner with config override
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SweepResult:
    """Compact result for one config across all symbols."""
    label: str
    total_return: float   # portfolio avg return %
    sharpe: float         # portfolio Sharpe
    max_dd: float         # worst single-symbol drawdown %
    win_rate: float       # overall win rate %
    n_trades: int         # total trades
    profit_factor: float  # aggregate PF
    calmar: float         # return / max_dd

    def score(self) -> float:
        """Composite score: Calmar-weighted, penalize low trade count."""
        if self.n_trades < 3:
            return -999.0
        # Primary: Calmar (risk-adj return).  Secondary: Sharpe.
        # Penalty: drawdown > 15% is unacceptable
        dd_penalty = max(0, self.max_dd - 15) * 0.5
        trade_bonus = min(self.n_trades / 20.0, 1.0)  # reward more trades up to 20
        return self.calmar * 0.5 + self.sharpe * 0.3 + trade_bonus * 0.2 - dd_penalty


def run_sweep_config(
    symbol_data: dict[str, dict],
    btc_close: np.ndarray,
    config: ElliottConfig,
    bt_config: BacktestConfig,
    use_relaxed: bool = True,
    label: str = "",
) -> SweepResult:
    """Run one configuration across all symbols, return aggregate metrics."""

    bt = VectorizedBacktest(bt_config)
    total_ret = 0.0
    total_trades = 0
    total_wins = 0
    worst_dd = 0.0
    active = 0
    equity_curves = []
    gross_profit = 0.0
    gross_loss = 0.0

    for sym, data in symbol_data.items():
        close = data["close"]
        high = data["high"]
        low = data["low"]
        min_len = min(len(close), len(btc_close))
        c, h, lo, bc = close[-min_len:], high[-min_len:], low[-min_len:], btc_close[-min_len:]

        if use_relaxed:
            signals, adj_close = compute_signals_relaxed(c, h, lo, bc, config)
        else:
            signals, adj_close = compute_elliott_signals(c, h, lo, bc, config)

        n_sig = int(np.sum(np.abs(signals) > 0))
        if n_sig == 0:
            continue

        result = bt.run_single_partial(adj_close, signals, symbol=sym)
        active += 1
        total_ret += result.total_return_pct
        total_trades += result.n_trades
        total_wins += int(result.win_rate / 100 * result.n_trades)
        worst_dd = max(worst_dd, result.max_drawdown_pct)
        equity_curves.append(result.equity_curve)

        # Profit factor aggregation
        for tr in (result.trades if hasattr(result, 'trades') and result.trades else []):
            r = tr.get('return_pct', 0)
            if r > 0:
                gross_profit += r
            else:
                gross_loss += abs(r)

    if active == 0:
        return SweepResult(label, 0, 0, 0, 0, 0, 0, 0)

    avg_ret = total_ret / active
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    # Portfolio Sharpe from aligned equity curves
    min_len = min(len(e) for e in equity_curves)
    aligned = np.array([e[:min_len] for e in equity_curves])
    port_eq = np.mean(aligned, axis=0)
    port_rets = np.diff(port_eq) / port_eq[:-1]
    if len(port_rets) > 1 and np.std(port_rets) > 0:
        sharpe = np.mean(port_rets) / np.std(port_rets) * np.sqrt(365 * 6)
    else:
        sharpe = 0.0

    # Portfolio drawdown
    peaks = np.maximum.accumulate(port_eq)
    dd = (peaks - port_eq) / peaks
    port_dd = float(np.max(dd)) * 100

    calmar = avg_ret / port_dd if port_dd > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)

    return SweepResult(label, avg_ret, sharpe, port_dd, wr, total_trades, pf, calmar)


# ═══════════════════════════════════════════════════════════════════════
#  Main sweep
# ═══════════════════════════════════════════════════════════════════════


def print_sweep_table(results: list[SweepResult], title: str):
    """Print compact results table sorted by score."""
    results.sort(key=lambda r: r.score(), reverse=True)
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")
    hdr = (
        f"{'#':>3} {'Label':<45} {'Ret%':>8} {'Sharpe':>8} {'MaxDD%':>8} "
        f"{'WR%':>6} {'Trades':>7} {'Calmar':>8} {'Score':>8}"
    )
    print(hdr)
    print("-" * 110)
    for i, r in enumerate(results[:20], 1):
        print(
            f"{i:>3} {r.label:<45} {r.total_return:>+8.2f} {r.sharpe:>8.2f} "
            f"{r.max_dd:>8.2f} {r.win_rate:>6.1f} {r.n_trades:>7d} "
            f"{r.calmar:>8.2f} {r.score():>8.3f}"
        )
    print(f"{'=' * 110}")


def make_config(**overrides) -> ElliottConfig:
    """Create ElliottConfig with selective overrides."""
    cfg = ElliottConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Elliott Wave Parameter Sweep")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--initial-capital", type=float, default=10000)
    args = parser.parse_args()

    from quantbox.core.settings import get_settings
    settings = get_settings()
    client = Client(settings.binance_api_key, settings.binance_api_secret)
    print("🔌 Connected to Binance")

    symbols = args.symbols or DEFAULT_SYMBOLS

    # ── Backtest engine config (4h bars) ──
    bt_config = BacktestConfig(
        initial_capital=args.initial_capital,
        cost_per_side=0.0005,
        funding_per_bar=0.01 / 100 / 2,  # per 4h bar
        bar_interval_minutes=240,
        risk_per_trade=0.02,
        max_leverage=20.0,
        max_concurrent=3,
    )

    # ── Download data ONCE ──
    print(f"\n📥 Downloading {args.days}-day 4h data for {len(symbols)} symbols + BTC...")
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
        if data:
            symbol_data[sym] = data
            print(f"  {sym}: {len(data['close'])} bars")
        else:
            print(f"  {sym}: ❌ failed")
        time.sleep(0.05)

    btc_close = btc_data["close"]
    print(f"\n✅ {len(symbol_data)} symbols ready")

    # ── Numba warmup ──
    print("\n⚙ Warming up Numba JIT...")
    d = np.random.randn(100).cumsum() + 100
    dh, dl = d + 1, d - 1
    compute_ema(d, 10); compute_rsi(d, 14)
    compute_macd(d, 12, 26, 9); compute_stochastic(dh, dl, d, 14, 3)
    compute_atr(dh, dl, d, 14); zigzag_pivots(dh, dl, 0.03)
    print("  JIT compiled ✓")

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1: Pattern detection sweep
    #  zigzag_threshold × min_pattern_quality × fib_c_impulse_tolerance
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: PATTERN DETECTION SWEEP")
    print("=" * 60)

    zz_vals = [0.02, 0.03, 0.04, 0.05]
    qual_vals = [20.0, 30.0, 40.0, 50.0]
    tol_vals = [0.08, 0.12, 0.18]

    stage1_results = []
    total_combos = len(zz_vals) * len(qual_vals) * len(tol_vals)
    combo_i = 0

    for zz, qual, tol in itertools.product(zz_vals, qual_vals, tol_vals):
        combo_i += 1
        label = f"zz={zz:.2f} qual={qual:.0f} tol={tol:.2f}"
        print(f"  [{combo_i}/{total_combos}] {label}", end="", flush=True)

        # Multi-scale thresholds scale with base
        thresholds = (zz, zz + 0.02, zz + 0.04)
        cfg = make_config(
            zigzag_threshold=zz,
            zigzag_thresholds=thresholds,
            min_pattern_quality=qual,
            fib_c_impulse_tolerance=tol,
        )

        result = run_sweep_config(
            symbol_data, btc_close, cfg, bt_config,
            use_relaxed=True, label=label,
        )
        stage1_results.append(result)
        print(f"  → ret={result.total_return:+.1f}% trades={result.n_trades} dd={result.max_dd:.1f}%")

    print_sweep_table(stage1_results, "STAGE 1: PATTERN DETECTION — TOP CONFIGS")

    # Best Stage 1
    best1 = max(stage1_results, key=lambda r: r.score())
    # Parse best params from label
    parts = best1.label.split()
    best_zz = float(parts[0].split("=")[1])
    best_qual = float(parts[1].split("=")[1])
    best_tol = float(parts[2].split("=")[1])
    print(f"\n  ★ Best Stage 1: {best1.label} (score={best1.score():.3f})")

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2: Risk/Exit management sweep
    #  trail_atr_mult × effective_leverage × divergence_lookback
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: RISK/EXIT SWEEP (using best Stage 1 params)")
    print("=" * 60)

    trail_vals = [1.5, 2.0, 2.5, 3.0]
    lev_vals = [1.5, 2.0, 3.0]
    div_vals = [30, 45, 60]

    stage2_results = []
    total_combos = len(trail_vals) * len(lev_vals) * len(div_vals)
    combo_i = 0

    best_thresholds = (best_zz, best_zz + 0.02, best_zz + 0.04)

    for trail, lev, div_lb in itertools.product(trail_vals, lev_vals, div_vals):
        combo_i += 1
        label = f"zz={best_zz:.2f} q={best_qual:.0f} t={best_tol:.2f} | trail={trail:.1f} lev={lev:.1f} div={div_lb}"
        print(f"  [{combo_i}/{total_combos}] trail={trail:.1f} lev={lev:.1f} div={div_lb}", end="", flush=True)

        cfg = make_config(
            zigzag_threshold=best_zz,
            zigzag_thresholds=best_thresholds,
            min_pattern_quality=best_qual,
            fib_c_impulse_tolerance=best_tol,
            trail_atr_mult=trail,
            effective_leverage=lev,
            divergence_lookback=div_lb,
        )

        result = run_sweep_config(
            symbol_data, btc_close, cfg, bt_config,
            use_relaxed=True, label=label,
        )
        stage2_results.append(result)
        print(f"  → ret={result.total_return:+.1f}% trades={result.n_trades} dd={result.max_dd:.1f}%")

    print_sweep_table(stage2_results, "STAGE 2: RISK/EXIT MANAGEMENT — TOP CONFIGS")

    best2 = max(stage2_results, key=lambda r: r.score())
    # Parse trail/lev/div from label
    exit_parts = best2.label.split("|")[1].strip().split()
    best_trail = float(exit_parts[0].split("=")[1])
    best_lev = float(exit_parts[1].split("=")[1])
    best_div = int(exit_parts[2].split("=")[1])
    print(f"\n  ★ Best Stage 2: trail={best_trail} lev={best_lev} div={best_div} (score={best2.score():.3f})")

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 3: Strict vs Relaxed Divergence
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 3: STRICT vs RELAXED DIVERGENCE")
    print("=" * 60)

    best_cfg = make_config(
        zigzag_threshold=best_zz,
        zigzag_thresholds=best_thresholds,
        min_pattern_quality=best_qual,
        fib_c_impulse_tolerance=best_tol,
        trail_atr_mult=best_trail,
        effective_leverage=best_lev,
        divergence_lookback=best_div,
    )

    res_relaxed = run_sweep_config(
        symbol_data, btc_close, best_cfg, bt_config,
        use_relaxed=True, label="BEST — Relaxed divergence",
    )
    res_strict = run_sweep_config(
        symbol_data, btc_close, best_cfg, bt_config,
        use_relaxed=False, label="BEST — Strict divergence (required)",
    )

    stage3 = [res_relaxed, res_strict]
    print_sweep_table(stage3, "STAGE 3: DIVERGENCE MODE COMPARISON")

    best_mode = "relaxed" if res_relaxed.score() >= res_strict.score() else "strict"
    best_overall = res_relaxed if best_mode == "relaxed" else res_strict

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 4: AB Test — Best Elliott vs Rubber Band baseline
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 4: AB TEST — ELLIOTT vs RUBBER BAND")
    print("=" * 60)

    # Run Rubber Band on same 4h data for fair comparison
    # RB uses compute_signals_with_regime on 4h bars
    from quantbox.strategy.rubberband import compute_signals_with_regime, RubberBandConfig
    rb_config = RubberBandConfig()
    rb_bt_config = BacktestConfig(
        initial_capital=args.initial_capital,
        cost_per_side=0.0005,
        funding_per_bar=0.01 / 100 / 2,
        bar_interval_minutes=240,  # 4h
        risk_per_trade=0.01,
        max_leverage=20.0,
        max_concurrent=5,
    )

    rb_bt = VectorizedBacktest(rb_bt_config)
    rb_total_ret = 0.0
    rb_trades = 0
    rb_wins = 0
    rb_active = 0
    rb_worst_dd = 0.0
    rb_equities = []

    for sym, data in symbol_data.items():
        close = data["close"]
        min_len = min(len(close), len(btc_close))
        c = close[-min_len:]
        bc = btc_close[-min_len:]

        # Use regime-filtered signals (same as live scalper core)
        sigs = compute_signals_with_regime(
            c, bc,
            lookback=rb_config.lookback,
            threshold=rb_config.threshold,
            target_retrace=rb_config.target_retrace,
            stop_mult=rb_config.stop_mult,
            max_hold=rb_config.max_hold,
            regime_lookback=rb_config.btc_regime_period,
            regime_threshold=rb_config.threshold,  # same threshold for regime
            stop_decay=rb_config.stop_decay,
        )
        n_sig = int(np.sum(np.abs(sigs) > 0))
        if n_sig == 0:
            continue

        result = rb_bt.run_single(c, sigs, symbol=sym)
        rb_active += 1
        rb_total_ret += result.total_return_pct
        rb_trades += result.n_trades
        rb_wins += int(result.win_rate / 100 * result.n_trades)
        rb_worst_dd = max(rb_worst_dd, result.max_drawdown_pct)
        rb_equities.append(result.equity_curve)

    if rb_active > 0 and rb_equities:
        rb_avg_ret = rb_total_ret / rb_active
        rb_wr = rb_wins / rb_trades * 100 if rb_trades > 0 else 0
        min_l = min(len(e) for e in rb_equities)
        aligned_rb = np.array([e[:min_l] for e in rb_equities])
        port_rb = np.mean(aligned_rb, axis=0)
        rb_rets = np.diff(port_rb) / port_rb[:-1]
        rb_sharpe = np.mean(rb_rets) / np.std(rb_rets) * np.sqrt(365 * 6) if np.std(rb_rets) > 0 else 0
        peaks = np.maximum.accumulate(port_rb)
        dd = (peaks - port_rb) / peaks
        rb_port_dd = float(np.max(dd)) * 100
        rb_calmar = rb_avg_ret / rb_port_dd if rb_port_dd > 0 else 0
    else:
        rb_avg_ret = rb_sharpe = rb_wr = rb_port_dd = rb_calmar = 0
        rb_trades = 0

    rb_result = SweepResult(
        "Rubber Band (4h baseline)", rb_avg_ret, rb_sharpe,
        rb_port_dd, rb_wr, rb_trades, 0, rb_calmar,
    )

    ab_results = [best_overall, rb_result]
    print_sweep_table(ab_results, "AB TEST: ELLIOTT (BEST) vs RUBBER BAND")

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════
    #  Final Summary
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  FINAL OPTIMAL CONFIGURATION")
    print("=" * 60)
    print(f"  zigzag_threshold:       {best_zz}")
    print(f"  zigzag_thresholds:      {best_thresholds}")
    print(f"  min_pattern_quality:    {best_qual}")
    print(f"  fib_c_impulse_tolerance: {best_tol}")
    print(f"  trail_atr_mult:         {best_trail}")
    print(f"  effective_leverage:     {best_lev}")
    print(f"  divergence_lookback:    {best_div}")
    print(f"  divergence_mode:        {best_mode}")
    print(f"  ────────────────────────────────")
    print(f"  Portfolio return:       {best_overall.total_return:+.2f}%")
    print(f"  Sharpe ratio:           {best_overall.sharpe:.2f}")
    print(f"  Max drawdown:           {best_overall.max_dd:.2f}%")
    print(f"  Win rate:               {best_overall.win_rate:.1f}%")
    print(f"  Total trades:           {best_overall.n_trades}")
    print(f"  Calmar ratio:           {best_overall.calmar:.2f}")
    print(f"  Composite score:        {best_overall.score():.3f}")
    print(f"  ────────────────────────────────")
    print(f"  Elapsed:                {elapsed:.0f}s")
    print("=" * 60)

    # Save results
    out_file = Path("logs/elliott_sweep_results.txt")
    with open(out_file, "w") as f:
        f.write(f"Elliott Wave Parameter Sweep — {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Days: {args.days}, Symbols: {len(symbol_data)}\n\n")
        f.write("OPTIMAL CONFIG:\n")
        f.write(f"  zigzag_threshold={best_zz}\n")
        f.write(f"  zigzag_thresholds={best_thresholds}\n")
        f.write(f"  min_pattern_quality={best_qual}\n")
        f.write(f"  fib_c_impulse_tolerance={best_tol}\n")
        f.write(f"  trail_atr_mult={best_trail}\n")
        f.write(f"  effective_leverage={best_lev}\n")
        f.write(f"  divergence_lookback={best_div}\n")
        f.write(f"  mode={best_mode}\n\n")
        f.write(f"RESULTS:\n")
        f.write(f"  Return: {best_overall.total_return:+.2f}%\n")
        f.write(f"  Sharpe: {best_overall.sharpe:.2f}\n")
        f.write(f"  MaxDD:  {best_overall.max_dd:.2f}%\n")
        f.write(f"  WR:     {best_overall.win_rate:.1f}%\n")
        f.write(f"  Trades: {best_overall.n_trades}\n")
        f.write(f"  Calmar: {best_overall.calmar:.2f}\n\n")
        f.write("AB TEST vs RUBBER BAND:\n")
        f.write(f"  RB Return: {rb_result.total_return:+.2f}%\n")
        f.write(f"  RB Sharpe: {rb_result.sharpe:.2f}\n")
        f.write(f"  RB Trades: {rb_result.n_trades}\n")

    print(f"\n💾 Results saved to {out_file}")


if __name__ == "__main__":
    main()
