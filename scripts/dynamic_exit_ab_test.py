#!/usr/bin/env python3
"""
Dynamic Exit A/B Test: 고정 exit_zscore vs half-life 기반 동적 exit 비교.

실제 바이낸스 캐시 데이터를 사용하여 두 전략의 성능을 비교한다.
결과를 JSON으로 저장하고 콘솔에 비교표를 출력한다.

사용법:
    uv run python scripts/dynamic_exit_ab_test.py
"""

import json
import sys
import time
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

from quantbox.data.real_data_loader import RealDataLoader
from quantbox.backtest.continuous_backtest import (
    ContinuousBacktestConfig,
    ContinuousPairBacktest,
)

# ─── Constants ───────────────────────────────────────────────────
INITIAL_CAPITAL = 1000.0
LEVERAGE = 2
MAX_TRADERS = 5
FORMATION_BARS = 240

# Base config shared by both strategies
BASE_PARAMS = dict(
    initial_capital=INITIAL_CAPITAL,
    leverage=LEVERAGE,
    max_traders=MAX_TRADERS,
    formation_bars=FORMATION_BARS,
    trading_fee=0.0005,
    slippage=0.001,
    funding_rate_8h=0.0001,
    bar_interval_minutes=15,
    entry_zscore=2.0,
    stop_loss_zscore=3.5,
    max_holding_bars=100,
    min_correlation=0.5,
    max_cointegration_pvalue=0.05,
    min_half_life=2.0,
    max_half_life=50.0,
    max_hurst=0.5,
    min_volume_rank=50,
)


@dataclass
class ABResult:
    """A/B 테스트 결과."""
    name: str
    total_return_pct: float
    n_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_bars: float
    avg_trade_pnl: float
    exit_reasons: dict
    in_market_pct: float
    avg_slot_utilization: float
    elapsed_sec: float


def load_data() -> dict[str, pd.DataFrame]:
    """캐시에서 실제 데이터 로드."""
    loader = RealDataLoader()
    available = loader.get_available_symbols()
    print(f"📊 Available symbols: {len(available)}")

    data = loader.load_all(min_bars=FORMATION_BARS + 200, align_data=True)
    if not data:
        raise ValueError("No data available in cache")

    # Use first symbol to get bar count
    first_sym = list(data.keys())[0]
    n_bars = len(data[first_sym])
    print(f"📊 Loaded {len(data)} symbols, {n_bars} bars each")
    return data


def run_backtest(name: str, config: ContinuousBacktestConfig, data: dict) -> ABResult:
    """단일 백테스트 실행 및 결과 요약."""
    print(f"\n{'='*60}")
    print(f"▶ Running: {name}")
    print(f"  exit_zscore={config.exit_zscore:.2f}, "
          f"use_dynamic_exit={config.use_dynamic_exit}")
    if config.use_dynamic_exit:
        print(f"  dynamic_exit_base={config.dynamic_exit_base:.2f}, "
              f"dynamic_exit_range={config.dynamic_exit_range:.2f}")
    print(f"{'='*60}")

    engine = ContinuousPairBacktest(config)
    t0 = time.time()
    results = engine.run(data, progress=True)
    elapsed = time.time() - t0

    m = results['metrics']
    trades_df = results['trades']

    return ABResult(
        name=name,
        total_return_pct=m['total_return_pct'],
        n_trades=m['n_trades'],
        win_rate=m['win_rate'],
        profit_factor=m['profit_factor'],
        sharpe_ratio=m['sharpe_ratio'],
        max_drawdown=m['max_drawdown'],
        avg_holding_bars=m['avg_holding_bars'],
        avg_trade_pnl=m['avg_trade'],
        exit_reasons=m['exit_reasons'],
        in_market_pct=m['in_market_pct'],
        avg_slot_utilization=m['avg_slot_utilization'],
        elapsed_sec=elapsed,
    )


def print_comparison(a: ABResult, b: ABResult):
    """두 결과를 나란히 비교 출력."""
    print(f"\n{'='*72}")
    print(f"{'A/B COMPARISON':^72}")
    print(f"{'='*72}")

    def row(label, va, vb, fmt=".2f", higher_is_better=True):
        if isinstance(va, dict) or isinstance(vb, dict):
            return
        diff = vb - va
        arrow = ""
        if abs(diff) > 1e-6:
            if (diff > 0) == higher_is_better:
                arrow = " ✅"
            else:
                arrow = " ❌"
        print(f"  {label:<28} {va:>12{fmt}}  {vb:>12{fmt}}  {diff:>+10{fmt}}{arrow}")

    print(f"  {'Metric':<28} {'[A] Static':>12}  {'[B] Dynamic':>12}  {'Diff':>10}")
    print(f"  {'-'*28} {'-'*12}  {'-'*12}  {'-'*10}")
    row("Total Return %", a.total_return_pct, b.total_return_pct)
    row("# Trades", a.n_trades, b.n_trades, ".0f")
    row("Win Rate %", a.win_rate, b.win_rate)
    row("Profit Factor", a.profit_factor, b.profit_factor)
    row("Sharpe Ratio", a.sharpe_ratio, b.sharpe_ratio)
    row("Max Drawdown %", a.max_drawdown, b.max_drawdown, ".2f", higher_is_better=False)
    row("Avg Holding Bars", a.avg_holding_bars, b.avg_holding_bars, ".1f", higher_is_better=False)
    row("Avg Trade PnL $", a.avg_trade_pnl, b.avg_trade_pnl)
    row("In Market %", a.in_market_pct, b.in_market_pct)
    row("Slot Utilization %", a.avg_slot_utilization, b.avg_slot_utilization)

    # Exit reason breakdown
    print(f"\n  {'Exit Reasons':^72}")
    print(f"  {'-'*28} {'-'*12}  {'-'*12}")
    all_reasons = set(list(a.exit_reasons.keys()) + list(b.exit_reasons.keys()))
    for reason in sorted(all_reasons):
        va = a.exit_reasons.get(reason, 0)
        vb = b.exit_reasons.get(reason, 0)
        diff = vb - va
        print(f"  {reason:<28} {va:>12}  {vb:>12}  {diff:>+10}")

    # Verdict
    print(f"\n{'─'*72}")
    score_b = 0
    if b.total_return_pct > a.total_return_pct:
        score_b += 3
    if b.sharpe_ratio > a.sharpe_ratio:
        score_b += 2
    if b.max_drawdown < a.max_drawdown:
        score_b += 2
    if b.win_rate > a.win_rate:
        score_b += 1
    if b.profit_factor > a.profit_factor:
        score_b += 1

    total = 9
    if score_b >= 6:
        verdict = f"🟢 Dynamic Exit WINS ({score_b}/{total} criteria)"
    elif score_b >= 4:
        verdict = f"🟡 INCONCLUSIVE ({score_b}/{total} criteria) — needs more data"
    else:
        verdict = f"🔴 Static Exit WINS ({total - score_b}/{total} criteria) — do NOT merge"
    print(f"  VERDICT: {verdict}")
    print(f"{'─'*72}")


def main():
    """A/B 테스트 실행."""
    data = load_data()

    # ── Strategy A: Static exit (baseline, current behavior) ──
    config_a = ContinuousBacktestConfig(
        **BASE_PARAMS,
        exit_zscore=0.5,
        use_dynamic_exit=False,
    )
    result_a = run_backtest("[A] Static exit_zscore=0.5", config_a, data)

    # ── Strategy B: Dynamic exit (half-life based) ──
    config_b = ContinuousBacktestConfig(
        **BASE_PARAMS,
        exit_zscore=0.5,  # fallback (unused when dynamic is on)
        use_dynamic_exit=True,
        dynamic_exit_base=0.3,
        dynamic_exit_range=0.7,
    )
    result_b = run_backtest("[B] Dynamic exit (hl-based)", config_b, data)

    # ── Print comparison ──
    print_comparison(result_a, result_b)

    # ── Save results ──
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_symbols': len(data),
        'data_bars': len(list(data.values())[0]),
        'strategy_a': asdict(result_a),
        'strategy_b': asdict(result_b),
    }
    output_path = Path(__file__).parent / 'dynamic_exit_ab_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == '__main__':
    main()
