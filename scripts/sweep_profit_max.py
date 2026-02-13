"""수익률 극대화를 위한 멀티 파라미터 그리드 스윕.

세션 2026-02-11 로그 분석 결과를 기반으로 핵심 파라미터를 탐색한다.
실제 .env 값을 기준(baseline)으로 하되, 각 파라미터를 변동시켜 최적 조합을 찾는다.

관찰된 패턴:
- mean_reversion 6/6 wins (+$131.51) → 핵심 수익원
- stop_loss 0/4 wins (-$37.36) → SL이 너무 타이트할 가능성
- max_holding 3/6 wins (+$17.08) → 보유 시간 늘리면 개선 가능
- |Z| ≥ 2.5 진입이 더 높은 승률
"""
import sys
import json
import time
import logging
import itertools
from dataclasses import asdict
from quantbox.data.real_data_loader import RealDataLoader
from quantbox.backtest.continuous_backtest import (
    ContinuousBacktestConfig, ContinuousPairBacktest,
)

logging.basicConfig(level=logging.WARNING)

# ── 유니버스: 실제 거래에 사용되는 상위 코인 ──
TOP_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'SUIUSDT', 'LINKUSDT',
    'LTCUSDT', 'DOTUSDT', 'AAVEUSDT', 'APTUSDT', 'NEARUSDT',
    'UNIUSDT', 'ZECUSDT', 'PEPEUSDT', 'JUPUSDT', 'WIFUSDT',
]

# ── 스윕할 파라미터 그리드 ──
# 현재 .env baseline: SL=1.5, MH=64(16h), LEV=2, ENTRY=2.0
# dynamic_exit: base=0.1, range=0.9
GRID = {
    'stop_loss_zscore': [1.5, 2.0, 2.5, 3.0],
    'max_holding_bars': [64, 96, 128],       # 16h, 24h, 32h
    'entry_zscore': [2.0, 2.3, 2.5],
    'leverage': [2, 3],
}

# ── 고정 파라미터 (baseline from .env) ──
FIXED = dict(
    initial_capital=2326.76,  # 실제 세션 시작 잔고
    exit_zscore=0.5,
    use_dynamic_exit=True,
    dynamic_exit_base=0.1,
    dynamic_exit_range=0.9,
    max_traders=5,
    trading_fee=0.0005,
    slippage=0.001,
    funding_rate_8h=0.0001,
    min_correlation=0.6,
    max_cointegration_pvalue=0.05,
    min_half_life=2.0,
    max_half_life=80.0,
    max_hurst=0.5,
)


def run_sweep():
    loader = RealDataLoader()
    data = loader.load_data(symbols=TOP_SYMBOLS, min_bars=300, align_data=True)
    print(f"Loaded {len(data)} symbols, {len(next(iter(data.values())))} bars each")
    if len(data) < 4:
        print("ERROR: Not enough data!")
        sys.exit(1)

    # 파라미터 조합 생성
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"\nTotal combinations: {total}")

    sep = "=" * 140
    hdr = (
        f"{'#':>3} {'SL':>4} {'MH_h':>5} {'ENTRY':>5} {'LEV':>3} "
        f"{'trades':>6} {'win%':>6} {'ret%':>8} {'avgRet':>8} {'PF':>6} "
        f"{'maxDD%':>7} {'sharpe':>7}  "
        f"{'mr':>3} {'mh':>3} {'sl':>3} {'elapsed':>4}"
    )
    print(f"\n{sep}\n{hdr}\n{sep}", flush=True)

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        cfg = ContinuousBacktestConfig(
            **FIXED,
            stop_loss_zscore=params['stop_loss_zscore'],
            max_holding_bars=params['max_holding_bars'],
            entry_zscore=params['entry_zscore'],
            leverage=params['leverage'],
        )

        bt = ContinuousPairBacktest(cfg)
        t0 = time.time()
        result = bt.run(data, progress=False)
        elapsed = time.time() - t0

        state = result['state']
        trades = state.trades
        metrics = result['metrics']
        n_trades = len(trades)

        if n_trades == 0:
            print(f"{i:>3} {params['stop_loss_zscore']:>4.1f} "
                  f"{params['max_holding_bars']*0.25:>5.0f} "
                  f"{params['entry_zscore']:>5.1f} {params['leverage']:>3} "
                  f"{'N/A':>6} ({elapsed:.0f}s)", flush=True)
            continue

        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(winners) / n_trades * 100
        total_ret = metrics.get('total_return_pct', 0)
        avg_ret = sum(t.net_return_pct for t in trades) / n_trades
        max_dd = metrics.get('max_drawdown', 0)
        sharpe = metrics.get('sharpe_ratio', 0)

        # Profit Factor
        gross_profit = sum(t.net_pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.net_pnl for t in losers)) if losers else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Exit reason breakdown
        exit_counts = {}
        for t in trades:
            r = t.exit_reason.value
            exit_counts[r] = exit_counts.get(r, 0) + 1

        mr = exit_counts.get('mean_reversion', 0)
        mh = exit_counts.get('max_holding', 0)
        sl = exit_counts.get('stop_loss', 0)

        row = {
            **params,
            'max_holding_hours': params['max_holding_bars'] * 0.25,
            'n_trades': n_trades,
            'win_rate': round(win_rate, 2),
            'total_return_pct': round(total_ret, 2),
            'avg_return_pct': round(avg_ret, 4),
            'profit_factor': round(pf, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 3),
            'mean_reversion_exits': mr,
            'max_holding_exits': mh,
            'stop_loss_exits': sl,
            'elapsed_sec': round(elapsed, 1),
        }
        results.append(row)

        print(
            f"{i:>3} {params['stop_loss_zscore']:>4.1f} "
            f"{params['max_holding_bars']*0.25:>5.0f} "
            f"{params['entry_zscore']:>5.1f} {params['leverage']:>3} "
            f"{n_trades:>6} {win_rate:>5.1f}% {total_ret:>7.2f}% "
            f"{avg_ret:>7.3f}% {pf:>5.2f} {max_dd:>6.1f}% {sharpe:>6.3f}  "
            f"{mr:>3} {mh:>3} {sl:>3} {elapsed:>3.0f}s",
            flush=True,
        )

    print(f"{sep}", flush=True)

    # ── 결과 정렬: Sharpe → Total Return ──
    results.sort(key=lambda x: (-x['sharpe_ratio'], -x['total_return_pct']))

    print("\n🏆 Top 5 by Sharpe Ratio:")
    for i, r in enumerate(results[:5], 1):
        print(
            f"  #{i}: SL={r['stop_loss_zscore']:.1f} MH={r['max_holding_hours']:.0f}h "
            f"ENTRY={r['entry_zscore']:.1f} LEV={r['leverage']}x "
            f"→ ret={r['total_return_pct']:.2f}% sharpe={r['sharpe_ratio']:.3f} "
            f"PF={r['profit_factor']:.2f} DD={r['max_drawdown_pct']:.1f}%"
        )

    # ── PF 기준 정렬 ──
    by_pf = sorted(results, key=lambda x: -x['profit_factor'])
    print("\n🏆 Top 5 by Profit Factor:")
    for i, r in enumerate(by_pf[:5], 1):
        print(
            f"  #{i}: SL={r['stop_loss_zscore']:.1f} MH={r['max_holding_hours']:.0f}h "
            f"ENTRY={r['entry_zscore']:.1f} LEV={r['leverage']}x "
            f"→ ret={r['total_return_pct']:.2f}% PF={r['profit_factor']:.2f} "
            f"sharpe={r['sharpe_ratio']:.3f} DD={r['max_drawdown_pct']:.1f}%"
        )

    # ── 현재 baseline과 비교 ──
    baseline = next(
        (r for r in results
         if r['stop_loss_zscore'] == 1.5
         and r['max_holding_bars'] == 64
         and r['entry_zscore'] == 2.0
         and r['leverage'] == 2),
        None,
    )
    if baseline:
        print(f"\n📊 Current .env baseline:")
        print(
            f"  SL=1.5 MH=16h ENTRY=2.0 LEV=2x "
            f"→ ret={baseline['total_return_pct']:.2f}% "
            f"sharpe={baseline['sharpe_ratio']:.3f} "
            f"PF={baseline['profit_factor']:.2f} "
            f"DD={baseline['max_drawdown_pct']:.1f}%"
        )

    # ── 결과 저장 ──
    out_path = "scripts/sweep_profit_max_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_sweep()
