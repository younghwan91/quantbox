"""Sweep max_holding_bars to find optimal value.

Uses top-volume symbols to keep runtime reasonable.
C(15,2)=105 pairs vs C(35,2)=595 pairs -> ~5.7x faster.
"""
import sys
import json
import time
import logging
from quantbox.data.real_data_loader import RealDataLoader
from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig, ContinuousPairBacktest

logging.basicConfig(level=logging.WARNING)

TOP_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'SUIUSDT', 'LINKUSDT',
    'LTCUSDT', 'DOTUSDT', 'AAVEUSDT', 'APTUSDT', 'NEARUSDT',
]

loader = RealDataLoader()
data = loader.load_data(symbols=TOP_SYMBOLS, min_bars=300, align_data=True)
print(f"Loaded {len(data)} symbols", flush=True)
if not data:
    print("ERROR: No data loaded!")
    sys.exit(1)

for sym, df in list(data.items())[:3]:
    print(f"  {sym}: {len(df)} bars", flush=True)

holding_values = [24, 32, 48, 64, 80, 100]
results_json = []

sep = "=" * 120
hdr = f"{'bars':>5} {'hours':>5} {'trades':>6} {'win%':>6} {'ret%':>8} {'avg_pnl':>8} {'avgH':>6} {'winH':>6} {'lossH':>6} {'maxDD':>6}  exit_reasons"
print(f"\n{sep}\n{hdr}\n{sep}", flush=True)

for bars in holding_values:
    hours = bars * 0.25
    config = ContinuousBacktestConfig(
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=2.0,
        max_holding_bars=bars,
        leverage=2,
        max_traders=5,
        use_dynamic_exit=True,
        dynamic_exit_base=0.3,
        dynamic_exit_range=0.7,
    )
    bt = ContinuousPairBacktest(config)
    t0 = time.time()
    result = bt.run(data, progress=False)
    elapsed = time.time() - t0

    # result is a dict: {'metrics', 'equity_curve', 'trades' (DataFrame), 'state', 'config'}
    state = result['state']
    trades = state.trades  # list of TradeResult objects
    metrics = result['metrics']

    n_trades = len(trades)
    if n_trades == 0:
        print(f"{bars:>5} {hours:>5.0f} {'N/A':>6} ({elapsed:.0f}s)", flush=True)
        continue

    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    win_rate = len(winners) / n_trades * 100
    total_ret = metrics.get('total_return_pct', 0)
    avg_pnl = sum(t.net_return_pct for t in trades) / n_trades

    exit_reasons = {}
    for t in trades:
        r = t.exit_reason.value
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    avg_bars_held = sum(t.bars_held for t in trades) / n_trades
    win_bars = sum(t.bars_held for t in winners) / len(winners) if winners else 0
    loss_bars = sum(t.bars_held for t in losers) / len(losers) if losers else 0

    max_dd = metrics.get('max_drawdown', 0)

    row = {
        'max_holding_bars': bars,
        'max_holding_hours': hours,
        'n_trades': n_trades,
        'win_rate': round(win_rate, 2),
        'total_return_pct': round(total_ret, 2),
        'avg_pnl_pct': round(avg_pnl, 4),
        'avg_hold_hours': round(avg_bars_held * 0.25, 1),
        'win_hold_hours': round(win_bars * 0.25, 1),
        'loss_hold_hours': round(loss_bars * 0.25, 1),
        'max_drawdown_pct': round(max_dd, 2),
        'exit_reasons': exit_reasons,
        'elapsed_sec': round(elapsed, 1),
    }
    results_json.append(row)

    print(f"{bars:>5} {hours:>5.0f} {n_trades:>6} {win_rate:>5.1f}% {total_ret:>7.2f}% {avg_pnl:>7.3f}% {avg_bars_held*0.25:>5.1f}h {win_bars*0.25:>5.1f}h {loss_bars*0.25:>5.1f}h {max_dd:>5.1f}%  {exit_reasons} ({elapsed:.0f}s)", flush=True)

print(f"{sep}", flush=True)

with open("scripts/sweep_max_holding_results.json", "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to scripts/sweep_max_holding_results.json")
