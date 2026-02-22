"""7-day win rate and PnL analysis from Binance Income API."""
import time
from quantbox.core.settings import get_settings
from binance.client import Client

settings = get_settings()
client = Client(settings.binance_api_key, settings.binance_api_secret)

end = int(time.time() * 1000)
start = end - 7 * 24 * 3600 * 1000

income = client.futures_income_history(
    incomeType='REALIZED_PNL', startTime=start, endTime=end, limit=1000,
)

wins = 0
losses = 0
total_win = 0.0
total_loss = 0.0
by_symbol: dict[str, float] = {}

# Each income entry is a partial fill — group by symbol to get per-trade PnL
for r in income:
    pnl = float(r['income'])
    sym = r['symbol']
    if sym not in by_symbol:
        by_symbol[sym] = 0.0
    by_symbol[sym] += pnl
    if pnl > 0:
        wins += 1
        total_win += pnl
    elif pnl < 0:
        losses += 1
        total_loss += pnl

total = wins + losses
wr = wins / total * 100 if total > 0 else 0

print(f'=== 7-Day Trade Stats ===')
print(f'Win: {wins}, Loss: {losses}, Total: {total}')
print(f'Win Rate: {wr:.1f}%')
print(f'Gross Win:  ${total_win:+.2f}')
print(f'Gross Loss: ${total_loss:+.2f}')
print(f'Net PnL:    ${total_win + total_loss:+.2f}')
print()
print('=== Per-Symbol Net PnL ===')
for sym, pnl in sorted(by_symbol.items(), key=lambda x: x[1]):
    marker = '✅' if pnl > 0 else '❌'
    print(f'  {marker} {sym:16s} ${pnl:+.2f}')

# Also count unique "trade rounds" — consecutive fills on same symbol
# grouped by realized PnL sign changes
print()
print('=== Unique Trade Count (by symbol close events) ===')
symbol_trades: dict[str, list[float]] = {}
for r in income:
    sym = r['symbol']
    pnl = float(r['income'])
    if sym not in symbol_trades:
        symbol_trades[sym] = []
    symbol_trades[sym].append(pnl)

trade_wins = 0
trade_losses = 0
for sym, pnls in symbol_trades.items():
    net = sum(pnls)
    if net > 0:
        trade_wins += 1
    elif net < 0:
        trade_losses += 1

trade_total = trade_wins + trade_losses
trade_wr = trade_wins / trade_total * 100 if trade_total > 0 else 0
print(f'Unique symbols traded: {len(symbol_trades)}')
print(f'Symbol-level wins: {trade_wins}, losses: {trade_losses}')
print(f'Symbol-level win rate: {trade_wr:.1f}%')
