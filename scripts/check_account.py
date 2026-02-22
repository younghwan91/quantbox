"""Quick Binance account status check via API."""
from binance.client import Client
from quantbox.core.settings import QuantBoxSettings

s = QuantBoxSettings()
c = Client(s.binance_api_key, s.binance_api_secret)

# Balance
bals = c.futures_account_balance()
for b in bals:
    if b['asset'] == 'USDT':
        print(f"USDT Balance: {b['balance']}, Available: {b['availableBalance']}")

# Open positions
positions = c.futures_position_information()
for p in positions:
    amt = float(p['positionAmt'])
    if amt != 0:
        entry = float(p['entryPrice'])
        mark = float(p['markPrice'])
        upnl = float(p['unRealizedProfit'])
        side = 'LONG' if amt > 0 else 'SHORT'
        pct = ((mark - entry) / entry * 100) if amt > 0 else ((entry - mark) / entry * 100)
        lev = p.get('leverage', '?')
        print(f"{p['symbol']} | {side} | Amt={amt} | Entry=${entry} | Mark=${mark:.4f} | uPnL=${upnl:.2f} ({pct:+.2f}%) | {lev}x")

# Open algo orders (SL/TP)
try:
    import requests, time, hashlib, hmac
    from urllib.parse import urlencode
    base = 'https://fapi.binance.com'
    ts = int(time.time() * 1000)
    params = {'timestamp': ts}
    qs = urlencode(params)
    sig = hmac.new(s.binance_api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    headers = {'X-MBX-APIKEY': s.binance_api_key}
    r = requests.get(f'{base}/fapi/v1/openOrders?{qs}&signature={sig}', headers=headers)
    orders = r.json()
    print(f"\nOpen orders: {len(orders)}")
    for o in orders:
        print(f"  {o['symbol']} {o['side']} {o['type']} qty={o['origQty']} stop={o.get('stopPrice','N/A')} status={o['status']}")
except Exception as e:
    # Fallback to python-binance
    orders = c.futures_get_open_orders()
    print(f"\nOpen orders: {len(orders)}")
    for o in orders:
        print(f"  {o['symbol']} {o['side']} {o['type']} qty={o['origQty']} stop={o.get('stopPrice','N/A')} status={o['status']}")
