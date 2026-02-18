"""Quick check: do ZigZag pivot prices match true bar-range extremes?"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from binance.client import Client
from quantbox.core.settings import get_settings
from quantbox.strategy.elliott import (
    ElliottConfig, zigzag_pivots, detect_impulse_abc_patterns,
)

import time
from datetime import datetime, timedelta, timezone


def download_4h(client, symbol, days=365):
    end_dt = datetime.now(timezone.utc)
    start_ms = int((end_dt - timedelta(days=days)).timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    all_k = []
    cur = start_ms
    while cur < end_ms:
        klines = client.futures_klines(symbol=symbol, interval="4h",
                                       startTime=cur, endTime=end_ms, limit=1000)
        if not klines:
            break
        all_k.extend(klines)
        cur = int(klines[-1][6]) + 1
        if len(klines) < 1000:
            break
        time.sleep(0.1)
    return {
        "high": np.array([float(k[2]) for k in all_k], dtype=np.float64),
        "low": np.array([float(k[3]) for k in all_k], dtype=np.float64),
        "close": np.array([float(k[4]) for k in all_k], dtype=np.float64),
    }


settings = get_settings()
client = Client(settings.binance_api_key, settings.binance_api_secret)
cfg = ElliottConfig()

# warmup
dummy = np.random.randn(100).cumsum() + 100
zigzag_pivots(dummy + 1, dummy - 1, 0.03)

for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    data = download_4h(client, sym, 365)
    high, low = data["high"], data["low"]

    for thresh in cfg.zigzag_thresholds:
        pidx, pprice, ptype, n_piv = zigzag_pivots(high, low, thresh)
        if n_piv < 9:
            continue

        pats = detect_impulse_abc_patterns(pidx, pprice, ptype, n_piv, cfg)

        for p in pats:
            # Get bar indices for W0..C
            # Find the 9-pivot window in the zigzag
            c_idx = p['c_idx']  # bar index where C is

            # Reconstruct indices from pattern
            w0, w1, w2, w3, w4, w5 = p['w0'], p['w1'], p['w2'], p['w3'], p['w4'], p['w5']
            a, b, c = p['a_price'], p['b_price'], p['c_price']

            # Find matching pivot indices
            w_indices = []
            for i in range(n_piv - 8):
                prices = [float(pprice[i + j]) for j in range(9)]
                if (abs(prices[0] - w0) < 1e-6 and abs(prices[5] - w5) < 1e-6
                        and abs(prices[8] - c) < 1e-6):
                    w_indices = [int(pidx[i + j]) for j in range(9)]
                    break

            if not w_indices:
                continue

            w0_idx, w1_idx, w2_idx, w3_idx, w4_idx, w5_idx = w_indices[:6]
            a_idx, b_idx, c_bar_idx = w_indices[6:]

            # True extremes between W0 and W5
            impulse_true_low = np.min(low[w0_idx:w5_idx + 1])
            impulse_true_high = np.max(high[w0_idx:w5_idx + 1])

            # Pivot-based measurement
            if p['signal'] == 1:  # Bullish: W0 is low, W5 is high
                pivot_low = w0
                pivot_high = w5
            else:  # Bearish: W0 is high, W5 is low
                pivot_low = w5
                pivot_high = w0

            diff_low = abs(impulse_true_low - pivot_low)
            diff_high = abs(impulse_true_high - pivot_high)
            pct_err_low = diff_low / pivot_low * 100 if pivot_low > 0 else 0
            pct_err_high = diff_high / pivot_high * 100 if pivot_high > 0 else 0

            impulse_pivot = abs(w5 - w0)
            impulse_true = impulse_true_high - impulse_true_low
            range_diff_pct = abs(impulse_true - impulse_pivot) / impulse_pivot * 100 if impulse_pivot > 0 else 0

            print(f"{sym} t={thresh:.3f} {p['type'][:8]}  "
                  f"pivot_range={impulse_pivot:.2f}  true_range={impulse_true:.2f}  "
                  f"range_err={range_diff_pct:.2f}%  "
                  f"low_err={pct_err_low:.3f}%  high_err={pct_err_high:.3f}%")
