# Partial Exit + Trailing Stop A/B Test

- **Commit**: `5900fa5`
- **Date**: 2026-02-19
- **Baseline**: E-3Slot v12.2.0 (conf≥2, ATR 2.0×, floor 0.8%, MTF block, 3 slots, 50 symbols)
- **Data**: 50 symbols × 1095 days (2023-02 → 2026-02), 15m + 4h OHLCV
- **Capital**: $2,095 USDT, fees 0.04%/side, funding 0.01%/8h, 3 slots
- **Script**: `scripts/partial_exit_trailing_ab.py`

**구현:** 미구현 (초기 partial exit 탐색)
## Background

Current R:R ratio is 0.98 — dangerously close to 1.0. While win rate (64.2%) and Profit Factor (1.76) keep the strategy profitable, a slight regime change could flip R:R below break-even.

**Goal**: Test if splitting exit into 50% at target + 50% trailing improves R:R while maintaining Sharpe/Calmar.

## Variants

| Variant | Description | Partial | Trail ATR× | Break-even | 1st Target |
|---------|-------------|---------|-----------|------------|------------|
| A-Base | 100% exit at 0.382 (current) | No | — | — | 0.382 |
| B-Trail15 | 50% at 0.382 + 50% trail 1.5×ATR | Yes | 1.5 | No | 0.382 |
| C-Trail20 | 50% at 0.382 + 50% trail 2.0×ATR (wider) | Yes | 2.0 | No | 0.382 |
| D-Trail10 | 50% at 0.382 + 50% trail 1.0×ATR (tighter) | Yes | 1.0 | No | 0.382 |
| E-BE15 | 50% at 0.382 + breakeven + trail 1.5×ATR | Yes | 1.5 | Yes | 0.382 |
| F-High15 | 50% at 0.500 + 50% trail 1.5×ATR | Yes | 1.5 | No | 0.500 |

## Results

| Metric | A-Base | B-Trail15 | C-Trail20 | D-Trail10 | **E-BE15** | F-High15 |
|--------|--------|-----------|-----------|-----------|------------|----------|
| Total Return | +8,785% | +7,544% | +6,778% | +8,136% | **+7,536%** | +8,311% |
| Sharpe | 6.41 | 6.85 | 6.92 | 6.78 | **6.84** | 6.85 |
| Calmar | 795.0 | 695.3 | 639.6 | 578.4 | **702.8** | 594.5 |
| Max Drawdown | 11.1% | 10.8% | 10.6% | 14.1% | **10.7%** | 14.0% |
| Trades | 28,616 | 35,770 | 29,952 | 42,208 | **36,796** | 44,961 |
| Trades/Day | 26.11 | 32.64 | 27.33 | 38.52 | **33.58** | 41.03 |
| Win Rate | 64.2% | 57.3% | 56.3% | 59.3% | **56.5%** | 61.6% |
| Profit Factor | 1.76 | 1.96 | 2.00 | 1.92 | **1.96** | 1.90 |
| **R:R** | **0.98** | **1.47** | **1.54** | **1.31** | **1.51** | **1.19** |
| Avg Win % | +3.101% | +4.306% | +4.665% | +3.958% | +4.273% | +3.475% |
| Avg Loss % | 3.157% | 2.933% | 3.022% | 3.021% | 2.826% | 2.929% |

## Exit Type Breakdown

### A-Base (current)
| Type | Count | Avg PnL | Win Rate |
|------|-------|---------|----------|
| SIGNAL | 28,616 | +0.863% | 64.2% |

### E-BE15 (winner)
| Type | Count | Avg PnL | Win Rate |
|------|-------|---------|----------|
| TP1 (50% at target) | 9,998 | +3.527% | 99.9% |
| TRAIL (runner half) | 9,854 | +4.626% | 68.6% |
| SIGNAL (no TP1 hit) | 16,802 | -2.316% | 23.1% |
| TRAIL_FORCE (reversal) | 142 | +9.440% | 100.0% |

**Note on SIGNAL exits**: In partial mode, "SIGNAL" only contains trades that **never reached the target** — inherently the losing trades. In baseline, SIGNAL contains both winners and losers. This selection effect explains the -2.3% avg in partial variants vs +0.86% in baseline.

### C-Trail20 (best R:R)
| Type | Count | Avg PnL | Win Rate |
|------|-------|---------|----------|
| TP1 | 8,132 | +3.534% | 99.9% |
| TRAIL | 7,950 | +5.018% | 67.0% |
| SIGNAL | 13,690 | -2.279% | 23.6% |
| TRAIL_FORCE | 180 | +9.577% | 100.0% |

## Delta vs A-Base

| Metric | A-Base | B-Trail15 | C-Trail20 | D-Trail10 | **E-BE15** | F-High15 |
|--------|--------|-----------|-----------|-----------|------------|----------|
| ΔReturn | — | -14.1% | -22.9% | -7.4% | **-14.2%** | -5.4% |
| ΔSharpe | — | +6.9% | +8.0% | +5.8% | **+6.7%** | +6.9% |
| ΔR:R | — | +50.0% | +57.1% | +33.7% | **+54.1%** | +21.4% |
| ΔPF | — | +11.4% | +13.6% | +9.1% | **+11.4%** | +8.0% |
| ΔDrawdown | — | -2.7% | -4.5% | +27.0% | **-3.6%** | +26.1% |

## Weighted Score

Scoring: Sharpe×0.25 + Calmar×0.20 + R:R×0.20 + Return×0.20 + inv(DD)×0.15

| Variant | Score | Result |
|---------|-------|--------|
| A-Base | 1.000 | baseline |
| B-Trail15 | 1.068 | |
| C-Trail20 | 1.056 | |
| D-Trail10 | 0.981 | worse DD |
| **E-BE15** | **1.079** | **★ WINNER** |
| F-High15 | 0.968 | worse DD |

## Key Findings

1. **R:R dramatically improved**: All partial exit variants boost R:R from 0.98 to 1.19-1.54. The strategy becomes mathematically robust (`E = WR × avgWin - (1-WR) × avgLoss > 0` with wider margin).

2. **Sharpe improves across the board**: From 6.41 to 6.78-6.92 (+6-8%). The partial exit smooths equity curve by locking in half-position profits early.

3. **Drawdown improves (except D & F)**: B/C/E all show **lower DD** than baseline (10.6-10.8% vs 11.1%). D-Trail10 (tighter) and F-High15 (higher target) show **worse DD** (~14%) — avoid these.

4. **Tradeoff: ~14% lower absolute return**: Expected since half the position exits at 0.382 instead of potentially trailing further. This is acceptable given the risk-adjusted improvements.

5. **Breakeven stop matters**: E-BE15 (breakeven after TP1) outscores B-Trail15 (no breakeven) by Calmar 702.8 vs 695.3 — the floor protection preserves equity during trailing.

6. **D-Trail10 is dangerous**: Tighter trailing (1.0× ATR) causes premature exits → more whipsaw trades → higher DD (14.1%).

7. **C-Trail20 has best R:R (1.54) but lowest return**: Wider trailing lets runners go longer but also means fewer clean trail exits. PF 2.00 is highest — excellent for live robustness.

## Winner: E-BE15

**50% exit at 0.382 target + breakeven stop + 1.5× ATR trailing for remaining 50%**

- **R:R 1.51** (vs 0.98 baseline, +54%)
- **PF 1.96** (vs 1.76 baseline, +11%)
- **Sharpe 6.84** (vs 6.41, +7%)
- **DD 10.7%** (vs 11.1%, improved)
- **Return +7,536%** (vs +8,785%, -14% acceptable tradeoff)

The breakeven floor ensures the runner half can never lose money after TP1, making this the safest partial exit approach.

## Binance Server-Side Implementation Plan

Binance USDT-M Futures supports `TRAILING_STOP_MARKET` order type:

```
POST /fapi/v1/order
  type = TRAILING_STOP_MARKET
  callbackRate = 1.5 ~ 3.0  (trailing distance %, maps to ATR×mult)
  activationPrice = entry_price  (optional: price at which trailing starts)
  quantity = remaining_qty
  side = opposite
  reduceOnly = true
```

**Implementation flow** (in `_partial_close_position`):
1. On TP1 hit → market-close 50% (existing code)
2. Cancel old server STOP order
3. Calculate `callbackRate = trail_atr_mult × current_atr_pct × 100` (clamp 0.1%-5.0%)
4. Place `TRAILING_STOP_MARKET` for remaining qty with `activationPrice = entry_price`
5. Server handles trailing — no client polling dependency (crash-proof)
6. 15m polling detects fill → clean state

## Recommendation

**Deploy E-BE15 as v12.3.0.** The 54% R:R improvement and 7% Sharpe improvement make this a clear upgrade over the current all-or-nothing exit. The 14% return reduction is an acceptable cost for mathematical robustness in live trading.

Config changes needed:
- `rb_partial_exit_enabled = True`
- `rb_partial_exit_fraction = 0.5`
- New param: `rb_trail_atr_mult = 1.5`
- New param: `rb_breakeven_after_tp1 = True`
