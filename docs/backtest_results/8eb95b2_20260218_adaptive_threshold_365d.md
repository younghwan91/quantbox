# Adaptive Threshold A/B Test — 365 Days

**Commit:** `8eb95b2` (2026-02-18)  
**Message:** feat: adaptive threshold A/B test infrastructure  
**Script:** `scripts/rb_adaptive_ab_test.py`, `scripts/rb_adaptive_sweep.py`  
**Data:** 365 days ending 2026-02-18, 20 symbols, 15m + 4h bars  
**Full Report:** [docs/adaptive_threshold_ab_test_report.md](../adaptive_threshold_ab_test_report.md)  

---

## Context

Fixed threshold (3.2%) produced zero entries in current live regime (BTC max 7d move = 2.83%). Adaptive threshold dynamically adjusts via ATR to generate entries in all volatility regimes.

## Experiment 1: Initial A/B Test (6 variants)

| Strategy | ΣRet% | Sharpe | MDD% | Trades | WR% |
|---|---|---|---|---|---|
| Fixed 3.2% (control) | +5,491 | 4.30 | 15.2% | 1,031 | 71.2% |
| Adaptive 2.0x, 48-bar | +92,250 | 7.38 | 15.3% | 5,863 | 66.5% |
| Adaptive 2.5x, 48-bar | +58,256 | 7.34 | 18.9% | 4,324 | 68.3% |
| Adaptive 3.0x, 48-bar | +30,063 | 7.10 | 14.3% | 2,792 | 70.3% |
| Adaptive 2.0x, 96-bar | +98,613 | 7.43 | 15.3% | 5,902 | 66.8% |
| **Adaptive 2.0x, 48-bar, f1.0%** | **+162,949** | **7.76** | **15.2%** | **9,022** | **65.5%** |

## Experiment 2: Walk-Forward Sweep (18 variants)

Grid: ATR mult (2.0/2.5/3.0) × Floor (1.0/1.2/1.5/1.8/2.0/2.5%) — Cap=6%, Period=48.

**Top 5 by OOS Sharpe (overfitting-resistant):**

| Variant | IS Sharpe | OOS Sharpe | OOS/IS Ratio | Full Sharpe | Last-30d Signals |
|---|---|---|---|---|---|
| 2.5x, f1.0% | 7.90 | 5.18 | 0.66 | 7.78 | 339 |
| 2.5x, f1.2% | 7.60 | 5.12 | 0.67 | 7.56 | 237 |
| 2.0x, f1.0% | 8.42 | 5.06 | 0.60 | 7.76 | 525 |
| 3.0x, f1.0% | 7.54 | 4.86 | 0.64 | 7.10 | 198 |
| 2.0x, f1.2% | 8.00 | 4.84 | 0.60 | 7.46 | 467 |

## Chosen Production Config

| Parameter | Value | Rationale |
|---|---|---|
| ATR Multiplier | 2.5× | Balance between signal count and quality |
| ATR Period | 48 bars (12h) | Standard responsiveness |
| Floor | 1.0% | Allows entries in compressed vol |
| Cap | 6.0% | Prevents deadlock in extreme vol |

**Full-period result (2.5x_f1):** Sharpe 7.78, MDD 18.9%, 4,324 trades, WR 68.3%, ΣRet +58,256%

## Experiment 3: Leverage Optimization

Conservatively chose **1.5x base × 2.5x MTF = 3.75x effective** to balance Sharpe vs drawdown. Higher leverage increases returns but degrades Sharpe monotonically.
