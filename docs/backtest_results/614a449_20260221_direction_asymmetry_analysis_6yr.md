# Direction Asymmetry Analysis — 6yr Full Data

| Field      | Value                                              |
|------------|----------------------------------------------------|
| Commit     | `614a449`                                          |
| Date       | 2026-02-21                                         |
| Config     | v12.4 I-HybTight (T150_S38_H25_PBT + M-2SSTSX)   |
| Data       | 2190d (6yr), 50 symbols, left-padded to max_bars   |
| Capital    | $2,095 → $134,100 (6,300.9%)                       |
| Trades     | 17,165 (L: 8,452 / S: 8,713)                       |

## Previous Bug

Prior analysis (`d3de815`) ran on only **721 days** (TONUSDT shortest symbol) due to missing left-padding.
This report uses the **correct 2,191-day** simulation with left-padding matching `reversal_ab_test.py`.

## Key Findings: Long vs Short Asymmetry

| Metric              |    ALL |   LONG |  SHORT | L−S Gap |
|---------------------|--------|--------|--------|---------|
| Trade count         | 17,165 |  8,452 |  8,713 |    −261 |
| Win rate %          |  55.8% |  56.0% |  55.7% |  +0.3pp |
| Profit factor       |   1.56 |   1.64 |   1.48 |   +0.16 |
| Avg win %           |  9.97% | 10.36% |  9.60% |  +0.76% |
| Avg loss %          |  5.21% |  4.92% |  5.49% |  −0.57% |
| **R:R**             | **1.91** | **2.10** | **1.75** | **+0.35** |
| **EV/trade %**      | **3.27%** | **3.64%** | **2.91%** | **+0.73%** |
| Total PnL $         | $149,283 | $80,818 | $68,465 | +$12,353 |
| Avg bars held       |   14.8 |   14.8 |   14.8 |    +0.0 |

**Longs earn +25% more EV per trade** (3.64% vs 2.91%). The gap is driven by:
- **R:R gap of +0.35** (longs have higher avg wins AND lower avg losses)
- PF gap of +0.16 (1.64 vs 1.48)
- WR gap is negligible (+0.3pp)

## Yearly Breakdown

| Year           |  L# |  S# | L:S  | L WR  | S WR  | Gap   | L EV%  | S EV%  | L PF | S PF |
|----------------|------|------|------|-------|-------|-------|--------|--------|------|------|
| Yr1 (2020-21)  | 1087 | 1089 | 1.0x | 66.4% | 63.8% | +2.6 | 6.544% | 4.706% | 2.62 | 2.14 |
| Yr2 (2021-22)  | 1497 | 1364 | 1.1x | 57.0% | 55.9% | +1.1 | 4.679% | 3.512% | 1.70 | 1.49 |
| Yr3 (2022-23)  | 1442 | 1510 | 0.9x | 56.0% | 53.9% | +2.1 | 3.319% | 2.434% | 1.52 | 1.33 |
| Yr4 (2023-24)  | 1400 | 1557 | 0.9x | 54.4% | 53.1% | +1.3 | 2.833% | 2.117% | 1.52 | 1.36 |
| Yr5 (2024-25)  | 1559 | 1610 | 1.0x | 53.6% | 55.2% | −1.6 | 2.830% | 2.710% | 1.49 | 1.44 |
| **Yr6 (2025-26)** | 1467 | 1578 | 0.9x | **51.5%** | **54.6%** | **−3.1** | 2.353% | 2.592% | 1.39 | 1.40 |

**Critical insight:** Long advantage **erodes over time** and **reverses in Yr5-6**. Shorts outperform longs in recent years. This means direction-asymmetric parameter tuning may be overfitting to early-period alpha that no longer exists.

## Portfolio Summary

| Metric       | Value        |
|-------------|-------------|
| Total return | 6,300.9%    |
| Max DD       | 7.6%        |
| Sharpe       | 5.56        |
| Calmar       | 833.9       |
| Profit factor| 1.56        |
| R:R          | 1.91        |

## Conclusion

Long-short asymmetry exists (R:R 2.10 vs 1.75) but is **declining** and **reversed in recent years**.
Symmetric parameters may actually be more robust going forward than direction-specific tuning.
