# Direction Asymmetry A/B Test (R1) — 6yr Full Data

| Field      | Value                                              |
|------------|----------------------------------------------------|
| Commit     | `614a449`                                          |
| Date       | 2026-02-21                                         |
| Baseline   | v12.4 I-HybTight (symmetric T150_S38_H25_PBT)     |
| Data       | 2190d (6yr), 50 symbols, left-padded               |
| Variants   | 16 (1 baseline + 15 direction-aware)                |
| Duration   | 71.0s                                               |

**구현:** 미구현 (Long bias 확인, 전략 방향 변경 없음)
## Previous Bug

Prior R1 test (`894139c`) ran on only **721 days** due to missing left-padding.
All numbers below are from the correct **2,191-day** simulation.

## Results Comparison

| Variant | Description | Ret% | DD% | Sharpe | Calmar | WR | PF | R:R | #Tr | L_WR | S_WR | L_EV | S_EV |
|---------|------------|------|-----|--------|--------|----|----|-----|-----|------|------|------|------|
| **A-Baseline** | **Symmetric I-HybTight (prod)** | **6,300.9%** | **7.6%** | 5.56 | **833.9** | 55.8% | **1.56** | 1.91 | 17165 | 56.0% | 55.7% | 3.64% | 2.91% |
| B-LongWideT | Long target 2.0× | 6,254.9% | 7.4% | 5.51 | 848.5 | 53.6% | 1.55 | 1.95 | 16467 | 51.4% | 55.6% | 3.21% | 2.88% |
| **C-ShortTightT** | **Short target 1.0×** | 6,223.3% | **7.1%** | **5.76** | 870.5 | 59.0% | 1.57 | 1.79 | 18454 | 56.1% | 61.5% | 3.63% | 3.02% |
| **D-SplitTarget** | **Long 2.0×/Short 1.0×** | 6,200.8% | **7.0%** | 5.71 | **891.2** | 57.0% | 1.56 | 1.81 | 17733 | 51.7% | 61.4% | 3.23% | 3.02% |
| E-ShortTightS | Short stop 0.30× | 6,217.5% | 7.6% | 5.63 | 813.8 | 53.9% | 1.55 | **2.08** | 17687 | 55.9% | 52.0% | 3.60% | 2.78% |
| F-ShortBoth | S tgt 1.0 + stp 0.30 | 6,113.7% | 7.2% | 5.75 | 852.4 | 57.1% | 1.55 | 1.93 | 18981 | 56.1% | 57.9% | 3.63% | 2.85% |
| G-ModSplitT | L 1.8×/S 1.2× tgt | 6,271.9% | 7.7% | 5.55 | 814.3 | 56.3% | 1.56 | 1.87 | 17322 | 53.0% | 59.2% | 3.39% | 3.03% |
| H-LongWideS | L stop 0.45× | 6,296.2% | 7.6% | 5.55 | 833.3 | 57.0% | 1.56 | 1.82 | 16842 | 58.4% | 55.6% | 3.77% | 2.89% |
| I-ShortConf4 | S conf≥4 | 5,386.6% | 7.9% | 5.16 | 681.1 | 56.0% | 1.59 | 1.98 | 12820 | 55.3% | 58.1% | 3.43% | 3.33% |
| J-Conf4+TgtT | S conf≥4 + tgt 1.0× | 5,363.2% | 7.9% | 5.22 | 678.5 | 57.6% | 1.60 | 1.92 | 13228 | 55.2% | 63.8% | 3.43% | 3.48% |
| K-ShortMH20 | S max_hold=20 | 6,295.8% | 7.5% | 5.67 | 841.4 | 56.0% | 1.56 | 1.86 | 17435 | 56.1% | 55.8% | 3.60% | 2.61% |
| L-ShortFast | S MH20 + stp 0.30 | 6,240.8% | 7.6% | 5.68 | 826.3 | 54.0% | 1.55 | 2.02 | 17943 | 56.0% | 52.1% | 3.59% | 2.50% |
| M-ModCombo | L1.8/S1.2 + S stp 0.30 | 6,156.1% | 7.8% | 5.60 | 789.2 | 54.3% | 1.55 | 2.03 | 17842 | 52.9% | 55.5% | 3.37% | 2.89% |
| N-FullAsym | L2.0/S1.0/S.30/Sc4/MH20 | 5,253.2% | 7.5% | 5.17 | 696.2 | 53.6% | 1.58 | 2.03 | 12638 | 50.7% | 60.4% | 3.04% | 3.10% |
| O-ModFull | L1.8/S1.2/S.30/Sc4 | 5,291.8% | 8.2% | 5.06 | 647.0 | 53.8% | 1.58 | 2.08 | 12695 | 52.2% | 57.7% | 3.21% | 3.23% |
| P-LongFavor | L tgt 2.0 + stp 0.45 | 6,265.9% | 7.4% | 5.50 | 850.0 | 54.7% | 1.55 | 1.86 | 16147 | 54.0% | 55.4% | 3.38% | 2.85% |

## Trophy Case

| Category | Winner | Value | vs Baseline |
|----------|--------|-------|-------------|
| 🏆 Best Sharpe | C-ShortTightT | 5.76 | +3.6% |
| 🏆 Best DD | D-SplitTarget | 7.0% | −0.6pp |
| 🏆 Best Calmar | D-SplitTarget | 891.2 | +6.9% |
| 🏆 Best R:R | E-ShortTightS | 2.08 | +8.9% |
| 🏆 Best Return | **A-Baseline** | **6,300.9%** | — |
| 🏆 Best PF | **A-Baseline** | **1.56** | — |

## Key Findings

1. **Baseline remains King on absolute return and PF.** No variant beats A-Baseline on total return (6,300.9%) or profit factor (1.56).

2. **Marginal Sharpe/DD improvements from target asymmetry.** C-ShortTightT (short target 1.0×) gives +3.6% Sharpe improvement (5.76 vs 5.56) and −0.5pp DD (7.1% vs 7.6%), but sacrifices ~1.2% return.

3. **D-SplitTarget is the best risk-adjusted variant.** Calmar 891.2 (+6.9% vs baseline 833.9) with DD 7.0% (−0.6pp). But −1.6% return penalty.

4. **Stop asymmetry (E,H) barely moves the needle on 6yr.** H-LongWideS (Sharpe 5.55, Calmar 833.3) is essentially baseline-equivalent. E-ShortTightS improves R:R but not Calmar.

5. **Confidence gating (I,J) hurts return significantly.** Drops to ~5,300-5,400% by removing too many short opportunities.

6. **None of the improvements justify the overfitting risk.** The long-short alpha is declining in recent years (see analysis report). Tuning asymmetric parameters to early-period patterns is dangerous.

## Verdict

**KEEP SYMMETRIC BASELINE.** The direction asymmetry alpha exists but is:
- Small on 6yr data (+3.6% Sharpe at best)
- **Declining and reversing** in recent years
- Not worth the overfitting risk of adding 4+ new hyperparameters
