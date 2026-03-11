# Direction Asymmetry R2: Combos & Fine-Tuning — 6yr Full Data

| Field      | Value                                              |
|------------|----------------------------------------------------|
| Commit     | `614a449`                                          |
| Date       | 2026-02-21                                         |
| Baseline   | v12.4 I-HybTight (symmetric T150_S38_H25_PBT)     |
| R1 Winners | H (L stop 0.45), E (S stop 0.30)                   |
| Data       | 2190d (6yr), 50 symbols, left-padded               |
| Variants   | 18 (1 baseline + 2 R1 refs + 15 combos)             |
| Duration   | 44.3s                                               |

**구현:** 미구현 (r1 후속 탐색)
## Previous Bug

Prior R2 test (`12e5c3b`) ran on only **721 days** due to missing left-padding.
All numbers below are from the correct **2,191-day** simulation.

## Results Comparison

| Variant | Description | Ret% | DD% | Sharpe | Calmar | WR | PF | R:R | #Tr |
|---------|------------|------|-----|--------|--------|----|----|-----|-----|
| **A-Baseline** | **Symmetric I-HybTight** | **6,300.9%** | **7.6%** | 5.56 | **833.9** | 55.8% | **1.56** | 1.91 | 17165 |
| R1-H | L stop 0.45 | 6,296.2% | 7.6% | 5.55 | 833.3 | 57.0% | 1.56 | 1.82 | 16842 |
| R1-E | S stop 0.30 | 6,217.5% | 7.6% | 5.63 | 813.8 | 53.9% | 1.55 | 2.08 | 17687 |
| HE-45_30 | L0.45 + S0.30 | 6,226.2% | 7.6% | 5.63 | 815.0 | 55.0% | 1.55 | 1.98 | 17366 |
| FL-40 | L0.40 + S0.30 | 6,249.0% | 7.6% | 5.65 | 817.9 | 54.2% | 1.55 | 2.05 | 17599 |
| FL-42 | L0.42 + S0.30 | 6,188.7% | 7.6% | 5.63 | 810.1 | 54.5% | 1.55 | 2.02 | 17501 |
| FL-48 | L0.48 + S0.30 | 6,246.0% | **8.9%** | 5.60 | 703.0 | 55.5% | 1.56 | 1.95 | 17225 |
| FL-50 | L0.50 + S0.30 | 6,278.6% | **8.9%** | 5.60 | 706.5 | 55.8% | 1.56 | 1.94 | 17167 |
| **FS-25** | **L0.45 + S0.25** | 6,210.8% | 7.6% | **5.73** | 812.4 | 53.6% | 1.55 | **2.09** | 17749 |
| FS-28 | L0.45 + S0.28 | 6,237.3% | 7.6% | 5.65 | 816.4 | 54.5% | 1.55 | 2.02 | 17519 |
| FS-33 | L0.45 + S0.33 | 6,169.9% | 7.6% | 5.60 | 807.6 | 55.7% | 1.54 | 1.91 | 17176 |
| FS-35 | L0.45 + S0.35 | 6,215.7% | **7.5%** | 5.59 | 824.0 | 56.3% | 1.55 | 1.86 | 17051 |
| HE+I | L0.45/S0.30 + S conf≥4 | 5,325.3% | 8.0% | 5.06 | 664.5 | 56.8% | **1.59** | 1.92 | 12600 |
| HE+K | L0.45/S0.30 + S MH20 | 6,229.2% | 7.5% | 5.72 | 832.9 | 55.1% | 1.55 | 1.93 | 17613 |
| HE+I+K | L0.45/S0.30 + Sc4 + MH20 | 5,300.5% | 7.8% | 5.07 | 681.6 | 56.8% | 1.59 | 1.90 | 12652 |
| HE+LH30 | L0.45/S0.30 + L MH30 | 6,054.7% | 7.9% | 5.53 | 765.3 | 54.8% | 1.53 | 2.01 | 17189 |
| HE+LH30K | L0.45/S0.30 + LMH30/SMH20 | 6,105.2% | 7.7% | 5.67 | 795.4 | 55.0% | 1.54 | 1.95 | 17433 |
| HE+LR4 | L0.45/S0.30 + L runner 4× | 6,226.2% | 7.6% | 5.63 | 815.0 | 55.0% | 1.55 | 1.98 | 17366 |

## Trophy Case

| Category | Winner | Value | vs Baseline |
|----------|--------|-------|-------------|
| 🏆 Best Sharpe | FS-25 | 5.73 | +3.1% |
| 🏆 Best DD | FS-35 | 7.5% | −0.1pp |
| 🏆 Best Calmar | **A-Baseline** | **833.9** | — |
| 🏆 Best Return | **A-Baseline** | **6,300.9%** | — |
| 🏆 Best PF | HE+I | 1.59 | +1.9% |

## H+E Stacking Analysis

| Config | Sharpe | DD% | Calmar |
|--------|--------|-----|--------|
| H alone (L stop 0.45) | 5.55 | 7.6% | 833.3 |
| E alone (S stop 0.30) | 5.63 | 7.6% | 813.8 |
| H+E combined | 5.63 | 7.6% | 815.0 |

**Verdict: SUBADDITIVE.** H+E stacking equals E alone. H adds nothing when E is present.
This makes sense — wider long stops (H) give longs more room, but tighter short stops (E) already capture the same edge by cutting short losers faster.

## Key Findings

1. **Baseline wins on Calmar (833.9) and Return (6,300.9%).** No combination beats it.

2. **FS-25 is the best Sharpe variant** (5.73, +3.1%) with the best R:R (2.09, +9.4%), but at the cost of −90bps return and identical DD.

3. **Wide long stops (L≥0.48) increase DD to 8.9%.** This is a dealbreaker — the Calmar drops from 833 to ~700.

4. **H+E stacking is subadditive.** No synergy between the two R1 winners. This suggests they're capturing the same underlying effect from different angles.

5. **Confidence gating (HE+I) kills return** (−15.5%) for +0.03 PF. Not worth it.

6. **HE+K (add S MH20) is the best non-baseline overall**: Sharpe 5.72, DD 7.5%, Calmar 832.9. But still doesn't beat baseline Calmar.

## Conclusion

**KEEP SYMMETRIC BASELINE.** The 6yr data tells a very different story from the (buggy) 2yr results:

| Metric | 2yr (buggy) FS-33 | 6yr FS-33 | 6yr Baseline |
|--------|-------------------|-----------|--------------|
| Sharpe | 5.64 | 5.60 | **5.56** |
| DD | 11.1% | 7.6% | **7.6%** |
| Calmar | 216.2 | 807.6 | **833.9** |
| Return | 2,366% | 6,170% | **6,301%** |

The direction asymmetry alpha that looked so promising on 2yr data **mostly disappears on 6yr data**.
The remaining edge (~3% Sharpe improvement) is not worth the additional complexity and overfitting risk,
especially since the long-short performance gap is **declining and reversing** in recent years (Yr5-6: shorts outperform longs).

**Production recommendation: No change. Keep I-HybTight symmetric.**
