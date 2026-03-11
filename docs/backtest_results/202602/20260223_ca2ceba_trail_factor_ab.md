# Trail Factor A/B Test — Round 4 (R12-Parity Baseline)

- **Commit:** `0f654c4` (branch: `feature/recent-regime-ab`)
- **Date:** 2026-02-23 13:09 UTC
- **Data:** 2190d (6yr), 50 symbols, 15m+4h candles
- **Baseline:** U-RR+VWS (R12-identical: `rr_size=True`, `vwma_stop='wide'`, close-based `gain` for ALL exit checks)
- **Focus:** Pre-TP1 trail factor sweep + stop/hold combos
- **Scoring:** Recency-weighted composite (Y5-6 = 2× weight)
- **Runtime:** 164.8s (VPVR 105.0s + 18 variants × ~2.5s)

**구현:** 미구현 (trail factor 민감도 탐색)
## Context

R1-R2 used stripped config (rr_size=off, vwma_stop=off) → baseline 801%.
R3 added VWMA/VPVR/RR but used `worst_gain`/`best_gain` (intra-bar) → baseline 1,553%.
**R4 fix:** Reverted to R12's close-based `gain` for ALL exit checks (TP1/stop/trail/runner).
Baseline now matches R12: **6,427% vs R12's 6,301%** (2% diff from VPVR timing edge effects).

### R3→R4 Changes (key fixes)
1. TP1 trigger: `best_gain >= target` → `gain >= target` (close-based)
2. TP1 PnL: `target * lev` → `gain * lev` (actual close, includes overshoot)
3. TP1 hit: added `pos['best_pnl'] = gain` and `pos['stepup_active'] = False` reset
4. Stop check: `worst_gain < -effective_stop` → `gain < -effective_stop`
5. Stop PnL: `-effective_stop` → `gain` (actual close, may recover)
6. best_pnl tracking: moved from top (using `best_gain`) to else-branch (using `gain`)
7. Trail/runner checks: `worst_gain` → `gain` everywhere
8. Runner PnL: `max(gain, trail_stop)` → `gain` (actual close)

## Results Summary

| Var | Desc | Ret% | DD% | Sharpe | Calmar | WR | PF | R:R | TP1% | #Tr |
|-----|------|-----:|----:|-------:|-------:|---:|---:|----:|-----:|----:|
| **A-Prod** | Production TF30 (baseline) | 6,427% | 6.4% | 5.77 | 1004.0 | 58.2% | 1.66 | 1.76 | 17% | 16,584 |
| B-TF20 | trail_factor=0.2 | 6,567% | 6.5% | 5.80 | 1009.6 | 57.3% | 1.68 | 1.85 | 18% | 16,547 |
| C-TF10 | trail_factor=0.10 | 6,478% | 6.4% | 5.81 | 1009.6 | 55.6% | 1.66 | 1.97 | 18% | 16,485 |
| D-TF15 | trail_factor=0.15 | 6,553% | 6.6% | 5.82 | 999.5 | 56.5% | 1.67 | 1.91 | 18% | 16,519 |
| E-TF18 | trail_factor=0.18 | 6,573% | 6.6% | 5.82 | 1001.7 | 57.0% | 1.68 | 1.87 | 18% | 16,534 |
| F-TF22 | trail_factor=0.22 | 6,518% | 6.5% | 5.79 | 1002.0 | 57.4% | 1.67 | 1.84 | 18% | 16,557 |
| G-TF25 | trail_factor=0.25 | 6,520% | 6.6% | 5.80 | 988.2 | 57.8% | 1.67 | 1.81 | 17% | 16,572 |
| H-TF20MH20 | TF20 + max_hold 20 | 6,556% | 6.4% | **6.00** | 1017.4 | 57.6% | 1.68 | 1.73 | 15% | 16,998 |
| I-TF20MH15 | TF20 + max_hold 15 | 6,159% | 6.6% | 6.07 | 928.1 | 57.6% | 1.66 | 1.58 | 11% | 17,436 |
| J-TF20S30 | TF20 + stop_mult 0.30 | 6,603% | 6.6% | 5.75 | 1005.9 | 55.1% | 1.66 | 2.01 | 17% | 17,103 |
| K-TF20S25 | TF20 + stop_mult 0.25 | 6,561% | 6.6% | 5.72 | 995.9 | 53.8% | 1.64 | **2.10** | 16% | 17,466 |
| L-TF15MH20 | TF15 + max_hold 20 | 6,598% | 6.3% | 6.03 | 1046.5 | 57.1% | 1.68 | 1.77 | 15% | 16,980 |
| **M-TF15S30** | TF15 + stop 0.30 ★ WINNER | **6,653%** | 6.4% | 5.79 | 1040.7 | 54.5% | 1.66 | 2.07 | 17% | 17,069 |
| N-TF20S30H20 | TF20 + S30 + MH20 | 6,638% | 6.3% | 5.95 | 1045.5 | 55.4% | 1.67 | 1.88 | 14% | 17,532 |
| O-TF20S30H15 | TF20 + S30 + MH15 | 6,239% | 6.6% | 6.05 | 950.4 | 55.8% | 1.64 | 1.71 | 10% | 17,966 |
| **P-TF15S30H20** | TF15+S30+MH20 | **6,700%** | **6.2%** | 6.00 | **1072.9** | 54.9% | 1.67 | 1.92 | 14% | 17,512 |
| Q-TF15S30H15 | TF15 + S30 + MH15 | 6,279% | 6.5% | **6.09** | 962.6 | 55.5% | 1.65 | 1.73 | 10% | 17,948 |
| R-TF20S25H20 | TF20 + S25 + MH20 | 6,598% | 6.4% | 5.91 | 1027.5 | 54.0% | 1.65 | 1.98 | 14% | 17,882 |

## Recency-Weighted Ranking

| Rank | Var | Score | vs Base | Verdict |
|-----:|-----|------:|--------:|---------|
| 1 | **M-TF15S30** | 1.090 | +9.0% | ★ WINNER |
| 2 | P-TF15S30H20 | 1.081 | +8.1% | beat base |
| 3 | K-TF20S25 | 1.080 | +8.0% | beat base |
| 4 | J-TF20S30 | 1.079 | +7.9% | beat base |
| 5 | C-TF10 | 1.078 | +7.8% | beat base |
| 6 | B-TF20 | 1.076 | +7.6% | beat base |
| 7 | E-TF18 | 1.075 | +7.5% | beat base |
| 8 | D-TF15 | 1.074 | +7.4% | beat base |
| 9 | N-TF20S30H20 | 1.072 | +7.2% | beat base |
| 10 | R-TF20S25H20 | 1.072 | +7.2% | beat base |
| 11 | F-TF22 | 1.070 | +7.0% | beat base |
| 12 | L-TF15MH20 | 1.067 | +6.7% | beat base |
| 13 | G-TF25 | 1.066 | +6.6% | beat base |
| 14 | H-TF20MH20 | 1.058 | +5.8% | beat base |
| 15 | Q-TF15S30H15 | 1.014 | +1.4% | beat base |
| 16 | O-TF20S30H15 | 1.007 | +0.7% | beat base |
| 17 | I-TF20MH15 | 0.997 | -0.3% | below base |

## Yearly Breakdown (Top 3)

### M-TF15S30 (trail_factor=0.15, stop_mult=0.30) ★ WINNER
| Year | Sharpe | R:R | TP1% |
|------|-------:|----:|-----:|
| Y1 | 10.5 | 2.03 | 21% |
| Y2 | 7.9 | 1.95 | 17% |
| Y3 | 6.4 | 1.94 | 16% |
| Y4 | 6.2 | 2.09 | 15% |
| Y5 | 6.1 | 2.04 | 17% |
| Y6 | 5.8 | 2.21 | 18% |

### P-TF15S30H20 (trail_factor=0.15, stop_mult=0.30, max_hold=20)
| Year | Sharpe | R:R | TP1% |
|------|-------:|----:|-----:|
| Y1 | 10.8 | 1.83 | 18% |
| Y2 | 8.2 | 1.87 | 14% |
| Y3 | 6.2 | 1.85 | 13% |
| Y4 | 6.2 | 1.89 | 12% |
| Y5 | 6.3 | 1.87 | 15% |
| Y6 | 6.7 | 2.08 | 15% |

### K-TF20S25 (trail_factor=0.20, stop_mult=0.25)
| Year | Sharpe | R:R | TP1% |
|------|-------:|----:|-----:|
| Y1 | 10.3 | 2.07 | 20% |
| Y2 | 7.8 | 2.00 | 17% |
| Y3 | 6.2 | 1.96 | 15% |
| Y4 | 5.5 | 2.14 | 15% |
| Y5 | 6.2 | 2.07 | 16% |
| Y6 | 6.0 | 2.24 | 17% |

## Exit Reason Breakdown

| Var | stop | time | trail | tp1 | trail_runner | time_runner |
|-----|-----:|-----:|------:|----:|-------------:|------------:|
| A-Prod | 5,947 | 4,238 | 1,531 | 2,434 | 2,434 | 0 |
| M-TF15S30 | 6,614 | 4,376 | 1,093 | 2,493 | 2,493 | 0 |
| P-TF15S30H20 | 6,610 | 5,662 | 832 | 2,204 | 2,204 | 2 |
| K-TF20S25 | 7,080 | 4,188 | 1,238 | 2,480 | 2,480 | 0 |

## Key Findings

1. **Baseline validated:** A-Prod = 6,427% (matches R12's 6,301% within 2%).
   All 18 variants tested on R12-identical simulate() logic.

2. **Differences are much smaller with correct baseline:**
   - R3 (intra-bar) showed massive C-TF10 dominance (+118% vs base).
   - **R4 (close-based) shows modest M-TF15S30 dominance (+226% absolute, +3.5% relative).**
   - The apparent trail factor signal in R3 was largely an artifact of intra-bar exit mechanics.

3. **Winner: M-TF15S30** (trail_factor=0.15, stop_mult=0.30)
   - Recency score: 1.090 (+9.0% vs base)
   - Ret 6,653%, DD 6.4%, R:R 2.07, TP1% 17%
   - Y6 Sharpe 5.8 — healthy recency performance.

4. **P-TF15S30H20 is the robust alternative:**
   - Best return (6,700%), best DD (6.2%), best Calmar (1072.9).
   - Y6 Sharpe 6.7 — strongest recent performance.
   - Trade-off: lower TP1% (14% vs 17%) due to shorter max_hold.

5. **Trail factor alone has marginal impact:**
   - TF10→TF30 spread is only ~50-140bp in return.
   - Real alpha comes from tighter stop (0.30 vs 0.382) + shorter hold (20 vs 25).

6. **16 of 17 variants beat baseline.** Only I-TF20MH15 (max_hold=15) slightly under.

## Recommendation

**M-TF15S30** (`pre_tp1_factor=0.15`, `stop_mult=0.30`) as top candidate:
- Best recency-weighted composite score (+9.0%).
- R:R improves from 1.76 → 2.07 (17% uplift).
- Same max_hold=25 (minimal production change, only 2 params).
- Risk: tighter stop (0.30 vs 0.382) increases stop-out rate by ~11%.

**Conservative alternative: P-TF15S30H20** (adds max_hold=20):
- Best absolute return, Calmar, and DD.
- Y6 Sharpe 6.7 (strongest recency signal).
- 3 parameter changes vs 2.

**Note:** Improvements are modest (~3-4% recency uplift vs well-optimized production).
Consider whether marginal gain justifies deployment risk.
