# R32: Exit / Risk Management A/B Test
**Commit:** `daf75c9` | **Date:** 2026-02-28 | **Base:** G-ALL deduped (R31)

**구현:** 미구현 (exit 최적화 탐색; r43/r47에서 발전)
## Objective
Test exit/risk management techniques from rubber band era on the vol straddle strategy (G-ALL deduped). Can we improve the baseline's fixed SL/TP/timeout with adaptive exits, trailing stops, partial closes, or alternative parameter sweeps?

## Methodology
- **Base strategy:** G-ALL deduped (15m+1h+4h, per-symbol dedup)
  - D_mom8: VR<0.45, momentum-8, SL=1%, TP=5%, cost=0.08%
  - S_d8: VR<0.55, squeeze≥8, straddle, SL=1.5%, TP=5%, cost=0.16%
- **Data:** 50 symbols × 6yr × 15m OHLCV (8.3M bars), resampled to 1h/4h
- **Variants:** 17 total, 4-core parallel execution (16s total runtime)
- **OOS:** 67/33 temporal split, top 6 + baseline

## Variants Tested

| ID | Description | D_mom8 SL | S_d8 SL | TP | Special |
|---|---|---|---|---|---|
| A-Baseline | Current G-ALL deduped | 1.0% | 1.5% | 5% | — |
| B-ATR-SL | ATR(14)-adaptive SL [0.3%, 3.0%] | ATR×1.5 | ATR×1.5 | 5% | Adaptive stop |
| C1-Trail-10 | Trailing stop 1.0% | 1.0% | 1.5% | 5% | Trail activation |
| C2-Trail-15 | Trailing stop 1.5% | 1.0% | 1.5% | 5% | Trail activation |
| C3-Trail-20 | Trailing stop 2.0% | 1.0% | 1.5% | 5% | Trail activation |
| D-Partial | 50% at TP1=2%, runner to 8% | 1.0% | 1.5% | 5% | D_mom8 only |
| E1-SL-050 | Tight SL sweep | 0.50% | 0.75% | 5% | — |
| E2-SL-075 | SL sweep | 0.75% | 1.0% | 5% | — |
| E3-SL-125 | SL sweep | 1.25% | 1.75% | 5% | — |
| E4-SL-200 | Wide SL sweep | 2.0% | 2.5% | 5% | — |
| F1-TP-030 | Low TP | 1.0% | 1.5% | 3% | — |
| F2-TP-070 | Higher TP | 1.0% | 1.5% | 7% | — |
| F3-TP-100 | Wide TP | 1.0% | 1.5% | 10% | — |
| G-ATR-Trail | ATR SL + Trail 1.5% | ATR×1.5 | ATR×1.5 | 5% | Combined |
| H1-Hold-Half | Max hold ×0.5 | 1.0% | 1.5% | 5% | Faster timeout |
| H2-Hold-1.5x | Max hold ×1.5 | 1.0% | 1.5% | 5% | Longer patience |
| H3-Hold-2x | Max hold ×2.0 | 1.0% | 1.5% | 5% | Much longer |

## Phase 1: Full-Sample Results

### Combined D+S (deduped)

| Rank | Variant | N | PF | EV(bp) | t-stat | WR% | R:R | Ret% | DD% | P(PF≥1) | Δ vs A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | F2-TP-070 | 1537 | 1.338 | +46.0 | 4.94 | 38.8 | 2.11 | +42,364 | 54.4 | 100% | +2.6bp |
| 2 | H3-Hold-2x | 1597 | 1.384 | +45.7 | 6.06 | 46.6 | 1.59 | +70,686 | 24.7 | 100% | +2.3bp |
| 3 | H2-Hold-1.5x | 1600 | 1.378 | +44.7 | 5.97 | 46.8 | 1.57 | +61,359 | 24.7 | 100% | +1.3bp |
| **4** | **A-Baseline** | **1602** | **1.382** | **+43.4** | **5.95** | **47.2** | **1.55** | **+51,992** | **25.2** | **100%** | **—** |
| 5 | E3-SL-125 | 1575 | 1.333 | +39.8 | 5.27 | 50.7 | 1.30 | +25,523 | 36.4 | 100% | -3.6bp |
| 6 | E2-SL-075 | 1710 | 1.348 | +35.6 | 5.44 | 37.1 | 2.29 | +23,385 | 35.6 | 100% | -7.8bp |
| 7 | E4-SL-200 | 1527 | 1.299 | +35.1 | 4.56 | 59.1 | 0.90 | +10,477 | 39.3 | 100% | -8.3bp |
| 8 | H1-Hold-Half | 1633 | 1.357 | +35.1 | 5.35 | 46.0 | 1.59 | +17,156 | 31.8 | 100% | -8.3bp |
| 9 | B-ATR-SL | 1583 | 1.373 | +35.0 | 5.20 | 37.9 | 2.25 | +14,324 | 32.9 | 100% | -8.4bp |
| 10 | D-Partial | 1646 | 1.320 | +34.2 | 5.13 | 51.4 | 1.25 | +15,090 | 25.2 | 100% | -9.2bp |
| 11 | F3-TP-100 | 1517 | 1.211 | +32.1 | 2.95 | 31.8 | 2.59 | +3,349 | 57.6 | 100% | -11.3bp |
| 12 | E1-SL-050 | 1815 | 1.362 | +31.3 | 5.42 | 30.0 | 3.18 | +16,964 | 24.0 | 100% | -12.1bp |
| 13 | F1-TP-030 | 1721 | 1.147 | +12.0 | 2.56 | 59.6 | 0.78 | +467 | 41.8 | 100% | -31.4bp |
| 14 | C3-Trail-20 | 1813 | 0.975 | -1.9 | -0.40 | 38.3 | 1.57 | -51 | 88.3 | 34% | -45.3bp |
| 15 | G-ATR-Trail | 1828 | 0.844 | -10.8 | -2.46 | 38.0 | 1.38 | -90 | 94.9 | 1% | -54.2bp |
| 16 | C2-Trail-15 | 1966 | 0.765 | -15.3 | -4.17 | 33.2 | 1.54 | -96 | 98.2 | 0% | -58.7bp |
| 17 | C1-Trail-10 | 2088 | 0.688 | -19.2 | -6.19 | 36.3 | 1.21 | -99 | 98.8 | 0% | -62.6bp |

### D_mom8 Only (deduped)

| Variant | N | PF | EV(bp) | t | WR% | R:R | Ret% | DD% |
|---|---|---|---|---|---|---|---|---|
| F2-TP-070 | 498 | 1.701 | +52.8 | 4.10 | 28.3 | 4.31 | +1,029 | 13.2 |
| **A-Baseline** | **509** | **1.574** | **+42.0** | **3.91** | **30.3** | **3.63** | **+630** | **14.2** |
| E2-SL-075 | 547 | 1.605 | +37.4 | 3.93 | 24.7 | 4.90 | +574 | 19.0 |
| H1-Hold-Half | 526 | 1.446 | +28.3 | 3.05 | 34.2 | 2.78 | +294 | 13.5 |

### S_d8 Only (deduped)

| Variant | N | PF | EV(bp) | t | WR% | R:R | Ret% | DD% |
|---|---|---|---|---|---|---|---|---|
| H3-Hold-2x | 1091 | 1.346 | +47.5 | 4.90 | 55.7 | 1.07 | +9,968 | 24.7 |
| H2-Hold-1.5x | 1093 | 1.342 | +46.8 | 4.85 | 55.7 | 1.07 | +9,370 | 24.7 |
| **A-Baseline** | **1093** | **1.333** | **+44.0** | **4.66** | **55.1** | **1.09** | **+7,033** | **25.2** |
| C1/C2/C3/G (trail) | — | <0.87 | negative | — | — | — | **total loss** | >88% |

## Phase 2: OOS Results (67/33 split)

### Combined D+S (deduped)

| Variant | IS_N | IS_PF | IS_EV | OOS_N | OOS_PF | OOS_EV | OOS_t | P(PF≥1) | CI_lo | CI_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| **H3-Hold-2x** | 895 | 1.222 | +27.2 | **702** | **1.602** | **+68.6** | **5.91** | **100%** | **1.373** | **1.889** |
| H2-Hold-1.5x | 898 | 1.227 | +27.6 | 702 | 1.580 | +65.9 | 5.72 | 100% | 1.355 | 1.860 |
| A-Baseline | 900 | 1.230 | +26.7 | 702 | 1.582 | +64.1 | 5.69 | 100% | 1.356 | 1.860 |
| E2-SL-075 | 963 | 1.193 | +20.0 | 747 | 1.538 | +54.2 | 5.28 | 100% | 1.319 | 1.783 |
| E3-SL-125 | 886 | 1.236 | +28.4 | 689 | 1.465 | +54.7 | 4.70 | 100% | 1.243 | 1.738 |
| F2-TP-070 | 874 | 1.335 | +44.4 | 663 | 1.335 | +47.1 | 3.27 | 100% | 1.117 | 1.568 |

### D_mom8 OOS

| Variant | IS_PF | OOS_PF | OOS_EV | OOS_t |
|---|---|---|---|---|
| E2-SL-075 | 1.094 | 2.560 | +84.5 | 5.03 |
| F2-TP-070 | 1.369 | 2.302 | +88.5 | 4.10 |
| A-Baseline | 1.170 | 2.320 | +85.2 | 4.67 |
| H3-Hold-2x | 1.169 | 2.155 | +83.2 | 4.27 |

### S_d8 OOS

| Variant | IS_PF | OOS_PF | OOS_EV | OOS_t |
|---|---|---|---|---|
| H3-Hold-2x | 1.238 | 1.480 | +62.8 | 4.40 |
| H2-Hold-1.5x | 1.241 | 1.466 | +61.0 | 4.29 |
| A-Baseline | 1.248 | 1.433 | +55.7 | 3.98 |
| F2-TP-070 | 1.326 | 1.176 | +30.0 | 1.65 |

## Key Findings

### 1. Trailing stops are catastrophic for straddle (C/G variants)
- All trailing stop variants **completely destroy** S_d8 performance (PF<0.87, >88% DD)
- The straddle relies on one leg going to TP(5%) while the other takes a capped SL
- Trailing stops prematurely close the profitable leg before it reaches TP, while the losing leg still takes its full SL
- **Rule: Never trail straddle legs. The fixed SL/TP structure IS the edge.**

### 2. Baseline is near-optimal
- A-Baseline ranks #4 full-sample, #3 OOS — already very strong
- Only H3-Hold-2x beats it OOS, and by a tiny margin (+4.5bp, +0.020 PF)
- The SL=1%/1.5%, TP=5% parameterization is a demonstrated sweet spot

### 3. Longer hold helps marginally (H2/H3)
- Extending max hold to 1.5x-2x improves S_d8 slightly (more time for TP to hit)
- D_mom8 is essentially unchanged (most trades resolve within baseline hold)
- H3-Hold-2x: OOS PF=1.602 vs Baseline PF=1.582 (+0.020, within noise)
- **Verdict: Marginal, not worth the complexity. Keep baseline hold.**

### 4. TP=7% (F2) boosts D_mom8 but hurts S_d8 OOS
- D_mom8 full-sample EV jumps from +42.0 to +52.8bp with TP=7%
- But S_d8 OOS PF drops from 1.433 to 1.176 — wider TP means fewer straddle wins
- Combined OOS EV is lower (+47.1 vs +64.1bp)
- **Verdict: TP=5% is the correct balance for D+S combined.**

### 5. ATR-adaptive SL (B) adds no value
- Full-sample: EV drops -8.4bp, DD increases +7.7%
- The vol regime filter (VR) already ensures entry only during specific vol conditions
- Adding adaptive SL on top adds noise without benefit
- **Verdict: Fixed SL is better given the entry filter already conditions on vol.**

### 6. Partial exit (D) for D_mom8 underperforms
- Split into 50% TP1=2% + runner to 8% with BE floor
- Full-sample EV drops from +42.0 to +14.9bp on D_mom8
- The TP1=2% target is too close — locks in small gains that don't compensate for runner BE stops
- **Verdict: Let the high R:R of D_mom8 (3.63:1) work. Don't partial exit.**

## Conclusion

**The baseline (A-Baseline) is already near-optimal.** The fixed SL/TP/timeout structure is the correct design for this strategy:
- Straddle's edge comes from the asymmetric payoff (one leg TP, one leg SL) — trailing stops destroy this
- D_mom8's high R:R (3.63:1) benefits from letting winners run to full 5% TP — partial exits reduce edge
- ATR-adaptive SL adds no value because entry conditions already filter for specific vol regimes
- Only longer holds show marginal OOS improvement (+4.5bp), not worth implementing

**Action: Keep current baseline parameters. No exit mechanism changes needed.**
