# R33: Creative Strategy A/B Test

| Field | Value |
|-------|-------|
| Commit | `da590bb` |
| Date | 2026-02-28 |
| Base | G-ALL deduped (15m+1h+4h) |
| Data | 50 symbols × 6yr × 15m (8.3M bars) |
| Execution | 4-core parallel, 285s |

## Objective

Test **structurally different** hypotheses beyond exit/SL/TP parameter sweeps (R32 already confirmed baseline SL/TP is near-optimal). R33 explores signal quality filters, cross-asset regime overlays, asymmetric straddle designs, entry timing, and anti-behavioral patterns.

## Variants (17)

| Code | Category | Description |
|------|----------|-------------|
| A0-Baseline | Reference | Current G-ALL deduped (SL=1%/1.5%, TP=5%) |
| A1-DeepSqueeze | Signal Quality | VR<0.35 for D, VR<0.45 for S (deeper squeeze) |
| A2-VolConfirm | Signal Quality | Require vol_pct ≥ 70 at entry |
| A3-LongSqueeze | Signal Quality | S_d8 → S_d16 (16-bar squeeze minimum) |
| B1-MomBias70 | Asymmetric | Straddle 70/30 size split favoring momentum |
| B2-AsymTP | Asymmetric | Straddle: mom-leg TP=7%, counter-leg TP=3% |
| B3-Confluence | Signal Quality | Only straddle when D_mom8 also signals |
| C1-BTC-Squeeze | Cross-Asset | Only trade when BTCUSDT VR<0.55 |
| C2-BTC-Neutral | Cross-Asset | Only trade when BTC |momentum_8|<1% |
| D1-AntiCrowd3 | Anti-Crowding | Skip if >3 coins signal on same bar |
| D2-AntiCrowd5 | Anti-Crowding | Skip if >5 coins signal on same bar |
| E1-Delay1bar | Entry Timing | Enter 1 bar after signal (confirmation) |
| E2-SMA-Prox | Entry Timing | Only straddle when price within 0.5% of SMA(20) |
| F1-AdaptHold | Adaptive | Hold time ∝ 1/VR (deeper squeeze → longer hold) |
| G1-Breakout | Inverse | VR>0.65 momentum directional (expansion trade) |
| H1-Cooldown5 | Behavioral | Skip 5 bars after a losing trade |
| H2-AntiMart | Behavioral | Anti-martingale: win→×1.5, loss→×0.5 |

## Phase 1: Full-Sample Results

### Combined D+S (deduped) — Ranked by EV

| # | Variant | N | PF | EV(bp) | t | WR% | DD% | Δ_vs_A0 |
|---|---------|---|----|--------|---|-----|-----|---------|
| 1 | C1-BTC-Squeeze | 535 | 1.532 | +53.2 | 4.40 | 49.5 | 20.2 | +9.8 |
| 2 | E2-SMA-Prox | 1391 | 1.469 | +49.9 | 6.50 | 46.9 | 22.9 | +6.5 |
| 3 | B3-Confluence | 854 | 1.529 | +48.7 | 5.29 | 41.0 | 18.9 | +5.3 |
| 4 | C2-BTC-Neutral | 1503 | 1.394 | +44.4 | 5.91 | 47.5 | 22.6 | +1.0 |
| 5 | **A0-Baseline** | **1602** | **1.382** | **+43.4** | **5.95** | **47.2** | **25.2** | **0.0** |
| 6 | F1-AdaptHold | 1602 | 1.379 | +43.3 | 5.91 | 47.3 | 25.5 | -0.1 |
| 7 | A3-LongSqueeze | 803 | 1.448 | +42.1 | 4.45 | 38.9 | 21.8 | -1.3 |
| 8 | D2-AntiCrowd5 | 1578 | 1.358 | +40.9 | 5.58 | 46.8 | 27.0 | -2.5 |
| 9 | H1-Cooldown5 | 1543 | 1.354 | +40.6 | 5.47 | 47.1 | 29.6 | -2.8 |
| 10 | H2-AntiMart | 1602 | 1.353 | +40.4 | 5.02 | 47.2 | 33.7 | -3.0 |
| 11 | E1-Delay1bar | 1619 | 1.340 | +39.6 | 5.40 | 46.6 | 30.9 | -3.8 |
| 12 | D1-AntiCrowd3 | 1529 | 1.335 | +38.5 | 5.17 | 46.4 | 26.3 | -4.9 |
| 13 | A2-VolConfirm | 167 | 1.279 | +37.3 | 1.53 | 49.7 | 22.4 | -6.1 |
| 14 | B2-AsymTP | 1610 | 1.336 | +36.8 | 4.92 | 48.8 | 45.9 | -6.6 |
| 15 | A1-DeepSqueeze | 104 | 1.294 | +28.7 | 1.13 | 44.2 | 12.2 | -14.7 |
| 16 | B1-MomBias70 | 1602 | 1.376 | +26.8 | 5.20 | 47.1 | 21.2 | -16.6 |
| 17 | G1-Breakout | 446624 | 0.905 | -8.5 | -25.62 | 17.1 | 100.0 | -51.9 |

### D_mom8 Highlights

| Variant | N | PF | EV(bp) | Note |
|---------|---|----|--------|------|
| C1-BTC-Squeeze | 188 | 2.186 | +69.8 | BTC compression → strong D filter |
| E1-Delay1bar | 514 | 1.678 | +49.0 | Delay helps D (unlike S) |
| A0-Baseline | 509 | 1.574 | +42.0 | Reference |
| G1-Breakout | 446624 | 0.905 | -8.5 | Expansion = no edge |

### S_d8 Highlights

| Variant | N | PF | EV(bp) | Note |
|---------|---|----|--------|------|
| B3-Confluence | 345 | 1.489 | +58.5 | D+S agreement → high quality |
| E2-SMA-Prox | 882 | 1.434 | +54.4 | Near SMA → mean-revert sweet spot |
| C2-BTC-Neutral | 1030 | 1.350 | +45.9 | Calm BTC → cleaner altcoin squeezes |
| A0-Baseline | 1093 | 1.333 | +44.0 | Reference |
| B1-MomBias70 | 1093 | 1.280 | +19.8 | Biasing straddle destroys symmetry |

## Phase 2: OOS Results (67/33 split)

### Combined D+S — Ranked by OOS EV

| # | Variant | IS_N | IS_PF | IS_EV | OOS_N | OOS_PF | OOS_EV | OOS_t | P≥1 | OOS_CI | Δ |
|---|---------|------|-------|-------|-------|--------|--------|-------|-----|--------|---|
| 1 | **B3-Confluence** | 499 | 1.324 | +30.2 | **357** | **1.801** | **+73.0** | **4.89** | 100% | 1.42-2.28 | **+8.9** |
| 2 | **E2-SMA-Prox** | 752 | 1.280 | +30.2 | **641** | **1.689** | **+72.1** | **6.18** | 100% | 1.42-1.97 | **+8.0** |
| 3 | **A3-LongSqueeze** | 448 | 1.200 | +19.0 | **358** | **1.759** | **+70.4** | **4.70** | 100% | 1.39-2.22 | **+6.3** |
| 4 | **C2-BTC-Neutral** | 830 | 1.192 | +22.5 | **687** | **1.638** | **+68.9** | **6.05** | 100% | 1.38-1.92 | **+4.8** |
| 5 | A0-Baseline | 900 | 1.230 | +26.7 | 702 | 1.582 | +64.1 | 5.69 | 100% | 1.36-1.86 | 0.0 |
| 6 | F1-AdaptHold | 900 | 1.242 | +28.2 | 702 | 1.558 | +62.0 | 5.49 | 100% | 1.34-1.83 | -2.1 |
| 7 | D2-AntiCrowd5 | 892 | 1.216 | +25.1 | 685 | 1.552 | +61.4 | 5.38 | 100% | 1.31-1.82 | -2.7 |
| 8 | C1-BTC-Squeeze | 263 | 1.392 | +37.9 | 271 | 1.502 | +54.6 | 3.07 | 100% | 1.17-1.95 | -9.5 |

### OOS S_d8 Detail

| Variant | OOS_N | OOS_PF | OOS_EV | OOS_t | P≥1 |
|---------|-------|--------|--------|-------|-----|
| E2-SMA-Prox | 440 | 1.537 | +66.1 | 4.46 | 100% |
| C2-BTC-Neutral | 488 | 1.478 | +60.2 | 4.26 | 100% |
| B3-Confluence | 156 | 1.457 | +57.3 | 2.31 | 99% |
| A0-Baseline | 501 | 1.433 | +55.7 | 3.98 | 100% |
| A3-LongSqueeze | 157 | 1.399 | +51.5 | 2.07 | 97% |

### OOS D_mom8 Detail

| Variant | OOS_N | OOS_PF | OOS_EV | OOS_t |
|---------|-------|--------|--------|-------|
| C1-BTC-Squeeze | 87 | 2.974 | +105.8 | 3.99 |
| C2-BTC-Neutral | 199 | 2.418 | +90.2 | 4.89 |
| A0-Baseline | 201 | 2.320 | +85.2 | 4.67 |

## Key Findings

### 1. Signal Confluence (B3) — Best OOS EV: +73.0bp (+8.9bp)
Only entering straddle when D_mom8 signal also fires creates a **high-conviction filter**. When both VR<0.45 (directional threshold) AND VR<0.55+dur≥8 (squeeze threshold) are satisfied simultaneously, the setup is structurally stronger. Tradeoff: reduces N from 702 to 357 (51% fewer trades).

**Structural hypothesis**: When compression is deep enough to trigger BOTH signal types, the subsequent expansion is more violent and predictable.

### 2. SMA Proximity (E2) — Best Risk-Adjusted: t=6.18, +72.1bp (+8.0bp)
Requiring straddle entry only when price is within 0.5% of SMA(20) improves quality while retaining 91% of trades (641 vs 702). **Highest t-statistic of all variants** (6.18 vs 5.69 baseline).

**Structural hypothesis**: Price near SMA = true mean-reversion range. When a squeeze happens near the mean, the breakout is more likely to be genuine rather than continuation of a trend.

### 3. Long Squeeze (A3) — Lowest DD: 12.1%, +70.4bp (+6.3bp)
Requiring 16 bars of squeeze duration (vs 8) roughly halves trade count but dramatically reduces drawdown (12.1% vs 20.6% baseline). PF=1.759 is excellent.

**Structural hypothesis**: Longer compression = more stored energy = higher probability of meaningful expansion.

### 4. BTC Neutral (C2) — Best Combined: t=6.05, N=687, +68.9bp (+4.8bp)
Trading altcoin squeezes only when BTC is range-bound (|momentum_8| < 1%) retains most trades (687 vs 702) while adding +4.8bp. This is the **most practical filter** — high N, high t, marginal improvement.

**Structural hypothesis**: When BTC is directional, altcoin movements are dominated by BTC correlation. Squeeze setups work best when BTC isn't overwhelming alt-specific compression signals.

### 5. Breakout (G1) — Complete Null: PF=0.905, N=446K, -100%
Trading VR>0.65 expansion generates **446,624 trades** (vs 1,602 baseline) with negative expectancy. This is the strongest possible confirmation that the compression/squeeze hypothesis IS the edge. Expansion = noise.

### 6. Momentum Bias (B1) Destroys Straddle Edge
Biasing straddle sizes 70/30 toward momentum direction reduces combined EV from +43.4bp to +26.8bp. The straddle's edge IS its symmetry — it profits from expansion regardless of direction at equal magnitude.

### 7. Anti-Crowding Has No Value (D1, D2)
Anti-crowding filters reduce trade count and EV. Multiple coins signaling simultaneously is NOT a negative sign — it indicates genuine market-wide compression, which is the thesis.

### 8. Entry Delay Hurts Overall (E1)
Delay by 1 bar reduces combined EV by -3.8bp. The signal's edge is strongest at moment of generation. However, for D_mom8 specifically, delay IMPROVED EV (+49.0bp vs +42.0bp) — possibly because directional trades benefit from momentum confirmation.

## Recommendations

### Immediate (Low Risk)
- **C2-BTC-Neutral**: Add BTC momentum filter as optional overlay. Minimal N reduction (687 vs 702), +4.8bp OOS improvement, highest t-stat with high N (6.05). Easy to implement in live system.

### Consider (Medium Risk)
- **E2-SMA-Prox**: SMA(20) proximity filter for straddle. +8.0bp OOS, t=6.18, N=641. Requires SMA computation but structurally sound.
- **A3-LongSqueeze**: Increase minimum squeeze duration to 16 for S_d8. +6.3bp, lowest DD (12.1%), but halves N.

### Monitor (Higher Risk)
- **B3-Confluence**: Best absolute OOS EV (+73bp) but only 357 trades. Need more OOS data to confirm statistical robustness at this N level.

### Reject
- B1-MomBias70: Destroys straddle symmetry
- B2-AsymTP: Higher DD, lower EV
- G1-Breakout: Complete null result
- D1/D2-AntiCrowd: No value
- H1-Cooldown5, H2-AntiMart: Behavioral filters add no value

## Action Items
1. Evaluate C2-BTC-Neutral + E2-SMA-Prox stacking in R34
2. Consider A3 (d≥16) as a conservative "quality mode" toggle
3. Keep baseline unchanged for now — improvements are marginal and need live validation
