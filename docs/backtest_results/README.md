# Backtest Results Archive

Each file in this folder records backtest/A/B test results tied to a specific git commit.

## Naming Convention

```
{commit_short}_{date}_{description}.md
```

Example: `5ec5007_20260219_3year_ab_test.md`

## Why This Exists

- Every code change may alter backtest results
- This archive lets you trace **which commit produced which numbers**
- Prevents "I remember the Sharpe was 5.6 but which version was that?" confusion

## Index

| Commit | Date | Description | Key Metrics |
|--------|------|-------------|-------------|
| `8eb95b2` | 2026-02-18 | v11.4 Adaptive Threshold (2.5x_f1) — 365d, 20 symbols | Sharpe 7.78, ΣRet +79,523% |
| `6618f55` | 2026-02-19 | 3-Year A/B Test (7 dimensions, 34 variants) — 1095d, 20 symbols | Sharpe 5.61, $2,095→$85,697 |
| `76d71af` | 2026-02-19 | Performance Assessment A/B/C (3 configs, 3yr portfolio) — 1095d, 20 symbols | **B wins**: Sharpe 6.07, Calmar 330.7, DD 10.1% |
| `f26299b` | 2026-02-19 | Bidirectional "Flip" Assessment (4 variants, 3yr portfolio) — 1095d, 20 symbols | **MTF Block wins**: more trades ≠ better; blocked signals are net losers |
| `6eb0212` | 2026-02-19 | Universe Expansion 20→30 symbols (4 variants, 3yr portfolio) — 1095d, 30 symbols | **B-30 wins**: Sharpe 7.09 (+19%), +58% return, same DD 9.7% |
| `dab1f80` | 2026-02-19 | Universe B-30/40/50 Sweep (3 sizes, 3yr portfolio) — 1095d, 50 symbols | **B-50 wins**: Sharpe 7.24, +6812%, DD 9.1%, Calmar 751 |
| `be25665` | 2026-02-19 | Filter Relaxation A/B (6 variants, 3yr portfolio) — 1095d, 50 symbols | **E-3Slot wins**: Sharpe 6.41, +8785%, DD 11.1%, 26.11 trades/day (+138%) |
| `5900fa5` | 2026-02-19 | Partial Exit + Trailing Stop A/B (6 variants, 3yr portfolio) — 1095d, 50 symbols | **E-BE15 wins**: R:R 0.98→1.51 (+54%), Sharpe 6.41→6.84, PF 1.76→1.96, DD 10.7% |
| `27d6339` | 2026-02-19 | Alpha Search R3+R4: Smart Exit + MDD Innovations (25 variants, 3yr portfolio) — 1095d, 50 symbols | **★ Y-Kitchen wins**: WR 67.1%, R:R 1.00, DD 7.2%, Sharpe 8.42, PF 2.07 — TRIPLE WIN vs v12.2 |
| `0690b0f` | 2026-02-20 | Alpha Search R5: Y-Kitchen Combos (19 variants, 3yr) — 1095d, 50 symbols | **G-YK+PT20**: WR 67.6%, R:R 1.80, DD 6.7%, Sharpe 8.17, PF 2.21 |
| `0690b0f` | 2026-02-20 | **R5 6yr Crash-Era Validation** (19 variants, 6yr) — 2190d, 50 symbols | **★ J-YK+2Slt wins**: DD 8.2%, Sharpe 5.33 — 3yr dominators (G,O) FAILED. Partial trailing amplifies crash DD. |
| `3fbf1d6` | 2026-02-20 | **R6 Combo Variants** (24 variants, 6yr) — 2190d, 50 symbols | **★★ M-2SSTSX wins**: DD 8.0%, Sharpe 5.50, PF 1.32 — 2slt+SoftTight+SteepX synergy dominates J. |
| `bdd85f0` | 2026-02-20 | **R7 3D Exit Grid Sweep** (101 variants, 6yr) — 2190d, 50 symbols | **★★★ T62_S79_PBT wins**: R:R 0.60→1.22 (+103%), DD 9.0%, PF 1.33 — Partial exit doubles R:R. |
| `3f63b4c` | 2026-02-20 | **R8 Stress Test** (5 configs × 4 tests, 6yr) — 2190d, 50 symbols | All PBT configs Grade A+. TierC-T100S38: R:R 1.66, BE margin +19.2pp, most regime-stable (96.9%). |
| `614a449` | 2026-02-21 | **Direction Asymmetry Analysis (6yr)** — 2190d, 50 symbols | Longs +25% EV/trade (3.64% vs 2.91%), R:R 2.10 vs 1.75. **But gap erodes: Yr5-6 shorts outperform longs.** |
| `614a449` | 2026-02-21 | **Direction Asymmetry A/B R1 (6yr)** — 16 variants, 2190d, 50 symbols | Best Sharpe: C-ShortTightT 5.76 (+3.6%). Best Calmar: D-SplitTarget 891.2. **Baseline wins on Return & PF.** |
| `614a449` | 2026-02-21 | **Direction Asymmetry R2 (6yr)** — 18 variants, 2190d, 50 symbols | Best Sharpe: FS-25 5.73 (+3.1%). H+E SUBADDITIVE. **Baseline beats all on Calmar (833.9) & Return (6,301%).** No production change. |
| `7f6bd77` | 2026-02-21 | **Adaptive Direction A/B — BTC 4h SMA Regime (6yr)** — 23 variants (single+dual SMA), 2190d, 50 symbols | Dual-confirm EV gap +0.463% (all 6yr positive). **P-DualStopTgt first to beat baseline on Return (6,358% vs 6,301%)** but DD 8.9% (+17%). Baseline wins DD, Calmar. No production change. |
| `a09a0da` | 2026-02-21 | **Alpha Search R10: Multi-Factor + VWMA S/R (6yr)** — 5-factor analysis + 17 variants, 2190d, 50 symbols | **VWMA(100) dist is #1 predictor (r=+0.136, monotonic, 2.1× EV spread).** K-VWTgW: Ret +1.0%, DD -0.1%, Calmar 852 (+2.2%). H-VWStW: Ret +4.0%, Sharpe 5.79. No production change. |
| `25b983d` | 2026-02-21 | **Alpha Search R11: VWMA + VPVR/POC R:R Maximization (6yr)** — 5-factor analysis + 22 variants, 2190d, 50 symbols | **★★★ U-RR+VWS breaks Calmar 1000** (1004.0, +20%). DD 6.4% (-16%), PF 1.66 (+6%). POC dist r=+0.069 (STRONG), VWMA-POC independent (r=0.033). Blend-wide pattern confirmed across POC/VA. No production change. |
| `0b7936d` | 2026-02-21 | **Alpha Search R12: Trailing Stop Innovation (6yr)** — 24 variants on U-RR+VWS baseline, 2190d, 50 symbols | **Runner is near-optimal:** 99.9% WR, 6.9-bar avg hold, 0 time exits. R:R ceiling +0.02. **B-T07 (ATR×0.7) best practical:** Calmar 1022 (+1.8%), Ret 6,528% (+1.6%), PF 1.67. Step-up/MH extension: zero effect. R:R bottleneck is stops, not runners. |
| `ca2ceba` | 2026-02-23 | **Trail Factor A/B R3 (U-RR+VWS baseline, 6yr)** — 18 variants, 2190d, 50 symbols, intra-bar SL | **★★★ C-TF10 (0.10): Ret 3,393%, DD 8.2%, Sharpe 4.47, Calmar 415.7.** All 17 variants beat baseline. M-TF15S30: DD 7.9% (best). Monotonic: lower trail_factor → better everything. |
| `1bce5d3` | 2026-02-23 | **Trail Factor A/B R4 (R12-parity fix, 6yr)** — 18 variants, 2190d, 50 symbols, close-based gain | **R4 fixes R3 exit parity.** A-Prod=6,427% (matches R12). **M-TF15S30 wins recency (+9.0%):** factor=0.15, stop=0.30. R:R 2.04, Calmar 985. |
| `cb1d9cb` | 2026-02-23 | **Adaptive Exit R13: Regime/Vol/Conf Conditioning (6yr)** — 18 variants, 2190d, 50 symbols | **K-ConfHold best raw:** Ret 6,561%, Sharpe 5.99, Calmar 1039, PF 1.69. **P-R4+RegTgt best R:R (2.05).** Q-R4+VolTgt recency winner (+9.7%) but Ret ↓11%. TP1% tradeoff: higher hit rate ≠ more profit. |
| `d80623f` | 2026-02-24 | **R21: Climax Volume Filter + ATR Stop Tuning (6yr)** — 15 variants, 2190d, 50 symbols | **★★★ E-Clmx×3.0+ATR2 → v12.6 production.** Calmar 728→**1171** (+60.8%), DD 7.7%→**5.8%** (−24.7%), WR 51%→**62.2%** (+11.1pp), Sharpe **6.37** (best). Ghost-SL eliminated. Stoch/MACD disabled; ATR stop is the single biggest driver. |
| `29eecab` | 2026-02-26 | **R22: Zero-Lookahead Alpha Diagnostic (6yr)** — 20 variants + 9 exit strategies + MFE/MAE, 2190d, 50 symbols | **❌ NO EDGE.** 4h MTF lookahead fix → PF=0.877 at **zero cost** (436K trades). 19/20 portfolio variants negative. Only O-WideTgt (2.5x tgt) PF=1.02 at zero cost. **Rubber band 15m mean-reversion signal has no theoretical alpha.** |
| `29eecab` | 2026-02-26 | **R23: New Alpha Search — 4 Dimensions (6yr)** — 27 variants (Timeframe/Momentum/Regime/Volume), 2190d, 50 symbols | **❌ 27/27 PF < 1.0 at zero cost.** Best: C2-Range-MR-0.2 (PF=0.905). Momentum worse than MR (0.817). Volume signals: WR ~50% = coin flip. Higher TF decreases PF. **OHLCV-based 15m crypto trading has no exploitable alpha.** |
| `06b99ce` | 2026-02-28 | **R31: Multi-TF Vol Straddle A/B (6yr)** — 9 variants (15m/1h/4h), dedup audit, 50 symbols | **G-ALL deduped:** N=1,602 (0.73/day, +46%), WR 47.2%, R:R 1.55, PF 1.382, Ret +51,992%, DD 25.2%, 6/6yr ✅. No lookahead. 12% overlap removed by per-symbol lock. |
| `daf75c9` | 2026-02-28 | **R32: Exit / Risk Management A/B (6yr)** — 21 variants (ATR stop, trailing, partial, SL/TP sweep), 50 symbols | **Baseline wins.** Fixed SL/TP is optimal. ATR/trailing stops destroy straddle symmetry. Partial close marginal. No production change. |
| `da590bb` | 2026-02-28 | **R33: Creative A/B (6yr)** — 17 variants (SMA filter, BTC regime, confluence, quality+, Elliott, VPVR), 50 symbols | **E2-SMA, C2-BTC, B3-Confluence, A3-Quality+ validated.** PF 1.735–1.943, DD 9.6–18.9%. Elliott/VPVR no edge. |
| `cd9c3bc` | 2026-02-28 | **R34: Multi-Condition Portfolio (6yr)** — 21 variants (16 combo + 3 threshold + 2 CW), 50 symbols | **CW-Lin +129bp EV×N (+101%).** CW-Tier +96.9bp EV×N (+51%). T3-Triple PF=1.943, DD=14.8%. All 4 filters are independent conviction layers. |
| `e359ba7` | 2026-02-28 | **R35: N-Boosting & Conviction Sizing (6yr)** — 18 variants (VR relax, dur relax, hold, SL, 4TF, CW), 50 symbols | **B1-Dur6 best risk-adjusted:** N+28%, EV -7.5bp, DD 24.9%. E1-4TF: N+17%, DD 22.5% (lowest). CW×N-boost: G4-CW-F4 EV×N +196% but DD 64.8%. No lookahead bias confirmed. |
