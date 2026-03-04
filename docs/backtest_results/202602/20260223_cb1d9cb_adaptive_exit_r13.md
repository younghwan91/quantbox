# R13: Adaptive Exit Parameters — Regime / Vol / Conf Conditioning

- **Commit:** `cb1d9cb`
- **Date:** 2026-02-23 14:22 UTC
- **Branch:** `feature/adaptive-exit-r13`
- **Script:** `research/active/r12_r20/r13_adaptive_exit.py`
- **Data:** 2190d (6yr), 50 symbols, 15m+4h

## Hypothesis

Fixed exit params (target=1.5, hold=25, trail_factor=0.3) are suboptimal because
market conditions at entry time vary. R4 showed TP1 hit rate is only **17%** — 83%
of trades never reach target. Exits should adapt to:

- **Regime** (BTC bull/choppy/bear) → trending = wider TP + longer hold
- **Volatility** (ATR% at entry) → high vol = wider TP/stop, low vol = tighter
- **Confidence** (3-6) → high conf = wider TP + longer hold

Per-position effective params are computed at entry time and stored in the position
dict, then used at exit time instead of fixed config values.

## Baseline Validation

| Metric | R4 Baseline | R13 A-Prod |
|--------|-------------|------------|
| Return | 6,427% | 6,427.3% |
| Max DD | 6.4% | 6.4% |
| Sharpe | 5.77 | 5.77 |
| Calmar | 1004.0 | 1004.0 |
| WR | 58.2% | 58.2% |
| TP1% | 17% | 17% |

✅ **Parity confirmed** — identical to R4 and R12.

## Results Summary (18 variants)

| Var | Description | Ret% | DD% | Sharpe | Calmar | WR | PF | R:R | TP1% | #Tr |
|-----|-------------|------|-----|--------|--------|----|----|-----|------|-----|
| A-Prod | Production fixed (baseline) | 6,427% | 6.4% | 5.77 | 1004.0 | 58.2% | 1.66 | 1.76 | 17% | 16,584 |
| B-RegTgt | Regime→target (B1.8/C1.5/R1.2) | 6,390% | 6.4% | 5.75 | 998.1 | 58.1% | 1.67 | 1.76 | 17% | 16,562 |
| C-RegHold | Regime→hold (B30/C25/R18) | 6,446% | 6.4% | 5.76 | 1006.9 | 58.1% | 1.67 | 1.76 | 17% | 16,599 |
| **D-RegTrail** | Regime→trail (B0.15/C0.30/R0.40) | **6,482%** | 6.4% | 5.77 | **1012.4** | 57.7% | 1.67 | **1.80** | 17% | 16,556 |
| E-RegAll | Regime→target+hold+trail | 6,369% | 6.4% | 5.79 | 994.8 | 57.6% | 1.67 | 1.78 | 17% | 16,455 |
| F-VolTgt | ATR%→target (ref=1.5%) | 5,433% | **6.0%** | 5.79 | 906.8 | **64.3%** | 1.60 | 1.43 | **30%** | 19,729 |
| G-VolHold | ATR%→hold (ref=1.5%) | 6,121% | 6.6% | **5.94** | 925.1 | 57.8% | 1.65 | 1.62 | 12% | 17,220 |
| H-VolStop | ATR%→stop (ref=1.5%) | 6,136% | 6.3% | 5.64 | 966.3 | 54.7% | 1.64 | **2.00** | 16% | 17,430 |
| I-VolAll | ATR%→target+hold+stop | 5,255% | 6.3% | 5.61 | 839.9 | 60.5% | 1.59 | 1.58 | 24% | 20,561 |
| J-ConfTgt | Conf→target (3→1.0, 6→2.0) | 5,937% | 6.7% | 5.81 | 881.2 | 62.9% | 1.63 | 1.58 | 26% | 18,684 |
| **K-ConfHold** | Conf→hold (3→18, 6→32) | **6,561%** | **6.3%** | **5.99** | **1039.3** | 58.4% | **1.69** | 1.66 | 14% | 17,086 |
| L-ConfAll | Conf→target+hold | 6,070% | 6.7% | 5.98 | 903.7 | 62.6% | 1.66 | 1.56 | 23% | 18,896 |
| M-RegVol | Regime tgt + Vol hold | 6,119% | 6.6% | 5.96 | 924.8 | 57.8% | 1.65 | 1.61 | 12% | 17,226 |
| N-RegConf | Regime tgt + Conf hold | 6,468% | 6.3% | 5.96 | 1024.6 | 58.3% | 1.68 | 1.65 | 14% | 17,070 |
| O-TriAdapt | Regime tgt + Vol stop + Conf hold | 6,300% | **6.2%** | 5.87 | 1010.5 | 54.8% | 1.66 | 1.89 | 13% | 17,935 |
| **P-R4+RegTgt** | TF15+S30 + Regime target | **6,523%** | 6.4% | 5.82 | 1020.3 | 54.4% | 1.65 | **2.05** | 17% | 17,060 |
| Q-R4+VolTgt | TF15+S30 + Vol target | 5,705% | 6.4% | 5.76 | 893.3 | 59.9% | 1.60 | 1.74 | **30%** | 20,273 |
| R-R4+RegAll | TF15+S30 + Regime all | 6,366% | 6.4% | 5.75 | 988.0 | 55.4% | 1.64 | 1.93 | 16% | 17,008 |

## Recency-Weighted Rankings

Weights: Y1-4 = 1×, Y5-6 = 2× | Sharpe 30%, PF 25%, R:R 20%, inv(DD) 15%, TP1% 10%

| Rank | Variant | Score | vs Base | Notes |
|------|---------|-------|---------|-------|
| 1 | Q-R4+VolTgt | 1.097 | +9.7% | ★ Recency winner (TP1% doubled to 30%, but Ret ↓11%) |
| 2 | P-R4+RegTgt | 1.075 | +7.5% | Best R:R (2.05), Ret 6,523% |
| 3 | J-ConfTgt | 1.067 | +6.7% | TP1% 26%, WR 62.9%, but R:R ↓ |
| 4 | H-VolStop | 1.065 | +6.5% | R:R 2.00, consistent across years |
| 5 | D-RegTrail | 1.064 | +6.4% | Simple & effective, Ret 6,482%, Calmar 1012 |
| 6 | F-VolTgt | 1.061 | +6.1% | TP1% 30% but Ret ↓15% |
| 7 | A-Prod | 1.059 | +5.9% | Baseline |
| ... | ... | ... | ... | |
| 17 | G-VolHold | 0.991 | -0.9% | Below base |
| 18 | M-RegVol | 0.988 | -1.2% | Below base |

## Key Winners by Category

| Category | Winner | Value |
|----------|--------|-------|
| 🏆 Best Return | **K-ConfHold** | 6,560.6% |
| 🏆 Best Sharpe | K-ConfHold | 5.99 |
| 🏆 Best Calmar | K-ConfHold | 1039.3 |
| 🏆 Best R:R | **P-R4+RegTgt** | 2.05 |
| 🏆 Best DD | F-VolTgt / O-TriAdapt | 6.0% / 6.2% |
| 🏆 Best PF | K-ConfHold | 1.69 |
| 🏆 Best TP1% | Q-R4+VolTgt | 30.5% |
| 🏆 Recency | Q-R4+VolTgt | 1.097 (+9.7%) |

## Analysis

### What Worked

1. **K-ConfHold (Conf→hold)** is the best raw performer across all core metrics:
   - Ret 6,561% (+2.1% vs base), Sharpe 5.99, Calmar 1039, PF 1.69
   - Logic: low confidence (3) → short hold (18 bars), high confidence (6) → long hold (32 bars)
   - "When uncertain, exit faster" — this is intuitively correct and empirically validated

2. **P-R4+RegTgt (TF15+S30 + Regime target)** achieves the best R:R (2.05):
   - Combines R4's tighter stop/trail (TF15+S30) with regime-adaptive target
   - Bull=1.8, Choppy=1.5, Bear=1.2 target retrace
   - Ret 6,523% (+1.5% vs base), Calmar 1020

3. **D-RegTrail** is the simplest improvement:
   - Just varies trail factor by regime (bull=0.15, choppy=0.30, bear=0.40)
   - Ret 6,482%, R:R 1.80, Calmar 1012 — all slightly better than base
   - Minimal complexity → high confidence in out-of-sample robustness

### What Didn't Work

1. **Vol-target scaling** (F-VolTgt, Q-R4+VolTgt) boosts TP1% dramatically (17%→30%)
   but at the cost of total return (-11% to -15%). Tighter targets in low-vol capture
   small profits, but profits per trade shrink more than the hit rate increase compensates.

2. **Vol-hold scaling** (G-VolHold, M-RegVol) performed below base. Vol-based hold
   durations create instability — sometimes holding too long in low vol (nothing happening)
   or cutting too short in high vol (missing the move).

### TP1% vs Total Return Tradeoff

| Variant | TP1% | Return | R:R | Trade-off |
|---------|------|--------|-----|-----------|
| A-Prod | 17% | 6,427% | 1.76 | Baseline |
| K-ConfHold | 14% | 6,561% | 1.66 | Less TP1 hits but bigger total P&L |
| P-R4+RegTgt | 17% | 6,523% | 2.05 | Same TP1%, much better R:R |
| Q-R4+VolTgt | 30% | 5,705% | 1.74 | Double TP1%, but -11% total return |
| D-RegTrail | 17% | 6,482% | 1.80 | Simple improvement, no TP1 change |

Higher TP1% does NOT equal higher returns. Per-trade profit matters more than hit rate.

## Recommendation

**Tier 1 — Implement candidates (further testing needed):**
- **K-ConfHold**: Best all-around. Confidence-adaptive hold is a low-complexity, high-impact change.
- **P-R4+RegTgt**: Best R:R. Layer regime-adaptive target on R4's tighter trail/stop.

**Tier 2 — Promising but needs R14 validation:**
- **D-RegTrail**: Simplest adaptive change. Could combine with K-ConfHold.
- **O-TriAdapt**: Best DD (6.2%), R:R 1.89, but complex (3 adaptive dimensions).

**Tier 3 — Reject:**
- Vol-scaled target/hold variants (F, G, I, M, Q): Return degradation not justified.
- Kitchen-sink combos (E-RegAll, R-R4+RegAll): Combining everything dilutes the gains.

## Next Steps

R14 should test **K-ConfHold + D-RegTrail combo** as the primary candidate:
- Confidence-adaptive hold (3→18, 4→22, 5→28, 6→32)
- Regime-adaptive trail (bull=0.15, choppy=0.30, bear=0.40)
- Optionally layer P-R4+RegTgt's regime-adaptive target

This is the "smart exit" stack: adapt WHAT to exit (hold by confidence), HOW to trail
(by regime), and WHERE to target (by regime).
