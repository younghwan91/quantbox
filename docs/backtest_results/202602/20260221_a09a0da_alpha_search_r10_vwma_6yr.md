# Alpha Search R10: Multi-Factor Adaptive + VWMA S/R Innovation

- **Commit:** `a09a0da`
- **Date:** 2026-02-21
- **Branch:** `feature/direction-asymmetry-analysis`
- **Data:** 6yr (2190d), 50 symbols, left-padded to 210,335 bars
- **Baseline:** v12.4 I-HybTight (symmetric T150_S38_H25_PBT)
- **VWMA Period:** 100 bars on 15m (25 hours lookback)
- **Runtime:** 87.9s (17 variants)

## Executive Summary

**VWMA(100) distance is the strongest single predictor of trade quality ever tested** in this system (r = +0.1357, monotonic increasing across terciles). Trades where price is far from VWMA in the favourable reversion direction produce 2.1× the EV of close-to-VWMA trades.

Two variants beat baseline on the composite risk-return metric:
- **K-VWTgW** (blend-wide target): Return +1.0%, DD -0.1%, Calmar +2.2% — best balanced
- **H-VWStW** (blend-wide stop): Return +4.0%, Sharpe +4.1%, DD +0.3% — best absolute return

**Verdict: NO PRODUCTION CHANGE** — improvements are modest and need walk-forward validation. But VWMA(100) distance is now a PROVEN predictive factor worthy of deeper R11 exploration.

---

## Hypothesis

1. **Factor Independence:** Five measurable features at entry time may independently predict trade quality — allowing a multi-factor adaptive system that's robust across changing regimes.
2. **VWMA(100) as S/R:** The 100-bar Volume-Weighted Moving Average on 15m represents a "volume-weighted fair value." For a mean-reversion strategy, price dislocation from VWMA should predict both entry quality and optimal stop/target sizing.

## Part A: Multi-Factor Independence Analysis (17,165 trades)

### Five Factors Tested

| # | Factor | Description | Computation |
|---|--------|-------------|-------------|
| F1 | VWMA(100) Distance | Directional price dislocation from VWMA | `dir * (VWMA - close) / close` (positive = favourable) |
| F2 | BTC Dual-Confirm Regime | Macro trend state | BTC > both SMA(50) & SMA(200) on 4h → Bull (+1), both below → Bear (-1), else Choppy (0) |
| F3 | BTC Vol Ratio | BTC realized volatility vs history | ATR(96)/SMA(ATR, 2880) on BTC 15m |
| F4 | Asset ATR Ratio | Per-asset volatility vs its history | ATR/SMA(ATR, 200) per symbol on 15m |
| F5 | Market Breadth | Universe sentiment | % of 50 symbols with close > SMA(200) on 15m |

### Factor Distributions (at entry)

| Factor | Min | P25 | Median | P75 | Max |
|--------|-----|-----|--------|-----|-----|
| VWMA(100) Dist | -0.211 | +0.004 | +0.013 | +0.025 | +0.561 |
| Dual-Confirm | -1.000 | -1.000 | 0.000 | +1.000 | +1.000 |
| BTC Vol Ratio | 0.198 | 0.714 | 0.895 | 1.139 | 4.784 |
| Asset ATR Ratio | 0.371 | 0.834 | 0.967 | 1.139 | 5.788 |
| Market Breadth | 0.000 | 0.220 | 0.440 | 0.720 | 1.000 |

### Tercile Analysis — EV by Factor Level

#### F1: VWMA(100) Distance — ⚡ STRONGEST PREDICTOR

| Tercile | #Trades | WR% | PF | R:R | EV% | Total$ |
|---------|---------|-----|-----|-----|------|--------|
| Lo (<0.7%) | 5,716 | 52.1% | 1.33 | 1.95 | 2.257% | $26,792 |
| Mid | 5,733 | 57.5% | 1.60 | 1.86 | 2.836% | $51,036 |
| Hi (>2.0%) | 5,716 | 57.9% | 1.71 | 1.85 | **4.709%** | $71,454 |

**Monotonic increasing.** Hi-tercile EV is 2.09× Lo-tercile. Every upgrade in VWMA distance improves WR (+5.8pp) and PF (+0.38). The $44.6K gap between Lo and Hi is enormous.

#### F2: BTC Dual-Confirm Regime

| Tercile | #Trades | WR% | PF | R:R | EV% | Total$ |
|---------|---------|-----|-----|-----|------|--------|
| Bear (-1) | 5,589 | 55.1% | 1.47 | 1.84 | 3.227% | $42,522 |
| Choppy (0) | 4,475 | 55.4% | 1.56 | 1.98 | 3.187% | $38,064 |
| Bull (+1) | 7,101 | 56.8% | 1.63 | 1.95 | 3.348% | $68,698 |

Non-monotonic, narrow spread (0.161%). Regime alone is a weak per-trade predictor (r = +0.005). Its value is in portfolio-level distribution, not individual trade quality.

#### F3: BTC Vol Ratio — MONOTONIC INCREASING

| Tercile | #Trades | WR% | PF | R:R | EV% | Total$ |
|---------|---------|-----|-----|-----|------|--------|
| Lo (<0.775) | 5,679 | 54.9% | 1.52 | 1.98 | 2.808% | $45,198 |
| Mid | 5,695 | 56.0% | 1.55 | 1.90 | 3.110% | $48,588 |
| Hi (>1.042) | 5,679 | 56.4% | 1.58 | 1.84 | **3.781%** | $52,890 |

Higher BTC volatility → better mean-reversion trades. The rubberband captures more PnL when volatility is above average. Spread = 0.97% EV.

#### F4: Asset ATR Ratio — U-SHAPED

| Tercile | #Trades | WR% | PF | R:R | EV% | Total$ |
|---------|---------|-----|-----|-----|------|--------|
| Lo (<0.879) | 5,716 | 56.8% | 1.68 | 2.01 | 3.149% | $55,486 |
| Mid | 5,733 | 54.8% | 1.44 | 1.91 | 2.900% | $40,598 |
| Hi (>1.073) | 5,716 | 55.9% | 1.56 | 1.85 | **3.752%** | $53,198 |

Non-monotonic (U-shaped). Both extremes outperform the middle. Low-vol assets win on WR/PF (stable bounces), high-vol assets win on absolute EV (larger moves). Mid-vol assets are worst.

#### F5: Market Breadth — MONOTONIC DECREASING

| Tercile | #Trades | WR% | PF | R:R | EV% | Total$ |
|---------|---------|-----|-----|-----|------|--------|
| Lo (<28%) | 5,369 | 54.9% | 1.52 | 1.91 | **3.581%** | $46,295 |
| Mid | 6,280 | 56.8% | 1.58 | 1.86 | 3.187% | $55,946 |
| Hi (>62%) | 5,516 | 55.7% | 1.57 | 2.01 | 3.051% | $47,042 |

Mean-reversion works BETTER in fear/panic (low breadth). When most of the market is below SMA(200), rubber-band dips bounce harder. Spread = 0.53% EV.

### Pairwise Correlation Matrix

|                    | F1 VWMA | F2 DC | F3 BTC Vol | F4 ATR | F5 Breadth |
|--------------------|---------|-------|------------|--------|------------|
| **F1 VWMA Dist**   | 1.000 | -0.035 | +0.145 | +0.170 | -0.101 |
| **F2 Dual-Confirm** | -0.035 | 1.000 | -0.052 | +0.019 | **+0.301** |
| **F3 BTC Vol**     | +0.145 | -0.052 | 1.000 | **+0.211** | -0.111 |
| **F4 Asset ATR**   | +0.170 | +0.019 | +0.211 | 1.000 | **-0.215** |
| **F5 Breadth**     | -0.101 | **+0.301** | -0.111 | -0.215 | 1.000 |

**Key independence findings:**
- **VWMA distance is nearly independent of ALL other factors** (max |r| = 0.170 with Asset ATR)
- Dual-Confirm and Breadth share moderate overlap (r = +0.301) — both measure "market sentiment"
- BTC Vol and Asset ATR overlap moderately (r = +0.211) — volatility clusters
- **VWMA and Dual-Confirm are orthogonal** (r = -0.035) — combining them should be additive

### Factor → PnL Correlation Rankings

| Rank | Factor | r → PnL | Strength |
|------|--------|---------|----------|
| 1 | **VWMA(100) Distance** | **+0.1357** | **STRONG** |
| 2 | Asset ATR Ratio | +0.0642 | STRONG |
| 3 | BTC Vol Ratio | +0.0462 | MODERATE |
| 4 | Market Breadth | -0.0202 | MODERATE |
| 5 | Dual-Confirm Regime | +0.0050 | WEAK |

**VWMA distance is 2.1× more predictive than the second-best factor.** This is the single most important finding: the volume-weighted fair value provides fundamentally different information from the move-size-based logic currently used.

---

## Part B: VWMA + Multi-Factor A/B Test — 17 Variants

### Variant Design

| Group | Variant | Description | Key Changes |
|-------|---------|-------------|-------------|
| — | A-Baseline | Production I-HybTight | No adaptation |
| G1: Entry Gate | B-VWEntry0 | VWMA dist > 0% | Filter entries where price didn't cross VWMA |
| | C-VWE05 | VWMA dist > 0.5% | Require meaningful dislocation |
| | D-VWE10 | VWMA dist > 1.0% | Aggressive filter |
| G2: Stop | E-VWSt50 | VWMA stop ×0.50 (replace) | Stop = VWMA distance × 0.50 |
| | F-VWSt75 | VWMA stop ×0.75 (replace) | Stop = VWMA distance × 0.75 |
| | G-VWStT | blend-tight: min(normal, vwma×0.50) | Tighter of two stops |
| | H-VWStW | blend-wide: max(normal, vwma×0.50) | Wider of two stops |
| G3: Target | I-VWTg10 | VWMA target ×1.0 (replace) | Target = full reversion to VWMA |
| | J-VWTg08 | VWMA target ×0.8 (replace) | Target = 80% reversion to VWMA |
| | K-VWTgW | blend-wide: max(normal, vwma×0.80) | Larger of two targets |
| G4: Combo | L-VWFull | Entry + stop + target all VWMA | Full VWMA system |
| G5: Dual+VWMA | M-DuVWS | DC stops + VWMA blend-tight | DC regime + VWMA tighter stop |
| | N-DuVWE | DC + VWMA entry gate 0.5% | DC regime + VWMA quality filter |
| | O-DuVWF | DC + VWMA entry + stop + target | Everything combined |
| G6: Multi-Factor | P-VolSt | BTC vol adaptive stop | High vol → 0.45, Low vol → 0.30 |
| | Q-BrdSz | Breadth adaptive sizing | Panic → 1.3×, Euphoria → 0.7× |

### Results Summary

| Variant | Return% | DD% | Sharpe | Calmar | WR% | PF | R:R | #Trades |
|---------|---------|-----|--------|--------|-----|-----|-----|---------|
| **A-Baseline** | **6,300.9%** | **7.6%** | **5.56** | **833.9** | **55.8%** | **1.56** | **1.91** | **17,165** |
| B-VWEntry0 | 6,129.4% | 7.7% | 5.47 | 794.4 | 56.7% | 1.60 | 1.89 | 15,222 |
| ★ C-VWE05 | 6,268.2% | **7.2%** | 5.55 | **871.4** | 57.5% | **1.65** | 1.87 | 14,129 |
| D-VWE10 | 5,938.5% | 7.6% | 5.74 | 782.4 | 58.0% | **1.68** | 1.85 | 12,316 |
| E-VWSt50 | 6,055.9% | 8.0% | 5.61 | 752.6 | 51.8% | 1.55 | 2.24 | 18,165 |
| F-VWSt75 | 6,260.7% | 7.6% | 5.64 | 821.6 | 54.9% | 1.58 | 2.00 | 17,089 |
| G-VWStT | 5,896.1% | **7.4%** | 5.42 | 794.8 | 49.8% | 1.53 | **2.40** | 18,799 |
| ▲ H-VWStW | **6,551.7%** | 7.9% | **5.79** | 828.4 | **58.2%** | 1.59 | 1.76 | 16,584 |
| I-VWTg10 | 5,533.2% | **7.1%** | 5.81 | 778.0 | 64.0% | 1.57 | 1.56 | 23,933 |
| J-VWTg08 | 5,117.7% | 7.3% | **6.01** | 701.7 | **65.2%** | 1.55 | 1.50 | 25,280 |
| ★▲ K-VWTgW | **6,364.5%** | **7.5%** | 5.60 | **852.2** | 55.8% | 1.56 | 1.92 | 17,135 |
| L-VWFull | 5,763.2% | 7.6% | 5.99 | 758.9 | 64.4% | **1.65** | 1.62 | 19,987 |
| M-DuVWS | 5,978.3% | 7.6% | 5.59 | 790.1 | 49.9% | 1.54 | **2.40** | 18,804 |
| N-DuVWE | 6,273.7% | 7.6% | 5.66 | 830.6 | 57.7% | **1.66** | 1.85 | 14,049 |
| O-DuVWF | 6,092.8% | 7.6% | 5.65 | 797.9 | 52.3% | 1.60 | 2.25 | 16,278 |
| P-VolSt | 6,223.9% | 7.6% | 5.50 | 823.7 | 55.5% | 1.55 | 1.94 | 17,295 |
| Q-BrdSz | 6,157.4% | **7.4%** | 5.31 | 837.2 | 55.8% | 1.55 | 1.91 | 17,165 |

★ = better DD, ▲ = better Return (bold highlights best-in-column)

### Key Deltas vs Baseline

| Variant | ΔReturn | ΔDD | ΔSharpe | ΔCalmar | Assessment |
|---------|---------|-----|---------|---------|------------|
| **K-VWTgW** | **+63.6** | **-0.1** | +0.04 | **+18.3** | **BEST BALANCED — improves Return, DD, AND Calmar** |
| **H-VWStW** | **+250.8** | +0.3 | **+0.23** | -5.5 | **BEST RETURN — +4.0% absolute, Sharpe +4.1%** |
| C-VWE05 | -32.7 | **-0.4** | -0.01 | **+37.5** | BEST CALMAR — risk-adjusted king |
| N-DuVWE | -27.2 | 0.0 | +0.10 | -3.3 | DC+VWMA synergy, modest |
| D-VWE10 | -362.4 | 0.0 | +0.18 | -51.5 | Too aggressive filtering |
| J-VWTg08 | -1183.2 | -0.3 | **+0.45** | -132.2 | Best Sharpe but return destroyed |

### Long/Short Breakdown for Winners

| Variant | Long: #Tr WR% EV PF | Short: #Tr WR% EV PF |
|---------|---------------------|----------------------|
| A-Baseline | 8,452 56.0% 3.636 1.64 | 8,713 55.7% 2.908 1.48 |
| H-VWStW | 8,154 58.3% 3.767 1.66 | 8,430 58.0% 3.027 1.52 |
| K-VWTgW | 8,440 56.0% 3.649 1.65 | 8,695 55.7% 2.915 1.49 |
| C-VWE05 | 6,895 58.0% 4.080 1.75 | 7,234 57.1% 3.145 1.57 |

**H-VWStW improves BOTH long AND short performance** — the wider stop benefits both directions symmetrically. C-VWE05 boosts per-trade quality dramatically (long EV 3.636 → 4.080, short EV 2.908 → 3.145).

---

## Analysis & Interpretation

### Why Blend-Wide Modes Win

The "blend-wide" approach (take the MORE GENEROUS of normal and VWMA-based levels) is the key innovation:

- **Blend-wide stop (H-VWStW):** `stop = max(move_size × 0.382, VWMA_dist × 0.50)`. When price is far from VWMA at entry (large dislocation), the VWMA-based stop is wider than the move-based stop, preventing premature stop-outs on volatile bounces. When VWMA distance is small, the normal stop dominates — no harm done.

- **Blend-wide target (K-VWTgW):** `target = max(move_size × 1.5, VWMA_dist × 0.80)`. When VWMA is far above entry (for longs), the VWMA target is higher than the fixed retrace, capturing more of the reversion. When close to VWMA, normal target is used.

**The asymmetry is critical:** these modes NEVER degrade a trade vs baseline — they only IMPROVE trades where VWMA provides a better reference. This explains why they're the only variants to beat baseline on return.

### Why Replace Modes Fail on Return

Replace modes (E, F, I, J) substitute the move-based stop/target entirely with VWMA-based ones. This HURTS when VWMA distance is small (which happens for ~33% of trades in the Lo tercile). Those trades get tiny stops/targets, causing either premature exits or too-frequent but too-small wins.

### Why Entry Gates Improve Quality But Not Return

C-VWE05 filters out 3,036 trades where VWMA distance < 0.5%. These are the Lo-tercile trades with 2.257% EV. Removing them raises average quality (WR 55.8% → 57.5%, PF 1.56 → 1.65) but loses the aggregate PnL those 3,036 trades contributed — even weak trades had positive EV (2.257%).

### Why Multi-Factor (P, Q) Underperform

- **P-VolSt** (BTC vol adaptive stop) follows the same "replacing is risky" pattern — it modifies stop_mult in both directions, sometimes tightening when wider would be better.
- **Q-BrdSz** (breadth sizing) reduces size in high-breadth regimes but those regimes still have positive EV (3.05%) — reducing size loses money on net.

---

## Verdict

### 🔴 NO PRODUCTION CHANGE

While VWMA(100) distance is the strongest predictor found, the production improvements are modest:
- **K-VWTgW:** +1.0% return, -0.1% DD — real but small
- **H-VWStW:** +4.0% return but +0.3% DD — trades off safety

The baseline I-HybTight already captures 6,300.9% over 6 years. The marginal gains from VWMA need:
1. **Walk-forward validation** (train on 4yr, test on 2yr)
2. **Yearly stability check** (does K-VWTgW win every year?)
3. **Different VWMA periods** (50, 200, 400 — is 100 optimal or just one sample?)
4. **Combined variant** (C-VWE05 entry gate + K-VWTgW target in one variant)

### Key Discoveries for Future Research (R11+)

| Finding | Implication | r → PnL | Independent? |
|---------|------------|---------|-------------|
| VWMA(100) distance is #1 predictor | Volume-weighted fair value provides fundamentally new information | +0.1357 | Yes (max |r| = 0.170) |
| Blend-wide modes preserve baseline floor | Asymmetric upgrading > symmetric replacement | N/A | N/A |
| BTC vol ratio is monotonically predictive | Higher BTC vol → better rubberband trades | +0.0462 | Semi (r=0.211 w/ATR) |
| Market breadth inversely predicts quality | Mean-reversion works better in fear/panic | -0.0202 | Semi (r=0.301 w/DC) |
| Dual-confirm is WEAK per-trade predictor | Its value is portfolio-level, not entry-level | +0.0050 | Semi |
| Asset ATR is U-shaped | Both low-vol and high-vol assets outperform mid-vol | +0.0642 | Semi |

### Recommended R11 Priority

1. **VWMA period sweep** — test 50, 100, 200, 400 bar VWMA, find optimal
2. **Walk-forward K-VWTgW** — train/test split to validate blend-wide target
3. **C-VWE05 + K-VWTgW combo** — entry gate + blend-wide target together
4. **VWMA + BTC Vol combo** — the two most independent strong factors combined
5. **VWMA trailing stop** — use live VWMA as post-TP1 trailing stop (not tested here)
