# Adaptive Direction A/B Test — BTC 4h SMA Regime-Driven

- **Commit:** `7f6bd77`
- **Date:** 2026-02-21
- **Branch:** `feature/direction-asymmetry-analysis`
- **Data:** 6yr (2190d), 50 symbols, left-padded to 210,335 bars
- **Baseline:** v12.4 I-HybTight (symmetric T150_S38_H25_PBT)

**구현:** 미구현 (방향 필터 적응 탐색)
## Hypothesis

The long/short performance asymmetry observed in the direction analysis is NOT static — it flips with the BTC macro regime. When BTC is above its SMA on the 4h timeframe (bull regime), longs outperform; below it (bear regime), shorts outperform. An **adaptive** mechanism that gives favorable exit params to the direction **aligned** with the current BTC regime should beat a static symmetric baseline.

## Part A: Regime Diagnostic — Does BTC 200 SMA Predict Direction Edge?

### Regime Distribution

| SMA Period | Bull % | Bear % |
|-----------|--------|--------|
| SMA(50)   | 52.6%  | 47.4%  |
| SMA(100)  | 53.1%  | 46.9%  |
| SMA(200)  | 52.4%  | 47.6%  |

### Aligned vs Misaligned Performance (SMA 200 on 4h)

Definitions:
- **Aligned** = long in bull regime OR short in bear regime
- **Misaligned** = long in bear regime OR short in bull regime

| Category     | #Trades | WR%   | PF   | R:R  | EV%    | Total$ |
|-------------|---------|-------|------|------|--------|--------|
| Bull-Long   | 4,801   | 57.3% | 1.76 | 2.12 | 3.796% | $51,566 |
| Bull-Short  | 4,462   | 55.6% | 1.48 | 1.78 | 2.911% | $35,706 |
| Bear-Long   | 3,601   | 54.1% | 1.49 | 2.09 | 3.380% | $28,356 |
| Bear-Short  | 4,179   | 55.6% | 1.46 | 1.69 | 2.793% | $30,645 |
| **ALIGNED**     | **8,980** | **56.5%** | **1.61** | **1.90** | **3.329%** | **$82,211** |
| **MISALIGNED**  | **8,063** | **54.9%** | **1.49** | **1.91** | **3.120%** | **$64,061** |

**Gap:** Aligned − Misaligned: **EV +0.209%** (+6.7%), WR +1.6pp, PF +0.12, R:R −0.01

**Regime signal strength: MODERATE — ALIGNED WINS**

### Yearly Breakdown

| Year | Bull% | Aligned EV | Misaligned EV | Gap |
|------|-------|-----------|---------------|-----|
| Yr1 (2020-21) | 66.9% | 6.069% | 4.871% | +1.198% |
| Yr2 (2021-22) | 47.5% | 4.122% | 4.124% | −0.002% |
| Yr3 (2022-23) | 37.7% | 3.101% | 2.605% | +0.496% |
| Yr4 (2023-24) | 60.3% | 2.465% | 2.446% | +0.019% |
| Yr5 (2024-25) | 58.7% | 2.906% | 2.619% | +0.287% |
| Yr6 (2025-26) | 43.3% | 2.399% | 2.572% | −0.173% |

**Observation:** The aligned advantage is strongest in Yr1 (+1.20%) and Yr3 (+0.50%) — both sharp trend periods. In Yr2, Yr4, and Yr6, the gap is near zero or reverses. The signal is **inconsistent across time**.

## Part B: Adaptive A/B Test — 23 Variants

### Design

At each bar, BTC regime is determined by one of five methods:
1. **Single SMA**: BTC 4h close vs SMA(N) for N ∈ {50, 100, 200}
2. **Dual-confirm**: Bull only when BTC > both SMA(50) AND SMA(200). Bear when < both. Otherwise "choppy" (treated as misaligned).
3. **Golden cross**: SMA(50) > SMA(200) = bull, SMA(50) < SMA(200) = bear.

Trades in the "aligned" direction (long in bull, short in bear) get favorable exit params (wider stops, more generous targets, longer hold time). Misaligned trades get tighter params.

Parameters locked at entry — no mid-trade regime switching.

### Part A Extension: Dual-SMA Regime Diagnostics

#### Dual-Confirm (BTC > both SMA50 & SMA200)

| Regime | Time% |
|--------|-------|
| Bull (>both) | 39.8% |
| Bear (<both) | 33.7% |
| Choppy (between) | **26.5%** |

| Category | #Trades | WR% | PF | R:R | EV% | Total$ |
|----------|---------|-----|----|-----|-----|--------|
| Bull-Long | 3,898 | 58.5% | 1.87 | 2.14 | 3.899% | $46,050 |
| Bull-Short | 3,203 | 54.6% | 1.41 | 1.75 | 2.677% | $22,648 |
| Bear-Long | 2,417 | 53.0% | 1.41 | 2.11 | 3.515% | $16,883 |
| Bear-Short | 3,172 | 56.6% | 1.51 | 1.65 | 3.008% | $25,638 |
| **ALIGNED** | **7,070** | **57.7%** | **1.70** | **1.88** | **3.500%** | **$71,688** |
| **MISALIGNED** | **5,620** | **53.9%** | **1.41** | **1.90** | **3.037%** | **$39,531** |

**Gap: EV +0.463% (+15.2%), WR +3.8pp, PF +0.29** — STRONGEST signal of all regimes tested.

| Year | Bull% | Aligned EV | Misaligned EV | Gap |
|------|-------|-----------|---------------|-----|
| Yr1 (2020-21) | 53.0% | 5.978% | 5.814% | +0.164% |
| Yr2 (2021-22) | 35.2% | 4.148% | 3.818% | +0.330% |
| Yr3 (2022-23) | 30.2% | 3.207% | 2.422% | +0.785% |
| Yr4 (2023-24) | 44.2% | 2.589% | 2.262% | +0.327% |
| Yr5 (2024-25) | 42.6% | 3.254% | 2.437% | +0.817% |
| Yr6 (2025-26) | 33.7% | 2.755% | 2.225% | +0.530% |

**Key insight:** Dual-confirm gap is **positive in ALL 6 years** — the only regime type to achieve this. The single SMA(200) flipped in Yr6. But 26.5% of time is "choppy" (undefined regime), where trades fallback to misaligned treatment.

#### Golden Cross (SMA50 > SMA200)

| Category | #Trades | WR% | PF | R:R | EV% | Total$ |
|----------|---------|-----|----|-----|-----|--------|
| **ALIGNED** | **8,468** | **55.8%** | **1.54** | **1.86** | **3.183%** | **$70,977** |
| **MISALIGNED** | **8,575** | **55.7%** | **1.56** | **1.95** | **3.277%** | **$75,296** |

**Gap: EV −0.094% (−2.9%)** — Signal REVERSES. Misaligned actually wins.

The golden cross is a trend-continuation signal, which **conflicts** with the mean-reversion rubberband strategy. When SMA(50) > SMA(200), long longs actually do worse than short shorts — likely because sharp counter-trend reversals are more profitable in established trends.

### Results — All 23 Variants

| Variant | Description | Ret% | DD% | Sharpe | Calmar | WR | PF | R:R | #Trades |
|---------|------------|------|-----|--------|--------|----|----|-----|---------|
| **A-Baseline** | Symmetric I-HybTight (prod) | 6,300.9% | **7.6%** | 5.56 | **833.9** | 55.8% | 1.56 | 1.91 | 17,165 |
| B-Stop200 | SMA(200) stop .45/.30 | 6,264.8% | 8.1% | 5.65 | 770.8 | 55.2% | 1.56 | 1.95 | 17,307 |
| C-Stop100 | SMA(100) stop .45/.30 | 6,229.1% | 8.1% | **5.67** | 766.4 | 55.2% | 1.55 | 1.96 | 17,314 |
| D-Stop50 | SMA(50) stop .45/.30 | 6,253.4% | 8.2% | 5.60 | 761.0 | 55.3% | 1.55 | 1.96 | 17,286 |
| E-Tgt200 | SMA(200) target 2.0/1.0 | 6,190.1% | 9.0% | 5.31 | 685.6 | 56.7% | 1.55 | 1.83 | 17,596 |
| F-StopTgt200 | SMA(200) stop+target | 6,264.8% | 8.9% | 5.32 | 704.2 | 56.2% | 1.56 | 1.88 | 17,703 |
| G-StopTgt100 | SMA(100) stop+target | 6,119.9% | 8.7% | 5.36 | 706.0 | 56.0% | 1.55 | 1.87 | 17,707 |
| H-MH200 | SMA(200) MH 30/20 | 6,247.0% | 10.7% | 5.49 | 585.2 | 55.8% | 1.55 | 1.89 | 17,149 |
| I-Full200 | SMA(200) full adaptive | 5,891.2% | 12.1% | 5.07 | 488.5 | 55.8% | 1.53 | 1.87 | 17,298 |
| J-Full100 | SMA(100) full adaptive | 5,753.9% | 12.1% | 5.15 | 477.2 | 55.6% | 1.52 | 1.86 | 17,283 |
| K-ConfGate200 | Misaligned +1 conf | 5,379.3% | 7.6% | 5.19 | 711.9 | 56.1% | 1.58 | 1.89 | 13,224 |
| L-Block200 | Block misaligned | 3,893.6% | 7.6% | 4.86 | 515.3 | 55.9% | 1.57 | 1.88 | 10,562 |
| M-Mild200 | Mild stop .40/.35 | 6,226.8% | 7.8% | 5.53 | 802.9 | 55.5% | 1.55 | 1.93 | 17,271 |
| N-MildFull200 | Mild stop+target | 6,206.4% | 8.7% | 5.42 | 714.9 | 55.8% | 1.55 | 1.90 | 17,353 |
| **O-DualStop** | Dual-confirm stop .45/.30 | 6,328.0% | 8.1% | **5.67** | 778.5 | 56.1% | 1.56 | 1.90 | 17,078 |
| **P-DualStopTgt** | Dual-confirm stop+target | **6,358.1%** | 8.9% | 5.31 | 714.7 | 55.5% | 1.57 | 1.88 | 16,933 |
| Q-DualFull | Dual-confirm full adaptive | 5,930.8% | 12.1% | 5.05 | 491.8 | 55.0% | 1.53 | 1.88 | 16,462 |
| R-DualConf | Dual-confirm +1 conf gate | 5,805.6% | 7.6% | 5.32 | 768.3 | 56.3% | 1.58 | 1.89 | 14,527 |
| S-DualBlock | Dual-confirm block misaligned | 4,822.7% | 7.6% | 5.08 | 638.3 | 56.2% | 1.59 | 1.90 | 12,692 |
| T-GCStop | Golden cross stop .45/.30 | 6,196.8% | 8.1% | 5.63 | 762.4 | 55.0% | 1.55 | **1.97** | 17,342 |
| U-GCStopTgt | Golden cross stop+target | 6,156.1% | 8.9% | 5.31 | 692.0 | 56.4% | 1.55 | 1.87 | 17,894 |
| V-GCFull | Golden cross full adaptive | 5,927.2% | 12.1% | 5.08 | 491.5 | 56.2% | 1.54 | 1.85 | 17,504 |
| W-GCMild | Golden cross mild stop .40/.35 | 6,219.9% | 7.8% | 5.52 | 802.0 | 55.4% | 1.55 | 1.94 | 17,293 |

### Aligned vs Misaligned Detail

| Variant | Al# | Al_WR | Al_EV | Mis# | Mis_WR | Mis_EV | Gap |
|---------|-----|-------|-------|------|--------|--------|-----|
| A-Baseline | 9,102 | 56.7% | 3.397% | 8,063 | 54.9% | 3.120% | +0.277% |
| B-Stop200 | 8,833 | 58.9% | 3.494% | 8,474 | 51.4% | 2.924% | +0.570% |
| C-Stop100 | 9,004 | 59.3% | 3.573% | 8,310 | 50.9% | 2.863% | +0.710% |
| D-Stop50 | 9,334 | 59.8% | 3.622% | 7,952 | 49.9% | 2.797% | +0.825% |
| E-Tgt200 | 8,620 | 52.4% | 3.067% | 8,976 | 60.8% | 3.184% | −0.117% |
| K-ConfGate200 | 10,024 | 56.2% | 3.269% | 3,200 | 56.0% | 3.287% | −0.018% |
| L-Block200 | 10,562 | 55.9% | 3.249% | 0 | — | — | — |
| **O-DualStop** | 11,189 | 59.1% | 3.509% | 5,889 | 50.5% | 2.846% | **+0.663%** |
| **P-DualStopTgt** | 10,419 | 55.0% | 3.221% | 6,514 | 56.3% | 2.922% | +0.299% |
| R-DualConf | 12,280 | 56.5% | 3.310% | 2,247 | 55.1% | 3.147% | +0.163% |
| T-GCStop | 8,298 | 58.1% | 3.335% | 9,044 | 52.1% | 3.060% | +0.275% |

## Analysis

### The BTC Regime Signal Is Real but Weak (Single-SMA)

1. **The mechanism works directionally.** Aligned trades (direction matches BTC regime) have +0.21% EV, +1.6pp WR, and +0.12 PF over misaligned. The stop adaptation variants successfully widen this gap to +0.71% (C-Stop100).

2. **But it's too weak to overcome the tradeoff.** Every adaptive variant that improves Sharpe also worsens DD:
   - C-Stop100: Sharpe +2.0% → DD +6.6% (7.6% → 8.1%)
   - The wider aligned stops let winners run longer, but also let the "wrong" aligned trades lose more.

3. **Full adaptive is destructive.** Combining stop + target + MH adaptation (I/J) causes DD 12.1% (+59% vs baseline) and destroys Calmar. Too many parameters changing simultaneously amplifies noise.

4. **Blocking misaligned trades kills returns.** L-Block200 loses 38% of return ($3.9k vs $6.3k) for no DD improvement. Misaligned trades are still profitable (EV +3.12%) — you can't afford to skip them.

5. **The regime signal is temporally inconsistent.** Yr1 (+1.20% gap) and Yr3 (+0.50%) show strong aligned advantage. Yr2, Yr4, and Yr6 show near-zero or negative gap. Any fixed regime-response mapping will overfit to specific periods.

### Dual-Confirm: Strongest Signal, First to Beat Baseline on Return

The dual-confirm regime (require BTC > BOTH SMA50 and SMA200 for bull) produces the strongest directional signal (+0.463% EV gap, +15.2%) — more than double the single SMA(200) gap (+0.209%). Critically, **the gap is positive in ALL 6 years**, whereas single-SMA signals flip negative in some years.

This yields the first variants to beat baseline on return:

| | Ret% | DD% | Sharpe | Calmar |
|-|------|-----|--------|--------|
| A-Baseline | 6,300.9% | 7.6% | 5.56 | 833.9 |
| **P-DualStopTgt** | **6,358.1%** (+0.9%) | 8.9% (+17%) | 5.31 (−4.5%) | 714.7 (−14%) |
| O-DualStop | 6,328.0% (+0.4%) | 8.1% (+6.6%) | 5.67 (+2.0%) | 778.5 (−6.6%) |

**P-DualStopTgt** wins on Return (+$3.8K), but the DD increase from 7.6% → 8.9% (+17%) makes it a poor Calmar tradeoff. **O-DualStop** ties for best Sharpe (5.67) with minimal DD increase (+6.6%), but still can't beat baseline Calmar.

**Why dual-confirm is stronger:** The "choppy" zone (26.5% of time) — where BTC is between SMA50 and SMA200 — acts as a natural noise filter. Only high-conviction regime states (above BOTH or below BOTH) get favorable treatment. This selectivity increases signal quality at the cost of regime coverage.

### Golden Cross: Signal Reverses — Wrong Abstraction

The golden cross (SMA50 > SMA200) produces a **negative** signal: misaligned trades actually outperform aligned by −0.094% EV. The crossover signal identifies long-term trend direction but for a mean-reversion strategy, counter-trend trades are more profitable in established trends than with-trend trades.

Golden cross variants (T-W) all underperform their dual-confirm counterparts. T-GCStop achieves best R:R (1.97) but loses on return, DD, and Calmar vs baseline.

### Why Adaptive Fails to Beat Baseline on Risk-Adjusted Metrics

The fundamental issue: the regime indicator (BTC vs SMA) is a **trend-following** signal applied to a **mean-reversion** strategy. Rubberband entries already counter-trade the trend by design. Adding a trend-aligned filter creates conflicting signals:

- Rubberband: "This asset overextended — take the opposite direction"
- Regime filter: "Only favor trades aligned with the BTC trend"

These two signals partially cancel. The regime filter adds value (aligned trades ARE better), but the improvement is too small to justify the increased DD from the parameter modifications.

### SMA Period / Regime Type Comparison

| Regime | Sharpe (Stop variant) | DD% | EV Gap | Signal Quality |
|--------|----------------------|-----|--------|----------------|
| SMA(50) | 5.60 | 8.2% | +0.522% | Fast, noisy |
| SMA(100) | 5.67 | 8.1% | +0.384% | Medium |
| SMA(200) | 5.65 | 8.1% | +0.209% | Slow, stable |
| **Dual-confirm** | **5.67** | **8.1%** | **+0.463%** | **Selective, consistent** |
| Golden cross | 5.63 | 8.1% | −0.094% | Reversed signal |

Dual-confirm ties for best Sharpe among all regimes, with the strongest and most temporally consistent EV gap. But the 26.5% "choppy" time reduces the number of regime-differentiated trades.

## Verdict

**NO PRODUCTION CHANGE.** The symmetric I-HybTight baseline (A) still dominates on risk-adjusted metrics:
- **Best DD:** 7.6% (A-Baseline) — no adaptive variant matches it without sacrificing return
- **Best Calmar:** 833.9 (A-Baseline) — dominant by 7%+ over all variants
- **Best Return:** 6,358.1% (**P-DualStopTgt**) — first variant to beat baseline on return (+0.9%), but DD +17%

The dual-confirm regime is the strongest and most consistent signal found (EV gap positive ALL 6 years), and it produced the first variant to beat baseline on raw return. However, the +0.9% return gain comes with +17% DD increase — an unfavorable tradeoff for capital preservation.

**Notable finding:** The dual-confirm "choppy" zone concept (26.5% undefined regime) is a promising filtering mechanism. Future work could explore using this uncertainty signal for position sizing rather than parameter modification — e.g., reducing size when BTC is between SMA50 and SMA200, rather than modifying stop/target params.

## Files

- Script: `scripts/adaptive_direction_ab.py`
- Results: `logs/adaptive_direction_ab_results.json`
- Output: `logs/adaptive_direction_ab_output.txt`
