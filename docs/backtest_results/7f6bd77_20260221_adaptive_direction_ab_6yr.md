# Adaptive Direction A/B Test — BTC 4h SMA Regime-Driven

- **Commit:** `7f6bd77`
- **Date:** 2026-02-21
- **Branch:** `feature/direction-asymmetry-analysis`
- **Data:** 6yr (2190d), 50 symbols, left-padded to 210,335 bars
- **Baseline:** v12.4 I-HybTight (symmetric T150_S38_H25_PBT)

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

## Part B: Adaptive A/B Test — 14 Variants

### Design

At each bar, BTC regime is determined by `BTC 4h close vs SMA(N)`. Trades in the "aligned" direction (long in bull, short in bear) get favorable exit params (wider stops, more generous targets, longer hold time). Misaligned trades get tighter params.

Parameters locked at entry — no mid-trade regime switching.

### Results

| Variant | Description | Ret% | DD% | Sharpe | Calmar | WR | PF | R:R | #Trades |
|---------|------------|------|-----|--------|--------|----|----|-----|---------|
| **A-Baseline** | Symmetric I-HybTight (prod) | **6,300.9%** | **7.6%** | 5.56 | **833.9** | 55.8% | 1.56 | 1.91 | 17,165 |
| B-Stop200 | SMA(200) stop .45/.30 | 6,264.8% | 8.1% | 5.65 | 770.8 | 55.2% | 1.56 | 1.95 | 17,307 |
| **C-Stop100** | SMA(100) stop .45/.30 | 6,229.1% | 8.1% | **5.67** | 766.4 | 55.2% | 1.55 | **1.96** | 17,314 |
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

## Analysis

### The BTC Regime Signal Is Real but Weak

1. **The mechanism works directionally.** Aligned trades (direction matches BTC regime) have +0.21% EV, +1.6pp WR, and +0.12 PF over misaligned. The stop adaptation variants successfully widen this gap to +0.71% (C-Stop100).

2. **But it's too weak to overcome the tradeoff.** Every adaptive variant that improves Sharpe also worsens DD:
   - C-Stop100: Sharpe +2.0% → DD +6.6% (7.6% → 8.1%)
   - The wider aligned stops let winners run longer, but also let the "wrong" aligned trades lose more.

3. **Full adaptive is destructive.** Combining stop + target + MH adaptation (I/J) causes DD 12.1% (+59% vs baseline) and destroys Calmar. Too many parameters changing simultaneously amplifies noise.

4. **Blocking misaligned trades kills returns.** L-Block200 loses 38% of return ($3.9k vs $6.3k) for no DD improvement. Misaligned trades are still profitable (EV +3.12%) — you can't afford to skip them.

5. **The regime signal is temporally inconsistent.** Yr1 (+1.20% gap) and Yr3 (+0.50%) show strong aligned advantage. Yr2, Yr4, and Yr6 show near-zero or negative gap. Any fixed regime-response mapping will overfit to specific periods.

### Why Adaptive Fails Here

The fundamental issue: the regime indicator (BTC vs SMA) is a **trend-following** signal applied to a **mean-reversion** strategy. Rubberband entries already counter-trade the trend by design. Adding a trend-aligned filter creates conflicting signals:

- Rubberband: "This asset overextended — take the opposite direction"
- Regime filter: "Only favor trades aligned with the BTC trend"

These two signals partially cancel. The regime filter adds value (aligned trades ARE better), but the improvement is too small to justify the increased DD from the parameter modifications.

### SMA Period Comparison

| SMA Period | Sharpe (Stop variant) | DD% | Signal Lag |
|-----------|----------------------|-----|------------|
| SMA(50)  | 5.60 | 8.2% | Fast, noisy |
| SMA(100) | 5.67 | 8.1% | Medium |
| SMA(200) | 5.65 | 8.1% | Slow, stable |

SMA(100) marginally wins on Sharpe but all three are within noise. No clear optimal period.

## Verdict

**NO PRODUCTION CHANGE.** The symmetric I-HybTight baseline (A) dominates on the metrics that matter most for capital preservation:
- **Best Return:** 6,300.9% (none beat it)
- **Best/Tied DD:** 7.6% (K, L match but with worse returns)
- **Best Calmar:** 833.9 (no variant comes close)

The +2% Sharpe gain from C-Stop100 is not worth the +6.6% DD increase. For a live system where drawdowns create psychological pressure and margin risk, the conservative choice is unambiguously correct.

The BTC regime signal is stored and documented for potential future use in a **softer** application — e.g., as a confidence tiebreaker when ranking candidates, rather than as a parameter modifier. But even that application would need further validation.

## Files

- Script: `scripts/adaptive_direction_ab.py`
- Results: `logs/adaptive_direction_ab_results.json`
- Output: `logs/adaptive_direction_ab_output.txt`
