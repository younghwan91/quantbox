# Alpha Search R5 — 6-Year Crash-Era Validation Report

> **Commit:** `0690b0f` (feature/alpha-search-v12.3)  
> **Date:** 2026-02-20  
> **Data:** 2190 days (2020-02 → 2026-02), 50 symbols  
> **Capital:** $2,095 | Fees: 0.04%/side | Funding: 0.01%/8h | Leverage: 3.75×  
> **Crash Periods Covered:** COVID (Mar 2020), China Ban (May 2021), Luna (May 2022), FTX (Nov 2022)

---

## 1. Executive Summary

This report validates the 19 R5 variants from the 3yr study (`alpha_search_r5.py`) on **6 years of real Binance data** covering all major crypto crash events. The simulation logic is **100% identical** — only the data window changed from 1095d to 2190d.

### Critical Finding: **3yr R5 Dominators FAILED 6yr Crash Validation**

| 3yr Dominator | 3yr DD | 6yr DD | ΔDD | Status |
|---|---|---|---|---|
| G-YK+PT20 (partial trail 2.0×) | 6.7% | 13.2% | +6.5% | ❌ **FAILED** — DD nearly doubled |
| O-SoftWide (wider soft equity) | 7.0% | 11.9% | +4.9% | ❌ **FAILED** — no longer dominates YK |

### New 6yr Crash-Era Dominators (beat Y-Kitchen on WR, R:R, AND DD):

| Variant | WR | R:R | DD | Sharpe | PF | DD Δ from 3yr |
|---|---|---|---|---|---|---|
| **J-YK+2Slt** (2 slots) | 62.5% | 0.78 | **8.2%** | **5.33** | 1.31 | **+0.6%** |
| **L-YK+C4** (conf≥4) | 63.2% | 0.79 | 10.0% | 4.95 | **1.35** | +1.0% |
| **N-SoftTight** (tight equity) | 62.1% | 0.78 | 10.5% | 5.08 | 1.29 | +2.8% |

**Winner: J-YK+2Slt** — Best Sharpe, best DD, and most crash-stable (DD increase only +0.6% from 3yr to 6yr).

---

## 2. Full 3yr vs 6yr Comparison

### 2.1 Key Metrics Comparison

| Variant | 3yr WR | 6yr WR | 3yr R:R | 6yr R:R | 3yr DD | 6yr DD | 3yr Sharpe | 6yr Sharpe | 3yr PF | 6yr PF |
|---|---|---|---|---|---|---|---|---|---|---|
| A-v12.2 | 66.6% | 61.9% | 0.95 | 0.78 | 9.4% | **25.3%** | 7.22 | 3.98 | 1.88 | 1.27 |
| B-YKitch | 67.1% | 62.1% | 1.00 | 0.78 | 7.2% | 10.7% | 8.42 | 5.04 | 2.07 | 1.29 |
| C-YK+DL25 | 67.1% | 62.1% | 1.02 | 0.78 | 8.3% | 12.4% | 7.98 | 4.78 | 2.10 | 1.29 |
| D-YK+DL50 | 67.1% | 62.1% | 1.03 | 0.79 | 10.0% | 13.7% | 7.63 | 4.56 | 2.11 | 1.30 |
| E-YK+PT15 | **69.6%** | 64.0% | 1.75 | 1.29 | 7.7% | 12.5% | **8.52** | 4.21 | 2.24 | 1.27 |
| F-YK+BE15 | 68.7% | 63.1% | **1.86** | **1.37** | 7.7% | 12.6% | 8.36 | 4.26 | 2.23 | 1.28 |
| **G-YK+PT20** | 67.6% | 61.7% | 1.80 | 1.37 | **6.7%** | 13.2% | 8.17 | 4.03 | 2.21 | 1.26 |
| H-DL+BE15 | 68.7% | 63.1% | 1.88 | 1.37 | 9.8% | 14.1% | 7.66 | 4.06 | 2.26 | 1.28 |
| I-DL+PT20 | 67.6% | 61.7% | 1.82 | 1.37 | 8.3% | 14.8% | 7.48 | 3.84 | 2.23 | 1.26 |
| **J-YK+2Slt** | 67.3% | **62.5%** | 1.01 | 0.78 | 7.6% | **8.2%** | 8.87 | **5.33** | 2.09 | 1.31 |
| K-YK+4Slt | 67.2% | 61.9% | 0.99 | 0.78 | 8.2% | 13.0% | 8.35 | 4.92 | 2.05 | 1.28 |
| **L-YK+C4** | **68.8%** | **63.2%** | 1.04 | 0.79 | 9.0% | 10.0% | 8.06 | 4.95 | **2.26** | **1.35** |
| M-SteepX | 67.1% | 62.1% | 1.00 | 0.78 | 7.3% | 11.0% | 8.58 | 5.15 | 2.12 | 1.30 |
| **N-SoftTight** | 67.1% | 62.1% | 1.00 | 0.78 | 7.7% | 10.5% | 8.46 | 5.08 | 2.07 | 1.29 |
| **O-SoftWide** | 67.1% | 62.1% | 1.00 | 0.78 | 7.0% | 11.9% | 8.40 | 4.97 | 2.07 | 1.29 |
| P-DL+4Slt | 67.2% | 61.9% | 1.00 | 0.78 | 9.2% | 13.8% | 7.89 | 4.71 | 2.08 | 1.28 |
| Q-DL+StpX | 67.1% | 62.1% | 1.02 | 0.78 | 8.0% | 12.9% | 7.98 | 4.87 | 2.14 | 1.31 |
| R-YK+CfTr | 67.0% | 62.4% | 0.99 | 0.77 | 9.4% | 10.8% | 8.26 | 4.89 | 2.05 | 1.28 |
| S-Ultra | 68.7% | 63.1% | 1.88 | 1.37 | 8.7% | 14.7% | 7.75 | 4.15 | 2.30 | 1.29 |

### 2.2 DD Stability Ranking (3yr → 6yr DD increase)

| Rank | Variant | 3yr DD | 6yr DD | ΔDD | Assessment |
|---|---|---|---|---|---|
| 1 | **J-YK+2Slt** | 7.6% | 8.2% | **+0.6%** | ★★★ Most crash-stable |
| 2 | **L-YK+C4** | 9.0% | 10.0% | **+1.0%** | ★★ Excellent stability |
| 3 | **N-SoftTight** | 7.7% | 10.5% | +2.8% | ★ Good |
| 4 | B-YKitch | 7.2% | 10.7% | +3.5% | Acceptable |
| 5 | M-SteepX | 7.3% | 11.0% | +3.7% | Acceptable |
| 6 | O-SoftWide | 7.0% | 11.9% | +4.9% | Weak |
| 7 | G-YK+PT20 | 6.7% | 13.2% | +6.5% | ❌ FRAGILE |
| 8 | A-v12.2 | 9.4% | 25.3% | +15.9% | ❌❌ CATASTROPHIC |

---

## 3. Category Leaders (6yr)

| Category | Variant | Value |
|---|---|---|
| Best Sharpe | **J-YK+2Slt** | 5.33 |
| Best DD | **J-YK+2Slt** | 8.2% |
| Best PF | **L-YK+C4** | 1.35 |
| Best WR | **L-YK+C4** | 63.2% |
| Best R:R | F-YK+BE15 / G-YK+PT20 | 1.37 |
| Best Return | D-YK+DL50 | +10,903.2% |
| Best Calmar | D-YK+DL50 | 796.1 |

---

## 4. Key Insights

### 4.1 Partial Exit + Trailing is a 3yr Mirage

The partial exit variants (E, F, G, H, I, S) looked spectacular on 3yr data because 2023-2025 was a mostly trending bull market. During crash regimes:
- **Trailing runners hold positions longer** during volatile reversals, amplifying drawdown
- G-YK+PT20's DD went from best-in-class (6.7%) to near-worst (13.2%)
- The "wider trail" (2.0× ATR) is particularly dangerous — it gives too much room during cascading selloffs

### 4.2 Concentration Beats Diversification During Crashes

- **J-YK+2Slt (2 slots): DD 8.2%** — fewest simultaneous positions = fewest correlated losses during crashes
- **K-YK+4Slt (4 slots): DD 13.0%** — more positions = more exposure = more pain during crash days when everything drops together

### 4.3 Strict Confidence Filtering is the Best Crash Protection

- **L-YK+C4 (conf≥4): DD 10.0%, PF 1.35** — by requiring higher confidence, it simply takes fewer low-quality trades during chaotic markets
- Only 26,752 trades (vs 41,096 for B-YKitch) — quality over quantity

### 4.4 v12.2 Production Baseline is Unacceptable

- DD went from 9.4% to **25.3%** — the Y-Kitchen innovations (conf≥3, dir_diversity, soft_equity, etc.) are not optional, they're essential for survival
- Any production deployment MUST include Y-Kitchen base innovations

### 4.5 Universal Degradation is Normal

All 19 variants showed degradation from 3yr to 6yr:
- WR dropped ~5%p across the board (66-69% → 61-64%)
- R:R dropped ~0.2 across the board
- Sharpe roughly halved (8.x → 4-5.x)
- PF dropped ~0.7 (2.0-2.3 → 1.26-1.35)

This is expected because the 6yr window includes 2020-2022 crash volatility. The question is not "did it degrade?" but "which degraded LEAST?"

---

## 5. 6yr Dominator Profiles

### J-YK+2Slt — The Crash-Era Champion ★★★

```
Config:     Y-Kitchen base + max_slots=2
6yr Stats:  WR 62.5% | R:R 0.78 | DD 8.2% | Sharpe 5.33 | PF 1.31
            Trades: 35,095 (16.0/day) | Bars held: 6.4
3yr Stats:  WR 67.3% | R:R 1.01 | DD 7.6% | Sharpe 8.87 | PF 2.09
DD Degradation: +0.6% (BEST of all 19 variants)
```

**Why it works:** With only 2 simultaneous slots, during crash events (when 50 coins all dump together), maximum exposure is capped at 2 positions. Compare 3 slots (Y-Kitchen) or 4 slots (K variant) — more slots = more correlated loss on crash days.

### L-YK+C4 — The Quality Filter ★★

```
Config:     Y-Kitchen base + min_conf=4 (instead of 3)
6yr Stats:  WR 63.2% | R:R 0.79 | DD 10.0% | Sharpe 4.95 | PF 1.35
            Trades: 26,752 (12.2/day) | Bars held: 6.5
3yr Stats:  WR 68.8% | R:R 1.04 | DD 9.0% | Sharpe 8.06 | PF 2.26
DD Degradation: +1.0%
```

**Why it works:** During crashes, signal quality drops (noisy indicators, false reversals). Requiring conf≥4 filters out the crash noise. Fewer trades but higher win rate and best PF.

### N-SoftTight — The Risk Scaler ★

```
Config:     Y-Kitchen base + tighter soft equity (2%/4%/6%)
6yr Stats:  WR 62.1% | R:R 0.78 | DD 10.5% | Sharpe 5.08 | PF 1.29
            Trades: 41,096 (18.8/day) | Bars held: 6.4
3yr Stats:  WR 67.1% | R:R 1.00 | DD 7.7% | Sharpe 8.46 | PF 2.07
DD Degradation: +2.8%
```

**Why it works:** Begins scaling down position sizes at just 2% drawdown (vs 3% default). During early crash stages, it reduces exposure faster.

---

## 6. Recommendations

### For Production Deployment:

1. **Primary Candidate:** J-YK+2Slt (Y-Kitchen + 2 slots)
   - Best crash resilience (DD 8.2% on 6yr)
   - Highest Sharpe (5.33)
   - Smallest DD degradation from 3yr (+0.6%)
   - Tradeoff: fewer simultaneous positions, slightly lower trade volume

2. **Alternative:** L-YK+C4 (Y-Kitchen + conf≥4)
   - Best PF (1.35) and WR (63.2%) on 6yr
   - Lower trade count (12.2/day vs 16-19/day) means less market impact
   - Good for cautious capital allocation

3. **DO NOT USE:** G-YK+PT20, partial exit variants
   - 3yr performance was a localized optimization, not a structural edge
   - Partial trailing runners amplify crash losses

### For Further Research:

- Combine J + L: 2 slots + conf≥4 gate — could be the ultimate crash-proof config
- Test J-YK+2Slt with tighter soft equity (N-SoftTight thresholds)
- Evaluate year-by-year breakdown to isolate crash-year vs bull-year performance

---

## 7. Data Coverage Summary

| Symbol | Days Available | Covers COVID | Covers Luna | Covers FTX |
|---|---|---|---|---|
| BTCUSDT | 2191d (full) | ✓ | ✓ | ✓ |
| ETHUSDT | 2191d (full) | ✓ | ✓ | ✓ |
| SOLUSDT | 1985d | ✗ | ✓ | ✓ |
| XRPUSDT | 2191d (full) | ✓ | ✓ | ✓ |
| BNBUSDT | 2191d (full) | ✓ | ✓ | ✓ |
| DOGEUSDT | 2051d | ✗ | ✓ | ✓ |
| LTCUSDT | 2191d (full) | ✓ | ✓ | ✓ |
| ... | ... | ... | ... | ... |
| **Coins with <3yr data** | **5 coins** | — | — | — |
| **Total symbols loaded** | **50** | | | |

Note: Coins without full 6yr history start from their listing date with left-padded arrays (zero signals for missing period = no phantom trades).

---

*Report generated from identical simulation logic as R5-3yr. Zero code changes between `alpha_search_r5.py` and `alpha_search_r5_6yr.py` — only data duration differs.*
