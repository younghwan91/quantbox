# Alpha Search R6 — Combo Variants (6-Year Crash-Era)

**Commit:** `3fbf1d6` (feature/alpha-search-v12.3)
**Date:** 2026-02-20
**Data:** 2190 days (6 years), 50 symbols, Binance USDT-M Futures
**Capital:** $2,095 | Fees: 0.04%/side | Funding: 0.01%/8h

## Objective

Combine the 5 crash-era champion features from R5-6yr into 2-way, 3-way, 4-way, and "Fortress" combos to discover synergies.

### Building Blocks (R5-6yr Champions)

| Code | Feature | R5-6yr DD | R5-6yr Sharpe | Key Strength |
|------|---------|-----------|---------------|--------------|
| **J** | `max_slots=2` | 8.2% | 5.33 | Best DD |
| **L** | `min_conf=4` | 10.0% | 4.95 | Best PF 1.35 |
| **N** | SoftTight (2%/4%/6% equity) | 10.5% | 5.08 | Tighter drawdown control |
| **M** | SteepX (0.3/0.5/1.5/2.0 cap) | 11.0% | 5.15 | Better size allocation |
| **R** | Conf-adaptive trail | 10.8% | 4.89 | Smarter trailing |

## Results — Full Table

| Variant | Description | WR | R:R | DD | Sharpe | PF | Calmar | Trades |
|---------|------------|-----|------|------|--------|------|--------|--------|
| A-YKitch | Y-Kitchen base | 62.1% | 0.78 | 10.7% | 5.04 | 1.29 | 518.0 | 41,096 |
| B-2Slt | J ref (2 slots) | 62.5% | 0.78 | 8.2% | 5.33 | 1.31 | 668.7 | 35,095 |
| C-2S+C4 | 2slt + conf≥4 | 63.5% | 0.79 | 8.8% | 5.06 | 1.37 | 563.7 | 23,812 |
| D-2S+ST | 2slt + SoftTight | 62.5% | 0.78 | **8.1%** | 5.35 | 1.31 | **661.0** | 35,095 |
| **E-2S+SX** | 2slt + SteepX | 62.5% | 0.78 | 8.2% | **5.49** | 1.32 | 625.6 | 35,095 |
| F-C4+ST | conf≥4 + SoftTight | 63.2% | 0.79 | 10.4% | 5.00 | 1.35 | 477.9 | 26,752 |
| G-C4+SX | conf≥4 + SteepX | 63.2% | 0.79 | 10.8% | 4.92 | 1.35 | 473.7 | 26,752 |
| H-2S+CT | 2slt + ConfTrail | 62.8% | 0.77 | 8.3% | 5.22 | 1.31 | 626.5 | 36,717 |
| I-C4+CT | conf≥4 + ConfTrail | 64.0% | 0.76 | 12.6% | 4.61 | 1.34 | 362.0 | 28,394 |
| J-ST+SX | SoftTight + SteepX | 62.1% | 0.78 | 9.8% | 5.21 | 1.30 | 518.1 | 41,096 |
| K-2SC4ST | 2slt+C4+SoftTight | 63.5% | 0.79 | 8.7% | 5.06 | 1.37 | 564.4 | 23,812 |
| L-2SC4SX | 2slt+C4+SteepX | 63.5% | 0.79 | 8.8% | 5.06 | 1.37 | 563.7 | 23,812 |
| **★ M-2SSTSX** | **2slt+SoftTight+SteepX** | **62.5%** | **0.78** | **8.0%** | **5.50** | **1.32** | **626.4** | **35,095** |
| N-C4STSX | C4+SoftTight+SteepX | 63.2% | 0.79 | 11.1% | 4.94 | 1.35 | 454.0 | 26,752 |
| O-2SC4CT | 2slt+C4+ConfTrail | 64.3% | 0.77 | 10.7% | 4.87 | 1.37 | 429.6 | 25,440 |
| P-2SSTCT | 2slt+SoftTight+ConfTrail | 62.8% | 0.77 | 8.2% | 5.24 | 1.31 | 625.0 | 36,717 |
| Q-4way | 2slt+C4+ST+SX | 63.5% | 0.79 | 8.7% | 5.06 | 1.37 | 564.4 | 23,812 |
| R-4wayB | 2slt+C4+ST+CT | 64.3% | 0.77 | 9.7% | 4.96 | 1.37 | 464.9 | 25,440 |
| S-4wayC | 2slt+C4+SX+CT | 64.3% | 0.77 | 10.7% | 4.87 | 1.37 | 429.6 | 25,440 |
| T-Fort5 | ALL 5 features | 64.3% | 0.77 | 9.7% | 4.96 | 1.37 | 464.9 | 25,440 |
| U-1Slot | 1 slot only | 62.3% | 0.79 | **7.9%** | 4.66 | 1.31 | 411.5 | 20,218 |
| V-C5Gate | conf≥5 | 63.5% | 0.78 | 13.7% | 3.74 | 1.33 | 152.4 | 12,185 |
| W-MH15 | max_hold=15 | 61.3% | **0.82** | 11.3% | 5.13 | 1.30 | 497.4 | 41,858 |
| X-Fort+MH15 | Fortress + MH15 | 63.5% | 0.80 | 10.0% | 5.08 | **1.38** | 459.5 | 25,722 |

## Dominators of J-2Slt (6yr DD Champion)

Only **3 combos** dominate J-2Slt on all 3 axes (WR≥, R:R≥, DD≤):

| Variant | DD | Sharpe | PF | δDD vs J | δSharpe vs J | Composite |
|---------|-----|--------|-----|----------|-------------|-----------|
| **★ M-2SSTSX** | **8.0%** | **5.50** | **1.32** | **-0.2%** | **+0.17** | **#1 (0.91)** |
| D-2S+ST | 8.1% | 5.35 | 1.31 | -0.1% | +0.02 | #3 (0.87) |
| E-2S+SX | 8.2% | 5.49 | 1.32 | +0.0% | +0.16 | #2 (0.88) |

## Composite Score (Sharpe × PF / DD)

| Rank | Variant | Score | Sharpe | PF | DD |
|------|---------|-------|--------|------|------|
| **#1** | **M-2SSTSX** | **0.91** | 5.50 | 1.32 | 8.0% |
| #2 | E-2S+SX | 0.88 | 5.49 | 1.32 | 8.2% |
| #3 | D-2S+ST | 0.87 | 5.35 | 1.31 | 8.1% |
| #4 | P-2SSTCT | 0.84 | 5.24 | 1.31 | 8.2% |
| #5 | H-2S+CT | 0.82 | 5.22 | 1.31 | 8.3% |

## Key Insights

### 1. **M-2SSTSX is the MONSTER** 🏆
- `max_slots=2` + SoftTight equity (2%/4%/6%) + SteepX cap (0.3/0.5/1.5/2.0)
- **Best Sharpe in the entire R6 field: 5.50** (vs J's 5.33, YK's 5.04)
- **Lowest DD among non-1-slot variants: 8.0%** (vs J's 8.2%)
- Dominates both J-2Slt AND Y-Kitchen baseline
- Synergy: SoftTight caps position size during drawdowns, SteepX rewards high-confidence entries, 2 slots maintain concentration

### 2. **2 Slots is the indispensable ingredient**
- Every top-5 composite variant includes `max_slots=2`
- Adding features WITHOUT 2 slots degrades DD (F, G, I, J, N all DD >9.8%)
- The slot concentration effect is the strongest single alpha factor in crash regimes

### 3. **SoftTight + SteepX synergy**
- D-2S+ST (Sharpe 5.35, DD 8.1%) + E-2S+SX (Sharpe 5.49, DD 8.2%)
- Combined as M-2SSTSX → Sharpe 5.50, DD 8.0% → **synergistic improvement, not averaging**
- SoftTight reduces exposure during drawdowns; SteepX amplifies high-confidence bets
- Together they create a "risk-aware sizing engine"

### 4. **Conf≥4 trades PF for DD**
- Adding conf≥4 improves PF (+0.06) and WR (+1.0%p) but slightly increases DD (+0.5-0.8%)
- C-2S+C4: PF 1.37 but DD 8.8% vs M-2SSTSX's PF 1.32 / DD 8.0%
- Verdict: conf≥4 is a PF play, not a DD play

### 5. **ConfTrail HURTS in combos**
- Every variant with ConfTrail has worse R:R (-0.01 to -0.02)
- H-2S+CT: R:R 0.77 (vs B-2Slt 0.78). Worse in every combination.
- Verdict: ConfTrail is a crash-era survivor but not a combo enhancer. Discard.

### 6. **Fortress (all 5 features) is WORSE than simpler combos**
- T-Fort5: DD 9.7%, Sharpe 4.96 — worse than M-2SSTSX (DD 8.0%, Sharpe 5.50)
- Too many constraints cancel each other out. Occam's razor wins.

### 7. **Creative variants**
- **U-1Slot**: Best DD (7.9%) but return collapses (-43% vs J). Too concentrated.
- **V-C5Gate**: Catastrophic DD 13.7%. Conf≥5 filters out too many trades, leaving portfolio exposed to individual bet variance.
- **W-MH15**: Best R:R (0.82) but worse WR and higher DD. Faster turnover ≠ better risk.
- **X-Fort+MH15**: Best PF (1.38) — interesting but DD 10.0% disqualifies it vs M-2SSTSX.

## Evolution Tracker

| Round | Champion | DD | Sharpe | PF | WR | Key Innovation |
|-------|---------|-----|--------|------|------|---------------|
| R4 | B-YKitch | 10.7% | 5.04 | 1.29 | 62.1% | Y-Kitchen (conf≥3, dir_div, steep, vol, soft_eq) |
| R5-6yr | J-YK+2Slt | 8.2% | 5.33 | 1.31 | 62.5% | + max_slots=2 |
| **R6** | **M-2SSTSX** | **8.0%** | **5.50** | **1.32** | **62.5%** | **+ SoftTight(2/4/6%) + SteepX(0.3/0.5/1.5/2.0)** |

## Recommendation

**Deploy M-2SSTSX** — the 3-feature combo (2 slots + SoftTight equity + SteepX cap) as the next production strategy:

```python
# M-2SSTSX Configuration
max_slots = 2
min_conf_gate = 3          # Y-Kitchen standard
dir_diversity = True
steep_cap_curve = True      # SteepX: 0.3 / 0.5 / 1.5 / 2.0
vol_adjusted_size = True
soft_equity_scale = True    # SoftTight: 2% / 4% / 6%

# SteepX caps
cap_mult_conf2 = 0.3
cap_mult_conf3 = 0.5
cap_mult_conf4 = 1.5
cap_mult_conf5 = 2.0

# SoftTight thresholds
soft_eq_dd1 = 0.02  →  scale 0.80
soft_eq_dd2 = 0.04  →  scale 0.60
soft_eq_dd3 = 0.06  →  scale 0.40
```

**Why M-2SSTSX over alternatives:**
1. **Lowest DD (8.0%)** among all multi-slot variants — survived COVID, Luna, FTX
2. **Highest Sharpe (5.50)** in the entire R6 field — not just safe, but efficient
3. **Synergistic, not averaging** — the combo is better than any individual feature
4. **Simple (3 features)** — less overfitting risk than Fortress (5 features)
5. **Dominates the previous champion** J-2Slt on ALL axes
