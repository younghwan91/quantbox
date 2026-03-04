# Universe Expansion Sweep — 30 Symbols × Multi-Slot Assessment

- **Commit:** `6eb0212`
- **Date:** 2026-02-19
- **Period:** 1095 days (3 years), 2023-02 → 2026-02
- **Config:** Option B (conf≥3) + MTF block + Adaptive 2.5×_f1
- **Capital:** $2,095 flat (no compounding)

## Purpose

Assess whether expanding the trading universe from 20 → 30 symbols can
increase trade frequency while maintaining quality, and whether adding
more parallel slots (3 or 4) further improves returns.

## Variants

| Variant | Symbols | Slots | Description |
|---------|---------|-------|-------------|
| B-20 | 20 | 2 | Current baseline |
| B-30 | 30 | 2 | More symbols → quality filter picks best |
| B-30-3s | 30 | 3 | More symbols + more parallelism |
| B-30-4s | 30 | 4 | Maximum parallelism |

All share identical signal generation (conf≥3, MTF block, 3.75× trend-aligned).

## Results

| Metric | B-20 | B-30 | B-30-3s | B-30-4s |
|--------|------|------|---------|---------|
| Total Return | +3,160% | +4,985% | +5,697% | +6,035% |
| Final Balance | $68,289 | $106,538 | $121,437 | $128,532 |
| **Sharpe** | **5.98** | **7.09** | **7.01** | **6.97** |
| Calmar | 329 | 513 | 421 | 435 |
| **Max Drawdown** | **9.6%** | **9.7%** | **13.5%** | **13.9%** |
| Total Trades | 5,960 | 8,833 | 10,747 | 12,007 |
| **Trades/Day** | **5.44** | **8.06** | **9.81** | **10.96** |
| Win Rate | 66.3% | 66.7% | 67.2% | 67.5% |
| Profit Factor | 2.17 | 2.28 | 2.31 | 2.33 |
| Avg PnL/Trade | +1.21% | +1.28% | +1.31% | +1.34% |
| Long Trades | 3,739 | 5,683 | 6,978 | 7,833 |
| Short Trades | 2,221 | 3,150 | 3,769 | 4,174 |
| Long WR | 67.2% | 67.7% | 68.3% | 68.9% |
| Short WR | 64.9% | 65.0% | 65.0% | 64.8% |

## Yearly Breakdown

### Year 1: 2023-02 → 2024-02 (Recovery/Bull)

| | B-20 | B-30 | B-30-3s | B-30-4s |
|---|------|------|---------|---------|
| Return | +1,218% | +1,858% | +2,091% | +2,196% |
| Trades | 1,877 | 2,687 | 3,224 | 3,551 |
| Win Rate | 68.4% | 68.9% | 69.3% | 69.6% |
| PF | 2.55 | 2.75 | 2.77 | 2.80 |
| Max DD | 9.6% | 9.7% | 13.5% | 13.9% |
| Sharpe | 7.73 | 9.27 | 9.19 | 9.14 |

### Year 2: 2024-02 → 2025-02 (Bull/Mixed)

| | B-20 | B-30 | B-30-3s | B-30-4s |
|---|------|------|---------|---------|
| Return | +1,323% | +2,122% | +2,480% | +2,628% |
| Trades | 2,111 | 3,139 | 3,847 | 4,304 |
| Win Rate | 65.6% | 66.3% | 66.7% | 67.0% |
| PF | 2.03 | 2.14 | 2.22 | 2.24 |
| Max DD | 2.8% | 2.1% | 2.0% | 1.8% |
| Sharpe | 7.33 | 10.24 | 10.89 | 11.09 |

### Year 3: 2025-02 → 2026-02 (Bear/Current)

| | B-20 | B-30 | B-30-3s | B-30-4s |
|---|------|------|---------|---------|
| Return | +1,070% | +1,672% | +1,875% | +2,000% |
| Trades | 1,965 | 2,999 | 3,668 | 4,144 |
| Win Rate | 65.1% | 65.4% | 65.8% | 66.2% |
| PF | 2.08 | 2.12 | 2.12 | 2.13 |
| Max DD | 1.2% | 1.2% | 1.4% | 1.4% |
| Sharpe | 7.74 | 10.55 | 10.58 | 10.63 |

## New Symbols Performance

| Symbol | Signals | Long | Short | L/S |
|--------|---------|------|-------|-----|
| SUIUSDT | 589 | 341 | 248 | 1.4 |
| AAVEUSDT | 518 | 377 | 141 | 2.7 |
| INJUSDT | 484 | 321 | 163 | 2.0 |
| UNIUSDT | 478 | 348 | 130 | 2.7 |
| NEARUSDT | 477 | 333 | 144 | 2.3 |
| APTUSDT | 467 | 327 | 140 | 2.3 |
| BCHUSDT | 439 | 332 | 107 | 3.1 |
| LTCUSDT | 410 | 326 | 84 | 3.9 |
| TONUSDT | 384 | 231 | 153 | 1.5 |
| TRXUSDT | 245 | 192 | 53 | 3.6 |

- Base 20 symbols: 10,959 total signals
- Extra 10 symbols: 4,491 total signals (**+41% supply**)

## Key Findings

### 1. Universe Expansion is a Free Upgrade
B-30 vs B-20 (same 2 slots):
- **+58% return** (+4,985% vs +3,160%)
- **+19% Sharpe** (7.09 vs 5.98)
- **+48% trade frequency** (8.06 vs 5.44/day)
- **Same drawdown** (9.7% vs 9.6%)
- **Higher avg PnL/trade** (+1.28% vs +1.21%)

More symbols → more signal candidates → portfolio picks the best → better quality.

### 2. Diminishing Returns from More Slots
| Upgrade | Return Δ | DD Δ | Sharpe Δ |
|---------|----------|------|----------|
| B-20 → B-30 (2s) | +58% | +0.1pp | +19% |
| B-30 → B-30-3s | +14% | +3.8pp | -1.1% |
| B-30-3s → B-30-4s | +6% | +0.4pp | -0.6% |

Going from 2→3→4 slots adds return but at the cost of drawdown and no Sharpe improvement.

### 3. Consistency Across All Market Regimes
All 4 variants are profitable in all 3 years. Y3 (bear) shows B-30
outperforming B-20 by +56% in a harder market — wider universe helps.

### 4. Short WR Plateau at ~65%
Shorts hit ceiling at 65% WR regardless of universe size or slot count.
Long WR scales with more slots (67.2% → 68.9%).

## Recommendation

**B-30 (30 symbols, 2 slots)** is the optimal upgrade:

- Highest Sharpe (7.09) — best risk-adjusted
- Highest Calmar (513) — best return-per-drawdown
- Drawdown contained (9.7%) — no additional risk
- +48% trade frequency solves the frequency concern
- Simple upgrade — just expand `PORTFOLIO_SYMBOLS` in `.env`

B-30-3s and B-30-4s offer more absolute return but push drawdown to 13.5-13.9%
without improving risk-adjusted metrics. Not worth the extra risk.

## Next Steps

1. Apply to live: expand `PORTFOLIO_SYMBOLS` from 20 → 30 in `.env`
2. Apply conf≥3: `RB_MIN_CONFIDENCE=3` in `.env`
3. Keep 2 slots: `MAX_CONCURRENT_POSITIONS=2`
