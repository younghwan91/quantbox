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
