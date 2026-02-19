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
