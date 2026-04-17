# r117 Donchian Backtest-Live Parity Audit

**Commit:** 03d21e4
**Date:** 2026-04-17 14:16 UTC
**Source:** data/journal.sqlite (live trades) vs sim_donchian_trailing (backtest)
**Scope:** All Donchian live trades to date (N=9).

## 1. Executive Summary

**Recommendation: PAUSE_DONCHIAN**

- avg(delta_pnl_pct)=-15.329% exceeds 2.0% threshold.
- worst-3 live trades avg delta -20.159% is below -3.0% threshold.

## 2. Aggregate Comparison

| Metric | Live | Sim (matched) | Delta |
|--------|------|---------------|-------|
| N | 9 | 4 | — |
| PF (USD) | 0.75 | None | — |
| Avg PnL % | 0.035 | 16.495 | -15.329 |
| Worst-3 avg delta % | — | — | -20.159 |

### Classification counts

- IDENTICAL (|delta| < 0.3%): 0
- SAME_SIGN_DIFFERENT_MAG: 3
- OPPOSITE_SIGN: 1
- NO_SIM_SIGNAL (live fired, sim did not): 5
- OUT_OF_CACHE (live entry after cache window end): 0

## 3. Per-Trade Detail

| Symbol | Entry | Dir | Live PnL% | Live USD | Live reason | Sim PnL% | Delta% | Class |
|--------|-------|-----|-----------|----------|-------------|----------|--------|-------|
| YFIUSDT | 2026-04-15T20:00 | LONG | -1.531 | $-10.82 | server_sl | — | — | NO_SIM_SIGNAL |
| EGLDUSDT | 2026-04-16T00:00 | LONG | 1.606 | $8.12 | server_trail | — | — | NO_SIM_SIGNAL |
| TRBUSDT | 2026-04-16T04:00 | LONG | 1.286 | $8.17 | server_trail | 2.122 | -0.836 | SAME_SIGN_DIFFERENT_MAG |
| FILUSDT | 2026-04-16T16:00 | LONG | -1.479 | $-10.71 | server_sl | — | — | NO_SIM_SIGNAL |
| SUSHIUSDT | 2026-04-16T16:00 | LONG | -1.558 | $-11.28 | server_sl | — | — | NO_SIM_SIGNAL |
| UNIUSDT | 2026-04-16T20:00 | LONG | -1.387 | $-7.87 | server_sl | — | — | NO_SIM_SIGNAL |
| TRBUSDT | 2026-04-17T00:00 | LONG | 1.885 | $7.16 | server_trail | 2.842 | -0.956 | SAME_SIGN_DIFFERENT_MAG |
| KSMUSDT | 2026-04-17T08:00 | LONG | -1.74 | $-7.25 | server_sl | 24.361 | -26.101 | OPPOSITE_SIGN |
| BANDUSDT | 2026-04-17T08:00 | LONG | 3.236 | $12.53 | server_trail | 36.657 | -33.421 | SAME_SIGN_DIFFERENT_MAG |

## 4. Methodology Notes

- Sim trades matched to live by entry_bar index within ±1 4h-bar tolerance (accounts for live-vs-sim bar-close timing micro-skew).
- Live entry_time beyond cache WINDOW_END (2026-04-12T16:18 UTC) cannot be replayed without cache extension — flagged OUT_OF_CACHE.
- Sim uses `sim_donchian_trailing` with cfg values from VolStraddleConfig (no hardcoded params).
- Pause thresholds: |avg_delta|>2.0% AND worst-3 avg < -3.0% (both conditions must fire).

## 5. Follow-ups

- Regardless of parity finding, live Donchian N=9 is too small for statistical confidence. Accumulate 20+ trades before drawing hard conclusions.
- If recommendation is KEEP_RUNNING, revisit audit when live N ≥ 20.