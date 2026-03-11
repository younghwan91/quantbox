# Parity Backtest Report: v12.5.3 Live Session (Feb 22-26, 2026)

**구현:** 해당없음 (Lookahead 버그 수정 문서 — v12.5.3)
## Summary

**Root Cause Found:** 4h MTF lookahead bias in `apply_mtf_confirmation()`.  
Backtest used final close of developing 4h bar (future data); live uses real-time developing price.

| Metric | Backtest (lookahead) | Backtest (fixed) | Live (v12.5.3) |
|--------|---------------------|------------------|----------------|
| Trades | 12 | **28** | 25 |
| Win Rate | 83.3% | **42.9%** | 36.0% |
| SL exits | 2 | **16** | 16 |
| TP/Runner exits | 10 | 12 | 9 |
| Net PnL (unlev) | +19.0% | **+8.66%** | +0.76% |
| Net PnL (4x lev) | +76.0% | **+34.6%** | +0.76% |

## Bug Mechanism

```
15m bar i=84 → 4h bar j = 84 // 16 = 5  (DEVELOPING bar, only 1h into 4h period)
close_4h[5] = 539.92  ← this is the FINAL close (known only at bar end)
SMA_4h = 566.97
539.92 < 566.97 → LONG BLOCKED (incorrectly — at bar 84, the 4h bar was young, price ~560)

Fix: j = j_raw - 1 = 4  ← last COMPLETED 4h bar
close_4h[4] = historical value ← no lookahead
```

**Impact on BTC crash period (Feb 22-24):**
- Old (lookahead): 4h bars that eventually closed down blocked LONGs even at bar start → unrealistically few entries → inflated WR
- Fixed: Uses previous completed bar → allows same entries as live → realistic SL churn during crash

## Remaining PnL Gap (+8.66% vs +0.76%)

| Factor | Estimated Impact | Notes |
|--------|-----------------|-------|
| Slippage + LimitChaser imperfect fills | -2~3% | Backtest assumes instant fill at close |
| Funding fees (8h resets) | -0.5~1% | 25 trades × avg 1.7h hold × 0.01%/8h |
| Pre-existing ETHUSDT trade | -1~2% | Live had carry-over position from previous session |
| Symbol selection timing | -1~2% | Backtest scans all 50 symbols simultaneously; live scans sequentially |
| Partial close execution lag | -0.5~1% | TP1 50% close has market impact in live |
| **Total estimated** | **-5~9%** | Accounts for most of the 8% gap |

## Files Modified

1. **`quantbox/strategy/rubberband/signals.py`** — `apply_mtf_confirmation()`: `j = j_raw - 1` instead of `j = j_raw`
2. **`scripts/backtest_live_period.py`** — `_find_4h_bar_idx()`: returns last completed bar
3. **`research/active/r12_r20/r15_adaptive_leverage.py`** — `apply_mtf_filter()`: same fix
4. **`research/active/r12_r20/r16_wr_calibrated_leverage.py`** — `apply_mtf_filter()`: same fix

## Tests Added

- `tests/test_mtf_lookahead.py` — 3 regression tests:
  - `test_mid_bar_uses_previous_completed_bar`: Signal inside developing 4h bar uses previous completed bar
  - `test_4h_boundary_also_uses_previous`: Signal at exact 4h boundary still uses previous completed bar
  - `test_trend_reversal_no_leakage`: 4h bar that reverses doesn't leak into earlier 15m signals

## Action Items

- [ ] Re-run 6-year backtest (r15, r16) with lookahead fix — ALL prior MTF-filtered results are inflated
- [x] Fix `apply_mtf_confirmation()` in signals.py
- [x] Fix parity backtest script
- [x] Fix r15 and r16 research scripts
- [x] Add regression tests
- [x] Update agents.md + copilot-instructions.md with Incident #2

## Live Session Context

- **Version:** v12.5.3 (commit `ca2ceba`)
- **Period:** 2026-02-22 16:32 → 2026-02-26 08:40 UTC
- **Balance:** $1,957.76
- **Market:** BTC crashed from $98k to $82k during this period
- **Features NOT present in v12.5.3:** Climax volume filter, ATR dynamic stop (added in v12.6)
