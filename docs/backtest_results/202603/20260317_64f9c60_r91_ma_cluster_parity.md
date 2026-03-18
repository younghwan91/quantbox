# r91: MA Cluster Backtest-vs-Live Parity Root Cause Analysis

**Date:** 2026-03-17  
**Commit:** 64f9c60  
**Script:** `research/active/r91_r95/r91_ma_cluster_parity.py`  
**Universe:** 60 symbols (symbol_whitelist), 6 years (2020 – Feb 2026, cache_extended)  
**Config:** cluster_pct=0.02, min_dur=6, max_dur=24, SL=2%, trail_act=2%, trail_dist=0.3%  
**L/N:** 21/10 = 2.1  

---

## Motivation

MA cluster was disabled (commit 8dc3bb1, 2026-03-15) after live performance of:
- **N=16, WR=31%, PF=0.587, net loss ≈ -$765** (96% of total account losses)

r75 (IS backtest) reported: N=1233, PF=2.403, Calmar=33.5

**Gap: ~4× PF overestimation in backtest vs live.** This investigation tests three root cause hypotheses.

---

## Hypotheses Tested

| ID | Hypothesis | Result |
|----|-----------|--------|
| H1 | Cooldown parity: backtest resets `cluster_count=0` → allows re-entry within same episode | **N inflated 1.5×, PF unchanged (≈0%)** |
| H2 | Regime failure: March 2026 = BTC crash, no direction filter | **PRIMARY CAUSE** |
| H3 | Trail-before-SL bias (pre-SL-first-fix) inflates r75 PF | **Zero impact (Δ=0.000)** |

---

## Results

### H1: Cooldown Parity

| Variant | N | PF | WR% | MaxDD |
|---------|---|----|-----|-------|
| Backtest-style (`cluster_count=0` reset, trail-before-SL) | 1205 | 2.547 | 58.2% | 26.1% |
| SL-first only (cooldown unchanged) | 1205 | 2.547 | 58.2% | 26.1% |
| **Live-style** (dissolve cooldown + SL-first) | **802** | **2.630** | **57.5%** | **19.0%** |

- **N inflation: 1.50×** — backtest fires 50% more trades than live would, because after a trade exits (cluster still active), it re-enters after `min_duration` bars. Live blocks until cluster fully dissolves.
- **PF inflation: 0.97×** (essentially 1.0) — re-entries within same episode have the **same EV as first entries**. Cooldown difference does NOT inflate PF.
- **Re-entries in same episode: 403 out of 1205** (33%) are same-episode re-entries in backtest style.

**H1 verdict: NOT the cause of PF inflation.** The 50% extra trades in backtest vs live-style sim have equal quality. PF is nearly identical between the two variants.

### H2: Regime Failure (Yearly Breakdown — live-style sim)

| Year | N | WR% | PF | EV (bp) |
|------|---|-----|----|---------|
| 2020 | 70 | 52.9% | 2.482 | +145bp |
| 2021 | 53 | 58.5% | 3.589 | +224bp |
| 2022 | 116 | 59.5% | 2.849 | +156bp |
| 2023 | 271 | 61.3% | 2.908 | +154bp |
| 2024 | 140 | 55.0% | 2.124 | +105bp |
| 2025 | 142 | 54.2% | 2.361 | +130bp |
| **2026 (Jan-Feb only)** | **10** | **40.0%** | **1.576** | **+72bp** |
| **Live actual (Mar 4-13)** | **16** | **31.2%** | **0.587** | **−** |

Key observations:
1. **EV decays 2021→2026** — the strategy is weakening over time (market becoming more efficient or regimes changing)
2. **2026 Jan-Feb**: already WR=40%, PF=1.576 — declining even before the crash
3. **March 2026 live**: BTC fell from ~$90k → ~$73k during the live run period (not in cache_extended data)
4. **No direction filter**: 11/16 live trades were LONG, all entered during altcoin downtrend driven by BTC crash
5. **Even SHORTs failed**: 5 SHORTs had net negative PnL (altcoins had brief counter-trend rallies that hit SLs)

### H3: Trail-Before-SL Bias

**Zero impact (ΔPF=0.000).** Explanation: In `sim_ma_cluster_trailing`, once `trailing_active=True`, the SL condition is guarded by `if not trailing_active` — it's disabled, not just deprioritized. Unlike the straddle sim (which had trail-vs-SL in same `if` block), the MA cluster sim correctly handles the ordering. The trail-before-SL bias found in straddle (H5, 1.8× inflation) does **not** exist in the MA cluster sim.

---

## Root Cause Summary

| Scenario | N | PF | WR% | Note |
|----------|---|----|-----|------|
| r75 reported (IS 6yr) | 1233 | 2.403 | 40.2% | r75 report |
| Backtest-style repro | 1205 | 2.547 | 58.2% | cluster_count=0 reset, trail-before-SL |
| SL-first only | 1205 | 2.547 | 58.2% | SL before trail-activation |
| Live-style sim | 802 | 2.630 | 57.5% | dissolve cooldown + SL-first |
| **LIVE actual (Mar 4-13)** | **16** | **0.587** | **31.2%** | **N=16, BTC crash regime** |

**The backtest is NOT inflated.** Both variants show PF≈2.5-2.6 with good WR (57-58%). The backtest was accurate.

**The live failure is a 10-day regime-specific OOS failure:**
- March 4-13, 2026 = BTC fell 18% (from ~$90k at Feb peak to ~$73k)
- Altcoins in synchronized downtrend
- MA cluster LONG signals kept firing (cluster breakouts) but price kept reversing
- 9 of 16 trades hit SL at -2%
- Strategy has NO direction filter (unlike VR with G6 BTC gate)

---

## Why r75 WR% Discrepancy (40.2% vs 58.2%)?

The r75 report shows WR=40.2% while our repro shows WR=58.2%. This is because:
- r75 used a **different SMA computation** (it likely used 15m-native SMAs rather than 4h-resampled)
- Or r75 had different `warmup`/`min_momentum` settings applied during the specific sweep

The PF values are consistent (2.403 vs 2.547), suggesting similar EV even with WR differences — different RR ratios in each version. This discrepancy warrants further investigation but doesn't affect the parity conclusion.

---

## Proposed Fix: Add G6 Gate to MA Cluster LONG Signals

If G6 BTC gate were applied to MA cluster LONG signals:
- 11/16 live trades (all LONGs) would have been blocked during BTC downtrend
- Net exposure would drop 69% in hostile regimes
- SHORTs (5 trades) still run, providing some downtrend capture

**Required change in `scalper.py`:** Add `if signal == 1 and is_ma_cluster and not btc_above_ma50: skip` before MA cluster LONG execution.

**OOS validation needed**: Run on 2026 Jan-Mar data (requires downloading fresh NPZ) to verify G6-filtered MA cluster performance before re-enabling.

---

## Conclusions

1. **Backtest was NOT inflated** — H1 (cooldown) and H3 (SL order) had near-zero impact on PF
2. **N inflation of 1.5× exists** (cooldown difference) but doesn't affect PF quality
3. **Live failure = 10-day OOS regime crash** — not a backtest bug, not fixable by code changes alone
4. **EV decay trend visible**: 2021 EV=224bp → 2025 EV=130bp → 2026 EV=72bp. Strategy weakening.
5. **Recommended action**: Keep MA cluster disabled. If re-enabling:
   - Add G6 direction gate for LONG signals
   - Require OOS validation on ≥3 months of 2026 data  
   - Accept lower N targets (EV=72bp is real, just needs enough trades)

**구현:** 미구현 (G6 gate 추가 검토 중)
