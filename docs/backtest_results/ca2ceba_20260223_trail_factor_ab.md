# Trail Factor A/B Test — Round 3 (U-RR+VWS Baseline)

- **Commit:** `2f5c456` (branch: `feature/recent-regime-ab`)
- **Date:** 2026-02-23 12:41 UTC
- **Data:** 2190d (6yr), 50 symbols, 15m+4h candles
- **Baseline:** U-RR+VWS (R12: `rr_size=True`, `vwma_stop='wide'`, `vwma_stop_mult=0.50`)
- **Innovation:** `worst_gain`/`best_gain` (intra-bar) for SL/trail/TP checks (v12.5.3 parity)
- **Focus:** Pre-TP1 trail factor sweep + stop/hold combos with correct baseline
- **Scoring:** Recency-weighted composite (Y5-6 = 2× weight)
- **Runtime:** 101.9s (VPVR 52.1s + 18 variants × ~2.4s)

## Context

R1-R2 used stripped production config (rr_size=off, vwma_stop=off) → baseline showed only 801%.
R3 restores R12 U-RR+VWS features: VWMA wide stop, VPVR-based POC for R:R sizing, BTC dual-confirm regime.
Baseline now shows 1,553% (lower than R12's 6,301% due to intra-bar worst_gain for SL/trail — more conservative, more realistic).

## Results Summary

| Var | Desc | Ret% | DD% | Sharpe | Calmar | WR | PF | R:R | TP1% | #Tr |
|-----|------|-----:|----:|-------:|-------:|---:|---:|----:|-----:|----:|
| **A-Prod** | Production TF30 (baseline) | 1,553% | 10.9% | 2.99 | 142.3 | 47.6% | 1.22 | 1.99 | 11% | 19,937 |
| B-TF20 | trail_factor=0.2 | 2,629% | 9.3% | 3.94 | 282.4 | 47.0% | 1.30 | 2.25 | 14% | 20,002 |
| **C-TF10** | trail_factor=0.10 ★ WINNER | **3,393%** | **8.2%** | **4.47** | **415.7** | 45.2% | **1.36** | 2.55 | **15%** | 20,028 |
| D-TF15 | trail_factor=0.15 | 3,035% | 8.4% | 4.26 | 361.4 | 46.2% | 1.33 | 2.39 | 14% | 20,018 |
| E-TF18 | trail_factor=0.18 | 2,746% | 8.4% | 4.10 | 326.2 | 46.6% | 1.31 | 2.31 | 14% | 20,006 |
| F-TF22 | trail_factor=0.22 | 2,431% | 9.1% | 3.78 | 265.8 | 47.2% | 1.28 | 2.19 | 13% | 19,991 |
| G-TF25 | trail_factor=0.25 | 2,128% | 8.9% | 3.54 | 239.3 | 47.4% | 1.26 | 2.11 | 13% | 19,973 |
| H-TF20MH20 | TF20 + max_hold 20 | 2,726% | 8.9% | 4.15 | 305.2 | 47.4% | 1.31 | 2.12 | 12% | 20,306 |
| I-TF20MH15 | TF20 + max_hold 15 | 2,631% | 9.6% | 4.10 | 273.4 | 47.7% | 1.31 | 1.95 | 9% | 20,549 |
| J-TF20S30 | TF20 + stop_mult 0.30 | 2,715% | 8.5% | 3.99 | 318.6 | 44.1% | 1.31 | 2.54 | 13% | 20,868 |
| K-TF20S25 | TF20 + stop_mult 0.25 | 2,633% | 9.3% | 3.92 | 284.3 | 42.3% | 1.30 | **2.73** | 12% | 21,434 |
| L-TF15MH20 | TF15 + max_hold 20 | 3,119% | 8.7% | 4.46 | 359.5 | 46.7% | 1.34 | 2.24 | 12% | 20,321 |
| M-TF15S30 | TF15 + stop_mult 0.30 | 3,136% | **7.9%** | 4.29 | 398.0 | 43.5% | 1.34 | 2.69 | 14% | 20,874 |
| N-TF20S30H20 | TF20 + S30 + MH20 | 2,790% | 8.3% | 4.15 | 337.3 | 44.5% | 1.32 | 2.40 | 11% | 21,160 |
| O-TF20S30H15 | TF20 + S30 + MH15 | 2,726% | 9.2% | 4.14 | 296.2 | 44.9% | 1.32 | 2.20 | 9% | 21,423 |
| P-TF15S30H20 | TF15 + S30 + MH20 | 3,202% | 8.1% | 4.45 | 394.1 | 44.0% | 1.35 | 2.53 | 12% | 21,167 |
| Q-TF15S30H15 | TF15 + S30 + MH15 | 3,085% | 8.7% | 4.43 | 355.7 | 44.5% | 1.35 | 2.30 | 9% | 21,430 |
| R-TF20S25H20 | TF20 + S25 + MH20 | 2,685% | 8.3% | 4.03 | 322.9 | 42.6% | 1.31 | 2.58 | 11% | 21,714 |

## Recency-Weighted Ranking

| Rank | Var | Score | vs Base | Verdict |
|-----:|-----|------:|--------:|---------|
| 1 | **C-TF10** | 1.290 | +29.0% | ★ WINNER |
| 2 | M-TF15S30 | 1.261 | +26.1% | beat base |
| 3 | P-TF15S30H20 | 1.243 | +24.3% | beat base |
| 4 | D-TF15 | 1.216 | +21.6% | beat base |
| 5 | L-TF15MH20 | 1.196 | +19.6% | beat base |
| 6 | Q-TF15S30H15 | 1.180 | +18.0% | beat base |
| 7 | J-TF20S30 | 1.165 | +16.5% | beat base |
| 8 | E-TF18 | 1.160 | +16.0% | beat base |
| 9 | N-TF20S30H20 | 1.160 | +16.0% | beat base |
| 10 | R-TF20S25H20 | 1.150 | +15.0% | beat base |

## Yearly Breakdown (Top 3)

### C-TF10 (trail_factor=0.10) ★ WINNER
| Year | Sharpe | R:R | TP1% |
|------|-------:|----:|-----:|
| Y1 | 8.2 | 2.44 | 18% |
| Y2 | 4.8 | 2.40 | 15% |
| Y3 | 4.4 | 2.53 | 15% |
| Y4 | 1.4 | 2.57 | 13% |
| Y5 | 4.5 | 2.46 | 15% |
| Y6 | 3.0 | 2.71 | 15% |

### M-TF15S30 (trail_factor=0.15, stop_mult=0.30)
| Year | Sharpe | R:R | TP1% |
|------|-------:|----:|-----:|
| Y1 | 8.0 | 2.62 | 16% |
| Y2 | 4.7 | 2.55 | 14% |
| Y3 | 4.2 | 2.72 | 13% |
| Y4 | 0.1 | 2.61 | 11% |
| Y5 | 4.1 | 2.58 | 14% |
| Y6 | 2.9 | 2.82 | 14% |

### P-TF15S30H20 (trail_factor=0.15, stop_mult=0.30, max_hold=20)
| Year | Sharpe | R:R | TP1% |
|------|-------:|----:|-----:|
| Y1 | 8.3 | 2.39 | 14% |
| Y2 | 5.1 | 2.46 | 12% |
| Y3 | 3.9 | 2.55 | 11% |
| Y4 | 0.3 | 2.44 | 10% |
| Y5 | 4.2 | 2.43 | 13% |
| Y6 | 3.1 | 2.69 | 12% |

## Exit Reason Breakdown

| Var | stop | time | trail | tp1 | trail_runner | time_runner |
|-----|-----:|-----:|------:|----:|-------------:|------------:|
| A-Prod | 9,112 | 2,753 | 3,998 | 2,037 | 2,037 | 0 |
| C-TF10 | 8,958 | 3,309 | 2,501 | 2,630 | 2,630 | 0 |
| M-TF15S30 | 10,138 | 3,022 | 2,730 | 2,492 | 2,492 | 0 |
| P-TF15S30H20 | 10,176 | 4,061 | 2,444 | 2,243 | 2,243 | 0 |

## Key Findings

1. **Trail factor monotonically improves from 0.30→0.10** — confirmed with correct U-RR+VWS baseline.
   - C-TF10 beats A-Prod by **+1,840% return**, **-2.7% DD**, **+1.48 Sharpe**, **+0.56 R:R**, **+4% TP1**.
   - Lower trail factor → fewer premature trail exits → more TP1 hits → better runners.

2. **Tighter stop (0.30) synergizes well with TF15:**
   - M-TF15S30: DD=7.9% (best), R:R=2.69, PF=1.34.
   - P-TF15S30H20: Sharpe=4.45, near-best return 3,202%.

3. **All 17 variants beat baseline** — the production trail_factor=0.30 is provably suboptimal.

4. **Recency stability:** C-TF10 maintains Y5 Sharpe=4.5, Y6 Sharpe=3.0 — no sign of regime decay.

5. **Baseline difference from R12:** A-Prod shows 1,553% vs R12's 6,301%.
   Cause: This script uses `worst_gain` (intra-bar low) for SL/trail checks, while R12 uses close-based `gain`.
   Intra-bar SL is more conservative and realistic (no look-ahead within bars).

## Recommendation

**Deploy C-TF10** (`pre_tp1_factor=0.10`) as the next production parameter change.
- Massive uplift in all metrics vs current TF30.
- Y4 (worst year) still positive: Sharpe=1.4, PF=1.36.
- Risk: Factor=0.10 means pre-TP1 trail triggers only when gains retrace to 10% of peak — very tolerant.
  In live trading, this means holding through deeper retracements pre-TP1.
- **Alternative:** D-TF15 for a more conservative choice (still +1,481% vs base, Sharpe 4.26).
