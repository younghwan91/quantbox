# Alpha Search R12: Trailing Stop Innovation — Maximize Runner R:R (6yr)

| Field | Value |
|-------|-------|
| Commit | `0b7936d` |
| Date | 2026-02-21 |
| Baseline | U-RR+VWS (R11 winner, Calmar 1004) |
| Data | 2190d (6yr), 50 symbols, 15m candles |
| Script | `scripts/alpha_search_r12.py` |
| Runtime | 112.2s |

## Goal

U-RR+VWS (R11 winner) achieved Calmar 1004, DD 6.4%, PF 1.66, Sharpe 5.77, but 
R:R dropped from 1.91 (base) to 1.76 due to wider VWMA stops.

**Target:** Recover R:R > 1.91 through trailing stop innovations while maintaining 
DD ≤ 7%, Calmar ≥ 800.

## Part A: Runner Diagnostic (Critical Finding)

Before testing innovations, we diagnosed the baseline runner behavior:

| Metric | Value | Implication |
|--------|-------|-------------|
| Total trades | 16,584 | |
| Pre-TP1 exits | 11,716 (70.6%) | Stops/time/trail — mostly losses |
| TP1 exits | 2,434 (14.7%) | 50% partial at target — all wins |
| Runner exits | 2,434 (14.7%) | 50% trailing post-TP1 |
| **Runner WR** | **99.9%** | Almost all runners profit |
| Runner avg PnL | 14.354% | Very high (leveraged) |
| Runner median PnL | 11.212% | Strong even at median |
| **Runner avg bars** | **6.9** | Exits in ~1.7 hours! |
| **Time exits** | **0** | Never hits max hold |
| Near break-even | 0.0% | No runners stall at entry |

### Why R:R Is Capped by Runner Composition

R:R = avg(win%) / avg(|loss%|). The denominator (losses) is entirely determined by 
the **stop mechanism** in pre-TP1 exits, which is identical across all trailing 
variants. The numerator (wins) includes TP1 exits, runners, and pre-TP1 positive 
exits. Runners are only **14.7% of total trades**.

Even making runners 10% more profitable changes overall R:R by ~0.01. This is the 
mathematical ceiling for trailing-only improvements.

## Part B: 24 Trailing Stop Variants

All variants inherit U-RR+VWS baseline (rr_size=True, vwma_stop=wide×0.50).

### Results Table

| Variant | Description | Ret% | DD% | Sharpe | Calmar | WR% | PF | R:R | #Tr |
|---------|-------------|-----:|----:|-------:|-------:|----:|---:|----:|----:|
| **A-Base** | U-RR+VWS baseline | 6,427% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,584 |
| **B-T07 ▲●** | Tighter trail ATR×0.7 | **6,528%** | 6.4% | 5.77 | **1022** | 58.2 | **1.67** | 1.76 | 16,774 |
| C-T15 | Looser trail ATR×1.5 | 6,190% | 6.6% | 5.59 | 944 | 58.1 | 1.66 | 1.77 | 16,196 |
| D-T20 | Very loose ATR×2.0 | 5,875% | 6.6% | 5.46 | 893 | 57.9 | 1.64 | 1.77 | 15,796 |
| E-SU30 | Step-up 30% at 2× | 6,427% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,584 |
| F-SU50 | Step-up 50% at 2× | 6,427% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,584 |
| G-SU50E | Step-up 50% at 1.5× | 6,427% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,584 |
| H-MP15 | Multi-phase 1.5→0.5 | 6,275% | 6.5% | 5.65 | 969 | 58.1 | 1.66 | 1.77 | 16,354 |
| I-MP20 ⬆ | Multi-phase 2.0→0.3 | 5,999% | 6.5% | 5.52 | 919 | 57.9 | 1.65 | **1.78** | 15,876 |
| J-MPrg | Progressive 2.0→0.3 | 6,371% | 6.4% | 5.71 | 991 | 58.1 | 1.66 | 1.77 | 16,486 |
| K-TtLn | Time linear 1.5→0.3 | 6,222% | 6.4% | 5.65 | 965 | 58.0 | 1.65 | 1.76 | 16,370 |
| L-TtSp | Time step →0.5 at 25b | 6,432% | 6.4% | 5.77 | 1005 | 58.2 | 1.66 | 1.76 | 16,589 |
| M-VwF5 | VWMA floor ×0.5 | 6,429% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,585 |
| N-VwF3 | VWMA floor ×0.3 | 6,427% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,584 |
| O-PcSt ◆ | POC static floor ×0.3 | 6,389% | 6.4% | **5.79** | 1004 | 58.1 | 1.66 | 1.76 | 16,620 |
| **P-PcDy ◆●** | POC dynamic floor ×0.3 | 6,404% | 6.4% | **5.79** | **1006** | 58.1 | 1.66 | 1.76 | 16,623 |
| Q-GAm | Gain-adaptive 0.5+0.5× | 6,357% | 6.4% | 5.73 | 993 | 58.2 | 1.66 | 1.77 | 16,476 |
| R-GAa | Gain-adaptive aggressive | 6,052% | 6.7% | 5.60 | 902 | 57.9 | 1.64 | 1.77 | 16,194 |
| S-MH4 | Max hold ×4.0 | 6,427% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,584 |
| T-MH5 | Max hold ×5.0 | 6,427% | 6.4% | 5.77 | 1004 | 58.2 | 1.66 | 1.76 | 16,584 |
| U-T15S | Loose+step-up | 6,190% | 6.6% | 5.59 | 944 | 58.1 | 1.66 | 1.77 | 16,196 |
| V-MPM4 | Multi-phase+MH4 | 6,276% | 6.5% | 5.65 | 969 | 58.1 | 1.66 | 1.77 | 16,353 |
| W-VwGA | VWMA floor+GA | 6,358% | 6.4% | 5.73 | 993 | 58.2 | 1.66 | 1.77 | 16,476 |
| X-Full | All combined | 6,200% | 6.6% | 5.59 | 943 | 58.1 | 1.66 | 1.77 | 16,199 |

### Runner Comparison

| Variant | R:R | RunWR | RunAvg% | RunMed% | AvgBars | Trail# | Time# |
|---------|----:|------:|--------:|--------:|--------:|-------:|------:|
| A-Base | 1.76 | 99.9% | 14.354 | 11.212 | 6.9 | 2,434 | 0 |
| B-T07 | 1.76 | 99.8% | 14.339 | 11.392 | **4.6** | 2,457 | 0 |
| I-MP20 | **1.78** | 99.1% | **14.533** | 10.371 | 17.0 | 2,292 | 24 |
| D-T20 | 1.77 | 99.1% | 14.446 | 10.328 | 18.4 | 2,254 | 45 |

## Key Findings

### 1. Runner Trailing Is Already Near-Optimal (The Big Finding)

The runner phase (post-TP1 50% trailing) is **already a 99.9% WR profit engine** 
with avg 6.9-bar hold time. No runners hit the time limit, none stall at break-even. 
The trailing stop improvements yield only **+0.02 R:R at best** (1.76→1.78) because:

- Runners are only **14.7% of total trades** — limited leverage on overall R:R
- Stop-outs (70.6% of trades) determine the loss side, which is unchanged
- **Mathematical ceiling:** Making the 14.7% runner exits 10% more profitable 
  shifts overall R:R by ~0.01

### 2. Dead Variants (Zero Effect)

| Variant | Why Zero Effect |
|---------|----------------|
| Step-up (E,F,G) | Runners exit in 6.9 bars — never reach the 2× or 1.5× trigger |
| Max hold extension (S,T) | 0 time exits — extending max hold changes nothing |
| VWMA floor (N) | VWMA floor × 0.3 is always below the existing ATR trail |

### 3. B-T07 Is the Practical Winner (Tighter Trail, ATR×0.7)

Counterintuitively, a **tighter** trail beats a looser one:

| Metric | A-Base | B-T07 | Delta |
|--------|--------|-------|-------|
| Return | 6,427% | **6,528%** | +1.6% |
| Calmar | 1004 | **1022** | +1.8% |
| PF | 1.66 | **1.67** | +0.6% |
| Trades | 16,584 | **16,774** | +190 |
| Runner bars | 6.9 | **4.6** | −33% |

**Why:** Tighter trail exits runners 33% faster (4.6 vs 6.9 bars), freeing slots 
for new entries. More slot turnover = more opportunities = higher total return. Each 
runner makes marginally less but the system makes more trades.

### 4. POC Dynamic Floor (P-PcDy) — Best Sharpe/Calmar Innovation

| Metric | A-Base | P-PcDy |
|--------|--------|--------|
| Sharpe | 5.77 | **5.79** |
| Calmar | 1004.0 | **1006.1** |
| DD | 6.4% | 6.4% |

Marginal but real: POC as dynamic trail floor provides volume-weighted support.

### 5. R:R Improvement Requires Different Approach

Trailing stop innovation is a **dead end for R:R**. The R:R bottleneck is:

- **Stop side:** 70.6% of trades exit as pre-TP1 stops/time/trail losses. The avg 
  loss size is set by `stop_mult=0.382`, which is constant.
- **To improve R:R, the focus should shift to:**
  1. **Partial fraction tuning:** Move from 50/50 to 30/70 (TP1/runner) — give the 
     99.9% WR runner more capital
  2. **Stop structure:** Tighter stops reduce avg loss → higher R:R (at cost of WR)
  3. **Entry quality gates:** Only take entries with higher implied R:R (filter more)

## Recommendations

1. **Adopt B-T07 (trail ATR×0.7)** as refinement — Calmar 1022, +1.6% return, 
   no R:R regression. Minimal code change: `trail_atr: float = 0.7` in production.
2. **R13 direction:** Test partial_frac variations (30/70, 40/60, 20/80) to give 
   the high-WR runner more capital. This directly impacts R:R because it increases 
   avg win (more runner weight) without changing avg loss.
3. **Optional:** Consider POC dynamic floor (P-PcDy) for marginal Sharpe boost.

## Status

- ★ = better DD  ◆ = better Sharpe  ▲ = better Return  ⬆ = better R:R  ● = better Calmar
- **Best R:R:** I-MP20 (1.78) — but worse Calmar (919) and Return (5,999%)
- **Best practical:** B-T07 — Calmar 1022, Return 6,528%, PF 1.67
