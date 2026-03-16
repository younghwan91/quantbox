<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-16 -->

# VolStraddle Strategy (VR Squeeze) — Agent Guide

**Status**: Production (Live Trading) | **Leverage**: 30x (Half-Kelly) | **Edge (r73, 5.67yr)**: PF=2.002, Calmar=1695.6

## Overview

VolStraddle is the primary production strategy for QuantBox, detecting volatility compression via Volatility Ratio (VR) and entering on squeeze release. Two entry modes:
- **D_mom8 (Directional)**: Single-sided LONG/SHORT when VR < 0.50
- **S_d8 (Straddle)**: Simultaneous LONG+SHORT when VR < 0.57 for ≥3 bars (Hedge Mode, Binance)

Multi-timeframe scanning (15m/30m/1h/4h, priority 4h > 1h > 30m > 15m) with zero lookahead (completed bars only). Trailing stops preserve explosive breakout profits beyond fixed TP. All parameters read from `VolStraddleConfig` (pydantic-settings via .env).

## VR Squeeze Algorithm

### ATR-Based Volatility Compression

**Volatility Ratio** measures relative volatility (compression):

```text
VR = ATR(8) / ATR(56)
```

- **ATR(8)**: 8-bar short-term volatility (EMA-smoothed True Range as % of close)
- **ATR(56)**: 56-bar long-term volatility (baseline, less reactive)
- **Low VR** (< threshold): Compression → volatility "squeezed"; often precedes explosive move
- **Threshold D_mom8**: VR < 0.50 (config: `d_vr_threshold`)
- **Threshold S_d8**: VR < 0.57 (config: `s_vr_threshold`)

**Why ATR(8)/ATR(56)?**
- Scale-invariant: works across symbols and timeframes (BTCUSDT, ALTUSDT, etc.)
- Non-parametric: detects relative volatility change, not absolute levels
- 6-year validated: R39 A1-D50 baseline, repeated in r47, r69, r73

### D_mom8 Directional Entry (Single-Sided)

Fires when VR < 0.50:
1. Direction determined by 8-bar momentum: `(close[i] / close[i-8] - 1)`
   - Positive momentum → LONG (+1)
   - Negative momentum → SHORT (-1)
   - Near-zero (|m| < 0.0001) → default LONG
2. No additional confirmation (pure VR trigger)
3. One position per symbol (per-symbol dedup)

**Exit**:
- **Trailing Stop** (priority 1): Once unrealized gain ≥ 1.0% (trail_activate), follow peak at 0.2% distance
- **Stop Loss** (priority 2): Hard 1.0% loss (d_sl)
- **Take Profit** (priority 3): 20% hard ceiling (d_tp, safety cap, rarely hit due to trail)
- **Max Hold** (priority 4): 192 bars @ 15m = 48h real-time

**Conviction Weighting (r34)** (optional):
- Score 0–4 based on E2 (SMA proximity), C2 (BTC neutral), B3 (confluence), A3 (quality+)
- Multipliers: [0.5, 1.0, 1.5, 2.0, 2.5] × base sizing
- Default: `cw_enabled=True`, `min_conviction_score=0` (all trades welcome)

### S_d8 Straddle Entry (Dual-Leg, Hedge Mode)

Fires when VR < 0.57 for ≥3 consecutive bars:
1. Opens LONG+SHORT simultaneously in same symbol (Binance Hedge Mode)
2. Independent SL/TP per leg (e.g., "SUIUSDT:LONG" + "SUIUSDT:SHORT")
3. No direction logic (neutral, bidirectional)

**Exit**:
- **Trailing Stop** (priority 1): Once unrealized gain ≥ 1.5% per leg, follow at 0.1% distance
- **Stop Loss** (priority 2): 1.5% loss per leg (s_sl)
- **Take Profit** (priority 3): 5% per leg (s_tp, safety)
- **Max Hold** (priority 4): 192 bars @ 15m = 48h real-time
- **Straddle Pair Coordination**: If one leg fills, retry second leg (max 2 retries, 2s delay). Orphan auto-close after 4 cycles (~60min).

**Slot Allocation**:
- Each straddle pair = 2 slots (one LONG, one SHORT)
- Max 8 slots for straddle (config: `straddle_max_slots`) → max 4 pairs
- Remaining slots (10 - 8 = 2) reserved for D_mom8 directional entries

### MA Cluster Squeeze (Legacy, Currently Disabled)

**Status**: R44d V17 variant (non-overlapping with VR, ~93.7% independence). Currently disabled (`ma_cluster_enabled=False`) due to live PF=0.309 (2024/2025 OOS degrade).

**When enabled**:
- 4h-only strategy: detects SMA(20/50/100/200) cluster (spread ≤ 2%)
- Entry on breakout + momentum direction
- 21x leverage, 2% SL, 8% TP, trailing stop (activate 2%, distance 0.3%)

Not documented here (archived strategy); modify VolStraddle for edge research.

## Edge Metrics (r73 Baseline)

**Production Portfolio** (D_mom8 + S_d8 + MA cluster straddle, 5.67 years, 60 symbols):

| Metric | Value | Notes |
| --- | --- | --- |
| **PF (Profit Factor)** | 2.002 | Total profit / Total loss |
| **Calmar Ratio** | 1695.6 | Annual RoR / Max DD |
| **Sharpe Ratio** | ~5.0+ | Risk-adjusted return |
| **Win Rate** | ~43.4% | Directional strategy; many small losses |
| **Max Drawdown** | 2.6% | Peak-to-trough (backtest) |
| **Years Profitable** | 5/5 | All years profitable |
| **Sample Size (N)** | ~2,336 trades | 15m TF entries |
| **Bootstrap CI (PF)** | [1.418, 1.634] | 95% confidence interval |
| **P(PF ≥ 1.0)** | 100% | No chance of breakeven |

**Note**: r73 combines D+S+MA. D_mom8 alone: PF~1.35; S_d8 alone: PF~3.94 (infrequent, high conviction).

## Configuration Parameters

**Location**: `quantbox/strategy/vol_straddle/config.py` (VolStraddleConfig, pydantic-settings)

Read via:
```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()  # Reads .env (prefix VS_)
```

**All ~100 parameters** (condensed by category):

| Section | Key Parameters | Default | Purpose |
| --- | --- | --- | --- |
| **VR Computation** | vr_short, vr_long | 8, 56 | ATR periods (R39 validated) |
| **D_mom8** | d_vr_threshold, d_sl, d_tp | 0.50, 0.01, 0.20 | Entry trigger & exits |
| **D_mom8 Trail** | d_trail_activate, d_trail_distance | 0.010, 0.002 | Trailing stop (r69: +38% OOS) |
| **S_d8** | s_vr_threshold, s_squeeze_dur, s_sl, s_tp | 0.57, 3, 0.015, 0.05 | Straddle params |
| **S_d8 Trail** | s_trail_activate, s_trail_distance | 0.015, 0.001 | Straddle trailing (r76: tight trail) |
| **Straddle Mode** | straddle_enabled, straddle_max_slots, hedge_mode | True, 8, True | Hedge Mode configuration |
| **Server-Side Trail** | trail_server_native | False | Binance TRAILING_STOP_MARKET (r90) |
| **Momentum** | momentum_period | 8 | Direction lookback |
| **Max Hold (bars)** | max_hold_15m, max_hold_30m, max_hold_1h, max_hold_4h | 192, 96, 48, 12 | 48h per TF |
| **Multi-TF Enable** | enable_15m, enable_30m, enable_1h, enable_4h | True, True, True, True | TF toggles |
| **Warmup (bars)** | warmup_15m, warmup_30m, warmup_1h, warmup_4h | 200, 100, 50, 13 | ~50h per TF |
| **Leverage** | base_leverage, max_leverage | 30.0, 30.0 | Half-Kelly (f*/2 × N / SL) |
| **Risk** | max_concurrent, risk_per_trade | 10, 0.01 | 10 slots, 1% per trade |
| **Costs** | fee_rate, slippage_rate, slippage_mode | 0.0002, 0.0002, "flat" | 0.02% maker + slippage |
| **Conviction (CW)** | cw_enabled, min_conviction_score, cw_multipliers | True, 0, [0.5, 1.0, 1.5, 2.0, 2.5] | Sizing boost (r34) |
| **E2: SMA Proximity** | e2_enabled, e2_sma_period, e2_threshold | True, 20, 0.005 | 0.5% from SMA mean |
| **C2: BTC Neutral** | c2_enabled, c2_btc_mom_threshold | True, 0.01 | BTC momentum < 1% |
| **B3: Confluence** | b3_enabled | True | D & S overlap |
| **A3: Quality+** | a3_enabled, a3_deep_vr_threshold, a3_long_squeeze_dur | True, 0.35, 16 | Extra filters |
| **F5: TBR Gate** | f5_tbr_enabled, f5_tbr_z_threshold, f5_tbr_z_window | False, 0.5, 192 | r47: disabled (2024/25 OOS degrade) |
| **G6: BTC Trend** | btc_long_filter_enabled, btc_long_ma_period | True, 50 | Suppress LONG if BTC down |
| **Symbols** | symbol_whitelist (60 syms) | See config.py | Production universe |
| **MA Cluster** | ma_cluster_enabled | False | Disabled (archived strategy) |
| **DD Scaling** | soft_equity_scale, soft_eq_dd1–dd3 | True, [1.5%, 3%, 5%] | Size scaling on drawdown |
| **Daily Loss** | daily_loss_limit_pct | 0.05 | 5% loss → halt entries |
| **Straddle Retry** | entry_retry_count, entry_retry_delay_s | 2, 2.0 | Second leg coordination |

**CRITICAL**: Never hardcode parameters. Always read from VolStraddleConfig():
```python
# ✓ Correct
cfg = VolStraddleConfig()
VR_LN = cfg.base_leverage / cfg.max_concurrent  # 30/10 = 3.0

# ✗ Wrong
VR_LN = 3.0  # Hardcoded, breaks with .env changes
```

## Signal Functions (9 Numba JIT)

**Location**: `quantbox/strategy/vol_straddle/signals.py`

All functions are **Numba-compiled** (`@numba.jit(nopython=True, cache=True)`); invisible to coverage.py. Verified via `tests/test_signals_unit.py`.

### Core Functions

| Function | Signature | Purpose |
| --- | --- | --- |
| `compute_vr()` | `(close, high, low, short_period, long_period) -> vr_array` | Volatility Ratio = ATR(short) / ATR(long) |
| `compute_momentum()` | `(close, lookback) -> momentum_array` | Simple momentum: `close[i] / close[i-lookback] - 1` |
| `compute_squeeze_dur()` | `(vr, threshold) -> duration_array` | Count consecutive bars where VR < threshold |
| `gen_vr_signal()` | `(vr, threshold, warmup) -> signal_array` | Binary signal (0/1) where VR < threshold |
| `gen_vr_dur_signal()` | `(vr, threshold, min_dur, warmup) -> signal_array` | Signal fires after ≥ min_dur consecutive squeeze bars |
| `make_direction_momentum()` | `(sig, momentum) -> directional_signal` | Assign +1 (LONG) or -1 (SHORT) based on momentum sign |
| `resample_ohlcv()` | `(close, high, low, volume, ratio) -> (c_out, h_out, l_out, v_out)` | Resample lower TF → higher TF (completed bars only) |
| `compute_sma()` | `(close, period) -> sma_array` | Simple Moving Average (for E2 proximity) |
| `check_ma_cluster()` | `(close, sma20, sma50, sma100, sma200, momentum, cluster_pct, min_duration, in_cooldown_arr, duration_out) -> int` | MA Cluster detection; stateful (cooldown tracking); returns +1/-1/0 |

**ZERO LOOKAHEAD** guarantee:
- `gen_vr_signal()` only checks bars ≥ warmup (first signal at bar `warmup`)
- `resample_ohlcv()` uses **completed bars only**: `n_out = n_in // ratio` (truncates remainder)
- Example: 200 × 15m bars → 50 × 1h bars (last 15m bar discarded if incomplete)

### Multi-TF Orchestration

Strategy's `_check_tf_signal()` method calls these in priority order:

```text
on_bar() → priority scanning:
  1. enable_4h? → _check_tf_signal(4h_data, "4h")
  2. enable_1h? → resample_ohlcv(15m→1h) → _check_tf_signal(..., "1h")
  3. enable_30m? → resample_ohlcv(15m→30m) → _check_tf_signal(..., "30m")
  4. enable_15m? → _check_tf_signal(15m_data, "15m")

  First non-zero signal wins (return immediately)
```

**Cross-TF References**:
All resample operations use the **completed bar formula**:

```python
j = (i // ratio) - 1  # Index in higher-TF array for bar i in lower-TF
```

Example (15m → 1h, ratio=4):
- 15m bars 0–3 → 1h bar 0
- 15m bars 4–7 → 1h bar 1
- etc.

This ensures **no lookahead**: bar i's 1h close is known only after bar i's 15m close completes.

## Entry & Exit Paths

### D_mom8 Directional

**Entry**:
1. Signal fires when VR < 0.50 (threshold) at current bar close
2. Direction assigned by momentum(8) sign
3. One position per symbol (dedup check before signal generation)

**Exit Sequence** (checked each cycle via `update_position()`):
1. **Trailing Stop**: If unrealized gain > trail_activate (1.0%), trail peak at distance (0.2%)
2. **Stop Loss**: Hard 1.0% loss (d_sl)
3. **Take Profit**: 20% cap (d_tp, dead in practice—trail exits first)
4. **Max Hold**: 192 bars (48h at 15m)
5. **Session Halt**: Daily loss limit or max DD exceeded

**Example Trade**:
- Entry: SUIUSDT at $5.00, VR=0.48, momentum +0.015 → LONG signal
- Position: SUIUSDT (no colon, direction implicit in entry price/side)
- Price: $5.00 → $5.10 (gain +2.0%)
- Trail activates (unrealized +1.0%): peak=$5.10, trail at $5.08
- Price: $5.09 → trail adjusted to $5.07 (peak trailing)
- Price: $5.06 → trail hit, close with ~0.2% loss from peak

### S_d8 Straddle (Hedge Mode)

**Entry**:
1. Signal fires when VR < 0.57 for ≥3 consecutive bars
2. Opens LONG+SHORT simultaneously in Binance Hedge Mode
3. Two positions created: `SUIUSDT:LONG` + `SUIUSDT:SHORT`
4. Slot cost: 2 slots (max_concurrent=10 allows ≤4 straddle pairs)

**Position Keys**:
- **LONG leg**: `symbol:LONG` (e.g., `SUIUSDT:LONG`)
- **SHORT leg**: `symbol:SHORT` (e.g., `SUIUSDT:SHORT`)
- **API calls**: Split key: `pos_key.split(':')[0]` to get ApiSymbol

**Exit Sequence** (per leg, independent):
1. **Trailing Stop**: If unrealized gain > 1.5%, trail at 0.1%
2. **Stop Loss**: 1.5% per leg (s_sl)
3. **Take Profit**: 5% per leg (s_tp)
4. **Max Hold**: 192 bars (48h)
5. **Pair Coordination**: If one leg orphaned, auto-close after 4 cycles

**Example Trade**:
- Entry: SUIUSDT squeeze detected
- Scalper.open_position() called twice:
  - `open_position(SUIUSDT:LONG, signal=2, ...)`
  - `open_position(SUIUSDT:SHORT, signal=2, ...)`
- Each leg tracked independently in active_positions dict
- Exit: LONG at +5%, SHORT at -1.5% (SL) → close both

## Hedge Mode

### Compound Position Keys

Hedge Mode requires **compound keys** to distinguish directional legs:

```python
# Live state (per symbol)
active_positions = {
    "SUIUSDT:LONG": {"entry_price": 5.00, "notional": 500, ...},
    "SUIUSDT:SHORT": {"entry_price": 5.00, "notional": 500, ...},
}

# When querying API (need ApiSymbol, no colon)
symbol = "SUIUSDT:LONG".split(':')[0]  # → "SUIUSDT"
```

**Why Compound Keys?**
- Binance Hedge Mode allows simultaneous LONG+SHORT in same symbol
- Strategy must track both independently (different entry prices, SL/TP)
- Scalper's ReconciliationMixin syncs both legs with exchange

### Independent SL/TP per Leg

Each leg has its own bracket:
- LONG leg: entry SL at (entry_price - 1.5%), TP at (entry_price + 5%)
- SHORT leg: entry SL at (entry_price + 1.5%), TP at (entry_price - 5%)

Filled independently (not coordinated exit unless one closes for some reason).

### Coordination on Straddle Pair

**Entry Coordination** (Step 0 safety):
- Scalper places first leg (LONG or SHORT, random)
- Waits for fill (verify via `_check_order_filled()`)
- Places second leg
- If second leg fails, **rollback**: close first leg + retry (up to 2 times, 2s delay)
- If rollback fails, **orphan**: mark as independent, auto-close after 4 cycles

**Config Controls**:
```python
entry_retry_count=2           # Retries for second leg
entry_retry_delay_s=2.0       # Delay between retries
rollback_retry_count=2        # Retries for rollback close
sl_cancel_retry_count=2       # Retries for counterpart SL cancel
orphan_auto_close_cycles=4    # Cycles before auto-close (~60min at 15m)
```

## Multi-TF Integration

### Priority Ordering (4h > 1h > 30m > 15m)

Higher timeframe signals have better quality (less whipsaw); scanned first:

1. **4h** (if enable_4h): Data fetched directly by scalper (no resampling noise)
2. **1h** (if enable_1h): Resampled from 15m (4:1 ratio)
3. **30m** (if enable_30m): Resampled from 15m (2:1 ratio)
4. **15m** (if enable_15m): Direct, highest frequency

**Per-Symbol Dedup**:
- First non-zero signal wins (return immediately)
- No multiple signals per bar (one position max per symbol)

### Cross-TF References (Completed Bars Only)

All resampling uses **completed bar indices**:

```python
def resample_ohlcv(close, high, low, volume, ratio=4):
    n_out = len(close) // ratio  # Truncate incomplete bars
    for i in range(n_out):
        start = i * ratio
        end = start + ratio
        c_out[i] = close[end - 1]  # LAST bar's close (completed)
        h_out[i] = max(high[start:end])
        l_out[i] = min(low[start:end])
```

**Example** (15m → 1h):
- 15m bars 0–3 → 1h bar 0 (close[3])
- 15m bars 4–7 → 1h bar 1 (close[7])
- If 200 × 15m bars → 50 × 1h bars (bars 0–49)
- Bar 200–203 (incomplete) → dropped (no bar 50 created)

This **guarantees zero lookahead**: 1h signal at bar i is computed only after 15m bar (i*4+3) closes.

### Warmup Requirements

Each TF has its own warmup (VR needs ~50h of history):

```python
warmup_15m=200      # ~50h of 15m bars
warmup_1h=50        # ~50h of 1h bars
warmup_4h=13        # ~52h of 4h bars
warmup_30m=100      # ~50h of 30m bars (resampled)
```

Strategy checks: `if len(close) < warmup + 1: return 0` before firing signal.

## Backtest Integration

### TradeEventSim (Canonical)

**Location**: `quantbox/backtest/trade_sim.py`

Entry point for research scripts:

```python
from quantbox.backtest.trade_sim import collect_trades, portfolio_sim, compute_metrics

# 1. Collect all trades for single symbol
trades = collect_trades(
    close, high, low, volume,
    sim_directional_trailing,  # or sim_straddle_trailing
    cfg, symbol
)

# 2. Simulate N-slot portfolio (N symbols × timeframe)
pnl_curve, positions_log = portfolio_sim(
    trades_by_symbol,
    max_slots=10,
    leverage=30.0,
    ...
)

# 3. Compute metrics
metrics = compute_metrics(pnl_curve)
# → {"PF": 1.35, "Calmar": 5.2, "MaxDD": 2.6%, ...}
```

### Dual Simulation Functions

| Function | Trade Type | Exit Logic |
| --- | --- | --- |
| `sim_directional_trailing()` | D_mom8 entries | Trailing stop (priority 1), SL, TP, max_hold |
| `sim_straddle_trailing()` | S_d8 entries (LONG+SHORT pair) | Per-leg trailing; both must exit (pair logic) |

Both are **Numba-compiled** for speed. Match live exit priority exactly.

### Parity Validation

Test file: `tests/test_backtest_live_parity.py`

Compares:
- Backtest exit prices vs live exit prices
- Same SL/TP/trailing logic
- Same leverage/sizing
- Delta < 0.1% acceptable (rounding, tick size)

When modifying exit logic:
1. Update `sim_directional_trailing()` + `sim_straddle_trailing()`
2. Run parity test
3. Update backtest scripts (import parity-validated functions)

## Critical CLAUDE.md Rules

These rules exist because of past real losses. Violations cause silent bugs or capital loss:

| # | Rule | Context |
| --- | --- | --- |
| **#1** | `futures_change_leverage()` only in entry paths, never close/reduce | Prevents accidental overleveraging during exits |
| **#2** | Cross-TF refs use completed bars only: `j = (i // ratio) - 1` | Eliminates lookahead bias in signal generation |
| **#3** | Slot sizing tests must include `n_active > 0` state | Catch edge cases (empty portfolio, first trade) |
| **#4** | `_check_order_filled()` handles both TRIGGERED and FINISHED | Async orders; can expire without fill |
| **#5** | Hedge Mode close uses `reduce_only=False` + leverage guard: `if not reduce_only and leverage > 1:` | Prevents close-side 2x deleveraging bugs |
| **#6** | Config params always read from `VolStraddleConfig()`, never hardcoded | .env changes silently break hardcoded values |

**Incident Context**:
- **r85**: Hardcoded SL=1% caused live losses when config changed to SL=0.5% (undetected for 3 hours)
- **r67**: Close side called `futures_change_leverage(1.0)` during reduce → doubled leverage mid-exit → margin call
- **r59**: Straddle pair entry lacked retry logic → orphaned SHORT legs (unrealized loss accumulation)
- **r44**: Lookahead in 1h resampling: used developing bar 4th candle → premature entries
- **r39**: Hardcoded 60 symbols list broke when whitelist config read from env → 0 trades for weeks

## For AI Agents

### Modifying Configuration

1. Edit `quantbox/strategy/vol_straddle/config.py` (pydantic-settings default)
2. **OR** set env var: `VS_<PARAM_NAME>=value` (reads from .env automatically)
3. Never hardcode; always read:
   ```python
   cfg = VolStraddleConfig()
   sl = cfg.d_sl  # Reads from .env or config.py default
   ```

**Validation**: VolStraddleConfig has 15+ cross-checks (SL<TP, leverage≥1, trail_distance<trail_activate, etc.). Invalid configs raise ValueError immediately.

### Adding Signal Functions

1. Write stateless Numba JIT function in `signals.py`:
   ```python
   @numba.jit(nopython=True, cache=True)
   def my_new_filter(close, high, low, threshold):
       # No object mutation, only array math
       result = np.zeros(len(close))
       for i in range(len(close)):
           if condition:
               result[i] = 1
       return result
   ```

2. Integrate into strategy's `_check_tf_signal()`:
   ```python
   filter_result = my_new_filter(close, high, low, cfg.my_threshold)
   if not np.any(filter_result[-1]):  # If filter rejects current bar
       return 0  # Skip signal
   ```

3. Add tests: `tests/test_signals_unit.py`
4. Update backtest scripts to import the same function (parity guarantee)

### Changing Exit Logic

1. Modify `update_position()` in VolStraddleStrategy
2. **ALSO update** backtest sim: `sim_directional_trailing()` + `sim_straddle_trailing()`
3. Run parity test: `pytest tests/test_backtest_live_parity.py -v`
4. If delta > 0.1%, investigate (rounding, tick size, or bug)

### Testing Multi-TF

When modifying cross-TF logic (resample, priority, etc.):

1. Write test using example 15m bars:
   ```python
   def test_mtf_resampling():
       c15m = np.array([100, 101, 102, 103, 104, 105, 106, 107, ...])
       c1h, _, _, _ = resample_ohlcv(c15m, h15m, l15m, v15m, 4)
       assert c1h[0] == 103  # 15m[3] is 1h[0] close (4th bar)
       assert len(c1h) == len(c15m) // 4  # No partial bars
   ```

2. Verify completed bar formula: `j = (i // ratio) - 1`

3. Check warmup: `if len(close) < warmup + 1: return 0`

## Dependencies

| Module | Role |
| --- | --- |
| `quantbox.strategy.vol_straddle.config` | VolStraddleConfig (pydantic-settings, .env reader) |
| `quantbox.strategy.vol_straddle.signals` | Numba JIT signal functions (compute_vr, gen_vr_signal, etc.) |
| `quantbox.strategy.protocol` | TradingStrategy Protocol (14-member interface) |
| `quantbox.backtest.trade_sim` | Canonical backtest engine (sim_directional_trailing, sim_straddle_trailing, collect_trades, portfolio_sim) |
| `quantbox.core.types` | ApiSymbol vs PositionKey type distinction |
| `quantbox.core.scalper_state` | State persistence (crash recovery) |
| `quantbox.scalper` | RubberBandScalper (live orchestrator) |
| `numpy` | Array operations (Numba-compiled) |
| `numba` | JIT compilation (nopython=True mode) |
| `pydantic` | Config validation + model_validator constraints |
| `pydantic-settings` | .env file reading (BaseSettings) |

## Cross-References

- **Strategy Parent**: See `quantbox/strategy/AGENTS.md` (TradingStrategy Protocol, architecture)
- **RubberBand Comparison**: See `quantbox/strategy/rubberband/AGENTS.md` (legacy strategy)
- **Backtest Details**: See `quantbox/backtest/AGENTS.md` (if created; trade_sim.py, metrics)
- **Scalper Integration**: See `quantbox/scalper.py` docstring + `AGENTS.md` (live flow)
- **Configuration**: See `.env.example` (all VS_* env vars)

## Releases & Changes

| Release | Key Changes | Date |
| --- | --- | --- |
| **r73 (current)** | Combined D+S+MA portfolio: PF=2.002, Calmar=1695.6, 5/5 years profitable | 2026-03 |
| **r90** | Server-side TRAILING_STOP_MARKET migration (trail_server_native=False for safe rollout) | 2026-03 |
| **r69** | Trail activation 0.5% → 1.0% (d_trail_activate); +38% OOS on validation set | 2026-02 |
| **r76** | S_d8 straddle trail distance 0.2% → 0.1%; PF 4.876 → 9.349 | 2026-02 |
| **r47** | D_mom8 trail parameters validated; disabled F5 TBR gate (2024/25 OOS degrade) | 2026-01 |
| **r43** | VR compression detector + trailing stop exit (R43 validated) | 2025-12 |

