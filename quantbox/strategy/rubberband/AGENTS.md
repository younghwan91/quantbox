<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-16 -->

# RubberBand Strategy (Legacy) — Agent Guide

**Status**: LEGACY / INFRASTRUCTURE ONLY | **Leverage**: 21x (Half-Kelly) | **Production**: NO

## Overview

RubberBandStrategy is a **non-production mean-reversion strategy** retained for architectural reference and fallback capability. It implements the same TradingStrategy Protocol as VolStraddle, allowing plug-and-play selection in scalper.py, but is no longer actively traded or developed.

**Key Point**: Do NOT modify this strategy for edge research. Use VolStraddle (VR Squeeze, 30x) for all new development. RubberBand serves as:
1. **Fallback**: Runnable alternative if VolStraddle becomes unstable
2. **Architecture Template**: Example of stateful strategy wrapper + signal functions
3. **Comparison Baseline**: Historical reference for v6–v12 signal lineage

## Status

| Aspect | Status |
| --- | --- |
| **Live Trading** | NO |
| **Backtesting** | Yes (via VectorizedBacktest) |
| **Active Development** | NO |
| **Code Maintenance** | Minimal (critical bugs only) |
| **Signal Research** | Use VolStraddle instead |
| **Configuration** | RubberBandConfig (pydantic-settings) |
| **Test Coverage** | tests/test_rubberband.py (basic) |

## Brief Description

RubberBandStrategy detects when price deviates far from moving averages and enters on reversion back to mean:

- **Signal**: ATR-based deviation bands around SMAs
- **Direction**: Momentum (8-bar) assigns LONG/SHORT
- **No Straddle Mode**: Single-sided directional entries only
- **Multi-TF**: Scans 15m, 1h, 4h (like VolStraddle)
- **Leverage**: 21x (Half-Kelly: f*/2=4.33%, from 38.6% WR, PF=1.285)

Example trigger:
- Price 10% above SMA(50) → enter SHORT (revert to mean)
- Exit on SL (2%), TP (4%), or max_hold (48h)

## Historical Context

### Deployment Timeline

| Era | Strategy | Status | Context |
| --- | --- | --- | --- |
| **v6–v8** | RubberBand V6–V8 | Active | Original adaptive mean-reversion; MACD/Stochastic confluence |
| **v9–v12** | RubberBand V9–V12 (E-3Slot) | Active | 3-slot risk-tiered sizing; MACD confirmation |
| **v13–v14.0** | RubberBand (legacy) | Inactive | Maintained but not deployed; VolStraddle v13+ primary |
| **v14.24+ (now)** | RubberBand (archived) | Infrastructure only | Protocol example; fallback only |

### Why VolStraddle Superseded RubberBand

1. **Edge Stability**: VR squeeze (VolStraddle) PF=2.002 vs RubberBand PF=1.285 (backtested)
2. **Frequency**: VR signals more frequent; RubberBand dry spells (weeks without signal)
3. **OOS Performance**: VolStraddle consistent 2020–2025; RubberBand eroded in 2023–2024
4. **Hedge Mode**: VolStraddle S_d8 straddle adds directionless edge; RubberBand directional only
5. **Multi-TF**: Both support multi-TF, but VR less susceptible to timeframe artifacts

### Current Role

- **Fallback**: If VolStraddle underperforms unexpectedly, can quickly switch to RubberBand via config flag
- **Code Reference**: Stateful strategy pattern (update_market_data, on_bar, update_position) mirrored by VolStraddle
- **Test Infrastructure**: Dual-engine backtest (VectorizedBacktest) still compares both

## Key Differences from VolStraddle

| Aspect | RubberBand | VolStraddle |
| --- | --- | --- |
| **Signal Basis** | ATR deviation bands (mean-reversion) | Volatility Ratio compression (squeeze) |
| **Entry Condition** | Price > ATR×2.0 from SMA | VR < 0.50 threshold |
| **Timeframe** | 15m, 1h, 4h | 15m, 30m, 1h, 4h |
| **Direction Logic** | Momentum(8) sign | Momentum(8) sign (same) |
| **Straddle Mode** | NO | YES (S_d8, Hedge Mode) |
| **Leverage** | 21x (per-trade risk 4.2%) | 30x (per-trade risk 3.0%) |
| **Exit Strategy** | SL/TP/max_hold (no trailing) | Trailing stop (priority 1) |
| **Sample Size (6yr)** | ~1,500 trades (sparse) | ~2,336 trades (frequent) |
| **PF (Profit Factor)** | 1.285 | 2.002 |
| **Status** | Legacy | Production |

## Protocol Compliance

RubberBandStrategy implements **all 14 members** of TradingStrategy Protocol:

```python
class RubberBandStrategy(TradingStrategy):
    # Properties
    @property
    def n_active(self) -> int: ...

    @property
    def active_positions(self) -> dict[str, dict]: ...

    # Data feed
    def update_market_data(...) -> None: ...
    def update_htf_data(...) -> None: ...

    # Signal generation
    def on_bar(...) -> int: ...
    def diagnose_bar(...) -> dict: ...

    # Position management
    def has_position(...) -> bool: ...
    def open_position(...) -> None: ...
    def close_position(...) -> dict | None: ...
    def update_position(...) -> str | None: ...

    # Query helpers
    def get_effective_stop_mult(...) -> float: ...
    def get_last_confidence(...) -> int: ...
    def was_mtf_confirmed(...) -> bool: ...
    def get_last_atr_pct(...) -> float: ...
```

This means RubberBandStrategy **can be used as a drop-in fallback** in scalper.py:

```python
# Switch strategies via config or CLI flag
if strategy_name == "vol_straddle":
    strategy = VolStraddleStrategy(cfg)
elif strategy_name == "rubberband":
    strategy = RubberBandStrategy(cfg)  # Works identically to VolStraddle

# Scalper doesn't care which; calls the 14 protocol methods
```

## For AI Agents

### Do NOT Modify for Production

- **No parameter tuning** for live trading
- **No signal function changes** without explicit user request
- **No backtest optimization** (time spent here is wasted; use VolStraddle)

### Use VolStraddle for Edge Research

When tasked with:
- "Improve strategy edge"
- "Optimize entry filters"
- "Add conviction weighting"
- "Test new stop-loss logic"

→ **Route all work to VolStraddle**, not RubberBand. The PF=1.285 baseline is stale.

### Reference Only: Architectural Lessons

Study RubberBandStrategy when learning:
- How to implement TradingStrategy Protocol
- State management pattern (per-symbol _positions dict)
- Multi-TF data storage (_htf_close_data, per-symbol arrays)
- Signal metadata (_last_confidence, _mtf_confirmed for scalper query)

Example: VolStraddle's `_last_signal_meta` dict mimics RubberBand's metadata pattern.

### Fallback Activation

If asked to "restore RubberBand as active strategy":

1. Set env var: `STRATEGY=rubberband`
2. Scalper imports: `RubberBandStrategy(RubberBandConfig())`
3. Verify parity test passes: `pytest tests/test_backtest_live_parity.py -v`
4. Monitor live for 48h+ before confirming stability

Rollback to VolStraddle if:
- Drawdown exceeds 10% (baseline: 2.6%)
- Win rate < 30% (baseline: ~43%)
- Zero trades for 5+ days (stale signal)

## Testing (Minimal)

| Test | Location | Purpose | Freq |
| --- | --- | --- | --- |
| **Unit Tests** | tests/test_rubberband.py | Signal functions, position state | On demand |
| **Backtest Parity** | tests/test_backtest_live_parity.py | Live vs backtest exit matching | On demand |
| **Protocol Compliance** | tests/conftest.py (strategy_fixtures) | Verify 14 members implemented | Pre-release |
| **Multi-TF** | tests/test_vol_straddle.py (similar pattern) | Resample, completed bars | On demand |

**Not modified** unless:
- Critical bug found (crashes, NaN in signals)
- Protocol member signature changes (rare)
- Backtest infrastructure changes (trade_sim.py updates)

## Dependencies

| Module | Role |
| --- | --- |
| `quantbox.strategy.rubberband.config` | RubberBandConfig (pydantic-settings) |
| `quantbox.strategy.rubberband.signals` | Signal functions (compute_atr_pct, compute_signals, etc.) |
| `quantbox.strategy.protocol` | TradingStrategy Protocol (14-member interface) |
| `quantbox.backtest.vectorized_backtest` | Fast array-based backtest (RubberBand primary) |
| `quantbox.core.types` | Type distinctions |
| `numpy` | Array operations |
| `numba` | JIT compilation (if signal functions use @jit) |
| `pydantic` | Config validation |
| `pydantic-settings` | .env file reading |

## Cross-References

- **Strategy Parent**: See `quantbox/strategy/AGENTS.md` (TradingStrategy Protocol, design patterns)
- **VolStraddle Production**: See `quantbox/strategy/vol_straddle/AGENTS.md` (primary strategy)
- **Config Example**: See `quantbox/strategy/rubberband/config.py` (RubberBandConfig)
- **Signal Functions**: See `quantbox/strategy/rubberband/signals.py` (compute_signals, etc.)
- **Scalper Integration**: See `quantbox/scalper.py` (strategy selection logic)

## Summary

RubberBandStrategy is **archived infrastructure** — a working example of the TradingStrategy Protocol, retained as a fallback, but not the focus of development. All new edge research and parameter tuning should target VolStraddle (v14.24+, PF=2.002). If RubberBand is ever needed, it will work immediately due to Protocol compliance; no modifications required.

