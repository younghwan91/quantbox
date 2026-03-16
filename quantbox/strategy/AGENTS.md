<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-16 -->

# Strategy Module — Agent Guide

## Overview

The Strategy module defines a **type-safe protocol** for trading strategies and provides two production implementations: **VolStraddleStrategy** (VR Squeeze, 30x, primary) and **RubberBandStrategy** (mean-reversion, 21x, legacy). Strategies are selected via structural subtyping—no inheritance required. The **TradingStrategy Protocol** is a 14-member `@runtime_checkable` interface that guarantees scalper.py can use any compliant strategy without modification.

This design decouples:
- **Live trading**: Stateful strategy objects (VolStraddleStrategy, RubberBandStrategy) in scalper.py
- **Backtesting**: Stateless signal functions (signals.py modules) + discrete trade simulators

## Protocol Overview

### The 14-Member TradingStrategy Protocol

All strategies must implement these members to be usable by `RubberBandScalper`:

| Category | Member | Signature | Role |
| --- | --- | --- | --- |
| **Properties** | `n_active` | `-> int` | Count of open positions (used for sizing) |
| **Properties** | `active_positions` | `-> dict[str, dict]` | Snapshot of all open positions (keyed by symbol or symbol:SIDE) |
| **Data Feed** | `update_market_data()` | `(symbol, volume, taker_buy, high, low) -> None` | Store 15m OHLCV before on_bar() |
| **Data Feed** | `update_htf_data()` | `(symbol, close_4h, high_4h, low_4h) -> None` | Store 4h data for multi-TF analysis |
| **Signal Gen** | `on_bar()` | `(symbol, close_prices, btc_close) -> int` | Generate signal: -1 (short), 0 (none), 1 (long), 2 (straddle) |
| **Signal Gen** | `diagnose_bar()` | `(symbol, close_prices, btc_close) -> dict` | Return debug dict (VR, momentum, conviction, etc.) |
| **Position Mgmt** | `has_position()` | `(symbol) -> bool` | Check if position open for symbol |
| **Position Mgmt** | `open_position()` | `(symbol, signal, entry_price, notional, leverage, move_size, entry_stop_mult, server_sl_order_id, server_tp_order_id) -> None` | Record new position |
| **Position Mgmt** | `close_position()` | `(symbol) -> dict \| None` | Remove and return position or None |
| **Position Mgmt** | `update_position()` | `(symbol, current_price, current_atr_pct, current_high, current_low) -> str \| None` | Evaluate exit conditions; return reason or None |
| **Query** | `get_effective_stop_mult()` | `(close_prices, symbol, move_size) -> float` | Return stop-loss multiplier for sizing |
| **Query** | `get_last_confidence()` | `(symbol) -> int` | Return last confidence score (0–N) |
| **Query** | `was_mtf_confirmed()` | `(symbol) -> bool` | Was last signal multi-TF confirmed? |
| **Query** | `get_last_atr_pct()` | `(symbol) -> float` | Return last ATR as fraction of price |

### Why Protocol Instead of Inheritance?

1. **Type safety without coupling**: mypy verifies scalper.py only calls protocol methods—no knowledge of actual strategy class needed.
2. **Structural subtyping** (duck typing with type checking): Any class with the 14 methods works. Zero inheritance overhead.
3. **Backward compatibility**: New strategies can be added without modifying RubberBandScalper.
4. **No runtime cost**: Protocol is purely a type-checking construct; zero overhead in production.
5. **Extensibility**: Signal functions stay stateless (for backtesting); wrapper strategies add statefulness (for live trading).

## Two Implementations

### VolStraddleStrategy (Production)

**Location**: `quantbox/strategy/vol_straddle/strategy.py`

**Purpose**: Primary production strategy using Volatility Ratio compression detection.

**Characteristics**:
- **VR Squeeze Signal**: `VR = ATR(8) / ATR(56) < 0.50` detects volatility compression
- **D_mom8 Directional**: Single-sided entry; direction set by 8-bar momentum
- **S_d8 Straddle**: Simultaneous LONG+SHORT (Hedge Mode) when squeeze duration ≥ 3 bars
- **Multi-TF**: Scans 15m, 30m, 1h, 4h independently; priority 4h > 1h > 30m > 15m
- **Leverage**: 30x (Half-Kelly: f*/2=3.07%, derived from 25.9% WR, PF=1.310)
- **Exit Logic**: Trailing stop (activate 1.0%, distance 0.2%) > SL (1%) > TP (5% safety) > max_hold (48h)
- **Edge (r73, 5.67yr)**: PF=2.002, Calmar=1695.6, 60 symbols, 6 consecutive profitable years

**Zero Lookahead**: Only completed bars used. 1h/4h resampled from 15m (remainder truncated). No developing bar data.

### RubberBandStrategy (Legacy)

**Location**: `quantbox/strategy/rubberband/strategy.py`

**Purpose**: Mean-reversion strategy using moving average deviations (non-production, fallback only).

**Characteristics**:
- **MA-Based**: Detects price deviation from SMAs; reverts to mean
- **Leverage**: 21x (Half-Kelly: f*/2=4.33%, derived from 38.6% WR, PF=1.285)
- **Signal**: Directional via momentum; no straddle mode
- **v6-v12 Lineage**: Historical versions retained for comparison
- **Status**: Not modified for live; use VolStraddle for edge research

## Design Patterns

### Structural Subtyping for Flexibility

Both strategies implement TradingStrategy Protocol without inheriting from a base class. This allows:
- Plug-and-play strategy selection in scalper.py (simple config flag)
- Type-safe protocol checks (mypy ensures scalper only calls 14 methods)
- Independent evolution: VolStraddle and RubberBand can diverge without coupling

### Stateful Wrappers (Live) vs Stateless Functions (Backtest)

**Live trading** (RubberBandScalper):
- Uses strategy objects: `strategy.on_bar(symbol, close_prices)` returns -1/0/1/2
- Maintains per-symbol state: positions, metadata, confidence, ATR
- Called once per 15m bar with new price data

**Backtesting** (TradeEventSim):
- Uses signal functions directly: `compute_vr()`, `gen_vr_signal()`, `make_direction_momentum()`
- Stateless Numba JIT functions (faster, vectorizable)
- Arrays passed in; arrays returned (no object mutation)
- Research scripts import `collect_trades()` + `portfolio_sim()` from trade_sim.py (canonical source)

### Per-Bar Signal Flow

Each 15m cycle follows this sequence:

```python
# 1. Feed data
strategy.update_market_data(symbol, volume, taker_buy, high, low)
strategy.update_htf_data(symbol, close_4h, high_4h, low_4h)

# 2. Generate signal
signal = strategy.on_bar(symbol, close_prices, btc_close)

# 3. Check exits
if strategy.has_position(symbol):
    exit_reason = strategy.update_position(symbol, current_price, atr_pct, high, low)
```

Scalper uses return values to:
- **signal == 0**: No action
- **signal == 1**: Enter LONG
- **signal == -1**: Enter SHORT
- **signal == 2**: Enter STRADDLE (VolStraddle only; Hedge Mode)

## Architectural Integration

### Scalper Uses Strategy Objects (Live)

`RubberBandScalper` in `scalper.py` composes 6 mixins and holds a strategy instance:

```python
class RubberBandScalper(LoopMixin, DataMixin, BracketMixin, ...):
    def __init__(self, strategy: TradingStrategy, ...):
        self.strategy = strategy  # Type-safe Protocol reference

    def on_cycle(self):
        signal = self.strategy.on_bar(symbol, close_prices, btc_close)
        if signal in (-1, 1, 2):
            # Open position via strategy.open_position()
```

Scalper doesn't know or care if it's VolStraddleStrategy or RubberBandStrategy—it only calls the 14 protocol members.

### Backtest Engines Use Signal Functions Directly (Vectorized)

**VectorizedBacktest** (RubberBand, fast):
- Imports signal functions: `compute_signals(), compute_atr_pct()`
- Passes 1D arrays; gets signal array back
- Close-to-close PnL; no intrabar SL/TP

**TradeEventSim** (VolStraddle, production):
- Imports signal functions: `compute_vr(), gen_vr_signal(), make_direction_momentum()`
- Canonical entry point: `collect_trades()` returns list of (entry_bar, exit_bar, direction, pnl)
- Discrete trade events; intrabar SL/TP; trailing stops; N-slot portfolio contention
- Resampling: `resample_ohlcv()` generates 1h/4h from 15m (completed bars only)

### No Inheritance, Zero Coupling

- VolStraddleStrategy and RubberBandStrategy are **completely independent classes**
- Neither inherits from the other; both just implement the Protocol
- Signal functions in vol_straddle/signals.py and rubberband/signals.py are separate modules
- Scalper composition: inject strategy instance at runtime (factory pattern in scalper.py)

## For AI Agents

### Adding a New Strategy

1. Create a new class (e.g., `class MyStrategy`).
2. Implement all 14 TradingStrategy Protocol members (copy template from protocol.py).
3. Add stateless signal functions to `my_strategy/signals.py` (Numba JIT if performance-critical).
4. In scalper.py, add a factory flag: `if strategy_name == "my_strategy": strategy = MyStrategy(cfg)`.
5. Write tests in `tests/test_my_strategy.py` (parity: backtest vs live).

### Working with TradingStrategy Type

When modifying scalper.py or any code that uses `strategy` parameter:

```python
def some_function(strategy: TradingStrategy) -> int:
    # Type checker ensures you only call Protocol methods
    signal = strategy.on_bar(symbol, close_prices)  # ✓ OK
    confidence = strategy.get_last_confidence(symbol)  # ✓ OK

    # These fail type check (not in Protocol):
    signal = strategy._internal_state  # ✗ Error: not in Protocol
```

Use `mcp__plugin_oh-my-claudecode_t__lsp_hover` or `lsp_goto_definition` to navigate protocol members.

### Modifying Signal Generation

When adding a new signal function (e.g., new filter):

1. Write it as a **stateless Numba JIT function** in `signals.py`:
   ```python
   @numba.jit(nopython=True, cache=True)
   def my_filter(close: np.ndarray, threshold: float) -> np.ndarray:
       # No object mutation, only array math
       return result_array
   ```

2. Call it from the strategy wrapper:
   ```python
   class MyStrategy:
       def on_bar(self, symbol, close_prices, btc_close=None):
           filter_result = my_filter(close_prices, 0.5)
           # Use filter_result to modify signal
   ```

3. Update backtest scripts to import and use the same function (parity guarantee).

## Dependencies

| Module | Role |
| --- | --- |
| `quantbox.strategy.protocol` | TradingStrategy Protocol definition |
| `quantbox.strategy.vol_straddle.config` | VolStraddleConfig (pydantic-settings) |
| `quantbox.strategy.vol_straddle.strategy` | VolStraddleStrategy implementation |
| `quantbox.strategy.vol_straddle.signals` | Numba JIT signal functions |
| `quantbox.strategy.rubberband.config` | RubberBandConfig |
| `quantbox.strategy.rubberband.strategy` | RubberBandStrategy implementation |
| `quantbox.strategy.rubberband.signals` | Mean-reversion signal functions |
| `quantbox.backtest.trade_sim` | TradeEventSim (canonical backtest engine) |
| `quantbox.backtest.vectorized_backtest` | VectorizedBacktest (fast array-based) |
| `quantbox.scalper` | RubberBandScalper (live orchestrator) |
| `quantbox.core.types` | ApiSymbol, PositionKey type distinctions |

## Cross-References

- **VolStraddle Details**: See `quantbox/strategy/vol_straddle/AGENTS.md`
- **RubberBand Details**: See `quantbox/strategy/rubberband/AGENTS.md`
- **Backtest Integration**: See `quantbox/backtest/AGENTS.md` (if created)
- **Scalper Usage**: See `quantbox/scalper.py` docstring + `scalper/AGENTS.md`
