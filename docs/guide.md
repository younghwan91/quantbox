# QuantBox Architecture & Reference Guide

New here? Start with the [README](../README.md).

<!-- toc -->
- [Architecture Overview](#architecture-overview)
- [Strategies, Safety Guards & Backtesting](#strategies-safety-guards--backtesting)
- [Testing & Project Structure](#testing--project-structure)
<!-- /toc -->

## Architecture Overview

```mermaid
flowchart LR
    Data["Data\n2-Tier Cache (LRU + Disk)"] --> Strategy["Strategy\nVolStraddle · MA Cluster"]
    Strategy -->|"TradingStrategy Protocol"| Engine["Core Engine\nscalper.py · loop_mixin"]
    Engine --> Exec["Execution Guards\nBrackets · ExposureGuard · ReconJournal"]
    Exec --> Exchange[("Binance USDT-M\nFutures API")]
    Strategy -.->|"Same signal logic"| BT["Dual-Engine Backtest\nvectorized_backtest · trade_sim (Numba)"]
    Engine -.-> Notify["Discord Bot · Dashboard · Journal"]
```

---

## Strategies, Safety Guards & Backtesting

<details>
<summary>Strategy Suite — VolStraddle, MA Cluster Squeeze, Legacy Strategy</summary>

### VolStraddle — VR Compression Detection
- **Volatility Ratio (VR) Compression**: Detects volatility compression when the short-term ATR to long-term ATR ratio drops below a threshold.
- **Multi-Timeframe**: Evaluates 15m/1h/4h signals independently with higher-timeframe priority.
- **D_mom8**: Enters directionally along 8-bar momentum when a compression signal fires.
- **S_d8 Straddle**: Enters both directions simultaneously (Hedge Mode) when compression persists for a configured number of bars.
- **Exits**: Stop-loss, safety-cap take-profit, trailing stop, and maximum holding duration.
- **No Lookahead**: References completed bars only, excluding the in-progress bar.

### MA Cluster Squeeze — 4h Moving Average Convergence Breakout
- **MA Cluster Detection**: Captures breakouts after SMA(20/50/100/200) lines stay tightly converged for a minimum duration.
- **4h Exclusive**: Operates solely on the higher timeframe with minimal overlap against VR signals.
- **Direction & Filters**: Enters in the momentum direction breaking out of the cluster, filtering out stale clusters and weak momentum signals.

### Legacy Strategy (Non-Production)
- **Rubber Band Scalper**: ATR-adaptive mean reversion with volume climax filter (`quantbox/strategy/rubberband/`).

</details>

<details>
<summary>Data Management & Live Trading Safety Guards</summary>

### Data Management
- **2-Tier Caching**: In-memory LRU + compressed disk cache.
- **Multi-Timeframe**: 1m, 5m, 15m, 1h, 4h, 1d.
- **Binance Integration**: Rate limiting and batch fetching.

### Live Execution Safety Guards
- **Server-Side Stop-Loss**: Binance STOP orders guarantee stop protection even during client crashes or network disconnects.
- **Leverage Verification**: Aborts order placement if leverage configuration fails.
- **Limit Chaser & Market Fallback Verification**: Reduces slippage via limit chasing and re-queries ACK responses to prevent phantom fills.
- **State Persistence**: Saves position state to JSON and reconciles against the exchange on crash recovery.
- **Max Drawdown Breaker · Daily Loss Limit**: Halts new entries and liquidates positions if drawdown from session peak or daily loss limit is breached.
- **Consecutive Error Guard · Close Failure Retention**: Liquidates all positions on consecutive errors; retains slot tracking on close failures for retry.
- **Rate Limiter + Circuit Breaker · In-Flight Order Tracker**: Tracks API rate limits and pending orders.
- **ExposureGuard · ReconJournal · Config Validators**: Session exposure caps, SQLite audit log, and cross-field config validation (`SL < TP`, `leverage >= 1`).
- **Discord Bot**: Remote monitoring and commands (position inspection, forced close, etc.).

</details>

<details>
<summary>Backtesting (Dual-Engine)</summary>

- **Signal-Array Engine** (`vectorized_backtest.py`): Fast single-asset scan using `±1` position arrays.
- **Trade-Event Engine** (`trade_sim.py`): Discrete trade simulation with High/Low intrabar SL/TP, N-slot portfolio contention, and Numba acceleration. Pipeline: `sim_directional_trades()` / `sim_straddle_trades()` → `collect_trades()` → `portfolio_sim()` → `compute_metrics()`.
- **Rolling Windows & Metrics**: Formation/Trading period simulation, Sharpe, Sortino, Calmar, Profit Factor, and Bootstrap CI.

</details>

---

## Testing & Project Structure

```bash
uv run pytest tests/ -v
```

| Path | Description |
|---|---|
| `quantbox/core/` | Config (`pydantic-settings`), event loop, state persistence, rate limiter, Discord commands, trade journal |
| `quantbox/execution/` | Server-side brackets, positions, exchange reconciliation, in-flight tracker, exposure cap, preflight, risk guards |
| `quantbox/data/` | Binance downloader, 2-tier cache, data loader |
| `quantbox/strategy/` | `vol_straddle` (VR compression + MA cluster), `rubberband` (legacy), `vol_selector`, `TradingStrategy` Protocol |
| `quantbox/backtest/` | `vectorized_backtest` (signal array), `trade_sim` (trade event) |
| `quantbox/notifications/` | Discord bot (two-way) & webhooks (one-way) |
| `quantbox/utils/` | Diagnostic logging |
| `quantbox/scalper.py` | Core engine entry point |
| `scripts/`, `research/`, `autoresearch/`, `docs/`, `tests/` | Utility scripts, research, docs ([STRATEGIES.md](STRATEGIES.md), [MANUAL.md](../MANUAL.md)), tests |
