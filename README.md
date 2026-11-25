<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="quantbox" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

**Binance USDT-M futures Volatility Ratio Squeeze and MA Cluster breakout trading system.**

Numba backtesting and live execution share the exact same `TradingStrategy` code path. Educational and research purposes only.

[![CI](https://github.com/younghwan91/quantbox/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/quantbox/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

[Quick start](#quick-start) · [Docs](docs/guide.md) · [Strategies](docs/STRATEGIES.md) · [Manual](MANUAL.md) · [한국어](README.ko.md)

</div>

## Try it now (no API keys)

```bash
git clone https://github.com/younghwan91/quantbox.git && cd quantbox
uv sync --dev
uv run pytest tests/test_backtest_live_parity.py tests/test_backtest_plausibility.py -m "not slow" --no-cov -q
```

Expected: `38 passed` in about 10 seconds. The first file checks that the live signal path makes the same decisions as the backtest engine (signal direction, multi-timeframe gating, leverage, dropping the open candle). The second checks the plausibility gate that rejects lookahead-looking backtests. Both use in-memory test data, so no exchange account or downloaded data is needed.

## Paper trading (needs Binance keys)

```bash
cp .env.example .env   # fill in BINANCE_API_KEY and BINANCE_API_SECRET
python -m quantbox.scalper --strategy vol_straddle --paper --once
```

One paper-trading cycle evaluates completed candles and logs signals without placing real orders. The full suite runs without credentials: `uv run pytest tests/ -v`.

## What it does

- **Dual volatility breakout signals** — `VolStraddle` (15m/1h/4h VR compression with `D_mom8` and `S_d8`) and `MA Cluster Squeeze` (4h SMA 20/50/100/200 breakout) share a unified 10-slot pool.
- **Unified live and backtest path** — Identical `TradingStrategy` Protocol signal logic runs across the Numba-accelerated backtest engines (`vectorized_backtest.py`, `trade_sim.py`) and the live scalper.
- **Multi-layer execution guards** — Server-side STOP brackets, Limit Chaser, Market Fallback verification, Max Drawdown Breaker, Daily Loss Limit, `ExposureGuard`, and SQLite `ReconJournal` protect live sessions.
- **Operational visibility** — Includes a real-time Streamlit dashboard, a trade journal CLI, and two-way Discord bot and webhook alerts.

## Strategies

| Strategy | Timeframe | Mechanism | Reference |
|---|---|---|---|
| **VolStraddle (`D_mom8` / `S_d8`)** | 15m · 1h · 4h | VR compression (`ATR(8)/ATR(56)`) + momentum or straddle | [docs/STRATEGIES.md](docs/STRATEGIES.md) |
| **MA Cluster Squeeze** | 4h | Breakout after SMA(20/50/100/200) convergence | [docs/STRATEGIES.md](docs/STRATEGIES.md) |
| **RubberBand (legacy)** | — | ATR-adaptive mean reversion (non-production) | `quantbox/strategy/rubberband/` |

## Configuration

API keys, Discord tokens, webhook URLs, and channel IDs are read exclusively from `.env` (gitignored). If required values are missing or left as `.env.example` placeholders (`your_...`), config validation refuses to start, including in `--paper` mode.

Strategy parameters live in `quantbox/strategy/*/config.py` (`VolStraddleConfig`), not `.env`.

## Monitoring & CLI

```bash
bash scripts/start_dashboard.sh start   # Streamlit dashboard at http://localhost:8501
bash scripts/start_dashboard.sh status  # Check dashboard status
bash scripts/start_dashboard.sh stop    # Stop dashboard

python3 scripts/journal.py              # Today's trades and overall summary
python3 scripts/journal.py symbols      # PnL ranking by symbol
python3 scripts/journal.py last 10      # Last 10 trades
python3 scripts/journal.py strategy     # Performance by strategy
```

## Caveats

- **Educational and research only** — Execution results are not presented as evidence of profitability; cryptocurrency futures carry substantial risk of loss.
- **Hedge Mode required for live orders** — Running without `--paper` places real orders; enable Binance Hedge Mode and configure `.env` first.
- **Completed bars only** — Signal and cross-timeframe logic read closed candles only to prevent lookahead bias.

## Docs

| Document | Contents |
|---|---|
| [Architecture & Reference Guide](docs/guide.md) | Architecture diagram, safety guards, dual-engine backtest, project structure |
| [Strategy Reference](docs/STRATEGIES.md) | Active strategies, entry/exit rules, and risk gates |
| [Operations Manual](MANUAL.md) | Detailed setup, live trading checklist, and operational commands |

## Validation write-ups

Out-of-sample findings, including what looked good and was not, are published in [crypto-quant-insights](https://github.com/younghwan91/crypto-quant-insights). Backtest archives for this repo are in [docs/backtest_results](docs/backtest_results/README.md) and are research records, not a track record.

## More from the same author

- [crypto-quant-insights](https://github.com/younghwan91/crypto-quant-insights) - fourteen chart-backed lessons from validating this system.
- [crypto-pair-trading](https://github.com/younghwan91/crypto-pair-trading) - event-driven statistical-arbitrage framework with shared backtest and live execution.
- [kiwoom-client](https://github.com/younghwan91/kiwoom-client) - Kiwoom REST API client for Korean equities.
- [fin-checkup](https://github.com/younghwan91/fin-checkup) - financial health check-up for Korean listed companies from DART filings.

## Contributing

Run the test suite before opening a pull request:

```bash
uv run pytest tests/ -v
```

Issues and pull requests are welcome.

## License

[MIT](LICENSE) © 2026 Younghwan Chae
