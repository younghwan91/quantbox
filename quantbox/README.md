# QuantBox 📈

**Professional Cryptocurrency Pair Trading Framework**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive quantitative trading framework for cryptocurrency pair trading with unified backtesting and live trading capabilities.

## 🌟 Features

### Core Architecture
- **Event-Driven Engine**: Unified architecture ensures identical behavior in backtest and live trading
- **Modular Design**: Plug-and-play components for strategies, filters, and execution
- **Type Safety**: Full type hints throughout the codebase

### Data Management
- **Two-Tier Caching**: Memory (LRU) + Disk (compressed) caching system
- **Multi-Timeframe Support**: Systematic handling of 1m, 5m, 15m, 1h, 4h, 1d data
- **Binance Integration**: Built-in downloader with rate limiting and batch processing

### Strategy Framework
- **Flexible Filter Pipeline**: Composable filters for pair selection
  - Correlation Filter
  - Cointegration (Engle-Granger) Filter
  - Half-Life Filter
  - Hurst Exponent Filter
  - Stationarity (ADF) Filter
  - Volume Filter
- **Pair Selection**: Statistical pair identification and ranking
- **Kalman Filter**: Online hedge ratio estimation

### Backtesting
- **Rolling Window**: Formation/trading period simulation
- **Walk-Forward Analysis**: Out-of-sample validation
- **Monte Carlo Simulation**: Robustness testing
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar, Profit Factor, etc.

### Optimization
- **Multiple Algorithms**:
  - Grid Search
  - Random Search
  - Differential Evolution
  - Bayesian Optimization
- **Sensitivity Analysis**: Parameter robustness testing
- **Multi-Objective**: Combined optimization criteria

### Visualization
- **Interactive Dashboard**: Streamlit-based UI
- **Plotly Charts**: Equity curves, drawdowns, trade analysis
- **HTML Reports**: Exportable backtest reports

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/trading_code.git
cd trading_code

# Install with pip
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
statsmodels>=0.13.0
numba>=0.54.0
pykalman>=0.9.5
ta>=0.10.0
python-binance>=1.0.0
pyyaml>=6.0
tqdm>=4.62.0
plotly>=5.0.0
streamlit>=1.10.0
pytest>=7.0.0
scikit-learn>=1.0.0
```

## 🚀 Quick Start

### Basic Backtest

```python
from quantbox import Config, BacktestRunner, PairTradingStrategy
from quantbox.data import DataDownloader

# Download data
downloader = DataDownloader()
data = downloader.download_multiple(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
    timeframe='1h',
    days=90
)

# Configure backtest
from quantbox.backtest.runner import BacktestConfig

config = BacktestConfig(
    initial_capital=10000.0,
    trading_fee=0.0005,
    entry_zscore=2.0,
    exit_zscore=0.5,
    formation_periods=240,
    trading_periods=24
)

# Run backtest
runner = BacktestRunner(config)
results = runner.run_backtest(data)

# View results
print(results['metrics'].summary())
```

### With Pair Selection

```python
from quantbox.strategy import PairSelector, FilterPipeline
from quantbox.strategy.filters import (
    CorrelationFilter, 
    CointegrationFilter,
    HalfLifeFilter
)

# Create filter pipeline
pipeline = FilterPipeline([
    CorrelationFilter(min_correlation=0.7),
    CointegrationFilter(max_pvalue=0.05),
    HalfLifeFilter(min_halflife=5, max_halflife=50),
])

# Select pairs
selector = PairSelector(filter_pipeline=pipeline)
candidates = selector.select_pairs(data)

# Get tradeable pairs
pairs = selector.select_tradeable_pairs(candidates, min_z_score=2.0)
print(f"Found {len(pairs)} tradeable pairs")
```

### Optimization

```python
from quantbox.optimization import (
    DifferentialEvolutionOptimizer,
    SharpeObjective
)

# Define parameter bounds
param_bounds = {
    'entry_zscore': (1.5, 3.0),
    'exit_zscore': (0.2, 1.0),
    'stop_loss_zscore': (2.5, 4.0),
}

# Run optimization
optimizer = DifferentialEvolutionOptimizer(
    param_bounds=param_bounds,
    objective_func=SharpeObjective(),
    population_size=15,
    max_iterations=30
)

result = optimizer.optimize(data)

print(f"Best params: {result.best_params}")
print(f"Best Sharpe: {result.best_score:.2f}")
```

### Walk-Forward Analysis

```python
from quantbox.backtest import WalkForwardAnalysis

wfa = WalkForwardAnalysis(
    train_periods=500,
    test_periods=100,
    step_periods=100
)

results = wfa.run(
    data=data,
    param_ranges={
        'entry_zscore': (1.5, 3.0),
        'exit_zscore': (0.2, 1.0),
    }
)

print(f"Average OOS return: {results['summary']['avg_test_return']:.2f}%")
print(f"Efficiency ratio: {results['summary']['avg_efficiency']:.2f}")
```

### Launch Dashboard

```bash
# Command line
python -m quantbox.ui.dashboard

# Or in Python
from quantbox.ui import run_dashboard
run_dashboard()
```

## 📁 Project Structure

```
quantbox/
├── __init__.py              # Package exports
├── core/                    # Core engine components
│   ├── config.py           # Configuration management
│   ├── engine.py           # Trading engine
│   ├── events.py           # Event system
│   ├── execution.py        # Order execution
│   └── portfolio.py        # Position management
├── data/                    # Data handling
│   ├── cache.py            # Caching system
│   ├── downloader.py       # Data download
│   └── preprocessor.py     # Data preprocessing
├── strategy/               # Strategy components
│   ├── base.py             # Strategy base classes
│   ├── filters.py          # Filter implementations
│   ├── pair_selector.py    # Pair selection
│   └── pair_strategy.py    # Pair trading strategy
├── backtest/               # Backtesting framework
│   ├── runner.py           # Backtest runner
│   ├── metrics.py          # Performance metrics
│   ├── analyzer.py         # Results analysis
│   └── walk_forward.py     # Walk-forward testing
├── optimization/           # Parameter optimization
│   ├── optimizer.py        # Optimization algorithms
│   ├── objective.py        # Objective functions
│   └── sensitivity.py      # Sensitivity analysis
└── ui/                     # User interface
    ├── components.py       # UI components
    └── dashboard.py        # Streamlit dashboard

tests/                      # Unit tests
├── conftest.py            # Test fixtures
├── test_config.py
├── test_events.py
├── test_portfolio.py
├── test_cache.py
├── test_filters.py
├── test_metrics.py
└── test_optimization.py
```

## 📊 Configuration

### YAML Configuration File

```yaml
mode: backtest  # backtest, paper, or live

exchange:
  name: binance
  testnet: true
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
  rate_limit: 1200

data:
  symbols:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
    - SOLUSDT
  timeframes:
    - 1h
    - 4h
  cache_enabled: true
  cache_dir: ./cache

strategy:
  entry_zscore: 2.0
  exit_zscore: 0.5
  stop_loss_zscore: 3.5
  max_holding_periods: 100
  min_correlation: 0.5
  max_cointegration_pvalue: 0.05
  min_halflife: 5
  max_halflife: 100

risk:
  max_position_size: 0.1
  max_portfolio_risk: 0.3
  max_daily_loss: 500.0
  max_drawdown: 0.2
  max_pairs: 5

backtest:
  initial_capital: 10000.0
  trading_fee: 0.0005
  slippage: 0.001
  formation_periods: 240
  trading_periods: 24
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_filters.py -v

# Run with coverage
pytest tests/ --cov=quantbox --cov-report=html
```

## 📈 Performance Metrics

The framework calculates comprehensive performance metrics:

| Metric | Description |
|--------|-------------|
| Total Return | Absolute return over the period |
| Annualized Return | CAGR |
| Sharpe Ratio | Risk-adjusted return (volatility) |
| Sortino Ratio | Risk-adjusted return (downside only) |
| Calmar Ratio | Return / Max Drawdown |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross Profit / Gross Loss |
| Average Trade | Mean P&L per trade |

## 🔧 Advanced Usage

### Custom Strategy

```python
from quantbox.strategy.base import StatefulStrategy
from quantbox.core.events import SignalType

class MyStrategy(StatefulStrategy):
    def on_bar(self, timestamp, data):
        # Your logic here
        z_score = self.calculate_zscore(data)
        
        if z_score > self.config['entry_zscore']:
            return {
                'type': SignalType.SHORT,
                'z_score': z_score,
                'strength': min(abs(z_score) / 3, 1.0)
            }
        elif z_score < -self.config['entry_zscore']:
            return {
                'type': SignalType.LONG,
                'z_score': z_score,
                'strength': min(abs(z_score) / 3, 1.0)
            }
        
        return None
```

### Custom Filter

```python
from quantbox.strategy.filters import BaseFilter, FilterResult

class MyCustomFilter(BaseFilter):
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "MyCustomFilter"
    
    def apply(self, series1, series2) -> FilterResult:
        # Your filter logic
        metric = self.calculate_metric(series1, series2)
        passed = metric > self.threshold
        
        return FilterResult(
            filter_name=self.name,
            passed=passed,
            value=metric,
            reason=f"Metric: {metric:.4f}"
        )
```

## ⚠️ Disclaimer

This software is for educational and research purposes only. Do not use it for actual trading without thorough testing and understanding of the risks involved. Cryptocurrency trading involves substantial risk of loss.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📮 Contact

For questions or feedback, please open an issue on GitHub.
