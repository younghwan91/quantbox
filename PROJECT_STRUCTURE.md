trading_code/
├── trader.py
├── pair_trading_model.py
├── pair_selector.py
├── main_trader.py
├── main_backtest.py
├── binance_downloader.py
├── backtest_preprocessor.py
├── binance.json
├── crypto_downloader.log
├── crypto_trader.log
├── data/
│   ├── backtest/
│   │   ├── precomputed_values.pkl
│   │   ├── pairs_for_backtest.csv
│   │   ├── trading_coins_backtest.csv
│   │   └── pairs_list.csv
│   └── futures/
│       ├── <coin_name>.pkl
│       └── ...
├── logs/
│   ├── crypto_downloader.log
│   ├── crypto_trader.log
│   └── ...
└── PROJECT_STRUCTURE.md

# Trading Code Project Structure and Connections

## Overview

The `trading_code` project is organized into several modules, each responsible for different aspects of cryptocurrency trading and backtesting. Below is an overview of how the main classes and functions are interconnected within the project.

## Modules and Connections

### `binance_downloader.py`
- **Class `BinanceDownloader`**
  - Responsible for downloading historical kline data from Binance.
  - **Methods:**
    - `download_data()`: Downloads live data.
    - `download_data_for_backtest()`: Downloads data for backtesting.
    - `save_trading_coins()`: Saves the list of trading coins to a CSV file.
  - **Usage:**
    - Utilized by `main_backtest.py` to fetch data for analyzing and backtesting trading strategies.

### `backtest_preprocessor.py`
- **Class `BacktestPreprocessor`**
  - Preprocesses downloaded data for backtesting.
  - **Methods:**
    - `prepare_pairs_for_backtest()`: Prepares and filters trading pairs based on historical data.
    - `_filter_pairs()`: Filters pairs using statistical tests.
    - `_save_pairs_info()`: Saves precomputed pairs data for later use.
  - **Usage:**
    - Called by `main_backtest.py` after data downloading to prepare data for the trading model.

### `pair_trading_model.py`
- **Class `PairTradingModel`**
  - Implements the pair trading strategy model.
  - **Methods:**
    - `load_precomputed_values()`: Loads precomputed values for model training/testing.
    - `simulate_trading()`: Simulates trading based on selected parameters.
    - `optimize_parameters()`: Optimizes trading parameters for better performance.
  - **Usage:**
    - Integrated into `main_backtest.py` for running simulations and optimizing strategies.

### `pair_selector.py`
- **Class `PairSelector`**
  - Selects optimal trading pairs based on statistical analysis.
  - **Methods:**
    - `select_trading_pairs()`: Coordinates the selection of trading pairs.
    - `save_pairs_and_data()`: Saves the selected pairs and their data.
    - `select_best_pair()`: Selects the best trading pair based on performance metrics.
  - **Usage:**
    - Utilized by `main_backtest.py` and `main_trader.py` to determine which pairs to trade.

- **Class `CoinDataLoader`**
  - **Methods:**
    - `load_selected_trading_coins()`: Loads the selected trading coins from a CSV file.
    - `load_coin_data()`: Loads coin data from pickle files.
  - **Usage:**
    - Used internally by `PairSelector` to gather and prepare coin data for analysis.

- **Class `CoinAnalysis`**
  - **Methods:**
    - `find_valid_coins()`: Filters coins based on statistical properties.
    - `calculate_normalized_close_prices()`: Normalizes closing prices.
    - `identify_pairs_candidates()`: Identifies potential trading pairs based on cointegration.
  - **Usage:**
    - Supports `PairSelector` by providing analytical methods to evaluate coin relationships.

### `trader.py`
- **Class `CryptoTrader`**
  - Handles the execution of trades based on selected pairs.
  - **Methods:**
    - `print_futures_balance()`: Prints the current futures account balance.
    - `run_trading_cycle()`: Executes the trading strategy in cycles.
  - **Usage:**
    - Utilized by `main_trader.py` to perform live trading based on the backtested strategy.

### `main_backtest.py`
- **Function `main()`**
  - Orchestrates the backtesting process by:
    1. Setting up logging.
    2. Downloading data using `BinanceDownloader`.
    3. Preprocessing data with `BacktestPreprocessor`.
    4. Selecting pairs via `PairSelector`.
    5. Running simulations and optimizing parameters using `PairTradingModel`.
  - **Usage:**
    - Entry point for running backtests to evaluate trading strategies before deploying them live.

### `main_trader.py`
- **Main Execution Script**
  - Manages live trading by:
    1. Setting up logging and configurations.
    2. Initializing `CryptoTrader` with Binance client and configurations.
    3. Executing trading cycles if enabled.
  - **Usage:**
    - Entry point for running the live trading bot based on the selected pairs and optimized parameters.

## Data Flow

1. **Data Acquisition**
   - `BinanceDownloader` fetches historical and live kline data from Binance.

2. **Data Preprocessing**
   - `BacktestPreprocessor` processes raw data to prepare it for modeling.

3. **Pair Selection**
   - `PairSelector` analyzes preprocessed data to select statistically significant trading pairs.

4. **Modeling and Optimization**
   - `PairTradingModel` runs simulations on selected pairs and optimizes trading parameters.

5. **Trading Execution**
   - `CryptoTrader` executes trades based on optimized models and real-time data.

## Logging

Both `main_backtest.py` and `main_trader.py` are configured to log detailed information to both console and rotating log files, aiding in monitoring and debugging the trading processes.

## Summary

The `trading_code` project is modular, with each component handling specific tasks in the trading pipeline. The main scripts `main_backtest.py` and `main_trader.py` orchestrate the workflow by integrating data downloading, preprocessing, pair selection, and trading operations through the interconnected classes and functions in their respective modules.

