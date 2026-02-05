#!/usr/bin/env python3
"""
Test backtest system with REAL Binance Futures data.

This script:
1. Downloads real data from Binance Futures
2. Runs backtests with the actual market data
3. Validates results with real trading conditions
"""

import sys
import os
import json
import gzip
import pickle
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger('RealDataTest')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def load_cached_data(cache_dir: str = "data/cache") -> Dict[str, pd.DataFrame]:
    """Load existing cached Binance data."""
    cache_path = Path(cache_dir)
    index_file = cache_path / "cache_index.json"
    
    if not index_file.exists():
        logger.warning("No cache index found")
        return {}
    
    with open(index_file) as f:
        index = json.load(f)
    
    data = {}
    for key, info in index.items():
        filepath = Path(info['filepath'])
        if not filepath.exists():
            continue
            
        symbol = info.get('metadata', {}).get('symbol', key.split('_')[0])
        
        try:
            with gzip.open(filepath, 'rb') as f:
                obj = pickle.load(f)
            
            # Handle CacheEntry objects
            if hasattr(obj, 'data'):
                df = obj.data
            else:
                df = obj
            
            if not isinstance(df, pd.DataFrame):
                continue
                
            # Ensure proper column names
            if 'Close' not in df.columns:
                if 'close' in df.columns:
                    df = df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    })
            
            if 'Close' not in df.columns:
                logger.warning(f"No Close column in {symbol}: {df.columns.tolist()}")
                continue
            
            data[symbol] = df
            logger.debug(f"Loaded {symbol}: {len(df)} bars")
            
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
    
    return data


def download_fresh_data(n_coins: int = 30, days: int = 30, interval: str = "1h") -> Dict[str, pd.DataFrame]:
    """
    Download fresh data from Binance Futures.
    
    Requires binance.json with API credentials in the project root.
    """
    try:
        from binance.client import Client
        from binance.enums import HistoricalKlinesType
    except ImportError:
        logger.error("binance package not installed. Run: pip install python-binance")
        return {}
    
    # Load API credentials
    creds_file = Path(__file__).parent / "binance.json"
    if not creds_file.exists():
        creds_file = Path(__file__).parent / "_legacy" / "binance.json"
    
    if not creds_file.exists():
        logger.error("binance.json not found. Please create it with your API credentials.")
        return {}
    
    with open(creds_file) as f:
        creds = json.load(f)
    
    client = Client(api_key=creds['api_key'], api_secret=creds['api_secret'])
    
    # Get top coins by volume
    logger.info("Fetching top coins by volume...")
    exchange_info = client.futures_exchange_info()
    
    trading_coins = [
        info['symbol']
        for info in exchange_info['symbols']
        if info['symbol'].endswith('USDT') and info['status'] == 'TRADING'
    ]
    
    # Get volumes and sort
    volume_dict = {}
    for coin in trading_coins[:100]:  # Check top 100
        try:
            ticker = client.get_ticker(symbol=coin)
            volume_dict[coin] = float(ticker['quoteVolume'])
        except:
            continue
    
    sorted_coins = sorted(volume_dict.items(), key=lambda x: x[1], reverse=True)
    selected_coins = [c[0] for c in sorted_coins[:n_coins]]
    
    logger.info(f"Selected {len(selected_coins)} coins: {selected_coins[:5]}...")
    
    # Download data
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    data = {}
    for coin in selected_coins:
        try:
            logger.info(f"Downloading {coin}...")
            klines = client.get_historical_klines(
                symbol=coin,
                interval=interval,
                start_str=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_str=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                klines_type=HistoricalKlinesType.FUTURES
            )
            
            if not klines:
                continue
            
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            
            df = df.astype({
                'Open': float, 'High': float, 'Low': float,
                'Close': float, 'Volume': float
            })
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms', utc=True)
            df.set_index('Open Time', inplace=True)
            
            data[coin] = df
            
        except Exception as e:
            logger.warning(f"Failed to download {coin}: {e}")
    
    logger.info(f"Downloaded {len(data)} coins")
    return data


def run_backtest_with_real_data(data: Dict[str, pd.DataFrame]) -> None:
    """Run backtest with real Binance data."""
    from quantbox.backtest.continuous_backtest import (
        ContinuousPairBacktest, ContinuousBacktestConfig
    )
    
    logger.info("="*60)
    logger.info("BACKTEST WITH REAL BINANCE FUTURES DATA")
    logger.info("="*60)
    
    # Prepare data - ensure all have same index
    symbols = list(data.keys())
    logger.info(f"Available symbols: {len(symbols)}")
    
    # Find symbols with sufficient data
    min_bars = 400
    valid_symbols = [s for s, df in data.items() if len(df) >= min_bars]
    logger.info(f"Symbols with >= {min_bars} bars: {len(valid_symbols)}")
    
    if len(valid_symbols) < 10:
        logger.warning("Not enough data for comprehensive test")
        # Lower minimum
        min_bars = 250
        valid_symbols = [s for s, df in data.items() if len(df) >= min_bars]
        logger.info(f"Symbols with >= {min_bars} bars: {len(valid_symbols)}")
    
    # Use only valid symbols
    aligned_data = {}
    for s in valid_symbols[:20]:  # Top 20
        df = data[s].copy()
        # Ensure timezone-naive for comparison
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        aligned_data[s] = df
    
    # Find common time range
    common_start = max(df.index.min() for df in aligned_data.values())
    common_end = min(df.index.max() for df in aligned_data.values())
    
    logger.info(f"Common period: {common_start} to {common_end}")
    
    # Align to common range
    final_data = {}
    for symbol, df in aligned_data.items():
        aligned = df.loc[common_start:common_end].copy()
        if len(aligned) >= 250:  # Minimum required
            final_data[symbol] = aligned
    
    bars = len(next(iter(final_data.values())))
    logger.info(f"Aligned {len(final_data)} symbols, {bars} bars each")
    
    if bars < 250:
        logger.error(f"Not enough data: {bars} bars (need >= 250)")
        return
    
    # Adjust formation bars based on available data
    formation_bars = min(200, bars - 50)
    
    # Run multiple configurations
    configs = [
        {
            'name': 'Conservative',
            'entry_zscore': 2.2,
            'exit_zscore': 0.5,
            'formation_bars': formation_bars,
            'max_cointegration_pvalue': 0.05,
            'min_correlation': 0.6,
        },
        {
            'name': 'Base',
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'formation_bars': formation_bars,
            'max_cointegration_pvalue': 0.05,
            'min_correlation': 0.5,
        },
        {
            'name': 'Aggressive', 
            'entry_zscore': 1.8,
            'exit_zscore': 0.3,
            'formation_bars': formation_bars,
            'max_cointegration_pvalue': 0.10,
            'min_correlation': 0.4,
        },
    ]
    
    results_list = []
    
    for cfg in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {cfg['name']}")
        logger.info(f"{'='*50}")
        
        try:
            config = ContinuousBacktestConfig(
                initial_capital=10000.0,
                trading_fee=0.0005,  # 0.05% Binance Futures maker fee
                slippage=0.0010,     # 0.10% slippage estimate
                formation_bars=cfg.get('formation_bars', 200),
                entry_zscore=cfg['entry_zscore'],
                exit_zscore=cfg['exit_zscore'],
                stop_loss_zscore=3.5,
                max_cointegration_pvalue=cfg.get('max_cointegration_pvalue', 0.05),
                min_correlation=cfg.get('min_correlation', 0.5),
                min_half_life=2.0,
                max_half_life=100.0,
            )
            
            engine = ContinuousPairBacktest(config)
            results = engine.run(final_data, progress=False)
            
            # Handle both dict and object results
            if isinstance(results, dict):
                total_return = results.get('total_return_pct', 0)
                sharpe = results.get('sharpe_ratio', 0)
                max_dd = results.get('max_drawdown_pct', 0)
                n_trades = results.get('n_trades', 0)
                win_rate = results.get('win_rate', 0)
                pf = results.get('profit_factor', 0)
                trades = results.get('trades', [])
            else:
                total_return = results.total_return_pct
                sharpe = results.sharpe_ratio
                max_dd = results.max_drawdown_pct
                n_trades = results.n_trades
                win_rate = results.win_rate
                pf = results.profit_factor
                trades = results.trades
            
            results_list.append({
                'config': cfg['name'],
                'return': total_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'trades': n_trades,
                'win_rate': win_rate,
                'profit_factor': pf,
            })
            
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info(f"Sharpe Ratio: {sharpe:.2f}")
            logger.info(f"Max Drawdown: {max_dd:.2f}%")
            logger.info(f"Total Trades: {n_trades}")
            logger.info(f"Win Rate: {win_rate:.1%}")
            logger.info(f"Profit Factor: {pf:.2f}")
            
            # Print some trade details
            if trades:
                logger.info(f"\nSample trades:")
                for t in trades[:3]:
                    if hasattr(t, 'symbol1'):
                        logger.info(f"  {t.symbol1}/{t.symbol2}: {t.pnl_pct:+.2f}% ({t.exit_reason})")
                    else:
                        logger.info(f"  {t}")
            
        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("REAL DATA BACKTEST SUMMARY")
    logger.info("="*60)
    
    print(f"\n{'Config':<15} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>10}")
    print("-"*65)
    for r in results_list:
        print(f"{r['config']:<15} {r['return']:>+9.2f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% {r['trades']:>8} {r['win_rate']:>9.1%}")


def verify_no_lookahead_with_real_data(data: Dict[str, pd.DataFrame]) -> None:
    """Verify no lookahead bias with real data."""
    from quantbox.backtest.continuous_backtest import (
        ContinuousPairBacktest, ContinuousBacktestConfig
    )
    
    logger.info("\n" + "="*60)
    logger.info("LOOKAHEAD BIAS TEST WITH REAL DATA")
    logger.info("="*60)
    
    # Use symbols with enough data
    min_bars = 400
    valid_symbols = [s for s, df in data.items() if len(df) >= min_bars][:15]
    subset_data = {s: data[s] for s in valid_symbols}
    
    if len(subset_data) < 6:
        # Lower threshold
        min_bars = 250
        valid_symbols = [s for s, df in data.items() if len(df) >= min_bars][:15]
        subset_data = {s: data[s] for s in valid_symbols}
    
    if len(subset_data) < 6:
        logger.warning("Not enough data for lookahead test")
        return
    
    # Align data
    aligned_data = {}
    for s, df in subset_data.items():
        df = df.copy()
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        aligned_data[s] = df
    
    common_start = max(df.index.min() for df in aligned_data.values())
    common_end = min(df.index.max() for df in aligned_data.values())
    
    final_data = {}
    for s, df in aligned_data.items():
        aligned = df.loc[common_start:common_end].copy()
        if len(aligned) >= 200:
            final_data[s] = aligned
    
    if len(final_data) < 6:
        logger.warning("Not enough aligned data for lookahead test")
        return
    
    total_bars = len(next(iter(final_data.values())))
    logger.info(f"Testing with {len(final_data)} symbols, {total_bars} bars each")
    
    split_point = int(total_bars * 0.7)
    formation_bars = min(100, split_point - 30)
    
    # Split data
    first_half = {s: df.iloc[:split_point] for s, df in final_data.items()}
    
    config = ContinuousBacktestConfig(
        formation_bars=formation_bars,
        entry_zscore=1.8,
        exit_zscore=0.4,
        min_correlation=0.3,
        max_cointegration_pvalue=0.15,
        max_half_life=200.0,
    )
    
    # Run on first half only
    engine1 = ContinuousPairBacktest(config)
    results1 = engine1.run(first_half, progress=False)
    
    # Run on full data
    engine2 = ContinuousPairBacktest(config)
    results2 = engine2.run(final_data, progress=False)
    
    # Get trade counts
    if isinstance(results1, dict):
        trades1 = results1.get('n_trades', 0)
        all_trades = results2.get('trades', [])
    else:
        trades1 = results1.n_trades
        all_trades = results2.trades
    
    trades_first_period_full = sum(
        1 for t in all_trades 
        if hasattr(t, 'exit_bar') and t.exit_bar < split_point
    ) if all_trades else 0
    
    logger.info(f"Trades in first period (standalone): {trades1}")
    logger.info(f"Trades in first period (from full run): {trades_first_period_full}")
    
    # The trade count might differ slightly due to edge effects
    # but large differences indicate lookahead
    if abs(trades1 - trades_first_period_full) <= 1:
        logger.info("✅ No obvious lookahead bias detected")
    else:
        logger.warning(f"⚠️ Potential issue: trade counts differ by {abs(trades1 - trades_first_period_full)}")


def main():
    """Main test function."""
    
    # Try to load cached data first
    logger.info("Checking for cached Binance data...")
    data = load_cached_data()
    
    if len(data) < 10:
        logger.info("Insufficient cached data, downloading fresh data...")
        data = download_fresh_data(n_coins=30, days=30, interval="1h")
    
    if len(data) < 6:
        logger.error("Could not load or download sufficient data")
        logger.error("Please ensure binance.json exists with valid API credentials")
        return
    
    logger.info(f"Loaded {len(data)} symbols with real market data")
    
    # Sample data info
    sample_symbol = next(iter(data.keys()))
    sample_df = data[sample_symbol]
    logger.info(f"Sample data ({sample_symbol}):")
    logger.info(f"  Period: {sample_df.index.min()} to {sample_df.index.max()}")
    logger.info(f"  Bars: {len(sample_df)}")
    logger.info(f"  Columns: {list(sample_df.columns)}")
    
    # Run backtest with real data
    run_backtest_with_real_data(data)
    
    # Verify no lookahead
    verify_no_lookahead_with_real_data(data)
    
    logger.info("\n" + "="*60)
    logger.info("REAL DATA TESTING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
