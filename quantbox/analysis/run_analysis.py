#!/usr/bin/env python3
"""
QuantBox Comprehensive Profitability Analysis

This script performs thorough backtesting and analysis:
1. Loads REAL market data from Binance Futures
2. Runs backtests with various parameter configurations
3. Analyzes profitability metrics
4. Validates for lookahead bias and cost accuracy
5. Generates detailed reports

IMPORTANT: This module uses ONLY real market data. No synthetic data is allowed.

Usage:
    python -m quantbox.analysis.run_analysis [--output results/]
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger('ProfitabilityAnalysis')


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    config_name: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade_pnl: float
    total_fees: float
    market_exposure: float
    
    def to_dict(self):
        return asdict(self)


class DataProvider:
    """
    Provides REAL market data for backtesting.
    
    This class loads actual Binance Futures data from the cache.
    NO synthetic data generation is available.
    """
    
    @staticmethod
    def load_real_data(
        symbols: List[str] = None,
        min_bars: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Load REAL market data from Binance Futures cache.
        
        Args:
            symbols: List of symbols to load. None = default symbols.
            min_bars: Minimum bars required per symbol.
            
        Returns:
            Dict mapping symbol names to OHLCV DataFrames
            
        Raises:
            ValueError: If no real data is available
        """
        from quantbox.data.real_data_loader import RealDataLoader
        
        loader = RealDataLoader()
        
        if symbols is None:
            # Use loader's default symbols
            data = loader.load_data(min_bars=min_bars, align_data=True)
        else:
            data = loader.load_data(symbols=symbols, min_bars=min_bars, align_data=True)
        
        if not data:
            raise ValueError(
                "No real market data available. "
                "Please run the data downloader to fetch Binance Futures data first."
            )
        
        n_bars = len(next(iter(data.values())))
        logger.info(f"Loaded REAL data: {len(data)} symbols, {n_bars} bars")
        
        return data


class ParameterSensitivity:
    """Analyze parameter sensitivity"""
    
    @staticmethod
    def generate_parameter_grid() -> List[Dict]:
        """Generate parameter combinations to test"""
        
        base_config = {
            'initial_capital': 10000.0,
            'trading_fee': 0.0005,
            'slippage': 0.001,
            'formation_bars': 240,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 3.5,
            'max_holding_bars': 100,
            'min_correlation': 0.6,
            'max_cointegration_pvalue': 0.05,
            'min_half_life': 2.0,
            'max_half_life': 50.0,
        }
        
        # Variations to test
        configs = [
            # Base case
            {**base_config, 'name': 'Base Case'},
            
            # Entry threshold sensitivity
            {**base_config, 'name': 'Lower Entry (1.5)', 'entry_zscore': 1.5},
            {**base_config, 'name': 'Higher Entry (2.5)', 'entry_zscore': 2.5},
            {**base_config, 'name': 'Higher Entry (3.0)', 'entry_zscore': 3.0},
            
            # Exit threshold sensitivity
            {**base_config, 'name': 'Tighter Exit (0.3)', 'exit_zscore': 0.3},
            {**base_config, 'name': 'Looser Exit (0.8)', 'exit_zscore': 0.8},
            
            # Formation period sensitivity
            {**base_config, 'name': 'Short Formation (120)', 'formation_bars': 120},
            {**base_config, 'name': 'Long Formation (360)', 'formation_bars': 360},
            
            # Stop loss sensitivity
            {**base_config, 'name': 'Tight Stop (3.0)', 'stop_loss_zscore': 3.0},
            {**base_config, 'name': 'Loose Stop (4.5)', 'stop_loss_zscore': 4.5},
            
            # Holding period sensitivity
            {**base_config, 'name': 'Short Hold (50)', 'max_holding_bars': 50},
            {**base_config, 'name': 'Long Hold (200)', 'max_holding_bars': 200},
            
            # Higher costs (stress test)
            {**base_config, 'name': 'High Costs', 'trading_fee': 0.001, 'slippage': 0.002},
            
            # Lower correlation threshold
            {**base_config, 'name': 'Lower Corr (0.5)', 'min_correlation': 0.5},
        ]
        
        return configs


def run_single_backtest(data: Dict[str, pd.DataFrame], config: Dict) -> AnalysisResult:
    """Run a single backtest with given configuration"""
    
    from quantbox.backtest.continuous_backtest import (
        ContinuousPairBacktest,
        ContinuousBacktestConfig
    )
    
    config_name = config.pop('name', 'Unnamed')
    
    bt_config = ContinuousBacktestConfig(**config)
    engine = ContinuousPairBacktest(bt_config)
    
    results = engine.run(data, progress=False)
    
    metrics = results['metrics']
    
    return AnalysisResult(
        config_name=config_name,
        total_return_pct=metrics.get('total_return_pct', 0),
        sharpe_ratio=metrics.get('sharpe_ratio', 0),
        max_drawdown_pct=metrics.get('max_drawdown', 0),
        win_rate=metrics.get('win_rate', 0),
        profit_factor=metrics.get('profit_factor', 0),
        n_trades=metrics.get('n_trades', 0),
        avg_trade_pnl=metrics.get('avg_trade', 0),
        total_fees=metrics.get('total_costs', 0),
        market_exposure=metrics.get('in_market_pct', 0)
    )


def run_comprehensive_analysis(
    data: Dict[str, pd.DataFrame] = None,
    configs: List[Dict] = None
) -> pd.DataFrame:
    """Run comprehensive backtesting analysis using REAL market data only."""
    
    logger.info("="*60)
    logger.info("QUANTBOX COMPREHENSIVE PROFITABILITY ANALYSIS")
    logger.info("="*60)
    
    # Load REAL market data if not provided
    if data is None:
        data = DataProvider.load_real_data(min_bars=500)
    
    # Generate parameter grid if not provided
    if configs is None:
        configs = ParameterSensitivity.generate_parameter_grid()
    
    logger.info(f"Testing {len(configs)} parameter configurations")
    logger.info(f"Data: {len(data)} symbols, {len(next(iter(data.values())))} bars")
    
    # Run backtests
    results = []
    
    for i, config in enumerate(configs):
        config_name = config.get('name', f'Config {i+1}')
        logger.info(f"\nRunning [{i+1}/{len(configs)}]: {config_name}")
        
        try:
            result = run_single_backtest(data.copy(), config.copy())
            results.append(result)
            
            logger.info(
                f"  Return: {result.total_return_pct:+.2f}% | "
                f"Sharpe: {result.sharpe_ratio:.2f} | "
                f"Trades: {result.n_trades}"
            )
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append(AnalysisResult(
                config_name=config_name,
                total_return_pct=0, sharpe_ratio=0, max_drawdown_pct=0,
                win_rate=0, profit_factor=0, n_trades=0,
                avg_trade_pnl=0, total_fees=0, market_exposure=0
            ))
    
    # Create results DataFrame
    results_df = pd.DataFrame([r.to_dict() for r in results])
    
    return results_df


def print_analysis_report(results_df: pd.DataFrame):
    """Print formatted analysis report"""
    
    print("\n" + "="*80)
    print("PROFITABILITY ANALYSIS REPORT")
    print("="*80)
    
    # Sort by Sharpe ratio
    sorted_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    print("\n📊 RESULTS RANKED BY SHARPE RATIO")
    print("-"*80)
    
    for i, row in sorted_df.iterrows():
        return_emoji = "🟢" if row['total_return_pct'] > 0 else "🔴"
        
        print(f"\n{return_emoji} {row['config_name']}")
        print(f"   Return: {row['total_return_pct']:+.2f}%  |  "
              f"Sharpe: {row['sharpe_ratio']:.2f}  |  "
              f"Max DD: {row['max_drawdown_pct']:.2f}%")
        print(f"   Trades: {row['n_trades']}  |  "
              f"Win Rate: {row['win_rate']:.1f}%  |  "
              f"Profit Factor: {row['profit_factor']:.2f}")
        print(f"   Avg Trade: ${row['avg_trade_pnl']:.2f}  |  "
              f"Total Costs: ${row['total_fees']:.2f}  |  "
              f"Exposure: {row['market_exposure']:.1f}%")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    profitable = results_df[results_df['total_return_pct'] > 0]
    
    print(f"\nProfitable configurations: {len(profitable)}/{len(results_df)}")
    print(f"Average return: {results_df['total_return_pct'].mean():+.2f}%")
    print(f"Best return: {results_df['total_return_pct'].max():+.2f}%")
    print(f"Worst return: {results_df['total_return_pct'].min():+.2f}%")
    print(f"Average Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
    print(f"Average trades: {results_df['n_trades'].mean():.1f}")
    
    # Best configuration
    best = sorted_df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION: {best['config_name']}")
    print(f"   Expected annual return: ~{best['total_return_pct'] * 4:.1f}% (if 1000 bars ≈ 3 months)")
    
    return sorted_df


def verify_no_lookahead_bias():
    """Run lookahead bias verification using REAL market data."""
    
    print("\n" + "="*80)
    print("LOOKAHEAD BIAS VERIFICATION")
    print("="*80)
    
    from quantbox.backtest.validation import (
        LookaheadBiasValidator,
        TransactionCostValidator,
        load_test_data
    )
    from quantbox.backtest.continuous_backtest import (
        ContinuousPairBacktest,
        ContinuousBacktestConfig
    )
    
    # Load REAL market data
    data = load_test_data(min_symbols=2, min_bars=300)
    symbols = list(data.keys())[:2]
    data = {symbols[0]: data[symbols[0]], symbols[1]: data[symbols[1]]}
    
    # Run backtest
    config = ContinuousBacktestConfig(formation_bars=100)
    engine = ContinuousPairBacktest(config)
    
    # Validate
    validator = LookaheadBiasValidator()
    is_valid, violations = validator.ran_comprehensive_test(engine, data)
    
    if is_valid:
        print("✅ No lookahead bias detected")
    else:
        print("❌ Lookahead bias violations:")
        for v in violations:
            print(f"   - {v}")
    
    # Verify transaction costs
    results = engine.run(data, progress=False)
    cost_validator = TransactionCostValidator()
    
    trades = results['state'].trades
    all_costs_valid = True
    
    for trade in trades:
        validation = cost_validator.validate_trade_costs(trade, config)
        if not all(validation.values()):
            all_costs_valid = False
            break
    
    if all_costs_valid:
        print("✅ Transaction costs correctly calculated")
    else:
        print("❌ Transaction cost errors detected")
    
    return is_valid and all_costs_valid


def main():
    """Main entry point"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           QUANTBOX PROFITABILITY ANALYSIS                     ║
    ║   Comprehensive Backtesting & Validation System               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Verify no lookahead bias
    print("\n[1/3] Verifying system integrity...")
    integrity_ok = verify_no_lookahead_bias()
    
    if not integrity_ok:
        print("\n⚠️  WARNING: System integrity issues detected!")
        print("    Results may not be reliable. Fix issues before proceeding.")
    
    # 2. Run comprehensive analysis
    print("\n[2/3] Running profitability analysis...")
    results_df = run_comprehensive_analysis()
    
    # 3. Print report
    print("\n[3/3] Generating report...")
    sorted_results = print_analysis_report(results_df)
    
    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"analysis_{timestamp}.csv"
    sorted_results.to_csv(results_file, index=False)
    print(f"\n📁 Results saved to: {results_file}")
    
    # Final verdict
    print("\n" + "="*80)
    avg_return = sorted_results['total_return_pct'].mean()
    profitable_pct = (sorted_results['total_return_pct'] > 0).mean() * 100
    
    if avg_return > 0 and profitable_pct > 50:
        print("✅ VERDICT: System shows potential profitability")
        print(f"   {profitable_pct:.0f}% of configurations are profitable")
    else:
        print("⚠️  VERDICT: System requires further optimization")
        print("   Consider adjusting parameters or market conditions")
    
    print("="*80)
    
    return sorted_results


if __name__ == "__main__":
    results = main()
