"""
Example usage of QuantBox framework.

This script demonstrates:
1. Data download and caching
2. Pair selection with filters
3. Backtesting
4. Parameter optimization
5. Results analysis
"""

import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_backtest():
    """Basic backtest example."""
    from quantbox import BacktestRunner, PairTradingStrategy
    from quantbox.backtest.runner import BacktestConfig
    from quantbox.data import DataDownloader
    
    logger.info("=" * 60)
    logger.info("Example 1: Basic Backtest")
    logger.info("=" * 60)
    
    # Download data
    downloader = DataDownloader()
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    logger.info(f"Downloading data for {symbols}")
    
    data = downloader.download_multiple(
        symbols=symbols,
        timeframe='1h',
        days=60
    )
    
    logger.info(f"Downloaded {len(data)} symbols")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        trading_fee=0.0005,
        slippage=0.001,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.5,
        formation_periods=120,
        trading_periods=24,
        max_pairs_per_window=5
    )
    
    # Run backtest
    runner = BacktestRunner(config)
    results = runner.run_backtest(data)
    
    # Print results
    metrics = results['metrics']
    logger.info("\n" + metrics.summary())
    
    return results


def example_with_filters():
    """Example with custom filter pipeline."""
    from quantbox import PairSelector, FilterPipeline
    from quantbox.strategy.filters import (
        CorrelationFilter,
        CointegrationFilter,
        HalfLifeFilter,
        HurstFilter
    )
    from quantbox.data import DataDownloader
    
    logger.info("=" * 60)
    logger.info("Example 2: Pair Selection with Filters")
    logger.info("=" * 60)
    
    # Download data
    downloader = DataDownloader()
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
    
    data = downloader.download_multiple(
        symbols=symbols,
        timeframe='1h',
        days=30
    )
    
    # Create filter pipeline
    pipeline = FilterPipeline([
        CorrelationFilter(min_correlation=0.6, max_correlation=0.98),
        CointegrationFilter(max_pvalue=0.1),
        HalfLifeFilter(min_halflife=3, max_halflife=60),
        HurstFilter(max_hurst=0.5),  # Mean-reverting
    ])
    
    # Select pairs
    selector = PairSelector(filter_pipeline=pipeline)
    candidates = selector.select_pairs(data)
    
    logger.info(f"\nFound {len(candidates)} candidate pairs:")
    for c in candidates[:5]:  # Top 5
        logger.info(f"  {c.symbol1}/{c.symbol2} - Score: {c.score:.4f}")
    
    # Get tradeable pairs
    tradeable = selector.select_tradeable_pairs(
        candidates,
        min_z_score=1.5,
        max_z_score=4.0
    )
    
    logger.info(f"\n{len(tradeable)} pairs ready to trade")
    
    return candidates


def example_optimization():
    """Parameter optimization example."""
    from quantbox.optimization import (
        DifferentialEvolutionOptimizer,
        SharpeObjective
    )
    from quantbox.data import DataDownloader
    
    logger.info("=" * 60)
    logger.info("Example 3: Parameter Optimization")
    logger.info("=" * 60)
    
    # Download data
    downloader = DataDownloader()
    data = downloader.download_multiple(
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
        timeframe='1h',
        days=60
    )
    
    # Define parameter bounds
    param_bounds = {
        'entry_zscore': (1.5, 3.0),
        'exit_zscore': (0.2, 1.0),
        'stop_loss_zscore': (2.5, 4.0),
    }
    
    # Create optimizer
    optimizer = DifferentialEvolutionOptimizer(
        param_bounds=param_bounds,
        objective_func=SharpeObjective(min_trades=5),
        population_size=10,
        max_iterations=15,  # Reduced for example
        maximize=True
    )
    
    logger.info("Running optimization...")
    result = optimizer.optimize(data)
    
    logger.info(f"\nBest Parameters:")
    for k, v in result.best_params.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"\nBest Sharpe Ratio: {result.best_score:.4f}")
    
    return result


def example_walk_forward():
    """Walk-forward analysis example."""
    from quantbox.backtest import WalkForwardAnalysis
    from quantbox.data import DataDownloader
    
    logger.info("=" * 60)
    logger.info("Example 4: Walk-Forward Analysis")
    logger.info("=" * 60)
    
    # Download data
    downloader = DataDownloader()
    data = downloader.download_multiple(
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        timeframe='1h',
        days=90
    )
    
    # Setup walk-forward
    wfa = WalkForwardAnalysis(
        train_periods=500,
        test_periods=100,
        step_periods=100,
        min_trades=3
    )
    
    # Run analysis
    logger.info("Running walk-forward analysis...")
    results = wfa.run(
        data=data,
        param_ranges={
            'entry_zscore': (1.5, 3.0),
            'exit_zscore': (0.2, 1.0),
        }
    )
    
    summary = results.get('summary', {})
    logger.info(f"\nWalk-Forward Summary:")
    logger.info(f"  Windows: {summary.get('num_windows', 0)}")
    logger.info(f"  Avg Test Return: {summary.get('avg_test_return', 0):.2f}%")
    logger.info(f"  Avg Efficiency: {summary.get('avg_efficiency', 0):.2f}")
    
    return results


def example_analysis():
    """Results analysis example."""
    from quantbox.backtest import BacktestAnalyzer
    
    logger.info("=" * 60)
    logger.info("Example 5: Results Analysis")
    logger.info("=" * 60)
    
    # Run a backtest first
    results = example_basic_backtest()
    
    # Create analyzer
    analyzer = BacktestAnalyzer(results)
    
    # Print summary
    logger.info(analyzer.summary())
    
    # Generate report (if matplotlib available)
    try:
        report_path = analyzer.generate_report(
            output_dir="reports",
            report_name="example_backtest"
        )
        logger.info(f"Report saved to: {report_path}")
    except Exception as e:
        logger.warning(f"Could not generate visual report: {e}")
    
    return analyzer


def main():
    """Run all examples."""
    logger.info("QuantBox Framework Examples")
    logger.info("=" * 60)
    
    try:
        # Basic backtest
        example_basic_backtest()
        
        # Pair selection
        example_with_filters()
        
        # Note: The following examples require more data/time
        # Uncomment to run:
        
        # example_optimization()
        # example_walk_forward()
        # example_analysis()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install dependencies with: pip install -e .[all]")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
