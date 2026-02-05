"""
QuantBox Analysis Module

Tools for analyzing backtest results and optimizing strategy parameters.
"""

from quantbox.analysis.run_analysis import (
    run_comprehensive_analysis,
    verify_no_lookahead_bias,
    DataProvider,
    ParameterSensitivity,
    AnalysisResult
)

__all__ = [
    'run_comprehensive_analysis',
    'verify_no_lookahead_bias',
    'DataProvider',
    'ParameterSensitivity',
    'AnalysisResult'
]
