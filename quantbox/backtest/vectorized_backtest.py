"""
Vectorized Backtest Engine for Single-Asset Strategies

Ultra-fast backtesting using Numba-compiled signal processing and
PnL calculation. Designed for rapid strategy iteration.

Features:
  - Numba-compiled inner loop (~100K bars/second)
  - Realistic cost model (fee + slippage + funding)
  - Position sizing with risk management
  - Multi-asset portfolio aggregation
  - Walk-forward testing support

Usage:
    from quantbox.backtest.vectorized_backtest import (
        VectorizedBacktest, BacktestConfig
    )
    from quantbox.strategy.rubberband import compute_signals

    config = BacktestConfig(initial_capital=10000)
    bt = VectorizedBacktest(config)
    signals = compute_signals(close, 8, 0.025, 0.5, 1.0, 8)
    result = bt.run_single(close, signals)
    portfolio = bt.run_portfolio(all_data, signal_func)
"""

import numpy as np
import numba
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BacktestConfig:
    """Configuration for vectorized backtest."""
    initial_capital: float = 10000.0
    cost_per_side: float = 0.0004    # 0.04% per side
    funding_per_bar: float = 0.01 / 100 / 32  # 0.01%/8h → per 15m bar
    bar_interval_minutes: int = 15
    risk_per_trade: float = 0.01     # 1% risk per trade
    max_leverage: float = 20.0
    max_concurrent: int = 5


@dataclass
class BacktestResult:
    """Results from a single-asset backtest."""
    symbol: str
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    n_trades: int
    bars_in_market: int
    profit_factor: float
    avg_trade_return: float
    equity_curve: np.ndarray
    trades: list[dict] = field(default_factory=list)


@numba.jit(nopython=True, cache=True)
def _backtest_core_partial(
    close: np.ndarray,
    signals: np.ndarray,
    cost_per_side: float,
    funding_per_bar: float,
) -> tuple:
    """Core backtest loop with delta-based costs for partial exits (Numba).

    Handles fractional signal magnitudes (e.g., ±0.5 from partial close).
    Cost is based on position DELTA (|new - old|), not close+reopen.
    Trades are counted as full lifecycle (entry to final flat).

    Identical to _backtest_core for non-partial scenarios (0↔±1↔∓1).
    """
    n = len(close)
    equity = np.ones(n)
    trade_returns = np.zeros(1000)
    position = 0.0
    prev_signal = 0.0
    n_trades = 0
    n_wins = 0
    bars_in_market = 0
    peak = 1.0
    max_dd = 0.0
    trade_entry_eq = 1.0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(1, n):
        ret = close[i] / close[i - 1] - 1.0
        bar_pnl = position * ret

        new_pos = signals[i]
        if new_pos != prev_signal:
            # Delta-based cost: only charge for the CHANGE in position
            cost = abs(new_pos - prev_signal) * cost_per_side
            bar_pnl -= cost

            # Trade close: signal goes to 0 from non-zero
            if new_pos == 0.0 and prev_signal != 0.0:
                eq_after = equity[i - 1] * (1.0 + bar_pnl)
                trade_ret = eq_after / trade_entry_eq - 1.0
                if n_trades < len(trade_returns):
                    trade_returns[n_trades] = trade_ret
                n_trades += 1
                if trade_ret > 0:
                    n_wins += 1
                    gross_profit += trade_ret
                else:
                    gross_loss += abs(trade_ret)

            # Trade open: signal goes from 0 to non-zero
            if prev_signal == 0.0 and new_pos != 0.0:
                trade_entry_eq = equity[i - 1] * (1.0 + bar_pnl)

        if position != 0.0:
            bar_pnl -= abs(position) * funding_per_bar
            bars_in_market += 1

        equity[i] = equity[i - 1] * (1.0 + bar_pnl)
        position = new_pos
        prev_signal = new_pos

        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd

    if position != 0.0:
        trade_ret = equity[-1] / trade_entry_eq - 1.0
        if n_trades < len(trade_returns):
            trade_returns[n_trades] = trade_ret
        n_trades += 1
        if trade_ret > 0:
            n_wins += 1
            gross_profit += trade_ret
        else:
            gross_loss += abs(trade_ret)

    total_return = equity[-1] / equity[0] - 1.0

    rets = np.empty(n - 1)
    for i in range(n - 1):
        rets[i] = equity[i + 1] / equity[i] - 1.0
    mean_r = np.mean(rets)
    std_r = np.std(rets)
    sharpe = mean_r / std_r * np.sqrt(96.0 * 365.25) if std_r > 0 else 0.0

    win_rate = n_wins / n_trades * 100.0 if n_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
    avg_ret = 0.0
    if n_trades > 0:
        s = 0.0
        for j in range(min(n_trades, len(trade_returns))):
            s += trade_returns[j]
        avg_ret = s / n_trades

    return (equity, total_return, sharpe, max_dd, win_rate,
            n_trades, bars_in_market, profit_factor, avg_ret,
            trade_returns[:min(n_trades, len(trade_returns))])


@numba.jit(nopython=True, cache=True)
def _backtest_core(
    close: np.ndarray,
    signals: np.ndarray,
    cost_per_side: float,
    funding_per_bar: float,
) -> tuple:
    """Core backtest loop (Numba-compiled).

    Returns:
        (equity_curve, total_return, sharpe, max_dd, win_rate,
         n_trades, bars_in_market, profit_factor, avg_trade_return,
         trade_returns array)
    """
    n = len(close)
    equity = np.ones(n)
    trade_returns = np.zeros(1000)  # pre-allocate
    position = 0.0
    prev_signal = 0.0
    n_trades = 0
    n_wins = 0
    bars_in_market = 0
    peak = 1.0
    max_dd = 0.0
    trade_entry_eq = 1.0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(1, n):
        ret = close[i] / close[i - 1] - 1.0
        bar_pnl = position * ret

        new_pos = signals[i]
        if new_pos != prev_signal:
            cost = abs(prev_signal) * cost_per_side + abs(new_pos) * cost_per_side
            bar_pnl -= cost

            if prev_signal != 0.0:
                eq_after = equity[i - 1] * (1.0 + bar_pnl)
                trade_ret = eq_after / trade_entry_eq - 1.0
                if n_trades < len(trade_returns):
                    trade_returns[n_trades] = trade_ret
                n_trades += 1
                if trade_ret > 0:
                    n_wins += 1
                    gross_profit += trade_ret
                else:
                    gross_loss += abs(trade_ret)

            if new_pos != 0.0:
                trade_entry_eq = equity[i - 1] * (1.0 + bar_pnl)

        if position != 0.0:
            bar_pnl -= abs(position) * funding_per_bar
            bars_in_market += 1

        equity[i] = equity[i - 1] * (1.0 + bar_pnl)
        position = new_pos
        prev_signal = new_pos

        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd

    # Close final position
    if position != 0.0:
        trade_ret = equity[-1] / trade_entry_eq - 1.0
        if n_trades < len(trade_returns):
            trade_returns[n_trades] = trade_ret
        n_trades += 1
        if trade_ret > 0:
            n_wins += 1
            gross_profit += trade_ret
        else:
            gross_loss += abs(trade_ret)

    total_return = equity[-1] / equity[0] - 1.0

    # Sharpe (annualized from 15m bars)
    rets = np.empty(n - 1)
    for i in range(n - 1):
        rets[i] = equity[i + 1] / equity[i] - 1.0
    mean_r = np.mean(rets)
    std_r = np.std(rets)
    sharpe = mean_r / std_r * np.sqrt(96.0 * 365.25) if std_r > 0 else 0.0

    win_rate = n_wins / n_trades * 100.0 if n_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
    avg_ret = 0.0
    if n_trades > 0:
        s = 0.0
        for j in range(min(n_trades, len(trade_returns))):
            s += trade_returns[j]
        avg_ret = s / n_trades

    return (equity, total_return, sharpe, max_dd, win_rate,
            n_trades, bars_in_market, profit_factor, avg_ret,
            trade_returns[:min(n_trades, len(trade_returns))])


class VectorizedBacktest:
    """Fast vectorized backtester for single-asset strategies."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run_single(
        self,
        close: np.ndarray,
        signals: np.ndarray,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run backtest on a single asset.

        Args:
            close: Close price array.
            signals: Signal array (-1, 0, +1).
            symbol: Symbol name for reporting.

        Returns:
            BacktestResult with all metrics.
        """
        equity, total_ret, sharpe, max_dd, wr, n_trades, bars_in, pf, avg_ret, _ = (
            _backtest_core(
                close.astype(np.float64),
                signals.astype(np.float64),
                self.config.cost_per_side,
                self.config.funding_per_bar,
            )
        )

        return BacktestResult(
            symbol=symbol,
            total_return_pct=total_ret * 100,
            sharpe=sharpe,
            max_drawdown_pct=max_dd * 100,
            win_rate=wr,
            n_trades=n_trades,
            bars_in_market=bars_in,
            profit_factor=pf,
            avg_trade_return=avg_ret * 100,
            equity_curve=equity,
        )

    def run_single_partial(
        self,
        close: np.ndarray,
        signals: np.ndarray,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run backtest with delta-based cost model for partial exits.

        Uses _backtest_core_partial which charges cost on position DELTA
        (|new - old|) instead of close+reopen. Required for partial
        exit signals (±0.5). Identical results for non-partial scenarios.
        """
        equity, total_ret, sharpe, max_dd, wr, n_trades, bars_in, pf, avg_ret, _ = (
            _backtest_core_partial(
                close.astype(np.float64),
                signals.astype(np.float64),
                self.config.cost_per_side,
                self.config.funding_per_bar,
            )
        )

        return BacktestResult(
            symbol=symbol,
            total_return_pct=total_ret * 100,
            sharpe=sharpe,
            max_drawdown_pct=max_dd * 100,
            win_rate=wr,
            n_trades=n_trades,
            bars_in_market=bars_in,
            profit_factor=pf,
            avg_trade_return=avg_ret * 100,
            equity_curve=equity,
        )

    def run_portfolio(
        self,
        data: dict[str, pd.DataFrame],
        signal_func: Callable[[np.ndarray], np.ndarray],
        btc_close: np.ndarray | None = None,
        signal_func_with_regime: Callable | None = None,
        leverage_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> dict:
        """Run portfolio backtest across multiple assets.

        Equal-weight allocation across all assets.

        Args:
            data: Symbol -> DataFrame with 'Close' column.
            signal_func: Function(close) -> signals.
            btc_close: BTC close prices for regime filter.
            signal_func_with_regime: Function(close, btc_close) -> signals.
            leverage_func: Optional function(close, raw_signals) -> leveraged_signals.
                Applied after signal generation. Use apply_confidence_leverage
                from rubberband.py to scale signals by move-confidence leverage.

        Returns:
            Dict with portfolio metrics, per-asset results, equity curve.
        """
        symbols = sorted(data.keys())
        n_bars = min(len(df) for df in data.values())
        n_symbols = len(symbols)

        per_asset: list[BacktestResult] = []
        portfolio_eq = np.ones(n_bars)

        for sym in symbols:
            close = data[sym]['Close'].values[:n_bars].astype(np.float64)

            if signal_func_with_regime is not None and btc_close is not None:
                sigs = signal_func_with_regime(close, btc_close[:n_bars])
            else:
                sigs = signal_func(close)

            # Apply leverage scaling (e.g., move-confidence leverage)
            if leverage_func is not None:
                sigs = leverage_func(close, sigs)

            result = self.run_single(close, sigs, symbol=sym)
            per_asset.append(result)

            # Equal-weight portfolio
            portfolio_eq *= result.equity_curve[:n_bars] ** (1.0 / n_symbols)

        # Portfolio metrics
        port_ret = (portfolio_eq[-1] - 1.0) * 100
        port_peak = np.maximum.accumulate(portfolio_eq)
        port_dd = np.max((port_peak - portfolio_eq) / port_peak) * 100

        rets = np.diff(portfolio_eq) / portfolio_eq[:-1]
        port_sharpe = (
            np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25)
            if np.std(rets) > 0 else 0.0
        )

        positive_count = sum(1 for r in per_asset if r.total_return_pct > 0)

        return {
            'portfolio_return_pct': port_ret,
            'portfolio_sharpe': port_sharpe,
            'portfolio_max_dd_pct': port_dd,
            'portfolio_equity': portfolio_eq,
            'pct_symbols_positive': positive_count / n_symbols * 100,
            'n_symbols': n_symbols,
            'total_trades': sum(r.n_trades for r in per_asset),
            'avg_win_rate': np.mean([r.win_rate for r in per_asset]),
            'per_asset': {r.symbol: r for r in per_asset},
        }

    def run_dd_managed_portfolio(
        self,
        data: dict[str, pd.DataFrame],
        signal_func: Callable[[np.ndarray], np.ndarray],
        selected_symbols: list[str],
        vol_lookback: int = 2880,
        inverse_vol_weight: bool = True,
        dd_dampening_threshold: float = 0.0,
        dd_dampening_scale: float = 0.5,
        leverage_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> dict:
        """Run portfolio backtest with drawdown management.

        Features:
          - Inverse-volatility weighting (high vol → lower weight)
          - DD dampening: when portfolio DD > threshold, reduce
            exposure by dd_dampening_scale factor

        Args:
            data: Symbol -> DataFrame with 'Close' column.
            signal_func: Function(close) -> signals.
            selected_symbols: Pre-selected symbols to trade.
            vol_lookback: Bars to compute realized vol for weighting.
            inverse_vol_weight: Weight by inverse realized vol vs equal.
            dd_dampening_threshold: Portfolio DD level to start reducing
                exposure (0=disabled). E.g. 0.05 = dampen at 5% DD.
            dd_dampening_scale: Exposure multiplier when DD exceeds
                threshold. 0.5 = halve positions. Applied linearly:
                full reduction at 2x threshold.

        Returns:
            Dict with portfolio metrics and equity curve.
        """
        from quantbox.strategy.vol_selector import compute_realized_vol

        n_symbols = len(selected_symbols)
        if n_symbols == 0:
            return {'portfolio_return_pct': 0.0, 'portfolio_sharpe': 0.0,
                    'portfolio_max_dd_pct': 0.0, 'portfolio_equity': np.ones(1)}

        n_bars = min(len(data[s]) for s in selected_symbols)

        # Pre-compute per-asset bar returns from equity curves
        per_asset_results = {}
        bar_rets = {}
        for sym in selected_symbols:
            close = data[sym]['Close'].values[:n_bars].astype(np.float64)
            sigs = signal_func(close)
            if leverage_func is not None:
                sigs = leverage_func(close, sigs)
            result = self.run_single(close, sigs, symbol=sym)
            per_asset_results[sym] = result
            eq = result.equity_curve[:n_bars]
            r = np.zeros(n_bars)
            for j in range(1, n_bars):
                r[j] = eq[j] / eq[j - 1] - 1.0
            bar_rets[sym] = r

        # Compute inverse-vol weights
        if inverse_vol_weight and n_symbols > 1:
            vols = np.zeros(n_symbols)
            for idx, sym in enumerate(selected_symbols):
                close_arr = data[sym]['Close'].values[:n_bars].astype(np.float64)
                start = max(0, n_bars - vol_lookback)
                vols[idx] = compute_realized_vol(close_arr[start:n_bars], np.sqrt(96 * 365.25))
            # Inverse-vol weight (avoid zero division)
            inv_vols = np.where(vols > 1e-8, 1.0 / vols, 0.0)
            total_inv = inv_vols.sum()
            if total_inv > 0:
                weights = inv_vols / total_inv
            else:
                weights = np.ones(n_symbols) / n_symbols
        else:
            weights = np.ones(n_symbols) / n_symbols

        # Build portfolio equity with DD dampening
        portfolio_eq = np.ones(n_bars)
        peak = 1.0

        for j in range(1, n_bars):
            # Portfolio bar return (weighted sum)
            port_bar_ret = 0.0
            for idx, sym in enumerate(selected_symbols):
                port_bar_ret += weights[idx] * bar_rets[sym][j]

            # DD dampening: reduce exposure when in drawdown
            if dd_dampening_threshold > 0 and peak > 0:
                current_dd = (peak - portfolio_eq[j - 1]) / peak
                if current_dd > dd_dampening_threshold:
                    # Linear scale: full dampening at 2x threshold
                    excess = min(current_dd - dd_dampening_threshold,
                                 dd_dampening_threshold)
                    dampening = 1.0 - (1.0 - dd_dampening_scale) * (
                        excess / dd_dampening_threshold)
                    port_bar_ret *= dampening

            portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + port_bar_ret)
            if portfolio_eq[j] > peak:
                peak = portfolio_eq[j]

        # Metrics
        port_ret = (portfolio_eq[-1] - 1.0) * 100
        port_peak = np.maximum.accumulate(portfolio_eq)
        port_dd = np.max((port_peak - portfolio_eq) / port_peak) * 100

        rets_arr = np.diff(portfolio_eq) / portfolio_eq[:-1]
        port_sharpe = (
            np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(96 * 365.25)
            if np.std(rets_arr) > 0 else 0.0
        )

        calmar = port_ret / port_dd if port_dd > 0 else 999.0
        positive_count = sum(1 for s in selected_symbols
                             if per_asset_results[s].total_return_pct > 0)
        total_trades = sum(per_asset_results[s].n_trades
                           for s in selected_symbols)

        return {
            'portfolio_return_pct': port_ret,
            'portfolio_sharpe': port_sharpe,
            'portfolio_max_dd_pct': port_dd,
            'portfolio_equity': portfolio_eq,
            'calmar': calmar,
            'weights': {sym: float(weights[i])
                        for i, sym in enumerate(selected_symbols)},
            'pct_symbols_positive': positive_count / n_symbols * 100,
            'n_symbols': n_symbols,
            'total_trades': total_trades,
            'avg_win_rate': np.mean([per_asset_results[s].win_rate
                                     for s in selected_symbols]),
            'per_asset': per_asset_results,
        }

    def walk_forward(
        self,
        data: dict[str, pd.DataFrame],
        signal_func: Callable,
        is_bars: int = 2880,
        oos_bars: int = 1440,
        step_bars: int = 1440,
    ) -> list[dict]:
        """Walk-forward analysis.

        Args:
            data: Symbol -> DataFrame.
            signal_func: Signal generation function.
            is_bars: In-sample window size.
            oos_bars: Out-of-sample window size.
            step_bars: Step between windows.

        Returns:
            List of per-window results.
        """
        symbols = sorted(data.keys())
        n_bars = min(len(df) for df in data.values())

        windows = []
        start = 0
        while start + is_bars + oos_bars <= n_bars:
            windows.append({
                'is_start': start,
                'is_end': start + is_bars,
                'oos_start': start + is_bars,
                'oos_end': start + is_bars + oos_bars,
            })
            start += step_bars

        results = []
        for wi, w in enumerate(windows):
            sym_rets = []
            sym_sharpes = []
            for sym in symbols:
                close = data[sym]['Close'].values.astype(np.float64)
                sigs = signal_func(close)
                r = self.run_single(
                    close[w['oos_start']:w['oos_end']],
                    sigs[w['oos_start']:w['oos_end']],
                    symbol=sym,
                )
                sym_rets.append(r.total_return_pct)
                sym_sharpes.append(r.sharpe)

            results.append({
                'window': wi + 1,
                'oos_start': w['oos_start'],
                'oos_end': w['oos_end'],
                'avg_return_pct': np.mean(sym_rets),
                'avg_sharpe': np.mean(sym_sharpes),
                'pct_positive': np.mean([r > 0 for r in sym_rets]) * 100,
            })

        return results
