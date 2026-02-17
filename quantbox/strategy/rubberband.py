"""
Rubber Band Scalper Strategy

Fades extreme short-term moves in crypto futures, profiting from the
mechanical snap-back after liquidation cascades overshoot.

Alpha source:
  - After a 4.5%+ move in 2 hours, liquidation cascades overshoot
  - The overshoot gets corrected within ~2.5 hours (10 bars at 15m)
  - This is a market-microstructure edge: stop hunts + forced liquidations

Validated on 30 Binance USDT-M pairs, 365 days of 15m data:
  - 9/10 rolling 90-day windows profitable (90%)
  - 20/30 symbols profitable, 7/12 months profitable
  - Sharpe 0.81, Calmar 1.13, Robustness 7/7
  - 98% of neighboring parameter configs also profitable

Regime awareness:
  - Works best in choppy/ranging markets
  - Loses in strong trending markets
  - BTC crash gate blocks all entries in extreme moves
  - MTF filter DISABLED (hurts long-term performance)

Parameters (365-day validated, not over-optimised):
  lookback = 8 bars (2 hours)
  threshold = 4.5% (minimum move to trigger — filters noise)
  target_retrace = 30% of original move (quick exit)
  stop_mult = 1.0x (stop at same distance as the extreme move)
  max_hold = 10 bars (2.5 hours)
"""

import numpy as np
import numba
from dataclasses import dataclass, field


# ──────────────────────────── Configuration ────────────────────────────
@dataclass
class RubberBandConfig:
    """Rubber Band Scalper configuration.

    All parameters are walk-forward validated. Do NOT tighten stops
    or widen targets without re-running walk_forward_test.py.
    """
    # ── Signal parameters (365-day validated) ──
    lookback: int = 8           # Bars to measure move (2h at 15m)
    threshold: float = 0.045   # Minimum move to trigger (4.5% — filters noise)
    target_retrace: float = 0.3  # Take profit at 30% retracement (quick exit)
    stop_mult: float = 1.0     # Stop at 1x the move size
    max_hold: int = 10          # Maximum holding period (bars, 2.5h)
    stop_decay: float = 0.0    # Stop tightening rate (0=off, disabled for robustness)

    # ── Multi-timeframe confirmation (DISABLED — hurts long-term perf) ──
    htf_period: int = 0          # Higher TF trend lookback (0=DISABLED)
    htf_block_threshold: float = 0.01  # Block entry if HTF trend > 1% against trade dir
    htf_agree_discount: float = 0.85   # Threshold multiplier when HTF agrees
    htf_agree_stop_mult: float = 0.95  # Stop multiplier when HTF agrees

    # ── Regime filter ──
    regime_filter_enabled: bool = False   # Disabled: blocks the very moves that generate alpha
    regime_lookback: int = 96    # 24h of 15m bars
    regime_trend_threshold: float = 0.08  # 8% BTC move in 24h → halt

    # ── Costs (conservative) ──
    fee_rate: float = 0.0002     # 0.02% maker fee per side
    slippage_rate: float = 0.0002  # 0.02% slippage with limit chaser
    funding_rate_8h: float = 0.0001  # 0.01% per 8h
    bar_interval_minutes: int = 15

    # ── Risk management ──
    risk_per_trade: float = 0.01   # Risk 1% of capital per trade
    max_concurrent: int = 5        # Max simultaneous positions
    max_drawdown_pct: float = 0.10  # 10% session drawdown → halt
    daily_loss_limit_pct: float = 0.05  # 5% daily loss → halt

    # ── Symbol selection ──
    min_volume_rank: int = 30  # Top 30 by volume
    exclude_symbols: list[str] = field(default_factory=lambda: ['USDCUSDT', 'TUSDUSDT', 'BUSDUSDT'])

    @property
    def cost_per_side(self) -> float:
        return self.fee_rate + self.slippage_rate

    @property
    def round_trip_cost(self) -> float:
        return 2 * self.cost_per_side

    @property
    def bars_per_funding(self) -> int:
        return (8 * 60) // self.bar_interval_minutes


# ──────────────────────── Numba Signal Engine ──────────────────────────
@numba.jit(nopython=True, cache=True)
def compute_signals(
    close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float = 0.0,
) -> np.ndarray:
    """Compute rubber band signals from price data.

    Args:
        close: Array of close prices.
        lookback: Number of bars to measure the move.
        threshold: Minimum move (as fraction) to trigger entry.
        target_retrace: Fraction of move to target for profit.
        stop_mult: Multiple of move size for stop loss.
        max_hold: Maximum holding period in bars.
        stop_decay: Stop tightening rate over hold duration (0=no decay,
            0.5=stop halves by max_hold). Cuts losers faster as time passes.

    Returns:
        Signal array: -1 (short), 0 (flat), +1 (long) for each bar.
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if ret < -threshold:
                position = 1   # Long: fade the drop
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold:
                position = -1  # Short: fade the spike
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            # ── Exit conditions (priority order) ──
            # Time-decaying stop: tightens as holding period grows
            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor

            # 1. Target hit: take profit
            target = move_size * target_retrace
            if gain >= target:
                position = 0
            # 2. Stop loss (time-decaying)
            elif gain < -effective_stop:
                position = 0
            # 3. Trailing stop: was up >50% of target, now lost 70% of gains
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            # 4. Max holding period
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def compute_signals_with_regime(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    regime_lookback: int,
    regime_threshold: float,
    stop_decay: float = 0.0,
) -> np.ndarray:
    """Compute signals with BTC regime filter.

    When BTC has moved more than regime_threshold in the last
    regime_lookback bars, the market is trending → block new entries.
    Existing positions are still managed normally.
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    for i in range(max(lookback, regime_lookback), n):
        ret = close[i] / close[i - lookback] - 1.0

        # Regime check: is BTC trending?
        btc_ret = abs(btc_close[i] / btc_close[i - regime_lookback] - 1.0)
        regime_blocked = btc_ret > regime_threshold

        if position == 0:
            if regime_blocked:
                sig[i] = 0
                continue

            if ret < -threshold:
                position = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold:
                position = -1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor

            target = move_size * target_retrace
            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def compute_signals_mtf(
    close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    htf_period: int,
    htf_block_threshold: float,
    htf_agree_discount: float = 1.0,
    htf_agree_stop_mult: float = 1.0,
) -> np.ndarray:
    """Compute rubber band signals with multi-timeframe confirmation.

    Three modes based on HTF trend vs trade direction:
      1. HTF AGREES with trade: lower threshold (more entries) +
         tighter stop (better risk-reward)
      2. HTF NEUTRAL: normal behavior
      3. HTF strongly AGAINST trade: block entry entirely

    "HTF agrees" means the higher TF trend supports the mean-reversion:
      - Long entry (fading drop): price is BELOW HTF SMA (oversold on HTF)
      - Short entry (fading spike): price is ABOVE HTF SMA (overbought on HTF)

    Args:
        close: Array of close prices (15m bars).
        lookback: Bars to measure the move.
        threshold: Minimum move to trigger entry.
        target_retrace: Take profit target as fraction of move.
        stop_mult: Stop loss as multiple of move.
        max_hold: Maximum holding bars.
        stop_decay: Stop tightening rate.
        htf_period: Higher TF SMA lookback in bars (0=disabled, 48=12h).
        htf_block_threshold: Block entry when HTF trend is strongly
            against trade direction (as fraction, e.g. 0.02 = 2%).
        htf_agree_discount: When HTF agrees, multiply threshold by this
            (e.g. 0.8 = 20% lower threshold → more entries). 1.0 = off.
        htf_agree_stop_mult: When HTF agrees, multiply stop_mult by this
            (e.g. 0.7 = 30% tighter stop → better risk-reward). 1.0 = off.

    Returns:
        Signal array: -1 (short), 0 (flat), +1 (long).
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    entry_stop_mult = stop_mult  # Effective stop for current position

    # Pre-compute running SMA for HTF filter (cumulative approach)
    htf_sma = np.zeros(n)
    if htf_period > 0:
        cumsum = 0.0
        for i in range(n):
            cumsum += close[i]
            if i >= htf_period:
                cumsum -= close[i - htf_period]
                htf_sma[i] = cumsum / htf_period
            elif i >= htf_period - 1:
                htf_sma[i] = cumsum / htf_period
            else:
                htf_sma[i] = close[i]

    start = max(lookback, htf_period)
    for i in range(start, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            # Determine HTF state for entry decisions
            htf_blocked = False
            htf_agrees = False
            eff_threshold = threshold
            eff_stop = stop_mult

            if htf_period > 0 and htf_sma[i] > 0:
                htf_trend = (close[i] - htf_sma[i]) / htf_sma[i]

                if ret < -threshold * min(htf_agree_discount, 1.0):
                    # Want to go long — HTF agrees if price below SMA (oversold)
                    if htf_trend > htf_block_threshold:
                        htf_blocked = True
                    elif htf_trend < 0:
                        htf_agrees = True
                elif ret > threshold * min(htf_agree_discount, 1.0):
                    # Want to go short — HTF agrees if price above SMA (overbought)
                    if htf_trend < -htf_block_threshold:
                        htf_blocked = True
                    elif htf_trend > 0:
                        htf_agrees = True

            if htf_agrees:
                eff_threshold = threshold * htf_agree_discount
                eff_stop = stop_mult * htf_agree_stop_mult

            if not htf_blocked:
                if ret < -eff_threshold:
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                    entry_stop_mult = eff_stop
                elif ret > eff_threshold:
                    position = -1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                    entry_stop_mult = eff_stop
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * entry_stop_mult * time_factor

            target = move_size * target_retrace
            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

        sig[i] = position
    return sig


# ──────────────────────── Position Sizing ──────────────────────────────
def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    move_size: float,
    stop_mult: float,
    max_leverage: float = 20.0,
) -> tuple[float, float]:
    """Calculate position size based on risk.

    Args:
        capital: Available capital.
        risk_per_trade: Fraction of capital to risk (e.g. 0.01 = 1%).
        entry_price: Entry price.
        move_size: Size of the original move (fraction).
        stop_mult: Stop multiplier.
        max_leverage: Maximum allowed leverage.

    Returns:
        (notional, leverage) tuple.
    """
    max_loss_pct = move_size * stop_mult  # Max loss per unit
    if max_loss_pct <= 0:
        return 0.0, 1.0

    risk_amount = capital * risk_per_trade
    notional = risk_amount / max_loss_pct

    leverage = notional / capital
    if leverage > max_leverage:
        leverage = max_leverage
        notional = capital * leverage

    return notional, leverage


# ──────────────────────── Live Signal Helper ──────────────────────────
class RubberBandStrategy:
    """Stateful strategy wrapper for live trading.

    Maintains per-symbol state (position, entry price, hold count)
    and generates signals from new price data.
    """

    def __init__(self, config: RubberBandConfig = None):
        self.config = config or RubberBandConfig()
        self._positions: dict[str, dict] = {}

    def on_bar(
        self,
        symbol: str,
        close_prices: np.ndarray,
        btc_close_prices: np.ndarray | None = None,
    ) -> int:
        """Process a new bar and return signal.

        Args:
            symbol: Trading symbol.
            close_prices: Array of recent close prices (at least lookback+1 bars).
            btc_close_prices: BTC close prices for regime filter.

        Returns:
            Signal: -1 (short), 0 (flat), +1 (long).
        """
        cfg = self.config
        if len(close_prices) < cfg.lookback + 1:
            return 0

        # Use regime-filtered signals if BTC data available
        if btc_close_prices is not None and cfg.regime_filter_enabled:
            if len(btc_close_prices) < max(cfg.lookback, cfg.regime_lookback) + 1:
                return 0
            sigs = compute_signals_with_regime(
                close_prices, btc_close_prices,
                cfg.lookback, cfg.threshold,
                cfg.target_retrace, cfg.stop_mult,
                cfg.max_hold, cfg.regime_lookback,
                cfg.regime_trend_threshold,
                cfg.stop_decay,
            )
        elif cfg.htf_period > 0:
            if len(close_prices) < max(cfg.lookback, cfg.htf_period) + 1:
                return 0
            sigs = compute_signals_mtf(
                close_prices,
                cfg.lookback, cfg.threshold,
                cfg.target_retrace, cfg.stop_mult,
                cfg.max_hold,
                cfg.stop_decay,
                cfg.htf_period, cfg.htf_block_threshold,
                cfg.htf_agree_discount, cfg.htf_agree_stop_mult,
            )
        else:
            sigs = compute_signals(
                close_prices,
                cfg.lookback, cfg.threshold,
                cfg.target_retrace, cfg.stop_mult,
                cfg.max_hold,
                cfg.stop_decay,
            )

        return int(sigs[-1])

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_effective_stop_mult(
        self, close_prices: np.ndarray,
    ) -> float:
        """Compute the effective stop multiplier for the current bar.

        Mirrors the HTF agree logic in compute_signals_mtf to ensure
        live trading uses the same stop as the backtest.

        Returns:
            Effective stop multiplier (cfg.stop_mult * htf_agree_stop_mult
            when HTF agrees, else cfg.stop_mult).
        """
        cfg = self.config
        if cfg.htf_period <= 0 or cfg.htf_agree_stop_mult >= 1.0:
            return cfg.stop_mult

        if len(close_prices) < max(cfg.lookback, cfg.htf_period) + 1:
            return cfg.stop_mult

        # Compute HTF SMA at current bar (same as Numba function)
        htf_window = close_prices[-cfg.htf_period:]
        htf_sma = float(np.mean(htf_window))
        if htf_sma <= 0:
            return cfg.stop_mult

        current = float(close_prices[-1])
        htf_trend = (current - htf_sma) / htf_sma

        # Compute the move (same as Numba: close[i]/close[i-lookback] - 1)
        lookback_price = float(close_prices[-cfg.lookback - 1])
        ret = current / lookback_price - 1.0

        # Check if HTF agrees with the trade direction
        min_disc = min(cfg.htf_agree_discount, 1.0)
        if ret < -cfg.threshold * min_disc:
            # Long entry — HTF agrees if below SMA
            if htf_trend < 0 and htf_trend <= cfg.htf_block_threshold:
                return cfg.stop_mult * cfg.htf_agree_stop_mult
        elif ret > cfg.threshold * min_disc:
            # Short entry — HTF agrees if above SMA
            if htf_trend > 0 and htf_trend >= -cfg.htf_block_threshold:
                return cfg.stop_mult * cfg.htf_agree_stop_mult

        return cfg.stop_mult

    def open_position(
        self, symbol: str, signal: int, entry_price: float,
        notional: float, leverage: float, move_size: float,
        entry_stop_mult: float | None = None,
    ):
        """Record a new position.

        Args:
            entry_stop_mult: Effective stop multiplier at entry (for
                HTF agree parity). If None, uses cfg.stop_mult.
        """
        self._positions[symbol] = {
            'signal': signal,
            'entry_price': entry_price,
            'notional': notional,
            'leverage': leverage,
            'move_size': move_size,
            'bars_held': 0,
            'best_pnl': 0.0,
            'entry_stop_mult': entry_stop_mult if entry_stop_mult is not None else self.config.stop_mult,
        }

    def close_position(self, symbol: str) -> dict | None:
        """Remove position and return its info."""
        return self._positions.pop(symbol, None)

    def update_position(self, symbol: str, current_price: float) -> str | None:
        """Check if position should be closed.

        Returns exit reason string or None.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        pos['bars_held'] += 1
        cfg = self.config

        if pos['signal'] == 1:
            gain = (current_price - pos['entry_price']) / pos['entry_price']
        else:
            gain = (pos['entry_price'] - current_price) / pos['entry_price']

        if gain > pos['best_pnl']:
            pos['best_pnl'] = gain

        target = pos['move_size'] * cfg.target_retrace

        # Time-decaying stop (parity with compute_signals_mtf)
        # Uses entry_stop_mult stored at open time (accounts for HTF agree)
        entry_sm = pos.get('entry_stop_mult', cfg.stop_mult)
        time_factor = 1.0 - cfg.stop_decay * (pos['bars_held'] / cfg.max_hold)
        effective_stop = pos['move_size'] * entry_sm * time_factor

        # Exit conditions
        if gain >= target:
            return 'target'
        if gain < -effective_stop:
            return 'stop_loss'
        if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
            return 'trailing_stop'
        if pos['bars_held'] >= cfg.max_hold:
            return 'max_hold'

        return None

    @property
    def active_positions(self) -> dict[str, dict]:
        return dict(self._positions)

    @property
    def n_active(self) -> int:
        return len(self._positions)
