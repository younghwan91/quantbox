"""
Rubber Band Scalper Strategy (v5.0 — Regime-Switch Dual Entry)

Fades extreme short-term moves in crypto futures, profiting from the
mechanical snap-back after liquidation cascades overshoot.

Alpha source:
  - After a 4.5%+ move in 2 hours, liquidation cascades overshoot
  - The overshoot gets corrected within ~2.5 hours (10 bars at 15m)
  - This is a market-microstructure edge: stop hunts + forced liquidations

v5.0: Regime-Switch Dual Entry
  - Detects BTC regime via SMA: BTC > SMA(96) = Risk-On, BTC < SMA(96) = Risk-Off
  - Risk-Off (BTC bearish) → Sympathy Pullback: long when alt dips + BTC falling
  - Risk-On (BTC bullish) → Classic Pullback: long when alt dips in its own uptrend
  - Each algo is high-conviction in its home regime → doubled confidence
  - 365-day A/B validated: Calmar 6.70 (vs 4.98 v4.4), Sharpe 1.89
  - WR 61.7%, R:R 1.25, DD 17.4%, 13/13 rolling 90d windows

v4.4: Stop/Target Optimization
  - Tighter stop (0.9x) + wider target (40%) → R:R 1.10
  - Flat 1.5x leverage (sweet spot), partial exit disabled

Parameters (v5.0, 365-day A/B validated):
  lookback = 8 bars (2 hours)
  threshold = 4.5% (minimum move to trigger)
  target_retrace = 40% of original move
  stop_mult = 0.9x (tighter stop)
  max_hold = 10 bars (2.5 hours)
  entry_mode = 'regime_switch' (dual-algo regime switching)
  btc_regime_period = 96 bars (24h SMA for BTC regime detection)
  asset_trend_period = 192 bars (48h SMA for asset trend in Risk-On)
  btc_directional_period = 32 bars (8h BTC momentum for Risk-Off)
  base_leverage = 1.5x (flat, Calmar 6.70)
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
    # ── Signal parameters (365-day validated, v5.0) ──
    lookback: int = 8           # Bars to measure move (2h at 15m)
    threshold: float = 0.045   # Minimum move to trigger (4.5% — filters noise)
    target_retrace: float = 0.4  # Take profit at 40% retracement (R:R 1.10)
    stop_mult: float = 0.9     # Stop at 0.9x the move size (DD 17.4%)
    max_hold: int = 10          # Maximum holding period (bars, 2.5h)
    stop_decay: float = 0.0    # Stop tightening rate (0=off, disabled for robustness)

    # ── Entry Mode (v5.0, A/B validated) ──
    # 'regime_switch': BTC regime detection → dual algo dispatch.
    #   Risk-Off (BTC < SMA) → sympathy pullback (BTC directional)
    #   Risk-On  (BTC > SMA) → classic pullback (asset trend)
    #   Calmar 6.70, Sharpe 1.89, WR 61.7%, R:R 1.25, 13/13 rolling.
    # 'btc_directional': v4.4 legacy. Long only when BTC falling.
    # 'raw': No filter (baseline).
    entry_mode: str = 'regime_switch'      # 'regime_switch' | 'btc_directional' | 'raw'
    btc_regime_period: int = 96            # BTC SMA period for regime detection (96 = 24h)
    asset_trend_period: int = 192          # Asset SMA for classic pullback (192 = 48h)

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

    # ── Dynamic Leverage (v5.0: flat 1.5x, A/B validated) ──
    # With regime-switch + stop=0.9, target=0.4, Calmar peaks at 1.5x (6.70).
    # Higher leverage scales return but DD grows: 2x=6.92, 2.5x=6.94, 3x=6.65.
    # 1.5x is the conservative sweet spot for live trading.
    dynamic_leverage_enabled: bool = True  # Enable leverage pipeline
    base_leverage: float = 1.5     # Flat leverage for all trades
    leverage_scale: float = 0.0    # No confidence scaling (flat leverage)
    max_leverage: float = 1.5      # Aligned with base (safety cap)

    # ── BTC Directional Alignment (v4.2, A/B validated) ──
    # Counter-trend entries only when BTC confirms: long when BTC falling,
    # short when BTC rising. Filters out idiosyncratic altcoin moves that
    # are less likely to revert. 365-day validated: 100% rolling 90-day
    # windows profitable, Calmar 4.04 vs 3.74 baseline (Top-10).
    btc_directional_enabled: bool = True   # Align entry with BTC momentum
    btc_directional_period: int = 32       # BTC momentum lookback (32 bars = 8h)

    # ── Capital Preservation (v4.4: DISABLED — hurts at tighter stop) ──
    # v4.3 partial exit hurt performance with stop=0.9 target=0.4:
    # Without partial: Calmar 4.96, With partial: Calmar 3.75 (-24%).
    # Tighter stop (0.9 vs 1.0) already limits DD to 12.7%.
    partial_exit_enabled: bool = False     # Disabled: hurts with tighter stop
    partial_exit_fraction: float = 0.5     # Fraction to close (0.5 = 50%)

    # ── R:R-Gated Leverage (v4.4, safety gate) ──
    # With stop=0.9, target=0.4: theoretical R:R = 0.44 > 0.30, gate always
    # passes. Kept as safety: if params changed to worse R:R, it disables leverage.
    rr_leverage_enabled: bool = True       # Gate leverage by R:R ratio
    min_rr_for_leverage: float = 0.30      # Minimum R:R to allow leverage

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
def compute_signals_with_btc_directional(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
) -> np.ndarray:
    """Compute signals with BTC directional alignment.

    Only enter counter-trend when BTC momentum confirms the move is
    market-wide (not idiosyncratic):
      - Long (price crashed): enter only if BTC also falling (market selloff → likely revert)
      - Short (price surged): enter only if BTC also rising (market euphoria → likely revert)

    This filters out idiosyncratic altcoin moves that may not revert.
    365-day A/B validated: Calmar +8%, 100% rolling 90-day windows profitable.

    Args:
        close: Alt close prices.
        btc_close: BTCUSDT close prices (same timeframe).
        lookback: Bars to measure the move.
        threshold: Minimum move to trigger entry.
        target_retrace: Take profit fraction.
        stop_mult: Stop loss multiple.
        max_hold: Maximum holding period.
        stop_decay: Stop tightening rate.
        btc_mom_period: BTC momentum lookback (32 bars = 8h recommended).

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
    start_bar = max(lookback, btc_mom_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                # BTC momentum check
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                if ret < -threshold and btc_mom < 0:
                    # Long: alt crashed AND BTC falling → correlated selloff, likely reverts
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold and btc_mom > 0:
                    # Short: alt surged AND BTC rising → correlated euphoria, likely reverts
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


# ────────── Regime-Switch Dual Entry (v5.0) ──────────
@numba.jit(nopython=True, cache=True)
def compute_signals_regime_switch(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    btc_regime_period: int,
    asset_trend_period: int,
) -> np.ndarray:
    """Regime-switching dual entry: different algo per BTC regime.

    BTC Regime Detection: BTC price vs SMA(btc_regime_period)
      - BTC < SMA → "Risk-Off" → SYMPATHY PULLBACK
        (Long when alt dips AND BTC is falling short-term)
      - BTC > SMA → "Risk-On" → CLASSIC PULLBACK
        (Long when alt dips AND alt's own trend is UP)

    Each algo is high-conviction in its home regime.
    365-day A/B validated: Calmar 6.70, Sharpe 1.89, WR 61.7%, R:R 1.25.
    13/13 rolling 90-day windows profitable.

    Args:
        close: Alt close prices.
        btc_close: BTCUSDT close prices (same timeframe).
        lookback: Bars to measure the move.
        threshold: Minimum move to trigger entry.
        target_retrace: Take profit fraction.
        stop_mult: Stop loss multiple.
        max_hold: Maximum holding period.
        stop_decay: Stop tightening rate.
        btc_mom_period: BTC momentum lookback for Risk-Off (32 bars = 8h).
        btc_regime_period: BTC SMA period for regime detection (96 = 24h).
        asset_trend_period: Asset SMA period for classic pullback (192 = 48h).

    Returns:
        Signal array: -1 (short), 0 (flat), +1 (long).
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute BTC SMA for regime detection
    btc_sma = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += btc_close[i]
        if i >= btc_regime_period:
            s -= btc_close[i - btc_regime_period]
        if i >= btc_regime_period - 1:
            btc_sma[i] = s / btc_regime_period

    # Pre-compute asset SMA for classic pullback (risk-on regime)
    asset_sma = np.full(n, np.nan)
    s2 = 0.0
    for i in range(n):
        s2 += close[i]
        if i >= asset_trend_period:
            s2 -= close[i - asset_trend_period]
        if i >= asset_trend_period - 1:
            asset_sma[i] = s2 / asset_trend_period

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_mom_period, btc_regime_period, asset_trend_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                # Determine BTC regime
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]  # BTC above SMA = bull

                if is_risk_on:
                    # ═══ RISK-ON: Classic Pullback ═══
                    # BTC bullish → alt dips are pullbacks in uptrend
                    if np.isnan(asset_sma[i]):
                        continue
                    is_uptrend = close[i] > asset_sma[i]

                    if ret < -threshold and is_uptrend:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and not is_uptrend:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                else:
                    # ═══ RISK-OFF: Sympathy Pullback ═══
                    # BTC bearish → use BTC directional (proven v4.4)
                    btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                    if ret < -threshold and btc_mom < 0:
                        position = 1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > threshold and btc_mom > 0:
                        position = -1
                        entry_price = close[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
        else:
            # ═══ EXIT LOGIC (same regardless of regime) ═══
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


# ────────── Partial Exit + Break-Even Stop (v4.3) ──────────
@numba.jit(nopython=True, cache=True)
def compute_signals_with_partial_exit(
    close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float = 0.0,
) -> np.ndarray:
    """Compute signals with partial exit + break-even stop.

    Capital preservation framework:
      Phase 1 (full position): on ANY profit → close 50%, transition to Phase 2.
      Phase 2 (half position): break-even stop replaces original stop.
        Remaining half runs to target or gets stopped at break-even ($0 risk).

    Output: ±1.0 (full position), ±0.5 (half position after partial close), 0.0 (flat).
    The backtest core treats signal magnitude as position size.

    Args:
        close: Close price array.
        lookback: Bars to measure the move.
        threshold: Minimum move to trigger entry.
        target_retrace: Take profit fraction of move.
        stop_mult: Stop loss multiple.
        max_hold: Maximum holding period.
        stop_decay: Stop tightening rate.

    Returns:
        Signal array with fractional positions.
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0       # Direction: +1, -1, or 0
    phase = 0           # 0=flat, 1=full position, 2=half position
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0

    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1.0

        if phase == 0:
            # ── Entry logic (same as compute_signals) ──
            if ret < -threshold:
                position = 1
                phase = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold:
                position = -1
                phase = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0

        elif phase == 1:
            # ── Full position: check exits, then partial close ──
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

            # Priority: full exit conditions checked FIRST
            if gain >= target:
                position = 0
                phase = 0
            elif gain < -effective_stop:
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0
            elif gain > 0:
                # ANY profit → partial close 50%, enter Phase 2
                phase = 2
                best_pnl = gain

        elif phase == 2:
            # ── Half position: break-even stop, target still active ──
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            target = move_size * target_retrace

            if gain >= target:
                position = 0
                phase = 0
            elif gain <= 0:
                # Break-even stop: $0 risk on remaining half
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0

        # Output signal: magnitude encodes position size
        if phase == 0:
            sig[i] = 0.0
        elif phase == 1:
            sig[i] = float(position)        # ±1.0
        elif phase == 2:
            sig[i] = float(position) * 0.5  # ±0.5

    return sig


@numba.jit(nopython=True, cache=True)
def compute_signals_partial_exit_btc_directional(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
) -> np.ndarray:
    """Partial exit + break-even stop + BTC directional alignment.

    Combines v4.2 BTC directional gate with v4.3 partial exit framework:
      - Entry: only when BTC momentum confirms (v4.2)
      - Exit: 50% partial close on any profit, break-even stop on remainder (v4.3)

    Output: ±1.0 (full), ±0.5 (half after partial close), 0.0 (flat).
    """
    n = len(close)
    sig = np.zeros(n)
    position = 0
    phase = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_mom_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if phase == 0:
            if abs(ret) > threshold:
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

                if ret < -threshold and btc_mom < 0:
                    position = 1
                    phase = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold and btc_mom > 0:
                    position = -1
                    phase = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0

        elif phase == 1:
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
                phase = 0
            elif gain < -effective_stop:
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0
            elif gain > 0:
                phase = 2
                best_pnl = gain

        elif phase == 2:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            target = move_size * target_retrace

            if gain >= target:
                position = 0
                phase = 0
            elif gain <= 0:
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0

        if phase == 0:
            sig[i] = 0.0
        elif phase == 1:
            sig[i] = float(position)
        elif phase == 2:
            sig[i] = float(position) * 0.5

    return sig


# ────────── R:R-Gated Leverage (v4.3) ──────────
@numba.jit(nopython=True, cache=True)
def compute_rr_gated_leverage(
    move_size: float,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    cost_per_side: float,
    base_leverage: float,
    leverage_scale: float,
    max_leverage: float,
    min_rr: float = 0.5,
) -> float:
    """Compute leverage gated by risk-reward ratio.

    Don't use leverage on trades with poor R:R. Only scale up when the
    expected gain/loss ratio justifies the leverage risk.

    "손익비가 높아야 레버리지를 쓴다"

    Args:
        move_size: Actual move that triggered entry.
        threshold: Minimum trigger threshold.
        target_retrace: Fraction of move targeted.
        stop_mult: Stop loss multiple of move.
        cost_per_side: Trading cost per side (fee + slippage).
        base_leverage: Min leverage when R:R is sufficient.
        leverage_scale: Scale factor for excess moves.
        max_leverage: Hard cap.
        min_rr: Minimum R:R ratio required for leverage. Below this → 1x.

    Returns:
        Leverage (1.0 if R:R insufficient, scaled if sufficient).
    """
    if move_size <= 0 or threshold <= 0:
        return 1.0

    round_trip = cost_per_side * 2.0
    expected_gain = move_size * target_retrace - round_trip
    expected_loss = move_size * stop_mult + round_trip

    if expected_loss <= 0 or expected_gain <= 0:
        return 1.0

    rr = expected_gain / expected_loss

    if rr < min_rr:
        return 1.0  # No leverage — R:R too low

    # R:R sufficient: scale by excess move (same as confidence leverage)
    excess = (move_size - threshold) / threshold
    lev = base_leverage + leverage_scale * max(excess, 0.0)
    return min(lev, max_leverage)


@numba.jit(nopython=True, cache=True)
def apply_rr_gated_leverage(
    close: np.ndarray,
    signals: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    cost_per_side: float,
    base_leverage: float,
    leverage_scale: float,
    max_leverage: float,
    min_rr: float = 0.5,
) -> np.ndarray:
    """Scale raw signals by R:R-gated leverage (Numba).

    Applies compute_rr_gated_leverage at each bar. Signals ±1 or ±0.5
    (from partial exit) get multiplied by the leverage only when R:R
    meets the minimum threshold.

    Args:
        close: Close price array.
        signals: Raw signals (±1, ±0.5, or 0).
        lookback: Move measurement lookback.
        threshold: Entry threshold.
        target_retrace: Take profit fraction.
        stop_mult: Stop loss multiple.
        cost_per_side: Fee + slippage per side.
        base_leverage: Min leverage when R:R met.
        leverage_scale: Excess move scale.
        max_leverage: Hard cap.
        min_rr: Minimum R:R for leverage.

    Returns:
        Leveraged signals array.
    """
    n = len(close)
    lev_signals = np.empty(n, dtype=np.float64)

    for i in range(n):
        s = signals[i]
        if s == 0.0 or i < lookback:
            lev_signals[i] = s
            continue

        move = abs(close[i] / close[i - lookback] - 1.0)

        round_trip = cost_per_side * 2.0
        expected_gain = move * target_retrace - round_trip
        expected_loss = move * stop_mult + round_trip

        if expected_loss <= 0 or expected_gain <= 0:
            lev_signals[i] = s  # Keep raw signal, no leverage
            continue

        rr = expected_gain / expected_loss

        if rr < min_rr:
            lev_signals[i] = s  # No leverage — R:R too low
            continue

        excess = (move - threshold) / threshold
        lev = base_leverage + leverage_scale * max(excess, 0.0)
        if lev > max_leverage:
            lev = max_leverage
        if lev < base_leverage:
            lev = base_leverage

        # Scale signal magnitude (preserves direction & partial fraction)
        if s > 0:
            lev_signals[i] = abs(s) * lev
        else:
            lev_signals[i] = -abs(s) * lev

    return lev_signals


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


# ──────────────────── Confidence Leverage ─────────────────────────────
def compute_confidence_leverage(
    move_size: float,
    threshold: float,
    base_leverage: float = 1.5,
    leverage_scale: float = 0.0,
    max_leverage: float = 1.5,
) -> float:
    """Compute leverage proportional to confidence (move size).

    v4.4: Flat 1.5x leverage for all trades. Confidence scaling disabled
    (scale=0). At stop=0.9, target=0.4, 1.5x is the sweet spot where
    Calmar peaks at 4.98 (higher than 1x=4.96).

    Formula: lev = base + scale * (move - threshold) / threshold
    Capped at max_leverage.

    Args:
        move_size: Actual move size (fraction, e.g. 0.06 = 6%).
        threshold: Minimum threshold (fraction, e.g. 0.045 = 4.5%).
        base_leverage: Flat leverage (1.5x for all trades).
        leverage_scale: Leverage gain per 100% excess move (0=flat).
        max_leverage: Hard cap on leverage.

    Returns:
        Computed leverage (float, between base_leverage and max_leverage).
    """
    if move_size <= 0 or threshold <= 0:
        return base_leverage
    excess = (move_size - threshold) / threshold
    lev = base_leverage + leverage_scale * max(excess, 0.0)
    return min(lev, max_leverage)


@numba.jit(nopython=True, cache=True)
def apply_confidence_leverage(
    close: np.ndarray,
    signals: np.ndarray,
    lookback: int,
    threshold: float,
    base_leverage: float,
    leverage_scale: float,
    max_leverage: float,
) -> np.ndarray:
    """Scale raw ±1 signals by move-confidence leverage (Numba).

    For backtest use: converts signal ±1 to ±leverage.
    The vectorized backtest treats signal magnitude as position size,
    so signal=±5 means 5x leveraged position.

    Args:
        close: Close price array.
        signals: Raw signals (±1 or 0).
        lookback: Bars to measure the move.
        threshold: Minimum threshold (fraction).
        base_leverage: Minimum leverage.
        leverage_scale: Scale factor.
        max_leverage: Hard cap.

    Returns:
        Leveraged signals array.
    """
    n = len(close)
    lev_signals = np.empty(n, dtype=np.float64)

    for i in range(n):
        s = signals[i]
        if s == 0.0 or i < lookback:
            lev_signals[i] = s
            continue

        # Compute move size at this bar
        move = abs(close[i] / close[i - lookback] - 1.0)
        excess = (move - threshold) / threshold
        lev = base_leverage + leverage_scale * max(excess, 0.0)
        if lev > max_leverage:
            lev = max_leverage
        if lev < base_leverage:
            lev = base_leverage

        # Scale direction by leverage
        if s > 0:
            lev_signals[i] = lev
        else:
            lev_signals[i] = -lev

    return lev_signals


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

        # ═══ v5.0 Regime-Switch Dual Entry (highest priority) ═══
        if btc_close_prices is not None and cfg.entry_mode == 'regime_switch':
            min_bars = max(
                cfg.lookback, cfg.btc_directional_period,
                cfg.btc_regime_period, cfg.asset_trend_period,
            ) + 1
            if len(btc_close_prices) < min_bars or len(close_prices) < min_bars:
                return 0
            sigs = compute_signals_regime_switch(
                close_prices, btc_close_prices,
                cfg.lookback, cfg.threshold,
                cfg.target_retrace, cfg.stop_mult,
                cfg.max_hold, cfg.stop_decay,
                cfg.btc_directional_period,
                cfg.btc_regime_period,
                cfg.asset_trend_period,
            )
            return int(sigs[-1])

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
        elif btc_close_prices is not None and cfg.btc_directional_enabled:
            min_bars = max(cfg.lookback, cfg.btc_directional_period) + 1
            if len(btc_close_prices) < min_bars or len(close_prices) < min_bars:
                return 0
            if cfg.partial_exit_enabled:
                sigs = compute_signals_partial_exit_btc_directional(
                    close_prices, btc_close_prices,
                    cfg.lookback, cfg.threshold,
                    cfg.target_retrace, cfg.stop_mult,
                    cfg.max_hold, cfg.stop_decay,
                    cfg.btc_directional_period,
                )
            else:
                sigs = compute_signals_with_btc_directional(
                    close_prices, btc_close_prices,
                    cfg.lookback, cfg.threshold,
                    cfg.target_retrace, cfg.stop_mult,
                    cfg.max_hold, cfg.stop_decay,
                    cfg.btc_directional_period,
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
            'partial_closed': False,   # v4.3: track partial exit state
        }

    def close_position(self, symbol: str) -> dict | None:
        """Remove position and return its info."""
        return self._positions.pop(symbol, None)

    def update_position(self, symbol: str, current_price: float) -> str | None:
        """Check if position should be closed or partially closed.

        v4.3: When partial_exit_enabled, returns 'partial_close' on first
        profit, then uses break-even stop for remaining half.

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

        if not pos.get('partial_closed', False):
            # ── Phase 1: Full position ──
            entry_sm = pos.get('entry_stop_mult', cfg.stop_mult)
            time_factor = 1.0 - cfg.stop_decay * (pos['bars_held'] / cfg.max_hold)
            effective_stop = pos['move_size'] * entry_sm * time_factor

            # Full exit conditions (priority order)
            if gain >= target:
                return 'target'
            if gain < -effective_stop:
                return 'stop_loss'
            if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                return 'trailing_stop'
            if pos['bars_held'] >= cfg.max_hold:
                return 'max_hold'

            # Partial close: any profit triggers 50% exit
            if cfg.partial_exit_enabled and gain > 0:
                pos['partial_closed'] = True
                pos['best_pnl'] = gain
                return 'partial_close'
        else:
            # ── Phase 2: Half position with break-even stop ──
            if gain >= target:
                return 'target'
            if gain <= 0:
                return 'breakeven_stop'
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
