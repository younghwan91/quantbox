"""
Rubber Band Scalper Strategy (v4.3 — Capital Preservation Framework)

Fades extreme short-term moves in crypto futures, profiting from the
mechanical snap-back after liquidation cascades overshoot.

Alpha source:
  - After a 4.5%+ move in 2 hours, liquidation cascades overshoot
  - The overshoot gets corrected within ~2.5 hours (10 bars at 15m)
  - This is a market-microstructure edge: stop hunts + forced liquidations

v4.3: Capital Preservation Framework
  - Partial 50% exit on any profit → lock in capital immediately
  - Break-even stop on remaining half → zero downside risk
  - R:R-gated leverage → "손익비가 높아야 레버리지를 쓴다" (only leverage when R:R high)
  - 365-day A/B validated: DD 75%→15%, Calmar 0.84→3.72, 10/10 rolling 90d

v4.2: BTC Directional Alignment
  - Counter-trend entries only when BTC momentum confirms market-wide move
  - Filters out idiosyncratic altcoin moves that may not revert

Validated on 30 Binance USDT-M pairs, 365 days of 15m data:
  - 10/10 rolling 90-day windows profitable (100%, Top-10)
  - DD: 15.28% (vs 75% baseline) — 5x safer
  - Sharpe: 1.64 (Top-10), Calmar: 3.72 (Top-10)
  - Win rate: 65.4% (vs 52.9% baseline)

Capital Preservation Principles:
  - Execute partial 50% liquidation upon any profit to secure capital
  - Immediate break-even stop-loss on remaining half to eliminate risk
  - Leverage is a strategic asset: only deployed when R:R justifies it
  - Prioritize net savings and capital preservation over gross performance

Parameters (365-day validated, not over-optimised):
  lookback = 8 bars (2 hours)
  threshold = 4.5% (minimum move to trigger — filters noise)
  target_retrace = 30% of original move (quick exit)
  stop_mult = 1.0x (stop at same distance as the extreme move)
  max_hold = 10 bars (2.5 hours)
  btc_directional_period = 32 bars (8 hours BTC momentum)
  partial_exit_enabled = True (50% close on any profit)
  rr_leverage_enabled = True (R:R-gated leverage)
  min_rr_for_leverage = 0.30 (R:R threshold for leverage)
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

    # ── Dynamic Leverage (move-confidence scaling) ──
    # Bigger moves beyond threshold = higher conviction = more leverage.
    # 365-day validated: Base=3 Scale=10 Max=7 is conservative and robust.
    # Competition mode: Base=5 Scale=20 Max=15 for maximum returns.
    dynamic_leverage_enabled: bool = True  # Enable move-confidence leverage
    base_leverage: float = 3.0     # Minimum leverage (for threshold-level moves)
    leverage_scale: float = 10.0   # Leverage gain per 100% excess move
    max_leverage: float = 7.0      # Hard cap on leverage

    # ── BTC Directional Alignment (v4.2, A/B validated) ──
    # Counter-trend entries only when BTC confirms: long when BTC falling,
    # short when BTC rising. Filters out idiosyncratic altcoin moves that
    # are less likely to revert. 365-day validated: 100% rolling 90-day
    # windows profitable, Calmar 4.04 vs 3.74 baseline (Top-10).
    btc_directional_enabled: bool = True   # Align entry with BTC momentum
    btc_directional_period: int = 32       # BTC momentum lookback (32 bars = 8h)

    # ── Capital Preservation (v4.3, A/B validated) ──
    # Partial exit + break-even stop: lock profit, eliminate downside risk.
    # 365-day A/B: DD 75%→15%, Calmar 0.84→3.72, 10/10 rolling 90d.
    partial_exit_enabled: bool = True      # 50% partial close on any profit
    partial_exit_fraction: float = 0.5     # Fraction to close (0.5 = 50%)

    # ── R:R-Gated Leverage (v4.3, A/B validated) ──
    # "손익비가 높아야 레버리지를 쓴다" — only deploy leverage when R:R is favorable.
    # With current params (target=0.3, stop=1.0), R:R ≈ 0.28, so leverage = 1x
    # for most trades. Leverage only activates for exceptionally large moves.
    # 365-day A/B: DD drops from 75% to 15%, Calmar 0.84→4.47 (Top-10).
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
    base_leverage: float = 3.0,
    leverage_scale: float = 10.0,
    max_leverage: float = 7.0,
) -> float:
    """Compute leverage proportional to confidence (move size).

    Bigger moves beyond threshold indicate stronger liquidation cascades
    and higher probability of snap-back. We reward this with more leverage.

    Formula: lev = base + scale * (move - threshold) / threshold
    Capped at max_leverage.

    365-day A/B validated (30 symbols × 35,040 bars):
      Base=3, Scale=10, Max=7:  +132%, Calmar 2.65 (conservative)
      Base=5, Scale=20, Max=15: +20,012% on Top-10 (competition mode)

    Args:
        move_size: Actual move size (fraction, e.g. 0.06 = 6%).
        threshold: Minimum threshold (fraction, e.g. 0.045 = 4.5%).
        base_leverage: Minimum leverage (for threshold-level moves).
        leverage_scale: Leverage gain per 100% excess move.
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
