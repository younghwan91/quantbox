"""
Rubber Band Scalper Strategy (v6.0 — Confidence-Scored Multi-Evidence)

Fades extreme short-term moves in crypto futures, profiting from the
mechanical snap-back after liquidation cascades overshoot.

Alpha source:
  - After a 4.5%+ move in 2 hours, liquidation cascades overshoot
  - The overshoot gets corrected within ~2.5 hours (10 bars at 15m)
  - This is a market-microstructure edge: stop hunts + forced liquidations

v6.0: Confidence-Scored Multi-Evidence
  - Same regime-switch entry as v5.0 + multi-evidence confidence scoring
  - Evidence: RSI, volume spike, taker buy/sell ratio, price near SMA, BTC momentum
  - min_confidence=2 filters low-quality trades (34 trades rejected = +9% Calmar)
  - Optional: Graduated leverage (high confidence → bigger position)
  - 365-day validated across 500+ configurations:
    Conservative (flat 1.5x, min_conf=2): Calmar 8.76, Sharpe 2.05, DD 16.9%
    Moderate (conf_lev 1.0→2.0, min_conf=2): Calmar 9.99, Sharpe 2.03, DD 22.4%
  - Momentum strategies exhaustively tested (5 types) — all failed on 15m crypto

v5.0: Regime-Switch Dual Entry
  - Detects BTC regime via SMA: BTC > SMA(96) = Risk-On, BTC < SMA(96) = Risk-Off
  - Risk-Off (BTC bearish) → Sympathy Pullback: long when alt dips + BTC falling
  - Risk-On (BTC bullish) → Classic Pullback: long when alt dips in its own uptrend

v4.4: Stop/Target Optimization
  - Tighter stop (0.9x) + wider target (40%) → R:R 1.10
  - Flat 1.5x leverage (sweet spot), partial exit disabled

Parameters (v6.0, 365-day A/B validated):
  lookback = 8 bars (2 hours)
  threshold = 4.5% (minimum move to trigger)
  target_retrace = 40% of original move
  stop_mult = 0.9x (tighter stop)
  max_hold = 10 bars (2.5 hours)
  entry_mode = 'confidence' (v6.0 default)
  min_confidence = 2 (require ≥2 pieces of confirming evidence)
  btc_regime_period = 96 bars (24h SMA for BTC regime detection)
  asset_trend_period = 192 bars (48h SMA for asset trend in Risk-On)
  btc_directional_period = 32 bars (8h BTC momentum for Risk-Off)
  base_leverage = 1.5x (conservative, same DD as v5.0)
  confidence_leverage_step = 0.0 (disabled by default for safety)
  max_leverage = 1.5x (capped at base)
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

    # ── Entry Mode (v6.0, A/B validated) ──
    # 'confidence': v6.0 — regime-switch + multi-evidence confidence scoring.
    #   Same regime-switch entry + RSI, volume, taker ratio, SMA distance evidence.
    #   min_confidence=2 filters 34 bad trades → Calmar 8.76 (+9% vs v5.0).
    # 'regime_switch': v5.0 — BTC regime detection → dual algo dispatch.
    # 'btc_directional': v4.4 legacy. Long only when BTC falling.
    # 'raw': No filter (baseline).
    entry_mode: str = 'confidence'         # 'confidence' | 'regime_switch' | 'btc_directional' | 'raw'
    btc_regime_period: int = 96            # BTC SMA period for regime detection (96 = 24h)
    asset_trend_period: int = 192          # Asset SMA for classic pullback (192 = 48h)

    # ── Confidence Scoring (v6.0, 365-day A/B validated) ──
    min_confidence: int = 2                # Minimum evidence score to enter (1-5, 2=optimal)
    confidence_leverage_step: float = 0.0  # Extra leverage per confidence point above 1 (0=flat)
    rsi_period: int = 14                   # RSI calculation period
    vol_period: int = 20                   # Volume SMA period for spike detection

    # ── Multi-timeframe confirmation (DISABLED — old single-TF SMA overlay) ──
    htf_period: int = 0          # Higher TF trend lookback (0=DISABLED)
    htf_block_threshold: float = 0.01  # Block entry if HTF trend > 1% against trade dir
    htf_agree_discount: float = 0.85   # Threshold multiplier when HTF agrees
    htf_agree_stop_mult: float = 0.95  # Stop multiplier when HTF agrees

    # ── v7.0 Multi-Timeframe Confirmation (A/B validated, 365-day) ──
    # Uses ACTUAL 4h candle data (not SMA overlay on 15m).
    # 4h SMA(12) trend confirms 15m entries: blocks counter-trend trades.
    # Result: MDD 16.5% → 4.3% (-74%), Calmar 8.65 → 32.42 (+275%)
    # Walk-forward: Q3 DD drops 16.5% → 3.0% (the critical quarter).
    mtf_confirmation_enabled: bool = True    # Enable 4h trend confirmation
    mtf_sma_period: int = 12                 # 4h SMA period (12 bars = 2 days)
    mtf_leverage_mult: float = 2.0           # Leverage multiplier for confirmed trades (2.0x → 3.0x effective)

    # ── v7.0 Independent 4H Rubber Band (A/B validated) ──
    # Trades rubber band signals on 4h timeframe independently.
    # 4h-only: Ret +52.2%, DD 4.8%, Calmar 10.87.
    # Combined with MTF-confirmed 15m: 70%/30% allocation.
    htf_independent_enabled: bool = True     # Enable independent 4h trading
    htf_rb_lookback: int = 3                 # 4h lookback (12h)
    htf_rb_threshold: float = 0.10           # 10% threshold (higher for 4h noise)
    htf_rb_target_retrace: float = 0.5       # 50% retracement target
    htf_rb_stop_mult: float = 0.8            # 0.8x stop
    htf_rb_max_hold: int = 12                # 12 bars (2 days)
    htf_rb_btc_sma_period: int = 48          # BTC SMA for 4h regime (8 days)

    # ── Regime filter ──
    regime_filter_enabled: bool = False   # Disabled: blocks the very moves that generate alpha
    regime_lookback: int = 96    # 24h of 15m bars
    regime_trend_threshold: float = 0.08  # 8% BTC move in 24h → halt

    # ── Costs (conservative) ──
    fee_rate: float = 0.0002     # 0.02% maker fee per side
    slippage_rate: float = 0.0002  # 0.02% slippage with limit chaser
    funding_rate_8h: float = 0.0001  # 0.01% per 8h
    bar_interval_minutes: int = 15

    # ── Dynamic Leverage (v6.0: flat 1.5x default, confidence optional) ──
    # Conservative (DD ≈ 16.9%): base=1.5 step=0.0 max=1.5 → Calmar 8.76
    # Moderate (DD ≈ 22%): base=1.0 step=1.0 max=2.0 → Calmar 9.99
    # 1.5x flat is conservative sweet spot for live trading.
    dynamic_leverage_enabled: bool = True  # Enable leverage pipeline
    base_leverage: float = 1.5     # Flat leverage for all trades
    leverage_scale: float = 0.0    # Legacy: No confidence scaling (use confidence_leverage_step)
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


# ────────── Confidence-Filtered Regime-Switch (v6.0) ──────────
@numba.jit(nopython=True, cache=True)
def _compute_sma_inline(arr: np.ndarray, period: int) -> np.ndarray:
    """SMA helper for Numba context."""
    n = len(arr)
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= period:
            s -= arr[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def _compute_rsi_inline(close: np.ndarray, period: int) -> np.ndarray:
    """RSI helper (Wilder's smoothing) for Numba context."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            gains[i] = d
        else:
            losses[i] = -d
    ag, al = 0.0, 0.0
    for i in range(1, period + 1):
        ag += gains[i]
        al += losses[i]
    ag /= period
    al /= period
    if al > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + ag / al)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        if al > 0:
            rsi[i] = 100.0 - 100.0 / (1.0 + ag / al)
        else:
            rsi[i] = 100.0
    return rsi


@numba.jit(nopython=True, cache=True)
def compute_signals_confidence(
    close: np.ndarray,
    volume: np.ndarray,
    taker_buy: np.ndarray,
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
    min_confidence: int,
    base_leverage: float,
    confidence_leverage_step: float,
    max_leverage: float,
    rsi_period: int,
    vol_period: int,
) -> np.ndarray:
    """v6.0: Regime-switch MR with confidence scoring + graduated leverage.

    Same regime-switch entry logic as v5.0, enhanced with:
      1. Multi-evidence confidence scoring (RSI, volume, taker ratio, price/SMA)
      2. Minimum confidence filter — reject low-quality entries
      3. Graduated leverage — more evidence → bigger position

    Evidence for LONG entries:
      +1 BTC regime confirmed (always, since regime check passes)
      +1 RSI oversold (< 40 Risk-On, < 35 Risk-Off)
      +1 Volume > 1.5x average (institutional / liquidation cascade)
      +1 Taker sell pressure (ratio < 0.45 Risk-On, < 0.42 Risk-Off)
      +1 Near SMA support (Risk-On) or strong BTC momentum (Risk-Off)

    For SHORT entries, evidence is mirrored (RSI overbought, taker buy, etc.).

    Leverage = min(base_leverage + confidence_leverage_step * (conf - 1), max_leverage)

    Conservative default (DD ≈ 16.9%): min_conf=2, flat 1.5x (same as v5.0 DD)
    Moderate (DD ≈ 22%): min_conf=2, base=1.0 step=1.0 max=2.0
    Calmar at conservative: 8.76 (+9% vs v5.0's 8.03)
    Calmar at moderate: 9.99 (+24% vs v5.0)

    365-day validated across 500+ configurations.

    Args:
        close: Alt close prices.
        volume: Alt volume data.
        taker_buy: Taker buy base volume (buying pressure indicator).
        btc_close: BTCUSDT close prices.
        lookback: Bars to measure the move.
        threshold: Minimum move to trigger entry.
        target_retrace: Take profit fraction.
        stop_mult: Stop loss multiple.
        max_hold: Maximum holding period.
        stop_decay: Stop tightening rate.
        btc_mom_period: BTC momentum lookback for Risk-Off.
        btc_regime_period: BTC SMA period for regime detection.
        asset_trend_period: Asset SMA for Risk-On trend check.
        min_confidence: Minimum confidence score to enter (1-5, recommended: 2).
        base_leverage: Base leverage for all entries.
        confidence_leverage_step: Additional leverage per confidence point above 1.
        max_leverage: Maximum leverage cap.
        rsi_period: RSI calculation period.
        vol_period: Volume SMA period for comparison.

    Returns:
        Signal array with leverage encoded: e.g., +1.5 = long at 1.5x leverage.
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute indicators
    btc_sma = _compute_sma_inline(btc_close, btc_regime_period)
    asset_sma = _compute_sma_inline(close, asset_trend_period)
    rsi = _compute_rsi_inline(close, rsi_period)
    vol_sma = _compute_sma_inline(volume, vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    lev = base_leverage
    start_bar = max(lookback, btc_mom_period, btc_regime_period, asset_trend_period,
                    rsi_period, vol_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                entered = False
                confidence = 1  # BTC regime alignment confirmed

                if is_risk_on:
                    # ═══ RISK-ON: Classic Pullback + Confidence ═══
                    if np.isnan(asset_sma[i]):
                        continue
                    is_uptrend = close[i] > asset_sma[i]

                    if ret < -threshold and is_uptrend:
                        # LONG: alt dipped in its own uptrend during BTC bull
                        if rsi[i] < 40:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.45:
                            confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02:
                            confidence += 1
                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > threshold and not is_uptrend:
                        # SHORT: alt surged but in downtrend → exhaustion
                        if rsi[i] > 60:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.55:
                            confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02:
                            confidence += 1
                        if confidence >= min_confidence:
                            position = -1
                            entered = True
                else:
                    # ═══ RISK-OFF: Sympathy Pullback + Confidence ═══
                    if ret < -threshold and btc_mom < 0:
                        # LONG: alt dips with BTC → sympathy → mean revert
                        if rsi[i] < 35:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.42:
                            confidence += 1
                        if btc_mom < -0.01:
                            confidence += 1
                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > threshold and btc_mom > 0:
                        # SHORT: alt pumps with BTC bounce → fades back
                        if rsi[i] > 65:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.58:
                            confidence += 1
                        if btc_mom > 0.01:
                            confidence += 1
                        if confidence >= min_confidence:
                            position = -1
                            entered = True

                if entered:
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                    lev = min(base_leverage + confidence_leverage_step * (confidence - 1),
                              max_leverage)
                    sig[i] = position * lev
                    continue

        else:
            # ═══ EXIT LOGIC (same as v5.0) ═══
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

            if position == 0:
                sig[i] = 0
                continue

        if position != 0:
            sig[i] = position * lev

    return sig


# ────────── v7.0 Multi-Timeframe Confirmation ──────────
@numba.jit(nopython=True, cache=True)
def apply_mtf_confirmation(
    signals_15m: np.ndarray,
    close_4h: np.ndarray,
    sma_period_4h: int,
    ratio: int = 16,
) -> np.ndarray:
    """Filter 15m signals by 4h trend direction (v7.0).

    Only allows entries that align with the 4h SMA trend:
      - Long signal: requires close_4h > SMA_4h (uptrend)
      - Short signal: requires close_4h < SMA_4h (downtrend)
    Counter-trend entries are zeroed out.

    A/B validated (365-day, 10 symbols):
      Without MTF: DD=16.5%, Calmar=8.65
      With MTF:    DD=2.2%,  Calmar=19.64

    Args:
        signals_15m: 15m signal array (from compute_signals_confidence).
        close_4h: 4h close prices array.
        sma_period_4h: SMA period on 4h bars (12 = 2 days optimal).
        ratio: Number of 15m bars per 4h bar (16 for USDT-M).

    Returns:
        Filtered signal array (same shape, counter-trend entries zeroed).
    """
    sma_4h = _compute_sma_inline(close_4h, sma_period_4h)
    n = len(signals_15m)
    out = signals_15m.copy()

    for i in range(n):
        if out[i] == 0.0:
            continue
        j = min(i // ratio, len(close_4h) - 1)
        if j < sma_period_4h or np.isnan(sma_4h[j]):
            continue

        trend_up = close_4h[j] > sma_4h[j]

        # Block counter-trend entries
        if out[i] > 0 and not trend_up:
            out[i] = 0.0
        elif out[i] < 0 and trend_up:
            out[i] = 0.0

    return out


@numba.jit(nopython=True, cache=True)
def apply_mtf_leverage(
    signals: np.ndarray,
    leverage_mult: float,
) -> np.ndarray:
    """Scale signal magnitude by leverage multiplier (v7.0).

    Used after MTF confirmation to amplify position size on high-confidence
    trades. The MTF filter removes ~75% of trades; remaining ones are
    trend-aligned and deserve increased leverage.

    With MTF confirmation + 2.0x leverage (effective 3.0x):
      Return: +103.7%, DD: 4.4%, Calmar: 23.75

    Args:
        signals: Signal array (magnitude encodes leverage).
        leverage_mult: Multiplier to apply (e.g., 2.0).

    Returns:
        Scaled signal array.
    """
    n = len(signals)
    out = np.zeros(n)
    for i in range(n):
        out[i] = signals[i] * leverage_mult
    return out


@numba.jit(nopython=True, cache=True)
def compute_signals_4h_independent(
    close: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    btc_sma_period: int,
) -> np.ndarray:
    """Independent 4h rubber band signals with BTC regime switch (v7.0).

    Trades 4h mean-reversion independently from 15m strategy.
    Different parameters optimized for 4h bar structure:
      - lookback=3 (12h), threshold=10%, max_hold=12 (2 days)
      - BTC SMA(48) for 8-day regime detection

    A/B validated (365-day, 10 symbols):
      Return: +52.2%, DD: 4.8%, Calmar: 10.87, 135 trades

    Args:
        close: 4h close prices.
        btc_close: 4h BTC close prices.
        lookback: Bars to measure move (3 = 12h).
        threshold: Min move to trigger (0.10 = 10%).
        target_retrace: TP at retrace fraction (0.5 = 50%).
        stop_mult: Stop at mult × move_size (0.8).
        max_hold: Max hold bars (12 = 2 days).
        btc_sma_period: BTC SMA for regime (48 = 8 days).

    Returns:
        Signal array (±1 for position direction).
    """
    n = len(close)
    sig = np.zeros(n)
    btc_sma = _compute_sma_inline(btc_close, btc_sma_period)
    btc_mom_period = min(lookback * 2, btc_sma_period)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_sma_period, btc_mom_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if position == 0:
            if abs(ret) > threshold:
                if np.isnan(btc_sma[i]):
                    continue

                is_risk_on = btc_close[i] > btc_sma[i]
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                entered = False

                if is_risk_on:
                    # Risk-On: buy dips, short rips
                    if ret < -threshold:
                        position = 1
                        entered = True
                    elif ret > threshold:
                        position = -1
                        entered = True
                else:
                    # Risk-Off: sympathy trades with BTC direction
                    if ret < -threshold and btc_mom < 0:
                        position = 1
                        entered = True
                    elif ret > threshold and btc_mom > 0:
                        position = -1
                        entered = True

                if entered:
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

            target = move_size * target_retrace
            effective_stop = move_size * stop_mult

            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                # Trailing: if gained >50% of target then dropped to <30% of peak
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
        # v6.0: volume & taker data for confidence scoring (per-symbol)
        self._volume_data: dict[str, np.ndarray] = {}
        self._taker_data: dict[str, np.ndarray] = {}
        # v7.0: higher timeframe close data for MTF confirmation (per-symbol)
        self._htf_close_data: dict[str, np.ndarray] = {}

    def update_market_data(
        self,
        symbol: str,
        volume: np.ndarray,
        taker_buy: np.ndarray,
    ) -> None:
        """Update volume and taker buy data for confidence scoring (v6.0).

        Call this before on_bar() with the latest volume/taker data arrays.

        Args:
            symbol: Trading symbol.
            volume: Volume array (same length as close_prices).
            taker_buy: Taker buy base volume array.
        """
        self._volume_data[symbol] = volume
        self._taker_data[symbol] = taker_buy

    def update_htf_data(
        self,
        symbol: str,
        close_4h: np.ndarray,
    ) -> None:
        """Update 4h close data for MTF confirmation (v7.0).

        Call this before on_bar() with the latest 4h candle closes.
        Used for: (1) MTF trend confirmation on 15m entries,
                  (2) Independent 4h rubber band signal generation.

        Args:
            symbol: Trading symbol.
            close_4h: 4h close prices array (at least mtf_sma_period bars).
        """
        self._htf_close_data[symbol] = close_4h

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

        # ═══ v6.0 Confidence-Scored Entry (highest priority) ═══
        if btc_close_prices is not None and cfg.entry_mode == 'confidence':
            min_bars = max(
                cfg.lookback, cfg.btc_directional_period,
                cfg.btc_regime_period, cfg.asset_trend_period,
                cfg.rsi_period, cfg.vol_period,
            ) + 1
            if len(btc_close_prices) < min_bars or len(close_prices) < min_bars:
                return 0
            # on_bar needs volume/taker_buy — stored in self._volume_data / self._taker_data
            vol_data = getattr(self, '_volume_data', {}).get(symbol)
            taker_data = getattr(self, '_taker_data', {}).get(symbol)
            if vol_data is None or taker_data is None:
                # Fallback to regime_switch if volume data not available
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
            sigs = compute_signals_confidence(
                close_prices, vol_data, taker_data, btc_close_prices,
                cfg.lookback, cfg.threshold,
                cfg.target_retrace, cfg.stop_mult,
                cfg.max_hold, cfg.stop_decay,
                cfg.btc_directional_period,
                cfg.btc_regime_period,
                cfg.asset_trend_period,
                cfg.min_confidence,
                cfg.base_leverage,
                cfg.confidence_leverage_step,
                cfg.max_leverage,
                cfg.rsi_period,
                cfg.vol_period,
            )

            # ═══ v7.0 MTF Confirmation ═══
            # Filter by 4h trend: block counter-trend entries
            htf_close = self._htf_close_data.get(symbol)
            if cfg.mtf_confirmation_enabled and htf_close is not None and len(htf_close) >= cfg.mtf_sma_period:
                sigs = apply_mtf_confirmation(sigs, htf_close, cfg.mtf_sma_period, 16)
                # Apply leverage boost to MTF-confirmed trades
                if cfg.mtf_leverage_mult > 1.0:
                    sigs = apply_mtf_leverage(sigs, cfg.mtf_leverage_mult)

            # Return signal with leverage magnitude
            last_sig = sigs[-1]
            if last_sig > 0:
                return 1
            elif last_sig < 0:
                return -1
            return 0

        # ═══ v5.0 Regime-Switch Dual Entry ═══
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

    def on_bar_4h(
        self,
        symbol: str,
        close_4h: np.ndarray,
        btc_close_4h: np.ndarray | None = None,
    ) -> int:
        """Generate independent 4h rubber band signal (v7.0).

        Separate from on_bar() — this runs on 4h candle data and generates
        independent entry/exit signals for the 4h timeframe strategy.

        Args:
            symbol: Trading symbol.
            close_4h: Array of 4h close prices.
            btc_close_4h: 4h BTC close prices for regime detection.

        Returns:
            Signal: -1 (short), 0 (flat), +1 (long).
        """
        cfg = self.config
        if not cfg.htf_independent_enabled:
            return 0
        if btc_close_4h is None:
            return 0
        min_bars = max(cfg.htf_rb_lookback, cfg.htf_rb_btc_sma_period) + 1
        if len(close_4h) < min_bars or len(btc_close_4h) < min_bars:
            return 0

        sigs = compute_signals_4h_independent(
            close_4h, btc_close_4h,
            cfg.htf_rb_lookback, cfg.htf_rb_threshold,
            cfg.htf_rb_target_retrace, cfg.htf_rb_stop_mult,
            cfg.htf_rb_max_hold, cfg.htf_rb_btc_sma_period,
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
