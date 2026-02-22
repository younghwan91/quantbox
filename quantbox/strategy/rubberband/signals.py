"""Numba-compiled signal engine and leverage/sizing functions.

All pure functions — no class state. Compiled with @numba.jit(nopython=True)
for hot-path performance (>10k iterations in backtest).
"""

from __future__ import annotations

import numba
import numpy as np


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


# ────────── ATR Computation (v11.4 Adaptive Threshold) ──────────
@numba.jit(nopython=True, cache=True)
def compute_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    period: int) -> np.ndarray:
    """ATR as percentage of close price (EMA-smoothed).

    Args:
        high: High prices array.
        low: Low prices array.
        close: Close prices array.
        period: ATR EMA smoothing period.

    Returns:
        Array of ATR/close ratios. NaN for bars before warm-up.
    """
    n = len(close)
    atr_pct = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    if n < period:
        return atr_pct
    atr = 0.0
    for i in range(period):
        atr += tr[i]
    atr /= period
    if close[period - 1] > 0:
        atr_pct[period - 1] = atr / close[period - 1]
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        atr = alpha * tr[i] + (1 - alpha) * atr
        if close[i] > 0:
            atr_pct[i] = atr / close[i]
    return atr_pct


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
def _compute_ema_inline(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA helper for Numba. Uses SMA warm-up then EMA smoothing."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += arr[i]
    out[period - 1] = s / period
    alpha = 2.0 / (period + 1.0)
    for i in range(period, n):
        out[i] = (arr[i] - out[i - 1]) * alpha + out[i - 1]
    return out


@numba.jit(nopython=True, cache=True)
def _compute_stochastic_inline(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    k_period: int = 14, d_period: int = 3,
) -> tuple:
    """Stochastic %K and %D arrays."""
    n = len(close)
    k = np.full(n, 50.0)
    d = np.full(n, 50.0)
    for i in range(k_period - 1, n):
        hh = high[i]
        ll = low[i]
        for j in range(i - k_period + 1, i):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        rng = hh - ll
        if rng > 0:
            k[i] = (close[i] - ll) / rng * 100.0
        else:
            k[i] = 50.0
    s = 0.0
    for i in range(n):
        s += k[i]
        if i >= d_period:
            s -= k[i - d_period]
        if i >= d_period - 1:
            d[i] = s / d_period
    return k, d


@numba.jit(nopython=True, cache=True)
def _compute_macd_inline(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9,
) -> tuple:
    """MACD line, signal line, histogram."""
    ema_fast = _compute_ema_inline(close, fast)
    ema_slow = _compute_ema_inline(close, slow)
    n = len(close)
    macd_line = np.full(n, 0.0)
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]
    signal_line = _compute_ema_inline(macd_line, signal_period)
    hist = np.full(n, 0.0)
    for i in range(n):
        if not np.isnan(signal_line[i]):
            hist[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, hist


@numba.jit(nopython=True, cache=True)
def compute_signals_confidence(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
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
    dynamic_target_enabled: bool = False,
    target_conf3: float = 0.45,
    target_conf4: float = 0.5,
    target_conf5: float = 0.55,
) -> np.ndarray:
    """v6.0→v8.0: Regime-switch MR with confidence scoring + graduated leverage.

    Same regime-switch entry logic as v5.0, enhanced with:
      1. Multi-evidence confidence scoring (RSI, volume, taker ratio, price/SMA)
      2. Minimum confidence filter — reject low-quality entries
      3. Graduated leverage — more evidence → bigger position
      4. Dynamic target (v8.0) — higher confidence → wider target (ride the wave)

    Evidence for LONG entries:
      +1 BTC regime confirmed (always, since regime check passes)
      +1 RSI oversold (< 40 Risk-On, < 35 Risk-Off)
      +1 Volume > 1.5x average (institutional / liquidation cascade)
      +1 Taker sell pressure (ratio < 0.45 Risk-On, < 0.42 Risk-Off)
      +1 Near SMA support (Risk-On) or strong BTC momentum (Risk-Off)

    For SHORT entries, evidence is mirrored (RSI overbought, taker buy, etc.).

    Leverage = min(base_leverage + confidence_leverage_step * (conf - 1), max_leverage)

    v8.0 Dynamic Target (A/B validated, winning strategy insight):
      When confident, ride the wave — scale target by confidence level:
        conf ≤ 2: target_retrace (default 0.4)
        conf = 3: target_conf3 (default 0.45)
        conf = 4: target_conf4 (default 0.50)
        conf ≥ 5: target_conf5 (default 0.55)
      DD=4.3% (unchanged), Calmar +9.3% (27.71 → 30.29)

    365-day validated across 500+ configurations.

    Args:
        close: Alt close prices.
        high: Alt high prices (for intra-bar TP detection).
        low: Alt low prices (for intra-bar SL detection).
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
        dynamic_target_enabled: Scale target by confidence (v8.0).
        target_conf3: Target retrace for confidence=3 (default 0.45).
        target_conf4: Target retrace for confidence=4 (default 0.50).
        target_conf5: Target retrace for confidence≥5 (default 0.55).

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
    trade_target_retrace = target_retrace  # per-trade target (may scale by confidence)
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
                confidence = 0  # v12.4.1: base=0 for backtest parity

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
                    # v8.0: Scale target by confidence (ride the wave when certain)
                    if dynamic_target_enabled:
                        if confidence >= 5:
                            trade_target_retrace = target_conf5
                        elif confidence >= 4:
                            trade_target_retrace = target_conf4
                        elif confidence >= 3:
                            trade_target_retrace = target_conf3
                        else:
                            trade_target_retrace = target_retrace
                    else:
                        trade_target_retrace = target_retrace
                    sig[i] = position * lev
                    continue

        else:
            # ═══ EXIT LOGIC ═══
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
                worst_gain = (low[i] - entry_price) / entry_price
                best_gain = (high[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
                worst_gain = (entry_price - high[i]) / entry_price
                best_gain = (entry_price - low[i]) / entry_price

            if best_gain > best_pnl:
                best_pnl = best_gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * trade_target_retrace

            # v12.5: Intra-bar SL/TP — SL priority on same-bar conflict
            if worst_gain < -effective_stop:
                position = 0
            elif best_gain >= target:
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


@numba.jit(nopython=True, cache=True)
def compute_signals_v9(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
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
    dynamic_target_enabled: bool = False,
    target_conf3: float = 0.45,
    target_conf4: float = 0.5,
    target_conf5: float = 0.55,
    stoch_enabled: bool = False,
    stoch_k_period: int = 5,
    stoch_d_period: int = 3,
    macd_enabled: bool = False,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 14,
    atr_period: int = 0,
    atr_multiplier: float = 2.5,
    threshold_floor: float = 0.01,
    threshold_cap: float = 0.06,
) -> np.ndarray:
    """v9.0: S+R+M Confluence Pipeline — regime-switch MR with Stochastic + MACD evidence.

    v11.4: Adaptive threshold — when atr_period > 0, per-bar threshold is
    clip(atr_multiplier × ATR%(atr_period), threshold_floor, threshold_cap).
    Falls back to fixed `threshold` when ATR is NaN or atr_period == 0.

    Extends compute_signals_confidence (v6.0→v8.0) with:
      - Stochastic %K/%D cross as confidence evidence (+3.7% Calmar)
      - MACD histogram reversal as confidence evidence (+7.6% Calmar)
      - Fibonacci exit discipline: target=0.382, stop=0.786
      - Combined: Calmar +33.6%, DD -8.5%

    Additional evidence for LONG:
      +1 Stochastic oversold cross (%K < 20, %K crosses above %D)
      +1 MACD histogram reversal (hist rising from below zero)

    Additional evidence for SHORT:
      +1 Stochastic overbought cross (%K > 80, %K crosses below %D)
      +1 MACD histogram reversal (hist falling from above zero)

    Args:
        close: Alt close prices.
        high: Alt high prices (for stochastic).
        low: Alt low prices (for stochastic).
        volume: Alt volume data.
        taker_buy: Taker buy base volume.
        btc_close: BTCUSDT close prices.
        lookback..target_conf5: Same as compute_signals_confidence.
        stoch_enabled: Enable stochastic %K/%D cross evidence.
        stoch_k_period: Stochastic %K period.
        stoch_d_period: Stochastic %D period.
        macd_enabled: Enable MACD histogram reversal evidence.
        macd_fast: MACD fast EMA period.
        macd_slow: MACD slow EMA period.
        macd_signal_period: MACD signal EMA period.

    Returns:
        Signal array with leverage encoded.

    WARNING (v12.4.1): This function retains DUAL Risk-On/Risk-Off branching
    for backward compatibility with research scripts (alpha_search_r2/r3,
    multi_alpha_sweep, etc.). Live trading (on_bar) uses Risk-Off ONLY —
    the Risk-On path was NEVER validated in the 6yr backtest (R:R 1.91).
    Production backtests should use reversal_ab_test.py scan_entries() which
    is Risk-Off only, matching on_bar() exactly.
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

    # v9.0: Pre-compute stochastic and MACD
    if stoch_enabled:
        stoch_k, stoch_d = _compute_stochastic_inline(
            high, low, close, stoch_k_period, stoch_d_period)
    else:
        stoch_k = np.full(n, 50.0)
        stoch_d = np.full(n, 50.0)

    if macd_enabled:
        _macd_line, _macd_signal, macd_hist = _compute_macd_inline(
            close, macd_fast, macd_slow, macd_signal_period)
    else:
        macd_hist = np.full(n, 0.0)

    # v11.4: Pre-compute ATR% for adaptive threshold
    if atr_period > 0:
        atr_pct = compute_atr_pct(high, low, close, atr_period)
    else:
        atr_pct = np.full(n, np.nan)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    lev = base_leverage
    trade_target_retrace = target_retrace
    start_bar = max(lookback, btc_mom_period, btc_regime_period, asset_trend_period,
                    rsi_period, vol_period)
    if atr_period > 0 and atr_period > start_bar:
        start_bar = atr_period

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        # v11.4: Per-bar adaptive threshold
        if atr_period > 0:
            atr_val = atr_pct[i]
            if np.isnan(atr_val):
                thr = threshold
            else:
                thr = atr_multiplier * atr_val
                if thr < threshold_floor:
                    thr = threshold_floor
                if thr > threshold_cap:
                    thr = threshold_cap
        else:
            thr = threshold

        if position == 0:
            if abs(ret) > thr:
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                entered = False
                confidence = 0  # v12.4.1: base=0 for backtest parity

                if is_risk_on:
                    # ═══ RISK-ON: Classic Pullback + Confidence ═══
                    if np.isnan(asset_sma[i]):
                        continue
                    is_uptrend = close[i] > asset_sma[i]

                    if ret < -thr and is_uptrend:
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
                        # v9.0: Stochastic oversold cross
                        if stoch_enabled and i > 0:
                            if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i - 1] <= stoch_d[i - 1]:
                                confidence += 1
                        # v9.0: MACD histogram reversal (rising from negative)
                        if macd_enabled and i > 0:
                            if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                                confidence += 1
                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > thr and not is_uptrend:
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
                        # v9.0: Stochastic overbought cross
                        if stoch_enabled and i > 0:
                            if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i - 1] >= stoch_d[i - 1]:
                                confidence += 1
                        # v9.0: MACD histogram reversal (falling from positive)
                        if macd_enabled and i > 0:
                            if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                                confidence += 1
                        if confidence >= min_confidence:
                            position = -1
                            entered = True
                else:
                    # ═══ RISK-OFF: Sympathy Pullback + Confidence ═══
                    if ret < -thr and btc_mom < 0:
                        # LONG: alt dips with BTC → sympathy → mean revert
                        if rsi[i] < 35:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.42:
                            confidence += 1
                        if btc_mom < -0.01:
                            confidence += 1
                        # v9.0: Stochastic oversold cross
                        if stoch_enabled and i > 0:
                            if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i - 1] <= stoch_d[i - 1]:
                                confidence += 1
                        # v9.0: MACD histogram reversal (rising from negative)
                        if macd_enabled and i > 0:
                            if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                                confidence += 1
                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > thr and btc_mom > 0:
                        # SHORT: alt pumps with BTC bounce → fades back
                        if rsi[i] > 65:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.58:
                            confidence += 1
                        if btc_mom > 0.01:
                            confidence += 1
                        # v9.0: Stochastic overbought cross
                        if stoch_enabled and i > 0:
                            if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i - 1] >= stoch_d[i - 1]:
                                confidence += 1
                        # v9.0: MACD histogram reversal (falling from positive)
                        if macd_enabled and i > 0:
                            if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
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
                    # v8.0: Scale target by confidence (ride the wave when certain)
                    if dynamic_target_enabled:
                        if confidence >= 5:
                            trade_target_retrace = target_conf5
                        elif confidence >= 4:
                            trade_target_retrace = target_conf4
                        elif confidence >= 3:
                            trade_target_retrace = target_conf3
                        else:
                            trade_target_retrace = target_retrace
                    else:
                        trade_target_retrace = target_retrace
                    sig[i] = position * lev
                    continue

        else:
            # ═══ EXIT LOGIC ═══
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
                worst_gain = (low[i] - entry_price) / entry_price
                best_gain = (high[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
                worst_gain = (entry_price - high[i]) / entry_price
                best_gain = (entry_price - low[i]) / entry_price

            if best_gain > best_pnl:
                best_pnl = best_gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * trade_target_retrace

            # v12.5: Intra-bar SL/TP — SL priority on same-bar conflict
            if worst_gain < -effective_stop:
                position = 0
            elif best_gain >= target:
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
    high: np.ndarray,
    low: np.ndarray,
    lookback: int,
    threshold: float,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float = 0.0,
    runner_max_hold_mult: float = 3.0,
) -> np.ndarray:
    """Compute signals with partial exit + break-even stop.

    v12.5: Uses high/low for intra-bar SL/TP detection to match live
    server-side behavior. SL checks worst-case (low for LONG, high for SHORT).
    TP checks best-case (high for LONG, low for SHORT).
    Same-bar conflict: SL takes priority (conservative).

    Capital preservation framework:
      Phase 1 (full position): TP1 hit → close 50%, transition to Phase 2.
      Phase 2 (half position): break-even stop replaces original stop.
        Remaining half runs with trailing or gets stopped at break-even.

    Output: ±1.0 (full position), ±0.5 (half position after partial close), 0.0 (flat).
    The backtest core treats signal magnitude as position size.

    Args:
        close: Close price array.
        high: High price array (for intra-bar TP detection).
        low: Low price array (for intra-bar SL detection).
        lookback: Bars to measure the move.
        threshold: Minimum move to trigger entry.
        target_retrace: Take profit fraction of move.
        stop_mult: Stop loss multiple.
        max_hold: Maximum holding period.
        stop_decay: Stop tightening rate.
        runner_max_hold_mult: Runner max hold = this × max_hold.

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
                worst_gain = (low[i] - entry_price) / entry_price
                best_gain = (high[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
                worst_gain = (entry_price - high[i]) / entry_price
                best_gain = (entry_price - low[i]) / entry_price

            # Update best_pnl with intra-bar best (not just close)
            if best_gain > best_pnl:
                best_pnl = best_gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * target_retrace

            # v12.5: SL/TP use intra-bar extremes for parity with live.
            # Same-bar conflict: SL takes priority (conservative).
            if worst_gain < -effective_stop:
                position = 0
                phase = 0
            elif best_gain >= target:
                phase = 2
                best_pnl = best_gain
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= max_hold:
                position = 0
                phase = 0
            elif gain > 0:
                # ANY profit at close → partial close 50%, enter Phase 2
                phase = 2
                best_pnl = gain

        elif phase == 2:
            # ── Half position: break-even stop, target still active ──
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
                worst_gain = (low[i] - entry_price) / entry_price
                best_gain = (high[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
                worst_gain = (entry_price - high[i]) / entry_price
                best_gain = (entry_price - low[i]) / entry_price

            if best_gain > best_pnl:
                best_pnl = best_gain

            target = move_size * target_retrace

            # v12.5: Runner exits using intra-bar worst for BE stop.
            # 1. Breakeven stop hit (worst_gain <= 0, intra-bar)
            # 2. Pre-TP1 trail: gave back too much unrealized profit
            # 3. Max hold exceeded (runner_max_hold_mult × max_hold)
            if worst_gain <= 0:
                # Break-even stop: $0 risk on remaining half
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= int(max_hold * runner_max_hold_mult):
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
    runner_max_hold_mult: float = 3.0,
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

            # v12.3.3: TP1 hit → partial close, enter Phase 2 runner
            if gain >= target:
                phase = 2
                best_pnl = gain
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

            if gain <= 0:
                position = 0
                phase = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
                phase = 0
            elif hold_count >= int(max_hold * runner_max_hold_mult):
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

