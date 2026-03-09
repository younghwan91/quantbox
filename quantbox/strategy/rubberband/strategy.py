"""Stateful strategy wrapper for live trading.

Maintains per-symbol state (position, entry price, hold count)
and generates signals from new price data.
"""

from __future__ import annotations

import logging

import numpy as np

from quantbox.strategy.rubberband.config import RubberBandConfig
from quantbox.strategy.rubberband.signals import (
    _compute_macd_inline,
    _compute_rsi_inline,
    _compute_sma_inline,
    _compute_stochastic_inline,
    compute_atr_pct,
    compute_signals,
    compute_signals_4h_independent,
    compute_signals_mtf,
    compute_signals_partial_exit_btc_directional,
    compute_signals_regime_switch,
    compute_signals_with_btc_directional,
    compute_signals_with_regime,
)

logger = logging.getLogger('Scalper')



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
        # v9.0: high/low data for stochastic (per-symbol)
        self._high_data: dict[str, np.ndarray] = {}
        self._low_data: dict[str, np.ndarray] = {}
        # v7.0: higher timeframe close data for MTF confirmation (per-symbol)
        self._htf_close_data: dict[str, np.ndarray] = {}
        # v10.2 PARITY: Track whether last signal per symbol was MTF-confirmed
        self._mtf_confirmed: dict[str, bool] = {}
        # v12.3: Store last confidence & ATR for sizing in scalper
        self._last_confidence: dict[str, int] = {}
        self._last_atr_pct: dict[str, float] = {}

    def update_market_data(
        self,
        symbol: str,
        volume: np.ndarray,
        taker_buy: np.ndarray,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
    ) -> None:
        """Update volume, taker buy, and OHLC data for confidence scoring.

        Call this before on_bar() with the latest data arrays.
        v6.0: volume + taker_buy for confidence scoring.
        v9.0: high + low for stochastic %K/%D evidence.

        Args:
            symbol: Trading symbol.
            volume: Volume array (same length as close_prices).
            taker_buy: Taker buy base volume array.
            high: High price array (for stochastic, v9.0).
            low: Low price array (for stochastic, v9.0).
        """
        self._volume_data[symbol] = volume
        self._taker_data[symbol] = taker_buy
        if high is not None:
            self._high_data[symbol] = high
        if low is not None:
            self._low_data[symbol] = low

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
        """Process a new bar and return entry signal for the LATEST bar only.

        CRITICAL: This method evaluates entry conditions at the LAST bar
        statelessly, identical to how the backtest evaluates each bar.
        Previously this ran compute_signals_v9 on 200 bars and took sigs[-1],
        which conflated old virtual positions with fresh entry signals.

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

        # ═══ v6.0+ Confidence-Scored Entry (stateless last-bar evaluation) ═══
        if btc_close_prices is not None and cfg.entry_mode == 'confidence':
            min_bars = max(
                cfg.lookback, cfg.btc_directional_period,
                cfg.btc_regime_period, cfg.asset_trend_period,
                cfg.rsi_period, cfg.vol_period,
            ) + 1
            if len(btc_close_prices) < min_bars or len(close_prices) < min_bars:
                return 0
            # on_bar needs volume/taker_buy — stored in self._volume_data / self._taker_data
            vol_data = self._volume_data.get(symbol)
            taker_data = self._taker_data.get(symbol)
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

            # ──────────────────────────────────────────────────────────
            # STATELESS last-bar entry evaluation (v13.0 parity fix)
            #
            # Previously: ran compute_signals_v9() on 200 bars and took
            # sigs[-1]. This caused TWO critical bugs:
            #   1. Ghost-position: sigs[-1] was non-zero from a STALE
            #      virtual entry 5-50 bars ago, not a fresh signal.
            #   2. Missed-signal: sigs[-1] was zero because the function
            #      was mid-position when bar 199 qualified for entry.
            #
            # Fix: evaluate the LAST bar directly using the same
            # confidence scoring logic as compute_signals_v9 (identical
            # entry conditions, no position state carried forward).
            # This is equivalent to what the backtest sees at each bar
            # when the strategy is flat (position == 0), which is the
            # only state that matters for _scan_entries().
            # ──────────────────────────────────────────────────────────
            n = len(close_prices)

            # v11.4: Compute adaptive threshold from ATR
            if cfg.atr_period > 0:
                high_data = self._high_data.get(symbol, close_prices)
                low_data = self._low_data.get(symbol, close_prices)
                atr_arr = compute_atr_pct(high_data, low_data, close_prices, cfg.atr_period)
                atr_val = float(atr_arr[-1])
                if np.isnan(atr_val):
                    thr = cfg.threshold  # fallback
                else:
                    thr = cfg.atr_multiplier * atr_val
                    if thr < cfg.threshold_floor:
                        thr = cfg.threshold_floor
                    if thr > cfg.threshold_cap:
                        thr = cfg.threshold_cap
            else:
                thr = cfg.threshold

            # 8-bar return at the last bar
            ret = close_prices[-1] / close_prices[-cfg.lookback - 1] - 1.0
            if abs(ret) <= thr:
                self._mtf_confirmed[symbol] = False
                return 0

            # BTC regime
            # BTC regime NaN guard (btc_sma needed for sanity check only;
            # v12.4.1 removed Risk-On/Risk-Off branching in favour of
            # universal Risk-Off logic that uses btc_mom instead)
            btc_sma_val = float(np.mean(btc_close_prices[-cfg.btc_regime_period:]))
            if np.isnan(btc_sma_val):
                self._mtf_confirmed[symbol] = False
                return 0

            # BTC momentum
            btc_mom = float(btc_close_prices[-1] / btc_close_prices[-cfg.btc_directional_period - 1] - 1.0)

            # RSI
            rsi_arr = _compute_rsi_inline(close_prices, cfg.rsi_period)
            rsi_val = float(rsi_arr[-1])

            # Volume
            vol_ratio = 0.0
            taker_ratio_val = 0.5
            vol_sma_arr = _compute_sma_inline(vol_data, cfg.vol_period)
            vol_sma_val = float(vol_sma_arr[-1])
            if not np.isnan(vol_sma_val) and vol_sma_val > 0:
                vol_ratio = float(vol_data[-1]) / vol_sma_val
            if float(vol_data[-1]) > 0:
                taker_ratio_val = float(taker_data[-1]) / float(vol_data[-1])

            # Stochastic
            stoch_k_val, stoch_d_val = 50.0, 50.0
            sk_prev, sd_prev = 50.0, 50.0
            if cfg.stoch_enabled:
                high_data = self._high_data.get(symbol, close_prices)
                low_data = self._low_data.get(symbol, close_prices)
                sk, sd = _compute_stochastic_inline(
                    high_data, low_data, close_prices,
                    cfg.stoch_k_period, cfg.stoch_d_period,
                )
                stoch_k_val, stoch_d_val = float(sk[-1]), float(sd[-1])
                if n > 1:
                    sk_prev, sd_prev = float(sk[-2]), float(sd[-2])

            # MACD
            macd_hist_val, macd_hist_prev = 0.0, 0.0
            if cfg.macd_enabled and n >= cfg.macd_slow + cfg.macd_signal_period:
                _, _, mhist = _compute_macd_inline(
                    close_prices, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
                )
                macd_hist_val = float(mhist[-1])
                if n > 1:
                    macd_hist_prev = float(mhist[-2])

            # ── Confidence scoring (v12.4.1: base=0 for backtest parity) ──
            # Previously confidence=1 ("BTC regime base point"), but
            # backtest scan_entries() starts at 0.  With min_confidence=3,
            # this +1 inflation admitted 2-evidence trades in live (blocked
            # in backtest) and over-sized all trades via SteepX.
            confidence = 0
            signal = 0

            # v12.4.1: ALWAYS use Risk-Off logic regardless of BTC regime.
            # The Risk-On path (uptrend pullback with looser thresholds)
            # was NEVER validated in the 6-year backtest which produced
            # R:R 1.91.  Using untested signal paths in production is
            # unacceptable.  Risk-Off sympathy logic was validated across
            # all market regimes (bull + bear) → use it universally.
            if ret < -thr and btc_mom < 0:
                # LONG: alt dips with BTC → sympathy → mean revert
                if rsi_val < 35: confidence += 1
                if vol_ratio > 1.5: confidence += 1
                if taker_ratio_val < 0.42: confidence += 1
                if btc_mom < -0.01: confidence += 1
                if cfg.stoch_enabled and n > 1:
                    if stoch_k_val < 20 and stoch_k_val > stoch_d_val and sk_prev <= sd_prev:
                        confidence += 1
                if cfg.macd_enabled:
                    if macd_hist_val > macd_hist_prev and macd_hist_prev < 0:
                        confidence += 1
                # v12.6: Climax volume filter (R21 [E] champion: vol > 3×vol_SMA)
                if cfg.climax_enabled and vol_sma_val > 0:
                    if vol_ratio >= cfg.climax_vol_mult:
                        confidence += 1
                if confidence >= cfg.min_confidence:
                    signal = 1

            elif ret > thr and btc_mom > 0:
                # SHORT: alt pumps with BTC bounce → fades back
                if rsi_val > 65: confidence += 1
                if vol_ratio > 1.5: confidence += 1
                if taker_ratio_val > 0.58: confidence += 1
                if btc_mom > 0.01: confidence += 1
                if cfg.stoch_enabled and n > 1:
                    if stoch_k_val > 80 and stoch_k_val < stoch_d_val and sk_prev >= sd_prev:
                        confidence += 1
                if cfg.macd_enabled:
                    if macd_hist_val < macd_hist_prev and macd_hist_prev > 0:
                        confidence += 1
                # v12.6: Climax volume filter (R21 [E] champion: vol > 3×vol_SMA)
                if cfg.climax_enabled and vol_sma_val > 0:
                    if vol_ratio >= cfg.climax_vol_mult:
                        confidence += 1
                if confidence >= cfg.min_confidence:
                    signal = -1

            # ═══ v7.0 MTF Confirmation (LIVE — inline latest-bar check) ═══
            htf_close = self._htf_close_data.get(symbol)
            mtf_applied = False

            if signal != 0 and cfg.mtf_confirmation_enabled and htf_close is not None and len(htf_close) >= cfg.mtf_sma_period:
                sma_4h = float(np.mean(htf_close[-cfg.mtf_sma_period:]))
                trend_up = float(htf_close[-1]) > sma_4h

                if signal > 0 and not trend_up:
                    signal = 0  # Block long in 4h downtrend
                elif signal < 0 and trend_up:
                    signal = 0  # Block short in 4h uptrend
                else:
                    mtf_applied = True

            self._mtf_confirmed[symbol] = mtf_applied

            # v12.3: Store confidence & ATR for sizing in scalper
            self._last_confidence[symbol] = confidence if signal != 0 else 0
            if cfg.atr_period > 0:
                self._last_atr_pct[symbol] = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.02
            else:
                self._last_atr_pct[symbol] = 0.02

            return signal

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
                    cfg.runner_max_hold_mult,
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

    def diagnose_bar(
        self,
        symbol: str,
        close_prices: np.ndarray,
        btc_close_prices: np.ndarray | None = None,
    ) -> dict:
        """Compute all intermediate diagnostic values for the latest bar.

        READ-ONLY: Does NOT modify any state. Call AFTER on_bar() to
        capture the same-bar context without side effects.

        Returns a dict with all pipeline stages for structured logging.
        Designed for parameter optimization — every filter gate is exposed.
        """
        cfg = self.config

        # v11.4: Compute adaptive threshold for diagnostics
        if cfg.atr_period > 0 and len(close_prices) >= cfg.atr_period:
            high_data = self._high_data.get(symbol, close_prices)
            low_data = self._low_data.get(symbol, close_prices)
            atr_arr = compute_atr_pct(high_data, low_data, close_prices, cfg.atr_period)
            _atr_val = float(atr_arr[-1])
            if np.isnan(_atr_val):
                _diag_thr = cfg.threshold
            else:
                _diag_thr = cfg.atr_multiplier * _atr_val
                if _diag_thr < cfg.threshold_floor:
                    _diag_thr = cfg.threshold_floor
                if _diag_thr > cfg.threshold_cap:
                    _diag_thr = cfg.threshold_cap
        else:
            _diag_thr = cfg.threshold

        diag: dict = {
            'symbol': symbol,
            'price': float(close_prices[-1]) if len(close_prices) > 0 else 0.0,
            'move_8bar': 0.0,
            'threshold': _diag_thr,
            'passed_threshold': False,
            'regime': None,          # 'risk_on' | 'risk_off' | None
            'btc_mom': 0.0,
            'asset_trend': None,     # 'up' | 'down' | None
            'entry_type': None,      # 'classic_long'|'classic_short'|'sympathy_long'|'sympathy_short'
            'confidence': 0,
            'min_confidence': cfg.min_confidence,
            'passed_confidence': False,
            'evidence': {},          # per-indicator breakdown
            'raw_signal': 0,         # signal from compute_signals before MTF
            'mtf_trend_up': None,    # True/False/None
            'mtf_blocked': False,
            'final_signal': 0,
        }

        if len(close_prices) < cfg.lookback + 1:
            return diag

        # 8-bar move
        ret = float(close_prices[-1] / close_prices[-cfg.lookback - 1] - 1.0)
        diag['move_8bar'] = ret

        if btc_close_prices is None or cfg.entry_mode != 'confidence':
            return diag

        min_bars = max(
            cfg.lookback, cfg.btc_directional_period,
            cfg.btc_regime_period, cfg.asset_trend_period,
            cfg.rsi_period, cfg.vol_period,
        ) + 1
        if len(btc_close_prices) < min_bars or len(close_prices) < min_bars:
            return diag

        n = len(close_prices)

        # ── Indicators (same as compute_signals_v9, last bar only) ──
        # BTC regime
        btc_sma_val = float(np.mean(btc_close_prices[-cfg.btc_regime_period:]))
        is_risk_on = float(btc_close_prices[-1]) > btc_sma_val
        diag['regime'] = 'risk_on' if is_risk_on else 'risk_off'

        # BTC momentum
        btc_mom = float(btc_close_prices[-1] / btc_close_prices[-cfg.btc_directional_period - 1] - 1.0)
        diag['btc_mom'] = btc_mom

        # Asset trend
        asset_sma_val = float(np.mean(close_prices[-cfg.asset_trend_period:]))
        is_uptrend = float(close_prices[-1]) > asset_sma_val
        diag['asset_trend'] = 'up' if is_uptrend else 'down'

        # RSI (approximate using last portion — same Wilder smoothing)
        rsi_arr = _compute_rsi_inline(close_prices, cfg.rsi_period)
        rsi_val = float(rsi_arr[-1])

        # Volume
        vol_data = self._volume_data.get(symbol)
        taker_data = self._taker_data.get(symbol)
        vol_ratio = 0.0
        taker_ratio_val = 0.5
        if vol_data is not None and len(vol_data) == n:
            vol_sma_arr = _compute_sma_inline(vol_data, cfg.vol_period)
            vol_sma_val = float(vol_sma_arr[-1])
            if not np.isnan(vol_sma_val) and vol_sma_val > 0:
                vol_ratio = float(vol_data[-1]) / vol_sma_val
            if taker_data is not None and len(taker_data) == n and float(vol_data[-1]) > 0:
                taker_ratio_val = float(taker_data[-1]) / float(vol_data[-1])

        # Stochastic
        stoch_k_val, stoch_d_val = 50.0, 50.0
        if cfg.stoch_enabled:
            high_data = self._high_data.get(symbol, close_prices)
            low_data = self._low_data.get(symbol, close_prices)
            sk, sd = _compute_stochastic_inline(high_data, low_data, close_prices,
                                                 cfg.stoch_k_period, cfg.stoch_d_period)
            stoch_k_val, stoch_d_val = float(sk[-1]), float(sd[-1])

        # MACD
        macd_hist_val, macd_hist_prev = 0.0, 0.0
        if cfg.macd_enabled and n >= cfg.macd_slow + cfg.macd_signal_period:
            _, _, mhist = _compute_macd_inline(close_prices, cfg.macd_fast,
                                                cfg.macd_slow, cfg.macd_signal_period)
            macd_hist_val = float(mhist[-1])
            macd_hist_prev = float(mhist[-2]) if n > 1 else 0.0

        # SMA distance
        sma_dist = abs(float(close_prices[-1]) - asset_sma_val) / asset_sma_val if asset_sma_val > 0 else 0.0

        diag['evidence'] = {
            'rsi': round(rsi_val, 2),
            'vol_ratio': round(vol_ratio, 3),
            'taker_ratio': round(taker_ratio_val, 4),
            'sma_dist': round(sma_dist, 4),
            'stoch_k': round(stoch_k_val, 2),
            'stoch_d': round(stoch_d_val, 2),
            'macd_hist': round(macd_hist_val, 6),
            'macd_hist_prev': round(macd_hist_prev, 6),
            # v12.6
            'climax_vol_ratio': round(vol_ratio, 3),
            'climax_threshold': cfg.climax_vol_mult if cfg.climax_enabled else None,
            'climax_fired': (cfg.climax_enabled and vol_ratio >= cfg.climax_vol_mult and vol_ratio > 0),
            'atr_stop_enabled': cfg.atr_stop_enabled,
            'atr_pct': round(self._last_atr_pct.get(symbol, float('nan')), 5),
        }

        # ── Threshold gate ──
        diag['passed_threshold'] = abs(ret) > _diag_thr
        if not diag['passed_threshold']:
            return diag

        # ── Confidence scoring (v12.4.1: base=0 for backtest parity) ──
        confidence = 0
        entry_type = None

        # v12.4.1: Always use Risk-Off (sympathy) logic — validated in backtest
        if ret < -_diag_thr and btc_mom < 0:
            entry_type = 'sympathy_long'
            if rsi_val < 35: confidence += 1
            if vol_ratio > 1.5: confidence += 1
            if taker_ratio_val < 0.42: confidence += 1
            if btc_mom < -0.01: confidence += 1
            if cfg.stoch_enabled and n > 1:
                sk_prev = float(sk[-2]) if cfg.stoch_enabled else 50.0
                sd_prev = float(sd[-2]) if cfg.stoch_enabled else 50.0
                if stoch_k_val < 20 and stoch_k_val > stoch_d_val and sk_prev <= sd_prev:
                    confidence += 1
            if cfg.macd_enabled:
                if macd_hist_val > macd_hist_prev and macd_hist_prev < 0:
                    confidence += 1
            # v12.6: Climax
            if cfg.climax_enabled and vol_ratio >= cfg.climax_vol_mult and not (vol_ratio == 0):
                confidence += 1
        elif ret > _diag_thr and btc_mom > 0:
            entry_type = 'sympathy_short'
            if rsi_val > 65: confidence += 1
            if vol_ratio > 1.5: confidence += 1
            if taker_ratio_val > 0.58: confidence += 1
            if btc_mom > 0.01: confidence += 1
            if cfg.stoch_enabled and n > 1:
                sk_prev = float(sk[-2]) if cfg.stoch_enabled else 50.0
                sd_prev = float(sd[-2]) if cfg.stoch_enabled else 50.0
                if stoch_k_val > 80 and stoch_k_val < stoch_d_val and sk_prev >= sd_prev:
                    confidence += 1
            if cfg.macd_enabled:
                if macd_hist_val < macd_hist_prev and macd_hist_prev > 0:
                    confidence += 1
            # v12.6: Climax
            if cfg.climax_enabled and vol_ratio >= cfg.climax_vol_mult and not (vol_ratio == 0):
                confidence += 1
        else:
            entry_type = 'blocked'

        diag['entry_type'] = entry_type
        diag['confidence'] = confidence
        diag['passed_confidence'] = confidence >= cfg.min_confidence

        # Determine raw signal (before MTF)
        raw_signal = 0
        if entry_type == 'sympathy_long' and confidence >= cfg.min_confidence:
            raw_signal = 1
        elif entry_type == 'sympathy_short' and confidence >= cfg.min_confidence:
            raw_signal = -1
        diag['raw_signal'] = raw_signal

        # ── MTF filter ──
        htf_close = self._htf_close_data.get(symbol)
        if cfg.mtf_confirmation_enabled and htf_close is not None and len(htf_close) >= cfg.mtf_sma_period:
            sma_4h = float(np.mean(htf_close[-cfg.mtf_sma_period:]))
            trend_up = float(htf_close[-1]) > sma_4h
            diag['mtf_trend_up'] = trend_up

            if raw_signal > 0 and not trend_up:
                diag['mtf_blocked'] = True
                diag['final_signal'] = 0
            elif raw_signal < 0 and trend_up:
                diag['mtf_blocked'] = True
                diag['final_signal'] = 0
            else:
                diag['final_signal'] = raw_signal
        else:
            diag['final_signal'] = raw_signal

        return diag

    def was_mtf_confirmed(self, symbol: str) -> bool:
        """Check if the last signal for a symbol was MTF-confirmed.

        Used by scalper._open_position() to apply mtf_leverage_mult,
        ensuring exact parity with the backtest signal pipeline:
        compute_signals_v9 → apply_mtf_confirmation → apply_mtf_leverage.

        Returns:
            True if the signal survived MTF confirmation filter.
        """
        return self._mtf_confirmed.get(symbol, False)

    def get_last_confidence(self, symbol: str) -> int:
        """Return last computed confidence for a symbol (v12.3 sizing)."""
        return self._last_confidence.get(symbol, 0)

    def get_last_atr_pct(self, symbol: str) -> float:
        """Return last computed ATR% for a symbol (v12.3 vol-sizing / PBT trail)."""
        return self._last_atr_pct.get(symbol, 0.02)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_effective_stop_mult(
        self, close_prices: np.ndarray,
        symbol: str | None = None,
        move_size: float | None = None,
    ) -> float:
        """Compute the effective stop multiplier for the current bar.

        v12.6: When atr_stop_enabled=True, returns an ATR-based dynamic stop
        expressed as a multiplier on move_size (= atr_stop_mult * ATR% / move_size).
        This mirrors the R21 backtest logic (champion [E]: Calmar 728→1171).

        Falls back to HTF-agree fixed stop when ATR stop is disabled or
        move_size/symbol are not provided.

        Args:
            close_prices: Recent close prices for HTF-agree computation.
            symbol: Symbol name for ATR lookup (v12.6 ATR stop path).
            move_size: Magnitude of the rubber-band move (v12.6 ATR stop path).

        Returns:
            Effective stop multiplier. Multiplied by move_size gives SL distance.
        """
        cfg = self.config

        # ── v12.6 ATR-based dynamic stop ──────────────────────────────────
        if cfg.atr_stop_enabled and symbol is not None and move_size is not None and move_size > 0:
            atr_pct = self._last_atr_pct.get(symbol, 0.02)
            # Clamp the absolute SL distance
            sl_dist = cfg.atr_stop_mult * atr_pct
            sl_dist = max(cfg.atr_stop_floor, min(cfg.atr_stop_cap, sl_dist))
            # Return as multiplier on move_size (parity with existing SL math:
            # sl_distance = move_size * eff_stop → eff_stop = sl_dist / move_size)
            return sl_dist / move_size

        # ── Legacy fixed stop (HTF-agree) ─────────────────────────────────
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
        server_sl_order_id: str | None = None,
        server_tp_order_id: str | None = None,
        **kwargs,
    ):
        """Record a new position.

        Args:
            entry_stop_mult: Effective stop multiplier at entry (for
                HTF agree parity). If None, uses cfg.stop_mult.
            server_sl_order_id: Exchange-side stop-loss order ID (v11.1).
                Used for cancellation when position is closed normally.
            server_tp_order_id: Exchange-side take-profit order ID (v11.2).
                Used for cancellation when position is closed normally.
        """
        if entry_price <= 0:
            raise ValueError(
                f"Invalid entry_price for {symbol}: {entry_price}"
            )
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
            'server_sl_order_id': server_sl_order_id,  # v11.1: server-side stop
            'server_tp_order_id': server_tp_order_id,  # v11.2: server-side TP
            # v12.3 PBT fields
            'confidence': kwargs.get('confidence', 0),  # for SteepX logging
            'highest_since_tp1': 0.0,   # track runner peak after TP1
            'trail_floor': None,        # breakeven floor after TP1 (None = not set)
        }

    def close_position(self, symbol: str) -> dict | None:
        """Remove position and return its info."""
        return self._positions.pop(symbol, None)

    def update_position(self, symbol: str, current_price: float,
                        current_atr_pct: float | None = None,
                        current_high: float | None = None,
                        current_low: float | None = None) -> str | None:
        """Check if position should be closed or partially closed.

        v12.5: Accepts current_high/current_low for intra-bar SL/TP detection.
        This serves as backup client-side check — primary TP1 is server-side (50% qty).
        SL is server-side (handled intra-bar), so client SL check is redundant
        but kept for safety.

        v12.3 PBT (Partial-Breakeven-Trail) exit mode:
          Phase 1 (full position): TP1 hit → partial close 50%, then Phase 2.
          Phase 2 (runner): breakeven floor + ATR trailing stop.

        Args:
            current_atr_pct: Current ATR as percentage (for runner trailing).
            current_high: Bar high price (for intra-bar TP1 detection).
            current_low: Bar low price (for intra-bar SL detection).

        Returns exit reason string or None.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        pos['bars_held'] += 1
        cfg = self.config

        # Close-based gain (used for trailing decisions)
        if pos['signal'] == 1:
            gain = (current_price - pos['entry_price']) / pos['entry_price']
        else:
            gain = (pos['entry_price'] - current_price) / pos['entry_price']

        # v12.5: Intra-bar extreme gains for SL/TP detection
        if current_high is not None and current_low is not None:
            if pos['signal'] == 1:
                worst_gain = (current_low - pos['entry_price']) / pos['entry_price']
                best_gain = (current_high - pos['entry_price']) / pos['entry_price']
            else:
                worst_gain = (pos['entry_price'] - current_high) / pos['entry_price']
                best_gain = (pos['entry_price'] - current_low) / pos['entry_price']
        else:
            worst_gain = gain
            best_gain = gain

        if best_gain > pos['best_pnl']:
            pos['best_pnl'] = best_gain

        target = pos['move_size'] * cfg.target_retrace
        entry_sm = pos.get('entry_stop_mult', cfg.stop_mult)
        time_factor = 1.0 - cfg.stop_decay * (pos['bars_held'] / cfg.max_hold)
        if time_factor < 0.1:
            time_factor = 0.1
        effective_stop = pos['move_size'] * entry_sm * time_factor

        if not pos.get('partial_closed', False):
            # ── Phase 1: Full position ──
            if cfg.partial_exit_enabled:
                # PBT mode: wait for TP1 hit to trigger partial close
                # v12.5: Use best_gain (intra-bar high/low) for TP1 detection
                # SL handled server-side; worst_gain here is backup
                if best_gain >= target:
                    pos['partial_closed'] = True
                    pos['highest_since_tp1'] = best_gain
                    if cfg.breakeven_after_partial:
                        pos['trail_floor'] = 0.0  # breakeven floor
                    else:
                        pos['trail_floor'] = -effective_stop
                    return 'partial_close'

                if worst_gain < -effective_stop:
                    return 'stop_loss'

                if pos['bars_held'] >= cfg.max_hold:
                    return 'max_hold'

                # Pre-TP1 trailing (only if gain exceeds activation threshold)
                # v12.3.3: configurable params (were hardcoded 0.5/0.3)
                # v12.5.3: Use worst_gain (intra-bar) for parity with Phase 2.
                # Close-based gain missed intra-bar breach → TONUSDT gave back
                # 1.10% best_pnl and exited at +0.07% (2026-02-22 incident).
                if (pos['best_pnl'] > target * cfg.pre_tp1_trail_activation
                        and worst_gain < pos['best_pnl'] * cfg.pre_tp1_trail_factor):
                    return 'trailing_stop'

            else:
                # Standard mode (no partial exit)
                # v12.5: best_gain for TP, worst_gain for SL
                if best_gain >= target:
                    return 'target'
                if worst_gain < -effective_stop:
                    return 'stop_loss'
                # v12.3.3: configurable pre-TP1 trail params
                # v12.5.3: worst_gain for intra-bar parity
                if (pos['best_pnl'] > target * cfg.pre_tp1_trail_activation
                        and worst_gain < pos['best_pnl'] * cfg.pre_tp1_trail_factor):
                    return 'trailing_stop'
                if pos['bars_held'] >= cfg.max_hold:
                    return 'max_hold'

        else:
            # ── Phase 2: Runner (after partial close) ──
            # Track runner peak (use best_gain for peak, worst_gain for stop)
            highest = pos.get('highest_since_tp1', best_gain)
            if best_gain > highest:
                pos['highest_since_tp1'] = best_gain
                highest = best_gain

            # ATR-based trailing stop for runner
            atr_val = current_atr_pct if current_atr_pct is not None else 0.02
            trail_stop = highest - cfg.trail_atr_mult * atr_val

            # Apply breakeven floor
            floor_val = pos.get('trail_floor')
            if floor_val is not None and trail_stop < floor_val:
                trail_stop = floor_val

            # v12.3.3: Store computed trail_stop for server-side ratcheting.
            # Scalper reads this after update_position() to sync server SL.
            pos['_computed_trail_stop'] = trail_stop

            # v12.5: worst_gain for intra-bar trail stop check
            if worst_gain <= trail_stop:
                return 'trail_runner'

            # v12.4 HybTight: Extended time exit for runner (configurable mult)
            if pos['bars_held'] >= cfg.max_hold * cfg.runner_max_hold_mult:
                return 'time_runner'

            # NOTE: No fixed TP for runner. The runner rides the trend
            # and exits ONLY via ATR trailing stop or time limit.
            # A fixed TP here would fire immediately since partial was
            # triggered at gain >= target (same level).

        return None

    @property
    def active_positions(self) -> dict[str, dict]:
        return dict(self._positions)

    @property
    def n_active(self) -> int:
        return len(self._positions)

