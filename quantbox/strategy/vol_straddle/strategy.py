"""VolStraddle Strategy — VR compression detector with multi-TF signals.

Production strategy implementing the R35 A1-D50 variant.
D_mom8: VR < 0.50 → directional entry via momentum(8).
Multi-TF: scans 15m, 1h, 4h independently; per-symbol dedup.

API contract matches RubberBandStrategy so the scalper can plug it in.

ZERO LOOKAHEAD:
  - All VR/momentum computed on COMPLETED bars only.
  - 1h and 4h data resampled from completed 15m bars (no developing bar).
  - Cross-TF signals use only their own completed TF data.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    compute_momentum,
    compute_squeeze_dur,
    compute_vr,
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)

logger = logging.getLogger('VolStraddle')


class VolStraddleStrategy:
    """VR compression (volatility squeeze) strategy.

    Detects VR < threshold on 15m/1h/4h and enters directionally
    using 8-bar momentum for direction assignment.

    Matches R35 A1-D50 backtest logic exactly for parity:
      - VR = ATR(7) / ATR(50), threshold 0.50 (D_mom8)
      - D direction = momentum(8) sign
      - SL = 1%, TP = 5%, max_hold = 192/48/12 per TF
      - Per-symbol dedup: one position per symbol at any time

    Args:
        config: VolStraddleConfig with strategy parameters.
    """

    def __init__(self, config: VolStraddleConfig | None = None):
        self.config = config or VolStraddleConfig()

        # ── Per-symbol state ──
        self._positions: dict[str, dict] = {}

        # ── Market data storage (from update_market_data) ──
        self._high_data: dict[str, np.ndarray] = {}
        self._low_data: dict[str, np.ndarray] = {}
        self._volume_data: dict[str, np.ndarray] = {}

        # ── HTF data (from update_htf_data) ──
        self._htf_4h: dict[str, dict[str, np.ndarray]] = {}

        # ── Signal metadata (set by on_bar, read by scalper) ──
        self._last_signal_meta: dict[str, dict] = {}

        # ── VR debug (per symbol per TF) ──
        self._last_vr_15m: dict[str, float] = {}
        self._last_vr_1h: dict[str, float] = {}
        self._last_vr_4h: dict[str, float] = {}
        # Legacy single VR for backward compat
        self._last_vr: dict[str, float] = {}

        # ── BTC data for C2 conviction condition ──
        self._btc_close: np.ndarray | None = None

    # ══════════════════════════════════════════════════════════════════
    #  Signal Generation
    # ══════════════════════════════════════════════════════════════════

    def on_bar(
        self, symbol: str, close_prices: np.ndarray,
        btc_close: np.ndarray | None = None,
    ) -> int:
        """Check VR compression signal across all enabled TFs.

        Scans 15m (from close_prices + stored high/low), 1h (resampled
        from 15m), and 4h (from stored HTF data or resampled).
        Returns the FIRST valid signal found (priority: 4h > 1h > 15m
        because higher TF signals have better quality per R31).

        ZERO LOOKAHEAD: only completed bars are used.
        - 15m: close_prices are from download_klines (developing bar dropped).
        - 1h: resampled from completed 15m bars (remainder truncated).
        - 4h: fetched separately (developing bar dropped by download_klines).

        Args:
            symbol: Trading symbol.
            close_prices: 15m close prices (200 completed bars).
            btc_close: BTC close prices (used for C2 conviction condition).

        Returns:
            +1 (long), -1 (short), or 0 (no signal).
        """
        self._current_symbol = symbol
        cfg = self.config

        # Store BTC close for C2 conviction condition
        if btc_close is not None:
            self._btc_close = btc_close

        # Per-symbol dedup: no signal when position already open
        # In straddle mode, both legs open simultaneously from one signal,
        # so if ANY leg exists we skip (no incremental straddle adds).
        if self.has_position(symbol):
            return 0

        high_15m = self._high_data.get(symbol)
        low_15m = self._low_data.get(symbol)
        vol_15m = self._volume_data.get(symbol)

        if high_15m is None or low_15m is None:
            return 0

        # Ensure arrays are same length (truncate to shortest)
        min_len = min(len(close_prices), len(high_15m), len(low_15m))
        if min_len < cfg.vr_long + 10:
            return 0
        c15 = close_prices[-min_len:]
        h15 = high_15m[-min_len:]
        l15 = low_15m[-min_len:]
        v15 = vol_15m[-min_len:] if vol_15m is not None else np.ones(min_len)

        # ── Check TFs in priority order (higher TF first) ──
        # Higher TF signals are less frequent but higher quality.

        # 4h: from stored HTF data (directly fetched by scalper)
        if cfg.enable_4h:
            data_4h = self._htf_4h.get(symbol)
            if data_4h is not None:
                sig_4h = self._check_tf_signal(
                    data_4h['close'], data_4h['high'], data_4h['low'],
                    '4h',
                )
                if sig_4h != 0:
                    return sig_4h

        # 1h: resampled from 15m (completed bars only)
        if cfg.enable_1h:
            # Need enough 15m bars so that resampled 1h array >= warmup_1h + 1
            # Each 1h bar = 4 × 15m bars, plus _check_tf_signal checks warmup
            if min_len >= 4 * (cfg.warmup_1h + 1):
                c1h, h1h, l1h, v1h = resample_ohlcv(c15, h15, l15, v15, 4)
                if len(c1h) >= cfg.warmup_1h + 1:
                    sig_1h = self._check_tf_signal(c1h, h1h, l1h, '1h')
                    if sig_1h != 0:
                        return sig_1h

        # 15m: direct from downloaded data
        if cfg.enable_15m:
            sig_15m = self._check_tf_signal(c15, h15, l15, '15m')
            if sig_15m != 0:
                return sig_15m

        return 0

    def _check_tf_signal(
        self,
        close: np.ndarray, high: np.ndarray, low: np.ndarray,
        tf: str,
    ) -> int:
        """Check VR compression signal for a single timeframe.

        Returns:
            +1 (long), -1 (short), 2 (straddle), or 0 (no signal).
        Stores signal metadata for the scalper to query.

        Signal priority: S_d8 straddle (signal=2) > D_mom8 directional.
        S_d8 requires sustained squeeze (duration ≥ min_dur), so it fires
        LATER than D_mom8. They don't compete in practice.

        CRITICAL CHECK: Only fires on the LAST bar (current bar close).
        Previous bars are used for VR computation but not for entry.
        """
        cfg = self.config
        warmup = cfg.get_warmup(tf)

        if len(close) < warmup + 1:
            return 0

        # Compute VR
        vr = compute_vr(close, high, low, cfg.vr_short, cfg.vr_long)

        # Store last VR for debug logging (even if no signal)
        vr_val = float(vr[-1]) if len(vr) > 0 and not np.isnan(vr[-1]) else 999.0
        self._last_vr[self._current_symbol] = vr_val
        # Store per-TF VR
        if tf == '15m':
            self._last_vr_15m[self._current_symbol] = vr_val
        elif tf == '1h':
            self._last_vr_1h[self._current_symbol] = vr_val
        elif tf == '4h':
            self._last_vr_4h[self._current_symbol] = vr_val

        # ── S_d8 straddle check (higher priority) ──
        # Requires sustained VR compression for min_dur bars.
        if cfg.straddle_enabled:
            s_sig = gen_vr_dur_signal(
                vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup,
            )
            if s_sig[-1] == 1:
                # Parity audit: check if D_mom8 would also fire
                # In R35 backtest, D and S run independently then dedup.
                # In live, S has priority. Log overlap for monitoring.
                d_sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
                d_overlap = bool(d_sig[-1] == 1)
                if d_overlap:
                    mom = compute_momentum(close, cfg.momentum_period)
                    d_dir = int(make_direction_momentum(d_sig, mom)[-1])
                    logger.info(
                        f"[{self._current_symbol}] PARITY: S_d8+D_mom8 overlap "
                        f"TF={tf} VR={vr_val:.3f} — D would be "
                        f"{'LONG' if d_dir == 1 else 'SHORT'} "
                        f"(straddle takes priority)"
                    )

                # ── R34 Conviction Score ──
                sq_dur = compute_squeeze_dur(vr, cfg.s_vr_threshold)
                sq_dur_val = int(sq_dur[-1])
                # B3: For S_d8, check if D_mom8 would also fire
                score, cond = self._compute_conviction_score(
                    close, vr, tf,
                    is_straddle=True,
                    d_overlap=d_overlap,
                    squeeze_dur=sq_dur_val,
                )

                # Min conviction filter (T3-Triple mode)
                if cfg.cw_enabled and score < cfg.min_conviction_score:
                    logger.debug(
                        f"[{self._current_symbol}] S_d8 blocked: "
                        f"conviction={score} < min={cfg.min_conviction_score} "
                        f"TF={tf} {cond}"
                    )
                    # Don't return — fall through to D_mom8 check
                else:
                    self._last_signal_meta[self._current_symbol] = {
                        'tf': tf,
                        'sl': cfg.s_sl,
                        'tp': cfg.s_tp,
                        'max_hold': cfg.get_max_hold(tf),
                        'vr': vr_val,
                        'momentum': 0.0,
                        'straddle': True,
                        'd_overlap': d_overlap,
                        'conviction_score': score,
                        'conviction_cond': cond,
                    }
                    logger.info(
                        f"[{self._current_symbol}] S_d8 straddle signal "
                        f"TF={tf} VR={vr_val:.3f} conviction={score}/4 "
                        f"{cond} "
                        f"(threshold={cfg.s_vr_threshold})"
                    )
                    return 2  # Straddle signal

        # ── D_mom8: VR < threshold → directional entry ──
        sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)

        # Only check the LAST bar (current bar close)
        if sig[-1] != 1:
            return 0

        # Direction from momentum
        mom = compute_momentum(close, cfg.momentum_period)
        dire = make_direction_momentum(sig, mom)

        direction = int(dire[-1])
        if direction == 0:
            return 0

        # ── R34 Conviction Score for D_mom8 ──
        # B3: Check if S_d8 would also fire (reverse overlap)
        s_overlap = False
        if cfg.straddle_enabled and cfg.b3_enabled:
            s_sig = gen_vr_dur_signal(
                vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup,
            )
            s_overlap = bool(s_sig[-1] == 1)

        score, cond = self._compute_conviction_score(
            close, vr, tf,
            is_straddle=False,
            d_overlap=s_overlap,  # B3: S would also fire
            squeeze_dur=0,  # Not relevant for D_mom8
        )

        # Min conviction filter (T3-Triple mode)
        if cfg.cw_enabled and score < cfg.min_conviction_score:
            logger.debug(
                f"[{self._current_symbol}] D_mom8 blocked: "
                f"conviction={score} < min={cfg.min_conviction_score} "
                f"TF={tf} {cond}"
            )
            return 0

        # Store metadata for scalper to query after on_bar
        # (which TF, SL/TP, max_hold for this specific signal)
        mom_val = float(mom[-1])
        self._last_signal_meta[self._current_symbol] = {
            'tf': tf,
            'sl': cfg.d_sl,
            'tp': cfg.d_tp,
            'max_hold': cfg.get_max_hold(tf),
            'vr': float(vr[-1]) if not np.isnan(vr[-1]) else 0.0,
            'momentum': mom_val,
            'conviction_score': score,
            'conviction_cond': cond,
        }
        logger.info(
            f"[{self._current_symbol}] D_mom8 signal "
            f"TF={tf} VR={vr_val:.3f} {'LONG' if direction == 1 else 'SHORT'} "
            f"mom={mom_val:.6f} conviction={score}/4 {cond} "
            f"(threshold={cfg.d_vr_threshold})"
        )
        return direction

    # Track current symbol for signal metadata storage
    _current_symbol: str = ''

    # ══════════════════════════════════════════════════════════════════
    #  Conviction Score (R34 CW-Tier)
    # ══════════════════════════════════════════════════════════════════

    def _compute_conviction_score(
        self,
        close: np.ndarray,
        vr: np.ndarray,
        tf: str,
        is_straddle: bool,
        d_overlap: bool,
        squeeze_dur: int = 0,
    ) -> tuple[int, dict[str, bool]]:
        """Compute conviction score (0-4) from R34 conditions.

        Conditions:
            E2: SMA Proximity — |close - SMA(N)| / SMA(N) < threshold
            C2: BTC Neutral — |BTC momentum(8)| < threshold
            B3: Confluence — both D_mom8 and S_d8 would fire
            A3: Quality+ — D: VR < deep threshold; S: duration ≥ long dur

        Args:
            close: Close prices for this timeframe.
            vr: VR array for this timeframe.
            tf: Timeframe label ('15m', '1h', '4h').
            is_straddle: True if S_d8 signal, False if D_mom8.
            d_overlap: True if both D and S would fire simultaneously.
            squeeze_dur: Current squeeze duration (for S_d8 A3 check).

        Returns:
            Tuple of (score, condition_details_dict).
        """
        cfg = self.config
        score = 0
        conditions: dict[str, bool] = {
            'e2_sma': False,
            'c2_btc': False,
            'b3_confluence': False,
            'a3_quality': False,
        }

        # ── E2: SMA Proximity ──
        if cfg.e2_enabled and len(close) >= cfg.e2_sma_period:
            sma = np.mean(close[-cfg.e2_sma_period:])
            if sma > 0:
                proximity = abs(close[-1] - sma) / sma
                if proximity < cfg.e2_threshold:
                    conditions['e2_sma'] = True
                    score += 1

        # ── C2: BTC Neutral ──
        if cfg.c2_enabled and self._btc_close is not None:
            btc = self._btc_close
            mom_period = cfg.momentum_period  # Same 8-bar momentum
            if len(btc) > mom_period:
                btc_mom = btc[-1] / btc[-mom_period - 1] - 1.0
                if abs(btc_mom) < cfg.c2_btc_mom_threshold:
                    conditions['c2_btc'] = True
                    score += 1

        # ── B3: Confluence (both D_mom8 and S_d8 would fire) ──
        if cfg.b3_enabled:
            if d_overlap:
                # When S_d8 fires: d_overlap was checked in _check_tf_signal
                # When D_mom8 fires: we need to check if S_d8 would also fire
                conditions['b3_confluence'] = True
                score += 1

        # ── A3: Quality+ ──
        if cfg.a3_enabled:
            vr_val = float(vr[-1]) if not np.isnan(vr[-1]) else 999.0
            if is_straddle:
                # S_d8: extended squeeze duration
                if squeeze_dur >= cfg.a3_long_squeeze_dur:
                    conditions['a3_quality'] = True
                    score += 1
            else:
                # D_mom8: extra deep VR compression
                if vr_val < cfg.a3_deep_vr_threshold:
                    conditions['a3_quality'] = True
                    score += 1

        return score, conditions

    # ══════════════════════════════════════════════════════════════════
    #  Signal Metadata Query
    # ══════════════════════════════════════════════════════════════════

    def get_last_signal_meta(self, symbol: str) -> dict:
        """Get metadata for the last signal generated for this symbol.

        Called by the scalper after on_bar() returns non-zero.

        Returns:
            Dict with 'tf', 'sl', 'tp', 'max_hold', 'vr', 'momentum'.
            Empty dict if no signal metadata available.
        """
        return self._last_signal_meta.get(symbol, {})

    def get_last_vr(self, symbol: str) -> float | None:
        """Get last computed VR value for this symbol (debug only).

        Returns VR value or None if not yet computed.
        """
        return self._last_vr.get(symbol)

    def get_vr_all_tf(self, symbol: str) -> dict[str, float]:
        """Get VR for all timeframes (debug only).

        Returns dict with '15m', '1h', '4h' keys.
        """
        return {
            '15m': self._last_vr_15m.get(symbol, 999.0),
            '1h': self._last_vr_1h.get(symbol, 999.0),
            '4h': self._last_vr_4h.get(symbol, 999.0),
        }

    # ══════════════════════════════════════════════════════════════════
    #  Position Management
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def parse_pos_key(key: str) -> tuple[str, str | None]:
        """Parse position key → (symbol, position_side or None).

        Compound keys: 'BTCUSDT:LONG' → ('BTCUSDT', 'LONG')
        Simple keys: 'BTCUSDT' → ('BTCUSDT', None)
        """
        if ':' in key:
            symbol, side = key.split(':', 1)
            return symbol, side
        return key, None

    def update_position(
        self, symbol: str, current_price: float,
        current_atr_pct: float | None = None,
        current_high: float | None = None,
        current_low: float | None = None,
    ) -> str | None:
        """Check if position should be closed.

        Simple SL/TP/max_hold exit — matches R35 backtest exactly.
        NO PBT, NO trailing, NO partial close.
        This ensures backtest-live parity.

        Uses intra-bar high/low for SL/TP detection when available,
        matching the backtest's bar-by-bar SL/TP checking.

        Args:
            current_price: Current close price.
            current_atr_pct: Not used (kept for interface compatibility).
            current_high: Bar high (for intra-bar SL/TP detection).
            current_low: Bar low (for intra-bar SL/TP detection).

        Returns:
            Exit reason string or None.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        pos['bars_held'] += 1
        entry = pos['entry_price']
        sl_pct = pos.get('sl', self.config.d_sl)
        tp_pct = pos.get('tp', self.config.d_tp)
        max_hold = pos.get('max_hold', self.config.max_hold_15m)

        # Intra-bar extremes for SL/TP check (parity with backtest
        # which checks high/low within each bar)
        if current_high is not None and current_low is not None:
            if pos['signal'] == 1:  # LONG
                worst_gain = (current_low - entry) / entry
                best_gain = (current_high - entry) / entry
            else:  # SHORT
                worst_gain = (entry - current_high) / entry
                best_gain = (entry - current_low) / entry
        else:
            gain = ((current_price - entry) / entry if pos['signal'] == 1
                    else (entry - current_price) / entry)
            worst_gain = gain
            best_gain = gain

        # Track best PnL for logging
        if best_gain > pos.get('best_pnl', 0.0):
            pos['best_pnl'] = best_gain

        # SL check (backtest uses low[j] for long, high[j] for short)
        if worst_gain <= -sl_pct:
            logger.info(
                f"[{symbol}] SL hit: worst_gain={worst_gain*100:+.3f}% "
                f"vs SL={sl_pct*100:.1f}% bars={pos['bars_held']} "
                f"entry={entry:.6f} "
                f"{'low' if pos['signal']==1 else 'high'}="
                f"{current_low if pos['signal']==1 else current_high}"
            )
            return 'stop_loss'

        # TP check (backtest uses high[j] for long, low[j] for short)
        if best_gain >= tp_pct:
            logger.info(
                f"[{symbol}] TP hit: best_gain={best_gain*100:+.3f}% "
                f"vs TP={tp_pct*100:.1f}% bars={pos['bars_held']} "
                f"entry={entry:.6f} "
                f"{'high' if pos['signal']==1 else 'low'}="
                f"{current_high if pos['signal']==1 else current_low}"
            )
            return 'target'

        # Max hold (time-based exit)
        if pos['bars_held'] >= max_hold:
            logger.info(
                f"[{symbol}] Max hold reached: bars={pos['bars_held']}/{max_hold} "
                f"best_gain={best_gain*100:+.3f}% worst_gain={worst_gain*100:+.3f}%"
            )
            return 'max_hold'

        return None

    # ══════════════════════════════════════════════════════════════════
    #  Position Lifecycle (interface methods)
    # ══════════════════════════════════════════════════════════════════

    def has_position(self, symbol: str) -> bool:
        """Check if a position exists for this symbol.

        In hedge mode, checks compound keys (symbol:LONG / symbol:SHORT)
        and also the plain key as fallback for backward compatibility.
        """
        if self.config.hedge_mode:
            return (
                f"{symbol}:LONG" in self._positions
                or f"{symbol}:SHORT" in self._positions
                or symbol in self._positions
            )
        return symbol in self._positions

    def open_position(
        self, symbol: str, signal: int, entry_price: float,
        notional: float, leverage: float, move_size: float,
        entry_stop_mult: float | None = None,
        server_sl_order_id: str | None = None,
        server_tp_order_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a new position.

        Stores TF-specific SL/TP/max_hold from signal metadata.
        In hedge mode, uses compound key (symbol:LONG / symbol:SHORT).

        Args:
            symbol: Trading symbol.
            signal: +1 (long) or -1 (short).
            entry_price: Fill price.
            notional: Position notional value.
            leverage: Applied leverage.
            move_size: Normalized to 1.0 for vol straddle.
            entry_stop_mult: SL as fraction (e.g. 0.01 = 1%).
            server_sl_order_id: Exchange SL order ID.
            server_tp_order_id: Exchange TP order ID.
            **kwargs: Extra fields (sl, tp, tf, max_hold, position_side).
        """
        if entry_price <= 0:
            raise ValueError(
                f"Invalid entry_price for {symbol}: {entry_price}"
            )

        # Determine position key (compound in hedge mode)
        position_side = kwargs.get('position_side')
        if position_side:
            pos_key = f"{symbol}:{position_side}"
        else:
            pos_key = symbol

        # Get TF-specific parameters from signal metadata or kwargs
        meta = self._last_signal_meta.get(symbol, {})
        sl = kwargs.get('sl', meta.get('sl', self.config.d_sl))
        tp = kwargs.get('tp', meta.get('tp', self.config.d_tp))
        tf = kwargs.get('tf', meta.get('tf', '15m'))
        max_hold = kwargs.get('max_hold', meta.get(
            'max_hold', self.config.get_max_hold(tf),
        ))

        self._positions[pos_key] = {
            'symbol': symbol,  # Actual Binance symbol (for API calls)
            'signal': signal,
            'entry_price': entry_price,
            'notional': notional,
            'leverage': leverage,
            'move_size': move_size,
            'bars_held': 0,
            'best_pnl': 0.0,
            # VolStraddle-specific
            'sl': sl,
            'tp': tp,
            'tf': tf,
            'max_hold': max_hold,
            # Server bracket IDs
            'server_sl_order_id': server_sl_order_id,
            'server_tp_order_id': server_tp_order_id,
            # Compatibility fields (used by scalper internals)
            'entry_stop_mult': entry_stop_mult or sl,
            'partial_closed': False,
            'confidence': 0,
            'highest_since_tp1': 0.0,
            'trail_floor': None,
            # Hedge mode
            'position_side': position_side,
        }

        logger.info(
            f"[{symbol}] Position opened: "
            f"{'LONG' if signal == 1 else 'SHORT'} "
            f"TF={tf} SL={sl*100:.1f}% TP={tp*100:.1f}% "
            f"max_hold={max_hold} VR={meta.get('vr', 0):.3f}"
            f"{f' (hedge:{position_side})' if position_side else ''}"
        )

    def close_position(self, symbol: str) -> dict | None:
        """Remove position and return its info.

        Accepts raw symbol or compound key (symbol:LONG / symbol:SHORT).
        """
        return self._positions.pop(symbol, None)

    @property
    def active_positions(self) -> dict[str, dict]:
        """Return copy of all active positions."""
        return dict(self._positions)

    @property
    def n_active(self) -> int:
        """Number of active positions."""
        return len(self._positions)

    # ══════════════════════════════════════════════════════════════════
    #  Market Data Updates
    # ══════════════════════════════════════════════════════════════════

    def update_market_data(
        self, symbol: str, volume: np.ndarray, taker_buy: np.ndarray,
        high: np.ndarray, low: np.ndarray,
    ) -> None:
        """Store 15m OHLCV data for VR computation.

        Called by scalper before on_bar() each cycle.

        Args:
            symbol: Trading symbol.
            volume: 15m volume array.
            taker_buy: Taker buy volume (not used by vol straddle).
            high: 15m high prices.
            low: 15m low prices.
        """
        self._high_data[symbol] = high
        self._low_data[symbol] = low
        self._volume_data[symbol] = volume

    def update_htf_data(
        self, symbol: str,
        close_4h: np.ndarray,
        high_4h: np.ndarray | None = None,
        low_4h: np.ndarray | None = None,
    ) -> None:
        """Store 4h OHLCV data for multi-TF VR computation.

        For vol straddle, we need full OHLCV (not just close) for VR.
        If high/low are not provided, 4h VR is computed from resampled
        15m data instead (done in on_bar).

        Args:
            symbol: Trading symbol.
            close_4h: 4h close prices.
            high_4h: 4h high prices (optional).
            low_4h: 4h low prices (optional).
        """
        if high_4h is not None and low_4h is not None:
            self._htf_4h[symbol] = {
                'close': close_4h,
                'high': high_4h,
                'low': low_4h,
            }
        else:
            # Without high/low, we can't compute VR on raw 4h data.
            # Fall back to resampled 15m data in on_bar.
            self._htf_4h.pop(symbol, None)

    # ══════════════════════════════════════════════════════════════════
    #  Compatibility Methods (used by scalper/positions internals)
    # ══════════════════════════════════════════════════════════════════

    def get_effective_stop_mult(
        self, close_prices: np.ndarray,
        symbol: str | None = None,
        move_size: float | None = None,
    ) -> float:
        """Return SL fraction for the current signal.

        For vol straddle, stop is a fixed percentage (not move-based).
        Returns d_sl (0.01 = 1%) since move_size is always 1.0.

        Compatible with scalper's _open_position interface.
        """
        meta = self._last_signal_meta.get(symbol, {}) if symbol else {}
        return meta.get('sl', self.config.d_sl)

    def get_last_confidence(self, symbol: str) -> int:
        """Return confidence level (always 0 — vol straddle has no scoring)."""
        return 0

    def was_mtf_confirmed(self, symbol: str) -> bool:
        """Return False — MTF confirmation is not used by vol straddle."""
        return False

    def get_last_atr_pct(self, symbol: str) -> float:
        """Return last known ATR% for the symbol (from stored 15m data)."""
        high = self._high_data.get(symbol)
        low = self._low_data.get(symbol)
        if high is None or low is None or len(high) < 2:
            return 0.02  # conservative default

        # Simple ATR estimate from last few bars
        n = min(14, len(high))
        atr_sum = 0.0
        for i in range(-n, 0):
            atr_sum += (high[i] - low[i]) / (0.5 * (high[i] + low[i]))
        return atr_sum / n

    def diagnose_bar(
        self, symbol: str, close_prices: np.ndarray,
        btc_close: np.ndarray | None = None,
    ) -> dict:
        """Return diagnostic info for the current bar (for JSONL logging)."""
        meta = self._last_signal_meta.get(symbol, {})
        return {
            'symbol': symbol,
            'strategy': 'vol_straddle',
            'vr': meta.get('vr', None),
            'tf': meta.get('tf', None),
            'momentum': meta.get('momentum', None),
        }

    # ══════════════════════════════════════════════════════════════════
    #  State Serialization (for crash recovery)
    # ══════════════════════════════════════════════════════════════════

    def to_state_dict(self) -> dict:
        """Serialize strategy state for persistence."""
        positions = {}
        for sym, pos in self._positions.items():
            positions[sym] = {
                k: v for k, v in pos.items()
                if not k.startswith('_')
            }
        return {
            'strategy_name': 'vol_straddle',
            'positions': positions,
        }

    def from_state_dict(self, state: dict) -> None:
        """Restore strategy state from persisted data."""
        self._positions = {}
        positions = state.get('positions', {})
        for sym, pos_data in positions.items():
            self._positions[sym] = pos_data
