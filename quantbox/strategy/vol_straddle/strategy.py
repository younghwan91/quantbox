"""VolStraddle Strategy — VR compression detector with multi-TF signals.

Production strategy implementing the R35 A1-D50 variant.
D_mom8: VR < 0.50 → directional entry via momentum(8).
Multi-TF: scans 15m, 1h, 4h independently; per-symbol dedup.

R44d V17: MA Cluster Squeeze (4h-only) as complementary strategy.
4 SMAs (20/50/100/200) cluster → breakout + momentum direction.
93.7% non-overlapping with VR squeeze — fires when VR is idle.

API contract matches RubberBandStrategy so the scalper can plug it in.

ZERO LOOKAHEAD:
  - All VR/momentum computed on COMPLETED bars only.
  - 1h and 4h data resampled from completed 15m bars (no developing bar).
  - Cross-TF signals use only their own completed TF data.
  - MA cluster uses only completed 4h bars (directly fetched).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    check_ma_cluster,
    compute_momentum,
    compute_sma,
    compute_squeeze_dur,
    compute_vr,
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)

logger = logging.getLogger("VolStraddle")

# bars_held increments per 15m cycle, so max_hold must be in 15m bars.
# Config stores max_hold in native TF bars — multiply to convert.
_TF_TO_15M = {"15m": 1, "30m": 2, "1h": 4, "4h": 16}


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
        self._taker_buy_data: dict[str, np.ndarray] = {}  # F5 TBR gate

        # ── HTF data (from update_htf_data) ──
        self._htf_4h: dict[str, dict[str, np.ndarray]] = {}

        # ── Signal metadata (set by on_bar, read by scalper) ──
        self._last_signal_meta: dict[str, dict] = {}

        # ── VR debug (per symbol per TF) ──
        self._last_vr_15m: dict[str, float] = {}
        self._last_vr_30m: dict[str, float] = {}
        self._last_vr_1h: dict[str, float] = {}
        self._last_vr_4h: dict[str, float] = {}
        # Legacy single VR for backward compat
        self._last_vr: dict[str, float] = {}

        # ── BTC data for C2 conviction condition ──
        self._btc_close: np.ndarray | None = None

        # ── MA Cluster Squeeze state (R44d V17) ──
        # Per-symbol cooldown for one-signal-per-episode logic.
        # Duration is computed statelessly from the 4h data (backward scan),
        # so only the cooldown flag needs per-symbol state.
        # in_cooldown: True after signal fired, reset when cluster dissolves.
        self._ma_cluster_cooldown: dict[str, np.ndarray] = {}  # 1-elem array for numba

        # ── MA Cluster scan state (populated every scan, even when no signal) ──
        # Used by scalper to produce "MA Cluster monitor" line parallel to
        # "Squeeze monitor". Keys: duration, spread, momentum, rejection.
        # rejection values: 'no_cluster', 'pre_duration', 'zombie',
        #   'cooldown', 'no_breakout', 'momentum_fail', 'fired'
        self._ma_cluster_state: dict[str, dict] = {}

        # ── Straddle rollback cooldown (per-symbol cycle counter) ──
        # After a half-open rollback, skip this symbol for N cycles
        # to prevent the LONG-open → SHORT-fail → rollback loop.
        self._straddle_rollback_cooldown: dict[str, int] = {}

        # ── Signal source tracking (populated by on_bar for each symbol) ──
        # Allows scalper to apply strategy-specific gates after on_bar returns.
        # Values: 'vr_directional' | 'vr_straddle' | 'ma_cluster' | 'none'
        self._last_signal_source: dict[str, str] = {}

    # ══════════════════════════════════════════════════════════════════
    #  Signal Generation
    # ══════════════════════════════════════════════════════════════════

    def on_bar(
        self,
        symbol: str,
        close_prices: np.ndarray,
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

        # Straddle rollback cooldown: skip symbol for N cycles after
        # a half-open rollback to prevent LONG→SHORT-fail→rollback loop.
        remaining = self._straddle_rollback_cooldown.get(symbol, 0)
        if remaining > 0:
            self._straddle_rollback_cooldown[symbol] = remaining - 1
            if remaining == 1:
                logger.info(f"[{symbol}] Straddle rollback cooldown expired")
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
                    data_4h["close"],
                    data_4h["high"],
                    data_4h["low"],
                    "4h",
                )
                if sig_4h != 0:
                    self._last_signal_source[symbol] = "vr_directional"
                    return sig_4h

        # 1h: resampled from 15m (completed bars only)
        if cfg.enable_1h:
            # Need enough 15m bars so that resampled 1h array >= warmup_1h + 1
            # Each 1h bar = 4 × 15m bars, plus _check_tf_signal checks warmup
            if min_len >= 4 * (cfg.warmup_1h + 1):
                c1h, h1h, l1h, v1h = resample_ohlcv(c15, h15, l15, v15, 4)
                if len(c1h) >= cfg.warmup_1h + 1:
                    sig_1h = self._check_tf_signal(c1h, h1h, l1h, "1h")
                    if sig_1h != 0:
                        self._last_signal_source[symbol] = "vr_directional"
                        return sig_1h

        # 30m: resampled from 15m (completed bars only) — r57 A0
        if cfg.enable_30m:
            if min_len >= 2 * (cfg.warmup_30m + 1):
                c30, h30, l30, v30 = resample_ohlcv(c15, h15, l15, v15, 2)
                if len(c30) >= cfg.warmup_30m + 1:
                    sig_30m = self._check_tf_signal(c30, h30, l30, "30m")
                    if sig_30m != 0:
                        self._last_signal_source[symbol] = "vr_directional"
                        return sig_30m

        # 15m: direct from downloaded data
        if cfg.enable_15m:
            sig_15m = self._check_tf_signal(c15, h15, l15, "15m")
            if sig_15m != 0:
                self._last_signal_source[symbol] = "vr_directional"
                return sig_15m

        # ── MA Cluster Squeeze (R44d V17, 4h-only) ──
        # Checked AFTER VR signals: VR has priority (more backtested).
        # Uses directly-fetched 4h data (no resampling needed).
        if cfg.ma_cluster_enabled and cfg.enable_4h:
            data_4h = self._htf_4h.get(symbol)
            if data_4h is not None:
                sig_ma = self._check_ma_cluster_signal(
                    data_4h["close"],
                    data_4h["high"],
                    data_4h["low"],
                    symbol,
                )
                if sig_ma != 0:
                    self._last_signal_source[symbol] = "ma_cluster"
                    return sig_ma

        self._last_signal_source[symbol] = "none"
        return 0

    def _check_tf_signal(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
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
        if tf == "15m":
            self._last_vr_15m[self._current_symbol] = vr_val
        elif tf == "30m":
            self._last_vr_30m[self._current_symbol] = vr_val
        elif tf == "1h":
            self._last_vr_1h[self._current_symbol] = vr_val
        elif tf == "4h":
            self._last_vr_4h[self._current_symbol] = vr_val

        # ── S_d8 straddle check (higher priority) ──
        # Requires sustained VR compression for min_dur bars.
        if cfg.straddle_enabled:
            s_sig = gen_vr_dur_signal(
                vr,
                cfg.s_vr_threshold,
                cfg.s_squeeze_dur,
                warmup,
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
                    close,
                    vr,
                    tf,
                    is_straddle=True,
                    d_overlap=d_overlap,
                    squeeze_dur=sq_dur_val,
                )

                # Min conviction filter (T3-Triple mode)
                if cfg.cw_enabled and score < cfg.min_conviction_score:
                    logger.info(
                        f"[{self._current_symbol}] S_d8 BLOCKED: "
                        f"conviction={score} < min={cfg.min_conviction_score} "
                        f"TF={tf} {cond} VR={vr_val:.3f}"
                    )
                    # Don't return — fall through to D_mom8 check
                else:
                    self._last_signal_meta[self._current_symbol] = {
                        "tf": tf,
                        "sl": cfg.s_sl,
                        "tp": cfg.s_tp,
                        "max_hold": cfg.get_max_hold(tf) * _TF_TO_15M.get(tf, 1),
                        "vr": vr_val,
                        "momentum": 0.0,
                        "straddle": True,
                        "strategy_type": "vr_straddle",
                        "d_overlap": d_overlap,
                        "conviction_score": score,
                        "conviction_cond": cond,
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

        # ── F5: TBR z-score Gate (R46 validated, D_mom8 only) ──
        if self.config.f5_tbr_enabled:
            tbr_z = self._compute_tbr_zscore(self._current_symbol)
            is_long = direction == 1
            blocked = (is_long and tbr_z > self.config.f5_tbr_z_threshold) or (
                not is_long and tbr_z < -self.config.f5_tbr_z_threshold
            )
            if blocked:
                logger.info(
                    f"[{self._current_symbol}] D_mom8 BLOCKED by F5-TBR: "
                    f"tbr_z={tbr_z:.3f} thresh={self.config.f5_tbr_z_threshold} "
                    f"dir={'LONG' if is_long else 'SHORT'} TF={tf}"
                )
                return 0

        # ── R34 Conviction Score for D_mom8 ──
        # B3: Check if S_d8 would also fire (reverse overlap)
        s_overlap = False
        if cfg.straddle_enabled and cfg.b3_enabled:
            s_sig = gen_vr_dur_signal(
                vr,
                cfg.s_vr_threshold,
                cfg.s_squeeze_dur,
                warmup,
            )
            s_overlap = bool(s_sig[-1] == 1)

        score, cond = self._compute_conviction_score(
            close,
            vr,
            tf,
            is_straddle=False,
            d_overlap=s_overlap,  # B3: S would also fire
            squeeze_dur=0,  # Not relevant for D_mom8
        )

        # Min conviction filter (T3-Triple mode)
        if cfg.cw_enabled and score < cfg.min_conviction_score:
            logger.info(
                f"[{self._current_symbol}] D_mom8 BLOCKED: "
                f"conviction={score} < min={cfg.min_conviction_score} "
                f"TF={tf} {cond} VR={vr_val:.3f}"
            )
            return 0

        # Store metadata for scalper to query after on_bar
        # (which TF, SL/TP, max_hold for this specific signal)
        mom_val = float(mom[-1])
        self._last_signal_meta[self._current_symbol] = {
            "tf": tf,
            "sl": cfg.d_sl,
            "tp": cfg.d_tp,
            "max_hold": cfg.get_max_hold(tf) * _TF_TO_15M.get(tf, 1),
            "vr": float(vr[-1]) if not np.isnan(vr[-1]) else 0.0,
            "momentum": mom_val,
            "conviction_score": score,
            "conviction_cond": cond,
        }
        logger.info(
            f"[{self._current_symbol}] D_mom8 signal "
            f"TF={tf} VR={vr_val:.3f} {'LONG' if direction == 1 else 'SHORT'} "
            f"mom={mom_val:.6f} conviction={score}/4 {cond} "
            f"(threshold={cfg.d_vr_threshold})"
        )
        return direction

    # Track current symbol for signal metadata storage
    _current_symbol: str = ""

    # ══════════════════════════════════════════════════════════════════
    #  MA Cluster Squeeze Signal (R44d V17)
    # ══════════════════════════════════════════════════════════════════

    def _check_ma_cluster_signal(
        self,
        close_4h: np.ndarray,
        high_4h: np.ndarray,
        low_4h: np.ndarray,
        symbol: str,
    ) -> int:
        """Check MA cluster squeeze signal on 4h data.

        Uses 4 SMAs (20/50/100/200) on 4h close. When all SMAs are
        within cluster_pct and this persists for min_duration bars,
        a breakout with momentum direction triggers entry.

        One signal per cluster episode: after firing, no more signals
        until the cluster dissolves (spread exceeds cluster_pct).

        ZERO LOOKAHEAD:
          - SMAs computed on completed 4h bars only.
          - Signal fires on the LAST completed bar.
          - Momentum uses only past data.

        Args:
            close_4h: 4h close prices (completed bars).
            high_4h: 4h high prices.
            low_4h: 4h low prices.
            symbol: Trading symbol.

        Returns:
            +1 (long), -1 (short), or 0 (no signal).
        """
        cfg = self.config
        periods = cfg.ma_cluster_sma_periods  # [20, 50, 100, 200]

        # Need enough data for SMA(200) + some warmup
        min_needed = max(periods) + 1
        if len(close_4h) < min_needed:
            return 0

        # Compute SMAs
        sma20 = compute_sma(close_4h, periods[0])
        sma50 = compute_sma(close_4h, periods[1])
        sma100 = compute_sma(close_4h, periods[2])
        sma200 = compute_sma(close_4h, periods[3])

        # Compute momentum(8) on 4h close
        mom = compute_momentum(close_4h, cfg.momentum_period)

        # Initialize per-symbol cooldown if needed
        if symbol not in self._ma_cluster_cooldown:
            self._ma_cluster_cooldown[symbol] = np.zeros(1, dtype=np.int64)

        cooldown_arr = self._ma_cluster_cooldown[symbol]
        duration_out = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close_4h,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cfg.ma_cluster_pct,
            cfg.ma_cluster_min_duration,
            cooldown_arr,
            duration_out,
            max_duration=cfg.ma_cluster_max_duration,
            min_momentum=cfg.ma_cluster_min_momentum,
        )

        # ── Always compute metrics (needed for near-miss logging + monitor) ──
        idx = len(close_4h) - 1
        ma_vals = [float(sma20[idx]), float(sma50[idx]), float(sma100[idx]), float(sma200[idx])]
        ma_max = max(ma_vals)
        ma_min = min(ma_vals)
        spread = (ma_max - ma_min) / close_4h[idx] * 100 if close_4h[idx] > 0 else 0.0
        mom_val = float(mom[-1]) if len(mom) > 0 and not np.isnan(mom[-1]) else 0.0
        dur = int(duration_out[0])

        # ── Determine rejection reason (for monitor + near-miss log) ──
        if sig != 0:
            rejection = "fired"
        elif dur == 0:
            rejection = "no_cluster"
        elif dur < cfg.ma_cluster_min_duration:
            rejection = f"pre_duration({dur}<{cfg.ma_cluster_min_duration})"
        elif cfg.ma_cluster_max_duration > 0 and dur > cfg.ma_cluster_max_duration:
            rejection = f"zombie({dur}>{cfg.ma_cluster_max_duration})"
        elif cooldown_arr[0] == 1:
            rejection = "cooldown"
        elif ma_min <= close_4h[idx] <= ma_max:
            rejection = "no_breakout"
        elif abs(mom_val) < cfg.ma_cluster_min_momentum:
            rejection = f"momentum_fail(|{mom_val:.4f}|<{cfg.ma_cluster_min_momentum:.3f})"
        else:
            rejection = "unknown"

        # ── Store per-symbol state for MA Cluster monitor line in scalper ──
        self._ma_cluster_state[symbol] = {
            "duration": dur,
            "spread": spread,
            "momentum": mom_val,
            "rejection": rejection,
        }

        # ── Log near-miss: cluster forming but signal rejected ──
        # Only log when cluster has been active long enough to be interesting
        # (dur >= 1) and not just absent (no_cluster). Silent no_cluster is fine.
        if sig == 0 and dur >= 1:
            if rejection.startswith("zombie"):
                logger.info(
                    f"[{symbol}] MA_CLUSTER zombie(skip): "
                    f"dur={dur}>{cfg.ma_cluster_max_duration} "
                    f"spread={spread:.2f}% mom={mom_val:.6f}"
                )
            elif rejection == "cooldown":
                logger.info(f"[{symbol}] MA_CLUSTER cooldown(skip): dur={dur} spread={spread:.2f}% mom={mom_val:.6f}")
            elif rejection == "no_breakout":
                logger.info(
                    f"[{symbol}] MA_CLUSTER forming(no breakout yet): "
                    f"dur={dur} spread={spread:.2f}% "
                    f"price_inside_band mom={mom_val:.6f}"
                )
            elif rejection.startswith("momentum_fail"):
                logger.info(
                    f"[{symbol}] MA_CLUSTER momentum_fail(skip): "
                    f"dur={dur} spread={spread:.2f}% "
                    f"|mom|={abs(mom_val):.4f}<{cfg.ma_cluster_min_momentum:.3f}"
                )
            elif rejection.startswith("pre_duration"):
                # Only log if more than half the minimum duration (near-miss)
                if dur >= cfg.ma_cluster_min_duration / 2:
                    logger.info(
                        f"[{symbol}] MA_CLUSTER building: "
                        f"dur={dur}/{cfg.ma_cluster_min_duration} "
                        f"spread={spread:.2f}% mom={mom_val:.6f}"
                    )

        if sig != 0:
            self._last_signal_meta[symbol] = {
                "tf": "4h",
                "sl": cfg.ma_cluster_sl,
                "tp": cfg.ma_cluster_tp,
                "max_hold": cfg.ma_cluster_max_hold_4h * _TF_TO_15M["4h"],
                "vr": 0.0,  # Not a VR signal
                "momentum": mom_val,
                "strategy_type": "ma_cluster",
                "ma_cluster_spread": spread,
                "ma_cluster_duration": dur,
                "leverage_override": cfg.ma_cluster_leverage,
            }
            logger.info(
                f"[{symbol}] MA_CLUSTER signal "
                f"TF=4h {'LONG' if sig == 1 else 'SHORT'} "
                f"spread={spread:.2f}% duration={dur} "
                f"mom={mom_val:.6f} "
                f"SL={cfg.ma_cluster_sl * 100:.0f}% TP={cfg.ma_cluster_tp * 100:.0f}% "
                f"leverage={cfg.ma_cluster_leverage:.0f}x"
            )
        return sig

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
            "e2_sma": False,
            "c2_btc": False,
            "b3_confluence": False,
            "a3_quality": False,
        }

        # ── E2: SMA Proximity ──
        if cfg.e2_enabled and len(close) >= cfg.e2_sma_period:
            sma = np.mean(close[-cfg.e2_sma_period :])
            if sma > 0:
                proximity = abs(close[-1] - sma) / sma
                if proximity < cfg.e2_threshold:
                    conditions["e2_sma"] = True
                    score += 1

        # ── C2: BTC Neutral ──
        if cfg.c2_enabled and self._btc_close is not None:
            btc = self._btc_close
            mom_period = cfg.momentum_period  # Same 8-bar momentum
            if len(btc) > mom_period:
                btc_mom = btc[-1] / btc[-mom_period - 1] - 1.0
                if abs(btc_mom) < cfg.c2_btc_mom_threshold:
                    conditions["c2_btc"] = True
                    score += 1

        # ── B3: Confluence (both D_mom8 and S_d8 would fire) ──
        if cfg.b3_enabled:
            if d_overlap:
                # When S_d8 fires: d_overlap was checked in _check_tf_signal
                # When D_mom8 fires: we need to check if S_d8 would also fire
                conditions["b3_confluence"] = True
                score += 1

        # ── A3: Quality+ ──
        if cfg.a3_enabled:
            vr_val = float(vr[-1]) if not np.isnan(vr[-1]) else 999.0
            if is_straddle:
                # S_d8: extended squeeze duration
                if squeeze_dur >= cfg.a3_long_squeeze_dur:
                    conditions["a3_quality"] = True
                    score += 1
            else:
                # D_mom8: extra deep VR compression
                if vr_val < cfg.a3_deep_vr_threshold:
                    conditions["a3_quality"] = True
                    score += 1

        return score, conditions

    def _compute_tbr_zscore(self, symbol: str) -> float:
        """Compute TBR z-score at the PREVIOUS bar (no lookahead).

        Uses window of f5_tbr_z_window bars EXCLUDING the current bar,
        matching R46b backtest logic: z[entry] = zscore of [i-w, i-1].

        Args:
            symbol: Trading symbol.

        Returns:
            TBR z-score float. Returns 0.0 if data insufficient.
        """
        cfg = self.config
        w = cfg.f5_tbr_z_window
        tb = self._taker_buy_data.get(symbol)
        vol = self._volume_data.get(symbol)
        if tb is None or vol is None or len(tb) < w + 1:
            return 0.0  # Insufficient data → pass filter (don't block)

        # Exclude current bar [−1] to match backtest: window = [−w−1 : −1]
        tb_window = tb[-(w + 1) : -1]
        vol_window = vol[-(w + 1) : -1]
        current_tb = tb[-1]
        current_vol = vol[-1]

        safe_vol = np.where(vol_window > 0, vol_window, 1.0)
        tbr_window = tb_window / safe_vol

        mu = float(np.mean(tbr_window))
        sigma = float(np.std(tbr_window))
        if sigma < 1e-9:
            return 0.0

        current_tbr = current_tb / current_vol if current_vol > 0 else 0.5
        return (current_tbr - mu) / sigma

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

        Returns dict with '15m', '30m', '1h', '4h' keys.
        """
        return {
            "15m": self._last_vr_15m.get(symbol, 999.0),
            "30m": self._last_vr_30m.get(symbol, 999.0),
            "1h": self._last_vr_1h.get(symbol, 999.0),
            "4h": self._last_vr_4h.get(symbol, 999.0),
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
        if ":" in key:
            symbol, side = key.split(":", 1)
            return symbol, side
        return key, None

    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_atr_pct: float | None = None,
        current_high: float | None = None,
        current_low: float | None = None,
    ) -> str | None:
        """Check if position should be closed.

        Directional trades: SL → Trailing Stop → Hard TP (R43 validated).
        Straddle trades: unchanged SL/TP/max_hold exit.

        Trailing stop (directional only):
        - Before trail_activate: fixed SL only.
        - After best_gain >= trail_activate: trail at peak - distance.
        - Hard TP at d_tp (12%) as safety ceiling.

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

        pos["bars_held"] += 1
        entry = pos["entry_price"]
        sl_pct = pos.get("sl", self.config.d_sl)
        tp_pct = pos.get("tp", self.config.d_tp)
        max_hold = pos.get("max_hold", self.config.max_hold_15m)

        # Intra-bar extremes for SL/TP check (parity with backtest
        # which checks high/low within each bar)
        if current_high is not None and current_low is not None:
            if pos["signal"] == 1:  # LONG
                worst_gain = (current_low - entry) / entry
                best_gain = (current_high - entry) / entry
            else:  # SHORT
                worst_gain = (entry - current_high) / entry
                best_gain = (entry - current_low) / entry
        else:
            gain = (current_price - entry) / entry if pos["signal"] == 1 else (entry - current_price) / entry
            worst_gain = gain
            best_gain = gain

        # Track best PnL (MFE — Maximum Favorable Excursion)
        if best_gain > pos.get("best_pnl", 0.0):
            pos["best_pnl"] = best_gain

        # Track worst gain (MAE — Maximum Adverse Excursion, stored as negative)
        if worst_gain < pos.get("worst_pnl", 0.0):
            pos["worst_pnl"] = worst_gain

        # ── Select trail params based on position type ──
        is_straddle = pos.get("is_straddle", False)
        strategy_type = pos.get("strategy_type", "vr_directional")
        if is_straddle:
            # r53 Option A: per-leg trailing for S_d8 straddle
            trail_activate = self.config.s_trail_activate
            trail_distance = self.config.s_trail_distance
        elif strategy_type == "ma_cluster":
            trail_activate = self.config.ma_cluster_trail_activate
            trail_distance = self.config.ma_cluster_trail_distance
        else:
            trail_activate = self.config.d_trail_activate
            trail_distance = self.config.d_trail_distance

        # Update peak gain tracking
        peak_gain = pos.get("peak_gain", 0.0)
        if best_gain > peak_gain:
            peak_gain = best_gain
            pos["peak_gain"] = peak_gain

        trail_active = pos.get("trail_active", False)

        # Activate trailing stop when peak gain reaches threshold
        if not trail_active and peak_gain >= trail_activate:
            trail_active = True
            pos["trail_active"] = True
            logger.info(
                f"[{symbol}] Trail ACTIVATED: peak_gain={peak_gain * 100:+.3f}% "
                f"≥ activate={trail_activate * 100:.1f}% "
                f"trail_stop={peak_gain - trail_distance:.4f}"
            )

        # Check trailing stop (must come before SL check)
        if trail_active:
            trail_stop_gain = peak_gain - trail_distance
            if worst_gain <= trail_stop_gain:
                exit_pnl = max(trail_stop_gain, -sl_pct)
                has_server_trail = bool(pos.get("server_trail_order_id"))
                if self.config.trail_server_native and has_server_trail:
                    # Server handles trail — Binance fires TRAILING_STOP_MARKET.
                    # Log for monitoring but don't trigger client-side close.
                    logger.debug(
                        f"[{symbol}] Server trail should fire: "
                        f"worst={worst_gain * 100:+.3f}% "
                        f"≤ trail_stop={trail_stop_gain * 100:+.3f}% "
                        f"(peak={peak_gain * 100:+.3f}%)"
                    )
                else:
                    logger.info(
                        f"[{symbol}] TRAIL HIT: worst={worst_gain * 100:+.3f}% "
                        f"≤ trail_stop={trail_stop_gain * 100:+.3f}% "
                        f"(peak={peak_gain * 100:+.3f}% - dist={trail_distance * 100:.1f}%) "
                        f"bars={pos['bars_held']} exit_pnl≈{exit_pnl * 100:+.3f}%"
                    )
                    return "trailing_stop"

        # SL check (only when trailing is NOT active — same as backtest)
        # B3 guard: sl_pct=0 would exit at ANY loss — skip if not configured
        if not trail_active and sl_pct > 0 and worst_gain <= -sl_pct:
            logger.info(
                f"[{symbol}] SL hit: worst_gain={worst_gain * 100:+.3f}% "
                f"vs SL={sl_pct * 100:.1f}% bars={pos['bars_held']} "
                f"entry={entry:.6f} "
                f"{'low' if pos['signal'] == 1 else 'high'}="
                f"{current_low if pos['signal'] == 1 else current_high}"
            )
            return "stop_loss"

        # Hard TP check (safety ceiling at 12%)
        if best_gain >= tp_pct:
            logger.info(
                f"[{symbol}] Hard TP hit: best_gain={best_gain * 100:+.3f}% "
                f"≥ TP={tp_pct * 100:.1f}% bars={pos['bars_held']} "
                f"entry={entry:.6f}"
            )
            return "target"

        # Max hold (time-based exit)
        if pos["bars_held"] >= max_hold:
            logger.info(
                f"[{symbol}] Max hold reached: bars={pos['bars_held']}/{max_hold} "
                f"best_gain={best_gain * 100:+.3f}% worst_gain={worst_gain * 100:+.3f}%"
            )
            return "max_hold"

        return None

    def _exit_simple(
        self,
        pos: dict,
        symbol: str,
        worst_gain: float,
        best_gain: float,
        sl_pct: float,
        tp_pct: float,
        max_hold: int,
    ) -> str | None:
        """Simple SL/TP/max_hold exit for straddle legs."""
        if worst_gain <= -sl_pct:
            logger.info(
                f"[{symbol}] SL hit: worst_gain={worst_gain * 100:+.3f}% "
                f"vs SL={sl_pct * 100:.1f}% bars={pos['bars_held']}"
            )
            return "stop_loss"
        if best_gain >= tp_pct:
            logger.info(
                f"[{symbol}] TP hit: best_gain={best_gain * 100:+.3f}% "
                f"vs TP={tp_pct * 100:.1f}% bars={pos['bars_held']}"
            )
            return "target"
        if pos["bars_held"] >= max_hold:
            logger.info(f"[{symbol}] Max hold reached: bars={pos['bars_held']}/{max_hold}")
            return "max_hold"
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
                f"{symbol}:LONG" in self._positions or f"{symbol}:SHORT" in self._positions or symbol in self._positions
            )
        return symbol in self._positions

    def open_position(
        self,
        symbol: str,
        signal: int,
        entry_price: float,
        notional: float,
        leverage: float,
        move_size: float,
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
            raise ValueError(f"Invalid entry_price for {symbol}: {entry_price}")

        # Determine position key (compound in hedge mode)
        position_side = kwargs.get("position_side")
        if position_side:
            pos_key = f"{symbol}:{position_side}"
        else:
            pos_key = symbol

        # Get TF-specific parameters from signal metadata or kwargs
        meta = self._last_signal_meta.get(symbol, {})
        sl = kwargs.get("sl", meta.get("sl", self.config.d_sl))
        tp = kwargs.get("tp", meta.get("tp", self.config.d_tp))
        tf = kwargs.get("tf", meta.get("tf", "15m"))
        max_hold = kwargs.get(
            "max_hold",
            meta.get(
                "max_hold",
                self.config.get_max_hold(tf),
            ),
        )

        self._positions[pos_key] = {
            "symbol": symbol,  # Actual Binance symbol (for API calls)
            "signal": signal,
            "entry_price": entry_price,
            "notional": notional,
            "leverage": leverage,
            "move_size": move_size,
            "bars_held": 0,
            "best_pnl": 0.0,
            "worst_pnl": 0.0,
            # VolStraddle-specific
            "sl": sl,
            "tp": tp,
            "tf": tf,
            "max_hold": max_hold,
            "is_straddle": kwargs.get("is_straddle", False),
            "straddle_pair_id": kwargs.get("straddle_pair_id", None),
            "strategy_type": kwargs.get("strategy_type", meta.get("strategy_type", "vr_directional")),
            # Balance at open time (for trade record accuracy)
            "balance_before": kwargs.get("balance_before"),
            # Entry timestamp (for trade journal)
            "opened_at": datetime.now(timezone.utc).isoformat(),
            # R43 Trailing Stop (directional only, VR squeeze only)
            "trail_active": False,
            "peak_gain": 0.0,
            # Server bracket IDs
            "server_sl_order_id": server_sl_order_id,
            "server_tp_order_id": server_tp_order_id,
            "server_trail_order_id": kwargs.get("server_trail_order_id"),
            # Compatibility fields (used by scalper internals)
            "entry_stop_mult": entry_stop_mult or sl,
            "partial_closed": False,
            "confidence": 0,
            "highest_since_tp1": 0.0,
            "trail_floor": None,
            # Hedge mode
            "position_side": position_side,
        }

        strategy_label = kwargs.get("strategy_type", meta.get("strategy_type", "vr_directional"))
        extra = ""
        if strategy_label == "ma_cluster":
            spread = meta.get("ma_cluster_spread", 0)
            dur = meta.get("ma_cluster_duration", 0)
            extra = f" MA_spread={spread:.2f}% dur={dur}"
        else:
            extra = f" VR={meta.get('vr', 0):.3f}"

        logger.info(
            f"[{symbol}] Position opened: "
            f"{'LONG' if signal == 1 else 'SHORT'} "
            f"TF={tf} SL={sl * 100:.1f}% TP={tp * 100:.1f}% "
            f"max_hold={max_hold} ({strategy_label}){extra}"
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
        self,
        symbol: str,
        volume: np.ndarray,
        taker_buy: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
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
        self._taker_buy_data[symbol] = taker_buy  # F5 TBR gate (R46)

    def update_htf_data(
        self,
        symbol: str,
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
                "close": close_4h,
                "high": high_4h,
                "low": low_4h,
            }
        else:
            # Without high/low, we can't compute VR on raw 4h data.
            # Fall back to resampled 15m data in on_bar.
            self._htf_4h.pop(symbol, None)

    # ══════════════════════════════════════════════════════════════════
    #  Compatibility Methods (used by scalper/positions internals)
    # ══════════════════════════════════════════════════════════════════

    def get_effective_stop_mult(
        self,
        close_prices: np.ndarray,
        symbol: str | None = None,
        move_size: float | None = None,
    ) -> float:
        """Return SL fraction for the current signal.

        For vol straddle, stop is a fixed percentage (not move-based).
        Returns d_sl (0.01 = 1%) since move_size is always 1.0.

        Compatible with scalper's _open_position interface.
        """
        meta = self._last_signal_meta.get(symbol, {}) if symbol else {}
        return meta.get("sl", self.config.d_sl)

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
        self,
        symbol: str,
        close_prices: np.ndarray,
        btc_close: np.ndarray | None = None,
    ) -> dict:
        """Return diagnostic info for the current bar (for JSONL logging).

        Always includes VR values from get_vr_all_tf() regardless of signal,
        so diagnostics capture market state even during dry spells.
        """
        meta = self._last_signal_meta.get(symbol, {})
        vr_all = self.get_vr_all_tf(symbol)
        # Use per-TF VR (always available after on_bar), not just signal meta
        min_vr = min(vr_all.values())
        return {
            "symbol": symbol,
            "strategy": "vol_straddle",
            "vr": meta.get("vr", None),
            "vr_15m": vr_all["15m"] if vr_all["15m"] < 999.0 else None,
            "vr_1h": vr_all["1h"] if vr_all["1h"] < 999.0 else None,
            "vr_4h": vr_all["4h"] if vr_all["4h"] < 999.0 else None,
            "vr_min": min_vr if min_vr < 999.0 else None,
            "tf": meta.get("tf", None),
            "momentum": meta.get("momentum", None),
        }

    # ══════════════════════════════════════════════════════════════════
    #  State Serialization (for crash recovery)
    # ══════════════════════════════════════════════════════════════════

    def to_state_dict(self) -> dict:
        """Serialize strategy state for persistence."""
        positions = {}
        for sym, pos in self._positions.items():
            positions[sym] = {k: v for k, v in pos.items() if not k.startswith("_")}
        return {
            "strategy_name": "vol_straddle",
            "positions": positions,
        }

    def from_state_dict(self, state: dict) -> None:
        """Restore strategy state from persisted data."""
        self._positions = {}
        positions = state.get("positions", {})
        for sym, pos_data in positions.items():
            self._positions[sym] = pos_data
