"""Position lifecycle management (open, partial close, close, pyramid).

Mixin class providing all position entry/exit methods including PBT
(Partial-then-BreakEven-Trail) flow and pyramid logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantbox.core.protocol import ScalperProtocol
    from quantbox.core.types import ApiSymbol, PositionKey, SignalValue

import logging
import math
from datetime import datetime, timezone

import numpy as np

from quantbox.core.terminal_ui import format_entry, format_exit
from quantbox.strategy.rubberband import (
    calculate_position_size,
    compute_confidence_leverage,
)
from quantbox.utils.diagnostics import _diag_logger

logger = logging.getLogger("Scalper")


class PositionMixin:
    """Mixin: Position lifecycle (open, partial close, close, pyramid)."""

    def _get_pair_counterpart(self: "ScalperProtocol", pos_key: str) -> str | None:
        """Find the counterpart position key for a straddle pair.

        Scans _positions for the position sharing the same straddle_pair_id.
        Returns None if no counterpart found or position is not a straddle.
        """
        pair_id = self.strategy._positions.get(pos_key, {}).get("straddle_pair_id")
        if not pair_id:
            return None
        for key, pos in self.strategy._positions.items():
            if key != pos_key and pos.get("straddle_pair_id") == pair_id:
                return key
        return None

    def _open_position(
        self: "ScalperProtocol",
        symbol: "ApiSymbol",
        signal: "SignalValue",
        close_prices: np.ndarray,
        balance: float,
        straddle_pair_id: str | None = None,
    ) -> bool:
        """Open a new position.

        Args:
            symbol: Trading symbol.
            signal: +1 (long) or -1 (short).
            close_prices: Recent close prices.
            balance: Current balance.
            straddle_pair_id: UUID linking LONG+SHORT legs of same straddle pair.
                Both legs receive the same ID. None for directional positions.

        Returns:
            True if position was opened.
        """
        cfg = self.config
        entry_price = close_prices[-1]

        # Measure the move that triggered the signal
        lookback_price = close_prices[-cfg.lookback - 1]
        move_size = abs(entry_price / lookback_price - 1.0)

        # ── VolStraddle: simple flat sizing (R35 parity) ──
        if self.strategy_name == "vol_straddle":
            vs = self.vs_config
            # move_size = 1.0 convention: SL/TP are absolute fractions
            move_size = 1.0

            # Dynamic capital allocation
            flat_per_slot = balance / vs.max_concurrent  # hard floor/ceiling
            if vs.dynamic_capital_enabled:
                n_active = self.strategy.n_active
                per_slot_capital = balance / max(1, n_active + 1)
                per_slot_capital = min(per_slot_capital, balance * vs.dynamic_capital_max_fraction)
                # Hard cap: never exceed flat sizing regardless of n_active
                if per_slot_capital > flat_per_slot:
                    logger.warning(
                        f"[{symbol}] dynamic_capital capped: "
                        f"${per_slot_capital:.2f} → ${flat_per_slot:.2f} "
                        f"(balance/max_concurrent={vs.max_concurrent})"
                    )
                    per_slot_capital = flat_per_slot
            else:
                per_slot_capital = flat_per_slot

            # SoftTight DD Scaling (safety mechanism)
            if vs.soft_equity_scale:
                hwm = self._high_water_mark if self._high_water_mark > 0 else balance
                dd_pct = (hwm - balance) / hwm if hwm > 0 else 0.0
                eq_scale = 1.0
                if dd_pct >= vs.soft_eq_dd3:
                    eq_scale = vs.soft_eq_scale3
                    per_slot_capital *= eq_scale
                elif dd_pct >= vs.soft_eq_dd2:
                    eq_scale = vs.soft_eq_scale2
                    per_slot_capital *= eq_scale
                elif dd_pct >= vs.soft_eq_dd1:
                    eq_scale = vs.soft_eq_scale1
                    per_slot_capital *= eq_scale
                if eq_scale < 1.0:
                    logger.info(
                        f"[{symbol}] SoftTight DD Scaling: DD={dd_pct * 100:.1f}% "
                        f"→ {eq_scale:.2f}x → slot=${per_slot_capital:.2f}"
                    )

            leverage = vs.base_leverage
            notional = per_slot_capital * leverage

            # Get signal metadata (SL/TP/TF from on_bar)
            sig_meta = self.strategy.get_last_signal_meta(symbol)
            sl_pct = sig_meta.get("sl", vs.d_sl)
            tp_pct = sig_meta.get("tp", vs.d_tp)
            tf_label = sig_meta.get("tf", "15m")
            max_hold = sig_meta.get("max_hold", vs.get_max_hold(tf_label))
            is_straddle = sig_meta.get("straddle", False)
            strategy_type = sig_meta.get("strategy_type", "vr_squeeze")
            # Split vr_squeeze into vr_straddle / vr_directional for dashboard
            if strategy_type == "vr_squeeze":
                strategy_type = "vr_straddle" if is_straddle else "vr_directional"

            # ── R44d V17: MA Cluster uses different leverage ──
            # Risk parity: 12x × 2% SL ≈ 25x × 1% SL
            if "leverage_override" in sig_meta:
                raw_lev = sig_meta["leverage_override"]
                leverage = max(1.0, min(125.0, float(raw_lev)))
                if leverage != raw_lev:
                    logger.warning(
                        f"[{symbol}] leverage_override clamped: {raw_lev} → {leverage:.0f} (must be [1, 125])"
                    )
                notional = per_slot_capital * leverage
                logger.info(f"[{symbol}] {strategy_type}: leverage={leverage:.0f}x (override, SL={sl_pct * 100:.0f}%)")

            # ── R34 CW-Tier Conviction Sizing (VR only, not MA Cluster) ──
            conviction_score = sig_meta.get("conviction_score", 0)
            if vs.cw_enabled and vs.cw_multipliers and "vr" in strategy_type:
                idx = min(conviction_score, len(vs.cw_multipliers) - 1)
                cw_mult = vs.cw_multipliers[idx]
                notional *= cw_mult
                logger.info(
                    f"[{symbol}] CW-Tier: score={conviction_score}/4 → "
                    f"{cw_mult:.1f}x → notional=${notional:.2f} "
                    f"({sig_meta.get('conviction_cond', {})})"
                )

            # ── Cap notional at Binance's max allowable for this symbol/leverage ──
            # Prevents -2027 "Exceeded the maximum allowable position" errors.
            sym_info = self.get_symbol_info(symbol)
            max_notional = sym_info.get("max_notional")
            if max_notional and notional > max_notional:
                logger.warning(
                    f"[{symbol}] Notional capped: ${notional:.2f} → ${max_notional:.2f} "
                    f"(Binance max_notional limit at current leverage)"
                )
                notional = max_notional

            # ── Cumulative notional pre-check: account for existing exchange positions ──
            # Prevents -2027 when partial fill or untracked position already exists.
            # _cached_exchange_notionals is set by _reconcile_positions() each cycle.
            if max_notional:
                pos_side_key = (
                    f"{symbol}:{'LONG' if signal == 1 else 'SHORT'}"
                    if hasattr(self, "vs_config") and self.vs_config.hedge_mode
                    else symbol
                )
                cached = getattr(self, "_cached_exchange_notionals", {})
                existing_notional = cached.get(pos_side_key, 0.0)
                if existing_notional > 0:
                    remaining_budget = max_notional - existing_notional
                    if remaining_budget <= 0:
                        logger.warning(
                            f"[{symbol}] Skipping entry: already at max notional "
                            f"(exchange={existing_notional:.0f} ≥ max={max_notional:.0f})"
                        )
                        return False
                    if notional > remaining_budget:
                        logger.warning(
                            f"[{symbol}] Notional reduced for cumulative cap: "
                            f"${notional:.0f} → ${remaining_budget:.0f} "
                            f"(existing=${existing_notional:.0f} + new would exceed ${max_notional:.0f})"
                        )
                        notional = remaining_budget

        else:
            # Position sizing — v10.0: dynamic capital allocation
            if cfg.dynamic_capital_enabled:
                n_active = self.strategy.n_active
                per_slot_capital = balance / max(1, n_active + 1)
                # Safety: cap at cfg.dynamic_capital_max_fraction of balance to prevent over-concentration
                per_slot_capital = min(per_slot_capital, balance * cfg.dynamic_capital_max_fraction)
            else:
                per_slot_capital = balance / cfg.max_concurrent

            # ═══ v12.3 PARITY: SteepX Confidence Sizing ═══
            conf_level = self.strategy.get_last_confidence(symbol)
            if cfg.steep_cap_curve and conf_level > 0:
                if conf_level >= 5:
                    cap_mult = cfg.cap_mult_conf5
                elif conf_level == 4:
                    cap_mult = cfg.cap_mult_conf4
                elif conf_level == 3:
                    cap_mult = cfg.cap_mult_conf3
                elif conf_level == 2:
                    cap_mult = cfg.cap_mult_conf2
                else:
                    cap_mult = cfg.cap_mult_conf2  # conf=1
                per_slot_capital *= cap_mult
                logger.debug(f"[{symbol}] SteepX: conf={conf_level} → {cap_mult:.1f}x → slot=${per_slot_capital:.2f}")

            # ═══ v12.3 PARITY: SoftTight DD Scaling ═══
            if cfg.soft_equity_scale:
                hwm = self._high_water_mark if self._high_water_mark > 0 else balance
                dd_pct = (hwm - balance) / hwm if hwm > 0 else 0.0
                if dd_pct >= cfg.soft_eq_dd3:
                    eq_scale = cfg.soft_eq_scale3
                elif dd_pct >= cfg.soft_eq_dd2:
                    eq_scale = cfg.soft_eq_scale2
                elif dd_pct >= cfg.soft_eq_dd1:
                    eq_scale = cfg.soft_eq_scale1
                else:
                    eq_scale = 1.0
                if eq_scale < 1.0:
                    per_slot_capital *= eq_scale
                    logger.debug(
                        f"[{symbol}] SoftTight: DD={dd_pct * 100:.1f}% → {eq_scale:.2f}x → slot=${per_slot_capital:.2f}"
                    )

            # ═══ v12.3 PARITY: Vol-Adjusted Sizing ═══
            if cfg.vol_adjusted_size:
                cur_atr = self.strategy.get_last_atr_pct(symbol)
                if cur_atr > 0:
                    vol_factor = cfg.vol_adj_median_atr / cur_atr
                    vol_factor = max(cfg.vol_adj_min, min(cfg.vol_adj_max, vol_factor))
                    per_slot_capital *= vol_factor
                    logger.debug(
                        f"[{symbol}] VolAdj: ATR={cur_atr * 100:.2f}% "
                        f"→ {vol_factor:.2f}x → slot=${per_slot_capital:.2f}"
                    )
            notional, leverage = calculate_position_size(
                capital=per_slot_capital,
                risk_per_trade=cfg.risk_per_trade,
                entry_price=entry_price,
                move_size=move_size,
                stop_mult=cfg.stop_mult,
                max_leverage=cfg.max_leverage,
            )

            # Dynamic leverage: override with move-confidence scaling
            if cfg.dynamic_leverage_enabled:
                if cfg.rr_leverage_enabled:
                    # v4.3: R:R-gated leverage — only deploy when R:R justifies it
                    from quantbox.strategy.rubberband import compute_rr_gated_leverage

                    conf_lev = compute_rr_gated_leverage(
                        move_size=move_size,
                        threshold=cfg.threshold,
                        target_retrace=cfg.target_retrace,
                        stop_mult=cfg.stop_mult,
                        cost_per_side=cfg.cost_per_side,
                        base_leverage=cfg.base_leverage,
                        leverage_scale=cfg.leverage_scale,
                        max_leverage=cfg.max_leverage,
                        min_rr=cfg.min_rr_for_leverage,
                    )
                else:
                    conf_lev = compute_confidence_leverage(
                        move_size=move_size,
                        threshold=cfg.threshold,
                        base_leverage=cfg.base_leverage,
                        leverage_scale=cfg.leverage_scale,
                        max_leverage=cfg.max_leverage,
                    )
                notional = per_slot_capital * conf_lev
                leverage = conf_lev

            # ═══ v10.2 PARITY FIX: Apply MTF leverage multiplier ═══
            if cfg.mtf_confirmation_enabled and cfg.mtf_leverage_mult > 1.0:
                if self.strategy.was_mtf_confirmed(symbol):
                    leverage *= cfg.mtf_leverage_mult
                    notional = per_slot_capital * leverage
                    logger.debug(
                        f"[{symbol}] MTF leverage boost: {leverage / cfg.mtf_leverage_mult:.2f}x → {leverage:.2f}x"
                    )

        if notional < 10.0:  # Internal strategy minimum ($10); Binance exchange min is $5 (see below)
            logger.debug(f"[{symbol}] Notional too small: ${notional:.2f}")
            return False

        # ── ExposureGuard: block entry if total exposure exceeds cap ──
        try:
            _exp_guard = getattr(self, "_exposure_guard", None)
            if _exp_guard is not None:
                current_notional = sum(abs(float(p.get("notional", 0))) for p in self.strategy._positions.values())
                if not _exp_guard.can_open(current_notional, notional, balance):
                    logger.warning(f"[{symbol}] Exposure cap hit — skipping entry")
                    return False
        except Exception as e:
            logger.error(f"[{symbol}] ExposureGuard check error: {e}")

        # T2-1 runtime guard: entry_price=0 → ZeroDivisionError in quantity calc
        if entry_price <= 0:
            logger.error(f"[{symbol}] INVARIANT VIOLATION: entry_price={entry_price!r} (must be > 0) — aborting open")
            return False

        quantity = notional / entry_price
        quantity = self._round_qty(quantity, symbol)
        if quantity <= 0:
            return False

        # Check exchange min notional (Binance API; default $5 when not in symbol info)
        info = self.get_symbol_info(symbol)
        min_notional = info.get("min_notional", 5.0)
        if quantity * entry_price < min_notional:
            logger.debug(f"[{symbol}] Below min notional ${min_notional}")
            return False

        side = "BUY" if signal > 0 else "SELL"
        if self.strategy_name == "vol_straddle":
            lev_int = max(1, min(math.ceil(leverage), int(self.vs_config.max_leverage)))
        else:
            lev_int = max(1, min(math.ceil(leverage), int(cfg.max_leverage)))

        # Hedge Mode: derive positionSide from signal
        pos_side = None
        if self.strategy_name == "vol_straddle" and self.vs_config.hedge_mode:
            pos_side = "LONG" if signal > 0 else "SHORT"

        # ── Exchange-level dedup (prevents double-entry after rapid restarts) ──
        # Even if internal state is stale, the exchange will tell us the truth.
        if not self.paper_mode and self.strategy_name == "vol_straddle":
            try:
                ex_positions = self.client.futures_position_information(symbol=symbol)
                # hedge_mode: pos_side='LONG'/'SHORT'; one-way mode: pos_side=None → 'BOTH'
                wanted_side = pos_side or "BOTH"
                already_open = any(
                    p.get("positionSide") == wanted_side and abs(float(p.get("positionAmt", 0))) > 0
                    for p in ex_positions
                )
                if already_open:
                    logger.warning(
                        f"[{symbol}] Exchange dedup: {wanted_side} already open "
                        f"on exchange — skipping entry to prevent duplication"
                    )
                    # Sync internal state to reflect reality
                    if not self.strategy.has_position(symbol):
                        logger.warning(
                            f"[{symbol}] State out of sync — "
                            f"position exists on exchange but not in internal state. "
                            f"No SL/TP protection. "
                            f"Reverse ghost check in _reconcile_positions() will auto-close."
                        )
                    return False
            except Exception as exc:
                logger.warning(f"[{symbol}] Exchange dedup check failed ({exc}), proceeding anyway")

        # Capture balance before order fills (used in trade record for accurate tracking).
        balance_before = self.get_balance()

        result = self._place_order(
            symbol,
            side,
            quantity,
            leverage=lev_int,
            position_side=pos_side,
        )
        if result is None:
            return False

        fill_price = float(result["avgPrice"])
        fill_qty = float(result["executedQty"])

        # Compute effective stop (v12.6: ATR-based if enabled, else HTF-agree fixed)
        eff_stop = self.strategy.get_effective_stop_mult(
            close_prices,
            symbol=symbol,
            move_size=move_size,
        )

        notional = fill_qty * fill_price
        direction = "LONG" if signal > 0 else "SHORT"

        if self.strategy_name == "vol_straddle":
            # ── VolStraddle brackets: fixed SL/TP (no PBT) ──
            # move_size=1.0, so SL distance = sl_pct, TP distance = tp_pct
            logger.info(
                f"[{symbol}] OPENED {direction} qty={fill_qty:.6f} "
                f"@ ${fill_price:.4f} (notional=${notional:.2f}, lev={lev_int}x, "
                f"TF={tf_label}, SL={sl_pct * 100:.1f}%, TP={tp_pct * 100:.1f}%, "
                f"{strategy_type}"
                f"{f', conviction={conviction_score}/4' if strategy_type == 'vr_squeeze' else ''})"
            )

            sl_order_id = self._place_server_stop_loss(
                symbol=symbol,
                signal=signal,
                quantity=fill_qty,
                entry_price=fill_price,
                move_size=move_size,
                stop_mult=sl_pct,
                position_side=pos_side,
            )
            if sl_order_id == "WOULD_TRIGGER":
                logger.warning(f"[{symbol}] SL WOULD_TRIGGER at entry — bracket repair next cycle")
                sl_order_id = None

            # Server-side trailing stop via regular API (POST /fapi/v1/order)
            trail_order_id: str | None = None
            if self.vs_config.trail_server_native:
                cr, ap, cs = self._get_trail_params(strategy_type, signal, fill_price)
                trail_order_id = self._place_trail_regular(
                    symbol=symbol,
                    side=cs,
                    quantity=fill_qty,
                    callback_rate=cr,
                    activation_price=ap,
                    position_side=pos_side,
                )
                # No TP needed — trail handles profit-taking
                tp_order_id = None
                trail_status = "OK" if trail_order_id else "FAILED"
                logger.info(
                    f"[{symbol}] Server bracket: SL={'OK' if sl_order_id else 'FAILED'} "
                    f"(algoId={sl_order_id}) TRAIL={trail_status} "
                    f"(orderId={trail_order_id}) callbackRate={cr}%"
                )
            else:
                # Legacy: full-qty TP (no partial — matches backtest SL/TP/max_hold exit)
                tp_order_id = self._place_server_take_profit(
                    symbol=symbol,
                    signal=signal,
                    quantity=fill_qty,
                    entry_price=fill_price,
                    move_size=move_size,
                    target_retrace=tp_pct,
                    position_side=pos_side,
                )
                if tp_order_id == "WOULD_TRIGGER":
                    logger.warning(f"[{symbol}] TP WOULD_TRIGGER at entry — client-side will handle TP")
                    tp_order_id = None

                sl_status = "OK" if sl_order_id else "FAILED"
                tp_status = "OK" if tp_order_id else "FAILED"
                logger.info(
                    f"[{symbol}] Server bracket: SL={sl_status} "
                    f"(algoId={sl_order_id}) TP={tp_status} "
                    f"(algoId={tp_order_id})"
                )

            # Record in strategy with TF-specific metadata
            self.strategy.open_position(
                symbol=symbol,
                signal=signal,
                entry_price=fill_price,
                notional=fill_qty * fill_price,
                leverage=lev_int,
                move_size=move_size,
                entry_stop_mult=sl_pct,
                server_sl_order_id=sl_order_id,
                server_tp_order_id=tp_order_id,
                server_trail_order_id=trail_order_id,
                sl=sl_pct,
                tp=tp_pct,
                tf=tf_label,
                max_hold=max_hold,
                position_side=pos_side,
                is_straddle=is_straddle,
                straddle_pair_id=straddle_pair_id,
                strategy_type=strategy_type,
                balance_before=balance_before,
            )

            # Paper balance cost
            if self.paper_mode:
                cost = fill_qty * fill_price * self.vs_config.cost_per_side
                self._paper_balance -= cost

            # Print + log entry
            _side_str = "LONG" if signal == 1 else "SHORT"
            print(
                format_entry(
                    symbol=symbol,
                    side=_side_str,
                    price=fill_price,
                    leverage=lev_int,
                    size_usd=fill_qty * fill_price,
                )
            )

            # Compute TP/SL prices for notification
            if signal == 1:
                tp_price = fill_price * (1.0 + tp_pct)
                sl_price = fill_price * (1.0 - sl_pct)
            else:
                tp_price = fill_price * (1.0 - tp_pct)
                sl_price = fill_price * (1.0 + sl_pct)

            balance_after = self.get_balance()
            n_active = self.strategy.n_active
            _dir = "LONG ↑" if signal == 1 else "SHORT ↓"
            _sl_tag = "🟢 서버" if sl_order_id else "🟡 클라이언트"
            _tp_tag = "🟢 서버" if tp_order_id else "🟡 클라이언트"
            _pos_lines = []
            for _sym, _p in self.strategy.active_positions.items():
                _pd = "L" if _p["signal"] == 1 else "S"
                _pos_lines.append(f"{_sym} {_pd} ${_p['entry_price']:.4f}")
            _pos_summary = "\n".join(_pos_lines) if _pos_lines else "없음"

            self.discord.send(
                embed={
                    "title": f"{'🟢' if signal == 1 else '🔴'} {symbol} {_dir} | VS 진입",
                    "color": 0x00C853 if signal == 1 else 0xFF1744,
                    "fields": [
                        {
                            "name": "전략",
                            "value": f"VolStraddle {'S' if is_straddle else ('MA' if strategy_type == 'ma_cluster' else 'D')} ({tf_label})",
                            "inline": True,
                        },
                        {"name": "진입가", "value": f"${fill_price:.4f}", "inline": True},
                        {"name": "포지션", "value": f"${fill_qty * fill_price:,.2f}", "inline": True},
                        {"name": "레버리지", "value": f"{lev_int}x", "inline": True},
                        {"name": "🎯 TP", "value": f"${tp_price:.4f} ({tp_pct * 100:.1f}%)", "inline": True},
                        {"name": "🛑 SL", "value": f"${sl_price:.4f} ({sl_pct * 100:.1f}%)", "inline": True},
                        {
                            "name": "Exit",
                            "value": (
                                "SL/TP"
                                if (is_straddle or strategy_type == "ma_cluster")
                                else f"Trail({vs.d_trail_activate * 100:.0f}%/{vs.d_trail_distance * 100:.1f}%)"
                            ),
                            "inline": True,
                        },
                        {"name": "SL 방식", "value": _sl_tag, "inline": True},
                        {
                            "name": "확신도" if strategy_type == "vr_squeeze" else "전략타입",
                            "value": (f"{conviction_score}/4" if strategy_type == "vr_squeeze" else "MA Cluster"),
                            "inline": True,
                        },
                        {"name": "슬롯", "value": f"{n_active}/{self.vs_config.max_concurrent}", "inline": True},
                        {"name": "── 활성 포지션 ──", "value": f"```\n{_pos_summary}\n```", "inline": False},
                        {"name": "잔고", "value": f"${balance_after:,.2f}", "inline": True},
                    ],
                    "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
                }
            )
            return True

        else:
            # ── Rubber Band brackets ──
            logger.info(
                f"[{symbol}] OPENED {direction} qty={fill_qty:.6f} "
                f"@ ${fill_price:.4f} (notional=${notional:.2f}, lev={lev_int}x, "
                f"move={move_size * 100:.2f}%, eff_stop={eff_stop:.4f})"
            )

        # ═══ Place server-side bracket orders via Algo API ═══
        sl_order_id = self._place_server_stop_loss(
            symbol=symbol,
            signal=signal,
            quantity=fill_qty,
            entry_price=fill_price,
            move_size=move_size,
            stop_mult=eff_stop,
            position_side=pos_side,
        )
        # v12.3.1: Guard against WOULD_TRIGGER sentinel — let bracket repair handle it
        if sl_order_id == "WOULD_TRIGGER":
            logger.warning(f"[{symbol}] SL WOULD_TRIGGER at entry — will rely on bracket repair next cycle")
            sl_order_id = None

        # v12.5: In PBT mode, place server TP for PARTIAL qty (50%).
        # Previously skipped entirely because "server TP closes 100%".
        # But server TP qty is user-specified — placing 50% qty makes
        # server handle the partial close intra-bar, fixing the critical
        # SL-TP asymmetry: SL fires intra-bar (server) but TP only at
        # bar close (client) → systematic loss bias.
        # When server TP (50%) fires, _sync_server_orders detects it
        # and transitions to Phase 2 runner (BE stop + ATR trailing).
        if cfg.partial_exit_enabled:
            tp_qty = self._round_qty(
                fill_qty * cfg.partial_exit_fraction,
                symbol,
            )
            tp_order_id = self._place_server_take_profit(
                symbol=symbol,
                signal=signal,
                quantity=tp_qty,
                entry_price=fill_price,
                move_size=move_size,
                target_retrace=cfg.target_retrace,
                position_side=pos_side,
            )
            if tp_order_id == "WOULD_TRIGGER":
                logger.warning(f"[{symbol}] TP WOULD_TRIGGER at entry — client-side will handle TP1")
                tp_order_id = None
            _tp_is_partial = bool(tp_order_id)
            if tp_order_id:
                logger.info(
                    f"[{symbol}] PBT: server TP for {cfg.partial_exit_fraction * 100:.0f}% "
                    f"qty={tp_qty:.6f} (intra-bar partial close)"
                )
            else:
                logger.info(f"[{symbol}] PBT: server TP failed — client-side handles partial close at TP1")
        else:
            _tp_is_partial = False
            # Non-PBT: captures intra-bar TP hits that 15m polling would miss.
            tp_order_id = self._place_server_take_profit(
                symbol=symbol,
                signal=signal,
                quantity=fill_qty,
                entry_price=fill_price,
                move_size=move_size,
                target_retrace=cfg.target_retrace,
                position_side=pos_side,
            )
            # v12.3.1: Guard against WOULD_TRIGGER sentinel for TP too
            if tp_order_id == "WOULD_TRIGGER":
                logger.warning(f"[{symbol}] TP WOULD_TRIGGER at entry — will rely on client-side TP")
                tp_order_id = None

        sl_status = "OK" if sl_order_id else "FAILED"
        tp_status = "OK" if tp_order_id else "FAILED"
        logger.info(
            f"[{symbol}] Server bracket: SL={sl_status} (algoId={sl_order_id}) TP={tp_status} (algoId={tp_order_id})"
        )

        # Record in strategy
        self.strategy.open_position(
            symbol=symbol,
            signal=signal,
            entry_price=fill_price,
            notional=fill_qty * fill_price,
            leverage=lev_int,
            move_size=move_size,
            entry_stop_mult=eff_stop,
            server_sl_order_id=sl_order_id,
            server_tp_order_id=tp_order_id,
            confidence=self.strategy.get_last_confidence(symbol),
            balance_before=balance_before,
        )

        # v12.5: Set partial TP flag AFTER open_position() created the dict
        if cfg.partial_exit_enabled and _tp_is_partial:
            pos_ref = self.strategy._positions.get(symbol)
            if pos_ref:
                pos_ref["server_tp_is_partial"] = True

        # Paper balance adjustment
        if self.paper_mode:
            cost = fill_qty * fill_price * cfg.cost_per_side
            self._paper_balance -= cost

        _side_str = "LONG" if signal == 1 else "SHORT"
        print(
            format_entry(
                symbol=symbol,
                side=_side_str,
                price=fill_price,
                leverage=lev_int,
                size_usd=fill_qty * fill_price,
            )
        )

        # v12.0: Entry diagnostic to JSONL
        if hasattr(self.strategy, "diagnose_bar"):
            entry_diag = self.strategy.diagnose_bar(symbol, close_prices)
            _diag_logger.log_entry(
                self._cycle_count,
                datetime.now(timezone.utc).isoformat(),
                symbol,
                signal,
                entry_diag,
                {
                    "notional": round(fill_qty * fill_price, 2),
                    "leverage": lev_int,
                    "move_size": round(move_size, 6),
                    "fill_price": fill_price,
                    "tp_pct": round(cfg.target_retrace * 100, 2),
                    "sl_mult": round(eff_stop, 4),
                },
            )

        # Compute TP/SL prices for notification
        tp_distance = move_size * cfg.target_retrace
        sl_distance = move_size * eff_stop
        if signal == 1:
            tp_price = fill_price * (1.0 + tp_distance)
            sl_price = fill_price * (1.0 - sl_distance)
        else:
            tp_price = fill_price * (1.0 - tp_distance)
            sl_price = fill_price * (1.0 + sl_distance)

        balance_after = self.get_balance()
        n_active = self.strategy.n_active
        _dir = "LONG ↑" if signal == 1 else "SHORT ↓"
        # Server bracket status
        _sl_tag = "🟢 서버" if sl_order_id else "🟡 클라이언트"
        _tp_tag = "🟢 서버" if tp_order_id else "🟡 클라이언트"
        # Active positions summary
        _pos_lines = []
        for _sym, _p in self.strategy.active_positions.items():
            _pd = "L" if _p["signal"] == 1 else "S"
            _pos_lines.append(f"{_sym} {_pd} ${_p['entry_price']:.4f}")
        _pos_summary = "\n".join(_pos_lines) if _pos_lines else "없음"

        self.discord.send(
            embed={
                "title": f"{'🟢' if signal == 1 else '🔴'} {symbol} {_dir} | 진입",
                "color": 0x00C853 if signal == 1 else 0xFF1744,
                "fields": [
                    {"name": "진입가", "value": f"${fill_price:.4f}", "inline": True},
                    {"name": "포지션", "value": f"${fill_qty * fill_price:,.2f}", "inline": True},
                    {"name": "레버리지", "value": f"{lev_int}x", "inline": True},
                    {"name": "이동폭", "value": f"{move_size * 100:.2f}%", "inline": True},
                    {"name": "🎯 TP", "value": f"${tp_price:.4f}", "inline": True},
                    {"name": "🛑 SL", "value": f"${sl_price:.4f}", "inline": True},
                    {"name": "TP 방식", "value": _tp_tag, "inline": True},
                    {"name": "SL 방식", "value": _sl_tag, "inline": True},
                    {"name": "슬롯", "value": f"{n_active}/{cfg.max_concurrent}", "inline": True},
                    {"name": "── 활성 포지션 ──", "value": f"```\n{_pos_summary}\n```", "inline": False},
                    {"name": "잔고", "value": f"${balance_after:,.2f}", "inline": True},
                ],
                "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
            }
        )
        return True

    def _partial_close_position(
        self: "ScalperProtocol",
        symbol: "PositionKey",
        current_price: float,
    ) -> bool:
        """Partially close 50% of a position (v4.3 capital preservation).

        Args:
            symbol: Position key (may be compound: 'BTCUSDT:LONG').

        Closes half the position to lock in profit. The remaining half
        continues with a break-even stop. Does NOT remove the position
        from the strategy — it stays tracked with partial_closed=True.

        Returns True if partial close succeeded.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        # Extract Binance symbol from compound key
        api_symbol = pos.get("symbol", symbol.split(":")[0])

        # v12.5 Race guard: If server already handled the partial close
        # intra-bar (_handle_server_partial_tp_fill sets this flag),
        # skip client-side partial close to avoid double execution.
        if pos.get("_server_partial_tp_filled"):
            logger.info(f"[{symbol}] Partial close skipped — server TP already handled partial fill")
            return False

        # v11.1: Cancel old server SL (quantity will change)
        sl_oid = pos.get("server_sl_order_id")
        if sl_oid:
            self._cancel_server_stop_loss(api_symbol, sl_oid)

        # v11.2: Cancel old server TP (quantity will change)
        tp_oid = pos.get("server_tp_order_id")
        if tp_oid:
            self._cancel_server_take_profit(api_symbol, tp_oid)

        fraction = self.config.partial_exit_fraction
        full_quantity = pos["notional"] / pos["entry_price"]
        close_qty = self._round_qty(full_quantity * fraction, api_symbol)
        if close_qty <= 0:
            return False

        # Reverse the entry side to close
        side = "SELL" if pos["signal"] == 1 else "BUY"

        # Hedge Mode: use positionSide instead of reduceOnly
        pos_side = pos.get("position_side")
        result = self._place_order(
            api_symbol,
            side,
            close_qty,
            reduce_only=(pos_side is None),
            position_side=pos_side,
        )
        if result is None:
            # Failed: revert partial_closed flag & clear stale order IDs.
            # Without clearing IDs, _sync_server_orders sees the old
            # (now-cancelled) IDs as valid and skips bracket repair,
            # leaving the position with ZERO server-side protection.
            pos["partial_closed"] = False
            pos["server_sl_order_id"] = None
            pos["server_tp_order_id"] = None
            # v12.3.2: Re-place server SL/TP immediately to close
            # the 15-minute protection gap (don't wait for bracket repair).
            if not self.paper_mode:
                new_sl = self._place_server_stop_loss(
                    symbol=api_symbol,
                    signal=pos["signal"],
                    quantity=self._round_qty(full_quantity, api_symbol),
                    entry_price=pos["entry_price"],
                    move_size=pos["move_size"],
                    stop_mult=pos.get("entry_stop_mult", self.config.stop_mult),
                    position_side=pos_side,
                )
                if new_sl and new_sl != "WOULD_TRIGGER":
                    pos["server_sl_order_id"] = new_sl
                # v12.5: PBT mode re-places 50% server TP.
                # Non-PBT: full qty server TP for intra-bar capture.
                if self.config.partial_exit_enabled:
                    tp_qty = self._round_qty(
                        full_quantity * self.config.partial_exit_fraction,
                        api_symbol,
                    )
                    new_tp = self._place_server_take_profit(
                        symbol=api_symbol,
                        signal=pos["signal"],
                        quantity=tp_qty,
                        entry_price=pos["entry_price"],
                        move_size=pos["move_size"],
                        target_retrace=self.config.target_retrace,
                        position_side=pos_side,
                    )
                    if new_tp and new_tp != "WOULD_TRIGGER":
                        pos["server_tp_order_id"] = new_tp
                        pos["server_tp_is_partial"] = True
                else:
                    new_tp = self._place_server_take_profit(
                        symbol=api_symbol,
                        signal=pos["signal"],
                        quantity=self._round_qty(full_quantity, api_symbol),
                        entry_price=pos["entry_price"],
                        move_size=pos["move_size"],
                        target_retrace=self.config.target_retrace,
                        position_side=pos_side,
                    )
                    if new_tp and new_tp != "WOULD_TRIGGER":
                        pos["server_tp_order_id"] = new_tp
                logger.info(
                    f"[{symbol}] Partial close failed — re-placed server "
                    f"SL={pos.get('server_sl_order_id')} "
                    f"TP={pos.get('server_tp_order_id')}"
                )
            return False

        # v12.3.1: Handle NO_POSITION — server order closed it between
        # our cancel and our close attempt (race condition)
        if isinstance(result, dict) and result.get("status") == "NO_POSITION":
            logger.warning(f"[{symbol}] Partial close got NO_POSITION — server order already closed the position")
            # v12.3.2: Record trade INLINE — don't call _record_server_fill
            # because close_position() already removes the position, and
            # _record_server_fill calls close_position() again → returns
            # None → trade silently dropped.
            closed_pos = self.strategy.close_position(symbol)
            if closed_pos:
                fill_price = pos["entry_price"]  # fallback
                try:
                    recent_trades = self.client.futures_account_trades(
                        symbol=api_symbol,
                        limit=10,
                    )
                    if recent_trades:
                        close_side = "SELL" if closed_pos["signal"] == 1 else "BUY"
                        for t in reversed(recent_trades):
                            if t.get("side") == close_side and t.get("realizedPnl", "0") != "0":
                                fill_price = float(t["price"])
                                break
                except Exception:
                    pass

                if closed_pos["signal"] == 1:
                    pnl_pct = (fill_price - closed_pos["entry_price"]) / closed_pos["entry_price"]
                else:
                    pnl_pct = (closed_pos["entry_price"] - fill_price) / closed_pos["entry_price"]
                pnl_usd = pnl_pct * closed_pos["notional"]
                cost = closed_pos["notional"] * self.config.round_trip_cost
                net_pnl = pnl_usd - cost
                trade = {
                    "symbol": api_symbol,
                    "signal": closed_pos["signal"],
                    "entry_price": closed_pos["entry_price"],
                    "exit_price": fill_price,
                    "pnl_pct": pnl_pct * 100,
                    "pnl_usd": net_pnl,
                    "notional": closed_pos["notional"],
                    "bars_held": closed_pos["bars_held"],
                    "reason": "server_fill_race",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "strategy_type": closed_pos.get("strategy_type", "vr_squeeze"),
                    "mfe_pct": closed_pos.get("best_pnl", 0) * 100,
                    "mae_pct": closed_pos.get("worst_pnl", 0) * 100,
                    "entry_time": closed_pos.get("opened_at"),
                    "balance_before": closed_pos.get("balance_before") or self._get_cycle_balance(),
                    "balance_after": self._get_cycle_balance(),
                    "straddle_pair_id": closed_pos.get("straddle_pair_id"),
                }
                self._trade_history.append(trade)
                self._journal.record(trade)
                self._save_state()
                logger.info(f"[{symbol}] NO_POSITION race recorded: PnL={pnl_pct * 100:+.2f}% (${net_pnl:+.2f})")
            return False

        fill_price = float(result["avgPrice"])

        # PnL on the partial close
        if pos["signal"] == 1:
            pnl_pct = (fill_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - fill_price) / pos["entry_price"]

        # v12.3.2: Use entry-price notional for PnL (not exit-price).
        # exit-price notional (close_qty * fill_price) overstates winners
        # and understates losers by fill_price/entry_price ratio.
        partial_notional = close_qty * pos["entry_price"]
        pnl_usd = pnl_pct * partial_notional
        # v12.3.2: Charge round_trip_cost (entry+exit) for backtest parity.
        # Backtest _backtest_core_partial charges delta-based cost at each
        # transition: 1.0*cps at entry + 0.5*cps at partial = 1.5*cps.
        # For the partial portion, round_trip covers its share of entry+exit.
        cost = partial_notional * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        # v12.3.2: Update remaining notional using ENTRY price, not fill price.
        # Using fill_price causes notional drift: remaining_qty computed as
        # (notional - close_qty*fill_price) / entry_price ≠ actual remaining.
        # This leaves dust on exchange (~1-3% of position) that becomes
        # an orphaned unmanaged position.
        remaining_qty_exact = full_quantity - close_qty
        pos["notional"] = remaining_qty_exact * pos["entry_price"]

        # v11.1: Place new server SL for remaining quantity
        remaining_qty = self._round_qty(
            pos["notional"] / pos["entry_price"],
            api_symbol,
        )
        if remaining_qty > 0 and not self.paper_mode:
            new_sl_oid = self._place_server_stop_loss(
                symbol=api_symbol,
                signal=pos["signal"],
                quantity=remaining_qty,
                entry_price=pos["entry_price"],
                move_size=pos["move_size"],
                stop_mult=0.0001,  # Break-even stop (tiny distance)
                position_side=pos_side,
            )
            pos["server_sl_order_id"] = new_sl_oid

            # v12.3 PBT: No server TP for runner — exits via client-side
            # ATR trailing (trail_runner / time_runner / target).
            # Server TP would conflict with the trailing logic.
            if not self.config.breakeven_after_partial:
                # Legacy mode: place TP for remaining half
                new_tp_oid = self._place_server_take_profit(
                    symbol=api_symbol,
                    signal=pos["signal"],
                    quantity=remaining_qty,
                    entry_price=pos["entry_price"],
                    move_size=pos["move_size"],
                    target_retrace=self.config.target_retrace,
                    position_side=pos_side,
                )
                pos["server_tp_order_id"] = new_tp_oid
            else:
                pos["server_tp_order_id"] = None

        # Paper mode — charge EXIT-side cost only.
        # Entry cost for the full position was already charged at _open_position.
        # Using round_trip_cost here would double-count the entry fee for
        # the partial portion (overcharge by 0.5 × cost_per_side per trade).
        if self.paper_mode:
            paper_exit_cost = partial_notional * self.config.cost_per_side
            self._paper_balance += pnl_usd - paper_exit_cost

        # Trade record (partial)
        trade = {
            "symbol": symbol,
            "signal": pos["signal"],
            "entry_price": pos["entry_price"],
            "exit_price": fill_price,
            "pnl_pct": pnl_pct * 100,
            "pnl_usd": net_pnl,
            "notional": partial_notional,
            "bars_held": pos["bars_held"],
            "reason": "partial_close",
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy_type": pos.get("strategy_type", "vr_squeeze"),
            "mfe_pct": pos.get("best_pnl", 0) * 100,
            "mae_pct": pos.get("worst_pnl", 0) * 100,
            "entry_time": pos.get("opened_at"),
            "balance_before": pos.get("balance_before") or self._get_cycle_balance(),
            "balance_after": self._get_cycle_balance(),
            "straddle_pair_id": pos.get("straddle_pair_id"),
        }
        self._trade_history.append(trade)
        self._journal.record(trade)

        # v12.4.5: Save state immediately after partial close.
        # Without this, a crash before end-of-cycle _save_state() leaves
        # the state file with 100% notional + partial_closed=False.
        # On restart: bracket repair uses 100% qty, _record_server_fill
        # records PnL on stale 100% notional, and the partial close
        # trade is lost from history entirely.
        self._save_state()

        logger.info(
            f"[{symbol}] PARTIAL CLOSE ({fraction * 100:.0f}%): "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct * 100:+.2f}% (${net_pnl:+.2f}) "
            f"→ remaining ${pos['notional']:.2f}, BE stop active"
        )
        _dir = "LONG ↑" if pos["signal"] == 1 else "SHORT ↓"
        # Remaining bracket status
        _r_sl = "🟢 서버 (BE)" if pos.get("server_sl_order_id") else "🟡 클라이언트 (BE)"
        _r_tp = "🟢 서버" if pos.get("server_tp_order_id") else "🟡 클라이언트"
        self.discord.send(
            embed={
                "title": f"🔒 {symbol} {_dir} | 부분청산 {fraction * 100:.0f}%",
                "color": 0xFFC107,
                "description": "원금 확보 완료. 나머지는 손익분기점 스톱으로 안전 운영.",
                "fields": [
                    {"name": "진입가", "value": f"${pos['entry_price']:.4f}", "inline": True},
                    {"name": "체결가", "value": f"${fill_price:.4f}", "inline": True},
                    {"name": "보유", "value": f"{pos['bars_held']}바", "inline": True},
                    {"name": "수익률", "value": f"{pnl_pct * 100:+.2f}%", "inline": True},
                    {"name": "확보금액", "value": f"${net_pnl:+.2f}", "inline": True},
                    {"name": "남은포지션", "value": f"${pos['notional']:,.2f}", "inline": True},
                    {"name": "TP 방식", "value": _r_tp, "inline": True},
                    {"name": "SL 방식", "value": _r_sl, "inline": True},
                ],
                "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
            }
        )
        return True

    def _close_position(
        self: "ScalperProtocol",
        symbol: "PositionKey",
        reason: str,
        current_price: float,
    ) -> bool:
        """Close an existing position.

        Args:
            symbol: Position key (may be compound: 'BTCUSDT:LONG').

        Returns True if closed successfully.  Also returns True when the
        exchange reports NO_POSITION (-4118/-2022) because the position is
        already gone — we just clean up our internal state.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        # pair_close_requested guard: skip duplicate close requests.
        # Set by coordination when requesting close of a counterpart leg.
        # Only skip if we are NOT already in the middle of executing the close.
        if pos.get("pair_close_requested") and pos.get("_pair_close_executing"):
            logger.info(f"[{symbol}] pair_close_requested already executing — skipping duplicate close")
            return False

        # Extract Binance symbol from compound key
        api_symbol = pos.get("symbol", symbol.split(":")[0])

        quantity = pos["notional"] / pos["entry_price"]
        quantity = self._round_qty(quantity, api_symbol)
        if quantity <= 0:
            return False

        # v11.1: Cancel server-side stop-loss BEFORE closing
        sl_oid = pos.get("server_sl_order_id")
        if sl_oid:
            self._cancel_server_stop_loss(api_symbol, sl_oid)

        # v11.2: Cancel server-side take-profit BEFORE closing
        tp_oid = pos.get("server_tp_order_id")
        if tp_oid:
            self._cancel_server_take_profit(api_symbol, tp_oid)

        # r90: Cancel server-side trailing stop BEFORE closing (regular API)
        trail_oid = pos.get("server_trail_order_id")
        if trail_oid:
            self._cancel_trail_regular(api_symbol, str(trail_oid))

        # Reverse the entry side
        side = "SELL" if pos["signal"] == 1 else "BUY"

        # Hedge Mode: use positionSide instead of reduceOnly
        pos_side = pos.get("position_side")
        result = self._place_order(
            api_symbol,
            side,
            quantity,
            reduce_only=(pos_side is None),  # One-Way only
            position_side=pos_side,
        )

        # ── NO_POSITION: exchange says position already gone ──
        # This typically means a server-side TP/SL filled between cycles.
        # We MUST record the trade — otherwise trade count and PnL stay at 0.
        if isinstance(result, dict) and result.get("status") == "NO_POSITION":
            logger.warning(
                f"[{symbol}] Position already closed on exchange — recording trade & cleaning up (reason={reason})"
            )
            # Pop the position from internal tracking
            closed_pos = self.strategy.close_position(symbol)
            if closed_pos:
                # Try to get actual fill price from recent exchange trades
                fill_price = current_price  # fallback
                try:
                    recent_trades = self.client.futures_account_trades(
                        symbol=api_symbol,
                        limit=10,
                    )
                    if recent_trades:
                        # Find the most recent CLOSE trade (opposite side)
                        close_side = "SELL" if closed_pos["signal"] == 1 else "BUY"
                        for t in reversed(recent_trades):
                            if t.get("side") == close_side and t.get("realizedPnl", "0") != "0":
                                fill_price = float(t["price"])
                                logger.info(
                                    f"[{symbol}] NO_POSITION fill recovered: "
                                    f"price=${fill_price:.4f} "
                                    f"realizedPnl={t.get('realizedPnl')}"
                                )
                                break
                except Exception as e:
                    logger.warning(
                        f"[{symbol}] Could not fetch fill info: {e}. "
                        f"Using current_price=${current_price:.4f} as fallback."
                    )

                # PnL calculation
                if closed_pos["signal"] == 1:
                    pnl_pct = (fill_price - closed_pos["entry_price"]) / closed_pos["entry_price"]
                else:
                    pnl_pct = (closed_pos["entry_price"] - fill_price) / closed_pos["entry_price"]

                pnl_usd = pnl_pct * closed_pos["notional"]
                # v12.3.2: Always charge round_trip_cost for parity.
                cost = closed_pos["notional"] * self.config.round_trip_cost
                net_pnl = pnl_usd - cost

                # Trade record
                trade = {
                    "symbol": api_symbol,
                    "signal": closed_pos["signal"],
                    "entry_price": closed_pos["entry_price"],
                    "exit_price": fill_price,
                    "pnl_pct": pnl_pct * 100,
                    "pnl_usd": net_pnl,
                    "notional": closed_pos["notional"],
                    "bars_held": closed_pos["bars_held"],
                    "reason": f"{reason}_no_position",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "strategy_type": closed_pos.get("strategy_type", "vr_squeeze"),
                    "mfe_pct": closed_pos.get("best_pnl", 0) * 100,
                    "mae_pct": closed_pos.get("worst_pnl", 0) * 100,
                    "entry_time": closed_pos.get("opened_at"),
                    "balance_before": closed_pos.get("balance_before") or self._get_cycle_balance(),
                    "balance_after": self._get_cycle_balance(),
                    "straddle_pair_id": closed_pos.get("straddle_pair_id"),
                }
                self._trade_history.append(trade)
                self._journal.record(trade)

                # v12.0: Exit diagnostic to JSONL
                _diag_logger.log_exit(
                    self._cycle_count,
                    datetime.now(timezone.utc).isoformat(),
                    trade,
                )

                _exit_side = "LONG" if closed_pos["signal"] == 1 else "SHORT"
                print(
                    format_exit(
                        symbol=symbol,
                        side=_exit_side,
                        entry_price=closed_pos["entry_price"],
                        exit_price=fill_price,
                        pnl_usd=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        reason=f"{reason} (no_position)",
                        bars_held=closed_pos["bars_held"],
                    )
                )
                logger.info(
                    f"[{symbol}] NO_POSITION trade recorded: {_exit_side} "
                    f"PnL={pnl_pct * 100:+.2f}% (${net_pnl:+.2f}) "
                    f"MFE={closed_pos.get('best_pnl', 0) * 100:+.2f}% "
                    f"MAE={closed_pos.get('worst_pnl', 0) * 100:+.2f}% "
                    f"exit=${fill_price:.4f} held={closed_pos['bars_held']} bars"
                )

                # Session stats
                cum_pnl = sum(t["pnl_usd"] for t in self._trade_history)
                n_trades = len(self._trade_history)
                wins_total = sum(1 for t in self._trade_history if t["pnl_usd"] > 0)
                wr = wins_total / n_trades * 100 if n_trades > 0 else 0
                balance_after = self.get_balance()
                hold_time = closed_pos["bars_held"] * 15  # minutes

                _dir_kr = "LONG ↑" if closed_pos["signal"] == 1 else "SHORT ↓"
                _reason_kr = reason.upper().replace("_", " ")
                n_remaining = self.strategy.n_active

                self.discord.send(
                    embed={
                        "title": f"{'✅' if net_pnl > 0 else '❌'} {symbol} {_dir_kr} | {_reason_kr} (서버체결)",
                        "color": 0x00C853 if net_pnl > 0 else 0xFF1744,
                        "fields": [
                            {"name": "진입가", "value": f"${closed_pos['entry_price']:.4f}", "inline": True},
                            {"name": "청산가", "value": f"${fill_price:.4f}", "inline": True},
                            {
                                "name": "보유시간",
                                "value": f"{hold_time}분 ({closed_pos['bars_held']}바)",
                                "inline": True,
                            },
                            {"name": "수익률", "value": f"{pnl_pct * 100:+.2f}%", "inline": True},
                            {"name": "순수익", "value": f"${net_pnl:+.2f}", "inline": True},
                            {"name": "체결방식", "value": "🟢 거래소(서버)", "inline": True},
                            {
                                "name": "── 세션 현황 ──",
                                "value": (
                                    f"누적 PnL: ${cum_pnl:+,.2f} | "
                                    f"{n_trades}건 | 승률 {wr:.0f}%\n"
                                    f"남은 포지션: {n_remaining}개 | "
                                    f"잔고: ${balance_after:,.2f}"
                                ),
                                "inline": False,
                            },
                        ],
                        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
                    }
                )
            self._save_state()
            return True

        if result is None:
            # Failed to close — position stays in state for retry
            # SL/TP already cancelled, so re-place them.
            # v12.3 PBT: Runner positions need BE stop, not original stop/TP.
            is_runner = pos.get("partial_closed") and self.config.breakeven_after_partial
            if is_runner:
                # Runner: use breakeven stop, no server TP
                new_sl_oid = self._place_server_stop_loss(
                    symbol=api_symbol,
                    signal=pos["signal"],
                    quantity=quantity,
                    entry_price=pos["entry_price"],
                    move_size=pos["move_size"],
                    stop_mult=0.0001,  # BE stop
                    position_side=pos.get("position_side"),
                )
                new_tp_oid = None
                # v12.4.4: Reset ratchet state so _ratchet_runner_sl
                # re-ratchets from breakeven on the next cycle.
                # Without this, stale _server_sl_pct from the previous
                # ratcheted level prevents re-ratcheting (<0.5% improvement),
                # leaving the server SL stuck at breakeven.
                pos["_server_sl_pct"] = 0.0
            else:
                # Phase 1: use original stop
                new_sl_oid = self._place_server_stop_loss(
                    symbol=api_symbol,
                    signal=pos["signal"],
                    quantity=quantity,
                    entry_price=pos["entry_price"],
                    move_size=pos["move_size"],
                    stop_mult=pos.get("entry_stop_mult", self.config.stop_mult),
                    position_side=pos.get("position_side"),
                )
                # v12.4.3: In PBT mode, do NOT re-place server TP.
                # Same bug class as v12.4.1 CRITICAL #1: server TP fires
                # for 100% quantity, killing the PBT partial close flow.
                # Scenario: Phase 1 trailing_stop close fails → server TP
                # re-placed → price recovers to TP → server TP closes 100%
                # → PBT runner never created.
                # Non-PBT: restore TP for intra-bar capture.
                if self.config.partial_exit_enabled:
                    new_tp_oid = None
                else:
                    # v14.0: Use per-position TP (not RubberBand default).
                    retry_tp = pos.get("tp", self.config.target_retrace)
                    new_tp_oid = self._place_server_take_profit(
                        symbol=api_symbol,
                        signal=pos["signal"],
                        quantity=quantity,
                        entry_price=pos["entry_price"],
                        move_size=pos["move_size"],
                        target_retrace=retry_tp,
                        position_side=pos.get("position_side"),
                    )
            # Update existing position's server order IDs
            # Guard against sentinel values being stored as order IDs
            if new_sl_oid and new_sl_oid != "WOULD_TRIGGER":
                pos["server_sl_order_id"] = new_sl_oid
            else:
                pos["server_sl_order_id"] = None
            # v12.3.1: Guard TP sentinel too
            if new_tp_oid and new_tp_oid != "WOULD_TRIGGER":
                pos["server_tp_order_id"] = new_tp_oid
            else:
                pos["server_tp_order_id"] = None
            self._save_state()

            # v12.3.1: SL WOULD_TRIGGER = price already past stop level.
            # Leaving position open with no SL for 15min is unacceptable.
            # Guard against infinite recursion if this is already a force close.
            if new_sl_oid == "WOULD_TRIGGER" and not reason.startswith("force_"):
                logger.error(
                    f"[{symbol}] SL WOULD_TRIGGER during re-placement — price past stop level, forcing immediate close"
                )
                return self._close_position(
                    symbol,
                    "force_sl_would_trigger",
                    current_price,
                )

            return False

        # ── Successful close — NOW remove from state ──
        pos = self.strategy.close_position(symbol)
        if pos is None:
            return False

        fill_price = float(result["avgPrice"])

        # PnL calculation
        if pos["signal"] == 1:
            pnl_pct = (fill_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - fill_price) / pos["entry_price"]

        pnl_usd = pnl_pct * pos["notional"]
        # v12.3.2: Always charge round_trip_cost for backtest parity.
        # Each portion (partial and runner) accounts for its share
        # of entry + exit costs. See _backtest_core_partial delta model.
        cost = pos["notional"] * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        # Paper balance
        if self.paper_mode:
            self._paper_balance += pnl_usd
            close_cost = quantity * fill_price * self.config.cost_per_side
            self._paper_balance -= close_cost

        # Trade record
        trade = {
            "symbol": api_symbol,
            "signal": pos["signal"],
            "entry_price": pos["entry_price"],
            "exit_price": fill_price,
            "pnl_pct": pnl_pct * 100,
            "pnl_usd": net_pnl,
            "notional": pos["notional"],
            "bars_held": pos["bars_held"],
            "reason": reason,
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy_type": pos.get("strategy_type", "vr_squeeze"),
            "mfe_pct": pos.get("best_pnl", 0) * 100,
            "mae_pct": pos.get("worst_pnl", 0) * 100,
            "entry_time": pos.get("opened_at"),
            "balance_before": pos.get("balance_before") or self._get_cycle_balance(),
            "balance_after": self._get_cycle_balance(),
            "straddle_pair_id": pos.get("straddle_pair_id"),
        }
        self._trade_history.append(trade)
        self._journal.record(trade)

        # v12.0: Exit diagnostic to JSONL
        _diag_logger.log_exit(self._cycle_count, datetime.now(timezone.utc).isoformat(), trade)

        _exit_side = "LONG" if pos["signal"] == 1 else "SHORT"
        print(
            format_exit(
                symbol=symbol,
                side=_exit_side,
                entry_price=pos["entry_price"],
                exit_price=fill_price,
                pnl_usd=net_pnl,
                pnl_pct=pnl_pct * 100,
                reason=reason,
                bars_held=pos["bars_held"],
            )
        )
        logger.info(
            f"[{symbol}] CLOSED ({reason}): {_exit_side} "
            f"PnL={pnl_pct * 100:+.2f}% (${net_pnl:+.2f}) "
            f"MFE={pos.get('best_pnl', 0) * 100:+.2f}% "
            f"MAE={pos.get('worst_pnl', 0) * 100:+.2f}% "
            f"held={pos['bars_held']} bars"
        )
        # Session stats for notification
        cum_pnl = sum(t["pnl_usd"] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins_total = sum(1 for t in self._trade_history if t["pnl_usd"] > 0)
        wr = wins_total / n_trades * 100 if n_trades > 0 else 0
        balance_after = self.get_balance()
        hold_time = pos["bars_held"] * 15  # minutes

        _dir_kr = "LONG ↑" if pos["signal"] == 1 else "SHORT ↓"
        _reason_kr = reason.upper().replace("_", " ")
        n_remaining = self.strategy.n_active
        # Remaining positions summary
        _pos_lines = []
        for _sym, _p in self.strategy.active_positions.items():
            _pd = "L" if _p["signal"] == 1 else "S"
            _pos_lines.append(f"{_sym} {_pd} ${_p['entry_price']:.4f}")
        _pos_summary = "\n".join(_pos_lines) if _pos_lines else "없음"

        self.discord.send(
            embed={
                "title": f"{'✅' if net_pnl > 0 else '❌'} {symbol} {_dir_kr} | {_reason_kr}",
                "color": 0x00C853 if net_pnl > 0 else 0xFF1744,
                "fields": [
                    {"name": "진입가", "value": f"${pos['entry_price']:.4f}", "inline": True},
                    {"name": "청산가", "value": f"${fill_price:.4f}", "inline": True},
                    {"name": "보유시간", "value": f"{hold_time}분 ({pos['bars_held']}바)", "inline": True},
                    {"name": "수익률", "value": f"{pnl_pct * 100:+.2f}%", "inline": True},
                    {"name": "순수익", "value": f"${net_pnl:+.2f}", "inline": True},
                    {"name": "체결방식", "value": "🔵 클라이언트", "inline": True},
                    {
                        "name": "── 세션 현황 ──",
                        "value": (
                            f"누적 PnL: ${cum_pnl:+,.2f} | {n_trades}건 | 승률 {wr:.0f}%\n잔고: ${balance_after:,.2f}"
                        ),
                        "inline": False,
                    },
                    {"name": "── 남은 포지션 ──", "value": f"```\n{_pos_summary}\n```", "inline": False},
                ],
                "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
            }
        )
        return True

    # ==================== Main Cycle ====================
