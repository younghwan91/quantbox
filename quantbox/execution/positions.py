"""Position lifecycle management (open, partial close, close, pyramid).

Mixin class providing all position entry/exit methods including PBT
(Partial-then-BreakEven-Trail) flow, Elliott add-to-position, and
pyramid logic.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import numpy as np

from quantbox.core.terminal_ui import format_entry, format_exit
from quantbox.strategy.elliott import ElliottConfig
from quantbox.strategy.rubberband import (
    calculate_position_size, compute_confidence_leverage,
)
from quantbox.utils.diagnostics import _diag_logger

logger = logging.getLogger('Scalper')


class PositionMixin:
    """Mixin: Position lifecycle (open, partial close, close, pyramid)."""

    def _open_position(
        self, symbol: str, signal: int, close_prices: np.ndarray,
        balance: float,
    ) -> bool:
        """Open a new position.

        Args:
            symbol: Trading symbol.
            signal: +1 (long) or -1 (short).
            close_prices: Recent close prices.
            balance: Current balance.

        Returns:
            True if position was opened.
        """
        cfg = self.config
        entry_price = close_prices[-1]

        # Measure the move that triggered the signal
        lookback_price = close_prices[-cfg.lookback - 1]
        move_size = abs(entry_price / lookback_price - 1.0)

        # ── Elliott: 2% exploratory sizing (winning_strategy §4.2) ──
        if self.strategy_name == 'elliott':
            ecfg = getattr(self, 'elliott_config', ElliottConfig())
            margin = balance * ecfg.exploratory_pct
            leverage = ecfg.nominal_leverage
            notional = margin * leverage
            if notional < 10.0:
                logger.debug(f"[{symbol}] Elliott notional too small: ${notional:.2f}")
                return False
        else:
            # Position sizing — v10.0: dynamic capital allocation
            if cfg.dynamic_capital_enabled:
                n_active = self.strategy.n_active
                per_slot_capital = balance / max(1, n_active + 1)
                # Safety: cap at 50% of balance to prevent over-concentration
                per_slot_capital = min(per_slot_capital, balance * 0.5)
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
                logger.debug(
                    f"[{symbol}] SteepX: conf={conf_level} → {cap_mult:.1f}x "
                    f"→ slot=${per_slot_capital:.2f}"
                )

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
                        f"[{symbol}] SoftTight: DD={dd_pct*100:.1f}% "
                        f"→ {eq_scale:.2f}x → slot=${per_slot_capital:.2f}"
                    )

            # ═══ v12.3 PARITY: Vol-Adjusted Sizing ═══
            if cfg.vol_adjusted_size:
                cur_atr = self.strategy.get_last_atr_pct(symbol)
                if cur_atr > 0:
                    vol_factor = cfg.vol_adj_median_atr / cur_atr
                    vol_factor = max(cfg.vol_adj_min, min(cfg.vol_adj_max, vol_factor))
                    per_slot_capital *= vol_factor
                    logger.debug(
                        f"[{symbol}] VolAdj: ATR={cur_atr*100:.2f}% "
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
                        f"[{symbol}] MTF leverage boost: "
                        f"{leverage/cfg.mtf_leverage_mult:.2f}x → {leverage:.2f}x"
                    )

        if notional < 10.0:  # Min notional $10
            logger.debug(f"[{symbol}] Notional too small: ${notional:.2f}")
            return False

        quantity = notional / entry_price
        quantity = self._round_qty(quantity, symbol)
        if quantity <= 0:
            return False

        # Check min notional
        info = self.get_symbol_info(symbol)
        min_notional = info.get('min_notional', 5.0)
        if quantity * entry_price < min_notional:
            logger.debug(f"[{symbol}] Below min notional ${min_notional}")
            return False

        side = 'BUY' if signal > 0 else 'SELL'
        lev_int = max(1, min(math.ceil(leverage), int(cfg.max_leverage)))

        result = self._place_order(symbol, side, quantity, leverage=lev_int)
        if result is None:
            return False

        fill_price = float(result['avgPrice'])
        fill_qty = float(result['executedQty'])

        # Compute effective stop (v12.6: ATR-based if enabled, else HTF-agree fixed)
        eff_stop = self.strategy.get_effective_stop_mult(
            close_prices, symbol=symbol, move_size=move_size,
        )

        notional = fill_qty * fill_price
        direction = 'LONG' if signal > 0 else 'SHORT'
        logger.info(
            f"[{symbol}] OPENED {direction} qty={fill_qty:.6f} "
            f"@ ${fill_price:.4f} (notional=${notional:.2f}, lev={lev_int}x, "
            f"move={move_size*100:.2f}%, eff_stop={eff_stop:.4f})"
        )

        # ═══ Place server-side bracket orders via Algo API ═══
        sl_order_id = self._place_server_stop_loss(
            symbol=symbol,
            signal=signal,
            quantity=fill_qty,
            entry_price=fill_price,
            move_size=move_size,
            stop_mult=eff_stop,
        )
        # v12.3.1: Guard against WOULD_TRIGGER sentinel — let bracket repair handle it
        if sl_order_id == 'WOULD_TRIGGER':
            logger.warning(
                f"[{symbol}] SL WOULD_TRIGGER at entry — "
                f"will rely on bracket repair next cycle"
            )
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
                fill_qty * cfg.partial_exit_fraction, symbol,
            )
            tp_order_id = self._place_server_take_profit(
                symbol=symbol,
                signal=signal,
                quantity=tp_qty,
                entry_price=fill_price,
                move_size=move_size,
                target_retrace=cfg.target_retrace,
            )
            if tp_order_id == 'WOULD_TRIGGER':
                logger.warning(
                    f"[{symbol}] TP WOULD_TRIGGER at entry — "
                    f"client-side will handle TP1"
                )
                tp_order_id = None
            _tp_is_partial = bool(tp_order_id)
            if tp_order_id:
                logger.info(
                    f"[{symbol}] PBT: server TP for {cfg.partial_exit_fraction*100:.0f}% "
                    f"qty={tp_qty:.6f} (intra-bar partial close)"
                )
            else:
                logger.info(
                    f"[{symbol}] PBT: server TP failed — "
                    f"client-side handles partial close at TP1"
                )
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
            )
            # v12.3.1: Guard against WOULD_TRIGGER sentinel for TP too
            if tp_order_id == 'WOULD_TRIGGER':
                logger.warning(
                    f"[{symbol}] TP WOULD_TRIGGER at entry — "
                    f"will rely on client-side TP"
                )
                tp_order_id = None

        sl_status = "OK" if sl_order_id else "FAILED"
        tp_status = "OK" if tp_order_id else "FAILED"
        logger.info(
            f"[{symbol}] Server bracket: SL={sl_status} "
            f"(algoId={sl_order_id}) TP={tp_status} "
            f"(algoId={tp_order_id})"
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
        )

        # v12.5: Set partial TP flag AFTER open_position() created the dict
        if cfg.partial_exit_enabled and _tp_is_partial:
            pos_ref = self.strategy._positions.get(symbol)
            if pos_ref:
                pos_ref['server_tp_is_partial'] = True

        # Paper balance adjustment
        if self.paper_mode:
            cost = fill_qty * fill_price * cfg.cost_per_side
            self._paper_balance -= cost

        _side_str = 'LONG' if signal == 1 else 'SHORT'
        print(format_entry(
            symbol=symbol,
            side=_side_str,
            price=fill_price,
            leverage=lev_int,
            size_usd=fill_qty * fill_price,
        ))
        logger.info(
            f"[{symbol}] OPENED {_side_str} "
            f"qty={fill_qty:.6f} @ ${fill_price:.4f} "
            f"(notional=${fill_qty * fill_price:.2f}, "
            f"lev={lev_int}x, move={move_size*100:.2f}%)"
        )

        # v12.0: Entry diagnostic to JSONL
        if hasattr(self.strategy, 'diagnose_bar'):
            entry_diag = self.strategy.diagnose_bar(symbol, close_prices)
            _diag_logger.log_entry(
                self._cycle_count,
                datetime.now(timezone.utc).isoformat(),
                symbol, signal, entry_diag,
                {'notional': round(fill_qty * fill_price, 2),
                 'leverage': lev_int, 'move_size': round(move_size, 6),
                 'fill_price': fill_price, 'tp_pct': round(cfg.target_retrace * 100, 2),
                 'sl_mult': round(eff_stop, 4)},
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
        _dir = 'LONG ↑' if signal == 1 else 'SHORT ↓'
        # Server bracket status
        _sl_tag = '🟢 서버' if sl_order_id else '🟡 클라이언트'
        _tp_tag = '🟢 서버' if tp_order_id else '🟡 클라이언트'
        # Active positions summary
        _pos_lines = []
        for _sym, _p in self.strategy.active_positions.items():
            _pd = 'L' if _p['signal'] == 1 else 'S'
            _pos_lines.append(f"{_sym} {_pd} ${_p['entry_price']:.4f}")
        _pos_summary = '\n'.join(_pos_lines) if _pos_lines else '없음'

        self.discord.send(embed={
            'title': f"{'🟢' if signal == 1 else '🔴'} {symbol} {_dir} | 진입",
            'color': 0x00C853 if signal == 1 else 0xFF1744,
            'fields': [
                {'name': '진입가', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': '포지션', 'value': f'${fill_qty * fill_price:,.2f}', 'inline': True},
                {'name': '레버리지', 'value': f'{lev_int}x', 'inline': True},
                {'name': '이동폭', 'value': f'{move_size*100:.2f}%', 'inline': True},
                {'name': '🎯 TP', 'value': f'${tp_price:.4f}', 'inline': True},
                {'name': '🛑 SL', 'value': f'${sl_price:.4f}', 'inline': True},
                {'name': 'TP 방식', 'value': _tp_tag, 'inline': True},
                {'name': 'SL 방식', 'value': _sl_tag, 'inline': True},
                {'name': '슬롯', 'value': f'{n_active}/{cfg.max_concurrent}', 'inline': True},
                {'name': '── 활성 포지션 ──', 'value': f'```\n{_pos_summary}\n```', 'inline': False},
                {'name': '잔고', 'value': f'${balance_after:,.2f}', 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    def _partial_close_position(
        self, symbol: str, current_price: float,
    ) -> bool:
        """Partially close 50% of a position (v4.3 capital preservation).

        Closes half the position to lock in profit. The remaining half
        continues with a break-even stop. Does NOT remove the position
        from the strategy — it stays tracked with partial_closed=True.

        Returns True if partial close succeeded.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        # v12.5 Race guard: If server already handled the partial close
        # intra-bar (_handle_server_partial_tp_fill sets this flag),
        # skip client-side partial close to avoid double execution.
        if pos.get('_server_partial_tp_filled'):
            logger.info(
                f"[{symbol}] Partial close skipped — "
                f"server TP already handled partial fill"
            )
            return False

        # v11.1: Cancel old server SL (quantity will change)
        sl_oid = pos.get('server_sl_order_id')
        if sl_oid:
            self._cancel_server_stop_loss(symbol, sl_oid)

        # v11.2: Cancel old server TP (quantity will change)
        tp_oid = pos.get('server_tp_order_id')
        if tp_oid:
            self._cancel_server_take_profit(symbol, tp_oid)

        fraction = self.config.partial_exit_fraction
        full_quantity = pos['notional'] / pos['entry_price']
        close_qty = self._round_qty(full_quantity * fraction, symbol)
        if close_qty <= 0:
            return False

        # Reverse the entry side to close
        side = 'SELL' if pos['signal'] == 1 else 'BUY'

        result = self._place_order(symbol, side, close_qty, reduce_only=True)
        if result is None:
            # Failed: revert partial_closed flag & clear stale order IDs.
            # Without clearing IDs, _sync_server_orders sees the old
            # (now-cancelled) IDs as valid and skips bracket repair,
            # leaving the position with ZERO server-side protection.
            pos['partial_closed'] = False
            pos['server_sl_order_id'] = None
            pos['server_tp_order_id'] = None
            # v12.3.2: Re-place server SL/TP immediately to close
            # the 15-minute protection gap (don't wait for bracket repair).
            if not self.paper_mode:
                new_sl = self._place_server_stop_loss(
                    symbol=symbol, signal=pos['signal'],
                    quantity=self._round_qty(full_quantity, symbol),
                    entry_price=pos['entry_price'],
                    move_size=pos['move_size'],
                    stop_mult=pos.get('entry_stop_mult', self.config.stop_mult),
                )
                if new_sl and new_sl != 'WOULD_TRIGGER':
                    pos['server_sl_order_id'] = new_sl
                # v12.5: PBT mode re-places 50% server TP.
                # Non-PBT: full qty server TP for intra-bar capture.
                if self.config.partial_exit_enabled:
                    tp_qty = self._round_qty(
                        full_quantity * self.config.partial_exit_fraction, symbol,
                    )
                    new_tp = self._place_server_take_profit(
                        symbol=symbol, signal=pos['signal'],
                        quantity=tp_qty,
                        entry_price=pos['entry_price'],
                        move_size=pos['move_size'],
                        target_retrace=self.config.target_retrace,
                    )
                    if new_tp and new_tp != 'WOULD_TRIGGER':
                        pos['server_tp_order_id'] = new_tp
                        pos['server_tp_is_partial'] = True
                else:
                    new_tp = self._place_server_take_profit(
                        symbol=symbol, signal=pos['signal'],
                        quantity=self._round_qty(full_quantity, symbol),
                        entry_price=pos['entry_price'],
                        move_size=pos['move_size'],
                        target_retrace=self.config.target_retrace,
                    )
                    if new_tp and new_tp != 'WOULD_TRIGGER':
                        pos['server_tp_order_id'] = new_tp
                logger.info(
                    f"[{symbol}] Partial close failed — re-placed server "
                    f"SL={pos.get('server_sl_order_id')} "
                    f"TP={pos.get('server_tp_order_id')}"
                )
            return False

        # v12.3.1: Handle NO_POSITION — server order closed it between
        # our cancel and our close attempt (race condition)
        if isinstance(result, dict) and result.get('status') == 'NO_POSITION':
            logger.warning(
                f"[{symbol}] Partial close got NO_POSITION — "
                f"server order already closed the position"
            )
            # v12.3.2: Record trade INLINE — don't call _record_server_fill
            # because close_position() already removes the position, and
            # _record_server_fill calls close_position() again → returns
            # None → trade silently dropped.
            closed_pos = self.strategy.close_position(symbol)
            if closed_pos:
                fill_price = pos['entry_price']  # fallback
                try:
                    recent_trades = self.client.futures_account_trades(
                        symbol=symbol, limit=10,
                    )
                    if recent_trades:
                        close_side = 'SELL' if closed_pos['signal'] == 1 else 'BUY'
                        for t in reversed(recent_trades):
                            if t.get('side') == close_side and t.get('realizedPnl', '0') != '0':
                                fill_price = float(t['price'])
                                break
                except Exception:
                    pass

                if closed_pos['signal'] == 1:
                    pnl_pct = (fill_price - closed_pos['entry_price']) / closed_pos['entry_price']
                else:
                    pnl_pct = (closed_pos['entry_price'] - fill_price) / closed_pos['entry_price']
                pnl_usd = pnl_pct * closed_pos['notional']
                cost = closed_pos['notional'] * self.config.round_trip_cost
                net_pnl = pnl_usd - cost
                trade = {
                    'symbol': symbol,
                    'signal': closed_pos['signal'],
                    'entry_price': closed_pos['entry_price'],
                    'exit_price': fill_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': net_pnl,
                    'notional': closed_pos['notional'],
                    'bars_held': closed_pos['bars_held'],
                    'reason': 'server_fill_race',
                    'time': datetime.now(timezone.utc).isoformat(),
                }
                self._trade_history.append(trade)
                self._save_state()
                logger.info(
                    f"[{symbol}] NO_POSITION race recorded: "
                    f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f})"
                )
            return False

        fill_price = float(result['avgPrice'])

        # PnL on the partial close
        if pos['signal'] == 1:
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']

        # v12.3.2: Use entry-price notional for PnL (not exit-price).
        # exit-price notional (close_qty * fill_price) overstates winners
        # and understates losers by fill_price/entry_price ratio.
        partial_notional = close_qty * pos['entry_price']
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
        pos['notional'] = remaining_qty_exact * pos['entry_price']

        # v11.1: Place new server SL for remaining quantity
        remaining_qty = self._round_qty(
            pos['notional'] / pos['entry_price'], symbol,
        )
        if remaining_qty > 0 and not self.paper_mode:
            new_sl_oid = self._place_server_stop_loss(
                symbol=symbol,
                signal=pos['signal'],
                quantity=remaining_qty,
                entry_price=pos['entry_price'],
                move_size=pos['move_size'],
                stop_mult=0.0001,  # Break-even stop (tiny distance)
            )
            pos['server_sl_order_id'] = new_sl_oid

            # v12.3 PBT: No server TP for runner — exits via client-side
            # ATR trailing (trail_runner / time_runner / target).
            # Server TP would conflict with the trailing logic.
            if not self.config.breakeven_after_partial:
                # Legacy mode: place TP for remaining half
                new_tp_oid = self._place_server_take_profit(
                    symbol=symbol,
                    signal=pos['signal'],
                    quantity=remaining_qty,
                    entry_price=pos['entry_price'],
                    move_size=pos['move_size'],
                    target_retrace=self.config.target_retrace,
                )
                pos['server_tp_order_id'] = new_tp_oid
            else:
                pos['server_tp_order_id'] = None

        # Paper mode — charge EXIT-side cost only.
        # Entry cost for the full position was already charged at _open_position.
        # Using round_trip_cost here would double-count the entry fee for
        # the partial portion (overcharge by 0.5 × cost_per_side per trade).
        if self.paper_mode:
            paper_exit_cost = partial_notional * self.config.cost_per_side
            self._paper_balance += pnl_usd - paper_exit_cost

        # Trade record (partial)
        trade = {
            'symbol': symbol,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': net_pnl,
            'notional': partial_notional,
            'bars_held': pos['bars_held'],
            'reason': 'partial_close',
            'time': datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade)

        # v12.4.5: Save state immediately after partial close.
        # Without this, a crash before end-of-cycle _save_state() leaves
        # the state file with 100% notional + partial_closed=False.
        # On restart: bracket repair uses 100% qty, _record_server_fill
        # records PnL on stale 100% notional, and the partial close
        # trade is lost from history entirely.
        self._save_state()

        logger.info(
            f"[{symbol}] PARTIAL CLOSE ({fraction*100:.0f}%): "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"→ remaining ${pos['notional']:.2f}, BE stop active"
        )
        _dir = 'LONG ↑' if pos['signal'] == 1 else 'SHORT ↓'
        # Remaining bracket status
        _r_sl = '🟢 서버 (BE)' if pos.get('server_sl_order_id') else '🟡 클라이언트 (BE)'
        _r_tp = '🟢 서버' if pos.get('server_tp_order_id') else '🟡 클라이언트'
        self.discord.send(embed={
            'title': f"🔒 {symbol} {_dir} | 부분청산 {fraction*100:.0f}%",
            'color': 0xFFC107,
            'description': '원금 확보 완료. 나머지는 손익분기점 스톱으로 안전 운영.',
            'fields': [
                {'name': '진입가', 'value': f'${pos["entry_price"]:.4f}', 'inline': True},
                {'name': '체결가', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': '보유', 'value': f'{pos["bars_held"]}바', 'inline': True},
                {'name': '수익률', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': '확보금액', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': '남은포지션', 'value': f'${pos["notional"]:,.2f}', 'inline': True},
                {'name': 'TP 방식', 'value': _r_tp, 'inline': True},
                {'name': 'SL 방식', 'value': _r_sl, 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True


    def _add_to_position(
        self, symbol: str, current_price: float, balance: float,
    ) -> bool:
        """Add primary capital (8%) to an exploratory Elliott position.

        Called when update_position returns 'confirm_entry' — the
        divergence + MACD/Stochastic confluence has been validated.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            balance: Current account balance.

        Returns:
            True if additional capital was deployed.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        ecfg = getattr(self, 'elliott_config', ElliottConfig())
        add_margin = balance * ecfg.primary_pct
        add_notional = add_margin * ecfg.nominal_leverage
        if add_notional < 10.0:
            return False

        add_qty = add_notional / current_price
        add_qty = self._round_qty(add_qty, symbol)
        if add_qty <= 0:
            return False

        side = 'BUY' if pos['signal'] == 1 else 'SELL'
        result = self._place_order(symbol, side, add_qty)
        if result is None:
            return False

        fill_price = float(result['avgPrice'])
        fill_qty = float(result['executedQty'])

        # Update position with blended entry
        old_notional = pos['notional']
        new_notional = fill_qty * fill_price
        total_notional = old_notional + new_notional
        blended_entry = (
            (pos['entry_price'] * old_notional + fill_price * new_notional)
            / total_notional
        )
        pos['entry_price'] = blended_entry
        pos['notional'] = total_notional

        logger.info(
            f"[{symbol}] CONFIRMED: Added 8% (${new_notional:.2f}) → "
            f"total ${total_notional:.2f} @ blended ${blended_entry:.4f}"
        )
        self.discord.send(embed={
            'title': f"✅ {symbol} 확정진입 | 다이버전스 검증",
            'color': 0x2196F3,
            'description': '다이버전스 확인. 10% 마진으로 업그레이드.',
            'fields': [
                {'name': '추가금액', 'value': f'${new_notional:.2f}', 'inline': True},
                {'name': '총포지션', 'value': f'${total_notional:.2f}', 'inline': True},
                {'name': '평균진입가', 'value': f'${blended_entry:.4f}', 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    def _pyramid_position(
        self, symbol: str, current_price: float, balance: float,
    ) -> bool:
        """Pyramid into a winning Elliott position using unrealized profits.

        Called when update_position returns 'pyramid' — Wave 3/C momentum
        confirmed with EMA alignment. Uses unrealized profit as margin.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            balance: Current account balance.

        Returns:
            True if pyramid order was filled.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        # Unrealized PnL
        if pos['signal'] == 1:
            gain_pct = (current_price - pos['entry_price']) / pos['entry_price']
        else:
            gain_pct = (pos['entry_price'] - current_price) / pos['entry_price']

        unrealized = gain_pct * pos['notional']
        if unrealized < 20.0:  # Min $20 profit for pyramid
            return False

        ecfg = getattr(self, 'elliott_config', ElliottConfig())
        # Use unrealized profit as margin for pyramid
        pyramid_notional = min(unrealized * ecfg.nominal_leverage, pos['notional'] * 0.5)
        pyramid_qty = pyramid_notional / current_price
        pyramid_qty = self._round_qty(pyramid_qty, symbol)
        if pyramid_qty <= 0:
            return False

        side = 'BUY' if pos['signal'] == 1 else 'SELL'
        result = self._place_order(symbol, side, pyramid_qty)
        if result is None:
            return False

        fill_price = float(result['avgPrice'])
        fill_qty = float(result['executedQty'])
        new_notional = fill_qty * fill_price

        # Update position
        old_notional = pos['notional']
        total_notional = old_notional + new_notional
        blended = (
            (pos['entry_price'] * old_notional + fill_price * new_notional)
            / total_notional
        )
        pos['entry_price'] = blended
        pos['notional'] = total_notional

        logger.info(
            f"[{symbol}] PYRAMID: Added ${new_notional:.2f} from "
            f"${unrealized:.2f} unrealized → total ${total_notional:.2f}"
        )
        self.discord.send(embed={
            'title': f"📈 {symbol} 피라미딩 | Wave 3/C 모멘텀",
            'color': 0x4CAF50,
            'description': '미실현 수익으로 추가진입. 스톱 BE로 이동.',
            'fields': [
                {'name': '추가금액', 'value': f'${new_notional:.2f}', 'inline': True},
                {'name': '총포지션', 'value': f'${total_notional:.2f}', 'inline': True},
                {'name': '미실현', 'value': f'${unrealized:.2f}', 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    def _close_position(
        self, symbol: str, reason: str, current_price: float,
    ) -> bool:
        """Close an existing position.

        Returns True if closed successfully.  Also returns True when the
        exchange reports NO_POSITION (-4118/-2022) because the position is
        already gone — we just clean up our internal state.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        quantity = pos['notional'] / pos['entry_price']
        quantity = self._round_qty(quantity, symbol)
        if quantity <= 0:
            return False

        # v11.1: Cancel server-side stop-loss BEFORE closing
        sl_oid = pos.get('server_sl_order_id')
        if sl_oid:
            self._cancel_server_stop_loss(symbol, sl_oid)

        # v11.2: Cancel server-side take-profit BEFORE closing
        tp_oid = pos.get('server_tp_order_id')
        if tp_oid:
            self._cancel_server_take_profit(symbol, tp_oid)

        # Reverse the entry side
        side = 'SELL' if pos['signal'] == 1 else 'BUY'

        result = self._place_order(symbol, side, quantity, reduce_only=True)

        # ── NO_POSITION: exchange says position already gone ──
        # This typically means a server-side TP/SL filled between cycles.
        # We MUST record the trade — otherwise trade count and PnL stay at 0.
        if isinstance(result, dict) and result.get('status') == 'NO_POSITION':
            logger.warning(
                f"[{symbol}] Position already closed on exchange — "
                f"recording trade & cleaning up (reason={reason})"
            )
            # Pop the position from internal tracking
            closed_pos = self.strategy.close_position(symbol)
            if closed_pos:
                # Try to get actual fill price from recent exchange trades
                fill_price = current_price  # fallback
                try:
                    recent_trades = self.client.futures_account_trades(
                        symbol=symbol, limit=10,
                    )
                    if recent_trades:
                        # Find the most recent CLOSE trade (opposite side)
                        close_side = 'SELL' if closed_pos['signal'] == 1 else 'BUY'
                        for t in reversed(recent_trades):
                            if t.get('side') == close_side and t.get('realizedPnl', '0') != '0':
                                fill_price = float(t['price'])
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
                if closed_pos['signal'] == 1:
                    pnl_pct = (fill_price - closed_pos['entry_price']) / closed_pos['entry_price']
                else:
                    pnl_pct = (closed_pos['entry_price'] - fill_price) / closed_pos['entry_price']

                pnl_usd = pnl_pct * closed_pos['notional']
                # v12.3.2: Always charge round_trip_cost for parity.
                cost = closed_pos['notional'] * self.config.round_trip_cost
                net_pnl = pnl_usd - cost

                # Trade record
                trade = {
                    'symbol': symbol,
                    'signal': closed_pos['signal'],
                    'entry_price': closed_pos['entry_price'],
                    'exit_price': fill_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': net_pnl,
                    'notional': closed_pos['notional'],
                    'bars_held': closed_pos['bars_held'],
                    'reason': f'{reason}_no_position',
                    'time': datetime.now(timezone.utc).isoformat(),
                }
                self._trade_history.append(trade)

                # v12.0: Exit diagnostic to JSONL
                _diag_logger.log_exit(
                    self._cycle_count,
                    datetime.now(timezone.utc).isoformat(),
                    trade,
                )

                _exit_side = 'LONG' if closed_pos['signal'] == 1 else 'SHORT'
                print(format_exit(
                    symbol=symbol,
                    side=_exit_side,
                    entry_price=closed_pos['entry_price'],
                    exit_price=fill_price,
                    pnl_usd=net_pnl,
                    pnl_pct=pnl_pct * 100,
                    reason=f'{reason} (no_position)',
                    bars_held=closed_pos['bars_held'],
                ))
                logger.info(
                    f"[{symbol}] NO_POSITION trade recorded: {_exit_side} "
                    f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
                    f"exit=${fill_price:.4f} held={closed_pos['bars_held']} bars"
                )

                # Session stats
                cum_pnl = sum(t['pnl_usd'] for t in self._trade_history)
                n_trades = len(self._trade_history)
                wins_total = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
                wr = wins_total / n_trades * 100 if n_trades > 0 else 0
                balance_after = self.get_balance()
                hold_time = closed_pos['bars_held'] * 15  # minutes

                _dir_kr = 'LONG ↑' if closed_pos['signal'] == 1 else 'SHORT ↓'
                _reason_kr = reason.upper().replace('_', ' ')
                n_remaining = self.strategy.n_active

                self.discord.send(embed={
                    'title': f"{'✅' if net_pnl > 0 else '❌'} {symbol} {_dir_kr} | {_reason_kr} (서버체결)",
                    'color': 0x00C853 if net_pnl > 0 else 0xFF1744,
                    'fields': [
                        {'name': '진입가', 'value': f'${closed_pos["entry_price"]:.4f}', 'inline': True},
                        {'name': '청산가', 'value': f'${fill_price:.4f}', 'inline': True},
                        {'name': '보유시간', 'value': f'{hold_time}분 ({closed_pos["bars_held"]}바)', 'inline': True},
                        {'name': '수익률', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                        {'name': '순수익', 'value': f'${net_pnl:+.2f}', 'inline': True},
                        {'name': '체결방식', 'value': '🟢 거래소(서버)', 'inline': True},
                        {'name': '── 세션 현황 ──', 'value': (
                            f'누적 PnL: ${cum_pnl:+,.2f} | '
                            f'{n_trades}건 | 승률 {wr:.0f}%\n'
                            f'남은 포지션: {n_remaining}개 | '
                            f'잔고: ${balance_after:,.2f}'
                        ), 'inline': False},
                    ],
                    'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
                })
            self._save_state()
            return True

        if result is None:
            # Failed to close — position stays in state for retry
            # SL/TP already cancelled, so re-place them.
            # v12.3 PBT: Runner positions need BE stop, not original stop/TP.
            is_runner = pos.get('partial_closed') and self.config.breakeven_after_partial
            if is_runner:
                # Runner: use breakeven stop, no server TP
                new_sl_oid = self._place_server_stop_loss(
                    symbol=symbol,
                    signal=pos['signal'],
                    quantity=quantity,
                    entry_price=pos['entry_price'],
                    move_size=pos['move_size'],
                    stop_mult=0.0001,  # BE stop
                )
                new_tp_oid = None
                # v12.4.4: Reset ratchet state so _ratchet_runner_sl
                # re-ratchets from breakeven on the next cycle.
                # Without this, stale _server_sl_pct from the previous
                # ratcheted level prevents re-ratcheting (<0.5% improvement),
                # leaving the server SL stuck at breakeven.
                pos['_server_sl_pct'] = 0.0
            else:
                # Phase 1: use original stop
                new_sl_oid = self._place_server_stop_loss(
                    symbol=symbol,
                    signal=pos['signal'],
                    quantity=quantity,
                    entry_price=pos['entry_price'],
                    move_size=pos['move_size'],
                    stop_mult=pos.get('entry_stop_mult', self.config.stop_mult),
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
                    new_tp_oid = self._place_server_take_profit(
                        symbol=symbol,
                        signal=pos['signal'],
                        quantity=quantity,
                        entry_price=pos['entry_price'],
                        move_size=pos['move_size'],
                        target_retrace=self.config.target_retrace,
                    )
            # Update existing position's server order IDs
            # Guard against sentinel values being stored as order IDs
            if new_sl_oid and new_sl_oid != 'WOULD_TRIGGER':
                pos['server_sl_order_id'] = new_sl_oid
            else:
                pos['server_sl_order_id'] = None
            # v12.3.1: Guard TP sentinel too
            if new_tp_oid and new_tp_oid != 'WOULD_TRIGGER':
                pos['server_tp_order_id'] = new_tp_oid
            else:
                pos['server_tp_order_id'] = None
            self._save_state()

            # v12.3.1: SL WOULD_TRIGGER = price already past stop level.
            # Leaving position open with no SL for 15min is unacceptable.
            # Guard against infinite recursion if this is already a force close.
            if (
                new_sl_oid == 'WOULD_TRIGGER'
                and not reason.startswith('force_')
            ):
                logger.error(
                    f"[{symbol}] SL WOULD_TRIGGER during re-placement — "
                    f"price past stop level, forcing immediate close"
                )
                return self._close_position(
                    symbol, 'force_sl_would_trigger', current_price,
                )

            return False

        # ── Successful close — NOW remove from state ──
        pos = self.strategy.close_position(symbol)
        if pos is None:
            return False

        fill_price = float(result['avgPrice'])

        # PnL calculation
        if pos['signal'] == 1:
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']

        pnl_usd = pnl_pct * pos['notional']
        # v12.3.2: Always charge round_trip_cost for backtest parity.
        # Each portion (partial and runner) accounts for its share
        # of entry + exit costs. See _backtest_core_partial delta model.
        cost = pos['notional'] * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        # Paper balance
        if self.paper_mode:
            self._paper_balance += pnl_usd
            close_cost = quantity * fill_price * self.config.cost_per_side
            self._paper_balance -= close_cost

        # Trade record
        trade = {
            'symbol': symbol,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': net_pnl,
            'notional': pos['notional'],
            'bars_held': pos['bars_held'],
            'reason': reason,
            'time': datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade)

        # v12.0: Exit diagnostic to JSONL
        _diag_logger.log_exit(self._cycle_count,
                              datetime.now(timezone.utc).isoformat(), trade)

        _exit_side = 'LONG' if pos['signal'] == 1 else 'SHORT'
        print(format_exit(
            symbol=symbol,
            side=_exit_side,
            entry_price=pos['entry_price'],
            exit_price=fill_price,
            pnl_usd=net_pnl,
            pnl_pct=pnl_pct * 100,
            reason=reason,
            bars_held=pos['bars_held'],
        ))
        logger.info(
            f"[{symbol}] CLOSED ({reason}): {_exit_side} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"held={pos['bars_held']} bars"
        )
        # Session stats for notification
        cum_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins_total = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
        wr = wins_total / n_trades * 100 if n_trades > 0 else 0
        balance_after = self.get_balance()
        hold_time = pos['bars_held'] * 15  # minutes

        _dir_kr = 'LONG ↑' if pos['signal'] == 1 else 'SHORT ↓'
        _reason_kr = reason.upper().replace('_', ' ')
        n_remaining = self.strategy.n_active
        # Remaining positions summary
        _pos_lines = []
        for _sym, _p in self.strategy.active_positions.items():
            _pd = 'L' if _p['signal'] == 1 else 'S'
            _pos_lines.append(f"{_sym} {_pd} ${_p['entry_price']:.4f}")
        _pos_summary = '\n'.join(_pos_lines) if _pos_lines else '없음'

        self.discord.send(embed={
            'title': f"{'✅' if net_pnl > 0 else '❌'} {symbol} {_dir_kr} | {_reason_kr}",
            'color': 0x00C853 if net_pnl > 0 else 0xFF1744,
            'fields': [
                {'name': '진입가', 'value': f'${pos["entry_price"]:.4f}', 'inline': True},
                {'name': '청산가', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': '보유시간', 'value': f'{hold_time}분 ({pos["bars_held"]}바)', 'inline': True},
                {'name': '수익률', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': '순수익', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': '체결방식', 'value': '🔵 클라이언트', 'inline': True},
                {'name': '── 세션 현황 ──', 'value': (
                    f'누적 PnL: ${cum_pnl:+,.2f} | '
                    f'{n_trades}건 | 승률 {wr:.0f}%\n'
                    f'잔고: ${balance_after:,.2f}'
                ), 'inline': False},
                {'name': '── 남은 포지션 ──', 'value': f'```\n{_pos_summary}\n```', 'inline': False},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    # ==================== Main Cycle ====================

