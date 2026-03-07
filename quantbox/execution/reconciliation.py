"""Position reconciliation with exchange reality.

Mixin class for syncing internal state with Binance exchange state:
server order fill detection, ghost position cleanup, and quantity
mismatch correction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger('Scalper')


class ReconciliationMixin:
    """Mixin: Exchange reconciliation (sync server orders, detect ghosts)."""

    def _sync_server_orders(self) -> None:
        """Check if any server-side TP/SL orders filled between cycles.

        v11.2: Reconciles internal state with exchange reality.
        Called at the start of each cycle before processing positions.
        If a server TP/SL fired intra-bar, we record the trade and cancel
        the counterpart order.
        """
        if self.paper_mode:
            return

        for sym in list(self.strategy.active_positions):
            pos = self.strategy._positions.get(sym)
            if not pos:
                continue

            # Extract Binance symbol from compound key (e.g. BTCUSDT:LONG → BTCUSDT).
            # pos['symbol'] may itself be a compound key if loaded from a corrupted
            # state file — always strip the suffix defensively.
            raw_sym = pos.get('symbol', sym)
            api_sym = raw_sym.split(':')[0] if ':' in raw_sym else raw_sym

            tp_oid = pos.get('server_tp_order_id')
            sl_oid = pos.get('server_sl_order_id')
            logger.debug(
                f"[{sym}] Sync check: sl_algoId={sl_oid} tp_algoId={tp_oid} "
                f"entry=${pos.get('entry_price', 0):.4f} "
                f"signal={'LONG' if pos.get('signal') == 1 else 'SHORT'}"
            )

            # Check TP order first (the happy path)
            if tp_oid:
                filled = self._check_order_filled(api_sym, tp_oid)
                if filled and filled.get('status') == 'FILLED':
                    fill_price = float(filled['avgPrice'])
                    fill_qty = float(filled['executedQty'])

                    # v12.5: PBT partial TP — server closed 50%, transition
                    # to Phase 2 runner instead of fully closing position.
                    if pos.get('server_tp_is_partial') and not pos.get('partial_closed'):
                        logger.info(
                            f"[{sym}] Server TP (PARTIAL) TRIGGERED intra-bar: "
                            f"fill=${fill_price:.4f} qty={fill_qty:.6f}"
                        )
                        self._handle_server_partial_tp_fill(
                            sym, pos, fill_price, fill_qty,
                        )
                        continue

                    logger.info(
                        f"[{sym}] Server TP TRIGGERED intra-bar: "
                        f"fill=${fill_price:.4f} qty={fill_qty:.6f}"
                    )
                    # Cancel the counterpart SL
                    if sl_oid:
                        self._cancel_server_stop_loss(api_sym, sl_oid)
                    self._record_server_fill(
                        sym, fill_price, fill_qty, 'server_tp',
                    )
                    continue
                elif filled and filled.get('status') == 'CANCELLED':
                    # v12.3.1: Externally cancelled — clear for bracket repair
                    pos['server_tp_order_id'] = None
                    tp_oid = None
                    logger.warning(
                        f"[{sym}] Server TP was externally cancelled — "
                        f"cleared for bracket repair"
                    )

            # Check SL order
            if sl_oid:
                filled = self._check_order_filled(api_sym, sl_oid)
                if filled and filled.get('status') == 'FILLED':
                    fill_price = float(filled['avgPrice'])
                    fill_qty = float(filled['executedQty'])
                    logger.info(
                        f"[{sym}] Server SL TRIGGERED intra-bar: "
                        f"fill=${fill_price:.4f} qty={fill_qty:.6f}"
                    )
                    # Cancel the counterpart TP
                    if tp_oid:
                        self._cancel_server_take_profit(api_sym, tp_oid)
                    self._record_server_fill(
                        sym, fill_price, fill_qty, 'server_sl',
                    )
                    continue
                elif filled and filled.get('status') == 'CANCELLED':
                    # v12.3.1: Externally cancelled — clear for bracket repair
                    pos['server_sl_order_id'] = None
                    sl_oid = None
                    logger.warning(
                        f"[{sym}] Server SL was externally cancelled — "
                        f"cleared for bracket repair"
                    )

            # ═══ v11.4.2 BRACKET REPAIR: Place missing SL/TP for restored positions ═══
            # Positions restored from state may have None algoIds (e.g. old session
            # failed to place them). Detect and repair to ensure server-side protection.
            missing_sl = sl_oid is None
            # v12.5: In PBT mode, server TP is now placed for 50% qty.
            # After partial close (Phase 2 runner), TP is intentionally None.
            # Don't treat post-partial absent TP as "missing".
            if self.config.partial_exit_enabled:
                # Pre-partial: TP should exist (server 50% TP)
                # Post-partial: TP intentionally absent (runner uses trailing)
                missing_tp = tp_oid is None and not pos.get('partial_closed', False)
            else:
                missing_tp = tp_oid is None

            if missing_sl or missing_tp:
                quantity = pos['notional'] / pos['entry_price']
                quantity = self._round_qty(quantity, api_sym)
                signal = pos['signal']
                entry_price = pos['entry_price']
                move_size = pos.get('move_size', 0.03)
                stop_mult = pos.get(
                    'entry_stop_mult', self.config.stop_mult,
                )

                if missing_sl:
                    # v12.3.1: Runner needs BE stop, not original stop
                    if pos.get('partial_closed') and self.config.breakeven_after_partial:
                        repair_stop_mult = 0.0001  # BE stop for runner
                    else:
                        repair_stop_mult = stop_mult
                    new_sl = self._place_server_stop_loss(
                        symbol=api_sym, signal=signal, quantity=quantity,
                        entry_price=entry_price, move_size=move_size,
                        stop_mult=repair_stop_mult,
                        position_side=pos.get('position_side'),
                    )
                    if new_sl == 'WOULD_TRIGGER':
                        # ═══ FORCE-CLOSE: Price blew past SL level ═══
                        # The stop price is already behind the market.
                        # Close the position immediately at market with
                        # reduceOnly to prevent the "margin insufficient"
                        # error that killed us before.
                        logger.critical(
                            f"[{sym}] BRACKET REPAIR → FORCE CLOSE: "
                            f"Price past SL. Closing at market."
                        )
                        self.discord.send(content=(
                            f"\U0001f6a8 **강제청산** [{sym}] — "
                            f"SL 가격 이탈 감지. 시장가 긴급 청산 실행."
                        ))
                        # v12.3.1: Fetch real price for force close
                        # (0.0 would corrupt PnL if NO_POSITION fallback)
                        _fc_price = entry_price  # safe fallback
                        try:
                            _book = self.client.futures_orderbook_ticker(
                                symbol=api_sym,
                            )
                            _fc_price = float(_book['bidPrice'])
                        except Exception:
                            pass
                        closed = self._close_position(
                            sym, 'force_sl_breach', _fc_price,
                        )
                        if closed:
                            logger.info(
                                f"[{sym}] FORCE CLOSE SUCCESS — "
                                f"position liquidated at market."
                            )
                        else:
                            logger.critical(
                                f"[{sym}] FORCE CLOSE FAILED — "
                                f"MANUAL INTERVENTION REQUIRED!"
                            )
                            self.discord.send(content=(
                                f"\U0001f6a8\U0001f6a8 **강제청산 실패** "
                                f"[{sym}] — 수동 청산 필요! 즉시 거래소 확인!"
                            ))
                        continue  # Skip TP repair — position gone
                    elif new_sl:
                        pos['server_sl_order_id'] = new_sl
                        # v12.4.4: Reset ratchet state when runner SL is
                        # re-placed at breakeven. Without this, _server_sl_pct
                        # retains the old ratcheted value, _ratchet_runner_sl
                        # sees insufficient improvement (<0.5%), and skips
                        # re-ratcheting — leaving server SL stuck at breakeven.
                        if repair_stop_mult < 0.001:  # BE stop for runner
                            pos['_server_sl_pct'] = 0.0
                        logger.info(
                            f"[{sym}] BRACKET REPAIR: SL placed "
                            f"algoId={new_sl}"
                        )
                    else:
                        logger.error(
                            f"[{sym}] BRACKET REPAIR FAILED: "
                            f"Could not place SL — position UNPROTECTED"
                        )

                if missing_tp:
                    # v12.5: PBT mode uses 50% qty server TP (pre-partial only).
                    # Non-PBT: full qty server TP for intra-bar capture.
                    if self.config.partial_exit_enabled:
                        tp_qty = self._round_qty(
                            quantity * self.config.partial_exit_fraction, api_sym,
                        )
                    else:
                        tp_qty = quantity
                    # v14.0: Use per-position TP (not RubberBand default).
                    # VolStraddle stores tp in position dict (VR=12%, MA=8%).
                    repair_tp = pos.get('tp', self.config.target_retrace)
                    new_tp = self._place_server_take_profit(
                        symbol=api_sym, signal=signal, quantity=tp_qty,
                        entry_price=entry_price, move_size=move_size,
                        target_retrace=repair_tp,
                        position_side=pos.get('position_side'),
                    )
                    if new_tp:
                        pos['server_tp_order_id'] = new_tp
                        if self.config.partial_exit_enabled:
                            pos['server_tp_is_partial'] = True
                        logger.info(
                            f"[{sym}] BRACKET REPAIR: TP placed "
                            f"algoId={new_tp} "
                            f"{'(50% partial)' if self.config.partial_exit_enabled else ''}"
                        )
                    else:
                        logger.warning(
                            f"[{sym}] BRACKET REPAIR: TP failed — "
                            f"client-side TP still active"
                        )

                if missing_sl or missing_tp:
                    self._save_state()
                    sl_id = pos.get('server_sl_order_id')
                    tp_id = pos.get('server_tp_order_id')
                    logger.info(
                        f"[{sym}] Bracket repair complete: "
                        f"SL={'OK' if sl_id else 'MISSING'} "
                        f"(algoId={sl_id}) "
                        f"TP={'OK' if tp_id else 'MISSING'} "
                        f"(algoId={tp_id})"
                    )

    def _reconcile_positions(self) -> None:
        """Reconcile internal position state with exchange reality.

        v12.5.1: Called each cycle AFTER _sync_server_orders.
        Detects ghost positions (closed on exchange but still in internal
        state) and quantity mismatches (ADL, manual intervention).
        """
        try:
            exchange_positions = self.client.futures_position_information()
        except Exception as e:
            logger.warning(f"Position reconciliation failed: {e}")
            return

        # Build map of actual exchange positions.
        # In Hedge Mode, Binance returns TWO entries per symbol
        # (positionSide=LONG and SHORT). Use compound key to avoid
        # one side overwriting the other → false ghost detection.
        exchange_map: dict[str, float] = {}
        for p in exchange_positions:
            amt = float(p.get('positionAmt', '0'))
            if amt != 0:
                pos_side = p.get('positionSide', 'BOTH')
                if pos_side in ('LONG', 'SHORT'):
                    key = f"{p['symbol']}:{pos_side}"
                else:
                    key = p['symbol']
                exchange_map[key] = amt

        # Check each internal position against exchange
        for sym in list(self.strategy.active_positions):
            pos = self.strategy._positions.get(sym)
            if not pos:
                continue

            # Build lookup key: compound (BTCUSDT:LONG) or plain (BTCUSDT).
            # Strip any compound suffix from pos['symbol'] defensively.
            raw_sym = pos.get('symbol', sym)
            api_sym = raw_sym.split(':')[0] if ':' in raw_sym else raw_sym
            pos_side = pos.get('position_side')
            if pos_side in ('LONG', 'SHORT'):
                lookup_key = f"{api_sym}:{pos_side}"
            else:
                lookup_key = api_sym

            if lookup_key not in exchange_map:
                # Ghost position: we think it's open, exchange says no
                logger.error(
                    f"[{sym}] RECONCILIATION: Ghost position detected! "
                    f"Internal entry=${pos['entry_price']:.4f} but "
                    f"exchange has NO position. Recording as unknown close."
                )
                self.discord.send(content=(
                    f"⚠️ **[{sym}] Ghost position detected** — "
                    f"Position missing on exchange. Cleaning up internal state."
                ))
                # ── Ghost Fill Audit (P1) ──────────────────────────
                # Retrieve and log ALL closing fills so we can diagnose
                # *why* _sync_server_orders missed the fill.
                fill_price = pos['entry_price']
                audit_fills: list[dict] = []
                try:
                    recent = self.client.futures_account_trades(
                        symbol=api_sym, limit=50,
                    )
                    close_side = 'SELL' if pos['signal'] == 1 else 'BUY'
                    for t in reversed(recent):
                        if (
                            t.get('side') == close_side
                            and float(t.get('realizedPnl', '0')) != 0
                        ):
                            audit_fills.append({
                                'orderId': t.get('orderId'),
                                'price': float(t['price']),
                                'qty': float(t.get('qty', '0')),
                                'realizedPnl': float(t.get('realizedPnl', '0')),
                                'commission': float(t.get('commission', '0')),
                                'commissionAsset': t.get('commissionAsset', ''),
                                'time': t.get('time'),          # epoch ms
                                'maker': t.get('maker', False),
                                'buyer': t.get('buyer', False),
                            })
                    if audit_fills:
                        fill_price = audit_fills[0]['price']  # most recent
                except Exception as e:
                    logger.warning(f"[{sym}] Ghost audit: trade fetch failed: {e}")

                # Also try to find the algo order that caused the fill
                algo_audit: dict | None = None
                if audit_fills:
                    try:
                        all_orders = self.client.futures_get_all_orders(
                            symbol=api_sym, limit=20,
                        )
                        fill_order_id = audit_fills[0]['orderId']
                        for o in reversed(all_orders):
                            if o.get('orderId') == fill_order_id:
                                algo_audit = {
                                    'orderId': o.get('orderId'),
                                    'clientOrderId': o.get('clientOrderId', ''),
                                    'type': o.get('type', ''),
                                    'side': o.get('side', ''),
                                    'origType': o.get('origType', ''),
                                    'status': o.get('status', ''),
                                    'stopPrice': float(o.get('stopPrice', '0')),
                                    'price': float(o.get('price', '0')),
                                    'updateTime': o.get('updateTime'),
                                }
                                break
                    except Exception as e:
                        logger.warning(f"[{sym}] Ghost audit: order fetch failed: {e}")

                # ── Structured audit log ──
                logger.error(
                    f"[{sym}] GHOST FILL AUDIT | "
                    f"entry=${pos['entry_price']:.4f} "
                    f"signal={'LONG' if pos['signal'] == 1 else 'SHORT'} "
                    f"bars_held={pos['bars_held']} "
                    f"notional=${pos['notional']:.2f}"
                )
                if audit_fills:
                    for i, af in enumerate(audit_fills[:5]):
                        ts_str = ''
                        if af['time']:
                            ts_str = datetime.fromtimestamp(
                                af['time'] / 1000, tz=timezone.utc,
                            ).strftime('%Y-%m-%d %H:%M:%S UTC')
                        logger.error(
                            f"  fill[{i}] orderId={af['orderId']} "
                            f"price={af['price']:.4f} qty={af['qty']:.6f} "
                            f"realizedPnl={af['realizedPnl']:+.4f} "
                            f"commission={af['commission']:.6f} "
                            f"{'MAKER' if af['maker'] else 'TAKER'} "
                            f"time={ts_str}"
                        )
                else:
                    logger.error(f"  [NO MATCHING FILLS FOUND in last 50 trades]")

                if algo_audit:
                    logger.error(
                        f"  trigger_order: id={algo_audit['orderId']} "
                        f"type={algo_audit['origType']} "
                        f"status={algo_audit['status']} "
                        f"stopPrice={algo_audit['stopPrice']:.4f} "
                        f"clientId={algo_audit['clientOrderId']}"
                    )

                # Check for internal state clues: were SL/TP algo IDs set?
                sl_oid = pos.get('server_sl_order_id')
                tp_oid = pos.get('server_tp_order_id')
                logger.error(
                    f"  internal_state: server_sl_id={sl_oid} "
                    f"server_tp_id={tp_oid} "
                    f"pbt_phase={pos.get('pbt_phase', 'N/A')} "
                    f"server_sl_pct={pos.get('_server_sl_pct', 'N/A')}"
                )

                # Discord notification with audit summary
                audit_desc = 'No matching fills found on exchange.'
                if audit_fills:
                    af0 = audit_fills[0]
                    ts_str = ''
                    if af0['time']:
                        ts_str = datetime.fromtimestamp(
                            af0['time'] / 1000, tz=timezone.utc,
                        ).strftime('%H:%M:%S UTC')
                    audit_desc = (
                        f"Fill: ${af0['price']:.4f} × {af0['qty']:.6f} "
                        f"@ {ts_str}\n"
                        f"PnL: {af0['realizedPnl']:+.4f} USDT\n"
                        f"Order: {algo_audit['origType'] if algo_audit else '?'} "
                        f"(id={af0['orderId']})"
                    )
                self.discord.send(embed={
                    'title': f"👻 [{sym}] Ghost Fill Audit",
                    'color': 0xFF6D00,
                    'description': audit_desc,
                    'fields': [
                        {'name': 'Entry', 'value': f"${pos['entry_price']:.4f}", 'inline': True},
                        {'name': 'Bars Held', 'value': str(pos['bars_held']), 'inline': True},
                        {'name': 'SL AlgoID', 'value': str(sl_oid or 'None'), 'inline': True},
                        {'name': 'TP AlgoID', 'value': str(tp_oid or 'None'), 'inline': True},
                        {'name': 'Total Fills', 'value': str(len(audit_fills)), 'inline': True},
                    ],
                })
                # v14.0.1: Cancel orphaned counterpart algo orders.
                # Without this, the remaining SL or TP stays on Binance
                # and could trigger on a future position for the same symbol.
                if sl_oid:
                    self._cancel_server_stop_loss(api_sym, sl_oid)
                if tp_oid:
                    self._cancel_server_take_profit(api_sym, tp_oid)
                self._record_server_fill(sym, fill_price, 0, 'ghost_reconcile')
                continue

            # Quantity mismatch check
            exchange_qty = abs(exchange_map[lookup_key])
            internal_qty = pos['notional'] / pos['entry_price']
            qty_diff_pct = abs(exchange_qty - internal_qty) / max(internal_qty, 1e-8) * 100
            if qty_diff_pct > 5:  # More than 5% mismatch
                logger.warning(
                    f"[{sym}] RECONCILIATION: Qty mismatch! "
                    f"Internal={internal_qty:.6f} Exchange={exchange_qty:.6f} "
                    f"Diff={qty_diff_pct:.1f}%. Correcting internal state."
                )
                # Update internal notional to match exchange reality
                pos['notional'] = exchange_qty * pos['entry_price']

    def _handle_server_partial_tp_fill(
        self, symbol: str, pos: dict,
        fill_price: float, fill_qty: float,
    ) -> None:
        """Handle server TP fill for 50% qty (PBT mode Phase 1 → Phase 2).

        v12.5: When server TP fires for partial qty, transition the
        remaining position to runner mode (BE stop + ATR trailing)
        instead of fully closing.

        Flow:
          1. Cancel old server SL (was for 100% qty)
          2. Record PnL on the closed 50%
          3. Update position: notional, partial_closed, runner state
          4. Place new BE SL for remaining 50% qty
          5. Save state immediately
        """
        # v12.5 Race guard: skip if client already handled partial close
        if pos.get('partial_closed'):
            logger.warning(
                f"[{symbol}] Server partial TP fill skipped — "
                f"client already processed partial close"
            )
            pos['server_tp_order_id'] = None
            pos['server_tp_is_partial'] = False
            return

        # Extract Binance symbol from compound key
        api_sym = pos.get('symbol', symbol.split(':')[0])

        # 1. Cancel old SL (100% qty, no longer valid)
        sl_oid = pos.get('server_sl_order_id')
        if sl_oid:
            self._cancel_server_stop_loss(api_sym, sl_oid)
            pos['server_sl_order_id'] = None

        # 2. PnL on the partial portion
        if pos['signal'] == 1:
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']

        partial_notional = fill_qty * pos['entry_price']
        pnl_usd = pnl_pct * partial_notional
        cost = partial_notional * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        # 3. Update position for runner mode
        full_quantity = pos['notional'] / pos['entry_price']
        remaining_qty_exact = full_quantity - fill_qty
        pos['notional'] = remaining_qty_exact * pos['entry_price']
        pos['partial_closed'] = True
        pos['_server_partial_tp_filled'] = True  # v12.5 race guard
        pos['server_tp_order_id'] = None
        pos['server_tp_is_partial'] = False
        pos['highest_since_tp1'] = pnl_pct if pnl_pct > 0 else 0.0
        pos['trail_floor'] = 0.0  # breakeven floor
        pos['best_pnl'] = max(pos.get('best_pnl', 0.0), pnl_pct)

        # 4. Place BE SL for remaining qty
        remaining_qty = self._round_qty(
            pos['notional'] / pos['entry_price'], api_sym,
        )
        if remaining_qty > 0 and not self.paper_mode:
            new_sl_oid = self._place_server_stop_loss(
                symbol=api_sym,
                signal=pos['signal'],
                quantity=remaining_qty,
                entry_price=pos['entry_price'],
                move_size=pos['move_size'],
                stop_mult=0.0001,  # Break-even stop
                position_side=pos.get('position_side'),
            )
            pos['server_sl_order_id'] = new_sl_oid
            # Reset ratchet state for fresh runner
            pos['_server_sl_pct'] = 0.0

        # Paper mode cost handling
        if self.paper_mode:
            paper_exit_cost = partial_notional * self.config.cost_per_side
            self._paper_balance += pnl_usd - paper_exit_cost

        # 5. Trade record (partial)
        trade = {
            'symbol': api_sym,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': net_pnl,
            'notional': partial_notional,
            'bars_held': pos['bars_held'],
            'reason': 'server_partial_tp',
            'time': datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade)
        self._save_state()

        fraction = self.config.partial_exit_fraction
        logger.info(
            f"[{symbol}] SERVER PARTIAL TP ({fraction*100:.0f}%): "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"→ remaining ${pos['notional']:.2f}, BE stop active"
        )
        _dir = 'LONG ↑' if pos['signal'] == 1 else 'SHORT ↓'
        _r_sl = '🟢 서버 (BE)' if pos.get('server_sl_order_id') else '🟡 클라이언트'
        self.discord.send(embed={
            'title': f"🔒 {symbol} {_dir} | 서버 TP 부분청산 {fraction*100:.0f}%",
            'color': 0x4CAF50,
            'description': (
                '서버 TP가 인트라바에서 50% 부분청산 완료.\n'
                '나머지 50%는 손익분기점 스톱 + ATR 트레일링.'
            ),
            'fields': [
                {'name': '진입가', 'value': f'${pos["entry_price"]:.4f}', 'inline': True},
                {'name': 'TP 체결가', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': '수익률', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': '확보금액', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': '남은 포지션', 'value': f'${pos["notional"]:.2f}', 'inline': True},
                {'name': '남은 SL', 'value': _r_sl, 'inline': True},
            ],
        })

    def _record_server_fill(
        self, symbol: str, fill_price: float, fill_qty: float, reason: str,
    ) -> None:
        """Record a trade closed by a server-side order (TP or SL).

        Removes the position from internal tracking and logs the PnL.
        """
        pos = self.strategy.close_position(symbol)
        if not pos:
            return

        # Extract Binance symbol from compound key for trade records
        api_sym = pos.get('symbol', symbol.split(':')[0])

        # PnL calculation
        if pos['signal'] == 1:
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']

        # v12.3.2: Use entry-price-based notional for PnL, not exit-price.
        # fill_qty * fill_price overstates winners and understates losers
        # by a factor of fill_price/entry_price (~3-5% error per trade).
        notional = pos['notional']
        pnl_usd = pnl_pct * notional
        # v12.3.2: Always charge round_trip_cost for backtest parity.
        # Both partial and runner portions must account for their share
        # of entry + exit costs. The backtest delta model charges
        # |delta| * cost_per_side per transition, totaling round_trip
        # over the full lifecycle.
        cost = notional * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        trade = {
            'symbol': api_sym,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': net_pnl,
            'notional': notional,
            'bars_held': pos['bars_held'],
            'reason': reason,
            'time': datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade)

        # v12.3.1: Save state immediately after recording server fill.
        # Without this, a crash before end-of-cycle _save_state() creates
        # a ghost position on restart (position gone on exchange but
        # still in our state file).
        self._save_state()

        emoji = '🎯' if reason == 'server_tp' else '🛑'
        logger.info(
            f"[{symbol}] {emoji} {reason.upper()} FILLED: "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"held={pos['bars_held']} bars"
        )
        cum_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins_total = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
        wr = wins_total / n_trades * 100 if n_trades > 0 else 0
        hold_time = pos['bars_held'] * 15

        _dir = 'LONG ↑' if pos['signal'] == 1 else 'SHORT ↓'
        _fill_type = '서버 TP 체결 (바 중간 익절)' if reason == 'server_tp' else '서버 SL 체결 (안전장치)'
        n_remaining = self.strategy.n_active
        self.discord.send(embed={
            'title': f"{emoji} {symbol} {_dir} | 서버체결",
            'color': 0x00C853 if net_pnl > 0 else 0xFF1744,
            'description': f"{_fill_type}",
            'fields': [
                {'name': '진입가', 'value': f'${pos["entry_price"]:.4f}', 'inline': True},
                {'name': '체결가', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': '보유시간', 'value': f'{hold_time}분 ({pos["bars_held"]}바)', 'inline': True},
                {'name': '수익률', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': '순수익', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': '체결방식', 'value': '🟢 거래소(서버)', 'inline': True},
                {'name': '── 세션 현황 ──', 'value': (
                    f'누적 PnL: ${cum_pnl:+,.2f} | '
                    f'{n_trades}건 | 승률 {wr:.0f}%\n'
                    f'남은 포지션: {n_remaining}개'
                ), 'inline': False},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })

    # ==================== Position Management ====================

