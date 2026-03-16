"""Position reconciliation with exchange reality.

Mixin class for syncing internal state with Binance exchange state:
server order fill detection, ghost position cleanup, and quantity
mismatch correction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantbox.core.protocol import ScalperProtocol

import logging
import os
import time
from datetime import datetime, timezone

from quantbox.execution.recon_journal import ReconEventType

logger = logging.getLogger("Scalper")


class ReconciliationMixin:
    """Mixin: Exchange reconciliation (sync server orders, detect ghosts)."""

    _cycle_balance: float | None

    def _sync_server_orders(self: "ScalperProtocol") -> None:
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
            raw_sym = pos.get("symbol", sym)
            api_sym = raw_sym.split(":")[0] if ":" in raw_sym else raw_sym

            # ── Step 2.1: Re-attempt counterpart SL cancel if flagged from previous cycle ──
            if pos.get("counterpart_sl_cancel_failed") and pos.get("is_straddle"):
                counterpart_key = self._get_pair_counterpart(sym)
                if counterpart_key:
                    counterpart = self.strategy._positions.get(counterpart_key)
                    if counterpart:
                        retry_sl_oid = counterpart.get("server_sl_order_id")
                        retry_api_sym = counterpart_key.split(":")[0]
                        if retry_sl_oid:
                            cfg_vs = getattr(self, "vs_config", None)
                            retry_count = cfg_vs.sl_cancel_retry_count if cfg_vs else 0
                            retry_delay = cfg_vs.sl_cancel_retry_delay_s if cfg_vs else 1.0
                            for attempt in range(retry_count + 1):
                                if self._cancel_server_stop_loss(retry_api_sym, retry_sl_oid):
                                    logger.info(f"[{sym}] 카운터파트 SL 재시도 취소 성공: {counterpart_key}")
                                    pos["counterpart_sl_cancel_failed"] = False
                                    break
                                if attempt < retry_count:
                                    time.sleep(retry_delay)
                            else:
                                logger.warning(f"[{sym}] 카운터파트 SL 재시도도 실패: {counterpart_key}")
                else:
                    # Counterpart no longer exists — clear the flag
                    pos["counterpart_sl_cancel_failed"] = False

            tp_oid = pos.get("server_tp_order_id")
            sl_oid = pos.get("server_sl_order_id")
            logger.debug(
                f"[{sym}] Sync check: sl_algoId={sl_oid} tp_algoId={tp_oid} "
                f"entry=${pos.get('entry_price', 0):.4f} "
                f"signal={'LONG' if pos.get('signal') == 1 else 'SHORT'}"
            )

            # Check TP order first (the happy path)
            if tp_oid:
                filled = self._check_order_filled(api_sym, tp_oid)
                if filled and filled.get("status") == "FILLED":
                    fill_price = float(filled["avgPrice"])
                    fill_qty = float(filled["executedQty"])

                    # v12.5: PBT partial TP — server closed 50%, transition
                    # to Phase 2 runner instead of fully closing position.
                    if pos.get("server_tp_is_partial") and not pos.get("partial_closed"):
                        logger.info(
                            f"[{sym}] Server TP (PARTIAL) TRIGGERED intra-bar: "
                            f"fill=${fill_price:.4f} qty={fill_qty:.6f}"
                        )
                        self._handle_server_partial_tp_fill(
                            sym,
                            pos,
                            fill_price,
                            fill_qty,
                        )
                        continue

                    logger.info(f"[{sym}] Server TP TRIGGERED intra-bar: fill=${fill_price:.4f} qty={fill_qty:.6f}")
                    # Cancel the counterpart SL (own leg's SL, with retry for straddle pairs)
                    if sl_oid:
                        cfg_vs = getattr(self, "vs_config", None)
                        if cfg_vs is not None and pos.get("is_straddle"):
                            sl_cancelled = False
                            for attempt in range(cfg_vs.sl_cancel_retry_count + 1):
                                if self._cancel_server_stop_loss(api_sym, sl_oid):
                                    sl_cancelled = True
                                    break
                                if attempt < cfg_vs.sl_cancel_retry_count:
                                    time.sleep(cfg_vs.sl_cancel_retry_delay_s)
                            if not sl_cancelled:
                                logger.warning(f"[{sym}] 자체 SL 취소 실패 ({cfg_vs.sl_cancel_retry_count + 1}회 시도)")
                        else:
                            self._cancel_server_stop_loss(api_sym, sl_oid)

                    # ── Step 2.1: Straddle counterpart SL cancel with retry ──
                    counterpart_key = self._get_pair_counterpart(sym) if pos.get("is_straddle") else None
                    if counterpart_key:
                        counterpart = self.strategy._positions.get(counterpart_key)
                        if counterpart:
                            counterpart_sl_oid = counterpart.get("server_sl_order_id")
                            counterpart_api_sym = counterpart_key.split(":")[0]
                            if counterpart_sl_oid:
                                cfg_vs = getattr(self, "vs_config", None)
                                retry_count = cfg_vs.sl_cancel_retry_count if cfg_vs else 0
                                retry_delay = cfg_vs.sl_cancel_retry_delay_s if cfg_vs else 1.0
                                counterpart_cancelled = False
                                for attempt in range(retry_count + 1):
                                    if self._cancel_server_stop_loss(counterpart_api_sym, counterpart_sl_oid):
                                        counterpart_cancelled = True
                                        break
                                    if attempt < retry_count:
                                        time.sleep(retry_delay)
                                if counterpart_cancelled:
                                    logger.info(
                                        f"[{sym}] 카운터파트 SL 취소 성공: {counterpart_key} algoId={counterpart_sl_oid}"
                                    )
                                else:
                                    logger.warning(
                                        f"[{sym}] 카운터파트 SL 취소 실패 ({retry_count + 1}회 시도): {counterpart_key}"
                                    )
                                    counterpart["counterpart_sl_cancel_failed"] = True
                                    self.discord.send(
                                        content=(
                                            f"⚠️ **[{counterpart_key}] 카운터파트 SL 취소 실패** — "
                                            f"다음 사이클에서 재시도"
                                        )
                                    )
                    # ── ReconJournal: record server TP fill ──
                    try:
                        _rj = getattr(self, "_recon_journal", None)
                        if _rj is not None:
                            _rj.record(
                                event_type=ReconEventType.SERVER_TP_FILL,
                                symbol=api_sym,
                                action=f"TP filled @ ${fill_price:.4f}",
                                result="success",
                                details={"fill_price": fill_price, "fill_qty": fill_qty},
                            )
                    except Exception:
                        pass
                    self._record_server_fill(
                        sym,
                        fill_price,
                        fill_qty,
                        "server_tp",
                    )
                    continue
                elif filled and filled.get("status") == "CANCELLED":
                    # v12.3.1: Externally cancelled — clear for bracket repair
                    pos["server_tp_order_id"] = None
                    tp_oid = None
                    logger.warning(f"[{sym}] Server TP was externally cancelled — cleared for bracket repair")

            # Check SL order
            if sl_oid:
                filled = self._check_order_filled(api_sym, sl_oid)
                if filled and filled.get("status") == "FILLED":
                    fill_price = float(filled["avgPrice"])
                    fill_qty = float(filled["executedQty"])
                    logger.info(f"[{sym}] Server SL TRIGGERED intra-bar: fill=${fill_price:.4f} qty={fill_qty:.6f}")
                    # Cancel the counterpart TP (own leg's TP)
                    if tp_oid:
                        self._cancel_server_take_profit(api_sym, tp_oid)

                    # ── Step 2.1: Straddle counterpart SL cancel with retry ──
                    counterpart_key = self._get_pair_counterpart(sym) if pos.get("is_straddle") else None
                    if counterpart_key:
                        counterpart = self.strategy._positions.get(counterpart_key)
                        if counterpart:
                            counterpart_sl_oid = counterpart.get("server_sl_order_id")
                            counterpart_api_sym = counterpart_key.split(":")[0]
                            if counterpart_sl_oid:
                                cfg_vs = getattr(self, "vs_config", None)
                                retry_count = cfg_vs.sl_cancel_retry_count if cfg_vs else 0
                                retry_delay = cfg_vs.sl_cancel_retry_delay_s if cfg_vs else 1.0
                                counterpart_cancelled = False
                                for attempt in range(retry_count + 1):
                                    if self._cancel_server_stop_loss(counterpart_api_sym, counterpart_sl_oid):
                                        counterpart_cancelled = True
                                        break
                                    if attempt < retry_count:
                                        time.sleep(retry_delay)
                                if counterpart_cancelled:
                                    logger.info(
                                        f"[{sym}] 카운터파트 SL 취소 성공: {counterpart_key} algoId={counterpart_sl_oid}"
                                    )
                                else:
                                    logger.warning(
                                        f"[{sym}] 카운터파트 SL 취소 실패 ({retry_count + 1}회 시도): {counterpart_key}"
                                    )
                                    counterpart["counterpart_sl_cancel_failed"] = True
                                    self.discord.send(
                                        content=(
                                            f"⚠️ **[{counterpart_key}] 카운터파트 SL 취소 실패** — "
                                            f"다음 사이클에서 재시도"
                                        )
                                    )
                    # ── ReconJournal: record server SL fill ──
                    try:
                        _rj = getattr(self, "_recon_journal", None)
                        if _rj is not None:
                            _rj.record(
                                event_type=ReconEventType.SERVER_SL_FILL,
                                symbol=api_sym,
                                action=f"SL filled @ ${fill_price:.4f}",
                                result="success",
                                details={"fill_price": fill_price, "fill_qty": fill_qty},
                            )
                    except Exception:
                        pass
                    self._record_server_fill(
                        sym,
                        fill_price,
                        fill_qty,
                        "server_sl",
                    )
                    continue
                elif filled and filled.get("status") == "CANCELLED":
                    # v12.3.1: Externally cancelled — clear for bracket repair
                    pos["server_sl_order_id"] = None
                    sl_oid = None
                    logger.warning(f"[{sym}] Server SL was externally cancelled — cleared for bracket repair")

            # ── r90: Check TRAILING_STOP_MARKET order (Algo API, same as SL/TP) ──
            trail_oid = pos.get("server_trail_order_id")
            if trail_oid and not str(trail_oid).startswith("paper_"):
                trail_status = self._check_trail_order_status(api_sym, str(trail_oid))
                if trail_status and trail_status.get("status") == "FILLED":
                    fill_price = float(trail_status.get("avgPrice", 0))
                    fill_qty = float(trail_status.get("executedQty", 0))
                    logger.info(f"[{sym}] Server TRAIL TRIGGERED: fill=${fill_price:.4f} qty={fill_qty:.6f}")
                    # Cancel orphaned SL
                    if sl_oid:
                        self._cancel_server_stop_loss(api_sym, sl_oid)
                    # Cancel orphaned TP (if any)
                    if tp_oid:
                        self._cancel_server_take_profit(api_sym, tp_oid)
                    # Straddle counterpart SL cancel
                    if pos.get("is_straddle"):
                        counterpart_key = self._get_pair_counterpart(sym)
                        if counterpart_key:
                            counterpart = self.strategy._positions.get(counterpart_key)
                            if counterpart:
                                c_sl = counterpart.get("server_sl_order_id")
                                c_api = counterpart_key.split(":")[0]
                                if c_sl:
                                    self._cancel_server_stop_loss(c_api, c_sl)
                    # ReconJournal
                    try:
                        _rj = getattr(self, "_recon_journal", None)
                        if _rj is not None:
                            _rj.record(
                                event_type=ReconEventType.SERVER_SL_FILL,  # reuse SL event type
                                symbol=api_sym,
                                action=f"Trail filled @ ${fill_price:.4f}",
                                result="success",
                                details={"fill_price": fill_price, "fill_qty": fill_qty, "type": "trail"},
                            )
                    except Exception:
                        pass
                    self._record_server_fill(sym, fill_price, fill_qty, "server_trail")
                    continue
                elif trail_status and trail_status.get("status") == "CANCELLED":
                    pos["server_trail_order_id"] = None
                    trail_oid = None
                    logger.warning(f"[{sym}] Server trail was externally cancelled — cleared for repair")

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
                missing_tp = tp_oid is None and not pos.get("partial_closed", False)
            else:
                missing_tp = tp_oid is None

            if missing_sl or missing_tp:
                quantity = pos["notional"] / pos["entry_price"]
                quantity = self._round_qty(quantity, api_sym)
                # F3: Guard — qty=0 after rounding (very small position) → skip repair
                if quantity <= 0:
                    logger.error(
                        f"[{sym}] Bracket repair SKIPPED: quantity={quantity:.6f} ≤ 0 "
                        f"(notional=${pos['notional']:.4f} / entry=${pos['entry_price']:.6f}). "
                        f"Manual close required."
                    )
                    self.discord.send(
                        content=(
                            f"⚠️ **[{sym}] Bracket repair skipped** — "
                            f"qty=0 after rounding (notional=${pos['notional']:.4f}). "
                            f"Manual close required."
                        )
                    )
                    continue
                signal = pos["signal"]
                entry_price = pos["entry_price"]
                move_size = pos.get("move_size", 0.03)
                stop_mult = pos.get(
                    "entry_stop_mult",
                    self.config.stop_mult,
                )

                if missing_sl:
                    # v12.3.1: Runner needs BE stop, not original stop
                    if pos.get("partial_closed") and self.config.breakeven_after_partial:
                        repair_stop_mult = 0.0001  # BE stop for runner
                    else:
                        repair_stop_mult = stop_mult
                    new_sl = self._place_server_stop_loss(
                        symbol=api_sym,
                        signal=signal,
                        quantity=quantity,
                        entry_price=entry_price,
                        move_size=move_size,
                        stop_mult=repair_stop_mult,
                        position_side=pos.get("position_side"),
                    )
                    if new_sl == "WOULD_TRIGGER":
                        # ═══ FORCE-CLOSE: Price blew past SL level ═══
                        # The stop price is already behind the market.
                        # Close the position immediately at market with
                        # reduceOnly to prevent the "margin insufficient"
                        # error that killed us before.
                        logger.critical(f"[{sym}] BRACKET REPAIR → FORCE CLOSE: Price past SL. Closing at market.")
                        self.discord.send(
                            content=(f"\U0001f6a8 **강제청산** [{sym}] — SL 가격 이탈 감지. 시장가 긴급 청산 실행.")
                        )
                        # v12.3.1: Fetch real price for force close
                        # (0.0 would corrupt PnL if NO_POSITION fallback)
                        _fc_price = entry_price  # safe fallback
                        try:
                            _book = self.client.futures_orderbook_ticker(
                                symbol=api_sym,
                            )
                            _fc_price = float(_book["bidPrice"] if signal == 1 else _book["askPrice"])
                        except Exception:
                            pass
                        closed = self._close_position(
                            sym,
                            "force_sl_breach",
                            _fc_price,
                        )
                        if closed:
                            logger.info(f"[{sym}] FORCE CLOSE SUCCESS — position liquidated at market.")
                        else:
                            logger.critical(f"[{sym}] FORCE CLOSE FAILED — MANUAL INTERVENTION REQUIRED!")
                            self.discord.send(
                                content=(
                                    f"\U0001f6a8\U0001f6a8 **강제청산 실패** "
                                    f"[{sym}] — 수동 청산 필요! 즉시 거래소 확인!"
                                )
                            )
                        continue  # Skip TP repair — position gone
                    elif new_sl:
                        pos["server_sl_order_id"] = new_sl
                        # v12.4.4: Reset ratchet state when runner SL is
                        # re-placed at breakeven. Without this, _server_sl_pct
                        # retains the old ratcheted value, _ratchet_runner_sl
                        # sees insufficient improvement (<0.5%), and skips
                        # re-ratcheting — leaving server SL stuck at breakeven.
                        if repair_stop_mult < 0.001:  # BE stop for runner
                            pos["_server_sl_pct"] = 0.0
                        logger.info(f"[{sym}] BRACKET REPAIR: SL placed algoId={new_sl}")
                        # ── ReconJournal: record bracket repair success ──
                        try:
                            _rj = getattr(self, "_recon_journal", None)
                            if _rj is not None:
                                _rj.record(
                                    event_type=ReconEventType.BRACKET_REPAIR,
                                    symbol=api_sym,
                                    action=f"SL placed algoId={new_sl}",
                                    result="success",
                                )
                        except Exception:
                            pass
                    else:
                        # Track consecutive SL repair failures.
                        # After MAX_REPAIR_FAILS attempts the position is likely
                        # in a bad state (API issues, corrupt symbol, etc.) —
                        # auto-close to cap the unprotected exposure window.
                        MAX_REPAIR_FAILS = 3
                        fail_count = pos.get("_repair_fail_count", 0) + 1
                        pos["_repair_fail_count"] = fail_count
                        logger.error(
                            f"[{sym}] BRACKET REPAIR FAILED ({fail_count}/{MAX_REPAIR_FAILS}): "
                            f"Could not place SL — position UNPROTECTED"
                        )
                        # ── ReconJournal: record bracket repair failure ──
                        try:
                            _rj = getattr(self, "_recon_journal", None)
                            if _rj is not None:
                                _rj.record(
                                    event_type=ReconEventType.BRACKET_REPAIR_FAIL,
                                    symbol=api_sym,
                                    action=f"SL placement failed ({fail_count}/{MAX_REPAIR_FAILS})",
                                    result="failure",
                                )
                        except Exception:
                            pass
                        if fail_count >= MAX_REPAIR_FAILS:
                            logger.critical(
                                f"[{sym}] BRACKET REPAIR EXHAUSTED ({MAX_REPAIR_FAILS} failures) — "
                                f"AUTO CLOSING to prevent extended SL exposure"
                            )
                            self.discord.send(
                                content=(
                                    f"\U0001f6a8 **수익보전 자동청산** [{sym}] — "
                                    f"SL 배치 {MAX_REPAIR_FAILS}회 연속 실패. 포지션 자동 종료."
                                )
                            )
                            _fc_price = entry_price
                            try:
                                _book = self.client.futures_orderbook_ticker(symbol=api_sym)
                                _fc_price = float(_book["bidPrice"] if signal == 1 else _book["askPrice"])
                            except Exception:
                                pass
                            closed = self._close_position(sym, "bracket_repair_exhausted", _fc_price)
                            if closed:
                                logger.info(f"[{sym}] AUTO CLOSE SUCCESS — unprotected position liquidated.")
                            else:
                                logger.critical(
                                    f"[{sym}] AUTO CLOSE FAILED — MANUAL INTERVENTION REQUIRED! "
                                    f"Position has no SL and could not be closed automatically."
                                )
                                self.discord.send(
                                    content=(
                                        f"\U0001f6a8\U0001f6a8 **자동청산 실패** [{sym}] — "
                                        f"수동 청산 필요! 즉시 거래소 확인!"
                                    )
                                )
                            continue  # Skip TP repair — position gone or in critical state

                if missing_tp:
                    # v12.5: PBT mode uses 50% qty server TP (pre-partial only).
                    # Non-PBT: full qty server TP for intra-bar capture.
                    if self.config.partial_exit_enabled:
                        tp_qty = self._round_qty(
                            quantity * self.config.partial_exit_fraction,
                            api_sym,
                        )
                    else:
                        tp_qty = quantity
                    # v14.0: Use per-position TP (not RubberBand default).
                    # VolStraddle stores tp in position dict (VR=12%, MA=8%).
                    repair_tp = pos.get("tp", self.config.target_retrace)
                    new_tp = self._place_server_take_profit(
                        symbol=api_sym,
                        signal=signal,
                        quantity=tp_qty,
                        entry_price=entry_price,
                        move_size=move_size,
                        target_retrace=repair_tp,
                        position_side=pos.get("position_side"),
                    )
                    if new_tp:
                        pos["server_tp_order_id"] = new_tp
                        if self.config.partial_exit_enabled:
                            pos["server_tp_is_partial"] = True
                        logger.info(
                            f"[{sym}] BRACKET REPAIR: TP placed "
                            f"algoId={new_tp} "
                            f"{'(50% partial)' if self.config.partial_exit_enabled else ''}"
                        )
                    else:
                        logger.warning(f"[{sym}] BRACKET REPAIR: TP failed — client-side TP still active")

                if missing_sl or missing_tp:
                    self._save_state()
                    sl_id = pos.get("server_sl_order_id")
                    tp_id = pos.get("server_tp_order_id")
                    logger.info(
                        f"[{sym}] Bracket repair complete: "
                        f"SL={'OK' if sl_id else 'MISSING'} "
                        f"(algoId={sl_id}) "
                        f"TP={'OK' if tp_id else 'MISSING'} "
                        f"(algoId={tp_id})"
                    )

            # ── r90: Trail bracket repair ──
            vs_cfg = getattr(self, "vs_config", None)
            if vs_cfg and getattr(vs_cfg, "trail_server_native", False):
                trail_oid = pos.get("server_trail_order_id")
                if trail_oid is None:
                    quantity = pos["notional"] / pos["entry_price"]
                    quantity = self._round_qty(quantity, api_sym)
                    if quantity > 0:
                        signal = pos["signal"]
                        entry_price = pos["entry_price"]
                        strategy_type = pos.get("strategy_type", "vr_squeeze")
                        cr, ap, cs = self._get_trail_params(strategy_type, signal, entry_price)
                        new_trail = self._place_trailing_stop_market(
                            symbol=api_sym,
                            side=cs,
                            quantity=quantity,
                            callback_rate=cr,
                            activation_price=ap,
                            position_side=pos.get("position_side"),
                        )
                        if new_trail:
                            pos["server_trail_order_id"] = new_trail
                            logger.info(f"[{sym}] BRACKET REPAIR: Trail placed orderId={new_trail} callbackRate={cr}%")
                            self._save_state()
                        else:
                            logger.warning(f"[{sym}] BRACKET REPAIR: Trail failed")

    def _reconcile_positions(self: "ScalperProtocol") -> None:
        """Reconcile internal position state with exchange reality.

        v12.5.1: Called each cycle AFTER _sync_server_orders.
        Detects ghost positions (closed on exchange but still in internal
        state) and quantity mismatches (ADL, manual intervention).
        """
        # 사이클 시작 시 balance 캐시 초기화 — _get_cycle_balance()가 새 값을 1회 fetch
        self._cycle_balance = None

        if self.paper_mode:
            return

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
        exchange_pos_detail: dict[str, dict] = {}  # full position data for journaling
        cached_notionals: dict[str, float] = {}  # {key: notional_usd} for -2027 pre-check
        for p in exchange_positions:
            amt = float(p.get("positionAmt", "0"))
            if amt != 0:
                pos_side = p.get("positionSide", "BOTH")
                if pos_side in ("LONG", "SHORT"):
                    key = f"{p['symbol']}:{pos_side}"
                else:
                    key = p["symbol"]
                exchange_map[key] = amt
                exchange_pos_detail[key] = p
                entry_px = float(p.get("entryPrice", 0) or 0)
                cached_notionals[key] = abs(amt) * entry_px

        # Cache for cumulative notional pre-check in _open_position()
        self._cached_exchange_notionals: dict[str, float] = cached_notionals

        # Check each internal position against exchange
        for sym in list(self.strategy.active_positions):
            pos = self.strategy._positions.get(sym)
            if not pos:
                continue

            # Build lookup key: compound (BTCUSDT:LONG) or plain (BTCUSDT).
            # Strip any compound suffix from pos['symbol'] defensively.
            raw_sym = pos.get("symbol", sym)
            api_sym = raw_sym.split(":")[0] if ":" in raw_sym else raw_sym
            pos_side = pos.get("position_side")
            if pos_side in ("LONG", "SHORT"):
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
                self.discord.send(
                    content=(
                        f"⚠️ **[{sym}] Ghost position detected** — "
                        f"Position missing on exchange. Cleaning up internal state."
                    )
                )
                # ── ReconJournal: record ghost detection ──
                try:
                    _rj = getattr(self, "_recon_journal", None)
                    if _rj is not None:
                        _rj.record(
                            event_type=ReconEventType.GHOST_DETECTED,
                            symbol=api_sym,
                            action="removed from state",
                            result="success",
                            details={
                                "entry_price": pos.get("entry_price"),
                                "bars_held": pos.get("bars_held"),
                                "position_side": pos.get("position_side"),
                            },
                        )
                except Exception:
                    pass
                # ── Ghost Fill Audit (P1) ──────────────────────────
                # Retrieve and log ALL closing fills so we can diagnose
                # *why* _sync_server_orders missed the fill.
                fill_price = pos["entry_price"]
                audit_fills: list[dict] = []
                try:
                    recent = self.client.futures_account_trades(
                        symbol=api_sym,
                        limit=50,
                    )
                    close_side = "SELL" if pos["signal"] == 1 else "BUY"
                    for t in reversed(recent):
                        if t.get("side") == close_side and float(t.get("realizedPnl", "0")) != 0:
                            audit_fills.append(
                                {
                                    "orderId": t.get("orderId"),
                                    "price": float(t["price"]),
                                    "qty": float(t.get("qty", "0")),
                                    "realizedPnl": float(t.get("realizedPnl", "0")),
                                    "commission": float(t.get("commission", "0")),
                                    "commissionAsset": t.get("commissionAsset", ""),
                                    "time": t.get("time"),  # epoch ms
                                    "maker": t.get("maker", False),
                                    "buyer": t.get("buyer", False),
                                }
                            )
                    if audit_fills:
                        fill_price = audit_fills[0]["price"]  # most recent
                except Exception as e:
                    logger.warning(f"[{sym}] Ghost audit: trade fetch failed: {e}")

                # Also try to find the algo order that caused the fill
                algo_audit: dict | None = None
                if audit_fills:
                    try:
                        all_orders = self.client.futures_get_all_orders(
                            symbol=api_sym,
                            limit=20,
                        )
                        fill_order_id = audit_fills[0]["orderId"]
                        for o in reversed(all_orders):
                            if o.get("orderId") == fill_order_id:
                                algo_audit = {
                                    "orderId": o.get("orderId"),
                                    "clientOrderId": o.get("clientOrderId", ""),
                                    "type": o.get("type", ""),
                                    "side": o.get("side", ""),
                                    "origType": o.get("origType", ""),
                                    "status": o.get("status", ""),
                                    "stopPrice": float(o.get("stopPrice", "0")),
                                    "price": float(o.get("price", "0")),
                                    "updateTime": o.get("updateTime"),
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
                        ts_str = ""
                        if af["time"]:
                            ts_str = datetime.fromtimestamp(
                                af["time"] / 1000,
                                tz=timezone.utc,
                            ).strftime("%Y-%m-%d %H:%M:%S UTC")
                        logger.error(
                            f"  fill[{i}] orderId={af['orderId']} "
                            f"price={af['price']:.4f} qty={af['qty']:.6f} "
                            f"realizedPnl={af['realizedPnl']:+.4f} "
                            f"commission={af['commission']:.6f} "
                            f"{'MAKER' if af['maker'] else 'TAKER'} "
                            f"time={ts_str}"
                        )
                else:
                    logger.error("  [NO MATCHING FILLS FOUND in last 50 trades]")

                if algo_audit:
                    logger.error(
                        f"  trigger_order: id={algo_audit['orderId']} "
                        f"type={algo_audit['origType']} "
                        f"status={algo_audit['status']} "
                        f"stopPrice={algo_audit['stopPrice']:.4f} "
                        f"clientId={algo_audit['clientOrderId']}"
                    )

                # Check for internal state clues: were SL/TP algo IDs set?
                sl_oid = pos.get("server_sl_order_id")
                tp_oid = pos.get("server_tp_order_id")
                logger.error(
                    f"  internal_state: server_sl_id={sl_oid} "
                    f"server_tp_id={tp_oid} "
                    f"pbt_phase={pos.get('pbt_phase', 'N/A')} "
                    f"server_sl_pct={pos.get('_server_sl_pct', 'N/A')}"
                )

                # Discord notification with audit summary
                audit_desc = "No matching fills found on exchange."
                if audit_fills:
                    af0 = audit_fills[0]
                    ts_str = ""
                    if af0["time"]:
                        ts_str = datetime.fromtimestamp(
                            af0["time"] / 1000,
                            tz=timezone.utc,
                        ).strftime("%H:%M:%S UTC")
                    audit_desc = (
                        f"Fill: ${af0['price']:.4f} × {af0['qty']:.6f} "
                        f"@ {ts_str}\n"
                        f"PnL: {af0['realizedPnl']:+.4f} USDT\n"
                        f"Order: {algo_audit['origType'] if algo_audit else '?'} "
                        f"(id={af0['orderId']})"
                    )
                self.discord.send(
                    embed={
                        "title": f"👻 [{sym}] Ghost Fill Audit",
                        "color": 0xFF6D00,
                        "description": audit_desc,
                        "fields": [
                            {"name": "Entry", "value": f"${pos['entry_price']:.4f}", "inline": True},
                            {"name": "Bars Held", "value": str(pos["bars_held"]), "inline": True},
                            {"name": "SL AlgoID", "value": str(sl_oid or "None"), "inline": True},
                            {"name": "TP AlgoID", "value": str(tp_oid or "None"), "inline": True},
                            {"name": "Total Fills", "value": str(len(audit_fills)), "inline": True},
                        ],
                    }
                )
                # v14.0.1: Cancel orphaned counterpart algo orders.
                # Without this, the remaining SL or TP stays on Binance
                # and could trigger on a future position for the same symbol.
                if sl_oid:
                    self._cancel_server_stop_loss(api_sym, sl_oid)
                if tp_oid:
                    self._cancel_server_take_profit(api_sym, tp_oid)
                self._record_server_fill(sym, fill_price, 0, "ghost_reconcile")
                continue

            # Quantity mismatch check
            exchange_qty = abs(exchange_map[lookup_key])
            internal_qty = pos["notional"] / pos["entry_price"]
            qty_diff_pct = abs(exchange_qty - internal_qty) / max(internal_qty, 1e-8) * 100
            if qty_diff_pct > self.config.reconciliation_qty_tolerance_pct:  # configurable mismatch threshold
                logger.warning(
                    f"[{sym}] RECONCILIATION: Qty mismatch! "
                    f"Internal={internal_qty:.6f} Exchange={exchange_qty:.6f} "
                    f"Diff={qty_diff_pct:.1f}%. Correcting internal state."
                )
                # Update internal notional to match exchange reality
                pos["notional"] = exchange_qty * pos["entry_price"]

        # ── Phase 2: Reverse ghost — exchange has position not in state ──
        # Build set of all keys already tracked internally.
        tracked_keys: set[str] = set()
        for sym in self.strategy.active_positions:
            pos = self.strategy._positions.get(sym)
            if not pos:
                continue
            raw_sym = pos.get("symbol", sym)
            api_sym = raw_sym.split(":")[0] if ":" in raw_sym else raw_sym
            pos_side = pos.get("position_side")
            if pos_side in ("LONG", "SHORT"):
                tracked_keys.add(f"{api_sym}:{pos_side}")
            else:
                tracked_keys.add(api_sym)

        # ORPHAN_AUTO_CLOSE=true 환경변수로 구버전(자동청산) 동작 복원 가능 (탈출구).
        # 기본값은 alert-only — 운영자가 !adopt 또는 !close-orphan 명령으로 처리.
        _auto_close_enabled = os.getenv("ORPHAN_AUTO_CLOSE", "").lower() == "true"

        for ex_key, ex_amt in exchange_map.items():
            if ex_key in tracked_keys:
                continue
            # Reverse ghost: exchange has a position with no internal state.
            parts = ex_key.split(":")
            api_sym = parts[0]
            pos_side_str = parts[1] if len(parts) == 2 else None

            if not _auto_close_enabled:
                # Alert-only — operator must decide via !adopt or !close-orphan
                logger.error(
                    f"[{ex_key}] REVERSE GHOST: Position on exchange has no internal "
                    f"state (no SL/TP protection). ALERT ONLY — use !adopt or !close-orphan."
                )
                self.discord.send(
                    content=(
                        f"🔍 **[{ex_key}] 고아 포지션 감지** — "
                        f"거래소에 보호 없는 포지션 존재 (qty={abs(ex_amt):.4f}).\n"
                        f"`!adopt {ex_key}` 또는 `!close-orphan {ex_key}` 명령으로 처리해주세요."
                    )
                )
                try:
                    _rj = getattr(self, "_recon_journal", None)
                    if _rj is not None:
                        _rj.record(
                            event_type=ReconEventType.REVERSE_GHOST,
                            symbol=api_sym,
                            action="orphan_alert_only (no auto-close)",
                            result="alert_sent",
                            details={"position_side": pos_side_str, "qty": abs(ex_amt)},
                        )
                except Exception:
                    pass
                continue

            # ORPHAN_AUTO_CLOSE=true: original auto-close behavior (escape hatch)
            close_side = "SELL" if pos_side_str == "LONG" else "BUY"
            logger.error(
                f"[{ex_key}] REVERSE GHOST: Position on exchange has no internal "
                f"state (no SL/TP protection). Auto-closing qty={abs(ex_amt):.6f}."
            )
            self.discord.send(
                content=(
                    f"⚠️ **[{ex_key}] Reverse ghost detected** — "
                    f"Unmanaged exchange position with no SL/TP. "
                    f"Auto-closing qty={abs(ex_amt):.4f} via MARKET order."
                )
            )
            try:
                order_kwargs: dict = dict(
                    symbol=api_sym,
                    side=close_side,
                    type="MARKET",
                    quantity=abs(ex_amt),
                )
                if pos_side_str in ("LONG", "SHORT"):
                    order_kwargs["positionSide"] = pos_side_str
                self.client.futures_create_order(**order_kwargs)
                logger.info(f"[{ex_key}] Reverse ghost auto-closed: {close_side} {abs(ex_amt):.6f} MARKET.")
                # ── TradeJournal: record reverse ghost in main journal.sqlite ──
                try:
                    _tj = getattr(self, "_journal", None)
                    if _tj is not None:
                        _pos_detail = exchange_pos_detail.get(ex_key, {})
                        _entry_px = float(_pos_detail.get("entryPrice", 0) or 0)
                        _mark_px = float(_pos_detail.get("markPrice", 0) or 0)
                        _upnl = float(_pos_detail.get("unRealizedProfit", 0) or 0)
                        _notional = abs(ex_amt) * _entry_px if _entry_px else 0
                        _pnl_pct = (_upnl / _notional * 100) if _notional else 0
                        _signal = 1 if pos_side_str == "LONG" else -1
                        _tj.record(
                            {
                                "symbol": api_sym,
                                "signal": _signal,
                                "strategy_type": "unknown",
                                "entry_price": _entry_px,
                                "exit_price": _mark_px,
                                "pnl_pct": _pnl_pct,
                                "pnl_usd": _upnl,
                                "mfe_pct": None,
                                "mae_pct": None,
                                "notional": _notional,
                                "bars_held": 0,
                                "reason": "reconciliation_close",
                                "time": datetime.now(timezone.utc).isoformat(),
                                "entry_time": None,
                                "balance_before": self._get_cycle_balance(),
                                "balance_after": self._get_cycle_balance(),
                                "straddle_pair_id": None,
                            }
                        )
                        logger.info(f"[{ex_key}] Reverse ghost journaled: PnL=${_upnl:+.2f}")
                except Exception as _je:
                    logger.warning(f"[{ex_key}] Reverse ghost journal failed: {_je}")
                # ── ReconJournal: record reverse ghost close ──
                try:
                    _rj = getattr(self, "_recon_journal", None)
                    if _rj is not None:
                        _rj.record(
                            event_type=ReconEventType.REVERSE_GHOST_CLOSED,
                            symbol=api_sym,
                            action=f"auto-closed {close_side} qty={abs(ex_amt):.6f}",
                            result="success",
                            details={"position_side": pos_side_str},
                        )
                except Exception:
                    pass
            except Exception as exc:
                logger.error(f"[{ex_key}] Reverse ghost auto-close FAILED: {exc}. Manual intervention required!")
                self.discord.send(
                    content=(
                        f"🚨 **[{ex_key}] Reverse ghost AUTO-CLOSE FAILED** — "
                        f"`{exc}`. Manual close required immediately!"
                    )
                )
                # ── ReconJournal: record failed reverse ghost close ──
                try:
                    _rj = getattr(self, "_recon_journal", None)
                    if _rj is not None:
                        _rj.record(
                            event_type=ReconEventType.REVERSE_GHOST,
                            symbol=api_sym,
                            action=f"auto-close FAILED: {exc}",
                            result="failure",
                            details={"position_side": pos_side_str},
                        )
                except Exception:
                    pass

        # ── Phase 3: Straddle pair integrity check ──
        # NOTE: 이 스트래들 고아 자동-청산 (VS_ORPHAN_AUTO_CLOSE_CYCLES)은 위의
        # ORPHAN_AUTO_CLOSE (거래소-전용 reverse ghost 자동청산)와 별개입니다.
        # 여기서는 내부 상태에서 카운터파트가 사라진 경우만 처리합니다.
        _cfg_vs = getattr(self, "vs_config", None)
        for sym in list(self.strategy.active_positions):
            pos = self.strategy._positions.get(sym)
            if not pos or not pos.get("is_straddle") or not pos.get("straddle_pair_id"):
                continue

            counterpart_key = self._get_pair_counterpart(sym)
            if counterpart_key is not None:
                # 페어 온전함 — 고아 추적 초기화
                pos.pop("straddle_orphan_since_cycle", None)
                continue

            # 카운터파트 없음 — 합법적 청산 여부 확인
            if pos.get("counterpart_legitimately_closed"):
                # 카운터파트가 server SL/TP로 정상 청산됨 — 생존 레그는 추세를 타도록 놔둠
                pos.pop("straddle_orphan_since_cycle", None)
                continue

            # 진짜 고아 — 카운터파트 없고 합법적 청산 기록도 없음
            if pos.get("straddle_orphan_since_cycle") is None:
                pos["straddle_orphan_since_cycle"] = self._cycle_count
                grace = _cfg_vs.orphan_auto_close_cycles if _cfg_vs else 4
                logger.warning(f"[{sym}] 스트래들 레그 고아 감지 — 카운터파트 없음. {grace}사이클 후 자동 청산 예정.")
                self.discord.send(
                    content=(f"🔍 **[{sym}] 스트래들 고아 감지** — 카운터파트 부재. {grace}사이클 후 자동 청산 예정.")
                )

            orphan_grace = _cfg_vs.orphan_auto_close_cycles if _cfg_vs else 4
            orphan_cycles = self._cycle_count - pos["straddle_orphan_since_cycle"]
            if orphan_cycles >= orphan_grace:
                logger.warning(f"[{sym}] 고아 스트래들 레그 자동 청산 — {orphan_cycles}사이클 동안 헤지 없음.")
                self.discord.send(
                    content=(
                        f"⚠️ **[{sym}] 고아 스트래들 자동 청산** — {orphan_cycles}사이클 동안 헤지 없음. 자동 청산 실행."
                    )
                )
                # 현재 포지션의 마크 가격 추정 (entry_price 사용)
                raw_sym = pos.get("symbol", sym)
                api_sym = raw_sym.split(":")[0] if ":" in raw_sym else raw_sym
                close_price = pos.get("entry_price", 0.0)
                try:
                    _book = self.client.futures_orderbook_ticker(symbol=api_sym)
                    close_price = float(_book["bidPrice"] if pos["signal"] == 1 else _book["askPrice"])
                except Exception:
                    pass
                closed = self._close_position(sym, "straddle_orphan_auto_close", close_price)
                if not closed:
                    logger.error(f"[{sym}] 고아 스트래들 자동 청산 실패 — 수동 개입 필요!")
            else:
                # 유예 기간 중 — SL 브라켓 확인
                if not pos.get("server_sl_order_id"):
                    logger.warning(f"[{sym}] 고아 스트래들 레그에 SL 브라켓 없음 — bracket repair 필요")

    def _handle_server_partial_tp_fill(
        self: "ScalperProtocol",
        symbol: str,
        pos: dict,
        fill_price: float,
        fill_qty: float,
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
        if pos.get("partial_closed"):
            logger.warning(f"[{symbol}] Server partial TP fill skipped — client already processed partial close")
            pos["server_tp_order_id"] = None
            pos["server_tp_is_partial"] = False
            return

        # Extract Binance symbol from compound key
        api_sym = pos.get("symbol", symbol.split(":")[0])

        # 1. Cancel old SL (100% qty, no longer valid)
        sl_oid = pos.get("server_sl_order_id")
        if sl_oid:
            self._cancel_server_stop_loss(api_sym, sl_oid)
            pos["server_sl_order_id"] = None

        # 2. PnL on the partial portion
        if pos["signal"] == 1:
            pnl_pct = (fill_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - fill_price) / pos["entry_price"]

        partial_notional = fill_qty * pos["entry_price"]
        pnl_usd = pnl_pct * partial_notional
        cost = partial_notional * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        # 3. Update position for runner mode
        full_quantity = pos["notional"] / pos["entry_price"]
        remaining_qty_exact = full_quantity - fill_qty
        pos["notional"] = remaining_qty_exact * pos["entry_price"]
        pos["partial_closed"] = True
        pos["_server_partial_tp_filled"] = True  # v12.5 race guard
        pos["server_tp_order_id"] = None
        pos["server_tp_is_partial"] = False
        pos["highest_since_tp1"] = pnl_pct if pnl_pct > 0 else 0.0
        pos["trail_floor"] = 0.0  # breakeven floor
        pos["best_pnl"] = max(pos.get("best_pnl", 0.0), pnl_pct)

        # 4. Place BE SL for remaining qty
        remaining_qty = self._round_qty(
            pos["notional"] / pos["entry_price"],
            api_sym,
        )
        if remaining_qty > 0 and not self.paper_mode:
            new_sl_oid = self._place_server_stop_loss(
                symbol=api_sym,
                signal=pos["signal"],
                quantity=remaining_qty,
                entry_price=pos["entry_price"],
                move_size=pos["move_size"],
                stop_mult=0.0001,  # Break-even stop
                position_side=pos.get("position_side"),
            )
            pos["server_sl_order_id"] = new_sl_oid
            # Reset ratchet state for fresh runner
            pos["_server_sl_pct"] = 0.0

        # Paper mode cost handling
        if self.paper_mode:
            paper_exit_cost = partial_notional * self.config.cost_per_side
            self._paper_balance += pnl_usd - paper_exit_cost

        # 5. Trade record (partial)
        trade = {
            "symbol": api_sym,
            "signal": pos["signal"],
            "entry_price": pos["entry_price"],
            "exit_price": fill_price,
            "pnl_pct": pnl_pct * 100,
            "pnl_usd": net_pnl,
            "notional": partial_notional,
            "bars_held": pos["bars_held"],
            "reason": "server_partial_tp",
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy_type": pos.get("strategy_type", "vr_squeeze"),
            "mfe_pct": pos.get("best_pnl", 0.0) * 100,
            "mae_pct": pos.get("worst_pnl", 0.0) * 100,
            "entry_time": pos.get("opened_at"),
            "balance_before": pos.get("balance_before") or self._get_cycle_balance(),
            "balance_after": self._get_cycle_balance(),
            "straddle_pair_id": pos.get("straddle_pair_id"),
        }
        self._trade_history.append(trade)

        _tj = getattr(self, "_journal", None)
        if _tj is not None:
            try:
                _tj.record(trade)
            except Exception as _je:
                logger.warning(f"[{symbol}] Journal write failed in _handle_server_partial_tp_fill: {_je}")

        self._save_state()

        fraction = self.config.partial_exit_fraction
        logger.info(
            f"[{symbol}] SERVER PARTIAL TP ({fraction * 100:.0f}%): "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct * 100:+.2f}% (${net_pnl:+.2f}) "
            f"→ remaining ${pos['notional']:.2f}, BE stop active"
        )
        _dir = "LONG ↑" if pos["signal"] == 1 else "SHORT ↓"
        _r_sl = "🟢 서버 (BE)" if pos.get("server_sl_order_id") else "🟡 클라이언트"
        self.discord.send(
            embed={
                "title": f"🔒 {symbol} {_dir} | 서버 TP 부분청산 {fraction * 100:.0f}%",
                "color": 0x4CAF50,
                "description": (
                    "서버 TP가 인트라바에서 50% 부분청산 완료.\n나머지 50%는 손익분기점 스톱 + ATR 트레일링."
                ),
                "fields": [
                    {"name": "진입가", "value": f"${pos['entry_price']:.4f}", "inline": True},
                    {"name": "TP 체결가", "value": f"${fill_price:.4f}", "inline": True},
                    {"name": "수익률", "value": f"{pnl_pct * 100:+.2f}%", "inline": True},
                    {"name": "확보금액", "value": f"${net_pnl:+.2f}", "inline": True},
                    {"name": "남은 포지션", "value": f"${pos['notional']:.2f}", "inline": True},
                    {"name": "남은 SL", "value": _r_sl, "inline": True},
                ],
            }
        )

    def _record_server_fill(
        self: "ScalperProtocol",
        symbol: str,
        fill_price: float,
        fill_qty: float,
        reason: str,
    ) -> None:
        """Record a trade closed by a server-side order (TP or SL).

        Removes the position from internal tracking and logs the PnL.
        """
        # ── Step 2.2: Straddle pair context before position is removed ──
        pair_id = None
        counterpart_key_before: str | None = None
        if self.strategy._positions.get(symbol, {}).get("is_straddle"):
            counterpart_key_before = self._get_pair_counterpart(symbol)
            pair_id = self.strategy._positions.get(symbol, {}).get("straddle_pair_id")

        pos = self.strategy.close_position(symbol)
        if not pos:
            return

        # Extract Binance symbol from compound key for trade records
        api_sym = pos.get("symbol", symbol.split(":")[0])

        # PnL calculation
        if pos["signal"] == 1:
            pnl_pct = (fill_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - fill_price) / pos["entry_price"]

        # v12.3.2: Use entry-price-based notional for PnL, not exit-price.
        # fill_qty * fill_price overstates winners and understates losers
        # by a factor of fill_price/entry_price (~3-5% error per trade).
        notional = pos["notional"]
        pnl_usd = pnl_pct * notional
        # v12.3.2: Always charge round_trip_cost for backtest parity.
        # Both partial and runner portions must account for their share
        # of entry + exit costs. The backtest delta model charges
        # |delta| * cost_per_side per transition, totaling round_trip
        # over the full lifecycle.
        cost = notional * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        _now = datetime.now(timezone.utc).isoformat()
        trade = {
            "symbol": api_sym,
            "signal": pos["signal"],
            "entry_price": pos["entry_price"],
            "exit_price": fill_price,
            "pnl_pct": pnl_pct * 100,
            "pnl_usd": net_pnl,
            "notional": notional,
            "bars_held": pos["bars_held"],
            "reason": reason,
            "time": _now,
            "strategy_type": pos.get("strategy_type", "vr_squeeze"),
            "mfe_pct": pos.get("best_pnl", 0.0) * 100,
            "mae_pct": pos.get("worst_pnl", 0.0) * 100,
            "entry_time": pos.get("opened_at"),
            "balance_before": pos.get("balance_before") or self._get_cycle_balance(),
            "balance_after": self._get_cycle_balance(),
            "straddle_pair_id": pair_id,
        }
        self._trade_history.append(trade)

        # Write to persistent journal (dashboard reads from here)
        _tj = getattr(self, "_journal", None)
        if _tj is not None:
            try:
                _tj.record(trade)
            except Exception as _je:
                logger.warning(f"[{symbol}] Journal write failed in _record_server_fill: {_je}")

        # v12.3.1: Save state immediately after recording server fill.
        # Without this, a crash before end-of-cycle _save_state() creates
        # a ghost position on restart (position gone on exchange but
        # still in our state file).
        self._save_state()

        # ── Step 2.2: Straddle pair context logging ──
        if pair_id:
            if counterpart_key_before:
                # Counterpart still open — pair half-closed
                counterpart_pos = self.strategy._positions.get(counterpart_key_before)
                if counterpart_pos:
                    # Verify counterpart brackets are intact after this leg's close
                    counterpart_has_sl = bool(counterpart_pos.get("server_sl_order_id"))
                    counterpart_has_tp = bool(counterpart_pos.get("server_tp_order_id"))
                    if not counterpart_has_sl:
                        logger.warning(
                            f"[{symbol}] 스트래들 반-청산 후 카운터파트 SL 없음: "
                            f"{counterpart_key_before} — bracket repair 필요"
                        )
                    logger.info(
                        f"[{symbol}] 스트래들 페어 반-청산 (pair_id={pair_id}): "
                        f"이 다리 PnL={net_pnl:+.2f}$, "
                        f"카운터파트 {counterpart_key_before} 아직 오픈 "
                        f"(SL={'OK' if counterpart_has_sl else '없음'} "
                        f"TP={'OK' if counterpart_has_tp else '없음'})"
                    )
                    # 카운터파트에 합법적 청산 플래그 설정 — Phase 3에서 survivor로 인식
                    counterpart_pos["counterpart_legitimately_closed"] = True
                    logger.info(
                        f"[{symbol}] 카운터파트 {counterpart_key_before}: "
                        f"legitimate_closed 플래그 설정 (server fill로 청산됨)"
                    )
                else:
                    logger.info(f"[{symbol}] 스트래들 페어 완전 청산 (pair_id={pair_id}): 양 다리 모두 청산됨")
            else:
                logger.info(f"[{symbol}] 스트래들 페어 완전 청산 (pair_id={pair_id}): 양 다리 모두 청산됨")

        emoji = "🎯" if reason == "server_tp" else "🛑"
        logger.info(
            f"[{symbol}] {emoji} {reason.upper()} FILLED: "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct * 100:+.2f}% (${net_pnl:+.2f}) "
            f"held={pos['bars_held']} bars"
        )
        cum_pnl = sum(t["pnl_usd"] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins_total = sum(1 for t in self._trade_history if t["pnl_usd"] > 0)
        wr = wins_total / n_trades * 100 if n_trades > 0 else 0
        hold_time = pos["bars_held"] * 15

        _dir = "LONG ↑" if pos["signal"] == 1 else "SHORT ↓"
        _fill_type = "서버 TP 체결 (바 중간 익절)" if reason == "server_tp" else "서버 SL 체결 (안전장치)"
        n_remaining = self.strategy.n_active
        self.discord.send(
            embed={
                "title": f"{emoji} {symbol} {_dir} | 서버체결",
                "color": 0x00C853 if net_pnl > 0 else 0xFF1744,
                "description": f"{_fill_type}",
                "fields": [
                    {"name": "진입가", "value": f"${pos['entry_price']:.4f}", "inline": True},
                    {"name": "체결가", "value": f"${fill_price:.4f}", "inline": True},
                    {"name": "보유시간", "value": f"{hold_time}분 ({pos['bars_held']}바)", "inline": True},
                    {"name": "수익률", "value": f"{pnl_pct * 100:+.2f}%", "inline": True},
                    {"name": "순수익", "value": f"${net_pnl:+.2f}", "inline": True},
                    {"name": "체결방식", "value": "🟢 거래소(서버)", "inline": True},
                    {
                        "name": "── 세션 현황 ──",
                        "value": (
                            f"누적 PnL: ${cum_pnl:+,.2f} | {n_trades}건 | 승률 {wr:.0f}%\n남은 포지션: {n_remaining}개"
                        ),
                        "inline": False,
                    },
                ],
                "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
            }
        )

    # ==================== Position Management ====================
