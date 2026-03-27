"""Discord bot command handlers.

Mixin class for processing interactive commands received via Discord
(!status, !close, !pause, !resume, !stop, !pnl, !blacklist).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantbox.core.protocol import ScalperProtocol

from quantbox.core.strategy_names import VR_DIRECTIONAL
from quantbox.core.types import to_api_symbol
from quantbox.notifications.bot import BotCommand, CommandType

logger = logging.getLogger("Scalper")


class CommandsMixin:
    """Mixin: Discord bot command processing and handlers."""

    def _process_bot_commands(self: "ScalperProtocol") -> None:
        """Poll the bot command queue and handle all pending commands.

        Called at the start of each cycle. Non-blocking.
        """
        if self._bot_runner is None:
            return

        for cmd in self._bot_runner.get_all_commands():
            try:
                result = self._handle_command(cmd)
                cmd.set_response(result)
            except Exception as e:
                logger.error(f"Bot command error: {e}")
                cmd.set_response(f"❌ Internal error: {e}")

    def _handle_command(self: "ScalperProtocol", cmd: BotCommand) -> str:
        """Dispatch a BotCommand to the appropriate handler.

        Args:
            cmd: The parsed bot command.

        Returns:
            Response text to send back to Discord.
        """
        handlers = {
            CommandType.STATUS: lambda: self._cmd_status(),
            CommandType.CLOSE_ALL: lambda: self._cmd_close_all(),
            CommandType.CLOSE_SLOT: lambda: self._cmd_close_slot(
                cmd.args.get("slot_id", -1) if cmd.args else -1,
            ),
            CommandType.PAUSE: lambda: self._cmd_pause(),
            CommandType.RESUME: lambda: self._cmd_resume(),
            CommandType.STOP: lambda: self._cmd_stop(),
            CommandType.PNL: lambda: self._cmd_pnl(),
            CommandType.BLACKLIST: lambda: self._cmd_blacklist(cmd.args),
            CommandType.ADOPT: lambda: self._cmd_adopt(cmd.args.get("position_key", "") if cmd.args else ""),
            CommandType.CLOSE_ORPHAN: lambda: self._cmd_close_orphan(
                cmd.args.get("position_key", "") if cmd.args else ""
            ),
        }
        handler = handlers.get(cmd.command)
        if handler is None:
            return "❓ Unknown command"
        return handler()

    def _cmd_status(self: "ScalperProtocol") -> str:
        """!status — Show balance, positions, and active state.

        v12.5.2: Reconcile with exchange BEFORE displaying, so ghost
        positions are cleaned up and the user sees real-time state.
        """
        # Real-time reconciliation: sync with exchange before display
        if not self.paper_mode:
            self._sync_server_orders()
            self._reconcile_positions()

        balance = self.get_balance()
        positions = self.strategy.active_positions
        n_trades = len(self._trade_history)
        total_pnl = sum(t["pnl_usd"] for t in self._trade_history)

        status = "⏸️ PAUSED" if self._paused else "▶️ ACTIVE"
        if self._drawdown_halt:
            status = "🛑 DD HALT"

        # 스트래들 페어 통계
        all_pos = self.strategy._positions
        straddle_legs = [p for p in all_pos.values() if p.get("is_straddle")]
        seen_pairs: set[str] = set()
        n_pairs = 0
        for p in straddle_legs:
            pid = p.get("straddle_pair_id")
            if pid and pid not in seen_pairs:
                seen_pairs.add(pid)
                n_pairs += 1
        n_orphans = sum(1 for p in straddle_legs if p.get("straddle_orphan_since_cycle") is not None)

        lines = [
            "```",
            f"📊 QuantBox Status — {status}",
            f"{'─' * 40}",
            f"Balance:    ${balance:,.2f}",
            f"Session PnL: ${total_pnl:+,.2f} ({n_trades} trades)",
            f"Positions:  {len(positions)}/{self.config.max_concurrent}",
            f"Straddle:   {n_pairs}쌍 | 고아={n_orphans}",
            f"Cycle:      #{self._cycle_count}",
            f"{'─' * 40}",
        ]

        if positions:
            for sym, pos in positions.items():
                direction = "L" if pos["signal"] == 1 else "S"
                bars = pos.get("bars_held", 0)
                max_hold = pos.get("max_hold", 192)
                strat = pos.get("strategy_type", VR_DIRECTIONAL)
                # Extract Binance symbol from compound key
                api_sym = to_api_symbol(sym, pos)
                entry_px = pos["entry_price"]
                # v12.5.2: Show real-time mark price uPnL
                try:
                    mark = self.client.futures_mark_price(symbol=api_sym)
                    cur_px = float(mark["markPrice"])
                    if pos["signal"] == 1:
                        upnl_pct = (cur_px - entry_px) / entry_px * 100
                    else:
                        upnl_pct = (entry_px - cur_px) / entry_px * 100
                except Exception:
                    cur_px = entry_px
                    upnl_pct = 0.0
                # SL/TP status
                sl_pct = pos.get("sl", 0)
                tp_pct = pos.get("tp", 0)
                sl_ok = "✓" if pos.get("server_sl_order_id") else "✗"
                tp_ok = "✓" if pos.get("server_tp_order_id") else "✗"
                # Trail status
                if pos.get("trail_active"):
                    peak = pos.get("peak_gain", 0) * 100
                    trail_str = f" trail✓({peak:+.1f}%)"
                else:
                    trail_str = " trail✗"
                lines.append(f"  [{strat[:6]}] {direction} {sym}")
                lines.append(f"    ${entry_px:.4f}→${cur_px:.4f} uPnL={upnl_pct:+.1f}% {bars}/{max_hold}bars")
                lines.append(f"    SL={sl_pct * 100:.1f}%{sl_ok} TP={tp_pct * 100:.0f}%{tp_ok}{trail_str}")
        else:
            lines.append("  No open positions")

        lines.append("```")
        return "\n".join(lines)

    def _cmd_close_all(self: "ScalperProtocol") -> str:
        """!close all — Close all open positions."""
        positions = list(self.strategy.active_positions.keys())
        if not positions:
            return "📭 No positions to close"

        closed = 0
        for sym in positions:
            # Extract Binance symbol from compound key for kline download
            pos = self.strategy._positions.get(sym, {})
            api_sym = to_api_symbol(sym, pos)
            price_arr = self.download_klines(api_sym, 2)
            price = float(price_arr[-1]) if price_arr is not None else 0.0
            if self._close_position(sym, "bot_close_all", price):
                closed += 1

        self._save_state()
        return f"✅ Closed {closed}/{len(positions)} positions"

    def _cmd_close_slot(self: "ScalperProtocol", slot_id: int) -> str:
        """!close <N> — Close the Nth active position (0-indexed).

        Since the scalper uses symbol names instead of fixed slots,
        positions are numbered in order of their dict keys.

        Args:
            slot_id: 0-based index of the position to close.
        """
        positions = list(self.strategy.active_positions.items())
        if not positions:
            return "📭 No positions to close"

        if slot_id < 0 or slot_id >= len(positions):
            symbols = [f"{i}: {s}" for i, (s, _) in enumerate(positions)]
            return f"❌ Invalid index {slot_id}. Active positions:\n" + "\n".join(symbols)

        sym, pos = positions[slot_id]
        # Extract Binance symbol from compound key for kline download
        api_sym = to_api_symbol(sym, pos)
        price_arr = self.download_klines(api_sym, 2)
        price = float(price_arr[-1]) if price_arr is not None else 0.0
        if self._close_position(sym, "bot_close", price):
            self._save_state()
            return f"✅ Closed position {slot_id}: {sym}"
        return f"❌ Failed to close {sym}"

    def _cmd_pause(self: "ScalperProtocol") -> str:
        """!pause — Stop new entries, keep monitoring existing."""
        if self._paused:
            return "⏸️ Already paused"
        self._paused = True
        logger.info("Bot command: PAUSED (no new entries)")
        return "⏸️ New entries paused. Existing positions still monitored."

    def _cmd_resume(self: "ScalperProtocol") -> str:
        """!resume — Resume normal operation."""
        if not self._paused:
            return "▶️ Already active"
        self._paused = False
        logger.info("Bot command: RESUMED")
        return "▶️ Trading resumed"

    def _cmd_stop(self: "ScalperProtocol") -> str:
        """!stop — Close all positions and shut down."""
        self._stop_requested = True
        # Close all positions first
        positions = list(self.strategy.active_positions.keys())
        closed = 0
        for sym in positions:
            # Extract Binance symbol from compound key for kline download
            pos = self.strategy._positions.get(sym, {})
            api_sym = to_api_symbol(sym, pos)
            price_arr = self.download_klines(api_sym, 2)
            price = float(price_arr[-1]) if price_arr is not None else 0.0
            if self._close_position(sym, "bot_stop", price):
                closed += 1
        self._save_state()
        logger.info(f"Bot command: STOP (closed {closed} positions)")
        return f"🛑 Stopping... Closed {closed} positions. Scalper will exit after this cycle."

    def _cmd_pnl(self: "ScalperProtocol") -> str:
        """!pnl — Session PnL summary."""
        trades = self._trade_history
        if not trades:
            return "📭 No trades this session"

        total_pnl = sum(t["pnl_usd"] for t in trades)
        wins = [t for t in trades if t["pnl_usd"] > 0]
        losses = [t for t in trades if t["pnl_usd"] <= 0]
        wr = len(wins) / len(trades) * 100 if trades else 0
        avg_win = (sum(t["pnl_usd"] for t in wins) / len(wins)) if wins else 0
        avg_loss = (sum(t["pnl_usd"] for t in losses) / len(losses)) if losses else 0
        best = max(trades, key=lambda t: t["pnl_usd"])
        worst = min(trades, key=lambda t: t["pnl_usd"])

        balance = self.get_balance()
        session_return = (
            (balance - self._initial_balance) / self._initial_balance * 100 if self._initial_balance > 0 else 0
        )

        return (
            f"```\n"
            f"📈 Session PnL Summary\n"
            f"{'─' * 36}\n"
            f"Total PnL:    ${total_pnl:+,.2f}\n"
            f"Session ROI:  {session_return:+.2f}%\n"
            f"Trades:       {len(trades)} "
            f"(W:{len(wins)} / L:{len(losses)})\n"
            f"Win Rate:     {wr:.1f}%\n"
            f"Avg Win:      ${avg_win:+,.2f}\n"
            f"Avg Loss:     ${avg_loss:+,.2f}\n"
            f"Best Trade:   {best['symbol']} ${best['pnl_usd']:+,.2f}\n"
            f"Worst Trade:  {worst['symbol']} ${worst['pnl_usd']:+,.2f}\n"
            f"Balance:      ${balance:,.2f}\n"
            f"```"
        )

    def _cmd_blacklist(self: "ScalperProtocol", args: dict | None = None) -> str:
        """!blacklist — View or clear the exclude list.

        Args:
            args: Optional dict with 'action': 'clear' to reset.
        """
        if args and args.get("action") == "clear":
            self.config.exclude_symbols.clear()
            return "✅ Blacklist cleared"

        excludes = self.config.exclude_symbols
        if not excludes:
            return "📭 Blacklist is empty"
        return f"🚫 Blacklist ({len(excludes)}): " + ", ".join(sorted(excludes))

    def _cmd_adopt(self: "ScalperProtocol", position_key: str) -> str:
        """!adopt <position_key> — Adopt an orphan exchange position into internal state.

        Queries the exchange for the position, creates an internal state entry,
        and alerts that brackets need to be placed on the next cycle.

        Args:
            position_key: Compound key like "SUIUSDT:LONG" or plain "SUIUSDT".
        """
        if not position_key:
            return "❌ 사용법: `!adopt <포지션키>` (예: `!adopt SUIUSDT:LONG`)"

        if self.paper_mode:
            return "❌ Paper mode에서는 !adopt 사용 불가"

        parts = position_key.split(":")
        api_sym = parts[0]
        pos_side = parts[1] if len(parts) == 2 and parts[1] in ("LONG", "SHORT") else None

        try:
            exchange_positions = self.client.futures_position_information(symbol=api_sym)
        except Exception as e:
            return f"❌ 거래소 조회 실패: {e}"

        # Find the matching position
        target_pos = None
        for ep in exchange_positions:
            amt = float(ep.get("positionAmt", 0))
            if abs(amt) < 1e-8:
                continue
            ep_side = ep.get("positionSide", "BOTH")
            if pos_side is None or ep_side == pos_side:
                target_pos = ep
                break

        if target_pos is None:
            return f"❌ [{position_key}] 거래소에 포지션 없음 (이미 청산됐거나 키 오류)"

        amt = float(target_pos.get("positionAmt", 0))
        entry_px = float(target_pos.get("entryPrice", 0) or 0)
        ep_side = target_pos.get("positionSide", "BOTH")

        if entry_px <= 0:
            return f"❌ [{position_key}] entryPrice={entry_px} 유효하지 않음 — 수동 처리 필요"

        notional = abs(amt) * entry_px
        signal = 1 if (ep_side == "LONG" or amt > 0) else -1

        # Check if already tracked (race condition guard)
        if self.strategy.has_position(api_sym) or position_key in self.strategy.active_positions:
            return f"⚠️ [{position_key}] 이미 내부 상태에 존재 — adopt 불필요"

        # Adopt: open position in strategy state with exchange data
        self.strategy.open_position(
            symbol=api_sym,
            signal=signal,
            entry_price=entry_px,
            notional=notional,
            leverage=int(float(target_pos.get("leverage", 1))),
            move_size=0.03,
            position_side=ep_side if ep_side in ("LONG", "SHORT") else None,
            strategy_type="adopted",
        )

        self._save_state()
        logger.info(f"[{position_key}] Adopted orphan position: qty={abs(amt):.6f} entry={entry_px:.4f}")
        self.discord.send(
            content=(
                f"✅ **[{position_key}] 고아 포지션 채택 완료** — "
                f"qty={abs(amt):.6f}, entry={entry_px:.4f}, notional=${notional:.2f}\n"
                f"⚠️ SL/TP 브라켓이 없습니다 — 다음 사이클에서 배치됩니다."
            )
        )
        return f"✅ [{position_key}] 채택 완료. SL/TP는 다음 사이클에서 배치됩니다."

    def _cmd_close_orphan(self: "ScalperProtocol", position_key: str) -> str:
        """!close-orphan <position_key> — Market close an orphan exchange position.

        Closes the position directly on the exchange without touching internal state.

        Args:
            position_key: Compound key like "SUIUSDT:LONG" or plain "SUIUSDT".
        """
        if not position_key:
            return "❌ 사용법: `!close-orphan <포지션키>` (예: `!close-orphan SUIUSDT:LONG`)"

        if self.paper_mode:
            return "❌ Paper mode에서는 !close-orphan 사용 불가"

        parts = position_key.split(":")
        api_sym = parts[0]
        pos_side = parts[1] if len(parts) == 2 and parts[1] in ("LONG", "SHORT") else None

        try:
            exchange_positions = self.client.futures_position_information(symbol=api_sym)
        except Exception as e:
            return f"❌ 거래소 조회 실패: {e}"

        # Find the matching position
        target_pos = None
        for ep in exchange_positions:
            amt = float(ep.get("positionAmt", 0))
            if abs(amt) < 1e-8:
                continue
            ep_side = ep.get("positionSide", "BOTH")
            if pos_side is None or ep_side == pos_side:
                target_pos = ep
                break

        if target_pos is None:
            return f"❌ [{position_key}] 거래소에 포지션 없음 (이미 청산됐거나 키 오류)"

        amt = float(target_pos.get("positionAmt", 0))
        ep_side = target_pos.get("positionSide", "BOTH")
        close_side = "SELL" if (ep_side == "LONG" or amt > 0) else "BUY"

        try:
            order_kwargs: dict = dict(
                symbol=api_sym,
                side=close_side,
                type="MARKET",
                quantity=abs(amt),
            )
            if ep_side in ("LONG", "SHORT"):
                order_kwargs["positionSide"] = ep_side
            self.client.futures_create_order(**order_kwargs)
            logger.info(f"[{position_key}] Orphan closed via !close-orphan: {close_side} qty={abs(amt):.6f}")
            self.discord.send(
                content=(f"✅ **[{position_key}] 고아 포지션 청산 완료** — {close_side} qty={abs(amt):.6f} MARKET 체결")
            )
            # Record in recon journal
            try:
                if self._recon_journal:
                    from quantbox.execution.recon_journal import ReconEventType

                    self._recon_journal.record(
                        event_type=ReconEventType.REVERSE_GHOST_CLOSED,
                        symbol=api_sym,
                        action=f"manual close-orphan {close_side} qty={abs(amt):.6f}",
                        result="success",
                        details={"position_side": ep_side, "qty": abs(amt)},
                    )
            except Exception:
                pass
            return f"✅ [{position_key}] 고아 포지션 청산 완료"
        except Exception as e:
            logger.error(f"[{position_key}] !close-orphan FAILED: {e}")
            return f"❌ [{position_key}] 청산 실패: {e}"

    # ==================== State Persistence ====================
