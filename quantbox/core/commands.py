"""Discord bot command handlers.

Mixin class for processing interactive commands received via Discord
(!status, !close, !pause, !resume, !stop, !pnl, !blacklist).
"""

from __future__ import annotations

import logging

from quantbox.notifications.bot import BotCommand, CommandType

logger = logging.getLogger("Scalper")


class CommandsMixin:
    """Mixin: Discord bot command processing and handlers."""

    def _process_bot_commands(self) -> None:
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

    def _handle_command(self, cmd: BotCommand) -> str:
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
        }
        handler = handlers.get(cmd.command)
        if handler is None:
            return "❓ Unknown command"
        return handler()

    def _cmd_status(self) -> str:
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

        lines = [
            "```",
            f"📊 QuantBox Status — {status}",
            f"{'─' * 40}",
            f"Balance:    ${balance:,.2f}",
            f"Session PnL: ${total_pnl:+,.2f} ({n_trades} trades)",
            f"Positions:  {len(positions)}/{self.config.max_concurrent}",
            f"Cycle:      #{self._cycle_count}",
            f"{'─' * 40}",
        ]

        if positions:
            for sym, pos in positions.items():
                direction = "L" if pos["signal"] == 1 else "S"
                bars = pos.get("bars_held", 0)
                best = pos.get("best_pnl", 0) * 100  # noqa: F841
                notional = pos.get("notional", 0)
                # Extract Binance symbol from compound key
                api_sym = pos.get("symbol", sym.split(":")[0])
                # v12.5.2: Show real-time mark price uPnL
                try:
                    mark = self.client.futures_mark_price(symbol=api_sym)
                    cur_px = float(mark["markPrice"])
                    entry_px = pos["entry_price"]
                    if pos["signal"] == 1:
                        upnl_pct = (cur_px - entry_px) / entry_px * 100
                    else:
                        upnl_pct = (entry_px - cur_px) / entry_px * 100
                except Exception:
                    upnl_pct = 0.0
                lines.append(
                    f"  {direction} {sym}: "
                    f"${pos['entry_price']:.4f} | "
                    f"${notional:.0f} | "
                    f"{bars}bars | uPnL={upnl_pct:+.1f}%"
                )
        else:
            lines.append("  No open positions")

        lines.append("```")
        return "\n".join(lines)

    def _cmd_close_all(self) -> str:
        """!close all — Close all open positions."""
        positions = list(self.strategy.active_positions.keys())
        if not positions:
            return "📭 No positions to close"

        closed = 0
        for sym in positions:
            # Extract Binance symbol from compound key for kline download
            pos = self.strategy._positions.get(sym, {})
            api_sym = pos.get("symbol", sym.split(":")[0])
            price_arr = self.download_klines(api_sym, 2)
            price = price_arr[-1] if price_arr is not None else 0
            if self._close_position(sym, "bot_close_all", price):
                closed += 1

        self._save_state()
        return f"✅ Closed {closed}/{len(positions)} positions"

    def _cmd_close_slot(self, slot_id: int) -> str:
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
        api_sym = pos.get("symbol", sym.split(":")[0])
        price_arr = self.download_klines(api_sym, 2)
        price = price_arr[-1] if price_arr is not None else 0
        if self._close_position(sym, "bot_close", price):
            self._save_state()
            return f"✅ Closed position {slot_id}: {sym}"
        return f"❌ Failed to close {sym}"

    def _cmd_pause(self) -> str:
        """!pause — Stop new entries, keep monitoring existing."""
        if self._paused:
            return "⏸️ Already paused"
        self._paused = True
        logger.info("Bot command: PAUSED (no new entries)")
        return "⏸️ New entries paused. Existing positions still monitored."

    def _cmd_resume(self) -> str:
        """!resume — Resume normal operation."""
        if not self._paused:
            return "▶️ Already active"
        self._paused = False
        logger.info("Bot command: RESUMED")
        return "▶️ Trading resumed"

    def _cmd_stop(self) -> str:
        """!stop — Close all positions and shut down."""
        self._stop_requested = True
        # Close all positions first
        positions = list(self.strategy.active_positions.keys())
        closed = 0
        for sym in positions:
            # Extract Binance symbol from compound key for kline download
            pos = self.strategy._positions.get(sym, {})
            api_sym = pos.get("symbol", sym.split(":")[0])
            price_arr = self.download_klines(api_sym, 2)
            price = price_arr[-1] if price_arr is not None else 0
            if self._close_position(sym, "bot_stop", price):
                closed += 1
        self._save_state()
        logger.info(f"Bot command: STOP (closed {closed} positions)")
        return f"🛑 Stopping... Closed {closed} positions. Scalper will exit after this cycle."

    def _cmd_pnl(self) -> str:
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

    def _cmd_blacklist(self, args: dict | None = None) -> str:
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

    # ==================== State Persistence ====================
