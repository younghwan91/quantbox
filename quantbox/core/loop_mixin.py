"""LoopMixin — timing, display, and summary logging for RubberBandScalper.

Extracted from scalper.py to reduce file size and isolate concerns.
Methods in this mixin rely on instance attributes set by RubberBandScalper.__init__:
    self.strategy         — active strategy instance
    self.strategy_name    — 'vol_straddle' / 'rubberband' / 'elliott'
    self.config           — RubberBandConfig
    self.vs_config        — VolStraddleConfig
    self._cfg             — active config (vs_config or config)
    self._trade_history   — list of completed trades
    self._high_water_mark — peak balance (float)
    self.paper_mode       — bool
    self._cycle_count     — int
    self._stop_requested  — bool
    self._process_bot_commands()  — from CommandsMixin
    self._sync_server_orders()    — from ReconciliationMixin
    self._save_state()            — from RubberBandScalper
"""

import logging
import time
from datetime import datetime, timedelta, timezone

from quantbox.core.bot_utils import GIT_COMMIT
from quantbox.core.settings import get_settings
from quantbox.core.terminal_ui import (
    print_banner,
    print_startup_complete,
    print_system_info,
)

logger = logging.getLogger("Scalper")


class LoopMixin:
    """Timing, display and periodic summary logging."""

    def _log_summary(self, balance: float) -> None:
        """Log periodic summary — always, even with 0 trades."""
        positions = self.strategy.active_positions
        n_trades = len(self._trade_history)
        wins = sum(1 for t in self._trade_history if t["pnl_usd"] > 0)
        total_pnl = sum(t["pnl_usd"] for t in self._trade_history)
        wr = wins / n_trades * 100 if n_trades > 0 else 0

        hwm = self._high_water_mark if self._high_water_mark > 0 else balance
        dd_pct = (hwm - balance) / hwm * 100 if hwm > 0 else 0.0

        # Time since last trade
        last_trade_ago = ""
        if self._trade_history:
            last_t = self._trade_history[-1].get("time", "")
            if last_t:
                try:
                    from datetime import datetime as _dt

                    lt = _dt.fromisoformat(last_t)
                    delta = datetime.now(timezone.utc) - lt
                    hours = delta.total_seconds() / 3600
                    last_trade_ago = f", LastTrade={hours:.1f}h ago"
                except Exception:
                    pass

        logger.info(
            f"[Summary] Trades={n_trades}, WR={wr:.0f}%, "
            f"PnL=${total_pnl:+.2f}, Balance=${balance:.2f}, "
            f"HWM=${hwm:.2f}, DD={dd_pct:.1f}%, "
            f"Active={len(positions)}{last_trade_ago}"
        )

        # Active position details
        for sym, pos in positions.items():
            logger.info(
                f"  [{sym}] {'L' if pos['signal'] == 1 else 'S'} "
                f"entry=${pos['entry_price']:.4f} "
                f"bars={pos['bars_held']}/{pos.get('max_hold', self.config.max_hold)} "
                f"MFE={pos.get('best_pnl', 0) * 100:+.2f}% "
                f"MAE={pos.get('worst_pnl', 0) * 100:+.2f}% "
                f"SL={pos.get('server_sl_order_id') or 'NONE'} "
                f"TP={pos.get('server_tp_order_id') or 'NONE'}"
            )

        # Periodic VR distribution (every 16 cycles ≈ 4h) when no positions
        if self.strategy_name == "vol_straddle" and len(positions) == 0 and self._cycle_count % 16 == 0:
            self._log_vr_distribution()

    def _log_vr_distribution(self) -> None:
        """Log top-10 closest-to-squeeze symbols with per-TF VR.

        Provides visibility into market state during extended no-signal
        periods. Fires every ~4h (16 cycles × 15m).
        """
        if not hasattr(self.strategy, "get_vr_all_tf"):
            return
        vr_entries: list[tuple[str, float, dict[str, float]]] = []
        for sym in self._cfg.symbol_whitelist or []:
            vr_all = self.strategy.get_vr_all_tf(sym)
            min_vr = min(vr_all.values())
            if min_vr < 999.0:
                vr_entries.append((sym, min_vr, vr_all))
        if not vr_entries:
            return
        vr_entries.sort(key=lambda x: x[1])
        top10 = vr_entries[:10]
        lines = [f"  [VR Distribution] Top-10 closest to squeeze (threshold={self.vs_config.d_vr_threshold}):"]
        for rank, (sym, min_vr, vr_all) in enumerate(top10, 1):
            lines.append(
                f"    {rank:2d}. {sym:<16s} min={min_vr:.3f} "
                f"(15m={vr_all['15m']:.3f} 1h={vr_all['1h']:.3f} 4h={vr_all['4h']:.3f})"
            )
        logger.info("\n".join(lines))

    def _print_startup_banner(self, balance: float, n_symbols: int) -> None:
        """Print the startup banner with system info."""
        settings = get_settings()
        mode_str = "PAPER" if self.paper_mode else "LIVE"
        print_banner()
        if GIT_COMMIT["short"]:
            logger.info(f"Git: {GIT_COMMIT['short']} — {GIT_COMMIT['message']}")
        if self.strategy_name == "vol_straddle":
            vc = self.vs_config
            print_system_info(
                mode=mode_str,
                strategy="vol_straddle",
                balance=balance,
                n_symbols=n_symbols,
                max_slots=vc.max_concurrent,
                leverage_base=vc.base_leverage,
                leverage_max=vc.max_leverage,
                lookback=vc.momentum_period,
                threshold=vc.d_vr_threshold,
                target=vc.d_tp,
                stop_mult=vc.d_sl,
                max_hold=vc.max_hold_15m,
                entry_mode="vr_compression",
                min_confidence=0.0,
                testnet=settings.binance_testnet,
                ma_cluster_enabled=vc.ma_cluster_enabled,
                ma_cluster_pct=vc.ma_cluster_pct,
                ma_cluster_min_duration=vc.ma_cluster_min_duration,
                ma_cluster_max_duration=vc.ma_cluster_max_duration,
                ma_cluster_min_momentum=vc.ma_cluster_min_momentum,
                ma_cluster_sl=vc.ma_cluster_sl,
                ma_cluster_tp=vc.ma_cluster_tp,
                ma_cluster_leverage=vc.ma_cluster_leverage,
            )
        else:
            print_system_info(
                mode=mode_str,
                strategy=self.strategy_name,
                balance=balance,
                n_symbols=n_symbols,
                max_slots=self.config.max_concurrent,
                leverage_base=self.config.base_leverage,
                leverage_max=self.config.max_leverage,
                lookback=self.config.lookback,
                threshold=self.config.threshold,
                target=self.config.target_retrace,
                stop_mult=self.config.stop_mult,
                max_hold=self.config.max_hold,
                entry_mode=self.config.entry_mode,
                min_confidence=self.config.min_confidence,
                testnet=settings.binance_testnet,
            )
        print_startup_complete(mode_str)

    def _next_bar_time(self) -> datetime:
        """Calculate the next 15-minute bar close time (+5s buffer)."""
        now = datetime.now(timezone.utc)
        next_boundary = ((now.minute // 15) + 1) * 15
        if next_boundary >= 60:
            target = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
        else:
            target = now.replace(minute=next_boundary, second=5, microsecond=0)
        return target

    def _wait_for_bar(self) -> None:
        """Wait for the next 15m bar close, polling every 5s for commands.

        While waiting:
        - Process bot commands (~5s latency for responsive interaction)
        - Sync server-side TP/SL order fills (every 60s to avoid API spam)
        - Check for stop requests
        """
        target = self._next_bar_time()
        remaining = (target - datetime.now(timezone.utc)).total_seconds()
        if remaining > 0:
            next_str = target.strftime("%H:%M:%S UTC")
            logger.info(f"Waiting {remaining:.0f}s for next bar close ({next_str})")

        BOT_POLL_INTERVAL = 5.0  # Check bot commands every 5s
        SYNC_INTERVAL = 60.0  # Sync server orders every 60s
        last_sync = time.monotonic()

        while True:
            now = datetime.now(timezone.utc)
            remaining = (target - now).total_seconds()
            if remaining <= 0:
                break

            # Process bot commands while waiting (~5s response time)
            self._process_bot_commands()
            if self._stop_requested:
                break

            # Sync server-side TP/SL fills periodically (not every 5s)
            if time.monotonic() - last_sync >= SYNC_INTERVAL:
                self._sync_server_orders()
                self._save_state()  # v12.3.1: persist after inter-cycle sync
                last_sync = time.monotonic()

            # Sleep 5s max for responsive bot commands
            sleep_secs = min(BOT_POLL_INTERVAL, remaining)
            time.sleep(sleep_secs)
