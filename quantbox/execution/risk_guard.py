"""RiskGuard — pre-entry risk checks for the scan loop.

Encapsulates the gate logic that was previously inlined in scalper.py's
candidate scan loop.  RiskGuard is a stateless helper: it receives config
at construction time and accepts state snapshots as method arguments so it
can be unit-tested without a live scalper instance.

Gates covered:
- G6: BTC trend MA gate (compute + LONG suppression)
- Straddle slot cap
- Direction diversity filter
- G7: Daily loss circuit-breaker threshold check

What stays in scalper.py:
- _daily_loss_blocked / _daily_loss_date state tracking (date reset, Discord alert)
- _drawdown_halt logic (also triggers position closes — too interleaved)
"""

from __future__ import annotations

import logging

import numpy as np

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

logger = logging.getLogger("Scalper")


class RiskGuard:
    """Stateless pre-entry risk gate helper.

    Args:
        vs_config: Active VolStraddleConfig instance.
    """

    def __init__(self, vs_config: VolStraddleConfig) -> None:
        self.cfg = vs_config

    # ── G6: BTC Trend Gate ────────────────────────────────────────────────

    def compute_btc_gate(self, btc_close: np.ndarray | None) -> float | None:
        """Compute BTC MA for G6 gate.

        Args:
            btc_close: Array of BTC close prices. May be None when disabled or
                data unavailable.

        Returns:
            MA value as float when G6 is active and data is sufficient,
            None otherwise (gate inactive — do not block LONG entries).
        """
        if not self.cfg.btc_long_filter_enabled or btc_close is None or len(btc_close) < self.cfg.btc_long_ma_period:
            return None
        btc_ma = float(np.mean(btc_close[-self.cfg.btc_long_ma_period :]))
        btc_cur = float(btc_close[-1])
        # 백테스트는 트레이드마다 이 게이트를 부른다 — f-string 평가를 지연시킨다.
        if logger.isEnabledFor(logging.INFO):
            btc_above_ma = btc_cur >= btc_ma
            logger.info(
                f"G6 BTC Trend Gate: BTC=${btc_cur:.1f} "
                f"MA{self.cfg.btc_long_ma_period}=${btc_ma:.1f} "
                f"-> {'ABOVE (LONG OK)' if btc_above_ma else 'BELOW (LONG BLOCKED)'}"
            )
        return btc_ma

    def allow_long(self, btc_close: np.ndarray | None, btc_ma: float | None) -> bool:
        """G6: return False when BTC is below MA and gate is active.

        Args:
            btc_close: Array of BTC close prices (used for current price).
            btc_ma: MA value returned by compute_btc_gate(). When None the
                gate is inactive and the method always returns True.

        Returns:
            True if LONG entry is allowed, False if suppressed by G6.
        """
        if btc_ma is None or btc_close is None:
            return True
        btc_cur = float(btc_close[-1])
        return btc_cur >= btc_ma

    # ── Direction diversity filter ────────────────────────────────────────

    def allow_direction(self, signal: int, active_positions: dict) -> tuple[bool, str]:
        """Direction diversity check.

        Blocks a new entry when too many positions already share the same
        direction (LONG or SHORT).  Straddle positions are direction-neutral
        and must be excluded by the caller before invoking this method.

        Args:
            signal: Entry signal — 1 for LONG, -1 for SHORT.
            active_positions: Dict of open positions (pos_key -> pos dict).
                Each pos dict must have a 'signal' key.

        Returns:
            (allowed, reason) — allowed=True when entry is permitted;
            reason is a human-readable explanation when blocked.
        """
        if not self.cfg.dir_diversity_enabled:
            return True, ""
        same_dir = sum(1 for p in active_positions.values() if p.get("signal") == signal)
        if same_dir >= self.cfg.max_same_direction:
            direction = "LONG" if signal == 1 else "SHORT"
            return (
                False,
                f"{same_dir} {direction} already open (max={self.cfg.max_same_direction})",
            )
        return True, ""

    # ── G7: Daily loss circuit-breaker threshold ──────────────────────────

    def check_daily_loss(self, realized_pnl: float, balance: float) -> bool:
        """Return True if today's realized PnL has exceeded the daily loss limit.

        State tracking (date reset, _daily_loss_blocked flag, Discord alert)
        remains in scalper.py.  This method only computes the threshold check.

        Args:
            realized_pnl: Today's cumulative realized PnL in USD (negative = loss).
            balance: Current account balance in USD.

        Returns:
            True if the loss limit is exceeded (block new entries),
            False if within limit.
        """
        if self.cfg.daily_loss_limit_pct <= 0:
            return False
        limit_usd = balance * self.cfg.daily_loss_limit_pct
        return realized_pnl < -limit_usd
