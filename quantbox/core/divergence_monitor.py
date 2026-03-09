"""Live-Backtest Divergence Monitor.

Compares live trade journal statistics against the v14.5.0 canonical backtest
baseline. Fires a DivergenceAlert when WR/PF diverge beyond thresholds.

Canonical baseline (v14.5.0, 60 syms, 6yr, commit 7399f4c):
  PF  = 1.563
  WR  = 43.4%
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── v14.5.0 canonical backtest baseline ──────────────────────────────────────
_BASELINE_PF: float = 1.563
_BASELINE_WR: float = 0.434  # 43.4%


@dataclass
class DivergenceAlert:
    """Fired when live performance diverges from backtest baseline."""

    pf_live: float
    pf_baseline: float
    wr_live: float
    wr_baseline: float
    n_trades: int
    triggered_by: list[str]  # e.g. ["pf"], ["wr"], ["pf", "wr"]
    message: str = field(default="")

    def __post_init__(self) -> None:
        if not self.message:
            parts = []
            if "pf" in self.triggered_by:
                parts.append(
                    f"PF {self.pf_live:.3f} < threshold {self._pf_threshold:.2f} (baseline {self.pf_baseline:.3f})"
                )
            if "wr" in self.triggered_by:
                gap = (self.wr_baseline - self.wr_live) * 100
                parts.append(
                    f"WR {self.wr_live * 100:.1f}% is {gap:.1f}pp below baseline {self.wr_baseline * 100:.1f}%"
                )
            self.message = "; ".join(parts)

    # store threshold for __post_init__ message formatting
    _pf_threshold: float = field(default=0.80, init=False, repr=False)


class DivergenceMonitor:
    """Monitors live trade performance vs backtest baseline.

    Args:
        min_trades: Minimum live trades before checking.
        pf_threshold: Alert if live PF drops below this (default 0.80).
        wr_gap_pp: Alert if live WR is X percentage-points below baseline.
        cooldown_hours: Minimum hours between consecutive alerts.
        cooldown_path: Path to JSON file storing last alert timestamp.
    """

    BASELINE_PF: float = _BASELINE_PF
    BASELINE_WR: float = _BASELINE_WR

    def __init__(
        self,
        min_trades: int = 30,
        pf_threshold: float = 0.80,
        wr_gap_pp: float = 10.0,
        cooldown_hours: float = 24.0,
        cooldown_path: Path | str | None = None,
    ) -> None:
        self.min_trades = min_trades
        self.pf_threshold = pf_threshold
        self.wr_gap_pp = wr_gap_pp
        self.cooldown_hours = cooldown_hours
        self._cooldown_path = Path(cooldown_path) if cooldown_path else Path("data/state/divergence_last_alert.json")

    # ── public API ────────────────────────────────────────────────────────────

    def check(self, stats: dict) -> DivergenceAlert | None:
        """Check live stats against baseline.

        Args:
            stats: dict from TradeJournal.get_stats() — must have keys
                   'n_trades', 'win_rate', 'profit_factor'.

        Returns:
            DivergenceAlert if a threshold is breached and cooldown elapsed,
            else None.
        """
        n = stats.get("n_trades", 0)
        if n < self.min_trades:
            logger.debug("Divergence check skipped: n=%d < min=%d", n, self.min_trades)
            return None

        if self._in_cooldown():
            logger.debug("Divergence check skipped: cooldown active")
            return None

        pf_live = stats.get("profit_factor", 0.0)
        wr_live = stats.get("win_rate", 0.0)

        triggered: list[str] = []
        if pf_live < self.pf_threshold:
            triggered.append("pf")
        if (self.BASELINE_WR - wr_live) * 100 >= self.wr_gap_pp:
            triggered.append("wr")

        if not triggered:
            return None

        alert = DivergenceAlert(
            pf_live=pf_live,
            pf_baseline=self.BASELINE_PF,
            wr_live=wr_live,
            wr_baseline=self.BASELINE_WR,
            n_trades=n,
            triggered_by=triggered,
        )
        alert._pf_threshold = self.pf_threshold
        # Re-generate message with threshold set
        parts = []
        if "pf" in triggered:
            parts.append(f"PF {pf_live:.3f} < threshold {self.pf_threshold:.2f} (baseline {self.BASELINE_PF:.3f})")
        if "wr" in triggered:
            gap = (self.BASELINE_WR - wr_live) * 100
            parts.append(f"WR {wr_live * 100:.1f}% is {gap:.1f}pp below baseline {self.BASELINE_WR * 100:.1f}%")
        alert.message = "; ".join(parts)

        self._save_cooldown()
        logger.warning("Divergence alert: %s", alert.message)
        return alert

    # ── cooldown helpers ──────────────────────────────────────────────────────

    def _in_cooldown(self) -> bool:
        try:
            if not self._cooldown_path.exists():
                return False
            data = json.loads(self._cooldown_path.read_text())
            last_str = data.get("last_alert_at")
            if not last_str:
                return False
            last_dt = datetime.fromisoformat(last_str)
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            return elapsed < self.cooldown_hours
        except Exception:
            return False

    def _save_cooldown(self) -> None:
        try:
            self._cooldown_path.parent.mkdir(parents=True, exist_ok=True)
            self._cooldown_path.write_text(json.dumps({"last_alert_at": datetime.now(timezone.utc).isoformat()}))
        except Exception as exc:
            logger.warning("Could not save divergence cooldown: %s", exc)
