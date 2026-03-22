"""
Lifecycle factory functions for RubberBandScalper component initialization.

Extracts setup logic from RubberBandScalper.__init__ into standalone functions
for better separation of concerns and testability.

Note: Client, DiscordNotifier, DiscordBotRunner construction stays in scalper.py
because tests widely patch those names at `quantbox.scalper.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from quantbox.core.divergence_monitor import DivergenceMonitor
from quantbox.core.heartbeat import Heartbeat
from quantbox.core.rate_limiter import RateLimiter
from quantbox.core.trade_journal import TradeJournal
from quantbox.execution.exposure import ExposureGuard
from quantbox.execution.inflight import InFlightTracker
from quantbox.execution.recon_journal import ReconJournal
from quantbox.strategy.vol_straddle import VolStraddleConfig

logger = logging.getLogger("Scalper")


def create_trade_journal(
    paper_mode: bool,
    journal_db_path: str | None = None,
) -> TradeJournal:
    """TradeJournal (SQLite) 인스턴스 생성.

    Args:
        paper_mode: True for paper trading session.
        journal_db_path: Override DB path. Defaults to data/journal*.sqlite.

    Returns:
        Initialized TradeJournal.
    """
    session_str = "paper" if paper_mode else "live"
    if journal_db_path is not None:
        db_path = journal_db_path
    else:
        db_path = "data/journal_paper.sqlite" if paper_mode else "data/journal.sqlite"
    return TradeJournal(db_path=db_path, session=session_str)


def create_divergence_monitor(vs_config: VolStraddleConfig) -> DivergenceMonitor:
    """Live-Backtest divergence monitor 생성.

    Args:
        vs_config: VolStraddle config for divergence thresholds.

    Returns:
        Initialized DivergenceMonitor.
    """
    return DivergenceMonitor(
        min_trades=vs_config.divergence_min_trades,
        pf_threshold=vs_config.divergence_pf_threshold,
        wr_gap_pp=vs_config.divergence_wr_gap_pp,
        cooldown_hours=vs_config.divergence_cooldown_hours,
    )


def create_reliability_modules(
    vs_config: VolStraddleConfig,
) -> dict:
    """Phase 3 reliability 모듈 생성 (각각 독립적으로 실패 허용).

    Args:
        vs_config: VolStraddle config for exposure guard settings.

    Returns:
        Dict with keys: rate_limiter, heartbeat, exposure_guard, inflight, recon_journal.
        Values are None if initialization failed.
    """
    modules: dict = {}

    try:
        modules["rate_limiter"] = RateLimiter()
    except Exception as e:
        logger.error(f"RateLimiter init failed: {e}")
        modules["rate_limiter"] = None

    try:
        modules["heartbeat"] = Heartbeat(stall_threshold_s=300.0)
    except Exception as e:
        logger.error(f"Heartbeat init failed: {e}")
        modules["heartbeat"] = None

    try:
        max_mult = vs_config.max_exposure_multiplier
        modules["exposure_guard"] = ExposureGuard(max_multiplier=max_mult)
    except Exception as e:
        logger.error(f"ExposureGuard init failed: {e}")
        modules["exposure_guard"] = None

    try:
        inflight_path = Path("data/state/inflight_orders.json")
        modules["inflight"] = InFlightTracker(persist_path=inflight_path)
    except Exception as e:
        logger.error(f"InFlightTracker init failed: {e}")
        modules["inflight"] = None

    try:
        modules["recon_journal"] = ReconJournal(Path("data/journal/recon.db"))
    except Exception as e:
        logger.error(f"ReconJournal init failed: {e}")
        modules["recon_journal"] = None

    return modules
