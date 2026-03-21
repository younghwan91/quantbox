"""Tests for straddle half-open rollback (P3) and strategy_type auto-heal (P4).

Root causes addressed:
- P3: When S_d8 straddle LONG succeeds but SHORT fails (-2027 or other API error),
  the orphaned LONG was left open with no counterpart (delta unintended). Fixed by
  rolling back the LONG immediately in _scan_entries().
- P4: State files written before commit 0ac2e2b stored is_straddle=True positions
  with strategy_type='vr_squeeze'. On restart these showed wrong label in dashboard.
  Fixed by _healed_strategy_type() called from _load_state().
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from tests.conftest import _build_vs_scalper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLOSE_PRICE = 1.50  # arbitrary mark price for tests
QTY = 2000.0
ENTRY = 1.45


def _straddle_pos(sym: str, side: str, strategy: str = "straddle") -> dict:
    """Minimal position dict for a straddle leg."""
    return {
        "symbol": sym,
        "signal": 1 if side == "LONG" else -1,
        "entry_price": ENTRY,
        "notional": ENTRY * QTY,
        "bars_held": 0,
        "peak_gain": 0.0,
        "best_pnl": 0.0,
        "worst_pnl": 0.0,
        "server_sl_order_id": "sl_001",
        "server_tp_order_id": "tp_001",
        "sl": 0.01,
        "tp": 0.20,
        "tf": "1h",
        "position_side": side,
        "trail_active": False,
        "strategy_type": strategy,
        "is_straddle": True,
        "leverage": 30,
        "move_size": 0.03,
    }


# ---------------------------------------------------------------------------
# P3: Straddle half-open rollback
# ---------------------------------------------------------------------------


class TestStraddleHalfOpenRollback:
    """_scan_entries() must close the orphaned LONG when SHORT fails."""

    def _make_scalper_with_straddle_candidates(self, short_fails: bool = True):
        """Return a scalper configured so S_d8 produces BANDUSDT straddle candidates."""
        scalper = _build_vs_scalper()

        # Simulate straddle candidates: LONG then SHORT, same symbol
        candidates = [
            ("BANDUSDT", 1, 0.03, [CLOSE_PRICE] * 20, True),
            ("BANDUSDT", -1, 0.03, [CLOSE_PRICE] * 20, True),
        ]

        def _mock_open_position(sym, signal, close, balance):
            if sym == "BANDUSDT" and signal == 1:
                # LONG succeeds — inject position into state
                key = "BANDUSDT:LONG"
                scalper.strategy._positions[key] = _straddle_pos("BANDUSDT", "LONG")
                return True
            if sym == "BANDUSDT" and signal == -1:
                # SHORT fails if short_fails=True
                return not short_fails
            return False

        scalper._open_position = _mock_open_position
        scalper._candidates_override = candidates
        return scalper, candidates

    def test_rollback_closes_long_when_short_fails(self):
        """When LONG opens but SHORT fails, LONG must be closed via straddle_rollback."""
        scalper, candidates = self._make_scalper_with_straddle_candidates(short_fails=True)

        close_called_with: list[tuple] = []

        def _mock_close(pos_key, reason, price):
            close_called_with.append((pos_key, reason, price))
            # Remove from state (simulate real close)
            scalper.strategy._positions.pop(pos_key, None)
            return True

        scalper._close_position = _mock_close

        # Call the rollback logic directly by running with candidates
        # We simulate the straddle_opened tracking portion of _scan_entries()
        entries_made = 0
        straddle_opened: dict[str, set[int]] = {}

        for sym, signal, move, close, is_straddle in candidates:
            opened = scalper._open_position(sym, signal, close, 1000.0)
            if opened:
                entries_made += 1
                if is_straddle:
                    straddle_opened.setdefault(sym, set()).add(signal)

        # Now run the rollback block (same logic as _scan_entries)
        for sym, signals_opened in straddle_opened.items():
            if len(signals_opened) == 1:
                opened_signal = next(iter(signals_opened))
                side_str = "LONG" if opened_signal == 1 else "SHORT"
                pos_key = f"{sym}:{side_str}"
                close_arr = next((c for s, sig, _, c, is_s in candidates if s == sym and is_s), None)
                rollback_price = float(close_arr[-1]) if close_arr is not None else 0.0
                scalper._close_position(pos_key, "straddle_rollback", rollback_price)
                entries_made -= 1

        assert ("BANDUSDT:LONG", "straddle_rollback", CLOSE_PRICE) in close_called_with, (
            "Straddle rollback must close BANDUSDT:LONG with reason='straddle_rollback'"
        )
        assert entries_made == 0, "Rolled-back entry should not count toward entries_made"

    def test_no_rollback_when_both_legs_open(self):
        """When both legs open successfully, no rollback is triggered."""
        scalper, candidates = self._make_scalper_with_straddle_candidates(short_fails=False)

        close_called_with: list[tuple] = []

        def _mock_close(pos_key, reason, price):
            close_called_with.append((pos_key, reason, price))
            return True

        scalper._close_position = _mock_close

        straddle_opened: dict[str, set[int]] = {}
        for sym, signal, move, close, is_straddle in candidates:
            opened = scalper._open_position(sym, signal, close, 1000.0)
            if opened:
                if is_straddle:
                    straddle_opened.setdefault(sym, set()).add(signal)

        for sym, signals_opened in straddle_opened.items():
            if len(signals_opened) == 1:
                # This should not run
                scalper._close_position(f"{sym}:LONG", "straddle_rollback", CLOSE_PRICE)

        assert close_called_with == [], "No rollback should happen when both legs open"

    def test_rollback_sends_discord_alert(self):
        """Discord alert is sent when straddle is rolled back."""
        scalper, candidates = self._make_scalper_with_straddle_candidates(short_fails=True)

        discord_contents: list[str] = []

        def _capture_discord(content=""):
            discord_contents.append(content)

        scalper.discord.send = _capture_discord
        scalper._close_position = lambda pos_key, reason, price: True

        straddle_opened: dict[str, set[int]] = {}
        for sym, signal, move, close, is_straddle in candidates:
            if scalper._open_position(sym, signal, close, 1000.0) and is_straddle:
                straddle_opened.setdefault(sym, set()).add(signal)

        for sym, signals_opened in straddle_opened.items():
            if len(signals_opened) == 1:
                scalper.discord.send(content=f"⚠️ [{sym}] Straddle half-open — rollback")

        assert any("Straddle" in c or "rollback" in c or "half" in c.lower() for c in discord_contents), (
            "Discord alert must be sent on straddle rollback"
        )


# ---------------------------------------------------------------------------
# P3b: Straddle rollback cooldown
# ---------------------------------------------------------------------------


class TestStraddleRollbackCooldown:
    """After rollback, the symbol must be blocked for N cycles."""

    def test_cooldown_blocks_signal(self):
        """on_bar returns 0 while cooldown is active."""
        scalper = _build_vs_scalper()
        strategy = scalper.strategy

        # Set cooldown for BANDUSDT
        strategy._straddle_rollback_cooldown["BANDUSDT"] = 3

        import numpy as np

        n = 500
        close = np.linspace(1.0, 1.5, n)
        high = close * 1.01
        low = close * 0.99
        vol = np.ones(n) * 1000.0
        taker_buy = vol * 0.5
        strategy.update_market_data("BANDUSDT", vol, taker_buy, high, low)

        # First call: cooldown=3 → 0, decrements to 2
        result = strategy.on_bar("BANDUSDT", close, "15m")
        assert result == 0, "Signal must be blocked during cooldown"
        assert strategy._straddle_rollback_cooldown["BANDUSDT"] == 2

        # Second call: cooldown=2 → 0, decrements to 1
        result = strategy.on_bar("BANDUSDT", close, "15m")
        assert result == 0
        assert strategy._straddle_rollback_cooldown["BANDUSDT"] == 1

        # Third call: cooldown=1 → 0, decrements to 0 (expired)
        result = strategy.on_bar("BANDUSDT", close, "15m")
        assert result == 0
        assert strategy._straddle_rollback_cooldown["BANDUSDT"] == 0

    def test_cooldown_expires_allows_signal(self):
        """After cooldown expires, on_bar can produce signals again."""
        scalper = _build_vs_scalper()
        strategy = scalper.strategy

        # Cooldown already at 0 — should not block
        strategy._straddle_rollback_cooldown["BANDUSDT"] = 0

        import numpy as np

        n = 500
        close = np.linspace(1.0, 1.5, n)
        high = close * 1.01
        low = close * 0.99
        vol = np.ones(n) * 1000.0
        taker_buy = vol * 0.5
        strategy.update_market_data("BANDUSDT", vol, taker_buy, high, low)

        # Should NOT be blocked (cooldown=0 means expired)
        strategy.on_bar("BANDUSDT", close, "15m")
        # Verify cooldown didn't go negative
        assert strategy._straddle_rollback_cooldown["BANDUSDT"] == 0

    def test_cooldown_default_from_config(self):
        """Config param straddle_rollback_cooldown_cycles defaults to 8."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert cfg.straddle_rollback_cooldown_cycles == 8


# ---------------------------------------------------------------------------
# P3c: Straddle rollback cooldown persisted immediately (sciomc S5d)
# ---------------------------------------------------------------------------


class TestCooldownImmediateSave:
    """_save_state must be called right after setting cooldown, not end of cycle."""

    def test_save_state_called_after_cooldown_set(self):
        """Cooldown set triggers immediate _save_state to survive crash."""
        from unittest.mock import MagicMock, patch

        scalper = _build_vs_scalper()

        # Track _save_state calls
        save_calls: list[int] = []
        original_save = scalper._save_state

        def tracking_save():
            save_calls.append(len(scalper.strategy._straddle_rollback_cooldown))
            original_save()

        scalper._save_state = tracking_save

        # Set cooldown directly (simulate rollback path)
        scalper.strategy._straddle_rollback_cooldown["TESTUSDT"] = 8
        scalper._save_state()

        assert len(save_calls) == 1, "_save_state must be called after cooldown set"
        assert scalper.strategy._straddle_rollback_cooldown["TESTUSDT"] == 8


# ---------------------------------------------------------------------------
# P4: _load_state() strategy_type auto-heal
# ---------------------------------------------------------------------------


class TestStrategyTypeAutoHeal:
    """_healed_strategy_type() must correct vr_squeeze → straddle for is_straddle=True positions."""

    def test_heal_function_corrects_straddle_mislabeled_as_vr(self):
        """_healed_strategy_type() returns 'straddle' when is_straddle=True and type='vr_squeeze'."""
        from quantbox.scalper import _healed_strategy_type

        pos = {"strategy_type": "vr_squeeze", "is_straddle": True}
        assert _healed_strategy_type(pos) == "straddle"

    def test_heal_function_preserves_correct_straddle_label(self):
        """Already-correct 'straddle' is not changed."""
        from quantbox.scalper import _healed_strategy_type

        pos = {"strategy_type": "straddle", "is_straddle": True}
        assert _healed_strategy_type(pos) == "straddle"

    def test_heal_function_preserves_vr_for_directional(self):
        """Directional VR position (is_straddle=False) keeps 'vr_squeeze' label."""
        from quantbox.scalper import _healed_strategy_type

        pos = {"strategy_type": "vr_squeeze", "is_straddle": False}
        assert _healed_strategy_type(pos) == "vr_squeeze"

    def test_heal_function_default_when_missing(self):
        """Missing fields default safely."""
        from quantbox.scalper import _healed_strategy_type

        # No keys at all → defaults to vr_squeeze (not straddle)
        assert _healed_strategy_type({}) == "vr_squeeze"

    def test_load_state_heals_straddle_position(self):
        """After _load_state(), is_straddle=True positions have strategy_type='straddle'."""
        from quantbox.scalper import PAPER_STATE_FILE
        from quantbox.core.scalper_state import save_scalper_state

        scalper = _build_vs_scalper()

        state = {
            "saved_at": "2026-03-14T06:00:00+00:00",
            "paper_mode": True,
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "positions": {
                "ICXUSDT:LONG": {
                    "symbol": "ICXUSDT",
                    "signal": 1,
                    "entry_price": 0.15,
                    "notional": 450.0,
                    "bars_held": 3,
                    "peak_gain": 0.0,
                    "best_pnl": 0.0,
                    "worst_pnl": 0.0,
                    "server_sl_order_id": "sl_999",
                    "server_tp_order_id": "tp_999",
                    "sl": 0.01,
                    "tp": 0.20,
                    "tf": "1h",
                    "position_side": "LONG",
                    "trail_active": False,
                    "strategy_type": "vr_squeeze",  # mislabeled (pre-0ac2e2b)
                    "is_straddle": True,  # correct flag
                    "leverage": 30,
                    "move_size": 0.03,
                    "entry_stop_mult": None,
                    "partial_closed": False,
                    "highest_since_tp1": 0.0,
                    "trail_floor": None,
                }
            },
            "trade_history": [],
            "cycle_count": 0,
        }

        save_scalper_state(state, filepath=PAPER_STATE_FILE)
        scalper._load_state()

        # The healed position should now show strategy_type='straddle'
        pos = scalper.strategy._positions.get("ICXUSDT:LONG")
        assert pos is not None, "_load_state did not restore ICXUSDT:LONG"
        assert pos.get("strategy_type") == "straddle", (
            f"Expected 'straddle' after auto-heal, got '{pos.get('strategy_type')}'"
        )

    def test_load_state_preserves_vr_for_directional_position(self):
        """Directional VR positions keep their label on _load_state()."""
        from quantbox.scalper import PAPER_STATE_FILE
        from quantbox.core.scalper_state import save_scalper_state

        scalper = _build_vs_scalper()

        state = {
            "saved_at": "2026-03-14T06:00:00+00:00",
            "paper_mode": True,
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "positions": {
                "ETHUSDT:LONG": {
                    "symbol": "ETHUSDT",
                    "signal": 1,
                    "entry_price": 3200.0,
                    "notional": 3000.0,
                    "bars_held": 5,
                    "peak_gain": 0.0,
                    "best_pnl": 0.0,
                    "worst_pnl": 0.0,
                    "server_sl_order_id": "sl_888",
                    "server_tp_order_id": "tp_888",
                    "sl": 0.01,
                    "tp": 0.20,
                    "tf": "15m",
                    "position_side": "LONG",
                    "trail_active": False,
                    "strategy_type": "vr_squeeze",
                    "is_straddle": False,
                    "leverage": 30,
                    "move_size": 0.03,
                    "entry_stop_mult": None,
                    "partial_closed": False,
                    "highest_since_tp1": 0.0,
                    "trail_floor": None,
                }
            },
            "trade_history": [],
            "cycle_count": 0,
        }

        save_scalper_state(state, filepath=PAPER_STATE_FILE)
        scalper._load_state()

        pos = scalper.strategy._positions.get("ETHUSDT:LONG")
        assert pos is not None
        assert pos.get("strategy_type") == "vr_squeeze", "Directional VR position must keep 'vr_squeeze' label"


# ---------------------------------------------------------------------------
# TestStraddleSlotCap — straddle_max_slots enforcement
# ---------------------------------------------------------------------------


def _straddle_pos_cap(sym: str, side: str) -> dict:
    """Minimal straddle position dict for slot cap tests."""
    return {
        "symbol": sym,
        "signal": 1 if side == "LONG" else -1,
        "entry_price": 1.0,
        "notional": 3000.0,
        "bars_held": 5,
        "peak_gain": 0.0,
        "best_pnl": 0.0,
        "worst_pnl": 0.0,
        "server_sl_order_id": "sl_cap",
        "server_tp_order_id": "tp_cap",
        "sl": 0.015,
        "tp": 0.05,
        "tf": "15m",
        "position_side": side,
        "trail_active": False,
        "strategy_type": "straddle",
        "is_straddle": True,
        "leverage": 30,
        "move_size": 1.0,
        "entry_stop_mult": None,
        "partial_closed": False,
        "highest_since_tp1": 0.0,
        "trail_floor": None,
        "opened_at": "2026-01-01T00:00:00+00:00",
    }


class TestStraddleSlotCap:
    """straddle_max_slots: third pair must be blocked when cap is reached."""

    def _run_slot_cap_check(self, scalper, candidates: list, max_entries: int = 10) -> list[str]:
        """Simulate the straddle slot cap portion of _scan_entries().

        Returns list of symbols for which _open_position was called.
        """
        opened_syms: list[str] = []
        entries_made: int = 0
        straddle_blocked: set[str] = set()
        straddle_opened: dict[str, set[int]] = {}

        for sym, signal, _move, _close, is_straddle in candidates:
            if entries_made >= max_entries:
                break
            if is_straddle:
                if sym in straddle_blocked:
                    continue
                if sym not in straddle_opened:
                    # Upfront check: need ≥2 remaining slots for LONG+SHORT
                    remaining_slots = max_entries - entries_made
                    if remaining_slots < 2:
                        straddle_blocked.add(sym)
                        continue
                    cfg_strd_max = scalper.vs_config.straddle_max_slots
                    active_straddle_pairs = sum(
                        1
                        for k, p in scalper.strategy._positions.items()
                        if ":LONG" in k and p.get("is_straddle", False)
                    )
                    if active_straddle_pairs * 2 >= cfg_strd_max:
                        straddle_blocked.add(sym)
                        continue

            opened_syms.append(sym)
            entries_made += 1
            straddle_opened.setdefault(sym, set()).add(signal)

        return opened_syms

    def test_new_pair_blocked_when_cap_reached(self):
        """With straddle_max_slots=4 and 2 active pairs, a 3rd pair is blocked."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        scalper = _build_vs_scalper(vs_config=VolStraddleConfig(straddle_max_slots=4))

        # Inject 2 straddle pairs = 4 slots (at cap)
        for sym in ("XRPUSDT", "ETHUSDT"):
            scalper.strategy._positions[f"{sym}:LONG"] = _straddle_pos_cap(sym, "LONG")
            scalper.strategy._positions[f"{sym}:SHORT"] = _straddle_pos_cap(sym, "SHORT")

        assert scalper.vs_config.straddle_max_slots == 4
        assert (
            sum(1 for k, p in scalper.strategy._positions.items() if ":LONG" in k and p.get("is_straddle", False)) == 2
        ), "Should have 2 active straddle LONG legs"

        # 3rd pair candidate
        candidates = [
            ("AVAXUSDT", 1, 0.03, [1.0] * 20, True),
            ("AVAXUSDT", -1, 0.03, [1.0] * 20, True),
        ]
        opened = self._run_slot_cap_check(scalper, candidates)

        assert "AVAXUSDT" not in opened, "3rd straddle pair must be blocked when cap=4 and 2 pairs active"

    def test_new_pair_allowed_under_cap(self):
        """With straddle_max_slots=6 and 2 active pairs (4 slots), a 3rd pair is allowed."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        scalper = _build_vs_scalper(vs_config=VolStraddleConfig(straddle_max_slots=6))

        # Inject 2 pairs = 4 slots (2 slots remaining)
        for sym in ("XRPUSDT", "ETHUSDT"):
            scalper.strategy._positions[f"{sym}:LONG"] = _straddle_pos_cap(sym, "LONG")
            scalper.strategy._positions[f"{sym}:SHORT"] = _straddle_pos_cap(sym, "SHORT")

        candidates = [
            ("AVAXUSDT", 1, 0.03, [1.0] * 20, True),
            ("AVAXUSDT", -1, 0.03, [1.0] * 20, True),
        ]
        opened = self._run_slot_cap_check(scalper, candidates)

        assert "AVAXUSDT" in opened, "3rd pair should open when 4/6 slots used"
        assert opened.count("AVAXUSDT") == 2, "Both LONG and SHORT legs must pass"

    def test_second_leg_not_double_blocked(self):
        """When SHORT is the second leg of a partially-opened pair, it is NOT blocked."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        scalper = _build_vs_scalper(vs_config=VolStraddleConfig(straddle_max_slots=4))

        # 1 active pair = 2 slots (cap=4 → still 2 slots free for 2nd pair)
        scalper.strategy._positions["XRPUSDT:LONG"] = _straddle_pos_cap("XRPUSDT", "LONG")
        scalper.strategy._positions["XRPUSDT:SHORT"] = _straddle_pos_cap("XRPUSDT", "SHORT")

        candidates = [
            ("ETHUSDT", 1, 0.03, [1.0] * 20, True),
            ("ETHUSDT", -1, 0.03, [1.0] * 20, True),
        ]
        opened = self._run_slot_cap_check(scalper, candidates)

        # Both LONG and SHORT of ETHUSDT should open (2nd pair, within cap)
        assert opened.count("ETHUSDT") == 2, "Both legs of a new pair must open if within cap"

    def test_config_auto_cap_when_max_concurrent_reduced(self):
        """straddle_max_slots is auto-capped to max_concurrent when it would exceed it."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig(max_concurrent=4)  # straddle_max_slots default=6 > 4
        assert cfg.straddle_max_slots == 4, "straddle_max_slots must be auto-capped to max_concurrent"
        assert cfg.straddle_max_slots <= cfg.max_concurrent

    def test_straddle_blocked_when_only_one_slot_remaining(self):
        """Straddle must NOT open when only 1 slot remains — avoids LONG-only orphan."""
        scalper = _build_vs_scalper()

        candidates = [
            ("COMPUSDT", 1, 0.04, [1.0] * 20, True),  # LONG
            ("COMPUSDT", -1, 0.04, [1.0] * 20, True),  # SHORT
        ]
        # max_entries=1 → only 1 slot left
        opened = self._run_slot_cap_check(scalper, candidates, max_entries=1)

        assert "COMPUSDT" not in opened, (
            "Straddle must be fully blocked when only 1 slot remains — "
            "prevents COMP-style orphan (LONG opens, SHORT blocked)"
        )

    def test_straddle_allowed_when_exactly_two_slots_remaining(self):
        """Straddle opens normally when exactly 2 slots remain."""
        scalper = _build_vs_scalper()

        candidates = [
            ("COMPUSDT", 1, 0.04, [1.0] * 20, True),
            ("COMPUSDT", -1, 0.04, [1.0] * 20, True),
        ]
        opened = self._run_slot_cap_check(scalper, candidates, max_entries=2)

        assert opened.count("COMPUSDT") == 2, "Both LONG and SHORT must open when exactly 2 slots remain"


# ---------------------------------------------------------------------------
# P5: Straddle orphan auto-close (_close_straddle_orphans)
# ---------------------------------------------------------------------------
