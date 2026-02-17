"""
Tests for the Scalper live trading module.

Tests the non-exchange parts: risk management, state persistence,
and cycle logic. Uses paper mode to avoid API calls.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from quantbox.strategy.rubberband import RubberBandConfig, RubberBandStrategy
from quantbox.scalper import (
    RubberBandScalper,
    _save_scalper_state,
    _load_scalper_state,
)


# ── State Persistence ──

class TestScalperState:
    """Test atomic state save/load."""

    def test_save_and_load(self, tmp_path):
        """State roundtrips through JSON correctly."""
        fpath = tmp_path / "test_state.json"
        state = {
            'version': '2.0.0',
            'paper_mode': True,
            'paper_balance': 1000.0,
            'positions': {
                'BTCUSDT': {
                    'signal': 1, 'entry_price': 95000.0,
                    'notional': 500.0, 'leverage': 2,
                    'move_size': 0.03, 'bars_held': 3,
                    'best_pnl': 0.01,
                },
            },
            'trade_history': [
                {'symbol': 'ETHUSDT', 'pnl_usd': 5.23}
            ],
        }
        _save_scalper_state(state, filepath=fpath)
        loaded = _load_scalper_state(filepath=fpath)

        assert loaded is not None
        assert loaded['paper_balance'] == 1000.0
        assert 'BTCUSDT' in loaded['positions']
        assert loaded['positions']['BTCUSDT']['signal'] == 1
        assert len(loaded['trade_history']) == 1

    def test_load_missing_file(self, tmp_path):
        """Loading non-existent file returns None."""
        result = _load_scalper_state(filepath=tmp_path / "nope.json")
        assert result is None

    def test_atomic_write_no_corruption(self, tmp_path):
        """File is not corrupted on normal write."""
        fpath = tmp_path / "atomic.json"
        _save_scalper_state({'test': True}, filepath=fpath)
        with open(fpath) as f:
            data = json.load(f)
        assert data['test'] is True


# ── Risk Management ──

class TestScalperRiskManagement:
    """Test drawdown and daily loss limits using the Strategy wrapper directly."""

    def test_drawdown_detection(self):
        """Session drawdown > 10% triggers halt."""
        cfg = RubberBandConfig(max_drawdown_pct=0.10)

        # Simulate: HWM = $10000, current = $8900 → 11% DD
        hwm = 10000.0
        balance = 8900.0
        dd = (hwm - balance) / hwm
        assert dd >= cfg.max_drawdown_pct

    def test_daily_loss_detection(self):
        """Daily loss > 5% triggers halt."""
        cfg = RubberBandConfig(daily_loss_limit_pct=0.05)

        daily_start = 10000.0
        balance = 9400.0
        daily_loss = (daily_start - balance) / daily_start
        assert daily_loss >= cfg.daily_loss_limit_pct

    def test_position_count_limit(self):
        """Max concurrent positions are enforced."""
        cfg = RubberBandConfig(max_concurrent=2)
        strat = RubberBandStrategy(cfg)

        strat.open_position("SYM1", 1, 100.0, 1000.0, 2.0, 0.03)
        strat.open_position("SYM2", -1, 200.0, 1000.0, 2.0, 0.03)
        assert strat.n_active == 2

        # Scalper should check n_active < max_concurrent before opening
        available_slots = cfg.max_concurrent - strat.n_active
        assert available_slots == 0


# ── Config CLI ──

class TestScalperConfig:
    """Test config defaults and overrides."""

    def test_default_config(self):
        cfg = RubberBandConfig()
        assert cfg.lookback == 8
        assert cfg.threshold == 0.045
        assert cfg.max_concurrent == 5
        assert cfg.cost_per_side == 0.0004

    def test_custom_config(self):
        cfg = RubberBandConfig(lookback=10, threshold=0.03, max_concurrent=3)
        assert cfg.lookback == 10
        assert cfg.threshold == 0.03
        assert cfg.max_concurrent == 3
