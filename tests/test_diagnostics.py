"""Tests for v12.0 diagnostic logging infrastructure.

Validates:
1. diagnose_bar() returns correct intermediate values without side effects
2. DiagnosticLogger writes valid JSONL
3. Zero impact on strategy signal generation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quantbox.strategy.rubberband import RubberBandConfig, RubberBandStrategy


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def strategy():
    """Default v10 strategy with confidence mode + MTF."""
    cfg = RubberBandConfig(
        lookback=8,
        threshold=0.032,
        entry_mode='confidence',
        min_confidence=1,
        mtf_confirmation_enabled=True,
        mtf_sma_period=12,
        macd_enabled=True,
        stoch_enabled=True,
    )
    return RubberBandStrategy(cfg)


@pytest.fixture
def make_prices():
    """Factory for synthetic price arrays with a controlled move."""
    def _make(n: int = 200, base: float = 100.0, move_pct: float = 0.0):
        prices = np.full(n, base)
        if move_pct != 0:
            # Last bar moves by move_pct relative to bar[-9]
            prices[-1] = base * (1 + move_pct)
        return prices
    return _make


@pytest.fixture
def btc_prices():
    """Flat BTC prices (neutral regime)."""
    return np.full(200, 95000.0)


# ─── diagnose_bar tests ─────────────────────────────────────────────


class TestDiagnoseBar:
    """diagnose_bar() correctness without side effects."""

    def test_returns_dict_with_required_keys(self, strategy, make_prices, btc_prices):
        close = make_prices(200, 100.0)
        strategy.update_market_data('ETHUSDT', np.ones(200), np.ones(200) * 0.5)
        strategy.update_htf_data('ETHUSDT', np.full(100, 95000.0))

        diag = strategy.diagnose_bar('ETHUSDT', close, btc_prices)

        required = {
            'symbol', 'price', 'move_8bar', 'threshold', 'passed_threshold',
            'regime', 'btc_mom', 'asset_trend', 'entry_type', 'confidence',
            'min_confidence', 'passed_confidence', 'evidence', 'raw_signal',
            'mtf_trend_up', 'mtf_blocked', 'final_signal',
        }
        assert required.issubset(diag.keys()), f"Missing: {required - diag.keys()}"

    def test_flat_market_no_threshold(self, strategy, btc_prices):
        """Flat market → passed_threshold = False, confidence stays 0."""
        close = np.full(200, 100.0)
        strategy.update_market_data('ETHUSDT', np.ones(200), np.ones(200) * 0.5)

        diag = strategy.diagnose_bar('ETHUSDT', close, btc_prices)

        assert diag['passed_threshold'] is False
        assert diag['confidence'] == 0
        assert diag['final_signal'] == 0
        assert abs(diag['move_8bar']) < 0.001

    def test_big_move_passes_threshold(self, strategy, btc_prices):
        """5% drop → passed_threshold True, entry_type set."""
        close = np.full(200, 100.0)
        close[-1] = 95.0  # -5% move
        vol = np.ones(200) * 1000
        taker = vol * 0.4  # sell pressure
        strategy.update_market_data('ETHUSDT', vol, taker)
        strategy.update_htf_data('ETHUSDT', np.full(100, 95000.0))

        diag = strategy.diagnose_bar('ETHUSDT', close, btc_prices)

        assert diag['passed_threshold'] is True
        assert diag['move_8bar'] == pytest.approx(-0.05, abs=0.001)
        assert diag['entry_type'] is not None
        assert diag['confidence'] >= 1

    def test_evidence_contains_indicators(self, strategy, btc_prices):
        """Evidence dict has all indicator values."""
        close = np.full(200, 100.0)
        close[-1] = 95.0
        vol = np.ones(200) * 1000
        taker = vol * 0.5
        strategy.update_market_data('ETHUSDT', vol, taker)

        diag = strategy.diagnose_bar('ETHUSDT', close, btc_prices)
        ev = diag['evidence']

        for key in ('rsi', 'vol_ratio', 'taker_ratio', 'sma_dist',
                     'stoch_k', 'stoch_d', 'macd_hist', 'macd_hist_prev'):
            assert key in ev, f"Missing evidence: {key}"
            assert isinstance(ev[key], float), f"{key} not float: {type(ev[key])}"

    def test_no_side_effects_on_state(self, strategy, btc_prices):
        """diagnose_bar() must not alter strategy internal state."""
        close = np.full(200, 100.0)
        close[-1] = 95.0
        vol = np.ones(200) * 1000
        taker = vol * 0.5
        strategy.update_market_data('ETHUSDT', vol, taker)
        strategy.update_htf_data('ETHUSDT', np.full(100, 95000.0))

        # Snapshot state before
        pos_before = dict(strategy._positions)
        mtf_before = dict(strategy._mtf_confirmed)

        strategy.diagnose_bar('ETHUSDT', close, btc_prices)
        strategy.diagnose_bar('ETHUSDT', close, btc_prices)  # Call twice

        # State unchanged
        assert strategy._positions == pos_before
        assert strategy._mtf_confirmed == mtf_before

    def test_diagnose_matches_on_bar_signal(self, strategy, btc_prices):
        """diagnose_bar().final_signal should match on_bar() output."""
        close = np.full(200, 100.0)
        close[-1] = 95.0
        vol = np.ones(200) * 1000
        taker = vol * 0.4
        strategy.update_market_data('ETHUSDT', vol, taker)
        strategy.update_htf_data('ETHUSDT', np.full(100, 95000.0))

        signal = strategy.on_bar('ETHUSDT', close, btc_prices)
        diag = strategy.diagnose_bar('ETHUSDT', close, btc_prices)

        # If on_bar returns non-zero, diagnose should agree
        # (Note: on_bar has full numba pipeline, diagnose uses Python mirror.
        #  They should agree on direction at minimum.)
        if signal != 0:
            assert diag['final_signal'] == signal, (
                f"on_bar={signal} vs diagnose={diag['final_signal']}"
            )

    def test_mtf_blocking_reported(self, strategy, btc_prices):
        """When MTF blocks a signal, mtf_blocked should be True."""
        close = np.full(200, 100.0)
        close[-1] = 95.0  # -5% → wants to go long
        vol = np.ones(200) * 1000
        taker = vol * 0.4
        strategy.update_market_data('ETHUSDT', vol, taker)

        # 4h data trending DOWN → should block long
        declining_4h = np.linspace(100, 90, 100)
        strategy.update_htf_data('ETHUSDT', declining_4h)

        diag = strategy.diagnose_bar('ETHUSDT', close, btc_prices)

        if diag['raw_signal'] == 1:
            assert diag['mtf_blocked'] is True
            assert diag['final_signal'] == 0


# ─── DiagnosticLogger tests ─────────────────────────────────────────


class TestDiagnosticLogger:
    """JSONL logging correctness."""

    def test_scan_writes_valid_jsonl(self):
        from quantbox.scalper import DiagnosticLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DiagnosticLogger(log_dir=Path(tmpdir))
            dl.log_scan(1, '2026-02-18T00:00:00Z', [
                {'symbol': 'ETHUSDT', 'move_8bar': -0.015, 'passed_threshold': False},
                {'symbol': 'SOLUSDT', 'move_8bar': 0.035, 'passed_threshold': True},
            ])
            # Force flush
            for h in dl._logger.handlers:
                h.flush()

            jsonl_path = Path(tmpdir) / 'diagnostics.jsonl'
            assert jsonl_path.exists()

            lines = jsonl_path.read_text().strip().split('\n')
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record['type'] == 'scan'
            assert record['cycle'] == 1
            assert len(record['symbols']) == 2
            assert record['symbols'][0]['symbol'] == 'ETHUSDT'

    def test_entry_exit_writes(self):
        from quantbox.scalper import DiagnosticLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DiagnosticLogger(log_dir=Path(tmpdir))
            dl.log_entry(5, '2026-02-18T01:00:00Z', 'XRPUSDT', 1,
                         {'confidence': 3, 'regime': 'risk_off'},
                         {'notional': 500.0, 'leverage': 3})
            dl.log_exit(8, '2026-02-18T02:00:00Z',
                        {'symbol': 'XRPUSDT', 'pnl_pct': 1.2, 'reason': 'target'})
            for h in dl._logger.handlers:
                h.flush()

            lines = (Path(tmpdir) / 'diagnostics.jsonl').read_text().strip().split('\n')
            assert len(lines) == 2

            entry = json.loads(lines[0])
            assert entry['type'] == 'entry'
            assert entry['symbol'] == 'XRPUSDT'

            exit_rec = json.loads(lines[1])
            assert exit_rec['type'] == 'exit'
            assert exit_rec['trade']['reason'] == 'target'

    def test_graceful_failure(self):
        """Logger should not crash if path is invalid."""
        from quantbox.scalper import DiagnosticLogger

        dl = DiagnosticLogger(log_dir=Path('/nonexistent/path/that/wont/work'))
        # Should not raise
        dl.log_scan(1, '2026-01-01', [{'x': 1}])


# ─── Zero-impact verification ───────────────────────────────────────


class TestZeroImpact:
    """Verify diagnostics don't change strategy behavior."""

    def test_on_bar_identical_with_and_without_diagnose(self, btc_prices):
        """Signal output must be identical whether diagnose_bar is called or not."""
        cfg = RubberBandConfig(threshold=0.032, min_confidence=1,
                               macd_enabled=True, stoch_enabled=True)

        s1 = RubberBandStrategy(cfg)
        s2 = RubberBandStrategy(cfg)

        close = np.full(200, 100.0)
        close[-1] = 95.0
        vol = np.ones(200) * 1000
        taker = vol * 0.45

        for s in (s1, s2):
            s.update_market_data('TEST', vol, taker)
            s.update_htf_data('TEST', np.full(100, 95000.0))

        # s1: on_bar only
        sig1 = s1.on_bar('TEST', close, btc_prices)

        # s2: on_bar + diagnose_bar
        sig2 = s2.on_bar('TEST', close, btc_prices)
        _ = s2.diagnose_bar('TEST', close, btc_prices)

        assert sig1 == sig2
        assert s1._positions == s2._positions
        assert s1._mtf_confirmed == s2._mtf_confirmed
