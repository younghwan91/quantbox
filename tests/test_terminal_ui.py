"""Tests for quantbox.core.terminal_ui — startup banner and formatted output."""

from datetime import datetime, timezone

import pytest

from quantbox.core.terminal_ui import (
    Color,
    format_cycle_header,
    format_entry,
    format_exit,
    format_scan_summary,
    format_session_summary,
    print_banner,
    print_system_info,
    print_startup_complete,
)


class TestFormatCycleHeader:
    """Cycle header contains essential info."""

    def test_contains_cycle_number(self):
        result = format_cycle_header(42, 1000.0, 2, 5)
        assert '42' in result

    def test_contains_balance(self):
        result = format_cycle_header(1, 5432.10, 0, 5)
        assert '5,432.10' in result

    def test_contains_slots(self):
        result = format_cycle_header(1, 1000.0, 3, 5)
        assert '3/5' in result


class TestFormatEntry:
    """Entry line shows direction, symbol, price, leverage, size."""

    def test_long_entry(self):
        result = format_entry('BTCUSDT', 'LONG', 50000.0, 5, 2500.0, confidence=4)
        assert 'LONG' in result
        assert 'BTCUSDT' in result
        assert '50000' in result
        assert '5x' in result
        assert 'C=4' in result

    def test_short_entry(self):
        result = format_entry('ETHUSDT', 'SHORT', 3000.0, 3, 1500.0)
        assert 'SHORT' in result
        assert 'ETHUSDT' in result


class TestFormatExit:
    """Exit line shows PnL, direction, reason."""

    def test_winning_trade(self):
        result = format_exit(
            'BTCUSDT', 'LONG', 50000.0, 51000.0,
            pnl_usd=48.50, pnl_pct=2.0, reason='target', bars_held=3,
        )
        assert '+' in result
        assert '48.50' in result
        assert 'target' in result
        assert '3bars' in result

    def test_losing_trade(self):
        result = format_exit(
            'ETHUSDT', 'SHORT', 3000.0, 3100.0,
            pnl_usd=-25.0, pnl_pct=-3.33, reason='stop', bars_held=8,
        )
        assert '-25.00' in result
        assert 'stop' in result


class TestFormatScanSummary:
    """Scan summary reports candidates or top movers."""

    def test_with_candidates(self):
        result = format_scan_summary(3, [('BTC', 0.05)], 0.032)
        assert '3 candidates' in result

    def test_no_candidates_shows_movers(self):
        movers = [('BTC', 0.028), ('ETH', 0.02), ('SOL', 0.015)]
        result = format_scan_summary(0, movers, 0.032)
        assert 'BTC' in result
        assert '>3.2%' in result

    def test_no_candidates_no_movers(self):
        result = format_scan_summary(0, [], 0.032)
        assert 'No signals' in result


class TestFormatSessionSummary:
    """Session summary contains key metrics."""

    def test_contains_metrics(self):
        result = format_session_summary(
            cycles=96, n_trades=12, win_rate=66.7,
            total_pnl=142.50, balance=10142.50, session_roi=1.43,
            best_trade={'symbol': 'BTCUSDT', 'pnl_usd': 55.0},
            worst_trade={'symbol': 'ETHUSDT', 'pnl_usd': -20.0},
        )
        assert '96' in result
        assert '12' in result
        assert '142.50' in result
        assert 'BTCUSDT' in result
        assert 'ETHUSDT' in result


class TestPrintBanner:
    """Banner prints without error."""

    def test_prints_quantbox(self, capsys):
        # Disable colors for assertion simplicity
        Color.disable()
        print_banner()
        captured = capsys.readouterr()
        # ASCII art uses block characters — check for the structure
        assert '██' in captured.out


class TestPrintSystemInfo:
    """System info displays config values."""

    def test_prints_without_error(self, capsys):
        Color.disable()
        print_system_info(
            mode='PAPER', strategy='rubberband', balance=10000.0,
            n_symbols=20, max_slots=5, leverage_base=3, leverage_max=10,
            lookback=8, threshold=0.032, target=0.5, stop_mult=1.2,
            max_hold=6, entry_mode='confidence', min_confidence=3,
        )
        captured = capsys.readouterr()
        assert 'PAPER' in captured.out
        assert '10,000.00' in captured.out
        assert '20 symbols' in captured.out
