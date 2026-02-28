"""Test: MTF 4h confirmation must NOT use current developing bar (lookahead).

Incident: Parity backtest showed 12 trades (83% WR) vs live 25 trades (36% WR).
Root cause: apply_mtf_confirmation used `j = i // ratio` which references
the CURRENT (developing) 4h bar whose final close was unknown at the time
of the 15m signal. This lookahead biased the filter toward perfect hindsight.

Fix: Use `j = i // ratio - 1` (last COMPLETED 4h bar).

These tests ensure:
  1. A signal at the START of a 4h bar uses the PREVIOUS bar's trend
  2. A signal at the END of a 4h bar still uses the PREVIOUS bar's trend
  3. Trend reversal within a 4h period doesn't leak into earlier 15m bars
"""

import numpy as np
import pytest
from quantbox.strategy.rubberband import apply_mtf_confirmation


def test_mid_bar_uses_previous_completed_bar():
    """Signal at 15m bar inside a 4h period should use PREVIOUS 4h bar.

    Setup: 4h bar j=5 is in downtrend (close < SMA), but j=4 was uptrend.
    A LONG signal at 15m bar 84 (inside 4h bar j=5) should check j=4,
    which is uptrend → LONG passes.
    With the bug (j=5), LONG would be blocked by the downtrend.
    """
    n_4h = 20
    close_4h = np.full(n_4h, 150.0)
    # Bars 0-4: uptrend (close > SMA)
    close_4h[:5] = np.linspace(140, 170, 5)
    # Bar 5: sudden crash → downtrend
    close_4h[5] = 120.0
    close_4h[6:] = np.linspace(118, 100, n_4h - 6)

    n_15m = n_4h * 16
    signals = np.zeros(n_15m)
    # Signal at bar 84 = 15m bar in 4h period j=5 (84//16=5)
    # Should use j=4 (completed, uptrend)
    signals[84] = 1.5  # LONG

    filtered = apply_mtf_confirmation(signals, close_4h, sma_period_4h=3, ratio=16)
    # j=4 is in uptrend → LONG should PASS
    assert filtered[84] == 1.5, (
        "LONG at start of 4h downtrend bar should pass using PREVIOUS bar's uptrend"
    )


def test_4h_boundary_also_uses_previous():
    """Signal at exact 4h bar boundary (i % ratio == 0) uses previous bar.

    At bar open, the new 4h bar has literally zero data. Must use j-1.
    """
    n_4h = 20
    close_4h = np.full(n_4h, 100.0)
    # Make bars 0-9 uptrend, bar 10 downtrend
    close_4h[:10] = np.linspace(90, 130, 10)
    close_4h[10] = 80.0  # crash at 4h bar 10

    n_15m = n_4h * 16
    signals = np.zeros(n_15m)
    # Signal at bar 160 = exact boundary of 4h bar j=10 (160//16=10)
    signals[160] = 1.5  # LONG

    filtered = apply_mtf_confirmation(signals, close_4h, sma_period_4h=3, ratio=16)
    # j=9 is in uptrend → LONG should pass (not blocked by j=10 downtrend)
    assert filtered[160] == 1.5, (
        "LONG at 4h boundary should use PREVIOUS completed bar"
    )


def test_trend_reversal_no_leakage():
    """4h bar that reverses mid-period shouldn't affect 15m bars at its start.

    4h bar j=6 starts up, ends down. The 15m bars during j=6 should
    NOT know the bar will end down — they use j=5 (completed).
    """
    n_4h = 20
    # Steady uptrend through bar 5, then bar 6 closes DOWN
    close_4h = np.linspace(100, 160, n_4h)
    close_4h[6] = 90.0  # Bar 6 closes much lower than it opened
    # But j=5 (completed) close = ~131.6, SMA(3) ≈ ~121 → uptrend

    n_15m = n_4h * 16
    signals = np.zeros(n_15m)
    # Several 15m bars during 4h period j=6
    signals[96] = -1.5   # SHORT at start of 4h bar 6
    signals[100] = -1.5  # SHORT mid-bar
    signals[110] = -1.5  # SHORT near end

    filtered = apply_mtf_confirmation(signals, close_4h, sma_period_4h=3, ratio=16)
    # j=5 is uptrend → SHORT should be BLOCKED (correct: short in uptrend)
    assert filtered[96] == 0.0, "SHORT in uptrend (using j=5) should be blocked"
    assert filtered[100] == 0.0, "SHORT in uptrend (using j=5) should be blocked"
    assert filtered[110] == 0.0, "SHORT in uptrend (using j=5) should be blocked"
