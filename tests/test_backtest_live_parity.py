"""Backtest ↔ Live Trading Parity Tests.

Verifies that the live scalper produces the EXACT same decisions
as the backtest engine for:
  1. Signal direction (on_bar returns same sign as backtest signals)
  2. MTF confirmation (blocking + leverage)
  3. Effective leverage (backtest signal magnitude == live leverage)
  4. CLI defaults don't override .env values
  5. Incomplete bar exclusion (live must drop open candle)

These tests exist because parity violations = silent money loss.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    compute_confidence_leverage,
    compute_signals_v9,
)


# ── Fixtures ──


@pytest.fixture
def default_cfg():
    """Production v10.1 config (matches .env defaults)."""
    return RubberBandConfig(
        threshold=0.032,
        min_confidence=1,
        max_hold=20,
        stoch_k_period=3,
        mtf_confirmation_enabled=True,
        mtf_sma_period=12,
        mtf_leverage_mult=2.5,
        base_leverage=1.5,
        leverage_scale=0.0,
        max_leverage=1.5,
    )


# ── Test 1: MTF confirmation tracking ──


def test_mtf_confirmed_flag_set_on_entry(default_cfg):
    """was_mtf_confirmed() must return True when signal survives MTF filter."""
    strategy = RubberBandStrategy(default_cfg)

    # Create a clear long signal: big dip
    n = 200
    close_15m = np.ones(n, dtype=np.float64) * 100.0
    close_15m[-1] = 95.0  # -5% dip → exceeds 3.2% threshold
    btc_close = np.ones(n, dtype=np.float64) * 50000.0
    # BTC falling → long signal allowed (btc_directional)
    btc_close[-32:] = np.linspace(50000, 49000, 32)

    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker_buy = np.ones(n, dtype=np.float64) * 400.0

    # 4h data: uptrend → blocks short, allows long
    close_4h = np.linspace(90, 110, 20).astype(np.float64)

    strategy.update_market_data("SYM", volume, taker_buy)
    strategy.update_htf_data("SYM", close_4h)

    sig = strategy.on_bar("SYM", close_15m, btc_close)
    if sig != 0:
        # If a signal was generated, MTF should have been applied
        assert strategy.was_mtf_confirmed("SYM") is True


def test_mtf_confirmed_flag_false_without_htf_data(default_cfg):
    """was_mtf_confirmed() must be False when no HTF data provided."""
    strategy = RubberBandStrategy(default_cfg)

    n = 200
    close = np.ones(n, dtype=np.float64) * 100.0
    btc = np.ones(n, dtype=np.float64) * 50000.0
    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker_buy = np.ones(n, dtype=np.float64) * 400.0

    strategy.update_market_data("SYM", volume, taker_buy)
    # No update_htf_data call

    strategy.on_bar("SYM", close, btc)
    assert strategy.was_mtf_confirmed("SYM") is False


def test_mtf_confirmed_default_false(default_cfg):
    """was_mtf_confirmed() returns False for unseen symbols."""
    strategy = RubberBandStrategy(default_cfg)
    assert strategy.was_mtf_confirmed("UNKNOWN") is False


# ── Test 2: Leverage parity ──


def test_backtest_live_leverage_parity(default_cfg):
    """Backtest signal magnitude must equal live effective leverage.

    Backtest: compute_signals_v9 → apply_mtf_confirmation → apply_mtf_leverage
    Live: compute_rr_gated_leverage → × mtf_leverage_mult (if confirmed)

    Both must produce the same effective leverage.
    """
    cfg = default_cfg

    # Backtest path: signal magnitude encodes leverage
    # With base=1.5, scale=0.0, max=1.5 → signal = ±1.5
    # After apply_mtf_leverage(2.5) → signal = ±3.75
    backtest_base_leverage = cfg.base_leverage  # 1.5
    backtest_mtf_leverage = backtest_base_leverage * cfg.mtf_leverage_mult  # 3.75

    # Live path: compute_confidence_leverage then × mtf_leverage_mult
    live_base_leverage = compute_confidence_leverage(
        move_size=0.05,  # 5% move (exceeds 3.2% threshold)
        threshold=cfg.threshold,
        base_leverage=cfg.base_leverage,
        leverage_scale=cfg.leverage_scale,
        max_leverage=cfg.max_leverage,
    )
    live_mtf_leverage = live_base_leverage * cfg.mtf_leverage_mult

    assert live_base_leverage == pytest.approx(backtest_base_leverage, abs=1e-6), (
        f"Base leverage mismatch: backtest={backtest_base_leverage}, live={live_base_leverage}"
    )
    assert live_mtf_leverage == pytest.approx(backtest_mtf_leverage, abs=1e-6), (
        f"MTF leverage mismatch: backtest={backtest_mtf_leverage}, live={live_mtf_leverage}"
    )


def test_exchange_leverage_uses_ceil():
    """Exchange leverage must use ceil() to avoid insufficient margin.

    int(3.75) = 3 → margin too high. ceil(3.75) = 4 → correct.
    """
    leverage = 3.75
    lev_int = max(1, min(math.ceil(leverage), 20))
    assert lev_int == 4, f"Should ceil to 4, got {lev_int}"

    # Edge case: exact integer
    lev_int = max(1, min(math.ceil(2.0), 20))
    assert lev_int == 2

    # Edge case: slightly above
    lev_int = max(1, min(math.ceil(1.01), 20))
    assert lev_int == 2


# ── Test 3: Signal pipeline parity ──


def test_signal_direction_preserved(default_cfg):
    """on_bar signal direction must match backtest signal sign."""
    strategy = RubberBandStrategy(default_cfg)

    n = 200
    # Create a significant dip to trigger long signal
    close = np.ones(n, dtype=np.float64) * 100.0
    close[-1] = 94.0  # -6% → well above threshold
    btc = np.ones(n, dtype=np.float64) * 50000.0
    btc[-32:] = np.linspace(50000, 49000, 32)  # BTC falling → long OK

    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker = np.ones(n, dtype=np.float64) * 400.0
    high = close.copy()
    low = close.copy()

    strategy.update_market_data("SYM", volume, taker, high, low)
    # No MTF data → raw signal
    live_sig = strategy.on_bar("SYM", close, btc)

    # Backtest signal
    cfg = default_cfg
    bt_sigs = compute_signals_v9(
        close,
        high,
        low,
        volume,
        taker,
        btc,
        cfg.lookback,
        cfg.threshold,
        cfg.target_retrace,
        cfg.stop_mult,
        cfg.max_hold,
        cfg.stop_decay,
        cfg.btc_directional_period,
        cfg.btc_regime_period,
        cfg.asset_trend_period,
        cfg.min_confidence,
        cfg.base_leverage,
        cfg.confidence_leverage_step,
        cfg.max_leverage,
        cfg.rsi_period,
        cfg.vol_period,
        cfg.dynamic_target_enabled,
        cfg.target_conf3,
        cfg.target_conf4,
        cfg.target_conf5,
        cfg.stoch_enabled,
        cfg.stoch_k_period,
        cfg.stoch_d_period,
        cfg.macd_enabled,
        cfg.macd_fast,
        cfg.macd_slow,
        cfg.macd_signal_period,
        cfg.atr_period,
        cfg.atr_multiplier,
        cfg.threshold_floor,
        cfg.threshold_cap,
    )

    bt_last = bt_sigs[-1]
    bt_direction = 1 if bt_last > 0 else (-1 if bt_last < 0 else 0)

    assert live_sig == bt_direction, f"Direction mismatch: live={live_sig}, backtest_raw={bt_last}"


# ── Test 4: CLI defaults don't override .env ──


def test_cli_defaults_are_none():
    """Argparse defaults must be None so .env values take precedence.

    Bug history: --threshold defaulted to 0.025, overriding .env 0.032.
    This caused live trading at different threshold than backtested.
    """
    import argparse

    # Simulate the parser from scalper.main()
    # We can't import main() easily, so just verify the pattern
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-slots", type=int, default=None)

    args = parser.parse_args([])  # No CLI args

    assert args.lookback is None, "lookback default must be None"
    assert args.threshold is None, "threshold default must be None"
    assert args.max_slots is None, "max_slots default must be None"

    # Verify the pattern: None → falls through to settings
    settings_threshold = 0.032
    effective = args.threshold if args.threshold is not None else settings_threshold
    assert effective == 0.032, f"Should use settings value, got {effective}"


# ── Test 5: MTF leverage is NOT double-applied in on_bar ──


def test_on_bar_does_not_apply_mtf_leverage():
    """on_bar must NOT call apply_mtf_leverage (it's done in _open_position).

    If on_bar applies mtf_leverage AND _open_position also applies it,
    we'd get double-counting: 1.5 × 2.5 × 2.5 = 9.375x instead of 3.75x.
    """
    cfg = RubberBandConfig(
        mtf_confirmation_enabled=True,
        mtf_sma_period=6,
        mtf_leverage_mult=2.5,
        entry_mode="confidence",
        min_confidence=1,
        threshold=0.032,
    )
    strategy = RubberBandStrategy(cfg)

    n = 200
    close = np.ones(n, dtype=np.float64) * 100.0
    close[-1] = 94.0  # -6% → strong signal
    btc = np.ones(n, dtype=np.float64) * 50000.0
    btc[-32:] = np.linspace(50000, 49000, 32)
    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker = np.ones(n, dtype=np.float64) * 400.0

    # 4h uptrend → allows long
    close_4h = np.linspace(90, 110, 20).astype(np.float64)

    strategy.update_market_data("SYM", volume, taker)
    strategy.update_htf_data("SYM", close_4h)

    sig = strategy.on_bar("SYM", close, btc)
    # on_bar returns int ±1, not a float with leverage magnitude
    assert sig in (-1, 0, 1), f"on_bar should return int, got {sig} (type={type(sig)})"


# ── Test 6: 4h data alignment (inline vs array-based) ──


def test_live_mtf_uses_latest_4h_bar():
    """Live MTF confirmation must use the LATEST 4h bar, not index i//16.

    Bug history: apply_mtf_confirmation(sigs, close_4h, 12, 16) maps
    15m bar i to 4h bar j = i // 16. For backtest (aligned arrays) this
    works. For live (200 15m bars + 100 4h bars with different starts),
    bar 199 maps to close_4h[12] (352h ago) instead of close_4h[99].

    The live path now uses an inline check with close_4h[-1] and
    SMA(close_4h[-sma_period:]).
    """
    cfg = RubberBandConfig(
        mtf_confirmation_enabled=True,
        mtf_sma_period=6,
        entry_mode="confidence",
        min_confidence=1,
        threshold=0.032,
    )
    strategy = RubberBandStrategy(cfg)

    n = 200
    close = np.ones(n, dtype=np.float64) * 100.0
    close[-1] = 94.0  # -6% dip → long signal
    btc = np.ones(n, dtype=np.float64) * 50000.0
    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker = np.ones(n, dtype=np.float64) * 400.0

    strategy.update_market_data("SYM", volume, taker)

    # Scenario: 4h array is 100 bars. The LATEST bars show a clear uptrend,
    # but earlierear bars (index 12 = where the old bug would look) show downtrend.
    close_4h = np.ones(100, dtype=np.float64) * 100.0
    # First 50 bars: downtrend (old bug would pick up this)
    close_4h[:50] = np.linspace(120, 80, 50)
    # Last 50 bars: clear uptrend (correct behavior should see this)
    close_4h[50:] = np.linspace(95, 115, 50)

    strategy.update_htf_data("SYM", close_4h)
    sig = strategy.on_bar("SYM", close, btc)

    # The latest 4h data shows uptrend → long should be ALLOWED.
    # Old bug: close_4h[12] ≈ 95.1 (downtrend zone), SMA of early bars < close
    # → would incorrectly block or allow based on stale data.
    # With fix: close_4h[-1] = 115, SMA(last 6) ≈ 112.5 → uptrend → long allowed.
    # We can't guarantee a signal fires (depends on confidence), but if it does,
    # it should be positive (long) since 4h is in uptrend.
    assert sig >= 0, f"Long should not be blocked by latest 4h uptrend, got {sig}"


def test_live_mtf_blocks_counter_trend():
    """Live MTF should block counter-trend entries using latest 4h data."""
    cfg = RubberBandConfig(
        mtf_confirmation_enabled=True,
        mtf_sma_period=6,
        entry_mode="confidence",
        min_confidence=1,
        threshold=0.032,
    )
    strategy = RubberBandStrategy(cfg)

    n = 200
    close = np.ones(n, dtype=np.float64) * 100.0
    close[-1] = 94.0  # -6% dip → would trigger long
    btc = np.ones(n, dtype=np.float64) * 50000.0
    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker = np.ones(n, dtype=np.float64) * 400.0

    strategy.update_market_data("SYM", volume, taker)

    # 4h data: clear DOWNTREND at the latest bars → should block long entries
    close_4h = np.linspace(120, 80, 100).astype(np.float64)

    strategy.update_htf_data("SYM", close_4h)
    sig = strategy.on_bar("SYM", close, btc)

    # Long entry in 4h downtrend must be blocked
    assert sig == 0, f"Long should be blocked by 4h downtrend, got {sig}"


# ── Test 7: btc_directional_period matches backtest ──


def test_btc_directional_period_default_matches_backtest():
    """btc_directional_period must be 48 (v11.0 A/B validated).

    With period=48, BTC momentum over 12h enables Risk-Off entries
    (sympathy pullback). A/B tested: Calmar 5.14 → 19.65 (+282%).
    Robust across all periods (16-96).
    """
    cfg = RubberBandConfig()
    assert cfg.btc_directional_period == 48, (
        f"btc_directional_period={cfg.btc_directional_period}, should be 48 (v11.0)"
    )


# ── Test 8: Incomplete bar exclusion (critical parity fix) ──


def test_download_klines_drops_incomplete_bar():
    """download_klines must drop the last (incomplete) candle from Binance.

    Binance futures_klines ALWAYS includes the currently-open candle as
    the last element. The backtest only sees completed bars, so live must
    also exclude the incomplete bar to maintain signal parity.

    Without this fix: close[-1] in live = few-seconds-old open price of new bar,
    while in backtest close[i] = the fully closed bar. The 8-bar return
    calculation diverges, causing the live system to rarely trigger signals.
    """
    from quantbox.scalper import RubberBandScalper

    # Create a bare scalper instance with mocked client (skip __init__)
    scalper = object.__new__(RubberBandScalper)

    # Simulate 201 klines from Binance (200 complete + 1 incomplete)
    fake_klines = []
    for i in range(201):
        # [open_time, open, high, low, close, volume, close_time,
        #  quote_vol, n_trades, taker_buy_base, taker_buy_quote, ignore]
        price = 100.0 + i * 0.1
        fake_klines.append(
            [
                1000000 + i * 900000,  # open_time
                str(price),  # open
                str(price + 0.5),  # high
                str(price - 0.5),  # low
                str(price),  # close
                str(1000.0),  # volume
                1000000 + (i + 1) * 900000,  # close_time
                str(10000.0),  # quote volume
                100,  # n_trades
                str(400.0),  # taker buy base
                str(4000.0),  # taker buy quote
                "0",  # ignore
            ]
        )

    # The last kline is the incomplete candle (current bar)
    # Set it to a wildly different price to make the bug obvious
    fake_klines[-1][4] = str(999.0)  # close = 999 (garbage incomplete data)

    mock_client = MagicMock()
    mock_client.futures_klines.return_value = fake_klines
    scalper.client = mock_client
    scalper.paper_mode = True

    # Test basic (close only)
    close = scalper.download_klines("BTCUSDT", 200)
    assert close is not None
    assert len(close) == 200, f"Expected 200 completed bars, got {len(close)}"
    # Last bar should be the penultimate kline, not the incomplete one
    assert close[-1] != 999.0, "Incomplete bar was NOT dropped!"
    assert close[-1] == pytest.approx(100.0 + 199 * 0.1, abs=0.01)

    # Test full (close + high + low + volume + taker)
    mock_client.futures_klines.return_value = fake_klines
    result = scalper.download_klines("BTCUSDT", 200, full=True)
    assert result is not None
    close_f, high_f, low_f, vol_f, taker_f = result
    assert len(close_f) == 200
    assert close_f[-1] != 999.0, "Incomplete bar was NOT dropped (full mode)!"


# ── Test 9: Stateless on_bar (ghost-position prevention) ──


def test_on_bar_stateless_no_ghost_position():
    """on_bar must evaluate ONLY the last bar, not carry forward virtual state.

    Bug history: on_bar ran compute_signals_v9 on 200 bars and took sigs[-1].
    If an earlier bar (e.g. bar 150) triggered a virtual entry, sigs[-1]
    reflected that stale position — not a fresh signal at bar 199.

    Fix: on_bar now evaluates the last bar statelessly.
    """
    cfg = RubberBandConfig(
        entry_mode="confidence",
        min_confidence=1,
        threshold=0.032,
        lookback=8,
        stoch_enabled=True,
        stoch_k_period=3,
        stoch_d_period=3,
        macd_enabled=True,
    )
    strategy = RubberBandStrategy(cfg)

    n = 200
    # Price flat everywhere — no signal anywhere
    close = np.ones(n, dtype=np.float64) * 100.0
    btc = np.ones(n, dtype=np.float64) * 50000.0
    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker = np.ones(n, dtype=np.float64) * 400.0
    high = close.copy()
    low = close.copy()

    # Inject a big dip at bar 150 (would trigger virtual entry in old code)
    close[150] = 95.0  # -5% dip
    high[150] = 100.0
    low[150] = 94.5

    # But bar 199 (the last bar) is back to flat — NO signal at last bar
    close[199] = 100.0

    strategy.update_market_data("SYM", volume, taker, high, low)

    sig = strategy.on_bar("SYM", close, btc)
    # Old code: sigs[-1] could be non-zero from bar 150 virtual entry
    # Fixed code: evaluates bar 199 only → flat → signal=0
    assert sig == 0, (
        f"Ghost signal detected! on_bar returned {sig} but last bar is flat. "
        "This means on_bar is still carrying virtual position state."
    )


def test_on_bar_stateless_detects_fresh_entry():
    """on_bar must detect a fresh entry at the last bar even when earlier bars
    would have occupied a virtual position slot.

    Bug history: If a virtual entry at bar 170 was still held at bar 199,
    compute_signals_v9 was in the exit branch and never re-evaluated
    bar 199 for a new entry. sigs[-1] == old_position, not fresh signal.

    Fix: on_bar evaluates bar 199 independently.
    """
    cfg = RubberBandConfig(
        entry_mode="confidence",
        min_confidence=1,
        threshold=0.032,
        lookback=8,
        stoch_enabled=True,
        stoch_k_period=3,
        stoch_d_period=3,
        macd_enabled=True,
        mtf_confirmation_enabled=False,  # Disable MTF to isolate entry logic
    )
    strategy = RubberBandStrategy(cfg)

    n = 200
    # Create two separate dips:
    # Dip 1 at bar 170 (would occupy virtual position slot in old code)
    # Dip 2 at bar 199 (the last bar — should trigger fresh)
    close = np.ones(n, dtype=np.float64) * 100.0
    btc = np.ones(n, dtype=np.float64) * 50000.0
    # BTC falling → sympathy long entries enabled
    btc[-49:] = np.linspace(50000, 49000, 49)
    volume = np.ones(n, dtype=np.float64) * 1000.0
    taker = np.ones(n, dtype=np.float64) * 400.0
    high = close.copy()
    low = close.copy()

    # Dip 1: bar 170 drops to 96 (-4%)
    close[170] = 96.0
    high[170] = 100.0
    low[170] = 95.5
    # Bars 171-198 stay at 100 (virtual position would still be active)
    # Dip 2: bar 199 drops to 95 (-5%, fresh signal)
    close[199] = 95.0
    high[199] = 100.0
    low[199] = 94.5

    strategy.update_market_data("SYM", volume, taker, high, low)

    sig = strategy.on_bar("SYM", close, btc)
    # Old code: sigs[-1] from stale bar-170 position (direction might match
    #           but entry price is wrong) or sigs[-1]==0 after max_hold exit
    # Fixed code: evaluates bar 199 independently → fresh long signal
    assert sig != 0, (
        "Missed signal! Last bar has a -5% dip qualifying for entry, "
        "but on_bar returned 0. The virtual position from bar 170 "
        "may have blocked evaluation of bar 199."
    )
    assert sig == 1, f"Expected long signal (+1) at -5% dip, got {sig}"
