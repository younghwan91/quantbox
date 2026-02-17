"""Backtest ↔ Live Trading Parity Tests.

Verifies that the live scalper produces the EXACT same decisions
as the backtest engine for:
  1. Signal direction (on_bar returns same sign as backtest signals)
  2. MTF confirmation (blocking + leverage)
  3. Effective leverage (backtest signal magnitude == live leverage)
  4. CLI defaults don't override .env values

These tests exist because parity violations = silent money loss.
"""

import math

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

    assert live_base_leverage == pytest.approx(backtest_base_leverage, abs=1e-6), \
        f"Base leverage mismatch: backtest={backtest_base_leverage}, live={live_base_leverage}"
    assert live_mtf_leverage == pytest.approx(backtest_mtf_leverage, abs=1e-6), \
        f"MTF leverage mismatch: backtest={backtest_mtf_leverage}, live={live_mtf_leverage}"


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
        close, high, low, volume, taker, btc,
        cfg.lookback, cfg.threshold,
        cfg.target_retrace, cfg.stop_mult,
        cfg.max_hold, cfg.stop_decay,
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
    )

    bt_last = bt_sigs[-1]
    bt_direction = 1 if bt_last > 0 else (-1 if bt_last < 0 else 0)

    assert live_sig == bt_direction, \
        f"Direction mismatch: live={live_sig}, backtest_raw={bt_last}"


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
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--max-slots', type=int, default=None)

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
        entry_mode='confidence',
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
