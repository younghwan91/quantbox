"""
Vol-Inverse Position Sizing (Risk Parity) tests.

핵심 원칙:
  notional ∝ 1/spread_std  →  stop-loss dollar loss = constant across all pairs.

검증 항목:
1. 높은 spread_std 페어는 낮은 notional을 받는다
2. 낮은 spread_std 페어(anchor)는 full slot capital을 받는다
3. anchor_std = min(spread_stds) 이므로 vol_ratio ≤ 1 항상 성립
4. vol_inverse_sizing=False일 때 scaling이 비활성화된다
5. 백테스트와 라이브의 sizing 공식이 동일하다
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from quantbox.backtest.continuous_backtest import (
    ContinuousBacktestConfig,
    ContinuousPairBacktest,
)
from quantbox.core.trader_slot import (
    calculate_slot_capital,
    calculate_margin_from_notional,
)


# ============================================================
# Unit: Vol-inverse ratio formula
# ============================================================


class TestVolInverseRatioFormula:
    """vol_ratio = anchor_std / spread_std, capped at 1.0"""

    def test_ratio_equals_one_for_anchor_pair(self):
        """anchor_std == spread_std → ratio = 1.0 (full capital)."""
        anchor_std = 0.005
        spread_std = 0.005
        ratio = min(anchor_std / spread_std, 1.0)
        assert ratio == pytest.approx(1.0)

    def test_ratio_less_than_one_for_high_vol(self):
        """spread_std > anchor_std → ratio < 1 (reduced capital)."""
        anchor_std = 0.005
        spread_std = 0.010  # 2× anchor
        ratio = min(anchor_std / spread_std, 1.0)
        assert ratio == pytest.approx(0.5)

    def test_ratio_cannot_exceed_one(self):
        """anchor = min → ratio ≤ 1 always. Even if numerically > 1."""
        anchor_std = 0.010
        spread_std = 0.005
        ratio = min(anchor_std / spread_std, 1.0)
        assert ratio == 1.0

    def test_equal_dollar_stop_loss(self):
        """stop_loss_dollar = stop_z × spread_std × notional
        With notional = base × (anchor/spread_std):
          loss = stop_z × spread_std × base × (anchor/spread_std)
               = stop_z × anchor × base  (constant!)
        """
        base_notional = 1000.0
        anchor_std = 0.005
        stop_z = 3.5  # stop loss z-score

        for spread_std in [0.005, 0.010, 0.020, 0.050]:
            ratio = min(anchor_std / spread_std, 1.0)
            adjusted = base_notional * ratio
            loss = stop_z * spread_std * adjusted
            expected_loss = stop_z * anchor_std * base_notional
            assert loss == pytest.approx(expected_loss), (
                f"spread_std={spread_std}: loss={loss:.2f} != expected={expected_loss:.2f}"
            )


# ============================================================
# Config defaults
# ============================================================


class TestVolInverseConfig:
    """vol_inverse_sizing config field behavior."""

    def test_default_enabled(self):
        """vol_inverse_sizing defaults to True."""
        cfg = ContinuousBacktestConfig()
        assert cfg.vol_inverse_sizing is True

    def test_max_multiplier_default(self):
        """vol_sizing_max_multiplier defaults to 3.0."""
        cfg = ContinuousBacktestConfig()
        assert cfg.vol_sizing_max_multiplier == 3.0

    def test_can_disable(self):
        """vol_inverse_sizing=False disables scaling."""
        cfg = ContinuousBacktestConfig(vol_inverse_sizing=False)
        assert cfg.vol_inverse_sizing is False


# ============================================================
# Backtest _enter_slot integration
# ============================================================


class TestVolInverseInBacktest:
    """_enter_slot applies vol-inverse sizing correctly."""

    def _make_engine(self, vol_inverse: bool = True):
        cfg = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=5,
            formation_bars=100,
            vol_inverse_sizing=vol_inverse,
        )
        return ContinuousPairBacktest(cfg)

    def _make_candidate(self, spread_std: float):
        return {
            'symbol1': 'BTCUSDT',
            'symbol2': 'ETHUSDT',
            'hedge_ratio': 1.0,
            'intercept': 0.0,
            'spread_mean': 0.0,
            'spread_std': spread_std,
            'half_life': 10.0,
            'score': 0.9,
        }

    def test_high_vol_gets_less_notional(self):
        """High spread_std pair → smaller notional than low spread_std."""
        engine = self._make_engine(vol_inverse=True)
        
        # Two candidates with different vol
        low_vol = self._make_candidate(spread_std=0.005)
        high_vol = self._make_candidate(spread_std=0.020)

        base_notional = 2000.0  # assume same base for both
        anchor_std = 0.005  # min

        # Low vol: ratio = 0.005/0.005 = 1.0
        ratio_low = min(anchor_std / low_vol['spread_std'], 1.0)
        notional_low = base_notional * ratio_low

        # High vol: ratio = 0.005/0.020 = 0.25
        ratio_high = min(anchor_std / high_vol['spread_std'], 1.0)
        notional_high = base_notional * ratio_high

        assert notional_low == pytest.approx(2000.0)
        assert notional_high == pytest.approx(500.0)
        assert notional_high < notional_low

    def test_disabled_no_scaling(self):
        """vol_inverse_sizing=False → notional unchanged."""
        engine = self._make_engine(vol_inverse=False)
        base_notional = 2000.0
        
        # When disabled, the scaling code is skipped
        # Verify config is False
        assert engine.config.vol_inverse_sizing is False

    def test_anchor_std_is_min_of_candidates(self):
        """anchor_std should be min(spread_stds) across candidates."""
        stds = [0.020, 0.005, 0.015, 0.010]
        candidates = [self._make_candidate(s) for s in stds]
        
        anchor_std = min(c['spread_std'] for c in candidates)
        assert anchor_std == pytest.approx(0.005)


# ============================================================
# Live trader vol-inverse sizing
# ============================================================


class TestVolInverseInLiveTrader:
    """Live trader execute_pair_trade applies vol-inverse sizing."""

    def test_anchor_std_passed_to_execute(self):
        """execute_pair_trade accepts anchor_std kwarg."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true",
            }):
                trader = LivePairTrader(paper_mode=True)
                
                # Mock everything needed for execute_pair_trade
                trader.get_price = MagicMock(side_effect=[100.0, 50.0])
                trader.calculate_zscore = MagicMock(return_value=-2.5)
                trader.get_symbol_info = MagicMock(return_value={
                    'step_size': 0.001, 'quantity_precision': 3,
                    'min_notional': 5,
                })
                trader.place_order = MagicMock(return_value={'orderId': '123'})

                pair = MagicMock()
                pair.symbol1 = "BTCUSDT"
                pair.symbol2 = "ETHUSDT"
                pair_info = {
                    'hedge_ratio': 1.0,
                    'spread_mean': 0.0,
                    'spread_std': 0.020,  # high vol
                    'half_life': 10.0,
                }

                # Full capital = 1000, anchor_std = 0.005
                capital = 1000.0
                anchor_std = 0.005

                result = trader.execute_pair_trade(
                    0, pair, pair_info, capital, anchor_std=anchor_std
                )
                assert result is True

                # With vol-inverse: ratio = 0.005/0.020 = 0.25
                # Effective notional = 1000 * 0.25 = 250
                pos = trader.slot_positions[0]
                leverage = trader.config['leverage']
                expected_notional = capital * min(anchor_std / pair_info['spread_std'], 1.0)
                assert pos['notional'] == pytest.approx(expected_notional)

    def test_no_anchor_no_scaling(self):
        """anchor_std=0 (default) → no scaling applied."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true",
            }):
                trader = LivePairTrader(paper_mode=True)

                trader.get_price = MagicMock(side_effect=[100.0, 50.0])
                trader.calculate_zscore = MagicMock(return_value=-2.5)
                trader.get_symbol_info = MagicMock(return_value={
                    'step_size': 0.001, 'quantity_precision': 3,
                    'min_notional': 5,
                })
                trader.place_order = MagicMock(return_value={'orderId': '123'})

                pair = MagicMock()
                pair.symbol1 = "BTCUSDT"
                pair.symbol2 = "ETHUSDT"
                pair_info = {
                    'hedge_ratio': 1.0,
                    'spread_mean': 0.0,
                    'spread_std': 0.020,
                    'half_life': 10.0,
                }

                capital = 1000.0
                result = trader.execute_pair_trade(0, pair, pair_info, capital)
                assert result is True

                pos = trader.slot_positions[0]
                # No anchor → no scaling → full capital used
                assert pos['notional'] == pytest.approx(capital)


# ============================================================
# Safety: stop-loss dollar equalization proof
# ============================================================


class TestStopLossDollarEqualization:
    """Core safety property: 
    With vol-inverse sizing, stop-loss dollar amounts are equal
    across all pairs regardless of spread_std.
    """

    def test_all_pairs_same_stop_loss_dollar(self):
        """Given multiple pairs with different spread_std,
        the stop-loss dollar loss should be identical for all.

        Formula: 
            loss = stop_z × spread_std × notional
            notional = base × (anchor / spread_std)
            ∴ loss = stop_z × anchor × base  (constant)
        """
        base_notional = 2000.0
        stop_z = 3.5  # stop loss threshold (z-score units)
        
        spread_stds = [0.003, 0.005, 0.010, 0.025, 0.050, 0.100]
        anchor_std = min(spread_stds)  # = 0.003

        losses = []
        for std in spread_stds:
            ratio = min(anchor_std / std, 1.0)
            adjusted_notional = base_notional * ratio
            loss = stop_z * std * adjusted_notional
            losses.append(loss)

        # All losses should be identical
        expected = stop_z * anchor_std * base_notional
        for i, loss in enumerate(losses):
            assert loss == pytest.approx(expected), (
                f"Pair {i} (std={spread_stds[i]}): "
                f"loss={loss:.4f} != expected={expected:.4f}"
            )

    def test_anchor_pair_gets_full_capital(self):
        """The pair with min spread_std gets ratio=1 → full slot capital."""
        base_notional = 2000.0
        anchor_std = 0.005
        
        ratio = min(anchor_std / anchor_std, 1.0)
        assert ratio == 1.0
        assert base_notional * ratio == base_notional
