"""
Tests for cached pair selection in run_cycle.

핵심 검증 포인트:
1. select_pairs()가 사이클당 정확히 1회만 호출되는지 검증
2. execute_pair_trade 실패 시 해당 심볼이 excluded에 추가되어 같은 페어 재시도가 방지되는지 검증
3. cached_candidates가 find_best_pair에 올바르게 전달되는지 검증
4. 기존 동작 (break on no result) 유지 검증
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from quantbox.core.trader_slot import (
    TraderSlotState,
    get_occupied_symbols,
    get_idle_slots,
    get_active_slots,
    calculate_slot_capital,
    is_pair_available,
)
from quantbox.strategy.pair_selector import PairCandidate


# ============================================================
# Helpers
# ============================================================


def _make_candidate(
    sym1: str, sym2: str, z_score: float = 2.5, score: float = 0.8
) -> PairCandidate:
    """테스트용 PairCandidate 생성."""
    return PairCandidate(
        symbol1=sym1,
        symbol2=sym2,
        passed=True,
        combined_score=score,
        correlation=0.9,
        cointegration_pvalue=0.01,
        half_life=10.0,
        hedge_ratio=1.0,
        hurst=0.4,
        current_z_score=z_score,
        spread_std=0.05,
    )


def _make_ohlcv_df(n: int = 200, base_price: float = 100.0) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame (가격만 필요)."""
    close = base_price + np.arange(n) * 0.01
    return pd.DataFrame({
        'Open': close,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.ones(n) * 1000,
    })


# ============================================================
# find_best_pair with cached_candidates tests
# ============================================================


class TestFindBestPairCaching:
    """find_best_pair의 cached_candidates 파라미터 동작 검증."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        monkeypatch.setenv("MAX_TRADERS", "5")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            trader = LivePairTrader(paper_mode=True)
            return trader

    def test_cached_candidates_skips_select_pairs(self, trader):
        """cached_candidates 전달 시 select_pairs가 호출되지 않아야 한다."""
        candidates = [_make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5)]
        data = {
            "BTCUSDT": _make_ohlcv_df(base_price=50000),
            "ETHUSDT": _make_ohlcv_df(base_price=3000),
        }

        trader.pair_selector.select_pairs = MagicMock(
            side_effect=AssertionError("select_pairs should NOT be called")
        )
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        result = trader.find_best_pair(
            data, excluded_symbols=set(),
            cached_candidates=candidates,
        )

        assert result is not None
        pair, pair_info = result
        assert pair.symbol1 == "BTCUSDT"
        assert pair.symbol2 == "ETHUSDT"
        trader.pair_selector.select_pairs.assert_not_called()

    def test_no_cached_candidates_calls_select_pairs(self, trader):
        """cached_candidates=None이면 select_pairs를 호출해야 한다."""
        candidates = [_make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5)]
        data = {
            "BTCUSDT": _make_ohlcv_df(base_price=50000),
            "ETHUSDT": _make_ohlcv_df(base_price=3000),
        }

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        result = trader.find_best_pair(
            data, excluded_symbols=set(),
            cached_candidates=None,
        )

        assert result is not None
        trader.pair_selector.select_pairs.assert_called_once()

    def test_excluded_symbols_filters_correctly(self, trader):
        """excluded_symbols에 포함된 심볼의 페어는 건너뛰어야 한다."""
        candidates = [
            _make_candidate("BTCUSDT", "ETHUSDT", z_score=3.0, score=0.9),
            _make_candidate("SOLUSDT", "AVAXUSDT", z_score=2.5, score=0.7),
        ]
        data = {
            "BTCUSDT": _make_ohlcv_df(base_price=50000),
            "ETHUSDT": _make_ohlcv_df(base_price=3000),
            "SOLUSDT": _make_ohlcv_df(base_price=100),
            "AVAXUSDT": _make_ohlcv_df(base_price=30),
        }

        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        result = trader.find_best_pair(
            data,
            excluded_symbols={"BTCUSDT"},
            cached_candidates=candidates,
        )

        assert result is not None
        pair, _ = result
        # BTCUSDT 포함 페어는 제외, SOLUSDT/AVAXUSDT 선택
        assert pair.symbol1 == "SOLUSDT"
        assert pair.symbol2 == "AVAXUSDT"

    def test_all_pairs_excluded_returns_none(self, trader):
        """모든 페어가 excluded이면 None 반환."""
        candidates = [
            _make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5),
        ]
        data = {
            "BTCUSDT": _make_ohlcv_df(base_price=50000),
            "ETHUSDT": _make_ohlcv_df(base_price=3000),
        }

        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        result = trader.find_best_pair(
            data,
            excluded_symbols={"BTCUSDT", "ETHUSDT"},
            cached_candidates=candidates,
        )

        assert result is None

    def test_empty_cached_candidates_returns_none(self, trader):
        """빈 cached_candidates -> None 반환."""
        data = {"BTCUSDT": _make_ohlcv_df()}

        result = trader.find_best_pair(
            data,
            excluded_symbols=set(),
            cached_candidates=[],
        )

        assert result is None


# ============================================================
# run_cycle: select_pairs 1회 호출 + execute 실패 시 excluded 추가
# ============================================================


class TestRunCycleCachedSearch:
    """run_cycle에서 select_pairs가 정확히 1회만 호출되는지,
    execute_pair_trade 실패 시 해당 심볼이 excluded에 추가되는지 검증.
    """

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        monkeypatch.setenv("MAX_TRADERS", "3")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            trader = LivePairTrader(paper_mode=True)

            # Mock external dependencies
            trader.get_available_balance = MagicMock(return_value=5000.0)
            trader.get_total_balance = MagicMock(return_value=5000.0)
            trader.get_trading_coins = MagicMock(
                return_value=["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "BNBUSDT"]
            )
            trader.download_multi_symbol_data = MagicMock(
                return_value={
                    "BTCUSDT": _make_ohlcv_df(base_price=50000),
                    "ETHUSDT": _make_ohlcv_df(base_price=3000),
                    "SOLUSDT": _make_ohlcv_df(base_price=100),
                    "AVAXUSDT": _make_ohlcv_df(base_price=30),
                    "BNBUSDT": _make_ohlcv_df(base_price=300),
                }
            )
            trader.discord = MagicMock()

            return trader

    def test_select_pairs_called_once_per_cycle(self, trader):
        """select_pairs가 사이클당 정확히 1회만 호출된다."""
        candidates = [
            _make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5),
        ]

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )
        # execute_pair_trade 성공
        trader.execute_pair_trade = MagicMock(return_value=True)

        # Simulate slot successfully entering position
        def side_effect_execute(slot_id, pair, pair_info, capital, **kwargs):
            trader.slots[slot_id].position_open = True
            trader.slots[slot_id].current_pair = (pair.symbol1, pair.symbol2)
            return True

        trader.execute_pair_trade = MagicMock(side_effect=side_effect_execute)

        trader.run_cycle()

        # select_pairs는 정확히 1회만 호출
        trader.pair_selector.select_pairs.assert_called_once()

    def test_select_pairs_called_once_even_with_multiple_idle_slots(self, trader):
        """idle slot이 3개여도 select_pairs는 1회만 호출된다.
        이전 코드에서는 idle slot마다 재호출하여 3회 호출되었다.
        """
        candidates = [
            _make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5, score=0.9),
            _make_candidate("SOLUSDT", "AVAXUSDT", z_score=2.3, score=0.7),
            _make_candidate("BNBUSDT", "BTCUSDT", z_score=2.1, score=0.5),
        ]

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)

        # select_tradeable_pairs는 모든 후보를 반환
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        call_count = 0

        def side_effect_execute(slot_id, pair, pair_info, capital, **kwargs):
            nonlocal call_count
            call_count += 1
            trader.slots[slot_id].position_open = True
            trader.slots[slot_id].current_pair = (pair.symbol1, pair.symbol2)
            return True

        trader.execute_pair_trade = MagicMock(side_effect=side_effect_execute)

        trader.run_cycle()

        # 핵심 검증: select_pairs는 반드시 1회만 호출
        trader.pair_selector.select_pairs.assert_called_once()

        # execute는 최소 1회 이상 호출 (available pairs에 따라)
        assert call_count >= 1

    def test_execute_fail_excludes_pair_symbols(self, trader):
        """execute_pair_trade 실패 시 해당 심볼이 excluded에 추가되어
        다음 slot에서 같은 pair를 재시도하지 않는다.
        """
        candidates = [
            _make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5, score=0.9),
            _make_candidate("SOLUSDT", "AVAXUSDT", z_score=2.3, score=0.7),
        ]

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        execute_calls = []

        def side_effect_execute(slot_id, pair, pair_info, capital, **kwargs):
            execute_calls.append({
                'slot_id': slot_id,
                'pair': (pair.symbol1, pair.symbol2),
            })
            if pair.symbol1 == "BTCUSDT":
                return False  # 첫 페어 실패
            # 두 번째 페어 성공
            trader.slots[slot_id].position_open = True
            trader.slots[slot_id].current_pair = (pair.symbol1, pair.symbol2)
            return True

        trader.execute_pair_trade = MagicMock(side_effect=side_effect_execute)

        trader.run_cycle()

        # 첫 slot: BTCUSDT/ETHUSDT 시도 → 실패 → excluded에 추가
        # 두 번째 slot: BTCUSDT/ETHUSDT 제외, SOLUSDT/AVAXUSDT 시도 → 성공
        assert len(execute_calls) >= 2
        assert execute_calls[0]['pair'] == ("BTCUSDT", "ETHUSDT")
        assert execute_calls[1]['pair'] == ("SOLUSDT", "AVAXUSDT")

    def test_execute_fail_same_pair_not_retried(self, trader):
        """execute_pair_trade 실패 후 같은 페어가 재시도되지 않는다.
        이전 버전에서는 동일한 best pair를 반복 시도했다.
        """
        # 하나뿐인 candidate
        candidates = [
            _make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5),
        ]

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        # execute 항상 실패
        trader.execute_pair_trade = MagicMock(return_value=False)

        trader.run_cycle()

        # execute는 최대 1회만 호출 (같은 페어 재시도 없음)
        # 첫 slot에서 실패 → excluded에 추가 → 다음 slot에서 해당 페어 제외 → no result → break
        assert trader.execute_pair_trade.call_count == 1

    def test_no_candidates_returns_early(self, trader):
        """select_pairs가 빈 리스트 반환 시 즉시 반환."""
        trader.pair_selector.select_pairs = MagicMock(return_value=[])

        result = trader.run_cycle()

        assert result is True
        trader.pair_selector.select_pairs.assert_called_once()

    def test_success_updates_excluded_for_next_slot(self, trader):
        """execute 성공 시 occupied_symbols가 excluded에 반영된다."""
        candidates = [
            _make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5, score=0.9),
            _make_candidate("SOLUSDT", "AVAXUSDT", z_score=2.3, score=0.7),
        ]

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        find_best_pair_calls = []
        original_find = trader.find_best_pair

        def track_find_best_pair(data, excluded_symbols=None, cached_candidates=None):
            find_best_pair_calls.append(set(excluded_symbols or set()))
            return original_find(
                data,
                excluded_symbols=excluded_symbols,
                cached_candidates=cached_candidates,
            )

        trader.find_best_pair = track_find_best_pair

        def side_effect_execute(slot_id, pair, pair_info, capital, **kwargs):
            trader.slots[slot_id].position_open = True
            trader.slots[slot_id].current_pair = (pair.symbol1, pair.symbol2)
            return True

        trader.execute_pair_trade = MagicMock(side_effect=side_effect_execute)

        trader.run_cycle()

        # 첫 호출: excluded = {}
        assert find_best_pair_calls[0] == set()
        # 두 번째 호출: excluded = {BTCUSDT, ETHUSDT}
        if len(find_best_pair_calls) >= 2:
            assert "BTCUSDT" in find_best_pair_calls[1]
            assert "ETHUSDT" in find_best_pair_calls[1]


# ============================================================
# find_best_pair: backward compatibility (cached_candidates=None)
# ============================================================


class TestFindBestPairBackwardCompatibility:
    """cached_candidates=None (기본값) 사용 시 기존 동작 유지 확인."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            trader = LivePairTrader(paper_mode=True)
            return trader

    def test_default_behavior_calls_select_pairs(self, trader):
        """cached_candidates 인자 없이 호출하면 select_pairs를 호출한다."""
        candidates = [_make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5)]
        data = {
            "BTCUSDT": _make_ohlcv_df(base_price=50000),
            "ETHUSDT": _make_ohlcv_df(base_price=3000),
        }

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        # cached_candidates 인자 없이 호출 (기존 API 호환)
        result = trader.find_best_pair(data, excluded_symbols=set())

        assert result is not None
        trader.pair_selector.select_pairs.assert_called_once()

    def test_no_tradeable_returns_none(self, trader):
        """tradeable pairs가 없으면 None 반환."""
        candidates = [_make_candidate("BTCUSDT", "ETHUSDT", z_score=1.0)]
        data = {
            "BTCUSDT": _make_ohlcv_df(base_price=50000),
            "ETHUSDT": _make_ohlcv_df(base_price=3000),
        }

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(return_value=[])

        result = trader.find_best_pair(data, excluded_symbols=set())

        assert result is None

    def test_pair_info_has_correct_keys(self, trader):
        """반환되는 pair_info에 필수 키가 포함되어 있어야 한다."""
        candidates = [_make_candidate("BTCUSDT", "ETHUSDT", z_score=2.5)]
        data = {
            "BTCUSDT": _make_ohlcv_df(base_price=50000),
            "ETHUSDT": _make_ohlcv_df(base_price=3000),
        }

        trader.pair_selector.select_pairs = MagicMock(return_value=candidates)
        trader.pair_selector.select_tradeable_pairs = MagicMock(
            return_value=candidates
        )

        result = trader.find_best_pair(
            data, excluded_symbols=set(),
            cached_candidates=candidates,
        )

        assert result is not None
        _, pair_info = result
        assert 'hedge_ratio' in pair_info
        assert 'spread_mean' in pair_info
        assert 'spread_std' in pair_info
        assert 'half_life' in pair_info
