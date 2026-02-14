"""
Tests for state persistence (save/load/clear/recovery).

상태 영속화 모듈의 단위 테스트.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quantbox.core.state_persistence import (
    save_state,
    load_state,
    clear_state,
    deserialize_entry_times,
    deserialize_positions,
    _serialize_slots,
    _serialize_positions,
    _serialize_entry_times,
    _serialize_trade_history,
)
from quantbox.core.trader_slot import TraderSlotState


@pytest.fixture
def tmp_state_file(tmp_path):
    """테스트용 임시 상태 파일 경로."""
    return tmp_path / "test_state.json"


@pytest.fixture
def sample_slots():
    """테스트용 슬롯 상태."""
    slots = [TraderSlotState(slot_id=i) for i in range(3)]
    slots[0].position_open = True
    slots[0].current_pair = ("BTCUSDT", "ETHUSDT")
    slots[0].position_info = {
        'symbol1': 'BTCUSDT',
        'symbol2': 'ETHUSDT',
        'entry_price1': 50000.0,
        'entry_price2': 3000.0,
        'hedge_ratio': 0.5,
        'qty1': 0.001,
        'qty2': -0.01,
        'entry_zscore': -2.5,
        'signal': 1,
        'spread_mean': 0.1,
        'spread_std': 0.01,
        'half_life': 10.0,
        'capital': 100.0,
        'notional': 500.0,
        'leverage': 5.0,
    }
    return slots


@pytest.fixture
def sample_positions():
    """테스트용 포지션 dict."""
    return {
        0: {
            'symbol1': 'BTCUSDT',
            'symbol2': 'ETHUSDT',
            'entry_price1': 50000.0,
            'entry_price2': 3000.0,
            'hedge_ratio': 0.5,
            'qty1': 0.001,
            'qty2': -0.01,
            'entry_zscore': -2.5,
            'signal': 1,
            'spread_mean': 0.1,
            'spread_std': 0.01,
            'half_life': 10.0,
            'capital': 100.0,
            'notional': 500.0,
            'leverage': 5.0,
            '_df1': 'should_be_excluded',  # underscore prefix → skip
        },
    }


@pytest.fixture
def sample_entry_times():
    """테스트용 진입 시각."""
    return {
        0: datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_trade_history():
    """테스트용 트레이드 이력."""
    return [
        {
            'slot_id': 0,
            'symbol1': 'BTCUSDT',
            'symbol2': 'ETHUSDT',
            'entry_time': datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            'exit_time': datetime(2025, 6, 1, 14, 0, 0, tzinfo=timezone.utc),
            'pnl': 5.50,
            'pnl_pct': 1.1,
            'reason': 'mean_reversion',
        },
    ]


class TestSerializeSlots:
    """슬롯 직렬화 테스트."""

    def test_serialize_dataclass_slots(self, sample_slots):
        result = _serialize_slots(sample_slots)
        assert len(result) == 3
        assert result[0]['slot_id'] == 0
        assert result[0]['position_open'] is True
        assert result[0]['current_pair'] == ['BTCUSDT', 'ETHUSDT']
        assert result[1]['position_open'] is False
        assert result[1]['current_pair'] is None

    def test_serialize_dict_slots(self):
        slots = [{'slot_id': 0, 'position_open': False}]
        result = _serialize_slots(slots)
        assert result == slots


class TestSerializePositions:
    """포지션 직렬화 테스트."""

    def test_int_keys_to_str(self, sample_positions):
        result = _serialize_positions(sample_positions)
        assert '0' in result
        assert 0 not in result

    def test_underscore_fields_excluded(self, sample_positions):
        result = _serialize_positions(sample_positions)
        assert '_df1' not in result['0']
        assert 'symbol1' in result['0']


class TestSerializeEntryTimes:
    """진입 시각 직렬화 테스트."""

    def test_datetime_to_iso(self, sample_entry_times):
        result = _serialize_entry_times(sample_entry_times)
        assert '0' in result
        assert '2025-06-01' in result['0']


class TestSerializeTradeHistory:
    """트레이드 이력 직렬화 테스트."""

    def test_datetime_fields_converted(self, sample_trade_history):
        result = _serialize_trade_history(sample_trade_history)
        assert isinstance(result[0]['entry_time'], str)
        assert result[0]['pnl'] == 5.50


class TestSaveLoadClear:
    """전체 save/load/clear 사이클 테스트."""

    def test_save_and_load(self, tmp_state_file, sample_slots, sample_positions,
                           sample_entry_times, sample_trade_history):
        save_state(
            slots=sample_slots,
            slot_positions=sample_positions,
            slot_entry_times=sample_entry_times,
            trade_history=sample_trade_history,
            paper_balance=1000.0,
            paper_mode=True,
            initial_balance=1000.0,
            session_high_water_mark=1050.0,
            daily_start_balance=990.0,
            daily_start_time='2025-06-01T00:00:00+00:00',
            filepath=tmp_state_file,
        )

        state = load_state(tmp_state_file)
        assert state is not None
        assert state['version'] == '1.8.0'
        assert state['paper_mode'] is True
        assert state['paper_balance'] == 1000.0
        assert state['initial_balance'] == 1000.0
        assert state['session_high_water_mark'] == 1050.0
        assert state['daily_start_balance'] == 990.0

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_state(tmp_path / "nonexistent.json")
        assert result is None

    def test_clear_removes_file(self, tmp_state_file, sample_slots):
        save_state(
            slots=sample_slots,
            slot_positions={},
            slot_entry_times={},
            trade_history=[],
            paper_balance=1000.0,
            paper_mode=True,
            filepath=tmp_state_file,
        )
        assert tmp_state_file.exists()
        clear_state(tmp_state_file)
        assert not tmp_state_file.exists()

    def test_clear_nonexistent_no_error(self, tmp_path):
        clear_state(tmp_path / "nonexistent.json")

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir" / "state.json"
        save_state(
            slots=[],
            slot_positions={},
            slot_entry_times={},
            trade_history=[],
            paper_balance=0,
            paper_mode=True,
            filepath=nested,
        )
        assert nested.exists()

    def test_atomic_write_no_corruption(self, tmp_state_file, sample_slots,
                                        sample_positions, sample_entry_times):
        """파일이 완전히 쓰여진 후에만 접근 가능해야 함."""
        save_state(
            slots=sample_slots,
            slot_positions=sample_positions,
            slot_entry_times=sample_entry_times,
            trade_history=[],
            paper_balance=1000.0,
            paper_mode=True,
            filepath=tmp_state_file,
        )
        # 파일이 유효한 JSON인지 확인
        with open(tmp_state_file) as f:
            data = json.load(f)
        assert 'slots' in data
        assert 'positions' in data

    def test_load_corrupted_file_returns_none(self, tmp_state_file):
        """손상된 JSON 파일 로드 시 None 반환."""
        tmp_state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_file.write_text("{ invalid json }")
        result = load_state(tmp_state_file)
        assert result is None


class TestDeserialize:
    """역직렬화 테스트."""

    def test_deserialize_entry_times(self):
        raw = {
            '0': '2025-06-01T12:00:00+00:00',
            '1': '2025-06-01T14:00:00+00:00',
        }
        result = deserialize_entry_times(raw)
        assert 0 in result
        assert 1 in result
        assert result[0].tzinfo is not None
        assert result[0].year == 2025

    def test_deserialize_entry_times_naive_gets_utc(self):
        raw = {'0': '2025-06-01T12:00:00'}
        result = deserialize_entry_times(raw)
        assert result[0].tzinfo == timezone.utc

    def test_deserialize_entry_times_invalid_skipped(self):
        raw = {'0': 'not-a-date', '1': '2025-06-01T12:00:00+00:00'}
        result = deserialize_entry_times(raw)
        assert 0 not in result
        assert 1 in result

    def test_deserialize_positions(self):
        raw = {'0': {'symbol1': 'BTCUSDT'}, '2': {'symbol1': 'ETHUSDT'}}
        result = deserialize_positions(raw)
        assert 0 in result
        assert 2 in result
        assert result[0]['symbol1'] == 'BTCUSDT'


class TestRoundTrip:
    """저장 → 로드 → 역직렬화 전체 라운드트립 테스트."""

    def test_positions_round_trip(self, tmp_state_file, sample_slots,
                                  sample_positions, sample_entry_times,
                                  sample_trade_history):
        save_state(
            slots=sample_slots,
            slot_positions=sample_positions,
            slot_entry_times=sample_entry_times,
            trade_history=sample_trade_history,
            paper_balance=1000.0,
            paper_mode=True,
            filepath=tmp_state_file,
        )

        state = load_state(tmp_state_file)
        positions = deserialize_positions(state['positions'])
        entry_times = deserialize_entry_times(state['entry_times'])

        assert 0 in positions
        assert positions[0]['symbol1'] == 'BTCUSDT'
        assert positions[0]['entry_price1'] == 50000.0
        assert 0 in entry_times
        assert entry_times[0].year == 2025

        # _df1 제외 확인
        assert '_df1' not in positions[0]

    def test_trade_history_round_trip(self, tmp_state_file, sample_trade_history):
        save_state(
            slots=[],
            slot_positions={},
            slot_entry_times={},
            trade_history=sample_trade_history,
            paper_balance=1000.0,
            paper_mode=True,
            filepath=tmp_state_file,
        )

        state = load_state(tmp_state_file)
        trades = state['trade_history']
        assert len(trades) == 1
        assert trades[0]['pnl'] == 5.50
        assert trades[0]['reason'] == 'mean_reversion'
