"""Dynamic exit z-score 기능 테스트.

Half-life 기반으로 exit_zscore를 동적 조정하는 기능의 로직 검증 및
백테스트/실거래 파리티 확인.
"""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace

from quantbox.backtest.continuous_backtest import (
    ContinuousBacktestConfig,
    ContinuousPairBacktest,
    ExitReason,
)


# ════════════════════════════════════════════════════════════════
# 1. Config 파라미터 존재 확인
# ════════════════════════════════════════════════════════════════

class TestDynamicExitConfig:
    """Config에 dynamic exit 관련 필드가 존재하는지 확인."""

    def test_backtest_config_has_dynamic_exit_fields(self):
        config = ContinuousBacktestConfig()
        assert hasattr(config, 'use_dynamic_exit')
        assert hasattr(config, 'dynamic_exit_base')
        assert hasattr(config, 'dynamic_exit_range')

    def test_backtest_config_defaults(self):
        config = ContinuousBacktestConfig()
        assert config.use_dynamic_exit is False
        assert config.dynamic_exit_base == 0.3
        assert config.dynamic_exit_range == 0.7

    def test_live_settings_has_dynamic_exit_fields(self):
        from quantbox.core.settings import QuantBoxSettings
        fields = QuantBoxSettings.model_fields
        assert 'use_dynamic_exit' in fields
        assert 'dynamic_exit_base' in fields
        assert 'dynamic_exit_range' in fields

    def test_live_config_dict_includes_dynamic_exit(self):
        """to_trader_config()에 dynamic exit 키가 포함되어야 한다."""
        from quantbox.core.settings import QuantBoxSettings
        with patch.dict('os.environ', {
            'BINANCE_API_KEY': 'test',
            'BINANCE_API_SECRET': 'test',
            'USE_DYNAMIC_EXIT': 'true',
            'DYNAMIC_EXIT_BASE': '0.4',
            'DYNAMIC_EXIT_RANGE': '0.6',
        }):
            settings = QuantBoxSettings()
            config = settings.to_trader_config()
            assert config['use_dynamic_exit'] is True
            assert config['dynamic_exit_base'] == 0.4
            assert config['dynamic_exit_range'] == 0.6


# ════════════════════════════════════════════════════════════════
# 2. 동적 exit z-score 계산 로직 (백테스트)
# ════════════════════════════════════════════════════════════════

class TestBacktestDynamicExitCalc:
    """ContinuousPairBacktest._get_effective_exit_zscore 테스트."""

    @pytest.fixture
    def engine_static(self):
        """Dynamic exit 비활성화 엔진."""
        config = ContinuousBacktestConfig(use_dynamic_exit=False)
        return ContinuousPairBacktest(config)

    @pytest.fixture
    def engine_dynamic(self):
        """Dynamic exit 활성화 엔진.
        
        min_half_life=2, max_half_life=50, base=0.3, range=0.7
        → exit_z = 0.3 + (hl-2)/(50-2) * 0.7
        """
        config = ContinuousBacktestConfig(
            use_dynamic_exit=True,
            dynamic_exit_base=0.3,
            dynamic_exit_range=0.7,
        )
        return ContinuousPairBacktest(config)

    def test_static_returns_fixed_exit(self, engine_static):
        """Dynamic 비활성화 시 고정 exit_zscore 반환."""
        pos = {'half_life': 10.0}
        assert engine_static._get_effective_exit_zscore(pos) == 0.5

    def test_dynamic_fast_pair(self, engine_dynamic):
        """빠른 회귀 (hl=2, 최소값) → base에 가까운 exit."""
        pos = {'half_life': 2.0}
        result = engine_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - 0.3) < 1e-6

    def test_dynamic_slow_pair(self, engine_dynamic):
        """느린 회귀 (hl=50, 최대값) → base+range에 가까운 exit."""
        pos = {'half_life': 50.0}
        result = engine_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - 1.0) < 1e-6

    def test_dynamic_mid_pair(self, engine_dynamic):
        """중간 회귀 (hl=26) → 보간된 exit."""
        pos = {'half_life': 26.0}
        result = engine_dynamic._get_effective_exit_zscore(pos)
        expected = 0.3 + (26 - 2) / (50 - 2) * 0.7
        assert abs(result - expected) < 1e-6

    def test_dynamic_clamps_below_min(self, engine_dynamic):
        """hl < min_half_life 시 base로 clamp."""
        pos = {'half_life': 0.5}
        result = engine_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - 0.3) < 1e-6

    def test_dynamic_clamps_above_max(self, engine_dynamic):
        """hl > max_half_life 시 base+range로 clamp."""
        pos = {'half_life': 100.0}
        result = engine_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - 1.0) < 1e-6

    def test_missing_half_life_uses_max(self, engine_dynamic):
        """position에 half_life가 없으면 max_half_life 사용 (보수적)."""
        pos = {}
        result = engine_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - 1.0) < 1e-6


# ════════════════════════════════════════════════════════════════
# 3. 동적 exit z-score 계산 로직 (라이브)
# ════════════════════════════════════════════════════════════════

class TestLiveDynamicExitCalc:
    """LivePairTrader._get_effective_exit_zscore 테스트."""

    @pytest.fixture
    def trader_static(self):
        from quantbox.live_trader import LivePairTrader
        with patch.dict('os.environ', {
            'BINANCE_API_KEY': 'test',
            'BINANCE_API_SECRET': 'test',
        }):
            with patch.object(LivePairTrader, '__init__', lambda self, **kw: None):
                t = LivePairTrader.__new__(LivePairTrader)
                t.config = {
                    'exit_zscore': 0.5,
                    'use_dynamic_exit': False,
                    'min_half_life': 3,
                    'max_half_life': 60,
                    'dynamic_exit_base': 0.3,
                    'dynamic_exit_range': 0.7,
                }
                return t

    @pytest.fixture
    def trader_dynamic(self):
        from quantbox.live_trader import LivePairTrader
        with patch.dict('os.environ', {
            'BINANCE_API_KEY': 'test',
            'BINANCE_API_SECRET': 'test',
        }):
            with patch.object(LivePairTrader, '__init__', lambda self, **kw: None):
                t = LivePairTrader.__new__(LivePairTrader)
                t.config = {
                    'exit_zscore': 0.5,
                    'use_dynamic_exit': True,
                    'min_half_life': 3,
                    'max_half_life': 60,
                    'dynamic_exit_base': 0.3,
                    'dynamic_exit_range': 0.7,
                }
                return t

    def test_static_returns_fixed(self, trader_static):
        pos = {'half_life': 10.0}
        assert trader_static._get_effective_exit_zscore(pos) == 0.5

    def test_dynamic_fast(self, trader_dynamic):
        pos = {'half_life': 3.0}
        result = trader_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - 0.3) < 1e-6

    def test_dynamic_slow(self, trader_dynamic):
        pos = {'half_life': 60.0}
        result = trader_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - 1.0) < 1e-6

    def test_dynamic_mid(self, trader_dynamic):
        pos = {'half_life': 31.5}
        expected = 0.3 + (31.5 - 3) / (60 - 3) * 0.7
        result = trader_dynamic._get_effective_exit_zscore(pos)
        assert abs(result - expected) < 1e-6


# ════════════════════════════════════════════════════════════════
# 4. 백테스트/실거래 파리티
# ════════════════════════════════════════════════════════════════

class TestDynamicExitParity:
    """백테스트와 실거래 로직의 파리티 검증."""

    def test_same_formula_produces_same_result(self):
        """동일 파라미터, 동일 half_life → 동일 exit_zscore."""
        from quantbox.live_trader import LivePairTrader
        config_vals = {
            'use_dynamic_exit': True,
            'dynamic_exit_base': 0.3,
            'dynamic_exit_range': 0.7,
            'min_half_life': 2.0,
            'max_half_life': 50.0,
            'exit_zscore': 0.5,
        }

        # Backtest
        bt_config = ContinuousBacktestConfig(**config_vals)
        bt_engine = ContinuousPairBacktest(bt_config)

        # Live
        with patch.object(LivePairTrader, '__init__', lambda self, **kw: None):
            live = LivePairTrader.__new__(LivePairTrader)
            live.config = config_vals

        for hl in [2.0, 10.0, 26.0, 40.0, 50.0]:
            pos = {'half_life': hl}
            bt_result = bt_engine._get_effective_exit_zscore(pos)
            live_result = live._get_effective_exit_zscore(pos)
            assert abs(bt_result - live_result) < 1e-10, (
                f"Parity violation at hl={hl}: backtest={bt_result}, live={live_result}"
            )

    def test_position_info_stores_half_life(self):
        """position_info에 half_life가 저장되는지 확인 (백테스트 코드 검사)."""
        import inspect
        source = inspect.getsource(ContinuousPairBacktest._enter_slot)
        assert "'half_life'" in source, (
            "_enter_slot의 position_info에 half_life 저장이 누락됨"
        )

    def test_disabled_by_default(self):
        """BacktestConfig 기본값은 비활성화, Settings 기본값은 활성화."""
        config = ContinuousBacktestConfig()
        assert config.use_dynamic_exit is False

        from quantbox.core.settings import QuantBoxSettings
        with patch.dict('os.environ', {
            'BINANCE_API_KEY': 'test',
            'BINANCE_API_SECRET': 'test',
        }, clear=True):
            settings = QuantBoxSettings()
            # settings.py 기본값은 True (라이브에서 동적 청산 활성)
            assert settings.use_dynamic_exit is True


# ════════════════════════════════════════════════════════════════
# 5. Half-life가 position_info에 저장되는지 구조 확인
# ════════════════════════════════════════════════════════════════

class TestHalfLifeInPositionInfo:
    """live_trader execute_pair_trade에서 half_life 저장 확인."""

    def test_live_position_data_has_half_life(self):
        """execute_pair_trade의 position_data에 half_life 키 존재."""
        import inspect
        from quantbox.live_trader import LivePairTrader
        source = inspect.getsource(LivePairTrader.execute_pair_trade)
        assert "'half_life'" in source, (
            "execute_pair_trade의 position_data에 half_life 저장이 누락됨"
        )
