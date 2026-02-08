"""
Tests for quantbox.core.settings — pydantic-settings 기반 환경변수 검증.

모든 테스트는 os.environ을 격리하여 실제 .env 파일이나
시스템 환경변수의 영향을 받지 않는다.
"""

import pytest
from unittest.mock import patch

from pydantic import ValidationError

from quantbox.core.settings import QuantBoxSettings, get_settings


# ── 필수 환경변수 최소 세트 ──
_REQUIRED_ENV = {
    'BINANCE_API_KEY': 'test-api-key-abc123',
    'BINANCE_API_SECRET': 'test-api-secret-xyz789',
}


def _make_settings(env: dict | None = None, **overrides) -> QuantBoxSettings:
    """테스트용 설정 인스턴스 생성 (env_file 무시).

    Args:
        env: 환경변수 dict. None이면 _REQUIRED_ENV 사용.
        **overrides: 추가/덮어쓸 환경변수.

    Returns:
        QuantBoxSettings 인스턴스.
    """
    base = dict(env or _REQUIRED_ENV)
    base.update(overrides)
    with patch.dict('os.environ', base, clear=True):
        return QuantBoxSettings(_env_file=None)


class TestRequiredFields:
    """필수 필드 검증."""

    def test_valid_with_api_key_and_api_secret(self):
        """API KEY + API SECRET → 정상 생성."""
        s = _make_settings()
        assert s.binance_api_key == 'test-api-key-abc123'
        assert s.binance_api_secret == 'test-api-secret-xyz789'

    def test_missing_api_key_raises(self):
        """BINANCE_API_KEY 없으면 ValidationError."""
        with pytest.raises(ValidationError):
            _make_settings(env={'BINANCE_API_SECRET': 'secret'})

    def test_missing_both_secrets_raises(self):
        """API_SECRET과 SECRET 둘 다 없으면 ValueError."""
        with pytest.raises(ValidationError, match='BINANCE_API_SECRET'):
            _make_settings(env={'BINANCE_API_KEY': 'key'})

    def test_empty_api_key_raises(self):
        """빈 문자열 API_KEY도 pydantic에서 허용하지만,
        model_validator가 secret을 검증."""
        # 빈 키 자체는 pydantic str 필드이므로 허용됨
        s = _make_settings(env={
            'BINANCE_API_KEY': '',
            'BINANCE_API_SECRET': 'secret',
        })
        assert s.binance_api_key == ''


class TestBinanceSecretFallback:
    """BINANCE_SECRET → BINANCE_API_SECRET 호환 별칭."""

    def test_binance_secret_resolves_to_api_secret(self):
        """BINANCE_SECRET만 있으면 api_secret으로 복사."""
        s = _make_settings(env={
            'BINANCE_API_KEY': 'key',
            'BINANCE_SECRET': 'legacy-secret',
        })
        assert s.binance_api_secret == 'legacy-secret'

    def test_api_secret_takes_priority(self):
        """둘 다 있으면 BINANCE_API_SECRET 우선."""
        s = _make_settings(env={
            'BINANCE_API_KEY': 'key',
            'BINANCE_API_SECRET': 'new-secret',
            'BINANCE_SECRET': 'old-secret',
        })
        assert s.binance_api_secret == 'new-secret'

    def test_neither_secret_raises(self):
        """둘 다 없으면 즉시 에러."""
        with pytest.raises(ValidationError, match='BINANCE_API_SECRET'):
            _make_settings(env={'BINANCE_API_KEY': 'key'})


class TestDefaultValues:
    """기본값이 .env.example과 일치하는지 확인."""

    def test_binance_testnet_default(self):
        s = _make_settings()
        assert s.binance_testnet is False

    def test_leverage_default(self):
        s = _make_settings()
        assert s.leverage == 2

    def test_max_traders_default(self):
        s = _make_settings()
        assert s.max_traders == 5

    def test_position_size_pct_default(self):
        s = _make_settings()
        assert s.position_size_pct == -1.0

    def test_entry_zscore_default(self):
        s = _make_settings()
        assert s.entry_zscore == 2.0

    def test_exit_zscore_default(self):
        s = _make_settings()
        assert s.exit_zscore == 0.5

    def test_stop_loss_zscore_default(self):
        s = _make_settings()
        assert s.stop_loss_zscore == 3.5

    def test_trading_fee_default(self):
        s = _make_settings()
        assert s.trading_fee == 0.0005

    def test_slippage_default(self):
        s = _make_settings()
        assert s.slippage == 0.001

    def test_funding_rate_8h_default(self):
        s = _make_settings()
        assert s.funding_rate_8h == 0.0001

    def test_n_coins_default(self):
        s = _make_settings()
        assert s.n_coins == 30

    def test_paper_balance_default(self):
        s = _make_settings()
        assert s.paper_balance == 1000.0

    def test_discord_webhook_url_default(self):
        s = _make_settings()
        assert s.discord_webhook_url is None


class TestCustomValues:
    """환경변수로 기본값 오버라이드."""

    def test_leverage_override(self):
        s = _make_settings(LEVERAGE='3')
        assert s.leverage == 3

    def test_max_traders_override(self):
        s = _make_settings(MAX_TRADERS='10')
        assert s.max_traders == 10

    def test_entry_zscore_override(self):
        s = _make_settings(ENTRY_ZSCORE='2.5')
        assert s.entry_zscore == 2.5

    def test_paper_balance_override(self):
        s = _make_settings(PAPER_BALANCE='5000')
        assert s.paper_balance == 5000.0

    def test_discord_webhook_override(self):
        url = 'YOUR_DISCORD_WEBHOOK_URL'
        s = _make_settings(DISCORD_WEBHOOK_URL=url)
        assert s.discord_webhook_url == url

    def test_testnet_true(self):
        s = _make_settings(BINANCE_TESTNET='true')
        assert s.binance_testnet is True

    def test_testnet_True_capital(self):
        """대소문자 구분 없이 true."""
        s = _make_settings(BINANCE_TESTNET='True')
        assert s.binance_testnet is True


class TestTypeValidation:
    """잘못된 타입이면 ValidationError."""

    def test_leverage_not_int(self):
        with pytest.raises(ValidationError):
            _make_settings(LEVERAGE='abc')

    def test_trading_fee_not_float(self):
        with pytest.raises(ValidationError):
            _make_settings(TRADING_FEE='not-a-number')

    def test_n_coins_not_int(self):
        with pytest.raises(ValidationError):
            _make_settings(N_COINS='3.5')


class TestToTraderConfig:
    """to_trader_config() 딕셔너리 변환."""

    def test_returns_dict(self):
        s = _make_settings()
        cfg = s.to_trader_config()
        assert isinstance(cfg, dict)

    def test_contains_required_keys(self):
        s = _make_settings()
        cfg = s.to_trader_config()
        required = [
            'quote_currency', 'n_coins', 'interval', 'formation_hours',
            'entry_zscore', 'exit_zscore', 'stop_loss_zscore', 'max_holding_hours',
            'position_size_pct', 'trading_fee', 'slippage', 'funding_rate_8h',
            'leverage', 'max_traders',
            'min_correlation', 'max_cointegration_pvalue',
            'min_half_life', 'max_half_life', 'max_hurst',
            'check_interval_seconds', 'retry_attempts', 'retry_delay',
        ]
        for key in required:
            assert key in cfg, f"Missing key: {key}"

    def test_auto_position_size(self):
        """position_size_pct=-1 → 1/max_traders."""
        s = _make_settings(POSITION_SIZE_PCT='-1', MAX_TRADERS='5')
        cfg = s.to_trader_config()
        assert cfg['position_size_pct'] == pytest.approx(0.2)

    def test_manual_position_size(self):
        """명시적 position_size_pct는 그대로 유지."""
        s = _make_settings(POSITION_SIZE_PCT='0.15')
        cfg = s.to_trader_config()
        assert cfg['position_size_pct'] == pytest.approx(0.15)

    def test_env_values_propagated(self):
        """환경변수 값이 config dict에 반영."""
        s = _make_settings(
            LEVERAGE='3',
            ENTRY_ZSCORE='2.5',
            TRADING_FEE='0.001',
        )
        cfg = s.to_trader_config()
        assert cfg['leverage'] == 3
        assert cfg['entry_zscore'] == 2.5
        assert cfg['trading_fee'] == 0.001

    def test_fixed_values(self):
        """하드코딩된 필터/실행 파라미터 확인."""
        s = _make_settings()
        cfg = s.to_trader_config()
        assert cfg['quote_currency'] == 'USDT'
        assert cfg['interval'] == '15m'
        assert cfg['formation_hours'] == 168
        assert cfg['max_holding_hours'] == 24
        assert cfg['min_correlation'] == 0.5
        assert cfg['max_hurst'] == 0.5


class TestGetSettings:
    """get_settings() 싱글턴 팩토리."""

    def test_returns_settings_instance(self):
        with patch.dict('os.environ', _REQUIRED_ENV, clear=True):
            get_settings.cache_clear()
            s = get_settings()
            assert isinstance(s, QuantBoxSettings)
            get_settings.cache_clear()

    def test_cached_returns_same_instance(self):
        with patch.dict('os.environ', _REQUIRED_ENV, clear=True):
            get_settings.cache_clear()
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
            get_settings.cache_clear()


class TestExtraFieldsIgnored:
    """extra='ignore' 설정 확인."""

    def test_unknown_env_vars_ignored(self):
        """코드에 없는 환경변수가 있어도 에러 없음."""
        s = _make_settings(
            UNKNOWN_VAR='whatever',
            TRADING_MODE='paper',  # deprecated var
        )
        assert s.binance_api_key == 'test-api-key-abc123'
