"""BTC Crash Gate 테스트.

BTC 급락/급등 시 신규 진입을 차단하는 매크로 방어 레이어를 검증한다.

테스트 범위:
- check_btc_crash_gate: BTC 수익률 기반 차단 로직
- execute_pair_trade 통합: BTC gate가 진입 흐름에서 올바르게 동작하는지
- Settings: 환경변수→config 전파
- Edge cases: API 에러, 데이터 부족, 비활성화
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════
# 1. Settings 테스트
# ═══════════════════════════════════════════════════════════

class TestBTCCrashGateSettings:
    """BTC crash gate 환경변수와 config 전파 테스트."""

    def test_default_settings(self, monkeypatch):
        """기본값이 올바르게 설정되는지 확인."""
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")

        from quantbox.core.settings import get_settings
        s = get_settings()
        assert s.btc_crash_gate_enabled is True
        assert s.btc_crash_window == 8
        assert s.btc_crash_threshold == 0.03

    def test_custom_settings(self, monkeypatch):
        """환경변수 커스텀 값이 반영되는지 확인."""
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BTC_CRASH_GATE_ENABLED", "false")
        monkeypatch.setenv("BTC_CRASH_WINDOW", "12")
        monkeypatch.setenv("BTC_CRASH_THRESHOLD", "0.05")

        from quantbox.core.settings import get_settings
        s = get_settings()
        assert s.btc_crash_gate_enabled is False
        assert s.btc_crash_window == 12
        assert s.btc_crash_threshold == 0.05

    def test_to_trader_config_includes_btc_gate(self, monkeypatch):
        """to_trader_config()에 BTC gate 설정이 포함되는지 확인."""
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")

        from quantbox.core.settings import get_settings
        cfg = get_settings().to_trader_config()
        assert 'btc_crash_gate_enabled' in cfg
        assert 'btc_crash_window' in cfg
        assert 'btc_crash_threshold' in cfg
        assert cfg['btc_crash_gate_enabled'] is True
        assert cfg['btc_crash_window'] == 8
        assert cfg['btc_crash_threshold'] == 0.03


# ═══════════════════════════════════════════════════════════
# 2. check_btc_crash_gate 단위 테스트
# ═══════════════════════════════════════════════════════════

class TestCheckBTCCrashGate:
    """check_btc_crash_gate() 메서드 단위 테스트."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            return LivePairTrader(paper_mode=True)

    def _make_klines(self, open_price: float, close_price: float, n_candles: int = 9):
        """BTC kline 데이터를 생성한다 (open→close 직선 보간).

        Args:
            open_price: 가장 오래된 봉의 시가.
            close_price: 가장 최근 봉의 종가.
            n_candles: 봉 수 (window+1).

        Returns:
            Binance futures_klines 형식의 리스트.
        """
        prices = np.linspace(open_price, close_price, n_candles)
        klines = []
        for i, p in enumerate(prices):
            klines.append([
                1000 * i,       # Open Time
                str(p),         # Open
                str(p * 1.001), # High
                str(p * 0.999), # Low
                str(p),         # Close
                "100.0",        # Volume
                1000 * (i + 1), # Close Time
                "10000.0",      # Quote Volume
                100,            # Trades
                "50.0",         # Taker Buy Base
                "5000.0",       # Taker Buy Quote
                "0",            # Ignore
            ])
        # 가장 오래된 봉: Close = open_price (live는 Close-to-Close 기준)
        # 가장 최근 봉: Close = close_price
        klines[0][4] = str(open_price)   # oldest Close
        klines[-1][4] = str(close_price)  # newest Close
        return klines

    def test_gate_blocks_on_crash(self, trader):
        """BTC -5% 하락 시 진입 차단."""
        klines = self._make_klines(70000, 66500, n_candles=9)  # -5%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is True
        assert ret < -0.03

    def test_gate_blocks_on_pump(self, trader):
        """BTC +5% 급등 시에도 진입 차단 (양방향)."""
        klines = self._make_klines(70000, 73500, n_candles=9)  # +5%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is True
        assert ret > 0.03

    def test_gate_passes_on_normal(self, trader):
        """BTC ±1% 범위 내 정상 움직임 시 통과."""
        klines = self._make_klines(70000, 70700, n_candles=9)  # +1%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False
        assert abs(ret) < 0.03

    def test_gate_passes_at_exact_threshold(self, trader):
        """BTC 정확히 ±3% 경계에서: 초과하지 않으므로 통과."""
        klines = self._make_klines(70000, 72100, n_candles=9)  # +3%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False  # 3% 이하이므로 통과

    def test_gate_blocks_just_above_threshold(self, trader):
        """BTC 3.01% — 임계값 초과 시 차단."""
        klines = self._make_klines(70000, 72110, n_candles=9)  # ~3.01%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is True

    def test_gate_disabled(self, trader):
        """비활성화 시 항상 통과."""
        trader.config['btc_crash_gate_enabled'] = False

        # 극단적 하락이라도 비활성 시 통과
        klines = self._make_klines(70000, 50000, n_candles=9)  # -28.6%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False
        assert ret == 0.0

    def test_gate_api_error_fail_open(self, trader):
        """API 에러 시 fail-open (차단하지 않음)."""
        trader.client.futures_klines = MagicMock(
            side_effect=Exception("API timeout")
        )

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False
        assert ret == 0.0

    def test_gate_empty_klines(self, trader):
        """빈 klines 반환 시 fail-open."""
        trader.client.futures_klines = MagicMock(return_value=[])

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False

    def test_gate_single_candle(self, trader):
        """봉 1개만 있을 때도 정상 동작."""
        klines = self._make_klines(70000, 70000, n_candles=2)
        trader.client.futures_klines = MagicMock(return_value=klines[:1])

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False

    def test_gate_custom_threshold(self, trader):
        """커스텀 threshold 적용."""
        trader.config['btc_crash_threshold'] = 0.01  # 1%로 타이트하게

        klines = self._make_klines(70000, 71500, n_candles=9)  # +2.14%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is True

    def test_gate_custom_window(self, trader):
        """커스텀 window가 API 호출에 반영되는지."""
        trader.config['btc_crash_window'] = 16

        klines = self._make_klines(70000, 70100, n_candles=17)  # 0.14%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False

        # futures_klines가 limit=17 (window+1)로 호출되었는지 확인
        call_args = trader.client.futures_klines.call_args
        assert call_args[1]['limit'] == 17

    def test_gate_klines_called_with_btcusdt(self, trader):
        """반드시 BTCUSDT로 조회하는지 확인."""
        klines = self._make_klines(70000, 70100, n_candles=9)
        trader.client.futures_klines = MagicMock(return_value=klines)

        trader.check_btc_crash_gate()

        call_args = trader.client.futures_klines.call_args
        assert call_args[1]['symbol'] == 'BTCUSDT'

    def test_gate_return_value_accuracy(self, trader):
        """수익률 계산 정확성 검증."""
        klines = self._make_klines(70000, 68600, n_candles=9)  # -2%
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is False  # 2% < 3% threshold
        assert abs(ret - (-0.02)) < 0.001


# ═══════════════════════════════════════════════════════════
# 3. execute_pair_trade 통합 테스트
# ═══════════════════════════════════════════════════════════

class TestBTCGateInExecutePairTrade:
    """BTC crash gate가 execute_pair_trade 진입 흐름에서 올바르게 동작하는지."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            return LivePairTrader(paper_mode=True)

    def _make_crash_klines(self):
        """BTC -5% 하락 klines."""
        prices = np.linspace(70000, 66500, 9)
        klines = []
        for i, p in enumerate(prices):
            klines.append([
                1000 * i, str(70000 if i == 0 else p), str(p * 1.001),
                str(p * 0.999), str(66500 if i == 8 else p),
                "100.0", 1000 * (i + 1), "10000.0", 100, "50.0", "5000.0", "0",
            ])
        return klines

    def test_btc_gate_blocks_before_other_gates(self, trader):
        """BTC gate가 z-cap/vol gate보다 먼저 실행되어 차단하는지."""
        from quantbox.strategy.pair_selector import PairCandidate

        crash_klines = self._make_crash_klines()
        trader.client.futures_klines = MagicMock(return_value=crash_klines)

        pair = PairCandidate(
            symbol1="ETHUSDT", symbol2="XRPUSDT",
            correlation=0.9, cointegration_pvalue=0.01,
            half_life=10.0, hurst=0.3, combined_score=5.0
        )
        pair_info = {
            'hedge_ratio': 1.0,
            'spread_mean': 0.0,
            'spread_std': 0.01,
            'half_life': 10.0,
        }

        # get_price 등이 호출되지 않아야 함 (BTC gate에서 먼저 차단)
        trader.get_price = MagicMock(return_value=100.0)

        result = trader.execute_pair_trade(0, pair, pair_info, 200.0)

        assert result is False
        # get_price는 호출되지 않아야 함 — BTC gate가 먼저 차단
        trader.get_price.assert_not_called()

    def test_btc_gate_passes_allows_normal_flow(self, trader):
        """BTC gate 통과 시 이후 게이트(z-cap 등)로 정상 진행."""
        from quantbox.strategy.pair_selector import PairCandidate

        # BTC 정상 (+0.5%)
        prices = np.linspace(70000, 70350, 9)
        normal_klines = []
        for i, p in enumerate(prices):
            normal_klines.append([
                1000 * i, str(70000 if i == 0 else p), str(p * 1.001),
                str(p * 0.999), str(70350 if i == 8 else p),
                "100.0", 1000 * (i + 1), "10000.0", 100, "50.0", "5000.0", "0",
            ])
        trader.client.futures_klines = MagicMock(return_value=normal_klines)

        pair = PairCandidate(
            symbol1="ETHUSDT", symbol2="XRPUSDT",
            correlation=0.9, cointegration_pvalue=0.01,
            half_life=10.0, hurst=0.3, combined_score=5.0
        )
        pair_info = {
            'hedge_ratio': 1.0,
            'spread_mean': 0.0,
            'spread_std': 0.01,
            'half_life': 10.0,
        }

        # z-score가 entry threshold 미달이면 False (BTC gate는 이미 통과)
        trader.get_price = MagicMock(return_value=100.0)
        trader.calculate_zscore = MagicMock(return_value=0.5)  # < entry_zscore

        result = trader.execute_pair_trade(0, pair, pair_info, 200.0)

        assert result is False
        # get_price가 호출됨 = BTC gate를 통과했다는 증거
        assert trader.get_price.call_count >= 1


# ═══════════════════════════════════════════════════════════
# 4. Feb 13 시나리오 재현 테스트
# ═══════════════════════════════════════════════════════════

class TestFeb13ScenarioReplay:
    """2026-02-13 BTC 크래시 시나리오를 재현하여 gate 동작 검증.

    당일 BTC $70K→$65K (-7.1%) → 모든 페어 진입이 차단되어야 함.
    """

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            return LivePairTrader(paper_mode=True)

    def test_feb13_btc_crash_blocked(self, trader):
        """BTC $70K→$65K (-7.1%) 시 진입 차단 확인."""
        # 2시간(8봉) 동안 $70K→$65K 하락 재현
        prices = np.linspace(70000, 65000, 9)
        klines = []
        for i, p in enumerate(prices):
            klines.append([
                1000 * i, str(70000 if i == 0 else p), str(p * 1.001),
                str(p * 0.999), str(65000 if i == 8 else p),
                "1000.0", 1000 * (i + 1), "100000.0", 500, "500.0", "50000.0", "0",
            ])
        trader.client.futures_klines = MagicMock(return_value=klines)

        blocked, ret = trader.check_btc_crash_gate()
        assert blocked is True
        assert abs(ret - (-70000 + 65000) / 70000) < 0.001

    def test_feb13_all_entries_blocked(self, trader):
        """크래시 중 여러 페어에 대한 진입 시도가 모두 차단되는지."""
        from quantbox.strategy.pair_selector import PairCandidate

        # BTC 크래시 klines
        prices = np.linspace(70000, 65000, 9)
        crash_klines = []
        for i, p in enumerate(prices):
            crash_klines.append([
                1000 * i, str(70000 if i == 0 else p), str(p * 1.001),
                str(p * 0.999), str(65000 if i == 8 else p),
                "1000.0", 1000 * (i + 1), "100000.0", 500, "500.0", "50000.0", "0",
            ])
        trader.client.futures_klines = MagicMock(return_value=crash_klines)

        pairs = [
            ("ETHUSDT", "XRPUSDT"),
            ("SOLUSDT", "AVAXUSDT"),
            ("DOTUSDT", "ATOMUSDT"),
        ]

        for sym1, sym2 in pairs:
            pair = PairCandidate(
                symbol1=sym1, symbol2=sym2,
                correlation=0.9, cointegration_pvalue=0.01,
                half_life=10.0, hurst=0.3, combined_score=5.0
            )
            pair_info = {
                'hedge_ratio': 1.0,
                'spread_mean': 0.0,
                'spread_std': 0.01,
                'half_life': 10.0,
            }
            result = trader.execute_pair_trade(0, pair, pair_info, 200.0)
            assert result is False, f"Entry should be blocked for {sym1}/{sym2}"


# ═══════════════════════════════════════════════════════════
# 5. Discord 상태 표시 테스트
# ═══════════════════════════════════════════════════════════

class TestBTCGateStatusDisplay:
    """Discord 상태 명령에 BTC gate 정보가 표시되는지."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            return LivePairTrader(paper_mode=True)

    def test_status_shows_btc_green(self, trader):
        """BTC 정상 시 🟢 표시."""
        normal_klines = []
        for i in range(9):
            normal_klines.append([
                1000 * i, "70000", "70100", "69900", "70100",
                "100.0", 1000 * (i + 1), "10000.0", 100, "50.0", "5000.0", "0",
            ])
        trader.client.futures_klines = MagicMock(return_value=normal_klines)

        status = trader._cmd_status()
        assert "🟢" in status
        assert "BTC" in status

    def test_status_shows_btc_red(self, trader):
        """BTC 크래시 시 🔴 표시."""
        prices = np.linspace(70000, 65000, 9)
        crash_klines = []
        for i, p in enumerate(prices):
            crash_klines.append([
                1000 * i, str(70000 if i == 0 else p), str(p * 1.001),
                str(p * 0.999), str(65000 if i == 8 else p),
                "1000.0", 1000 * (i + 1), "100000.0", 500, "500.0", "50000.0", "0",
            ])
        trader.client.futures_klines = MagicMock(return_value=crash_klines)

        status = trader._cmd_status()
        assert "🔴" in status
        assert "BTC" in status
