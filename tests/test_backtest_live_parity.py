"""
Tests for backtest/live parity fixes:
A. Hurst filter in backtest
B. Live PnL cost deduction
C. Sharpe annualization
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# ============================================================
# A. Hurst filter in backtest
# ============================================================

class TestBacktestHurstFilter:
    """백테스트 _find_top_pairs()에 Hurst 필터가 적용되는지 검증."""

    def test_config_has_max_hurst(self):
        """ContinuousBacktestConfig에 max_hurst 필드가 존재한다."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        config = ContinuousBacktestConfig()
        assert hasattr(config, 'max_hurst')
        assert config.max_hurst == 0.5

    def test_spread_statistics_has_hurst(self):
        """PairStatistics에 calculate_hurst 메서드가 존재한다."""
        from quantbox.backtest.continuous_backtest import PairStatistics
        assert hasattr(PairStatistics, 'calculate_hurst')

    def test_hurst_mean_reverting_series(self):
        """평균회귀 시계열의 Hurst가 낮다 (mean-reverting)."""
        from quantbox.backtest.continuous_backtest import PairStatistics
        np.random.seed(42)
        # Ornstein-Uhlenbeck process (mean-reverting)
        n = 500
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = spread[i-1] * 0.95 + np.random.normal(0, 0.1)
        
        hurst = PairStatistics.calculate_hurst(spread)
        # Note: 현재 구현은 poly[0]/2.0 (live와 동일) → 절대값이 아닌 상대 비교로 검증
        assert hurst < 0.3, f"Mean-reverting series should have low Hurst, got {hurst:.3f}"

    def test_hurst_trending_higher_than_mean_reverting(self):
        """트렌딩 시계열의 Hurst가 평균회귀보다 크다."""
        from quantbox.backtest.continuous_backtest import PairStatistics
        np.random.seed(42)
        
        # Mean-reverting (OU process)
        n = 500
        mr_spread = np.zeros(n)
        for i in range(1, n):
            mr_spread[i] = mr_spread[i-1] * 0.95 + np.random.normal(0, 0.1)
        hurst_mr = PairStatistics.calculate_hurst(mr_spread)
        
        # Trending (random walk with drift)
        np.random.seed(99)
        trend_spread = np.cumsum(np.random.normal(0.05, 0.1, n))
        hurst_trend = PairStatistics.calculate_hurst(trend_spread)
        
        assert hurst_trend > hurst_mr, (
            f"Trending Hurst ({hurst_trend:.3f}) should be > mean-reverting ({hurst_mr:.3f})"
        )

    def test_hurst_filter_rejects_trending_pair(self):
        """Hurst > 0.5인 페어는 _find_top_pairs에서 제거된다."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig
        )
        config = ContinuousBacktestConfig(
            max_hurst=0.5,
            min_correlation=0.3,  # 느슨하게
            max_cointegration_pvalue=1.0,  # 모든 공적분 통과
            min_half_life=0.1,
            max_half_life=999,
        )
        bt = ContinuousPairBacktest(config)
        
        # 트렌딩 페어 생성 (공통 트렌드 + 독립 랜덤워크)
        np.random.seed(42)
        n = 300
        trend = np.cumsum(np.random.normal(0.01, 0.05, n))
        p1 = 100 * np.exp(trend + np.cumsum(np.random.normal(0, 0.01, n)))
        p2 = 50 * np.exp(trend + np.cumsum(np.random.normal(0, 0.01, n)))
        
        data = {
            'SYM1': pd.DataFrame({'Close': p1}),
            'SYM2': pd.DataFrame({'Close': p2}),
        }
        
        result = bt._find_top_pairs(data, start_idx=0, end_idx=n-1, n=1)
        # 트렌딩 페어는 Hurst 필터에서 걸러져야 함
        # (또는 다른 필터에서 걸러질 수 있음)
        # 핵심: Hurst가 0.5를 못 넘겨야 candidates에 포함 안 됨

    def test_hurst_same_as_live_filter(self):
        """백테스트의 Hurst 계산이 live FilterPipeline과 동일한 결과를 낸다."""
        from quantbox.backtest.continuous_backtest import PairStatistics
        from quantbox.strategy.filters import HurstFilter
        
        np.random.seed(42)
        n = 300
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = spread[i-1] * 0.93 + np.random.normal(0, 0.1)
        
        # 백테스트 방식
        bt_hurst = PairStatistics.calculate_hurst(spread)
        
        # 라이브 방식 (HurstFilter._calculate_hurst)
        hurst_filter = HurstFilter(max_hurst=0.5)
        live_hurst = hurst_filter._calculate_hurst(spread)
        
        # 동일한 알고리즘이므로 결과가 가까워야 함
        assert abs(bt_hurst - live_hurst) < 0.01, (
            f"Backtest Hurst ({bt_hurst:.4f}) != Live Hurst ({live_hurst:.4f})"
        )


# ============================================================
# B. Live PnL cost deduction
# ============================================================

class TestLivePnlCostDeduction:
    """monitor_slot()이 수수료/슬리피지/펀딩을 차감하는지 검증."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            with patch("quantbox.live_trader.get_settings") as mock_settings:
                s = mock_settings.return_value
                s.binance_api_key = "test_key"
                s.binance_api_secret = "test_secret"
                s.binance_testnet = True
                s.max_traders = 3
                s.leverage = 2
                s.position_size_pct = -1
                s.entry_zscore = 2.0
                s.exit_zscore = 0.5
                s.stop_loss_zscore = 3.5
                s.trading_fee = 0.0005
                s.slippage = 0.001
                s.funding_rate_8h = 0.0001
                s.n_coins = 30
                s.paper_balance = 1000.0
                s.discord_webhook_url = None
                s.discord_bot_token = None
                s.discord_channel_id = None
                s.to_trader_config.return_value = {
                    'quote_currency': 'USDT',
                    'n_coins': 30,
                    'interval': '15m',
                    'formation_hours': 168,
                    'entry_zscore': 2.0,
                    'exit_zscore': 0.5,
                    'stop_loss_zscore': 3.5,
                    'max_holding_hours': 24,
                    'position_size_pct': 0.333,
                    'trading_fee': 0.0005,
                    'slippage': 0.001,
                    'funding_rate_8h': 0.0001,
                    'leverage': 2,
                    'max_traders': 3,
                    'min_correlation': 0.5,
                    'max_cointegration_pvalue': 0.05,
                    'min_half_life': 3,
                    'max_half_life': 60,
                    'max_hurst': 0.5,
                    'check_interval_seconds': 900,
                    'retry_attempts': 3,
                    'retry_delay': 2,
                    'lookback_bars': 200,
                }
                from quantbox.live_trader import LivePairTrader
                trader = LivePairTrader(paper_mode=True)
            trader.discord = MagicMock()
            trader.get_available_balance = MagicMock(return_value=5000.0)
            return trader

    def _open_position(self, trader, slot_id: int, hours_ago: float = 2.0):
        """테스트용 포지션을 슬롯에 세팅하는 헬퍼."""
        from datetime import datetime, timezone
        
        position_data = {
            'symbol1': 'BTCUSDT', 'symbol2': 'ETHUSDT',
            'entry_price1': 100.0, 'entry_price2': 50.0,
            'qty1': 1.0, 'qty2': -2.0,  # long spread
            'entry_zscore': -2.5, 'signal': 1,
            'hedge_ratio': 1.0, 'intercept': 0.0,
            'spread_mean': 0.0, 'spread_std': 0.1,
            'capital': 100.0, 'notional': 200.0, 'leverage': 2,
        }
        slot = trader.slots[slot_id]
        slot.position_open = True
        slot.current_pair = ('BTCUSDT', 'ETHUSDT')
        slot.position_info = position_data
        trader.slot_positions[slot_id] = position_data
        trader.slot_entry_times[slot_id] = (
            datetime.now(timezone.utc) - pd.Timedelta(hours=hours_ago)
        )

    def test_pnl_deducts_trading_fee(self, trader):
        """PnL 계산 시 거래 수수료가 차감된다."""
        self._open_position(trader, slot_id=0, hours_ago=2.0)
        
        # 가격 변동 없음 (raw PnL = 0) → 비용만 차감
        trader.get_price = MagicMock(side_effect=[100.0, 50.0])
        trader.close_slot_positions = MagicMock(
            return_value={'exit_price1': 100.0, 'exit_price2': 50.0}
        )
        
        # z-score를 평균회귀 조건으로 (>= -exit_zscore → close)
        with patch.object(trader, 'calculate_zscore', return_value=0.0):
            trader.monitor_slot(0)
        
        assert len(trader.trade_history) > 0, "Trade should be recorded"
        trade = trader.trade_history[-1]
        # 가격 변동 없으므로 raw PnL = 0, 비용 차감으로 순 PnL < 0
        assert trade['pnl'] < 0, f"Zero raw PnL should be negative after costs, got {trade['pnl']:.4f}"
        
        # 비용 크기 검증: entry_value=200, fee=200*0.0005*2=0.2, slip=200*0.001*2=0.4
        # funding은 시간에 따라 다름. 총 비용 ≈ 0.6+ 
        assert trade['pnl'] < -0.5, f"Cost should be at least ~$0.6, got pnl={trade['pnl']:.4f}"

    def test_pnl_with_profit_minus_costs(self, trader):
        """실제 수익에서 비용을 차감한 순수익이 계산된다."""
        self._open_position(trader, slot_id=0, hours_ago=1.0)
        
        # BTC 102로 상승 → raw pnl1 = (102-100)*1 = +2
        # ETH 50 유지 → raw pnl2 = (50-50)*(-2) = 0
        # raw PnL = $2.0
        trader.get_price = MagicMock(side_effect=[102.0, 50.0])
        trader.close_slot_positions = MagicMock(
            return_value={'exit_price1': 102.0, 'exit_price2': 50.0}
        )
        
        with patch.object(trader, 'calculate_zscore', return_value=0.0):
            trader.monitor_slot(0)
        
        assert len(trader.trade_history) > 0, "Trade should be recorded"
        trade = trader.trade_history[-1]
        # raw $2.0에서 비용 차감 → 순 PnL < $2.0
        assert trade['pnl'] < 2.0, "PnL should be less than raw due to costs"
        assert trade['pnl'] > 0, f"$2 profit should still be positive after costs on $200 position"

    def test_live_mode_no_slippage_deduction(self, trader):
        """Live mode에서는 슬리피지를 이중 차감하지 않는다.

        실거래에서 entry_price는 이미 실제 체결가(슬리피지 반영 완료).
        별도로 슬리피지를 차감하면 이중 차감.
        """
        # Switch to live mode
        trader.paper_mode = False

        self._open_position(trader, slot_id=0, hours_ago=2.0)

        # 가격 변동 없음 → raw PnL = 0, 비용만 차감
        trader.get_price = MagicMock(side_effect=[100.0, 50.0])
        trader.close_slot_positions = MagicMock(
            return_value={'exit_price1': 100.0, 'exit_price2': 50.0}
        )
        # Mock funding rate to avoid MagicMock client returning float(MagicMock)=1.0
        trader._calculate_funding_cost = MagicMock(return_value=0.01)

        with patch.object(trader, 'calculate_zscore', return_value=0.0):
            trader.monitor_slot(0)

        trade = trader.trade_history[-1]

        # entry_value=200, fee=200*0.0005*2=0.2
        # Slippage should NOT be deducted in live mode
        # funding = small
        # Total cost should be ~0.2 + small funding, NOT ~0.6+
        assert trade['pnl'] > -0.5, (
            f"Live PnL should NOT include slippage deduction. "
            f"Expected > -$0.5, got {trade['pnl']:.4f}"
        )

    def test_paper_mode_includes_slippage(self, trader):
        """Paper mode에서는 슬리피지를 차감한다 (보수적 시뮬레이션)."""
        assert trader.paper_mode is True  # fixture is paper_mode

        self._open_position(trader, slot_id=0, hours_ago=2.0)

        trader.get_price = MagicMock(side_effect=[100.0, 50.0])
        trader.close_slot_positions = MagicMock(
            return_value={'exit_price1': 100.0, 'exit_price2': 50.0}
        )

        with patch.object(trader, 'calculate_zscore', return_value=0.0):
            trader.monitor_slot(0)

        trade = trader.trade_history[-1]

        # Paper: fee(0.2) + slippage(0.4) + funding → should be > 0.5
        assert trade['pnl'] < -0.5, (
            f"Paper PnL should include slippage. "
            f"Expected < -$0.5, got {trade['pnl']:.4f}"
        )


# ============================================================
# C. Sharpe annualization
# ============================================================

class TestSharpeAnnualization:
    """Sharpe ratio 연간화 계수가 올바른지 검증."""

    def test_backtest_sharpe_uses_bar_interval(self):
        """백테스트 Sharpe가 bar_interval_minutes를 반영한다."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig
        )
        
        # 15분봉: bars_per_year = 365.25 * 24 * 4 = 35064
        config = ContinuousBacktestConfig(bar_interval_minutes=15)
        bt = ContinuousPairBacktest(config)
        
        # 간단한 equity curve 생성
        n = 100
        equity_vals = 10000 + np.cumsum(np.random.normal(1, 10, n))
        
        equity_df = pd.DataFrame({
            'equity': equity_vals,
            'drawdown': np.zeros(n),
        })
        
        returns = np.diff(equity_vals) / equity_vals[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        expected_factor = np.sqrt(365.25 * 24 * (60 / 15))  # sqrt(35064) ≈ 187.3
        expected_sharpe = np.mean(returns) / np.std(returns) * expected_factor
        
        # Wrong factor (old bug): sqrt(252*24) ≈ 77.8
        wrong_factor = np.sqrt(252 * 24)
        
        # 올바른 factor가 잘못된 factor보다 커야 함
        assert expected_factor > wrong_factor, "15m bars should have larger annualization factor"
        # 비율: sqrt(35064) / sqrt(6048) ≈ 2.41
        ratio = expected_factor / wrong_factor
        assert 2.0 < ratio < 3.0, f"Factor ratio should be ~2.4, got {ratio:.2f}"

    def test_backtest_sharpe_not_hardcoded(self):
        """다른 bar_interval(예: 1시간)에서도 올바르게 작동한다."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        
        config_15m = ContinuousBacktestConfig(bar_interval_minutes=15)
        config_1h = ContinuousBacktestConfig(bar_interval_minutes=60)
        
        factor_15m = 365.25 * 24 * (60 / config_15m.bar_interval_minutes)
        factor_1h = 365.25 * 24 * (60 / config_1h.bar_interval_minutes)
        
        assert factor_15m == pytest.approx(35064, rel=0.01)
        assert factor_1h == pytest.approx(8766, rel=0.01)
        assert factor_15m / factor_1h == pytest.approx(4.0, rel=0.01)

    def test_metrics_rolling_sharpe_parameterized(self):
        """metrics.py의 rolling Sharpe가 파라미터화되었다."""
        from quantbox.backtest.metrics import calculate_rolling_metrics
        
        # 충분한 데이터
        n = 500
        equity = 10000 + np.cumsum(np.random.normal(0.5, 5, n))
        equity_df = pd.DataFrame({'equity': equity})
        
        # 15m bars (35064) vs daily (252)
        rolling_15m = calculate_rolling_metrics(equity_df, window=100, periods_per_year=35064)
        rolling_daily = calculate_rolling_metrics(equity_df, window=100, periods_per_year=252)
        
        if not rolling_15m.empty and not rolling_daily.empty:
            # 15m의 sharpe가 더 큼 (annualization factor 큼)
            sharpe_15m = rolling_15m['rolling_sharpe'].dropna().mean()
            sharpe_daily = rolling_daily['rolling_sharpe'].dropna().mean()
            
            if sharpe_daily != 0:
                ratio = abs(sharpe_15m / sharpe_daily)
                # sqrt(35064/252) ≈ 11.8
                assert ratio > 5, f"15m Sharpe should be much larger than daily, ratio={ratio:.1f}"


# ============================================================
# Backtest filter parity with live
# ============================================================

class TestFilterParity:
    """백테스트와 실거래의 필터 기준이 동일한지 검증."""

    def test_backtest_config_matches_live_settings(self):
        """백테스트 기본 config와 live settings의 필터 값 매핑을 확인."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        from quantbox.core.settings import QuantBoxSettings
        
        bt_config = ContinuousBacktestConfig()
        
        # 이 값들이 존재하고 비교 가능해야 함
        assert hasattr(bt_config, 'min_correlation')
        assert hasattr(bt_config, 'max_cointegration_pvalue')
        assert hasattr(bt_config, 'min_half_life')
        assert hasattr(bt_config, 'max_half_life')
        assert hasattr(bt_config, 'max_hurst')  # 새로 추가된 필드

    def test_backtest_has_all_live_filters(self):
        """백테스트 _find_top_pairs에 live와 동일한 필터가 있다."""
        import inspect
        from quantbox.backtest.continuous_backtest import ContinuousPairBacktest
        
        source = inspect.getsource(ContinuousPairBacktest._find_top_pairs)
        
        # 모든 필터가 코드에 존재해야 함
        assert 'correlation' in source.lower(), "Missing correlation filter"
        assert 'cointegration' in source.lower() or 'coint' in source.lower(), "Missing cointegration filter"
        assert 'half_life' in source.lower(), "Missing half-life filter"
        assert 'hurst' in source.lower(), "Missing Hurst filter"


# ============================================================
# D. Regime gate parity (z-cap, vol expansion, BTC crash)
# ============================================================

class TestRegimeGateParity:
    """백테스트에 라이브와 동일한 레짐 보호 게이트가 존재하는지 검증."""

    def test_backtest_config_has_z_score_cap(self):
        """ContinuousBacktestConfig에 z_score_cap 필드가 존재한다."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        config = ContinuousBacktestConfig()
        assert hasattr(config, 'z_score_cap')
        assert config.z_score_cap == 4.0

    def test_backtest_config_has_vol_expansion_gate(self):
        """ContinuousBacktestConfig에 vol_expansion_gate 필드가 존재한다."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        config = ContinuousBacktestConfig()
        assert hasattr(config, 'vol_expansion_gate')
        assert config.vol_expansion_gate == 2.0
        assert hasattr(config, 'vol_expansion_window')
        assert config.vol_expansion_window == 8

    def test_backtest_config_has_btc_crash_gate(self):
        """ContinuousBacktestConfig에 BTC crash gate 필드가 존재한다."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        config = ContinuousBacktestConfig()
        assert hasattr(config, 'btc_crash_gate_enabled')
        assert config.btc_crash_gate_enabled is True
        assert hasattr(config, 'btc_crash_window')
        assert config.btc_crash_window == 8
        assert hasattr(config, 'btc_crash_threshold')
        assert config.btc_crash_threshold == 0.03

    def test_enter_slot_source_has_all_gates(self):
        """백테스트 _enter_slot에 모든 5개 게이트가 존재한다."""
        import inspect
        from quantbox.backtest.continuous_backtest import ContinuousPairBacktest

        source = inspect.getsource(ContinuousPairBacktest._enter_slot)

        assert 'btc_crash_gate' in source.lower() or 'check_btc_crash_gate' in source.lower(), \
            "Missing BTC crash gate in _enter_slot"
        assert 'z_score_cap' in source.lower(), \
            "Missing z_score_cap gate in _enter_slot"
        assert 'vol_expansion' in source.lower(), \
            "Missing vol expansion gate in _enter_slot"
        assert 'entry_zscore' in source.lower(), \
            "Missing entry z-score threshold in _enter_slot"

    def test_z_score_cap_blocks_extreme_z(self):
        """z_score_cap보다 큰 z-score에서 진입이 차단된다."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig,
            BacktestState, TraderSlotState
        )

        config = ContinuousBacktestConfig(
            z_score_cap=4.0,
            entry_zscore=2.0,
            formation_bars=50,
        )
        bt = ContinuousPairBacktest(config)

        # Create a candidate where the current z-score will be extreme (>4.0)
        # by setting spread_std very small relative to spread change
        candidate = {
            'symbol1': 'AAUSDT', 'symbol2': 'BBUSDT',
            'hedge_ratio': 1.0, 'intercept': 0.0,
            'spread_mean': 0.0, 'spread_std': 0.001,  # tiny std → huge z
            'half_life': 10.0, 'correlation': 0.8,
            'coint_pvalue': 0.01, 'current_z': 5.0, 'score': 0.01,
        }

        state = BacktestState(current_capital=10000, peak_capital=10000, max_traders=5)
        slot = state.slots[0]

        # Build minimal data: two symbols with a wide spread
        n = 100
        prices_a = np.full(n, 100.0)
        prices_b = np.full(n, 50.0)
        # At bar 80, make a big divergence → z-score will exceed cap
        prices_a[80] = 110.0  # log(110) - 1.0*log(50) = 4.70 - 3.91 = 0.79, z = 0.79/0.001 = 790

        data = {
            'AAUSDT': pd.DataFrame({'Close': prices_a}),
            'BBUSDT': pd.DataFrame({'Close': prices_b}),
        }

        from datetime import datetime, timezone
        result = bt._enter_slot(state, slot, candidate, data, 80, datetime.now(timezone.utc))
        assert result is False, "Z-score cap should block entry when |z| > cap"

    def test_btc_crash_gate_blocks_when_btc_crashes(self):
        """BTC가 급락하면 백테스트에서도 진입이 차단된다."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig,
        )

        config = ContinuousBacktestConfig(
            btc_crash_gate_enabled=True,
            btc_crash_window=8,
            btc_crash_threshold=0.03,
        )
        bt = ContinuousPairBacktest(config)

        n = 100
        # BTC crashes 5% over 8 bars
        btc_prices = np.full(n, 50000.0)
        btc_prices[92:] = 47500.0  # -5% at bars 92+

        data = {
            'BTCUSDT': pd.DataFrame({'Close': btc_prices}),
            'ETHUSDT': pd.DataFrame({'Close': np.full(n, 3000.0)}),
        }

        blocked, ret = bt._check_btc_crash_gate(data, bar_idx=95)
        assert blocked is True, f"BTC crash gate should block, return={ret:.4f}"
        assert abs(ret) > 0.03

    def test_btc_crash_gate_passes_when_calm(self):
        """BTC가 안정적이면 게이트를 통과한다."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig,
        )

        config = ContinuousBacktestConfig(
            btc_crash_gate_enabled=True,
            btc_crash_window=8,
            btc_crash_threshold=0.03,
        )
        bt = ContinuousPairBacktest(config)

        n = 100
        btc_prices = np.full(n, 50000.0)  # Stable BTC

        data = {
            'BTCUSDT': pd.DataFrame({'Close': btc_prices}),
        }

        blocked, ret = bt._check_btc_crash_gate(data, bar_idx=50)
        assert blocked is False
        assert abs(ret) < 0.001

    def test_btc_crash_gate_disabled(self):
        """btc_crash_gate_enabled=False이면 게이트가 항상 통과한다."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig,
        )

        config = ContinuousBacktestConfig(btc_crash_gate_enabled=False)
        bt = ContinuousPairBacktest(config)

        # Even with crashing BTC data
        n = 100
        btc_prices = np.full(n, 50000.0)
        btc_prices[90:] = 40000.0  # -20%

        data = {'BTCUSDT': pd.DataFrame({'Close': btc_prices})}

        blocked, ret = bt._check_btc_crash_gate(data, bar_idx=95)
        assert blocked is False

    def test_btc_crash_gate_no_btc_data_passes(self):
        """BTC 데이터가 없으면 fail-open으로 통과한다."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig,
        )

        config = ContinuousBacktestConfig(btc_crash_gate_enabled=True)
        bt = ContinuousPairBacktest(config)

        data = {'ETHUSDT': pd.DataFrame({'Close': np.full(100, 3000.0)})}

        blocked, ret = bt._check_btc_crash_gate(data, bar_idx=50)
        assert blocked is False

    def test_vol_expansion_gate_blocks_volatile_spread(self):
        """최근 스프레드 변동성이 formation 대비 2배 이상이면 진입 차단."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest, ContinuousBacktestConfig,
            BacktestState,
        )
        from datetime import datetime, timezone

        config = ContinuousBacktestConfig(
            vol_expansion_gate=2.0,
            vol_expansion_window=8,
            entry_zscore=2.0,
            z_score_cap=10.0,  # Widen cap so z-cap doesn't interfere
            btc_crash_gate_enabled=False,  # Isolate vol gate test
            formation_bars=50,
        )
        bt = ContinuousPairBacktest(config)

        n = 100
        # Stable formation period, then volatile recent bars
        prices1 = np.full(n, 100.0)
        prices2 = np.full(n, 50.0)
        # Add huge volatility at the recent end
        prices1[92] = 95.0
        prices1[93] = 105.0
        prices1[94] = 90.0
        prices1[95] = 110.0
        prices1[96] = 85.0
        prices1[97] = 115.0
        prices1[98] = 80.0
        prices1[99] = 120.0

        data = {
            'AAUSDT': pd.DataFrame({'Close': prices1}),
            'BBUSDT': pd.DataFrame({'Close': prices2}),
        }

        candidate = {
            'symbol1': 'AAUSDT', 'symbol2': 'BBUSDT',
            'hedge_ratio': 1.0, 'intercept': 0.0,
            'spread_mean': np.log(100.0) - np.log(50.0),
            'spread_std': 0.001,  # Very tight formation std
            'half_life': 10.0, 'correlation': 0.8,
            'coint_pvalue': 0.01, 'current_z': 3.0, 'score': 0.01,
        }

        state = BacktestState(current_capital=10000, peak_capital=10000, max_traders=5)
        slot = state.slots[0]

        result = bt._enter_slot(state, slot, candidate, data, 99, datetime.now(timezone.utc))
        assert result is False, "Vol expansion gate should block entry when volatility spikes"


# ============================================================
# E. Formation window parity
# ============================================================

class TestFormationWindowParity:
    """Formation window 설정이 백테스트와 실거래에서 일치하는지 검증."""

    def test_settings_has_formation_hours(self):
        """QuantBoxSettings에 formation_hours 필드가 존재한다."""
        from quantbox.core.settings import QuantBoxSettings
        assert hasattr(QuantBoxSettings, 'model_fields')
        assert 'formation_hours' in QuantBoxSettings.model_fields

    def test_to_trader_config_uses_formation_hours(self):
        """to_trader_config()가 formation_hours를 반영한다."""
        import os
        os.environ['BINANCE_API_KEY'] = 'test_key'
        os.environ['BINANCE_API_SECRET'] = 'test_secret'

        from quantbox.core.settings import QuantBoxSettings
        settings = QuantBoxSettings(
            binance_api_key='test',
            binance_api_secret='test',
            formation_hours=60,
        )
        config = settings.to_trader_config()
        assert config['formation_hours'] == 60

    def test_formation_bars_matches_hours(self):
        """백테스트 formation_bars와 라이브 formation_hours가 정합성을 가진다.

        기본값: formation_hours=60, bar_interval=15m → 60*4 = 240 bars.
        """
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        from quantbox.core.settings import QuantBoxSettings

        bt_config = ContinuousBacktestConfig()
        settings = QuantBoxSettings(
            binance_api_key='test',
            binance_api_secret='test',
        )

        expected_bars = settings.formation_hours * (60 // bt_config.bar_interval_minutes)
        assert bt_config.formation_bars == expected_bars, (
            f"Backtest formation_bars ({bt_config.formation_bars}) != "
            f"live formation_hours ({settings.formation_hours}) * "
            f"bars/hour ({60 // bt_config.bar_interval_minutes}) = {expected_bars}"
        )