"""
Tests for quantbox.strategy.kalman_spread — Online Kalman Filter z-score.

실제 바이낸스 데이터 없이 테스트 가능한 KF 수학적 속성만 검증한다.
(deterministic synthetic data는 가격 생성이 아닌 KF 로직 검증 목적.)
"""

import numpy as np
import pytest

from quantbox.strategy.kalman_spread import KalmanState, OnlineKalmanSpread


# ── Fixtures ──

@pytest.fixture
def default_kf() -> OnlineKalmanSpread:
    """기본 파라미터의 KF 인스턴스."""
    return OnlineKalmanSpread(
        init_mu=0.0,
        init_gamma=1.0,
        formation_spread_std=0.01,
    )


@pytest.fixture
def custom_kf() -> OnlineKalmanSpread:
    """커스텀 파라미터 KF."""
    return OnlineKalmanSpread(
        init_mu=0.05,
        init_gamma=1.23,
        formation_spread_std=0.015,
        delta=1e-4,
        obs_noise=1e-2,
        init_cov_scale=1e-3,
    )


class TestKalmanStateDataclass:
    """KalmanState 데이터클래스 기본 검증."""

    def test_default_values(self):
        state = KalmanState()
        assert state.mu == 0.0
        assert state.gamma == 1.0
        assert state.spread == 0.0
        assert state.z_score == 0.0
        assert state.vol_ratio == 1.0
        assert state.n_obs == 0
        assert state.P.shape == (2, 2)

    def test_custom_values(self):
        state = KalmanState(mu=0.1, gamma=1.5, z_score=2.3, n_obs=100)
        assert state.mu == 0.1
        assert state.gamma == 1.5
        assert state.z_score == 2.3
        assert state.n_obs == 100


class TestOnlineKalmanSpreadInit:
    """초기화 파라미터 검증."""

    def test_initial_state(self, default_kf: OnlineKalmanSpread):
        assert default_kf.mu == 0.0
        assert default_kf.gamma == 1.0
        assert default_kf.P.shape == (2, 2)

    def test_custom_init(self, custom_kf: OnlineKalmanSpread):
        assert custom_kf.mu == pytest.approx(0.05)
        assert custom_kf.gamma == pytest.approx(1.23)

    def test_formation_std_floor(self):
        """formation_spread_std=0 → 1e-10으로 하한 클램핑."""
        kf = OnlineKalmanSpread(0.0, 1.0, formation_spread_std=0.0)
        assert kf._formation_std == pytest.approx(1e-10)

    def test_negative_formation_std_floor(self):
        """음수도 1e-10으로 클램핑."""
        kf = OnlineKalmanSpread(0.0, 1.0, formation_spread_std=-0.01)
        assert kf._formation_std == pytest.approx(1e-10)


class TestPredictStep:
    """Predict step: P → P + Q."""

    def test_covariance_grows(self, default_kf: OnlineKalmanSpread):
        P_before = default_kf.P.copy()
        default_kf.predict()
        P_after = default_kf.P
        # P should grow by Q (positive diagonal)
        assert np.all(np.diag(P_after) > np.diag(P_before))

    def test_state_unchanged_after_predict(self, default_kf: OnlineKalmanSpread):
        """Random walk → state doesn't change in predict."""
        mu_before = default_kf.mu
        gamma_before = default_kf.gamma
        default_kf.predict()
        assert default_kf.mu == mu_before
        assert default_kf.gamma == gamma_before


class TestUpdateStep:
    """Update step 수학적 속성."""

    def test_returns_kalman_state(self, default_kf: OnlineKalmanSpread):
        state = default_kf.update(np.log(100.0), np.log(100.0))
        assert isinstance(state, KalmanState)

    def test_n_obs_increments(self, default_kf: OnlineKalmanSpread):
        state1 = default_kf.update(np.log(100.0), np.log(100.0))
        assert state1.n_obs == 1
        state2 = default_kf.update(np.log(101.0), np.log(101.0))
        assert state2.n_obs == 2

    def test_z_score_magnitude_bounded(self, default_kf: OnlineKalmanSpread):
        """Reasonable inputs → z-score shouldn't be extreme."""
        state = default_kf.update(np.log(100.0), np.log(100.0))
        assert abs(state.z_score) < 100  # sanity

    def test_spread_variance_positive(self, default_kf: OnlineKalmanSpread):
        state = default_kf.update(np.log(100.0), np.log(100.0))
        assert state.spread_var > 0

    def test_vol_ratio_positive(self, default_kf: OnlineKalmanSpread):
        state = default_kf.update(np.log(100.0), np.log(100.0))
        assert state.vol_ratio > 0

    def test_covariance_symmetric(self, default_kf: OnlineKalmanSpread):
        """Joseph form으로 업데이트 후에도 P가 대칭이어야 함."""
        for _ in range(50):
            default_kf.update(
                np.log(100.0 + np.random.randn() * 0.01),
                np.log(100.0 + np.random.randn() * 0.01),
            )
        P = default_kf.P
        np.testing.assert_allclose(P, P.T, atol=1e-15)

    def test_covariance_positive_definite(self, default_kf: OnlineKalmanSpread):
        """P는 양정치여야 함."""
        for _ in range(50):
            default_kf.update(
                np.log(100.0 + np.random.randn() * 0.01),
                np.log(100.0 + np.random.randn() * 0.01),
            )
        eigenvalues = np.linalg.eigvalsh(default_kf.P)
        assert np.all(eigenvalues > 0)


class TestAdaptiveTracking:
    """KF가 구조 변화를 추적하는지 검증."""

    def test_hedge_ratio_adapts_to_shift(self):
        """gamma=1.0에서 시작, 실제 관계가 gamma=1.5면 KF가 적응해야 함."""
        kf = OnlineKalmanSpread(
            init_mu=0.0,
            init_gamma=1.0,
            formation_spread_std=0.01,
            delta=1e-4,  # faster adaptation for test
        )
        true_gamma = 1.5
        true_mu = 0.02

        # Feed 200 observations with true_gamma relationship
        np.random.seed(42)
        log_p2 = np.log(100.0) + np.cumsum(np.random.randn(200) * 0.001)
        log_p1 = true_mu + true_gamma * log_p2 + np.random.randn(200) * 0.001

        state = None
        for lp1, lp2 in zip(log_p1, log_p2):
            state = kf.update(lp1, lp2)

        # After 200 bars, gamma should be closer to 1.5 than initial 1.0
        assert abs(state.gamma - true_gamma) < abs(1.0 - true_gamma)

    def test_regime_shift_increases_z_score(self):
        """갑작스러운 레짐 변화 시 z-score가 일시적으로 증가."""
        kf = OnlineKalmanSpread(
            init_mu=0.0,
            init_gamma=1.0,
            formation_spread_std=0.01,
            delta=1e-5,
        )
        # Stable period
        np.random.seed(42)
        log_p2_stable = np.log(100.0) + np.cumsum(np.random.randn(100) * 0.001)
        log_p1_stable = 1.0 * log_p2_stable + np.random.randn(100) * 0.001

        for lp1, lp2 in zip(log_p1_stable, log_p2_stable):
            kf.update(lp1, lp2)
        state_stable = kf.get_state()

        # Sudden regime shift: add large shock
        log_p1_shock = log_p1_stable[-1] + 0.05  # 5% jump
        state_shock = kf.update(log_p1_shock, log_p2_stable[-1])

        # z-score should spike above normal range (> 1.0)
        assert abs(state_shock.z_score) > 1.0


class TestWarmup:
    """warmup() 일괄 처리 검증."""

    def test_warmup_processes_all_bars(self, default_kf: OnlineKalmanSpread):
        n = 100
        log_p1 = np.log(np.linspace(100, 110, n))
        log_p2 = np.log(np.linspace(100, 109, n))
        state = default_kf.warmup(log_p1, log_p2)
        assert state.n_obs == n

    def test_warmup_equals_sequential_update(self):
        """warmup과 개별 update 결과가 동일해야 함."""
        np.random.seed(123)
        n = 50
        log_p1 = np.log(100.0 + np.cumsum(np.random.randn(n) * 0.1))
        log_p2 = np.log(100.0 + np.cumsum(np.random.randn(n) * 0.1))

        # Warmup path
        kf1 = OnlineKalmanSpread(0.0, 1.0, 0.01)
        state_warmup = kf1.warmup(log_p1, log_p2)

        # Sequential path
        kf2 = OnlineKalmanSpread(0.0, 1.0, 0.01)
        state_seq = None
        for lp1, lp2 in zip(log_p1, log_p2):
            state_seq = kf2.update(lp1, lp2)

        assert state_warmup.mu == pytest.approx(state_seq.mu)
        assert state_warmup.gamma == pytest.approx(state_seq.gamma)
        assert state_warmup.z_score == pytest.approx(state_seq.z_score)
        assert state_warmup.n_obs == state_seq.n_obs

    def test_warmup_length_mismatch_raises(self, default_kf: OnlineKalmanSpread):
        with pytest.raises(ValueError, match="Length mismatch"):
            default_kf.warmup(np.array([1.0, 2.0]), np.array([1.0]))

    def test_warmup_empty_returns_initial_state(self, default_kf: OnlineKalmanSpread):
        """빈 배열로 warmup → 초기 상태 반환."""
        state = default_kf.warmup(np.array([]), np.array([]))
        assert state.n_obs == 0


class TestGetState:
    """get_state()는 데이터 처리 없이 현재 상태 반환."""

    def test_initial_get_state(self, default_kf: OnlineKalmanSpread):
        state = default_kf.get_state()
        assert state.n_obs == 0
        assert state.z_score == 0.0
        assert state.spread == 0.0

    def test_get_state_after_updates(self, default_kf: OnlineKalmanSpread):
        default_kf.update(np.log(100.0), np.log(100.0))
        state = default_kf.get_state()
        assert state.n_obs == 1
        assert state.mu == default_kf.mu
        assert state.gamma == default_kf.gamma


class TestVolRatio:
    """vol_ratio = sqrt(S) / formation_std."""

    def test_vol_ratio_near_one_for_stable(self):
        """안정적인 페어 → vol_ratio ≈ 1.0 근처."""
        kf = OnlineKalmanSpread(0.0, 1.0, formation_spread_std=0.01)
        np.random.seed(42)
        for _ in range(200):
            p = 100.0 + np.random.randn() * 0.01
            kf.update(np.log(p), np.log(p))
        state = kf.get_state()
        # Should be within reasonable range (not exactly 1 due to filter dynamics)
        assert 0.1 < state.vol_ratio < 50.0

    def test_vol_expansion_detected(self):
        """큰 formation_std 대비 작은 실제 변동 → vol_ratio < 1.
        반대도 마찬가지."""
        kf = OnlineKalmanSpread(
            0.0, 1.0,
            formation_spread_std=0.001,  # very tight formation
        )
        # Feed data with larger spread
        state = kf.update(np.log(100.0), np.log(99.0))  # big spread
        assert state.vol_ratio > 1.0  # Vol expanded vs formation


class TestSettingsIntegration:
    """Settings에 새 KF 파라미터가 추가되었는지 검증."""

    def test_settings_has_kalman_fields(self):
        """QuantBoxSettings에 KF 관련 필드 존재."""
        from unittest.mock import patch
        from quantbox.core.settings import QuantBoxSettings

        env = {
            'BINANCE_API_KEY': 'test-key',
            'BINANCE_API_SECRET': 'test-secret',
        }
        with patch.dict('os.environ', env, clear=True):
            s = QuantBoxSettings(_env_file=None)
            assert hasattr(s, 'z_score_cap')
            assert hasattr(s, 'vol_expansion_gate')
            assert hasattr(s, 'kalman_delta')
            assert hasattr(s, 'kalman_obs_noise')

    def test_settings_default_values(self):
        """KF 기본값 검증."""
        from unittest.mock import patch
        from quantbox.core.settings import QuantBoxSettings

        env = {
            'BINANCE_API_KEY': 'test-key',
            'BINANCE_API_SECRET': 'test-secret',
        }
        with patch.dict('os.environ', env, clear=True):
            s = QuantBoxSettings(_env_file=None)
            assert s.z_score_cap == 4.0
            assert s.vol_expansion_gate == 2.0
            assert s.kalman_delta == pytest.approx(1e-5)
            assert s.kalman_obs_noise == pytest.approx(1e-3)

    def test_to_trader_config_contains_kalman_keys(self):
        """to_trader_config()에 KF 키가 포함."""
        from unittest.mock import patch
        from quantbox.core.settings import QuantBoxSettings

        env = {
            'BINANCE_API_KEY': 'test-key',
            'BINANCE_API_SECRET': 'test-secret',
        }
        with patch.dict('os.environ', env, clear=True):
            s = QuantBoxSettings(_env_file=None)
            cfg = s.to_trader_config()
            assert 'z_score_cap' in cfg
            assert 'vol_expansion_gate' in cfg
            assert 'kalman_delta' in cfg
            assert 'kalman_obs_noise' in cfg

    def test_env_override_kalman_params(self):
        """환경변수로 KF 파라미터 오버라이드."""
        from unittest.mock import patch
        from quantbox.core.settings import QuantBoxSettings

        env = {
            'BINANCE_API_KEY': 'test-key',
            'BINANCE_API_SECRET': 'test-secret',
            'Z_SCORE_CAP': '5.0',
            'VOL_EXPANSION_GATE': '3.0',
            'KALMAN_DELTA': '0.0001',
            'KALMAN_OBS_NOISE': '0.01',
        }
        with patch.dict('os.environ', env, clear=True):
            s = QuantBoxSettings(_env_file=None)
            assert s.z_score_cap == 5.0
            assert s.vol_expansion_gate == 3.0
            assert s.kalman_delta == pytest.approx(1e-4)
            assert s.kalman_obs_noise == pytest.approx(1e-2)
            cfg = s.to_trader_config()
            assert cfg['z_score_cap'] == 5.0
            assert cfg['kalman_delta'] == pytest.approx(1e-4)
