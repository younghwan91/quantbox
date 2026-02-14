"""Online Kalman Filter for adaptive spread tracking.

Provides real-time adaptive hedge ratio (gamma), intercept (mu), and
innovation-based z-score for pair trading.

Unlike the legacy batch KalmanModel, this runs **online** — one observation
at a time — making it suitable for live trading where we receive a new
15m bar every check cycle.

Key benefits over fixed (formation-period) parameters:
    1. Hedge ratio adapts to structural changes → catches regime shifts.
    2. Innovation z-score uses the filter's own uncertainty estimate (S)
       as the denominator → automatic vol-expansion adjustment.
    3. When the pair relationship breaks, innovation explodes briefly then
       the filter adapts → z-score drops toward 0 → triggers mean_reversion
       exit instead of waiting for stop_loss.

State vector: [mu, gamma]
    mu    = intercept of log-price relationship
    gamma = hedge ratio (slope)

Observation model:
    log(price1) = mu + gamma * log(price2) + noise

References:
    - Elliott, van der Hoek, Malcolm (2005) "Pairs Trading"
    - Montana et al. (2009) "Flexible least squares for temporal data mining"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Snapshot of the Kalman filter state at a given time step.

    Attributes:
        mu: Current intercept estimate.
        gamma: Current hedge ratio estimate.
        P: State covariance matrix (2x2).
        spread: Latest spread (innovation).
        spread_var: Innovation variance (S).
        z_score: Innovation z-score (spread / sqrt(S)).
        vol_ratio: sqrt(S) / formation_spread_std — measures vol expansion.
        n_obs: Number of observations processed so far.
    """
    mu: float = 0.0
    gamma: float = 1.0
    P: np.ndarray = field(default_factory=lambda: np.eye(2) * 1e-4)
    spread: float = 0.0
    spread_var: float = 1.0
    z_score: float = 0.0
    vol_ratio: float = 1.0
    n_obs: int = 0


class OnlineKalmanSpread:
    """Online Kalman Filter for adaptive pair spread tracking.

    Lightweight 2-state filter (mu, gamma) that processes one observation
    at a time. Designed for integration with LivePairTrader.monitor_slot().

    Usage::

        kf = OnlineKalmanSpread(
            init_mu=0.05,
            init_gamma=1.23,
            formation_spread_std=0.015,
        )

        # Each 15m bar:
        state = kf.update(log_price1, log_price2)
        print(state.z_score, state.vol_ratio)

    Args:
        init_mu: Initial intercept from formation period OLS.
        init_gamma: Initial hedge ratio from formation period OLS.
        formation_spread_std: Spread std from formation period (for vol_ratio).
        delta: Process noise scaling factor. Controls how fast the filter
            adapts. Larger = faster adaptation but noisier estimates.
            Default 1e-5 is conservative for 15m crypto pairs.
        obs_noise: Observation noise variance (R). Default 1e-3.
        init_cov_scale: Initial state covariance scaling. Default 1e-4.
    """

    def __init__(
        self,
        init_mu: float,
        init_gamma: float,
        formation_spread_std: float,
        *,
        delta: float = 1e-5,
        obs_noise: float = 1e-3,
        init_cov_scale: float = 1e-4,
    ):
        # State vector: [mu, gamma]
        self._x = np.array([init_mu, init_gamma], dtype=np.float64)
        # State covariance
        self._P = np.eye(2, dtype=np.float64) * init_cov_scale
        # Process noise (Q)
        self._Q = np.eye(2, dtype=np.float64) * delta
        # Observation noise (R) — scalar
        self._R = obs_noise
        # Formation spread std for vol_ratio calculation
        self._formation_std = max(formation_spread_std, 1e-10)
        # Observation count
        self._n_obs = 0
        # Transition matrix: random walk → F = I
        # (mu and gamma follow random walks)

    @property
    def mu(self) -> float:
        """Current intercept estimate."""
        return float(self._x[0])

    @property
    def gamma(self) -> float:
        """Current hedge ratio estimate."""
        return float(self._x[1])

    @property
    def P(self) -> np.ndarray:
        """Current state covariance (2x2)."""
        return self._P.copy()

    def predict(self) -> None:
        """Prediction step: propagate state and covariance.

        Since F = I (random walk), state doesn't change.
        Covariance grows by Q.
        """
        # x_pred = F @ x = x (random walk)
        # P_pred = F @ P @ F.T + Q = P + Q
        self._P = self._P + self._Q

    def update(self, log_price1: float, log_price2: float) -> KalmanState:
        """Process one observation and return the updated state.

        This performs predict + update in one call for simplicity.

        Args:
            log_price1: log(price) of the first asset.
            log_price2: log(price) of the second asset.

        Returns:
            KalmanState with updated estimates and innovation z-score.
        """
        # --- Predict ---
        self.predict()

        # --- Update ---
        # Observation model: y = H @ x + noise
        # y = log_price1, H = [1, log_price2]
        y = log_price1
        H = np.array([1.0, log_price2], dtype=np.float64)

        # Innovation (prediction error)
        y_hat = H @ self._x
        innovation = y - y_hat  # = spread (adaptive)

        # Innovation covariance: S = H @ P @ H.T + R
        S = H @ self._P @ H + self._R

        # Kalman gain: K = P @ H.T / S
        K = (self._P @ H) / S

        # State update: x = x + K * innovation
        self._x = self._x + K * innovation

        # Covariance update: P = (I - K @ H) @ P
        # Joseph form for numerical stability:
        # P = (I - K H) P (I - K H)^T + K R K^T
        I_KH = np.eye(2) - np.outer(K, H)
        self._P = I_KH @ self._P @ I_KH.T + np.outer(K, K) * self._R

        self._n_obs += 1

        # Compute z-score and vol_ratio
        S_safe = max(S, 1e-20)
        sqrt_S = np.sqrt(S_safe)
        z_score = innovation / sqrt_S
        vol_ratio = sqrt_S / self._formation_std

        return KalmanState(
            mu=float(self._x[0]),
            gamma=float(self._x[1]),
            P=self._P.copy(),
            spread=float(innovation),
            spread_var=float(S),
            z_score=float(z_score),
            vol_ratio=float(vol_ratio),
            n_obs=self._n_obs,
        )

    def get_state(self) -> KalmanState:
        """Return current state without processing new data."""
        S = self._P[0, 0] + self._R  # approximate
        sqrt_S = np.sqrt(max(S, 1e-20))
        return KalmanState(
            mu=float(self._x[0]),
            gamma=float(self._x[1]),
            P=self._P.copy(),
            spread=0.0,
            spread_var=float(S),
            z_score=0.0,
            vol_ratio=sqrt_S / self._formation_std,
            n_obs=self._n_obs,
        )

    def warmup(
        self,
        log_prices1: np.ndarray,
        log_prices2: np.ndarray,
    ) -> KalmanState:
        """Run the filter over historical data to warm up state.

        Call this once at position entry with formation-period data
        so the filter has converged estimates before live updates begin.

        Args:
            log_prices1: Array of log(price1) values (chronological).
            log_prices2: Array of log(price2) values (chronological).

        Returns:
            Final KalmanState after processing all warmup data.
        """
        if len(log_prices1) != len(log_prices2):
            raise ValueError(
                f"Length mismatch: {len(log_prices1)} vs {len(log_prices2)}"
            )

        state = None
        for lp1, lp2 in zip(log_prices1, log_prices2):
            state = self.update(lp1, lp2)

        if state is None:
            return self.get_state()

        logger.debug(
            f"KF warmup complete: {self._n_obs} bars, "
            f"mu={self.mu:.6f}, gamma={self.gamma:.4f}"
        )
        return state
