import numpy as np
from pykalman import KalmanFilter

class KalmanModel:
    def __init__(self, model_type, Y, split_index, dt=1.0):

        """
        Parameters:
            model_type (str): Specifies the model configuration. E.g.,
                              'mu_gamma', 'mu_dot_gamma', 'mu_gamma_dot',
                              'mu_gamma_ddot', 'mu_dot_gamma_ddot', 'mu_ddot_gamma_ddot'
            Y (np.ndarray): Data array (T x 2) with columns for observation and regressor.
            dt (float): Time step (default is 1).
        """
        self.model_type = model_type
        self.Y = Y
        self.dt = dt
        self.T = Y.shape[0]
        self.split_index = split_index
        # Set up the model matrices
        self._initialize_model()
        
    def _initialize_model(self):
        # Common initial OLS to estimate mu and gamma
        # X = np.column_stack([np.ones(self.T), self.Y[:, 1]])
        X = np.column_stack([np.ones(self.split_index), self.Y[:self.split_index, 1]])
        beta_est, _, _, _ = np.linalg.lstsq(X, self.Y[:self.split_index, 0], rcond=None)
        init_mu, init_gamma = beta_est[0], beta_est[1]
        
        # Define common observation noise covariance
        self.R = np.array([[1e3]]) # 5
        # Process noise scale and initial covariance (can be adjusted per model)
        noise_scale = 1e-6
        
        # Build model based on the model type
        if self.model_type == 'mu_gamma':
            # State vector: [mu, gamma]
            self.state_dim = 2
            self.init_state = np.array([init_mu, init_gamma])
            self.F = np.eye(2)  # Random walk
            self.Q = noise_scale * np.eye(2)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1]])
        elif self.model_type == 'mu_dot_gamma':
            # State vector: [mu, gamma, mu_dot]
            self.state_dim = 3
            self.init_state = np.array([init_mu, init_gamma, 0.0])
            self.F = np.array([
                [1, 0, self.dt],
                [0, 1,      0],
                [0, 0,      1]
            ])
            self.Q = noise_scale * np.eye(3)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1], np.zeros(self.T)])
        
        elif self.model_type == 'mu_ddot_gamma':
            # State vector: [mu, gamma, mu_dot, mu_ddot]
            self.state_dim = 4
            self.init_state = np.array([init_mu, init_gamma, 0.0, 0.0])
            self.F = np.array([
                [1, 0, self.dt, 0.5*self.dt*self.dt],
                [0, 1,      0,                0],
                [0, 0,      1,                self.dt],
                [0, 0,      0,                     1]
            ])
            self.Q = noise_scale * np.eye(4)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1],
                                       np.zeros(self.T), np.zeros(self.T)])
            
        elif self.model_type == 'mu_gamma_dot':
            # State vector: [mu, gamma, gamma_dot]
            self.state_dim = 3
            self.init_state = np.array([init_mu, init_gamma, 0.0])
            self.F = np.array([
                [1, 0,      0],
                [0, 1, self.dt],
                [0, 0,      1]
            ])
            self.Q = noise_scale * np.eye(3)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1], np.zeros(self.T)])

        elif self.model_type == 'mu_gamma_auto':
            # State vector: [mu, gamma, gamma_dot]
            self.state_dim = 3
            self.init_state = np.array([init_mu, init_gamma, 0.0])
            self.F = np.array([
                [1, 0,      0],
            [0, 1,         0],
                [0, 0,      0.1]
            ])
            self.Q = noise_scale * np.eye(3)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1], np.ones(self.T)])

        elif self.model_type == 'mu_gamma_ddot':
            # State vector: [mu, gamma, gamma_dot, gamma_ddot]
            self.state_dim = 4
            self.init_state = np.array([init_mu, init_gamma, 0.0, 0.0])
            self.F = np.array([
                [1, 0,      0,        0],
                [0, 1, self.dt, 0.5*self.dt*self.dt],
                [0, 0,      1,        self.dt],
                [0, 0,      0,             1]
            ])
            self.Q = noise_scale * np.eye(4)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1],
                                       np.zeros(self.T), np.zeros(self.T)])
            
        elif self.model_type == 'mu_dot_gamma_dot':
            # State vector: [mu, gamma, mu_dot, gamma_dot]
            self.state_dim = 4
            self.init_state = np.array([init_mu, init_gamma, 0.0, 0.0])
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            self.Q = noise_scale * np.eye(4)         
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1],
                                        np.zeros(self.T), np.zeros(self.T)])

        elif self.model_type == 'mu_dot_gamma_ddot':
            # State vector: [mu, gamma, mu_dot, gamma_dot, gamma_ddot]
            self.state_dim = 5
            self.init_state = np.array([init_mu, init_gamma, 0.0, 0.0, 0.0])
            self.F = np.array([
                [1, 0, self.dt,      0,         0],
                [0, 1,      0, self.dt, 0.5*self.dt*self.dt],
                [0, 0,      1,      0,         0],
                [0, 0,      0,      1,         self.dt],
                [0, 0,      0,      0,             1]
            ])
            self.Q = noise_scale * np.eye(5)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1],
                                       np.zeros(self.T), np.zeros(self.T),
                                       np.zeros(self.T)])
            
        elif self.model_type == 'mu_ddot_gamma_ddot':
            # State vector: [mu, gamma, mu_dot, mu_ddot, gamma_dot, gamma_ddot]
            self.state_dim = 6
            self.init_state = np.array([init_mu, init_gamma, 0.0, 0.0, 0.0, 0.0])
            self.F = np.array([
                [1, 0, self.dt, 0.5*self.dt*self.dt,    0,                0],
                [0, 1,      0,                0, self.dt, 0.5*self.dt*self.dt],
                [0, 0,      1,                self.dt,    0,                0],
                [0, 0,      0,                     1,    0,                0],
                [0, 0,      0,                     0,    1,          self.dt],
                [0, 0,      0,                     0,    0,                1]
            ])
            self.Q = noise_scale * np.eye(6)
            obs_mat = np.column_stack([np.ones(self.T), self.Y[:, 1],
                                       np.zeros(self.T), np.zeros(self.T),
                                       np.zeros(self.T), np.zeros(self.T)])
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Reshape observation matrix to match pykalman expectations (T x obs_dim x state_dim)
        self.H = obs_mat.reshape(self.T, 1, self.state_dim)
        # Initial state covariance: small uncertainty
        self.init_cov = noise_scale * np.eye(self.state_dim)
    
    def run_smoother(self):
        """Create the Kalman filter and run the smoother."""
        self.kf = KalmanFilter(
            transition_matrices=self.F,
            observation_matrices=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            initial_state_mean=self.init_state,
            initial_state_covariance=self.init_cov
        )
        # self.kf.em(self.Y[:, 0], n_iter=20)
        self.state_means, self.state_covs = self.kf.filter(self.Y[:, 0])
        return self.state_means
    
    def forecast_state(self, steps=5):
        """Forecast the state vector a given number of steps ahead."""
        if not hasattr(self, 'state_means'):
            raise RuntimeError("Run the smoother first before forecasting.")
        last_state = self.state_means[-1]
        F_power = np.linalg.matrix_power(self.F, steps)
        return F_power.dot(last_state)

    def forecast_state_with_covariance(self, steps=5):
        """Forecast the state vector and its covariance a given number of steps ahead."""
        if not hasattr(self, 'state_means') or not hasattr(self, 'state_covs'):
            raise RuntimeError("Run the smoother first before forecasting.")
        
        last_state = self.state_means[-1]
        last_cov = self.state_covs[-1]
        
        # Compute F^steps
        F_power = np.linalg.matrix_power(self.F, steps)
        forecast_state = F_power.dot(last_state)
        
        # Propagate the covariance: F^n * last_cov * (F^n)^T
        forecast_cov = F_power.dot(last_cov).dot(F_power.T)
        
        # Accumulate the process noise over the forecast horizon:
        # sum_{k=0}^{steps-1} F^k * Q * (F^k)^T
        noise_sum = np.zeros_like(self.F)
        F_power_k = np.eye(self.F.shape[0])
        for _ in range(steps):
            noise_sum += F_power_k.dot(self.Q).dot(F_power_k.T)
            F_power_k = self.F.dot(F_power_k)
        
        forecast_cov += noise_sum

        return forecast_state, forecast_cov