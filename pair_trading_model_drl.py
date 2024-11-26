# pair_trading_model_drl_mlp.py

import numpy as np
import pandas as pd
import logging
import pickle
import matplotlib.pyplot as plt

# Import DRL libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor 

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PairTradingModel')


class PairTradingEnv(gym.Env):
    """
    Custom Environment for Pair Trading using Gymnasium
    Modified to use MLP-based DRL with single scalar inputs per step.
    Each episode terminates when a position is closed.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, trading_data, window_split_index, trading_fee=0.0005):
        super(PairTradingEnv, self).__init__()

        self.data = trading_data.reset_index(drop=True)
        self.trading_fee = trading_fee
        self.window_split_index = window_split_index

        # Define action space: 0 = Hold, 1 = Buy (Long), 2 = Sell (Short), 3 = Close Position
        self.action_space = spaces.Discrete(4)

        # Define observation space: Single scalar values for each feature plus position
        # Features: 'spread', 'z_score_spread', 'vol1', 'vol2', 'rsi1', 'rsi2'
        # Total observation size: 6 features + 1 position = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.window_split_index  # Start after the initial split
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.total_profit = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnls = []

        return self._next_observation(), {}

    def _next_observation(self):
        """
        Get the next observation from the trading data using current step's values
        """
        # Extract current step's features
        spread = self.data['spread'].iloc[self.current_step]
        z_score_spread = self.data['z_score_spread'].iloc[self.current_step]
        vol1 = self.data['vol1'].iloc[self.current_step]
        vol2 = self.data['vol2'].iloc[self.current_step]
        rsi1 = self.data['rsi1'].iloc[self.current_step]
        rsi2 = self.data['rsi2'].iloc[self.current_step]

        # Create observation array
        obs = np.array([
            spread,
            z_score_spread,
            vol1,
            vol2,
            rsi1,
            rsi2,
            self.position
        ], dtype=np.float32)

        return obs

    def update_data(self, trading_data, window_split_index):
        """
        Update the environment with new trading data and reset
        """
        self.data = trading_data.reset_index(drop=True)
        self.window_split_index = window_split_index
        self.reset()

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        terminated = False
        truncated = False
        info = {}

        # Get the current spread value
        spread = self.data['spread'].iloc[self.current_step]

        # Initialize reward
        reward = 0.0

        # Process the action
        if action == 1 and self.position == 0:
            # Open long position
            self.position = 1
            self.entry_price = spread
            info['position_change'] = 'long_open'
        elif action == 2 and self.position == 0:
            # Open short position
            self.position = -1
            self.entry_price = spread
            info['position_change'] = 'short_open'
        elif action == 3 and self.position != 0:
            # Close position
            profit = (spread - self.entry_price) * self.position - self.trading_fee
            self.total_profit += profit
            self.position = 0
            self.entry_price = 0.0
            info['position_change'] = 'position_closed'
            reward += profit

            # Terminate the episode since the position is closed
            terminated = True

        # Update unrealized P&L
        if self.position != 0:
            self.unrealized_pnl = (spread - self.entry_price) * self.position
        else:
            self.unrealized_pnl = 0.0

        # Record unrealized P&L for risk calculation
        self.unrealized_pnls.append(self.unrealized_pnl)

        # Calculate risk (standard deviation of unrealized P&L)
        if len(self.unrealized_pnls) > 1:
            risk = np.std(self.unrealized_pnls)
        else:
            risk = 0.0

        # Reward is unrealized profit minus risk penalty
        risk_aversion_coefficient = 0.1  # Adjust as needed
        reward += self.unrealized_pnl - risk_aversion_coefficient * risk

        # Move to the next time step
        self.current_step += 1

        # Check if we've reached the end of the data
        if self.current_step >= len(self.data):
            terminated = True
            # Close any open positions at the end
            if self.position != 0:
                profit = (spread - self.entry_price) * self.position - self.trading_fee
                self.total_profit += profit
                self.position = 0
                self.entry_price = 0.0
                reward += profit
            info['total_profit'] = self.total_profit

        # Get the next observation
        if not terminated and not truncated:
            obs = self._next_observation()
        else:
            obs = np.array([], dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment to the screen (optional)
        """
        profit = self.total_profit
        print(f'Step: {self.current_step}, Position: {self.position}, Total Profit: {profit}')

    def seed(self, seed=None):
        """
        Set the seed for the environment's random number generator.
        """
        np.random.seed(seed)


class CustomMLP(BaseFeaturesExtractor):
    """
    Custom MLP Feature Extractor for MLP-based DRL
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        # Observation space is expected to be (7,) as defined in PairTradingEnv
        super(CustomMLP, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class TrainingMonitorCallback(BaseCallback):
    """
    Callback for monitoring training progress.
    """
    def __init__(self, verbose=0):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_profits = []

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        if done:
            episode_reward = sum(self.locals['rewards'])
            total_profit = self.locals['infos'][0].get('total_profit', 0.0)
            self.episode_rewards.append(episode_reward)
            self.episode_profits.append(total_profit)
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Reward = {episode_reward:.5f}, "
                      f"Profit = {total_profit:.5f}")
        return True


class PairTradingModelDRL:
    """
    Class encapsulating the Pair Trading DRL model using PPO.
    """
    def __init__(self, config):
        self.timeframe = config['TIMEFRAME']
        self.days = config['DAYS']
        self.trading_fee = config.get('TRADING_FEE', 0.0005)
        self.window_size = 1  # Not used in MLP-based DRL
        self.model = None  # DRL model
        self.episodes = []  # List to store each episode's trading window data

    def load_precomputed_values(self, filepath):
        """
        Load precomputed trading data from a pickle file.
        """
        try:
            with open(filepath, 'rb') as f:
                all_values = pickle.load(f)
            logger.info(f"Loaded precomputed values from {filepath}")

            # Sort and ensure each record is treated as an episode
            all_values = sorted(all_values, key=lambda x: x['window_start'])
            logger.info(f"Total number of records (episodes): {len(all_values)}")

            # Process each record
            for idx, record in enumerate(all_values):
                spread = record['spread']
                vol1 = record['vol1']
                vol2 = record['vol2']
                window_split_index = record['split_index']
                rsi1 = record['rsi1']
                rsi2 = record['rsi2']

                zero_mean_spread = spread - np.mean(spread[:window_split_index])
                z_score_spread = zero_mean_spread / np.std(zero_mean_spread[:window_split_index])
                normalized_vol1 = (vol1 - np.mean(vol1[:window_split_index])) / np.std(vol1[:window_split_index])
                normalized_vol2 = (vol2 - np.mean(vol2[:window_split_index])) / np.std(vol2[:window_split_index])

                df = pd.DataFrame({
                    'spread': zero_mean_spread,
                    'z_score_spread': z_score_spread,
                    'vol1': normalized_vol1,
                    'vol2': normalized_vol2,
                    'rsi1': rsi1,
                    'rsi2': rsi2
                })

                self.episodes.append((df, window_split_index))
                logger.debug(f"Loaded episode {idx + 1} with {len(df)} trading steps")

            logger.info(f"Total number of trading episodes: {len(self.episodes)}")

            # Split episodes into training and testing sets
            data_split_index = int(len(self.episodes) * 0.8)  # 80% for training
            self.training_episodes = self.episodes[:data_split_index]
            self.testing_episodes = self.episodes[data_split_index:]
            logger.info(f"Split data into {len(self.training_episodes)} training and {len(self.testing_episodes)} testing episodes")

        except Exception as e:
            logger.error(f"Error loading precomputed values: {e}")  

    def train_agent(self, total_timesteps=100000):
        """
        Train the PPO agent on the training episodes.
        """
        if not hasattr(self, 'training_episodes'):
            logger.error("Training data not loaded.")
            return

        total_episodes = len(self.training_episodes)
        timesteps_per_episode = total_timesteps // total_episodes

        policy_kwargs = dict(
            features_extractor_class=CustomMLP,
            features_extractor_kwargs=dict(features_dim=64),
        )

        # Initialize the environment with the first episode and wrap it
        initial_episode = self.training_episodes[0]
        env = DummyVecEnv([lambda: PairTradingEnv(
            trading_data=initial_episode[0],
            window_split_index=initial_episode[1],
            trading_fee=self.trading_fee
        )])
        self.model = PPO('MlpPolicy',
                         env,
                         policy_kwargs=policy_kwargs,
                         verbose=1,)

        callback = TrainingMonitorCallback(verbose=1)

        # Loop over each episode sequentially
        for idx, episode in enumerate(self.training_episodes):
            logger.info(f"Training on episode {idx + 1}/{total_episodes}")

            # Update the environment with the new episode data
            env.envs[0].update_data(trading_data=episode[0], window_split_index=episode[1])

            # Train the model on the current episode
            self.model.set_env(env)
            self.model.learn(total_timesteps=timesteps_per_episode,
                             callback=callback,
                             reset_num_timesteps=False
                             )

        env.close()
        logger.info("Training completed.")

    def test_agent(self, confidence_threshold=0.8):
        """
        Test the trained agent on the testing data.
        Each testing episode corresponds to a separate trading window.
        Uses confidence threshold to decide whether to trade or hold.
        """
        if self.model is None:
            logger.error("Model is not trained yet.")
            return

        if not hasattr(self, 'testing_episodes'):
            logger.error("Testing data not loaded.")
            return

        total_reward = 0.0
        total_profit = 0.0

        for idx, episode in enumerate(self.testing_episodes):
            env = PairTradingEnv(episode[0], episode[1], trading_fee=self.trading_fee)
            obs, _ = env.reset()
            episode_reward = 0.0

            while True:
                # Get action with confidence check
                action = self._select_action_with_confidence(obs, confidence_threshold)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            total_reward += episode_reward
            total_profit += env.total_profit
            logger.info(f"Episode {idx + 1}: Reward = {episode_reward:.2f}, Profit = {env.total_profit:.2f}")

        logger.info(f"Total Test Reward: {total_reward:.2f}, Total Test Profit: {total_profit:.2f}")

    def simulate_trading(self, confidence_threshold=0.8):
        """
        Simulate trading using the trained agent on the testing data and collect statistics.
        Uses confidence threshold to decide whether to trade or hold.
        """
        if self.model is None:
            logger.error("Model is not trained yet.")
            return

        if not hasattr(self, 'testing_episodes'):
            logger.error("Testing data not loaded.")
            return

        # Initialize lists to collect simulation data
        all_positions = []
        all_profits = []
        all_steps = []

        for idx, episode in enumerate(self.testing_episodes):
            env = PairTradingEnv(episode[0], episode[1], trading_fee=self.trading_fee)
            obs, _ = env.reset()
            total_reward = 0.0
            positions = []
            profits = []
            steps = []

            while True:
                # Get action with confidence check
                action = self._select_action_with_confidence(obs, confidence_threshold)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Record positions and profits
                positions.append(env.position)
                profits.append(env.total_profit)
                steps.append(env.current_step)

                if terminated or truncated:
                    break

            all_positions.extend(positions)
            all_profits.extend(profits)
            all_steps.extend(steps)

            logger.info(f"Simulation for Episode {idx + 1}: Total Profit = {env.total_profit:.2f}")

        # Plot positions over time
        plt.figure(figsize=(14, 7))
        plt.plot(all_steps, all_positions, label='Position', color='blue')
        plt.title('Positions Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot profits over time
        plt.figure(figsize=(14, 7))
        plt.plot(all_steps, all_profits, label='Total Profit', color='green')
        plt.title('Total Profit Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Profit')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _select_action_with_confidence(self, obs, confidence_threshold=0.8):
        """
        Select action based on confidence. If the highest action probability is below
        the threshold, default to holding (action=0).
        """
        self.model.policy.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs.cpu().numpy()[0]
        
        max_prob = np.max(action_probs)
        if max_prob >= confidence_threshold:
            action = np.argmax(action_probs)
        else:
            action = 0  # Hold

        return action

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        """
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.error("No model to save.")

    def load_model(self, filepath):
        """
        Load a trained model from a file.
        """
        try:
            self.model = PPO.load(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
