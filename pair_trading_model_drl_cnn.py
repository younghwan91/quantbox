# pair_trading_model_drl_cnn.py

import numpy as np
import pandas as pd
import logging
import pickle
import matplotlib.pyplot as plt

# Import DRL libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv  # Added DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn

logger = logging.getLogger('PairTradingModel')

class PairTradingEnv(gym.Env):
    """
    Custom Environment for Pair Trading using Gymnasium
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, trading_data, window_split_index, window_size=10, trading_fee=0.0005):
        super(PairTradingEnv, self).__init__()

        self.data = trading_data.reset_index(drop=True)
        self.trading_fee = trading_fee
        self.window_size = window_size
        self.window_split_index = window_split_index

        # Define action space: 0 = Hold, 1 = Buy (Long), 2 = Sell (Short), 3 = Close Position
        self.action_space = spaces.Discrete(4)

        # Define observation space: Moving window of spread values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 1), dtype=np.float32
        )

        # Initialize environment variables
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        """
        super().reset(seed=seed)
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = self.window_split_index  # Start after the initial window
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.total_profit = 0.0

        # Return initial observation and info
        return self._next_observation(), {}

    def _next_observation(self):
        """
        Get the next observation from the trading data using a sliding window
        """
        window_start = self.current_step - self.window_size
        window_end = self.current_step
        
        spread_window = self.data['spread'].iloc[window_start:window_end].values
        obs = spread_window.reshape(-1, 1).astype(np.float32)
        return obs

    def step(self, action):
        """
        Execute one time step within the environment
        """
        reward = 0.0
        terminated = False
        truncated = False

        # Get current spread
        spread = self.data['spread'].iloc[self.current_step]

        # Implement action logic
        if action == 0:
            # Hold
            pass
        elif action == 1:
            # Buy (Long)
            if self.position == 0:
                self.position = 1
                self.entry_price = spread
            elif self.position == -1:
                # Close short position and open long
                profit = self.entry_price - spread - (2 * self.trading_fee)
                reward += profit
                self.total_profit += profit
                self.position = 1
                self.entry_price = spread
        elif action == 2:
            # Sell (Short)
            if self.position == 0:
                self.position = -1
                self.entry_price = spread
            elif self.position == 1:
                # Close long position and open short
                profit = spread - self.entry_price - (2 * self.trading_fee)
                reward += profit
                self.total_profit += profit
                self.position = -1
                self.entry_price = spread
        elif action == 3:
            # Close Position
            if self.position == 1:
                profit = spread - self.entry_price - (2 * self.trading_fee)
                reward += profit
                self.total_profit += profit
            elif self.position == -1:
                profit = self.entry_price - spread - (2 * self.trading_fee)
                reward += profit
                self.total_profit += profit
            self.position = 0
            self.entry_price = 0.0

            # 포지션이 종료되었으므로 에피소드 종료
            terminated = True

        # Move to the next step
        self.current_step += 1

        # Check if the end of data is reached
        if self.current_step >= len(self.data):
            terminated = True
            # Close any open positions
            if self.position != 0:
                if self.position == 1:
                    profit = spread - self.entry_price - (2 * self.trading_fee)
                elif self.position == -1:
                    profit = self.entry_price - spread - (2 * self.trading_fee)
                reward += profit
                self.total_profit += profit
                self.position = 0
                self.entry_price = 0.0

        # Get next observation
        if not terminated and not truncated:
            obs = self._next_observation()
        else:
            obs = np.zeros((self.window_size, 1), dtype=np.float32)

        info = {}
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


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom 1D CNN feature extractor for Stable Baselines3
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[1]  # Should be 1

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the size of the output of the CNN to define the linear layer input size
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, observation_space.shape[0])
            cnn_output_size = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, window_size, 1)
        # Need to transpose to (batch_size, channels, window_size)
        x = observations.transpose(1, 2)
        x = self.cnn(x)
        x = self.linear(x)
        return x

# Top-level environment creation function
def make_pair_trading_env(trading_data, window_split_index, window_size, trading_fee, seed):
    """
    Top-level function to create a PairTradingEnv instance.
    This function is picklable and can be used with SubprocVecEnv.
    """
    def _init():
        env = PairTradingEnv(trading_data, window_split_index, window_size, trading_fee)
        env.seed(seed)
        return env
    return _init

class PairTradingModelDRL:
    def __init__(self, config):
        self.timeframe = config['TIMEFRAME']
        self.days = config['DAYS']
        self.trading_fee = config.get('TRADING_FEE', 0.0005)
        self.window_size = config.get('WINDOW_SIZE', 10)
        self.model = None  # DRL model
        self.episodes = []  # List to store each episode's trading window data

    def load_precomputed_values(self, filepath):
        """
        Load precomputed values from a pickle file and prepare training and testing data.
        Each record is split into a formation window and a trading window based on the window ratio.
        Observations in the trading window use a sliding window approach.
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
                window_split_index = record['split_index']  # Use local variable

                # Create DataFrame for the trading window
                df = pd.DataFrame({'spread': spread})

                # Store a tuple of (trading_data, window_split_index)
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
        Train the DRL agent using the training data.
        Each training episode corresponds to a separate trading window.
        """
        if not hasattr(self, 'training_episodes'):
            logger.error("Training data not loaded.")
            return

        batch_size = 2  # Number of parallel environments
        total_episodes = len(self.training_episodes)
        batches = [self.training_episodes[i:i + batch_size] for i in range(0, total_episodes, batch_size)]

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=64),
        )

        # Initialize the DRL agent (PPO algorithm) with the custom CNN policy using the first batch
        initial_env_fns = [
            make_pair_trading_env(
                trading_data=episode[0],
                window_split_index=episode[1],
                window_size=self.window_size,
                trading_fee=self.trading_fee,
                seed=i
            ) for i, episode in enumerate(batches[0])
        ]
        self.model = PPO('CnnPolicy', SubprocVecEnv(initial_env_fns),
                        policy_kwargs=policy_kwargs, verbose=1)

        for epoch, batch in enumerate(batches):
            logger.info(f"Training on batch {epoch + 1}/{len(batches)}")
            env_fns = [
                make_pair_trading_env(
                    trading_data=episode[0],
                    window_split_index=episode[1],
                    window_size=self.window_size,
                    trading_fee=self.trading_fee,
                    seed=i
                ) for i, episode in enumerate(batch)
            ]
            env = SubprocVecEnv(env_fns)
            self.model.set_env(env)
            self.model.learn(total_timesteps=total_timesteps // len(batches))
            env.close()

        logger.info("Training completed.")

    def test_agent(self):
        """
        Test the trained agent on the testing data.
        Each testing episode corresponds to a separate trading window.
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
            env = PairTradingEnv(episode[0], episode[1], self.window_size, self.trading_fee)
            obs, _ = env.reset()
            episode_reward = 0.0

            while True:
                # Get action from the trained model
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            total_reward += episode_reward
            total_profit += env.total_profit
            logger.info(f"Episode {idx + 1}: Reward = {episode_reward:.2f}, Profit = {env.total_profit:.2f}")

        logger.info(f"Total reward on testing data: {total_reward}")
        logger.info(f"Total profit on testing data: {total_profit}")

    def simulate_trading(self):
        """
        Simulate trading using the trained agent on the testing data and collect statistics.
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
            env = PairTradingEnv(episode[0], episode[1], self.window_size, self.trading_fee)
            obs, _ = env.reset()
            total_reward = 0.0
            positions = []
            profits = []
            steps = []

            while True:
                # Get action from the trained model
                action, _states = self.model.predict(obs, deterministic=True)
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

        # Plot positions and profits over time
        plt.figure(figsize=(14, 7))
        plt.plot(all_steps, all_positions, label='Position')
        plt.title('Positions Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Position')
        plt.legend()
        plt.show()

        plt.figure(figsize=(14, 7))
        plt.plot(all_steps, all_profits, label='Total Profit')
        plt.title('Total Profit Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Profit')
        plt.legend()
        plt.show()

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
