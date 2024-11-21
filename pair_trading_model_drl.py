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
from stable_baselines3.common.callbacks import BaseCallback

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

        # Define observation space: Moving window of spread, vol1, and vol2 values
        # Updated to have 3 channels instead of 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size * 3 + 2,), dtype=np.float32
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
        
        # Extract spread, vol1, and vol2
        spread_window = self.data['spread'].iloc[window_start:window_end].values
        vol1_window = self.data['vol1'].iloc[window_start:window_end].values
        vol2_window = self.data['vol2'].iloc[window_start:window_end].values
        
        # Stack the features to create a multi-channel observation
        obs = np.stack([spread_window, vol1_window, vol2_window], axis=1).astype(np.float32)
        obs = obs.flatten()  # Flatten to a 1D array
        # Calculate unrealized profit
        spread = self.data['spread'].iloc[self.current_step - 1]
        if self.position == 1:
            unrealized_profit = spread - self.entry_price
        elif self.position == -1:
            unrealized_profit = self.entry_price - spread
        else:
            unrealized_profit = 0.0
        # Append current position
        obs = np.concatenate([obs, [self.position, unrealized_profit]], axis=0)
        return obs

    def update_data(self, trading_data, window_split_index):
        self.data = trading_data.reset_index(drop=True)
        self.window_split_index = window_split_index
        self.reset()

    def step(self, action):
        """
        Execute one time step within the environment
        """
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Get current spread
        spread = self.data['spread'].iloc[self.current_step]
        
        time_penalty = -0.00001 if self.position != 0 else 0.0
        reward += time_penalty    

        # Calculate Unrealized P&L before the action
        if self.position == 1:
            unrealized_pnl_before = spread - self.entry_price
        elif self.position == -1:
            unrealized_pnl_before = self.entry_price - spread
        else:
            unrealized_pnl_before = 0.0

        # Implement action logic
        if action == 0:
            # Hold position
            pass  # No action needed
        elif action == 1:
            # Buy (Long)
            if self.position == 0:
                # Open long position
                self.position = 1
                self.entry_price = spread
                self.total_profit -= self.trading_fee  # Pay trading fee
                info['position_change'] = 'long_open'
            elif self.position == -1:
                pass
        elif action == 2:
            # Sell (Short)
            if self.position == 0:
                # Open short position
                self.position = -1
                self.entry_price = spread
                self.total_profit -= self.trading_fee  # Pay trading fee
                info['position_change'] = 'short_open'
            elif self.position == 1:
                pass
        elif action == 3:
            # Close Position
            if self.position != 0:
                # Calculate realized P&L
                if self.position == 1:
                    profit = spread - self.entry_price - self.trading_fee
                    info['position_change'] = 'long_close'
                elif self.position == -1:
                    profit = self.entry_price - spread - self.trading_fee
                    info['position_change'] = 'short_close'
                self.total_profit += profit
                reward += profit  # Reward is the realized profit
                self.position = 0
                self.entry_price = 0.0
            else:
                pass

        # Calculate Unrealized P&L after the action
        if self.position == 1:
            unrealized_pnl_after = spread - self.entry_price
        elif self.position == -1:
            unrealized_pnl_after = self.entry_price - spread
        else:
            unrealized_pnl_after = 0.0

        # Calculate change in unrealized P&L
        delta_unrealized_pnl = unrealized_pnl_after - unrealized_pnl_before

        # Total reward is realized P&L plus change in unrealized P&L
        reward += delta_unrealized_pnl

        # Move to the next step
        self.current_step += 1

        # Check if the end of data is reached
        if self.current_step >= len(self.data):
            terminated = True
            # Close any open positions
            if self.position != 0:
                # Calculate final realized P&L
                if self.position == 1:
                    profit = spread - self.entry_price - self.trading_fee
                elif self.position == -1:
                    profit = self.entry_price - spread - self.trading_fee
                self.total_profit += profit
                reward += profit  # Add final realized profit to reward
                self.position = 0
                self.entry_price = 0.0
            info['total_profit'] = self.total_profit
       
        # Get next observation
        if not terminated and not truncated:
            obs = self._next_observation()
        else:
            obs = []

        
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
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.window_size = (observation_space.shape[0] - 2) // 3  # Adjust for position
        n_input_channels = 3  # spread, vol1, vol2

        # CNN layers for time-series data
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output size
        with torch.no_grad():
            sample_timeseries = torch.zeros(1, n_input_channels, self.window_size)
            cnn_output_size = self.cnn(sample_timeseries).shape[1]

        # Linear layer to combine CNN output and position
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size + 2, features_dim),  # +1 for position scalar
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, window_size * 3 + 1)
        # Separate time-series data and position
        timeseries = observations[:, :-2]
        scalars = observations[:, -2:]

        # Reshape timeseries to (batch_size, channels, window_size)
        timeseries = timeseries.view(observations.shape[0], self.window_size, 3)
        timeseries = timeseries.permute(0, 2, 1)  # (batch_size, channels, window_size)

        # Process timeseries through CNN
        x = self.cnn(timeseries)

        # Concatenate CNN output with position
        x = torch.cat((x, scalars), dim=1)

        # Process through linear layers
        x = self.linear(x)
        return x


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_profits = []
        self.rewards = []
        self.actions = []
        self.positions = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        action = self.locals['actions'][0]

        self.rewards.append(reward)
        self.actions.append(action)

        # Access the current position from the environment
        position = self.training_env.envs[0].position
        self.positions.append(position)

        if done:
            total_reward = sum(self.rewards)
            total_profit = self.locals['infos'][0].get('total_profit', 0.0)
            self.episode_rewards.append(total_reward)
            self.episode_profits.append(total_profit)

            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: Reward = {total_reward:.5f}, Profit = {total_profit:.5f}")
                # print(f"Actions taken: {self.actions}")
                # print(f"Positions: {self.positions}")

            # Reset for next episode
            self.rewards = []
            self.actions = []
            self.positions = []

        return True

    
class PairTradingModelDRL:
    def __init__(self, config):
        self.timeframe = config['TIMEFRAME']
        self.days = config['DAYS']
        self.trading_fee = config.get('TRADING_FEE', 0.0005)
        self.window_size = config['WINDOW_SIZE']
        self.model = None  # DRL model
        self.episodes = []  # List to store each episode's trading window data

    def load_precomputed_values(self, filepath):

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
                vol1 = record['vol1']  # Ensure vol1 is present
                vol2 = record['vol2']  # Ensure vol2 is present
                window_split_index = record['split_index']  # Use local variable
                
                zero_mean_spread = spread - np.mean(spread[:window_split_index])
                normalized_vol1 = (vol1 - np.mean(vol1[:window_split_index])) / np.std(vol1[:window_split_index])
                normalized_vol2 = (vol2 - np.mean(vol2[:window_split_index])) / np.std(vol2[:window_split_index])
                # Create DataFrame for the trading window
                df = pd.DataFrame({
                    'spread': zero_mean_spread,
                    'vol1': normalized_vol1,
                    'vol2': normalized_vol2,
                })

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

        if not hasattr(self, 'training_episodes'):
            logger.error("Training data not loaded.")
            return

        # Limit training episodes for faster training
        # self.training_episodes = self.training_episodes[:100]
        total_episodes = len(self.training_episodes)
        timesteps_per_episode = total_timesteps // total_episodes

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=64),
        )

        # Initialize the environment with the first episode and wrap it
        initial_episode = self.training_episodes[0]
        env = DummyVecEnv([lambda: PairTradingEnv(
            trading_data=initial_episode[0],
            window_split_index=initial_episode[1],
            window_size=self.window_size,
            trading_fee=self.trading_fee
        )])
        self.model = PPO('CnnPolicy', 
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

            spreads = []
            positions = []
            actions = []
            steps = []
            position_changes = []

            while True:
                # Get action from the trained model
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                current_step = env.current_step - 1
                current_spread = env.data['spread'].iloc[current_step]
                spreads.append(current_spread)
                positions.append(env.position)
                actions.append(action)
                steps.append(current_step)

                if 'position_change' in info:
                    position_changes.append({
                        'step': current_step,
                        'type': info['position_change'],
                        'price': current_spread
                    })

                if terminated or truncated:
                    break

            total_reward += episode_reward
            total_profit += env.total_profit
            logger.info(f"Episode {idx + 1}: Reward = {episode_reward:.2f}, Profit = {env.total_profit:.2f}")

            # Plotting the spread and positions
            plt.figure(figsize=(14, 7))
            plt.plot(steps, spreads, label='Spread')

            markers = {
                'long_open': {'color': 'green', 'marker': '^', 'label': 'Long Open'},
                'short_open': {'color': 'red', 'marker': 'v', 'label': 'Short Open'},
                'long_close': {'color': 'green', 'marker': 'x', 'label': 'Long Close'},
                'short_close': {'color': 'red', 'marker': 'x', 'label': 'Short Close'},
            }
            handled_labels = set()
            for change in position_changes:
                step = change['step']
                price = change['price']
                change_type = change['type']
                marker_info = markers[change_type]
                label = marker_info['label'] if marker_info['label'] not in handled_labels else None
                plt.scatter(step, price, color=marker_info['color'], marker=marker_info['marker'], label=label)
                handled_labels.add(marker_info['label'])

            plt.axhline(y=0, color='black', linewidth=2)
            plt.title(f'Episode {idx + 1} Spread and Positions')
            plt.xlabel('Steps')
            plt.ylabel('Spread')
            plt.legend(loc='best')
            plt.show()


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
