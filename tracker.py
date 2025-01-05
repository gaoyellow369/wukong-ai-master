import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from log import log

class RewardTracker:
    def __init__(self, train_data_dir):
        self.train_data_dir = train_data_dir
        self.episode_rewards = []  # Save the total rewards for each game
        self.total_rewards = []  # Save the cumulative rewards for each action
        self.boss_healths = []  # Save the Boss's health at the end of each game
        self.episode_num = 0  # Record the current number of games

        # Create directory
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)

        # Initialize Boss health data file path
        self.boss_healths_file_path = os.path.join(train_data_dir, 'boss_healths.csv')
        self.fight_time_file_path = os.path.join(train_data_dir, 'fight_times.csv')

        # If the file does not exist, create the file and write the header
        if not os.path.exists(self.boss_healths_file_path):
            pd.DataFrame(columns=['Episode', 'Boss Health']).to_csv(self.boss_healths_file_path, index=False)

    def add_reward(self, reward):
        """Called after each action, saves the accumulated reward."""
        self.total_rewards.append(reward)

    def end_episode(self, boss_health, fight_time_seconds=None):
        """Called at the end of each game, saves the total reward, Boss health and resets the accumulated rewards."""
        if fight_time_seconds < 10:
            log(f"Fight time is too short: {fight_time_seconds}")
            self.total_rewards.clear()
            return
        total_reward = sum(self.total_rewards)
        self.episode_rewards.append(total_reward)
        self.boss_healths.append(boss_health)
        self.episode_num += 1

        # Save the reward data for the current game to a file
        self.save_episode_data()

        # Clear the reward data for the current game
        self.total_rewards.clear()

        # Save Boss health data at the end of each game
        self.save_boss_health_data()
        if fight_time_seconds is not None:
            self.save_fight_time_data(fight_time_seconds)

    def save_episode_data(self):
        """Saves the reward data for each game to a CSV file, using a timestamp to avoid overwriting."""
        episode_df = pd.DataFrame({
            'Action': range(len(self.total_rewards)),
            'Reward': self.total_rewards
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Use timestamp
        episode_file_path = os.path.join(self.train_data_dir, f'episode_{self.episode_num}_rewards_{timestamp}.csv')
        episode_df.to_csv(episode_file_path, index=False)
        log(f"Reward data has been saved to {episode_file_path}")

    def save_boss_health_data(self):
        """Saves Boss health data to the same CSV file."""
        # Create a data frame with the current game number and corresponding Boss health
        current_data = pd.DataFrame({
            'Episode': [self.episode_num],
            'Boss Health': [self.boss_healths[-1]]  # Only save the Boss health for the current game
        })

        # Append to file
        current_data.to_csv(self.boss_healths_file_path, mode='a', header=False, index=False)
        log(f"Boss health data has been saved to {self.boss_healths_file_path}")
    
    def save_fight_time_data(self, time_seconds):
        current_data = pd.DataFrame({
            'Episode': [self.episode_num],
            'Fight Time': [time_seconds]  # Only save the Boss health for the current game
        })

        # Append to file
        current_data.to_csv(self.fight_time_file_path, mode='a', header=False, index=False)
        log(f"Fight time data has been saved to {self.fight_time_file_path}")

    def save_overall_data(self):
        """Saves the total rewards data for all games, using a timestamp to avoid overwriting."""
        overall_df = pd.DataFrame({'Episode': range(1, len(self.episode_rewards) + 1),
                                   'Reward': self.episode_rewards})
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Use timestamp
        overall_file_path = os.path.join(self.train_data_dir, f'overall_rewards_{timestamp}.csv')
        overall_df.to_csv(overall_file_path, mode='a', index=False)
        log(f"Overall reward data has been saved to {overall_file_path}")

        self.episode_rewards.clear()
