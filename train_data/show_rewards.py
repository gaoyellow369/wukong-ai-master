import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_data_from_directory(directory, prefix):
    """Load CSV data files named with a specified prefix from a specified directory, and accumulate the total sum of the 'Reward' column for each file."""
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith(prefix)]
    rewards = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            if 'Reward' in df.columns:
                total_reward = df['Reward'].sum()  # Accumulate the Reward values throughout the file
                rewards.append(total_reward)
                print(f"Reading file: {file_path}, Total Reward value: {total_reward}")
            else:
                print(f"Warning: 'Reward' column not found in file")
        except Exception as e:
            print(f"Unable to read file {file_path}: {e}")
    return rewards

def plot_total_rewards(rewards, save_path):
    """Plot the curve of total rewards at the end of each game."""
    if rewards:  # Ensure there is data to plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rewards) + 1), rewards, marker='o')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Rewards per Episode')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Reward curve chart has been saved to {save_path}")
    else:
        print("No reward data to plot.")

def main(directory, prefix):
    """Main function, generates reward curve charts and data files."""
    rewards = load_data_from_directory(directory, prefix)
    
    # Get the parent directory of the current directory
    parent_directory = os.path.dirname(directory)
    
    # Create 'image' folder in the directory at the same level as data
    image_directory = os.path.join(parent_directory, 'image')
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    
    if rewards:
        # Generate reward curve chart
        plot_total_rewards(rewards, os.path.join(image_directory, f'total_rewards_curve_{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    else:
        print(f"No valid reward data files named with prefix '{prefix}' found.")

if __name__ == "__main__":
    # Replace with the directory where you save your data
    data_directory = './data'
    
    # Draw the curve for all game rewards
    prefix_type = 'episode_'  # Or 'overall_rewards' to draw the curve with the sum of rewards every 10 games
    
    main(data_directory, prefix_type)
