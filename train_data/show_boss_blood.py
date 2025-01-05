import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_boss_health(csv_file_path):
    """Plot Boss health curve."""
    if not os.path.isfile(csv_file_path):
        print(f"CSV file {csv_file_path} does not exist.")
        return

    # Read CSV file
    df = pd.read_csv(csv_file_path)
    if 'Episode' not in df.columns or 'Boss Health' not in df.columns:
        print("The CSV file is missing 'Episode' or 'Boss Health' column.")
        return

    # Plot curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Boss Health'], marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Boss Health')
    plt.title('Boss Health per Episode')
    plt.grid(True)

    # Set the save path to the image directory at the same level as the data directory
    image_dir = os.path.join(os.path.dirname(csv_file_path), '../image')
    os.makedirs(image_dir, exist_ok=True)  # If the image directory does not exist, create it

    # Save the plot result
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plot_file_path = os.path.join(image_dir, f'boss_health_plot_{timestamp}.png')
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Boss health curve has been saved to {plot_file_path}")

# Usage example
if __name__ == "__main__":
    csv_file_path = './data/boss_healths.csv'  # Replace with actual file path
    plot_boss_health(csv_file_path)
