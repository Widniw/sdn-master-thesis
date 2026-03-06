import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """Smooths the RL noise to create the thick, clean lines seen in academic papers."""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def main():
    # 1. Point directly to your specific log folder
    log_dir = "./training_logs/seed_999/"
    
    if not os.path.exists(log_dir):
        print(f"Error: Could not find the directory '{log_dir}'")
        return

    print(f"Loading data from {log_dir}...")
    
    # 2. Extract the data using Stable-Baselines3 built-in tools
    try:
        df = load_results(log_dir)
        x, y = ts2xy(df, 'timesteps')
    except Exception as e:
        print(f"Failed to load logs. Ensure monitor.csv exists in the folder. Error: {e}")
        return

    # 3. Apply the smoothing window (Adjust this number if the line is too spiky or too flat)
    smoothing_window = 200
    if len(y) > smoothing_window:
        y_smoothed = moving_average(y, window=smoothing_window)
        x_smoothed = x[len(x) - len(y_smoothed):]
    else:
        print("Not enough data to smooth! Plotting raw data instead.")
        y_smoothed = y
        x_smoothed = x

    # 4. Generate the Paper-Style Graph
    print("Generating graph...")
    plt.figure(figsize=(8, 5)) # Matching the slightly wider aspect ratio of IEEE papers
    
    # Plot the thick red line, just like Figure 5a
    plt.plot(x_smoothed, y_smoothed, color='red', linewidth=2.5, label='Proposed (DDPG)')
    
    # Optional: Plot the raw data faintly in the background to show the true variance
    plt.plot(x, y, color='red', alpha=0.15) 

    # 5. Format to match the paper's aesthetics
    plt.title("Reward vs. Step", fontsize=14, fontweight='bold')
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    
    # The paper uses a clean grid
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    
    # Save a high-resolution copy for your document
    output_filename = "figure_5a_recreation.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Graph saved as '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    main()