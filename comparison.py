import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """
    Smooths out the extremely noisy RL rewards using a mathematical moving average.
    This creates the clean, thick lines you see in academic papers.
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_algorithm(log_dir_base, color, label, window=150):
    """Scans the directory for seed folders and plots their learning curves."""
    if not os.path.exists(log_dir_base):
        print(f"⚠️ Directory {log_dir_base} not found. Skipping...")
        return

    # Find all folders that start with "seed_"
    seed_dirs = [d for d in os.listdir(log_dir_base) if d.startswith("seed_")]
    
    if not seed_dirs:
        print(f"⚠️ No seed folders found in {log_dir_base}. Skipping...")
        return

    for i, seed_dir in enumerate(seed_dirs):
        full_path = os.path.join(log_dir_base, seed_dir)
        
        try:
            # load_results automatically merges the 4 monitor.csv files from your 4 CPU cores
            df = load_results(full_path)
            
            # ts2xy extracts the Timesteps (x) and the Rewards (y)
            x, y = ts2xy(df, 'timesteps')
            
            # Apply the moving average to smooth the line
            if len(y) > window:
                y_smoothed = moving_average(y, window=window)
                # Trim the X-axis to match the smoothed Y-axis
                x_smoothed = x[len(x) - len(y_smoothed):]
                
                # Only add the label to the legend for the very first line so it doesn't duplicate
                current_label = label if i == 0 else None
                
                # Plot the line! alpha=0.6 makes them slightly transparent so overlapping lines look good
                plt.plot(x_smoothed, y_smoothed, color=color, alpha=0.6, label=current_label, linewidth=2)
        except Exception as e:
            print(f"Could not load data for {full_path}: {e}")

def main():
    print("Generating learning curves...")
    
    # Set up the figure size (10x6 is standard for academic papers)
    plt.figure(figsize=(10, 6))
    
    # Plot both algorithms
    plot_algorithm("./training_logs/", color="blue", label="DDPG (Dijkstra Link Weights)")
    plot_algorithm("./ppo_training_logs/", color="red", label="PPO (Flow Path Selection)")
    
    # Thesis-level formatting
    plt.title("Learning Curve Comparison: DDPG vs. PPO", fontsize=16, fontweight='bold')
    plt.xlabel("Total Timesteps", fontsize=14)
    plt.ylabel("Reward (Moving Average)", fontsize=14)
    
    # Adds a subtle background grid for readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Put the legend in the bottom right corner so it doesn't block the data
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    
    # Save a high-resolution PNG for your thesis document
    output_filename = "thesis_learning_curves_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n🎉 Graph saved successfully as '{output_filename}'!")
    
    # Display the interactive window
    plt.show()

if __name__ == "__main__":
    main()