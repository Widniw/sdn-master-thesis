import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from network_env import NetworkEnv  # Make sure this points to your Dijkstra environment!

def main():
    print("Loading environment and model...")
    env = NetworkEnv()
    
    # Load your best trained DDPG model
    # Update this filename to your actual best model!
    best_model_path = "ddpg_sdn_routing_seed_5555" 
    model = DDPG.load(best_model_path, env=env)

    num_test_episodes = 100
    naive_delays = []
    ddpg_delays = []

    print(f"Starting evaluation over {num_test_episodes} traffic scenarios...\n")

    for episode in range(num_test_episodes):
        # 1. Reset the environment to generate a new random traffic matrix
        obs, _ = env.reset()
        
        # --- TEST 1: THE NAIVE METHOD ---
        # To simulate standard Dijkstra, we set all link weights to exactly 1.0
        # This forces the algorithm to strictly use hop-count (shortest path)
        naive_action = np.ones(env.action_space.shape)
        
        # We step the environment using the naive weights
        _, _, _, _, naive_info = env.step(naive_action)
        naive_delays.append(naive_info['avg_delay'])

        # We must reset the environment using the EXACT SAME SEED so the DDPG agent 
        # has to solve the exact same traffic matrix that the Naive method just tried.
        env.reset(seed=episode) 

        # --- TEST 2: THE DDPG AGENT ---
        # Let the AI look at the traffic and output its optimized link weights
        # deterministic=True tells the AI to use its best guess without random exploration
        ddpg_action, _ = model.predict(obs, deterministic=True)
        
        _, _, _, _, ddpg_info = env.step(ddpg_action)
        ddpg_delays.append(ddpg_info['avg_delay'])

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_test_episodes} completed...")

    # --- CALCULATE QUICK STATISTICS ---
    print("\n" + "="*40)
    print("FINAL RESULTS (Over 100 Scenarios):")
    print("="*40)
    print(f"Naive Average Delay: {np.mean(naive_delays):.2f}s (Median: {np.median(naive_delays):.2f}s)")
    print(f"DDPG Average Delay:  {np.mean(ddpg_delays):.2f}s (Median: {np.median(ddpg_delays):.2f}s)")
    print("="*40)

    # --- GENERATE THE THESIS GRAPH ---
    print("Generating Box Plot...")
    plt.figure(figsize=(8, 6))
    
    # Create a boxplot comparing the two lists of delays
    box_data = [naive_delays, ddpg_delays]
    plt.boxplot(box_data, labels=['Naive (Shortest Path)', 'DDPG (Optimized Weights)'], 
                patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))

    # Formatting for academia
    plt.title("End-to-End Delay: Naive Routing vs. DDPG Agent", fontsize=14, fontweight='bold')
    plt.ylabel("Average Flow Delay (Seconds)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a text box with the exact improvement percentage
    improvement = ((np.mean(naive_delays) - np.mean(ddpg_delays)) / np.mean(naive_delays)) * 100
    plt.text(0.95, 0.95, f"AI Improvement: {improvement:.1f}%", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig("thesis_delay_comparison_boxplot.png", dpi=300)
    print("Graph saved as 'thesis_delay_comparison_boxplot.png'!")
    
    plt.show()

if __name__ == "__main__":
    main()