import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from network_env import NetworkEnv  
import networkx as nx

def main():
    print("Loading environment and model...")
    env = NetworkEnv()
    
    # Load your best trained DDPG model
    best_model_path = "ddpg_sdn_routing_article_universal_model" 
    model = DDPG.load(best_model_path, env=env)

    num_test_episodes = 100
    naive_rewards = []
    ddpg_rewards = []

    print(f"Starting evaluation over {num_test_episodes} traffic scenarios...\n")

    for episode in range(num_test_episodes):
        # 1. Reset the environment to generate a new random traffic matrix
        obs, _ = env.reset(seed=episode) # Set seed early to be safe
        
        # --- TEST 1: THE NAIVE METHOD ---
        naive_action = np.ones(env.action_space.shape)
        
        # Capture the REWARD (the second variable returned by step)
        _, naive_reward, _, _, _ = env.step(naive_action)
        naive_rewards.append(naive_reward)
        
        weights_dict = nx.get_edge_attributes(env.model.G, 'weight')
        weights_list = list(weights_dict.values())
        print(f"Naive episode {episode}:{weights_list = }") # Commented out to reduce terminal spam

        # Reset the environment using the EXACT SAME SEED for fair comparison
        env.reset(seed=episode) 

        # --- TEST 2: THE DDPG AGENT ---
        ddpg_action, _ = model.predict(obs, deterministic=True)
        
        # Capture the REWARD
        _, ddpg_reward, _, _, _ = env.step(ddpg_action)
        ddpg_rewards.append(ddpg_reward)
        
        weights_dict = nx.get_edge_attributes(env.model.G, 'weight')
        weights_list = list(weights_dict.values())
        print(f"DRL episode {episode}:{weights_list = }")

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_test_episodes} completed...")

    # --- CALCULATE QUICK STATISTICS ---
    print("\n" + "="*40)
    print("FINAL RESULTS (Over 100 Scenarios):")
    print("="*40)
    print(f"Naive Average Reward: {np.mean(naive_rewards):.4f} (Median: {np.median(naive_rewards):.4f})")
    print(f"DDPG Average Reward:  {np.mean(ddpg_rewards):.4f} (Median: {np.median(ddpg_rewards):.4f})")
    print("="*40)

    # --- GENERATE THE THESIS GRAPH ---
    print("Generating Box Plot...")
    plt.figure(figsize=(8, 6))
    
    # Create a boxplot comparing the two lists of rewards
    box_data = [naive_rewards, ddpg_rewards]
    plt.boxplot(box_data, labels=['Naive (Shortest Path)', 'DDPG (Optimized Weights)'], 
                patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))

    # Formatting for academia
    plt.title("Total Reward: Naive Routing vs. DDPG Agent", fontsize=14, fontweight='bold')
    plt.ylabel("Reward", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate improvement (Flipped because Higher Reward = Better)
    improvement = ((np.mean(ddpg_rewards) - np.mean(naive_rewards)) / np.mean(naive_rewards)) * 100
    plt.text(0.95, 0.05, f"AI Improvement: {improvement:.2f}%", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig("thesis_reward_comparison_boxplot.png", dpi=300)
    print("Graph saved as 'thesis_reward_comparison_boxplot.png'!")
    
    plt.show()

if __name__ == "__main__":
    main()