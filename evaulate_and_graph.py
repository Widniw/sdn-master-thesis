import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, PPO
from network_env import NetworkEnv  
from flow_based_network_env import FlowBasedNetworkEnv
import networkx as nx

def main():
    print("Loading environment and model...")
    ddpg_env = NetworkEnv()
    flowbased_env = FlowBasedNetworkEnv()
    
    # Load your best trained DDPG model
    best_model_path = "./models/article_dijkstra/ddpg_sdn_routing_200000_steps" 
    article_model = DDPG.load(best_model_path, env = ddpg_env)

    best_model_path = "./models/my_approach_flow_based/ppo_correct_discrete_3_paths_1200000_steps"
    flowbased_model = PPO.load(best_model_path, env = flowbased_env)

    num_test_episodes = 100
    naive_rewards = []
    ddpg_rewards = []
    flowbased_rewards = []

    print(f"Starting evaluation over {num_test_episodes} traffic scenarios...\n")

    for episode in range(num_test_episodes):
        obs, _ = ddpg_env.reset(seed=episode) 
        flowbased_obs, _ = flowbased_env.reset(seed=episode)

        # --- TEST 1: THE NAIVE METHOD ---
        naive_action = np.ones(ddpg_env.action_space.shape)
        
        # Capture the REWARD (the second variable returned by step)
        _, naive_reward, _, _, _ = ddpg_env.step(naive_action)
        naive_rewards.append(naive_reward)
        
        # weights_dict = nx.get_edge_attributes(ddpg_env.model.G, 'weight')
        # weights_list = list(weights_dict.values())
        # print(f"Naive episode {episode}:{weights_list = }") # Commented out to reduce terminal spam

        # Reset the environment using the EXACT SAME SEED for fair comparison
        ddpg_env.reset(seed=episode) 

        # --- TEST 2: THE DDPG AGENT ---
        ddpg_action, _ = article_model.predict(obs, deterministic=True)
        
        # Capture the REWARD
        _, ddpg_reward, _, _, _ = ddpg_env.step(ddpg_action)
        ddpg_rewards.append(ddpg_reward)
        
        # weights_dict = nx.get_edge_attributes(ddpg_env.model.G, 'weight')
        # weights_list = list(weights_dict.values())
        # print(f"DRL episode {episode}:{weights_list = }")

        truncated = False
        while not truncated:
            flowbased_action, _ = flowbased_model.predict(flowbased_obs, deterministic=True)
            flowbased_obs, flowbased_reward, _, truncated, _ = flowbased_env.step(flowbased_action)

        flowbased_rewards.append(flowbased_reward)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_test_episodes} completed...")

    # --- CALCULATE QUICK STATISTICS ---
    print("\n" + "="*40)
    print("FINAL RESULTS (Over 100 Scenarios):")
    print("="*40)
    print(f"Naive Average Reward: {np.mean(naive_rewards):.4f} (Median: {np.median(naive_rewards):.4f})")
    print(f"DDPG Average Reward:  {np.mean(ddpg_rewards):.4f} (Median: {np.median(ddpg_rewards):.4f})")
    print(f"Flowbased Average Reward:  {np.mean(flowbased_rewards):.4f} (Median: {np.median(flowbased_rewards):.4f})")

    print("="*40)

    # --- GENERATE THE THESIS GRAPH ---
    print("Generating Box Plot...")
    plt.figure(figsize=(8, 6))
    
    # Create a boxplot comparing the two lists of rewards
    box_data = [naive_rewards, ddpg_rewards, flowbased_rewards]
    plt.boxplot(box_data, labels=['Naive (Shortest Path)', 'DDPG (Optimized Weights)', 'Flowbased'], 
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