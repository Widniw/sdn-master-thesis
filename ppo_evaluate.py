import numpy as np
import matplotlib.pyplot as plt
import random
from stable_baselines3 import PPO
from flow_based_network_env import NetworkEnv  
from pathlib import Path
import warnings

# Suppress minor matplotlib/gym warnings for a clean console
warnings.filterwarnings("ignore")

def main():
    print("Loading network environment and 350k PPO model...")
    env = NetworkEnv()
    
    # Point exactly to the 350k checkpoint
    base_dir = Path(__file__).resolve().parent
    checkpoint_path = base_dir / "models" / "my_approach_flow_based" / "ppo_discrete_1850000_steps.zip" 
    
    # Load the model
    model = PPO.load(checkpoint_path, env=env, device="cpu")

    num_test_episodes = 100
    naive_rewards = []
    ppo_rewards = []

    print(f"\nStarting evaluation over {num_test_episodes} unseen traffic scenarios...")
    print("-" * 50)

    for episode in range(num_test_episodes):
        # We use a high seed range (10000+) to guarantee data the AI hasn't memorized
        test_seed = 10000 + episode 
        
        # --- TEST 1: NAIVE METHOD (Dijkstra Shortest Path) ---
        random.seed(test_seed)
        np.random.seed(test_seed)
        obs, _ = env.reset(seed=test_seed) 
        
        # Action 0 means "use path 0" for all 625 flows
        naive_action = np.zeros(env.action_space.shape, dtype=int)
        _, naive_reward, _, _, _ = env.step(naive_action)
        naive_rewards.append(naive_reward)

        # --- TEST 2: THE PPO AGENT ---
        # Reset with the exact same seed so the traffic matrix is identical
        random.seed(test_seed)
        np.random.seed(test_seed)
        obs, _ = env.reset(seed=test_seed) 

        # deterministic=True strips away random exploration
        ppo_action, _ = model.predict(obs, deterministic=True)
        _, ppo_reward, _, _, _ = env.step(ppo_action)
        ppo_rewards.append(ppo_reward)

        # --- LIVE TELEMETRY ---
        # Print the exact distribution of paths the AI chose for this specific matrix
        unique_actions, counts = np.unique(ppo_action, return_counts=True)
        action_distribution = dict(zip(unique_actions, counts))
        
        # Only print every 10th episode to keep the console clean
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_test_episodes}")
            print(f"  AI Route Choices: {action_distribution}")
            print(f"  Naive Reward: {naive_reward:.4f} | PPO Reward: {ppo_reward:.4f}")
            print("-" * 50)

    # --- CALCULATE FINAL STATISTICS ---
    print("\n" + "="*50)
    print("FINAL RESULTS (Over 100 Unseen Scenarios):")
    print("="*50)
    print(f"Naive Average Reward: {np.mean(naive_rewards):.4f} (Median: {np.median(naive_rewards):.4f})")
    print(f"PPO Average Reward:   {np.mean(ppo_rewards):.4f} (Median: {np.median(ppo_rewards):.4f})")
    print("="*50)

    # --- GENERATE THE BOX PLOT ---
    print("\nGenerating Box Plot...")
    plt.figure(figsize=(8, 6))
    
    box_data = [naive_rewards, ppo_rewards]
    plt.boxplot(box_data, labels=['Naive (Shortest Path)', 'PPO (AI Rerouted)'], 
                patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))

    plt.title("Total Reward: Naive Routing vs. PPO Agent (350k Steps)", fontsize=14, fontweight='bold')
    plt.ylabel("Reward", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    improvement = ((np.mean(ppo_rewards) - np.mean(naive_rewards)) / np.mean(naive_rewards)) * 100
    plt.text(0.95, 0.05, f"AI Improvement: {improvement:.2f}%", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig("thesis_ppo_300k_evaluation.png", dpi=300)
    print("Graph saved as 'thesis_ppo_300k_evaluation.png'!")

if __name__ == "__main__":
    main()