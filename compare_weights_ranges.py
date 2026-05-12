import numpy as np
from stable_baselines3 import DDPG
from network_env import NetworkEnv
from network_env_weight10 import NetworkEnv10weight
import matplotlib.pyplot as plt

def compare_histogram():
    dijkstra_env = NetworkEnv()
    dijkstra_env_weight10 = NetworkEnv10weight()
    
    num_episodes = 1000

    dijkstra_average_rewards = []
    djikstra_weight10_average_rewards = []

    dijkstra_model_path = "./models/article_dijkstra/ddpg_sdn_routing_200000_steps" 
    dijkstra_model = DDPG.load(dijkstra_model_path, env = dijkstra_env)

    dijkstra_weight10_model_path = "./models/weight_range_comparison/ddpg_sdn_routing_400000_steps_10_weights"
    dijkstra_weight10_model = DDPG.load(dijkstra_weight10_model_path, env = dijkstra_env_weight10)


    for episode in range(1000, 1000 + num_episodes):
        dijkstra_obs, _ = dijkstra_env.reset(seed=episode)
        dijkstra_action, _ = dijkstra_model.predict(dijkstra_obs, deterministic=True)

        # Apply the weights to the network and calculate results
        obs, dijkstra_reward, terminated, truncated, info = dijkstra_env.step(dijkstra_action)

        dijkstra_average_rewards.append(dijkstra_reward)

        dijkstra_weight10_obs, _ = dijkstra_env_weight10.reset(seed = episode)
        dijkstra_weight10_action, _ = dijkstra_weight10_model.predict(dijkstra_weight10_obs, deterministic=True)

        obs, dijkstra_weight10_reward, terminated, truncated, info = dijkstra_env_weight10.step(dijkstra_weight10_action)

        djikstra_weight10_average_rewards.append(dijkstra_weight10_reward)
        
        print(f"Scenario {episode + 1}:")
        print(f"Dijkstra reward = {dijkstra_reward}")
        print(f"Dijkstra weight 10 reward = {dijkstra_weight10_reward}")
        print("-" * 50)

    # print("\n=== Final Evaluation Results ===")

    # # Helper function to calculate 95% Confidence Interval
    # def get_ci(data):
    #     # 1.96 is the Z-value for 95% confidence
    #     return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

    # === PLOTTING ===

    # 1. Calculate the Means
    dijkstra_mean = np.mean(dijkstra_average_rewards)
    dijkstra_weight10_mean = np.mean(djikstra_weight10_average_rewards)

    # Print the exact means to the console
    print(f"Dijkstra mean: {dijkstra_mean:.3f}")
    print(f"Dijkstra Weight 10 Mean: {dijkstra_weight10_mean:.3f}")

    # # 2. Calculate the 95% Confidence Intervals
    # naive_ci = get_ci(naive_sums_of_mice_flows_interactions_on_elephant)
    # dijkstra_ci = get_ci(djikstra_sums_of_mice_flows_interactions_on_elephant)
    # flowbased_ci = get_ci(flowbased_sums_of_mice_flow_interactions_on_elephant)

    # # 3. Setup data for the chart
    # labels = ['Naive', 'Dijkstra (DDPG)', 'Flow-based (PPO)']
    # means = [naive_mean, dijkstra_mean, flowbased_mean]
    # errors = [naive_ci, dijkstra_ci, flowbased_ci]
    
    # # Matching the colors used in previous graphs (red, blue, green style)
    # colors = ['#ff4d4d', '#4d4dff', '#4daf4a'] 

    # # 4. Create the figure and plot
    # plt.figure(figsize=(10, 6))
    
    # # Plot the bars with yerr (error bars) and capsize (horizontal lines on error bars)
    # bars = plt.bar(labels, means, yerr=errors, capsize=10, color=colors, alpha=0.85, edgecolor='black')

    # # Add text labels on top of each bar for exact mean values
    # for bar in bars:
    #     yval = bar.get_height()
    #     # Offset the text slightly above the maximum error bar
    #     plt.text(bar.get_x() + bar.get_width()/2, yval + (max(errors) * 0.2) + 0.2, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

    # # 5. Formatting
    # plt.ylabel('Average Number of Interceptions', fontsize=12, fontweight='bold')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # # Add padding to the top of the y-axis so the text labels and error bars don't get cut off
    # plt.ylim(0, max(means) + max(errors) + max(means)*0.15)

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    compare_histogram()