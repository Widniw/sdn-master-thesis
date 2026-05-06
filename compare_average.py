import os
import numpy as np
from stable_baselines3 import PPO, DDPG
from network_env import NetworkEnv
from flow_based_network_env import FlowBasedNetworkEnv
import matplotlib.pyplot as plt

def compare_average():
    dijkstra_env = NetworkEnv()
    flow_based_env = FlowBasedNetworkEnv()
    
    num_episodes = 100

    dijkstra_total_delays = []
    dijkstra_total_packet_losses = []
    dijkstra_total_rewards = []

    naive_total_delays = []
    naive_total_packet_losses = []
    naive_total_rewards = []

    flow_based_total_delays = []
    flow_based_total_packet_losses = []
    flow_based_total_rewards = []

    article_model_path = "./models/article_dijkstra/ddpg_sdn_routing_200000_steps" 
    article_model = DDPG.load(article_model_path, env=dijkstra_env)

    flow_based_model_path = "./models/my_approach_flow_based/ppo_correct_discrete_3_paths_1200000_steps" 
    flow_based_model = PPO.load(flow_based_model_path, env=flow_based_env)

    print(f"Starting evaluation of {num_episodes} episodes. Please wait...")

    for episode in range(1000, 1000 + num_episodes):
        # 1. Dijkstra (Article) Model
        dijkstra_obs, _ = dijkstra_env.reset(seed=episode)
        dijkstra_action, _ = article_model.predict(dijkstra_obs, deterministic=True)
        obs, dijkstra_reward, terminated, truncated, info = dijkstra_env.step(dijkstra_action)
        
        dijkstra_total_delays.append(info.get('avg_delay', 0))
        dijkstra_total_packet_losses.append(info.get('packet_loss', 0))
        dijkstra_total_rewards.append(dijkstra_reward)

        # 2. Naive Baseline
        naive_obs, _ = dijkstra_env.reset(seed=episode)
        naive_action = np.ones(155)
        obs, naive_reward, terminated, truncated, info = dijkstra_env.step(naive_action)

        naive_total_delays.append(info.get('avg_delay', 0))
        naive_total_packet_losses.append(info.get('packet_loss', 0))
        naive_total_rewards.append(naive_reward)

        # 3. Flow Based Model
        flowbased_obs, _ = flow_based_env.reset(seed=episode)
        flowbased_action, _ = flow_based_model.predict(flowbased_obs, deterministic=True)
        truncated_flow = False

        while not truncated_flow:
            flowbased_obs, flow_based_reward, terminated_flow, truncated_flow, info = flow_based_env.step(flowbased_action)
            flowbased_action, _ = flow_based_model.predict(flowbased_obs, deterministic=True)

        flow_based_total_delays.append(info.get('avg_delay', 0))
        flow_based_total_packet_losses.append(info.get('packet_loss', 0))
        flow_based_total_rewards.append(flow_based_reward)

        # Reduced printing: Update progress every 100 episodes
        if (episode + 1) % 10 == 0:
            print(f"Processed {episode + 1 - 1000} / {num_episodes} scenarios...")

    print("\n=== Final Evaluation Results ===")
    print(f"Dijkstra Average Delay: \t\t {np.mean(dijkstra_total_delays):.4f} s")
    print(f"Dijkstra Average Packet Loss: \t {np.mean(dijkstra_total_packet_losses):.2f} pkts/s")
    print(f"Dijkstra Average Reward: \t\t {np.mean(dijkstra_total_rewards):.4f}")
    
    print(f"Flow Based Average Delay: \t\t {np.mean(flow_based_total_delays):.4f} s")
    print(f"Flow Based Average Packet Loss: \t {np.mean(flow_based_total_packet_losses):.2f} pkts/s")
    print(f"Flow Based Average Reward: \t\t {np.mean(flow_based_total_rewards):.4f}")
    
    print(f"Naive Average Delay: \t\t {np.mean(naive_total_delays):.4f} s")
    print(f"Naive Average Packet Loss: \t\t {np.mean(naive_total_packet_losses):.2f} pkts/s")
    print(f"Naive Average Reward: \t\t {np.mean(naive_total_rewards):.4f}")

    # Helper function to calculate 95% Confidence Interval
    def get_ci(data):
        # 1.96 is the Z-value for 95% confidence
        return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

    # Metrics to plot
    plot_data = [
        ("Average Delay (s)", naive_total_delays, dijkstra_total_delays, flow_based_total_delays),
        ("Packet Loss (pkts/s)", naive_total_packet_losses, dijkstra_total_packet_losses, flow_based_total_packet_losses),
        ("Average Reward", naive_total_rewards, dijkstra_total_rewards, flow_based_total_rewards)
    ]

    labels = ['Naive', 'Dijkstra', 'Flow-based']
    colors = ['red', 'blue', 'green']

    # Create a figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Performance Comparison across Models (with 95% CI)', fontweight='bold', fontsize=16)

    for i, (title, naive_data, dij_data, flow_data) in enumerate(plot_data):
        means = [np.mean(naive_data), np.mean(dij_data), np.mean(flow_data)]
        cis = [get_ci(naive_data), get_ci(dij_data), get_ci(flow_data)]
        
        # Plot the bars with error caps
        axes[i].bar(labels, means, yerr=cis, capsize=8, color=colors, alpha=0.7, edgecolor='black')
        axes[i].set_title(title, fontweight='bold', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        if title == "Average Delay (s)":
            axes[i].set_ylim(bottom = 6)
        
        if title == "Packet Loss (pkts/s)":
            axes[i].set_ylim(bottom = 8000)

        if title == "Average Reward":
            axes[i].set_ylim(bottom = 0.7)

    plt.tight_layout()

    # Create /images directory if it doesn't exist
    save_dir = './images'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure as an SVG file BEFORE plt.show()
    save_path = os.path.join(save_dir, 'model_comparison_bars.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"\nFigure successfully saved to: {save_path}")

    # Render the plot to screen
    plt.show()

if __name__ == "__main__":
    compare_average()