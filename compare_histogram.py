import numpy as np
from stable_baselines3 import PPO, DDPG
from network_env import NetworkEnv
from flow_based_network_env import FlowBasedNetworkEnv
import matplotlib.pyplot as plt

def compare_histogram():
    dijkstra_env = NetworkEnv()
    flow_based_env = FlowBasedNetworkEnv()
    
    num_episodes = 1000

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
    article_model = DDPG.load(article_model_path, env = dijkstra_env)

    flow_based_model_path = "./models/my_approach_flow_based/ppo_correct_discrete_3_paths_1200000_steps" 
    flow_based_model = PPO.load(flow_based_model_path, env = flow_based_env)

    for episode in range(1000, 1000 + num_episodes):
        dijkstra_obs, _ = dijkstra_env.reset(seed=episode)
        flowbased_obs, _ = flow_based_env.reset(seed=episode)

        dijkstra_action, _ = article_model.predict(dijkstra_obs, deterministic=True)

        # Apply the weights to the network and calculate results
        obs, dijkstra_reward, terminated, truncated, info = dijkstra_env.step(dijkstra_action)
        
        dijkstra_avg_delay = info.get('avg_delay', 0)
        dijkstra_packet_loss = info.get('packet_loss', 0)
        
        dijkstra_total_delays.append(dijkstra_avg_delay)
        dijkstra_total_packet_losses.append(dijkstra_packet_loss)
        dijkstra_total_rewards.append(dijkstra_reward)

        naive_obs, _ = dijkstra_env.reset(seed=episode)
        naive_action = np.ones(155)
        obs, naive_reward, terminated, truncated, info = dijkstra_env.step(naive_action)

        naive_avg_delay = info.get('avg_delay', 0)
        naive_packet_loss = info.get('packet_loss', 0)
        
        naive_total_delays.append(naive_avg_delay)
        naive_total_packet_losses.append(naive_packet_loss)
        naive_total_rewards.append(naive_reward)

        flowbased_action, _ = flow_based_model.predict(flowbased_obs, deterministic = True)

        truncated = False

        while not truncated:
            flowbased_obs, flow_based_reward, terminated, truncated, info = flow_based_env.step(flowbased_action)

            flowbased_action, _ = flow_based_model.predict(flowbased_obs, deterministic = True)

        flow_based_avg_delay = info.get('avg_delay', 0)
        flow_based_packet_loss = info.get('packet_loss', 0)
        
        flow_based_total_delays.append(flow_based_avg_delay)
        flow_based_total_packet_losses.append(flow_based_packet_loss)
        flow_based_total_rewards.append(flow_based_reward)


        print(f"Scenario {episode + 1}:")
        print(f" Dijkstra Average Delay: {dijkstra_avg_delay:.4f} s")
        print(f" Dijkstra  Packet Loss:   {dijkstra_packet_loss:.2f} pkts/s")
        print(f" Dijkstra Reward Given:  {dijkstra_reward:.4f}")
        print(f" Flow based Average Delay: {flow_based_avg_delay:.4f} s")
        print(f" Flow based  Packet Loss:   {flow_based_packet_loss:.2f} pkts/s")
        print(f" Flow based Reward Given:  {flow_based_reward:.4f}")
        print(f" Naive based Average Delay: {naive_avg_delay:.4f} s")
        print(f" Naive based  Packet Loss:   {naive_packet_loss:.2f} pkts/s")
        print(f" Naive based Reward Given:  {naive_reward:.4f}")

        print("-" * 50)

    print("\n=== Final Evaluation Results ===")
    print(f"Dijkstra Average Delay across {num_episodes} scenarios: \t\t {np.mean(dijkstra_total_delays):.4f} s")
    print(f"Dijkstra Average Packet Loss across {num_episodes} scenarios: \t {np.mean(dijkstra_total_packet_losses):.2f} pkts")
    print(f"Dijkstra Average Reward across {num_episodes} scenarios: \t\t {np.mean(dijkstra_total_rewards):.4f}")
    print(f"Flow Based Average Delay across {num_episodes} scenarios: \t\t {np.mean(flow_based_total_delays):.4f} s")
    print(f"Flow Based Average Packet Loss across {num_episodes} scenarios: \t {np.mean(flow_based_total_packet_losses):.2f} pkts/s")
    print(f"Flow Based Average Reward across {num_episodes} scenarios: \t\t {np.mean(flow_based_total_rewards):.4f}")
    print(f"Naive Average Delay across {num_episodes} scenarios: \t\t {np.mean(naive_total_delays):.4f} s")
    print(f"Naive Average Packet Loss across {num_episodes} scenarios: \t\t {np.mean(naive_total_packet_losses):.2f} pkts")
    print(f"Naive Average Reward across {num_episodes} scenarios: \t\t {np.mean(naive_total_rewards):.4f}")

    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Define the number of buckets (bins). 
    # For 10 episodes, 5 to 8 bins usually looks best.
    num_bins = 20

    # Plot the true frequency histograms with transparency so overlaps are visible
    plt.hist(dijkstra_total_rewards, bins=num_bins, color='blue', edgecolor='white', alpha=0.6, label='Article Model')
    plt.hist(naive_total_rewards, bins=num_bins, color='red', edgecolor='white', alpha=0.6, label='Naive Baseline')
    plt.hist(flow_based_total_rewards, bins=num_bins, color='green', edgecolor='white', alpha=0.6, label='Flow Based Model')
    
    # Add labels and title
    plt.xlabel('Reward', fontweight='bold', fontsize=12)
    plt.ylabel('Number of Scenarios', fontweight='bold', fontsize=12)
    # plt.title('Reward Distribution for 3 Approaches', fontweight='bold', fontsize=14)
    
    # Add a legend and grid for readability
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Render the plot
    plt.tight_layout()
    plt.show()

    plt.savefig()

if __name__ == "__main__":
    compare_histogram()