import numpy as np
from stable_baselines3 import PPO, DDPG
from network_env_elephant import NetworkEnv
from flow_based_network_env_elephant import FlowBasedNetworkEnv
import matplotlib.pyplot as plt

def compare_histogram():
    dijkstra_env = NetworkEnv()
    flow_based_env = FlowBasedNetworkEnv()
    
    num_episodes = 100

    naive_sums_of_mice_flows_interactions_on_elephant = []
    djikstra_sums_of_mice_flows_interactions_on_elephant = []
    flowbased_sums_of_mice_flow_interactions_on_elephant = []

    article_model_path = "./models/elephant/5x5_grid/dijkstra-based/ddpg_sdn_elephant_routing_300000_steps" 
    article_model = DDPG.load(article_model_path, env = dijkstra_env)

    flow_based_model_path = "./models/elephant/5x5_grid/flow-based/ppo_elephant_discrete_3_paths_1400000_steps" 
    flow_based_model = PPO.load(flow_based_model_path, env = flow_based_env)

    for episode in range(1000, 1000 + num_episodes):
        dijkstra_obs, _ = dijkstra_env.reset(seed=episode)
        flowbased_obs, _ = flow_based_env.reset(seed=episode)

        dijkstra_action, _ = article_model.predict(dijkstra_obs, deterministic=True)

        # Apply the weights to the network and calculate results
        obs, dijkstra_reward, terminated, truncated, info = dijkstra_env.step(dijkstra_action)

        elephant_flow_path = dijkstra_env.flows_paths[dijkstra_env.elephant_flow]
        first_3_switches_of_elephant_flow_path = elephant_flow_path[:4]
        set_elephant_flow_path = set(first_3_switches_of_elephant_flow_path)

        dijkstra_sum_of_mice_flows_interactions = 0
        for flow, path in dijkstra_env.flows_paths.items(): 
            shared_elemets = len(set(path) & set_elephant_flow_path)
            dijkstra_sum_of_mice_flows_interactions += shared_elemets
        
        djikstra_sums_of_mice_flows_interactions_on_elephant.append(dijkstra_sum_of_mice_flows_interactions)

        naive_obs, _ = dijkstra_env.reset(seed=episode)
        naive_action = np.ones(155)
        obs, naive_reward, terminated, truncated, info = dijkstra_env.step(naive_action)

        elephant_flow_path = dijkstra_env.flows_paths[dijkstra_env.elephant_flow]
        first_3_switches_of_elephant_flow_path = elephant_flow_path[:4]
        set_elephant_flow_path = set(first_3_switches_of_elephant_flow_path)

        naive_sum_of_mice_flows_interactions = 0
        for flow, path in dijkstra_env.flows_paths.items():
            shared_elemets = len(set(path) & set_elephant_flow_path)
            naive_sum_of_mice_flows_interactions += shared_elemets
        
        naive_sums_of_mice_flows_interactions_on_elephant.append(naive_sum_of_mice_flows_interactions)

        flowbased_action, _ = flow_based_model.predict(flowbased_obs, deterministic = True)

        truncated = False

        while not truncated:
            flowbased_obs, flow_based_reward, terminated, truncated, info = flow_based_env.step(flowbased_action)

            flowbased_action, _ = flow_based_model.predict(flowbased_obs, deterministic = True)
        
        elephant_flow_path = flow_based_env.flows_paths[flow_based_env.elephant_flow]
        first_3_switches_of_elephant_flow_path = elephant_flow_path[:4]
        set_elephant_flow_path = set(first_3_switches_of_elephant_flow_path)

        flowbased_sum_of_mice_flows_interactions = 0
        for flow, path in flow_based_env.flows_paths.items():
            shared_elemets = len(set(path) & set_elephant_flow_path)
            flowbased_sum_of_mice_flows_interactions += shared_elemets
        
        flowbased_sums_of_mice_flow_interactions_on_elephant.append(flowbased_sum_of_mice_flows_interactions)

        print(f"Scenario {episode + 1}:")
        print(f"{dijkstra_sum_of_mice_flows_interactions = }")
        print(f"{naive_sum_of_mice_flows_interactions = }")
        print(f"{flowbased_sum_of_mice_flows_interactions = }")
        print("-" * 50)

    print("\n=== Final Evaluation Results ===")

    # Helper function to calculate 95% Confidence Interval
    def get_ci(data):
        # 1.96 is the Z-value for 95% confidence
        return 1.96 * (np.std(data, ddof=1) / np.sqrt(len(data)))

    # === PLOTTING ===

    # 1. Calculate the Means
    naive_mean = np.mean(naive_sums_of_mice_flows_interactions_on_elephant)
    dijkstra_mean = np.mean(djikstra_sums_of_mice_flows_interactions_on_elephant)
    flowbased_mean = np.mean(flowbased_sums_of_mice_flow_interactions_on_elephant)

    # Print the exact means to the console
    print(f"Naive Mean: {naive_mean:.2f}")
    print(f"Dijkstra Mean: {dijkstra_mean:.2f}")
    print(f"Flow-based Mean: {flowbased_mean:.2f}")

    # 2. Calculate the 95% Confidence Intervals
    naive_ci = get_ci(naive_sums_of_mice_flows_interactions_on_elephant)
    dijkstra_ci = get_ci(djikstra_sums_of_mice_flows_interactions_on_elephant)
    flowbased_ci = get_ci(flowbased_sums_of_mice_flow_interactions_on_elephant)

    # 3. Setup data for the chart
    labels = ['Naive', 'Dijkstra (DDPG)', 'Flow-based (PPO)']
    means = [naive_mean, dijkstra_mean, flowbased_mean]
    errors = [naive_ci, dijkstra_ci, flowbased_ci]
    
    # Matching the colors used in previous graphs (red, blue, green style)
    colors = ['#ff4d4d', '#4d4dff', '#4daf4a'] 

    # 4. Create the figure and plot
    plt.figure(figsize=(10, 6))
    
    # Plot the bars with yerr (error bars) and capsize (horizontal lines on error bars)
    bars = plt.bar(labels, means, yerr=errors, capsize=10, color=colors, alpha=0.85, edgecolor='black')

    # Add text labels on top of each bar for exact mean values
    for bar in bars:
        yval = bar.get_height()
        # Offset the text slightly above the maximum error bar
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(errors) * 0.2) + 0.2, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

    # 5. Formatting
    plt.ylabel('Average Number of Interceptions', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add padding to the top of the y-axis so the text labels and error bars don't get cut off
    plt.ylim(0, max(means) + max(errors) + max(means)*0.15)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_histogram()