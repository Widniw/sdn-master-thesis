import numpy as np
from stable_baselines3 import PPO
from network_env import NetworkEnv
from flow_based_network_env import FlowBasedNetworkEnv

def sanity_check():
    dijkstra_env = NetworkEnv()
    flow_based_env = FlowBasedNetworkEnv()
    
    num_episodes = 1

    dijkstra_total_delays = []
    dijkstra_total_packet_losses = []
    dijkstra_total_rewards = []

    flow_based_total_delays = []
    flow_based_total_packet_losses = []
    flow_based_total_rewards = []

    print(f"\n Running sanity check over {num_episodes} different traffic scenarios...")
    print("-" * 50)

    for episode in range(num_episodes):
        dijkstra_env.reset(seed=episode)
        flow_based_env.reset(seed=episode)

        dijkstra_action = np.ones(130)

        # Apply the weights to the network and calculate results
        obs, dijkstra_reward, terminated, truncated, info = dijkstra_env.step(dijkstra_action)
        
        dijkstra_avg_delay = info.get('avg_delay', 0)
        dijkstra_packet_loss = info.get('packet_loss', 0)
        
        dijkstra_total_delays.append(dijkstra_avg_delay)
        dijkstra_total_packet_losses.append(dijkstra_packet_loss)
        dijkstra_total_rewards.append(dijkstra_reward)

        truncated = False

        while not truncated:
            obs, flow_based_reward, terminated, truncated, info = flow_based_env.step(0)

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


        print("-" * 50)

    print("\n=== Final Evaluation Results ===")
    print(f"Dijkstra Average Delay across {num_episodes} scenarios: \t\t {np.mean(dijkstra_total_delays):.4f} s")
    print(f"Dijkstra Average Packet Loss across {num_episodes} scenarios: \t {np.mean(dijkstra_total_packet_losses):.2f} pkts/s")
    print(f"Dijkstra Average Reward across {num_episodes} scenarios: \t\t {np.mean(dijkstra_total_rewards):.4f} s")
    print(f"Flow Based Average Delay across {num_episodes} scenarios: \t\t {np.mean(flow_based_total_delays):.4f} s")
    print(f"Flow Based Average Packet Loss across {num_episodes} scenarios: \t {np.mean(flow_based_total_packet_losses):.2f} pkts/s")
    print(f"Flow Based Average Reward across {num_episodes} scenarios: \t\t {np.mean(flow_based_total_rewards):.4f} pkts/s")



if __name__ == "__main__":
    sanity_check()