import numpy as np
from flow_based_network_env_simplified import FlowBasedNetworkEnv

def compare_histogram():
    flow_based_env = FlowBasedNetworkEnv()
    
    num_episodes = 10

    naive_total_delays = []
    naive_total_packet_losses = []
    naive_total_rewards = []

    for episode in range(1000, 1000 + num_episodes):
        flowbased_obs, _ = flow_based_env.reset(seed=episode)

        flowbased_action = 0
        truncated = False

        while not truncated:
            flowbased_obs, naive_based_reward, terminated, truncated, info = flow_based_env.step(flowbased_action)

        naive_based_avg_delay = info.get('avg_delay', 0)
        naive_based_packet_loss = info.get('packet_loss', 0)
        
        naive_total_delays.append(naive_based_avg_delay)
        naive_total_packet_losses.append(naive_based_packet_loss)
        naive_total_rewards.append(naive_based_reward)


        print(f"Scenario {episode + 1}:")
        print(f" Naive based Average Delay: {naive_based_avg_delay:.4f} s")
        print(f" Naive based  Packet Loss:   {naive_based_packet_loss:.2f} pkts/s")
        print(f" Naive based Reward Given:  {naive_based_reward:.4f}")

        print("-" * 50)

    print("\n=== Final Evaluation Results ===")
    print(f"Naive Average Delay across {num_episodes} scenarios: \t\t {np.mean(naive_total_delays):.4f} s")
    print(f"Naive Average Packet Loss across {num_episodes} scenarios: \t\t {np.mean(naive_total_packet_losses):.2f} pkts")
    print(f"Naive Average Reward across {num_episodes} scenarios: \t\t {np.mean(naive_total_rewards):.4f}")


if __name__ == "__main__":
    compare_histogram()