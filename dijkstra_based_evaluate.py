import numpy as np
from stable_baselines3 import DDPG
from network_env import NetworkEnv

def evaluate_model():
    print("Loading the Network Environment...")
    env = NetworkEnv()
    
    print("Loading the trained DDPG model...")
    try:
        model = DDPG.load("ddpg_sdn_routing")
    except FileNotFoundError:
        print("Error: 'ddpg_sdn_routing.zip' not found. Please run train.py first!")
        return

    num_episodes = 10
    total_delays = []
    total_packet_losses = []

    print(f"\nEvaluating the trained agent over {num_episodes} different traffic scenarios...")
    print("-" * 50)

    for episode in range(num_episodes):
        # Reset the environment to generate a new set of random flows
        obs, _ = env.reset()
        
        # Predict the optimal link weights based on the ATVM state
        # We use deterministic=True to disable exploration noise during evaluation
        action, _states = model.predict(obs, deterministic=True)
        
        # Apply the weights to the network and calculate results
        obs, reward, terminated, truncated, info = env.step(action)
        
        avg_delay = info.get('avg_delay', 0)
        packet_loss = info.get('packet_loss', 0)
        
        total_delays.append(avg_delay)
        total_packet_losses.append(packet_loss)
        
        print(f"Scenario {episode + 1}:")
        print(f"  Average Delay: {avg_delay:.4f} s")
        print(f"  Packet Loss:   {packet_loss:.2f} pkts/s")
        print(f"  Reward Given:  {reward:.4f}")
        print("-" * 50)

    print("\n=== Final Evaluation Results ===")
    print(f"Average Delay across {num_episodes} scenarios:       {np.mean(total_delays):.4f} s")
    print(f"Average Packet Loss across {num_episodes} scenarios: {np.mean(total_packet_losses):.2f} pkts/s")

if __name__ == "__main__":
    evaluate_model()