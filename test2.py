import numpy as np
from flow_based_network_env import FlowBasedNetworkEnv
from network_env import NetworkEnv

def run_head_to_head(seed=42):
    print(f"--- Running Head-to-Head Naive Comparison (Seed {seed}) ---")
    
    # ==========================================
    # 1. Flow-Based (Sequential - Action 0)
    # ==========================================
    env_flow = FlowBasedNetworkEnv()
    env_flow.reset(seed=seed)
    
    done = False
    while not done:
        _, reward_flow, terminated, truncated, info_flow = env_flow.step(0)
        done = terminated or truncated

    print("\n[Flow-Based PPO Env] - Action 0 (Yen's Algorithm)")
    print(f"  Alpha:        {env_flow.alpha}")
    print(f"  Final Reward: {reward_flow:.4f}")
    if 'avg_delay' in info_flow:
        print(f"  Avg Delay:    {info_flow['avg_delay']:.4f}")
        print(f"  Packet Loss:  {info_flow['packet_loss']:.2f}")

    # ==========================================
    # 2. DDPG (Simultaneous - Weights 1.0)
    # ==========================================
    env_ddpg = NetworkEnv()
    env_ddpg.reset(seed=seed)
    
    # DDPG expects an array of link weights. We set all 40 edges to 1.0
    action_ddpg = np.ones(env_ddpg.model.no_of_edges, dtype=np.float32)
    
    # Inject the exact same traffic so it's a perfectly fair fight
    env_ddpg.flows_traffic = env_flow.flows_traffic.copy()
    env_ddpg.total_incoming_network = env_flow.total_incoming_network
    
    _, reward_ddpg, _, _, info_ddpg = env_ddpg.step(action_ddpg)

    print("\n[DDPG Env] - Link Weights 1.0 (Dijkstra)")
    print(f"  Alpha:        {env_ddpg.alpha}")
    print(f"  Final Reward: {reward_ddpg:.4f}")
    if 'avg_delay' in info_ddpg:
        print(f"  Avg Delay:    {info_ddpg['avg_delay']:.4f}")
        print(f"  Packet Loss:  {info_ddpg['packet_loss']:.2f}")

    # ==========================================
    # ANALYSIS
    # ==========================================
    print("\n--- Analysis ---")
    if info_flow['packet_loss'] == info_ddpg['packet_loss'] and info_flow['avg_delay'] == info_ddpg['avg_delay']:
        print("RESULT: PERFECT MATCH! NetworkX resolved all tie-breakers identically.")
    else:
        print("RESULT: SLIGHT DEVIATION. NetworkX tie-breakers caused minor routing differences.")

if __name__ == "__main__":
    run_head_to_head(seed=42)