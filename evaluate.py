import numpy as np
import time
import networkx as nx
from stable_baselines3 import PPO, DDPG

# Import your environments (Make sure these match your exact file names!)
from flow_based_network_env import FlowBasedNetworkEnv
from network_env import NetworkEnv

def evaluate_algorithms(num_episodes=50):
    # 1. Load the Environments
    env_ppo = FlowBasedNetworkEnv()
    env_ddpg = NetworkEnv() 
    
    print("Loading Trained Models...")
    try:
        # REPLACE THESE PATHS WITH YOUR ACTUAL BEST .ZIP FILES
        ddpg_model = DDPG.load("./models/article_dijkstra/ddpg_sdn_routing_200000_steps")
        ppo_model = PPO.load("./models/my_approach_flow_based/ppo_correct_discrete_3_paths_100000_steps")
    except Exception as e:
        print(f"Error loading models. Please check your file paths!\nDetails: {e}")
        return

    # Metrics storage
    results = {
        "Naive": {"reward": [], "delay": [], "packet_loss": [], "time": []},
        "DDPG": {"reward": [], "delay": [], "packet_loss": [], "time": []},
        "PPO": {"reward": [], "delay": [], "packet_loss": [], "time": []}
    }

    print(f"\n--- Starting Evaluation over {num_episodes} Episodes ---")

    for episode in range(num_episodes):
        # ==========================================
        # TRAFFIC GENERATION (The Master Record)
        # ==========================================
        # We generate the traffic using the PPO env, then freeze it to inject everywhere else
        obs_ppo, _ = env_ppo.reset(seed=episode)
        
        frozen_traffic = env_ppo.flows_traffic.copy()
        frozen_sdm = env_ppo.SDM.copy()
        frozen_active_keys = env_ppo.active_flow_keys.copy()
        frozen_total_incoming = env_ppo.total_incoming_network

        # ==========================================
        # TEST 1: The PPO Agent (Sequential Flow-Based)
        # ==========================================
        start_time = time.time()
        done = False
        
        while not done:
            action, _states = ppo_model.predict(obs_ppo, deterministic=True)
            obs_ppo, reward_ppo, terminated, truncated, info_ppo = env_ppo.step(action)
            done = terminated or truncated

        ppo_time = time.time() - start_time
        
        results["PPO"]["reward"].append(reward_ppo)
        results["PPO"]["delay"].append(info_ppo['avg_delay'])
        results["PPO"]["packet_loss"].append(info_ppo['packet_loss'])
        results["PPO"]["time"].append(ppo_time)


        # ==========================================
        # TEST 2: DDPG (Simultaneous Link Weights)
        # ==========================================
        obs_ddpg, _ = env_ddpg.reset(seed=episode)
        
        # INJECT THE MASTER TRAFFIC INTO DDPG
        env_ddpg.flows_traffic = frozen_traffic.copy()
        env_ddpg.total_incoming_network = frozen_total_incoming
        
        # CRITICAL: Recalculate DDPG's initial ATVM state so it isn't blind
        for u, v, data in env_ddpg.model.G.edges(data=True):
            data['weight'] = 1.0
        all_paths = dict(nx.all_pairs_dijkstra_path(env_ddpg.model.G, weight="weight"))
        env_ddpg.flows_paths = { (src, dst): all_paths[src][dst] for (src, dst), traffic in env_ddpg.flows_traffic.items() }
        _, _, switch_AVTM_matrix = env_ddpg.model.calculate_measurements(env_ddpg.flows_traffic, env_ddpg.flows_paths)
        obs_ddpg = switch_AVTM_matrix.flatten()
        
        start_time = time.time()
        
        # DDPG plays the game in 1 single step
        action, _states = ddpg_model.predict(obs_ddpg, deterministic=True)
        _, reward_ddpg, terminated, truncated, info_ddpg = env_ddpg.step(action)
        
        ddpg_time = time.time() - start_time
        
        results["DDPG"]["reward"].append(reward_ddpg)
        results["DDPG"]["delay"].append(info_ddpg['avg_delay'])
        results["DDPG"]["packet_loss"].append(info_ddpg['packet_loss'])
        results["DDPG"]["time"].append(ddpg_time)


        # ==========================================
        # TEST 3: Naive Shortest Path (Baseline ECMP)
        # ==========================================
        env_ppo.reset(seed=episode)
        
        # INJECT THE MASTER TRAFFIC
        env_ppo.flows_traffic = frozen_traffic.copy()
        env_ppo.SDM = frozen_sdm.copy()
        env_ppo.active_flow_keys = frozen_active_keys.copy()
        env_ppo.total_incoming_network = frozen_total_incoming
        
        start_time = time.time()
        done = False
        
        while not done:
            # Action 0 is hardcoded in your PPO environment to be the absolute shortest path
            action = 0 
            _, reward_naive, terminated, truncated, info_naive = env_ppo.step(action)
            done = terminated or truncated

        naive_time = time.time() - start_time

        results["Naive"]["reward"].append(reward_naive)
        results["Naive"]["delay"].append(info_naive['avg_delay'])
        results["Naive"]["packet_loss"].append(info_naive['packet_loss'])
        results["Naive"]["time"].append(naive_time)

        # Print live progress
        print(f"Episode {episode + 1:02d}/{num_episodes} | Naive: {reward_naive:.3f} | DDPG: {reward_ddpg:.3f} | PPO: {reward_ppo:.3f}")

    # ==========================================
    # PRINT FINAL RESULTS FOR THESIS GRAPHS
    # ==========================================
    print("\n" + "="*50)
    print("FINAL AVERAGED RESULTS (Over 50 Random Topologies)")
    print("="*50)
    
    for algo in ["Naive", "DDPG", "PPO"]: 
        avg_rew = np.mean(results[algo]["reward"])
        avg_delay = np.mean(results[algo]["delay"])
        avg_loss = np.mean(results[algo]["packet_loss"])
        avg_time = np.mean(results[algo]["time"]) * 1000 
        
        print(f"Algorithm: {algo}")
        print(f"  Average Reward:      {avg_rew:.4f}")
        print(f"  Average Delay:       {avg_delay:.4f}")
        print(f"  Average Packet Loss: {avg_loss:.4f}")
        print(f"  Avg Inference Time:  {avg_time:.2f} ms")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_algorithms(num_episodes=10)