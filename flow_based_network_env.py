from utils import json2networkx
from switch import Switch
import networkx as nx
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from NetworkModel import NetworkModel
import itertools

class FlowBasedNetworkEnv(gym.Env):
    """Custom Environment for SDN DRL Routing using K-Shortest Paths and SDM"""
    metadata = {'render_modes': ['console']}

    def __init__(self):
        super(FlowBasedNetworkEnv, self).__init__()

        base_dir = Path(__file__).resolve().parent
        topology_path = base_dir / "topologies" / "mesh5x5.json"

        self.G = json2networkx(topology_path)
        self.model = NetworkModel(self.G)
                
        # Hyperparameters from the paper
        self.alpha = 0.9  
        self.mu_max = 3000.0  
        self.K_max = 10000.0  
        self.max_hops = 25 
        
        self.no_of_flows = 150
        self.k_paths = 7  # The AI can choose between the 3 shortest paths for any flow

        # --- PRE-COMPUTE K-SHORTEST PATHS ---
        # We do this once during startup so the training loop runs lightning fast
        print(f"Pre-computing {self.k_paths}-shortest paths for all valid pairs...")
        self.all_k_paths = {}
        nodes = list(self.G.nodes())
        for src in nodes:
            for dst in nodes:
                if src != dst:
                    self.all_k_paths[(src, dst)] = list(itertools.islice(nx.shortest_simple_paths(self.G, src, dst), self.k_paths))
        print("Path pre-computation complete!")

        # --- THE NEW ACTION SPACE ---
        # 625 discrete path choices (mapped from continuous [0.0, 0.999])
        # Index i = (src_idx * 25) + dst_idx
        self.action_space = spaces.MultiDiscrete([self.k_paths] * 625)
        
        # --- THE NEW OBSERVATION SPACE ---
        # State: ATVM (625) + SDM/Traffic Demand Matrix (625) = 1250 total inputs
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1250,), dtype=np.float32)

        self.max_steps = 1 # 10 steps per episode to let the gradients stabilize
        self.current_step = 0
        self.max_possible_delay = self.max_hops * (self.K_max / self.mu_max)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        super().reset(seed=seed)
        random.seed(seed)

        self.flows_traffic = {}
        self.flows_paths = {}
        self.total_incoming_network = 0
        
        # Initialize an empty Source Destination Matrix (SDM)
        self.SDM = np.zeros((25, 25), dtype=np.float32)
        
        # Generate random traffic
        for _ in range(self.no_of_flows):
            random_hosts = random.sample(range(0, 25), 2)
            traffic_rate = random.uniform(10, 300)
            self.total_incoming_network += traffic_rate
            
            src_idx = random_hosts[0]
            dst_idx = random_hosts[1]
            
            flow_key = (f"10.0.1.{src_idx}", f"10.0.1.{dst_idx}")
            self.flows_traffic[flow_key] = traffic_rate
            
            # Place traffic in the matrix and normalize it (max generated is 300)
            self.SDM[src_idx, dst_idx] = min(1.0, traffic_rate / 300.0)

            self.flows_paths[flow_key] = self.all_k_paths[flow_key][0]

        # Calculate initial measurements
        _, _, switch_AVTM_matrix = self.model.calculate_measurements(self.flows_traffic, self.flows_paths)
            
        state = np.concatenate((switch_AVTM_matrix.flatten(), self.SDM.flatten()))
        
        return state, {}

    def step(self, action):
        self.flows_paths = {}
        
        # Map the 625 actions ONLY to the active flows
        for (src_ip, dst_ip), _ in self.flows_traffic.items():
            # Extract raw integer indices (0 to 24)
            src_idx = int(src_ip.split('.')[-1]) 
            dst_idx = int(dst_ip.split('.')[-1])
            
            # Find the EXACT index in the 625-length action array
            action_idx = (src_idx * 25) + dst_idx
            
            # Grab the AI's continuous choice [0.0, 0.999]
            path_idx = action[action_idx]
                        
            self.flows_paths[(src_ip, dst_ip)] = self.all_k_paths[(src_ip, dst_ip)][path_idx]

        # Calculate physics for the chosen paths
        avg_delay, total_packet_loss, switch_AVTM_matrix = self.model.calculate_measurements(self.flows_traffic, self.flows_paths)
        
        # Calculate Reward
        r_d = 1.0 - min(avg_delay / self.max_possible_delay, 1.0)
        r_p = 1.0 - min(total_packet_loss / self.total_incoming_network, 1.0) if self.total_incoming_network > 0 else 1.0
        reward = self.alpha * r_d + (1 - self.alpha) * r_p
        
        # Build Next State
        next_state = np.concatenate((switch_AVTM_matrix.flatten(), self.SDM.flatten()))
        
        self.current_step += 1
        terminated = False 
        truncated = self.current_step >= self.max_steps 
        info = {'avg_delay': avg_delay, 'packet_loss': total_packet_loss, 'flows_paths': self.flows_paths}
        
        return next_state, reward, terminated, truncated, info