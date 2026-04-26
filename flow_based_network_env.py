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
        self.k_paths = 3  # The AI can choose between the 3 shortest paths for any flow

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
        self.action_space = spaces.Discrete(n = self.k_paths)
        
        # --- THE NEW OBSERVATION SPACE ---
        # State: ATVM (625) + Source Node (25) + Destination Node (25) = 675
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(675,), dtype=np.float32)

        self.max_possible_delay = self.max_hops * (self.K_max / self.mu_max)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)

        self.flows_traffic = {}
        self.flows_paths = {}
        self.idx_to_flow = {}
        self.total_incoming_network = 0
        self.flow_no = 0
        
        # Generate random traffic
        for i in range(self.no_of_flows):
            src, dst = random.sample(range(0, 25), 2)
            flow_key = (f"10.0.1.{src}", f"10.0.1.{dst}")
            
            if flow_key in self.flows_traffic.keys():
                continue

            self.idx_to_flow[self.flow_no] = flow_key

            self.flow_no += 1

            traffic_rate = random.uniform(10, 300)
            self.total_incoming_network += traffic_rate
                        
            self.flows_traffic[flow_key] = traffic_rate
        
        self.flow_no = 0
        
        # Calculate initial measurements
        switch_AVTM_matrix = np.zeros(625)

        (src_ip, dst_ip) = self.idx_to_flow[self.flow_no] 
        src_idx = int(src_ip.split('.')[-1]) 
        dst_idx = int(dst_ip.split('.')[-1])

        src_state = np.zeros(25)
        src_state[src_idx] = 1

        dst_state = np.zeros(25)
        dst_state[dst_idx] = 1

        state = np.concatenate((switch_AVTM_matrix.flatten(), src_state, dst_state))
        
        return state, {}

    def step(self, action):        
        (src_ip, dst_ip) = self.idx_to_flow[self.flow_no]

        self.flows_paths[(src_ip, dst_ip)] = self.all_k_paths[(src_ip, dst_ip)][action]

        temp_flows_traffic = {}
        for flow in range(self.flow_no + 1):
            flow_key = self.idx_to_flow[flow]
            temp_flows_traffic[flow_key] = self.flows_traffic[flow_key]

        # Calculate physics for the chosen paths
        avg_delay, total_packet_loss, switch_AVTM_matrix = self.model.calculate_measurements(temp_flows_traffic, self.flows_paths)

        src_idx = int(src_ip.split('.')[-1]) 
        dst_idx = int(dst_ip.split('.')[-1])

        src_state = np.zeros(25)
        src_state[src_idx] = 1

        dst_state = np.zeros(25)
        dst_state[dst_idx] = 1

        state = np.concatenate((switch_AVTM_matrix.flatten(), src_state, dst_state))
        info = {'avg_delay': avg_delay, 'packet_loss': total_packet_loss, 'flows_paths': self.flows_paths}
        terminated = False

        self.flow_no += 1

        if self.flow_no >= len(self.flows_traffic.keys()):
        
            # Calculate Reward
            r_d = 1.0 - min(avg_delay / self.max_possible_delay, 1.0)
            r_p = 1.0 - min(total_packet_loss / self.total_incoming_network, 1.0) if self.total_incoming_network > 0 else 1.0
            reward = self.alpha * r_d + (1 - self.alpha) * r_p

            truncated = True
        else:
            reward = 0
            truncated = False


        
        return state, reward, terminated, truncated, info