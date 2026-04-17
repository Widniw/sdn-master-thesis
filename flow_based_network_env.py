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

        self.action_space = spaces.Discrete(self.k_paths)
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1253,), dtype=np.float32)

        self.current_flow_idx = 0

        self.max_possible_delay = self.max_hops * (self.K_max / self.mu_max)
        self.active_flow_keys = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)

        self.current_flow_idx = 0
        self.flows_traffic = {}
        self.flows_paths = {} 
        self.active_flow_keys = []
        self.total_incoming_network = 0
        self.SDM = np.zeros((25, 25), dtype=np.float32)
        
        # 1. Generate all 150 flows up front
        for i in range(self.no_of_flows):
            random_hosts = random.sample(range(0, 25), 2)
            traffic_rate = random.uniform(10, 300)
            self.total_incoming_network += traffic_rate
            
            src_idx, dst_idx = random_hosts[0], random_hosts[1]
            flow_key = (f"10.0.1.{src_idx}", f"10.0.1.{dst_idx}", i)  # i makes it unique
            
            self.flows_traffic[flow_key] = traffic_rate
            self.active_flow_keys.append(flow_key)
            self.SDM[src_idx, dst_idx] = min(1.0, traffic_rate / 300.0)

        # 2. Get the blank starting network state
        _, _, switch_AVTM_matrix = self.model.calculate_measurements({}, {})
        
        # 3. Build the state for Flow #1
        state = self._build_state(switch_AVTM_matrix)
        return state, {}

    def step(self, action):
        current_flow = self.active_flow_keys[self.current_flow_idx]
        src, dst, _ = current_flow
        self.flows_paths[current_flow] = self.all_k_paths[(src, dst)][action]
        
        self.current_flow_idx += 1
        
        terminated = self.current_flow_idx >= self.no_of_flows
        truncated = False
        info = {}

        if terminated:
            avg_delay, total_packet_loss, switch_AVTM_matrix = self.model.calculate_measurements(self.flows_traffic, self.flows_paths)
            
            r_d = 1.0 - min(avg_delay / self.max_possible_delay, 1.0)
            r_p = 1.0 - min(total_packet_loss / self.total_incoming_network, 1.0) if self.total_incoming_network > 0 else 1.0
            reward = self.alpha * r_d + (1 - self.alpha) * r_p
            
            info = {'avg_delay': avg_delay, 'packet_loss': total_packet_loss}
            
            next_state = np.zeros(1253, dtype=np.float32)
            
        else:
            # THE GAME CONTINUES: Give a reward of 0 and get the next state
            reward = 0.0
            
            # Fast-update the ATVM based on current paths to show the AI the growing congestion
            _, _, switch_AVTM_matrix = self.model.calculate_measurements(
                {k: self.flows_traffic[k] for k in self.active_flow_keys[:self.current_flow_idx]}, 
                self.flows_paths
            )
            next_state = self._build_state(switch_AVTM_matrix)

        return next_state, reward, terminated, truncated, info

    def _build_state(self, avtm):
        """Helper function to create the 1253-length array"""
        current_flow = self.active_flow_keys[self.current_flow_idx]
        src, dst, _ = current_flow
        src_idx = int(current_flow[0].split('.')[-1])
        dst_idx = int(current_flow[1].split('.')[-1])
        traffic_rate = self.flows_traffic[current_flow] / 300.0 # Normalize

        flow_info = np.array([src_idx / 24.0, dst_idx / 24.0, traffic_rate], dtype=np.float32)
        return np.concatenate((avtm.flatten(), self.SDM.flatten(), flow_info))