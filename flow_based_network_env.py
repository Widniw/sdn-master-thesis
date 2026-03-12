from utils import json2networkx
import networkx as nx
from traffic_leaving_mm1k import traffic_leaving_mm1k
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import islice
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from NetworkModel import NetworkModel


class NetworkEnv(gym.Env):
    """Custom Environment that follows gymnasium interface for SDN DRL Routing"""
    metadata = {'render_modes': ['console']}

    def __init__(self):
        super(NetworkEnv, self).__init__()
        
        # Load topology
        base_dir = Path(__file__).resolve().parent
        topology_path = base_dir / "topologies" / "mesh5x5.json"

        G = json2networkx(topology_path)
        self.model = NetworkModel(G)

        # Hyperparameters from the paper
        self.alpha = 0.9 # Weight factor for delay vs packet loss
        self.mu_max = 3000.0 # Max service rate
        self.K_max = 10000.0 # Max queue capacity
        self.max_hops = 25 # Absolute worst-case path length for scaling delay

        self.no_of_flows = 150
        self.k_paths = 2

        self.action_space = spaces.MultiDiscrete([self.k_paths] * self.no_of_flows)

        # State: ATVM (Aggregated Traffic Volume Matrix) normalized [0, 1]
        # Shape is (25, 25) flattened to 1D for the neural network
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.model.no_of_switches * self.model.no_of_switches,), dtype=np.float32)

        self.max_steps = 10
        self.current_step = 0

        self.max_possible_delay = self.max_hops * (self.K_max / self.mu_max)

        print("Pre-calculating K-Shortest Paths for all host pairs... (This takes a few seconds)")
        self.all_k_paths = {}
        for src in range(1, 26):
            for dst in range(1, 26):
                if src != dst:
                    h_src = f"10.0.0.{src}"
                    h_dst = f"10.0.0.{dst}"
                    # Find the top K paths and store them
                    paths = list(islice(nx.shortest_simple_paths(self.G, h_src, h_dst), self.k_paths))
                    self.all_k_paths[(h_src, h_dst)] = paths
        print("Paths pre-calculated!")

    def reset(self, seed=None, options=None):
        # Reset the step counter
        self.current_step = 0

        super().reset(seed=seed)

        self.flows = [] 
        
        for i in range(self.no_of_flows):
            random_hosts = random.sample(range(1, 26), 2)
            traffic_rate = random.randint(10, 300)
            
            self.flows.append({
                'id': f"flow_{i}", # Unique ID in case two flows share the same src/dst
                'src': f"10.0.0.{random_hosts[0]}",
                'dst': f"10.0.0.{random_hosts[1]}",
                'traffic': traffic_rate
            })
            
        state = self._calculate_state()
        return state, {}

    def step(self, action):
        self.flows_paths = {}

        # 1. Apply the paths chosen by the RL agent!
        for i, flow_data in enumerate(self.flows):
            flow_id = flow_data['id']
            src = flow_data['src']
            dst = flow_data['dst']
            traffic = flow_data['traffic']
            
            path_choice = action[i] 
                            
            # Fetch the chosen path
            chosen_path = self.all_k_paths[(src, dst)][path_choice]
            self.flows_paths[flow_id] = chosen_path

            # Load the traffic onto the chosen path
            for u, v in zip(chosen_path, chosen_path[1:]):
                self.G[u][v]['flows'][flow_id] = traffic

        # rd(t) and rp(t) formulas
        r_d = 1.0 - min(avg_flow_delay / max_possible_delay, 1.0)
        r_p = 1.0 - min(total_packet_loss / total_incoming_network, 1.0) if total_incoming_network > 0 else 1.0

        # Total Reward R(st, at)
        reward = self.alpha * r_d + (1 - self.alpha) * r_p

        # 5. Get Next State (ATVM)
        next_state = self._calculate_state()

        # Environment step logic
        # Replace the old terminated/truncated lines with this:
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps # Ends the episode after 100 steps
        info = {'avg_flow_delay': avg_flow_delay, 'total_packet_loss': total_packet_loss}

        return next_state, reward, terminated, truncated, info
