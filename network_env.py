from utils import json2networkx
import networkx as nx
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from NetworkModel import NetworkModel

class NetworkEnv(gym.Env):
    """Custom Environment that follows gymnasium interface for SDN DRL Routing"""
    metadata = {'render_modes': ['console']}

    def __init__(self):
        super(NetworkEnv, self).__init__()

        base_dir = Path(__file__).resolve().parent
        topology_path = base_dir / "topologies" / "mesh5x5.json"

        G = json2networkx(topology_path)
        self.model = NetworkModel(G)
                
        # Hyperparameters from the paper
        self.alpha = 0.9  # Weight factor for delay vs packet loss 
        self.mu_max = 3000.0  # Max service rate [cite: 441]
        self.K_max = 10000.0  # Max queue capacity [cite: 441]
        self.max_hops = 25 # Absolute worst-case path length for scaling delay
        
        # Actions: Link weights bounded between 1 and 5 [cite: 426, 444]
        self.action_space = spaces.Box(low=1.0, high=5.0, shape=(self.model.no_of_edges,), dtype=np.float32)
        
        # State: ATVM (Aggregated Traffic Volume Matrix) normalized [0, 1] 
        # Shape is (25, 25) flattened to 1D for the neural network
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.model.no_of_switches * self.model.no_of_switches,), dtype=np.float32)

        self.max_steps = 1
        self.current_step = 0

        self.max_possible_delay = self.max_hops * (self.K_max / self.mu_max)

    def reset(self, seed=None, options=None):
        # Reset the step counter
        self.current_step = 0

        super().reset(seed=seed)
        
        self.flows_traffic = {}
        no_of_flows = 150
        self.total_incoming_network = 0
        for _ in range(no_of_flows):
            random_hosts = random.sample(range(0, 25), 2)
            traffic_rate = random.uniform(10, 300)
            self.total_incoming_network += traffic_rate
            self.flows_traffic[(f"10.0.1.{random_hosts[0]}", f"10.0.1.{random_hosts[1]}")] = traffic_rate
            
        # Initialize default weights to 1
        for u, v, data in self.model.G.edges(data=True):
            data['weight'] = 1.0

        all_paths = dict(nx.all_pairs_dijkstra_path(self.model.G, weight="weight"))

        self.flows_paths = {}
        for (src, dst), traffic in self.flows_traffic.items():
            path = all_paths[src][dst] 
            self.flows_paths[(src, dst)] = path

        avg_delay, total_packet_loss, switch_AVTM_matrix = self.model.calculate_measurements(self.flows_traffic, self.flows_paths)
            
        # Calculate initial ATVM state
        state = switch_AVTM_matrix.flatten()
        return state, {}

    def step(self, action):
        # 1. Apply new link weights from the RL agent
        for i, (u, v) in enumerate(self.model.edges):
            self.model.G[u][v]['weight'] = action[i]
            
        all_paths = dict(nx.all_pairs_dijkstra_path(self.model.G, weight="weight"))

        self.flows_paths = {}
        for (src, dst), traffic in self.flows_traffic.items():
            path = all_paths[src][dst] 
            self.flows_paths[(src, dst)] = path

        avg_delay, total_packet_loss, switch_AVTM_matrix = self.model.calculate_measurements(self.flows_traffic, self.flows_paths)
        
        # rd(t) and rp(t) formulas [cite: 296, 330]
        r_d = 1.0 - min(avg_delay / self.max_possible_delay, 1.0)
        r_p = 1.0 - min(total_packet_loss / self.total_incoming_network, 1.0) if self.total_incoming_network > 0 else 1.0
        
        # Total Reward R(st, at) 
        reward = self.alpha * r_d + (1 - self.alpha) * r_p
        
        # 5. Get Next State (ATVM)
        next_state = switch_AVTM_matrix.flatten()
        
        # Environment step logic
# Replace the old terminated/truncated lines with this:
        self.current_step += 1
        terminated = False 
        truncated = self.current_step >= self.max_steps # Ends the episode after 100 steps
        info = {'avg_delay': avg_delay, 'packet_loss': total_packet_loss}
        
        return next_state, reward, terminated, truncated, info
