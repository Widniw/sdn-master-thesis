from utils import json2networkx
from switch import Switch
import networkx as nx
from traffic_leaving_mm1k import traffic_leaving_mm1k
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

        self.G = json2networkx(topology_path)
        self.model = NetworkModel(self.G)
                
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

        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset the step counter
        self.current_step = 0

        super().reset(seed=seed)
        
        self.flows_traffic = {}

        temp_random = random.Random(42)
        
        self.total_incoming_network = 0

        # Generate 150 random flows with demand between 10 and 300 [cite: 442]
        for _ in range(150):
            random_hosts = temp_random.sample(range(0, 25), 2)
            traffic_rate = temp_random.randint(10, 300)
            self.flows_traffic[(f"10.0.1.{random_hosts[0]}", f"10.0.1.{random_hosts[1]}")] = traffic_rate

            self.total_incoming_network += traffic_rate
        

            
        # Initialize default weights to 1
        for u, v, data in self.G.edges(data=True):
            data['weight'] = 1.0

        all_paths = dict(nx.all_pairs_dijkstra_path(self.G, weight="weight"))

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
            self.G[u][v]['weight'] = action[i]
            
        all_paths = dict(nx.all_pairs_dijkstra_path(self.G, weight="weight"))

        self.flows_paths = {}
        for (src, dst), traffic in self.flows_traffic.items():
            path = all_paths[src][dst] 
            self.flows_paths[(src, dst)] = path

        avg_delay, total_packet_loss, switch_AVTM_matrix = self.model.calculate_measurements(self.flows_traffic, self.flows_paths)

        max_possible_delay = self.max_hops * (self.K_max / self.mu_max)
        
        # rd(t) and rp(t) formulas [cite: 296, 330]
        r_d = 1.0 - min(avg_delay / max_possible_delay, 1.0)
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

    def _calculate_state(self):
        """Calculates the ATVM and normalizes it to [0,1] """
        AVTM_matrix = np.zeros((self.no_of_switches, self.no_of_switches), dtype=np.float32)
        
        for src_node, _ in self.switches:
            for dst_node, _ in self.switches:
                src_idx = int(src_node) - 1
                dst_idx = int(dst_node) - 1
                
                if self.G.has_edge(src_node, dst_node):
                    edge_flows = self.G[src_node][dst_node].get('flows', {})
                    sum_flows = sum(edge_flows.values())
                    # Normalize by mu_max [cite: 274]
                    normalized_flow = min(1.0, sum_flows / self.mu_max) 
                    AVTM_matrix[src_idx][dst_idx] = normalized_flow
                    
        return AVTM_matrix.flatten()