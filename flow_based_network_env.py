from utils import json2networkx
from switch import Switch
import matplotlib.pyplot as plt
import networkx as nx
import copy
from traffic_leaving_mm1k import traffic_leaving_mm1k
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import islice
from concurrent.futures import ThreadPoolExecutor


class NetworkEnv(gym.Env):
    """Custom Environment that follows gymnasium interface for SDN DRL Routing"""
    metadata = {'render_modes': ['console']}

    def __init__(self):
        super(NetworkEnv, self).__init__()
        
        # Load topology
        self.G = json2networkx("topologies/mesh5x5.json")
        self.switches = [(node, attr) for node, attr in self.G.nodes(data=True) if isinstance(attr.get('data'), Switch)]
        self.no_of_switches = len(self.switches)
        self.edges_list = list(self.G.edges())
        self.no_of_edges = len(self.edges_list)
        self.switches_delay = {}

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
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.no_of_switches * self.no_of_switches,), dtype=np.float32)

        self.max_steps = 100
        self.current_step = 0

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
        for u, v, data in self.G.edges(data=True):
            data['flows'] = {}

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

        mm1k_stabilization_rounds = 18
        for round in range(mm1k_stabilization_rounds):
            self.temp_flows_on_edges = {}

            with ThreadPoolExecutor(max_workers=8) as executor:
                for switch, attributes in self.switches:
                    executor.submit(self._calculate_temp_flows_on_edges, switch, attributes)

            for (u, v), _flows in self.temp_flows_on_edges.items():
                for _flow_id, _traffic in _flows.items():
                    self.G[u][v]['flows'][_flow_id] = _traffic

        # Calculating self.switches_delay
        for switch, attributes in self.switches:
            total_incoming = 0
            for u, v, data in self.G.in_edges(switch, data=True):
                total_incoming += sum(data.get('flows', {}).values())

            if total_incoming == 0:
                self.switches_delay[switch] = 0
                continue

            total_outgoing = 0
            for u, v, data in self.G.out_edges(switch, data=True):
                total_outgoing += sum(data.get('flows', {}).values())

            switch_obj = self.G.nodes[switch]["data"]

            service_rate = switch_obj.service_rate
            queue_capacity = switch_obj.queue_capacity

            ro = total_incoming / service_rate

            if ro == 1:
                ro = 0.999

            if ro < 1.0:
                term1 = ro / (1 - ro)
                term2 = ((queue_capacity + 1) * (ro ** (queue_capacity + 1))) / (1 - (ro ** (queue_capacity + 1)))
            elif ro > 1.0: 
                term1 = ro / (1 - ro)
                term2 = (queue_capacity + 1) / ((ro ** -(queue_capacity + 1)) - 1)

            L_system = term1 - term2

            exp_delay_at_switch = L_system / total_outgoing

            self.switches_delay[switch] = exp_delay_at_switch

        flow_delays = {}
        total_incoming_network = 0
        for flow_data in self.flows:
            total_incoming_network += flow_data['traffic']
            flow_path = self.flows_paths[flow_data['id']] 

            total_delay = 0
            for switch in flow_path[1:-1]:
                total_delay += self.switches_delay.get(switch, 0)

            flow_delays[flow_data['id']] = total_delay

        avg_flow_delay = sum(flow_delays.values()) / len(flow_delays)

        total_packet_loss = 0
        for switch, attributes in self.switches:
            total_incoming = 0
            for u, v, data in self.G.in_edges(switch, data=True):
                total_incoming += sum(data.get('flows', {}).values())

            total_outgoing = 0
            for u, v, data in self.G.out_edges(switch, data=True):
                total_outgoing += sum(data.get('flows', {}).values())

            packet_loss = total_incoming - total_outgoing
            total_packet_loss += packet_loss

        # 4. Calculate Rewards based on the paper's formulas
        # Max theoretical delay = sum of (K_n / mu_n) over max path
        max_possible_delay = self.max_hops * (self.K_max / self.mu_max)

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
                    # Normalize by mu_max
                    normalized_flow = min(1.0, sum_flows / self.mu_max)
                    AVTM_matrix[src_idx][dst_idx] = normalized_flow

        return AVTM_matrix.flatten()
    
    def _calculate_temp_flows_on_edges(self, switch, attributes):
            incoming_flows = {}

            for u, v, data in self.G.in_edges(switch, data=True):
                flows_on_edge = data.get('flows', {})

                if flows_on_edge:
                    incoming_flows[(u, v)] = flows_on_edge

            leaving_flows_values = traffic_leaving_mm1k(incoming_flows, attributes["data"].service_rate, attributes["data"].queue_capacity)

            for incoming_edge, flows_dict in leaving_flows_values.items():
                # incoming_edge to (poprzedni_węzeł, obecny_switch)
                for flow_id, new_traffic_amount in flows_dict.items():
                    flow_path = self.flows_paths[flow_id]

                    curr_index = flow_path.index(switch)

                    next_hop = flow_path[curr_index + 1]

                    if (switch, next_hop) not in self.temp_flows_on_edges:
                        self.temp_flows_on_edges[(switch, next_hop)] = {}

                    # Nadpisujemy wartość przepływu nową, mniejszą wartością obliczoną przez MM1K
                    self.temp_flows_on_edges[(switch, next_hop)][flow_id] = new_traffic_amount

