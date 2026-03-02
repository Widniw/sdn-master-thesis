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
        self.alpha = 0.1  # Weight factor for delay vs packet loss 
        self.mu_max = 3000.0  # Max service rate [cite: 441]
        self.K_max = 10000.0  # Max queue capacity [cite: 441]
        self.max_hops = 25 # Absolute worst-case path length for scaling delay
        
        # Actions: Link weights bounded between 1 and 5 [cite: 426, 444]
        self.action_space = spaces.Box(low=1.0, high=5.0, shape=(self.no_of_edges,), dtype=np.float32)
        
        # State: ATVM (Aggregated Traffic Volume Matrix) normalized [0, 1] 
        # Shape is (25, 25) flattened to 1D for the neural network
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.no_of_switches * self.no_of_switches,), dtype=np.float32)

        self.max_steps = 100 
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset the step counter
        self.current_step = 0

        super().reset(seed=seed)
        
        self.flows = {}
        
        # Generate 150 random flows with demand between 10 and 300 [cite: 442]
        for _ in range(150):
            random_hosts = random.sample(range(1, 26), 2)
            traffic_rate = random.randint(10, 300)
            self.flows[(f"10.0.0.{random_hosts[0]}", f"10.0.0.{random_hosts[1]}")] = traffic_rate
            
        # Initialize default weights to 1
        for u, v, data in self.G.edges(data=True):
            data['weight'] = 1.0
            
        # Calculate initial ATVM state
        state = self._calculate_state()
        return state, {}

    def step(self, action):
        # 1. Apply new link weights from the RL agent
        for i, (u, v) in enumerate(self.edges_list):
            self.G[u][v]['weight'] = action[i]
            
        # Clear previous flows on edges
        for u, v, data in self.G.edges(data=True):
            data['flows'] = {}
            
        flows_paths = {}

        # 2. Route traffic using Dijkstra
        for flow_name, traffic in self.flows.items():
            path = nx.dijkstra_path(self.G, source=flow_name[0], target=flow_name[1], weight="weight")

            flows_paths[flow_name] = path

            for u, v in zip(path, path[1:]):
                self.G[u][v]['flows'][flow_name] = traffic
                
        for i in range(18):
            temp_flows_on_edges = {}

            for switch, attributes in self.switches:
                incoming_flows = {}

                for u, v, data in self.G.in_edges(switch, data=True):
                    flows_on_edge = data.get('flows', {})
                    
                    if flows_on_edge:
                        incoming_flows[(u, v)] = flows_on_edge

                leaving_flows_values = traffic_leaving_mm1k(incoming_flows, attributes["data"].service_rate, attributes["data"].queue_capacity)

                for incoming_edge, flows_dict in leaving_flows_values.items():
                    # incoming_edge to (poprzedni_węzeł, obecny_switch)
                    
                    for flow_id, new_traffic_amount in flows_dict.items():
                        # flow_id to np. ("10.0.0.1", "10.0.0.2")
                        
                            current_path = flows_paths[flow_id]
                            
                            # Znajdujemy indeks obecnego switcha na ścieżce przepływu
                            curr_index = current_path.index(switch)
                            
                            # Sprawdzamy, czy to nie jest koniec ścieżki
                            next_hop = current_path[curr_index + 1]

                            if (switch, next_hop) not in temp_flows_on_edges:
                                temp_flows_on_edges[(switch, next_hop)] = {}          
                                        
                            # Nadpisujemy wartość przepływu nową, mniejszą wartością obliczoną przez MM1K
                            temp_flows_on_edges[(switch, next_hop)][flow_id] = new_traffic_amount
    
            for (u, v), _flows in temp_flows_on_edges.items():
                for _flow_id, _traffic in _flows.items():
                    self.G[u][v]['flows'][_flow_id] = _traffic

        self.switches_delay = {}

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
                # Wzór standardowy dla rho < 1
                term1 = ro / (1 - ro)
                term2 = ((queue_capacity + 1) * (ro ** (queue_capacity + 1))) / (1 - (ro ** (queue_capacity + 1)))
                        
            else: # ro > 1
                # Wzór przekształcony dla rho > 1 (korzysta z ujemnych potęg, by uniknąć nieskończoności)
                # Term1: ro / (1 - ro) jest bezpieczne (daje ujemną liczbę)
                term1 = ro / (1 - ro)
                        
                # Term2: Dzielimy licznik i mianownik przez ro^(K+1)
                # Otrzymujemy: (K+1) / (ro^(-K-1) - 1)
                term2 = (queue_capacity + 1) / ((ro ** -(queue_capacity + 1)) - 1)
            
            L_system = term1 - term2

            exp_delay_at_switch = L_system / total_outgoing

            self.switches_delay[switch] = exp_delay_at_switch
        
        delays = []
        total_incoming_network = 0
        total_packet_loss = 0

        for flow_name, traffic in self.flows.items():
            total_incoming_network += traffic

            # print(f"For flow {flow_name} with traffic volume: {traffic}")
            dijkstra_path = nx.dijkstra_path(self.G, source = flow_name[0], target = flow_name[1], weight = "weight")
            # print(f"{dijkstra_path = }")

            total_delay = 0

            for switch in dijkstra_path[1:-1]:
                total_delay += self.switches_delay[switch]
            
            # print(f"{round(total_delay,2) = }s")

            delays.append(total_delay)

        avg_delay = sum(delays) / len(delays)

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
        # avg_delay = sum(delays) / len(delays) if delays else 0
        # total_packet_loss = total_incoming_network - total_outgoing_network
        
        # Max theoretical delay = sum of (K_n / mu_n) over max path 
        max_possible_delay = self.max_hops * (self.K_max / self.mu_max)
        
        # rd(t) and rp(t) formulas [cite: 296, 330]
        r_d = 1.0 - min(avg_delay / max_possible_delay, 1.0)
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