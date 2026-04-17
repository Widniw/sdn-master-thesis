from switch import Switch
import numpy as np
from pathlib import Path
from utils import json2networkx
import random
import networkx as nx

class NetworkModel:
    def __init__(self, G):
        self.G = G
        self.switches = [(node, attr) for node, attr in self.G.nodes(data=True) if isinstance(attr.get('data'), Switch)]
        self.no_of_nodes = len(self.G.nodes)
        self.no_of_switches = len(self.switches)
        self.edges = [edge for edge in self.G.edges()]
        self.no_of_edges = len(self.edges)
        self.mu_max = 3000

        self.node_to_index = {}
        for i, node in enumerate(self.G.nodes()):
            self.node_to_index[node] = i
    
    def calculate_measurements(self, flows_traffic, flows_paths):
        self.flows_traffic = flows_traffic
        self.flows_paths = flows_paths

        self.ziped_flow_paths = {}

        for flow, path in self.flows_paths.items():
            zipped_path = []
            for u, v in zip(path, path[1:]):
                zipped_path.append((u,v))
            self.ziped_flow_paths[flow] = zipped_path

        no_of_flows = len(self.flows_paths)
        self.flow_to_index = {}
        for i, flow in enumerate(self.flows_traffic):
            self.flow_to_index[flow] = i
        
        self.full_AVTM_matrix = np.zeros((self.no_of_nodes, self.no_of_nodes, no_of_flows), dtype=np.float32)

        for flow_name, traffic in self.flows_traffic.items():
            flow_idx = self.flow_to_index[flow_name]

            for u, v in self.ziped_flow_paths[flow_name]:
                u_idx = self.node_to_index[u]
                v_idx = self.node_to_index[v]

                self.full_AVTM_matrix[u_idx, v_idx, flow_idx] = traffic
        
        for i in range(18):
            temp_full_AVTM_matrix = self.full_AVTM_matrix.copy()

            switch_leaving_prob = {}

            for switch, attributes in self.switches:
                v_idx = self.node_to_index[switch]

                total_incoming = self.full_AVTM_matrix[:, v_idx, :].sum()

                service_rate = attributes["data"].service_rate
                queue_capacity = attributes["data"].queue_capacity
                ro = total_incoming / service_rate

                if ro == 1:
                    ro = 0.999

                if ro < 1.0:
                    leaving_probability = (1 - ro**queue_capacity) / (1 - ro**(queue_capacity + 1))
                        
                else:
                    numerator = (1.0 / ro) - (ro ** -(queue_capacity + 1))
                    denominator = 1.0 - (ro ** -(queue_capacity + 1))
                    leaving_probability = numerator / denominator
                
                switch_leaving_prob[switch] = leaving_probability
            
            for flow, zipped_path in self.ziped_flow_paths.items():
                last_index = len(zipped_path) - 1

                for (u, v) in zipped_path:
                    curr_index = zipped_path.index((u, v))

                    if curr_index == last_index:
                        continue

                    u_idx = self.node_to_index[u]
                    v_idx = self.node_to_index[v]
                    flow_idx = self.flow_to_index[flow]

                    traffic = self.full_AVTM_matrix[u_idx, v_idx, flow_idx]

                    outgoing_traffic = traffic * switch_leaving_prob[v]

                    next_edge_index = curr_index + 1

                    next_u, next_v = zipped_path[next_edge_index]
                    next_u_idx = self.node_to_index[next_u]
                    next_v_idx = self.node_to_index[next_v]

                    temp_full_AVTM_matrix[next_u_idx, next_v_idx, flow_idx] = outgoing_traffic
                
            
            self.full_AVTM_matrix = temp_full_AVTM_matrix.copy()

        delay_at_switch = {}

        for switch, attributes in self.switches:
                v_idx = self.node_to_index[switch]

                total_incoming = self.full_AVTM_matrix[:, v_idx, :].sum()

                if total_incoming == 0:
                    delay_at_switch[switch] = 0
                    continue

                total_outgoing = self.full_AVTM_matrix[v_idx, :, :].sum()

                service_rate = attributes["data"].service_rate
                queue_capacity = attributes["data"].queue_capacity
                ro = total_incoming / service_rate        

                if ro == 1:
                    L_system = queue_capacity / total_outgoing
                elif ro < 1.0:
                    term1 = ro / (1 - ro)
                    term2 = ((queue_capacity + 1) * (ro ** (queue_capacity + 1))) / (1 - (ro ** (queue_capacity + 1)))    
                else: # ro > 1
                    term1 = ro / (1 - ro)
                    term2 = (queue_capacity + 1) / ((ro ** -(queue_capacity + 1)) - 1)
                
                L_system = term1 - term2

                exp_delay_at_switch = L_system / total_outgoing

                delay_at_switch[switch] = exp_delay_at_switch                    

        delays = np.zeros(no_of_flows)
        total_incoming_network = 0

        for i, ((src, dst, _), traffic) in enumerate(self.flows_traffic.items()):
            total_incoming_network += traffic

            path = self.flows_paths[(src, dst, _)]

            total_delay = 0

            for switch in path[1:-1]:
                total_delay += delay_at_switch[switch]

            delays[i] = total_delay

        avg_delay = np.mean(delays) if len(delays) > 0 else 0.0

        total_packet_loss = 0

        for switch, attributes in self.switches:
            v_idx = self.node_to_index[switch]
            total_incoming = self.full_AVTM_matrix[:, v_idx, :].sum()
            total_outgoing = self.full_AVTM_matrix[v_idx, :, :].sum()

            packet_loss = total_incoming - total_outgoing
            total_packet_loss += packet_loss
    
        switch_AVTM_matrix = np.zeros((self.no_of_switches, self.no_of_switches), dtype=np.float32)

        for src_switch, attributes in self.switches:
            for dst_switch, attributes in self.switches:
                u_idx = self.node_to_index[src_switch]
                v_idx = self.node_to_index[dst_switch]

                raw_traffic = self.full_AVTM_matrix[u_idx, v_idx, :].sum()

                # 2. IMPLEMENT EQUATION 10: Normalize and bound to [0, 1]
                normalized_traffic = min(1.0, raw_traffic / self.mu_max)

                # 3. Store in the final ATVM matrix
                switch_AVTM_matrix[int(src_switch)][int(dst_switch)] = normalized_traffic

        return avg_delay, total_packet_loss, switch_AVTM_matrix

    
    def calculate_lightweight_avtm(self, flows_traffic, flows_paths):
        switch_AVTM_matrix = np.zeros((self.no_of_switches, self.no_of_switches), dtype=np.float32)
        
        for flow_key, traffic in flows_traffic.items():
            path = flows_paths[flow_key]
            for u, v in zip(path, path[1:]):
                # Skip edges that involve host nodes
                if u not in self.switches or v not in self.switches:
                    continue
                u_idx = self.node_to_index[u]
                v_idx = self.node_to_index[v]
                switch_AVTM_matrix[u_idx][v_idx] += traffic / self.mu_max
                switch_AVTM_matrix[u_idx][v_idx] = min(1.0, switch_AVTM_matrix[u_idx][v_idx])
        
        return switch_AVTM_matrix
                
if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    topology_path = base_dir / "topologies" / "mesh5x5.json"

    G = json2networkx(topology_path)
    model = NetworkModel(G)

    # weights = [1.       , 5.       , 1.       , 1.       , 1.       , 1.       ,
    #    1.       , 1.       , 1.       , 1.       , 5.       , 5.       ,
    #    1.       , 1.       , 5.       , 1.       , 5.       , 1.       ,
    #    1.       , 5.       , 5.       , 1.       , 1.       , 5.       ,
    #    1.2316306, 5.       , 5.       , 1.2761637, 1.       , 1.       ,
    #    5.       , 5.       , 1.       , 1.       , 1.       , 5.       ,
    #    1.       , 5.       , 5.       , 5.       , 1.       , 5.       ,
    #    5.       , 5.       , 5.       , 5.       , 5.       , 1.       ,
    #    1.       , 2.7109962, 1.       , 5.       , 1.       , 1.0602391,
    #    1.2330025, 5.       , 1.       , 1.       , 1.       , 5.       ,
    #    5.       , 5.       , 1.       , 5.       , 5.       , 5.       ,
    #    1.       , 5.       , 5.       , 5.       , 5.       , 5.       ,
    #    5.       , 5.       , 1.       , 1.       , 1.       , 1.       ,
    #    5.       , 1.       , 5.       , 1.       , 5.       , 5.       ,
    #    1.       , 5.       , 1.       , 5.       , 1.       , 5.       ,
    #    5.       , 5.       , 5.       , 5.       , 1.       , 1.       ,
    #    5.       , 1.       , 5.       , 1.       , 5.       , 5.       ,
    #    1.       , 1.       , 1.       , 1.       , 5.       , 1.       ,
    #    5.       , 1.       , 1.       , 1.       , 5.       , 1.       ,
    #    1.       , 5.       , 1.       , 3.6593628, 1.       , 1.       ,
    #    5.       , 1.       , 1.       , 1.       , 1.       , 1.       ,
    #    1.       , 5.       , 5.       , 5.]
    
    # for i, (u, v, attr) in enumerate(G.edges(data = True)):
    #     attr['weight'] = weights[i]

    alpha = 0.9
    mu_max = 3000
    K_max = 10000
    max_hops = 25
    seed = 412158

    random.seed(seed)
    
    flows_traffic = {}
    no_of_flows = 150
    total_incoming_network = 0
    for flow in range(no_of_flows):
        random_hosts = random.sample(range(0, 25), 2)
        random_traffic_rate = random.uniform(10, 300)
        total_incoming_network += random_traffic_rate
        flows_traffic[(f"10.0.1.{random_hosts[0]}",f"10.0.1.{random_hosts[1]}")] = random_traffic_rate

    # flows_traffic = {("10.0.1.0", "10.0.1.1"): 3,
    #          ("10.0.1.1", "10.0.1.0"): 2} 

    all_paths = dict(nx.all_pairs_dijkstra_path(model.G, weight="weight"))

    flows_paths = {}
    for (src, dst), traffic in flows_traffic.items():
        path = all_paths[src][dst] 
        flows_paths[(src, dst)] = path

    
    avg_delay, total_packet_loss, switch_AVTM_matrix = model.calculate_measurements(flows_traffic, flows_paths)
    print(f"{avg_delay =}")
    print(f"{total_packet_loss = }")
    print(f"{switch_AVTM_matrix = }")

    max_possible_delay = max_hops * (K_max / mu_max)
    
    # rd(t) and rp(t) formulas [cite: 296, 330]
    r_d = 1.0 - min(avg_delay / max_possible_delay, 1.0)
    r_p = 1.0 - min(total_packet_loss / total_incoming_network, 1.0) if total_incoming_network > 0 else 1.0
    
    # Total Reward R(st, at) 
    reward = alpha * r_d + (1 - alpha) * r_p
    print(f"{reward = }")
