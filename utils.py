import networkx as nx
import json
from switch import Switch

def json2networkx(json_data):
    with open(json_data, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()

    for node, attributes in data.items():
        switch_obj = Switch(service_rate = attributes["service_rate"],
                            queue_capacity = attributes["queue_capacity"],
                            aggr_arrival_rate = 1)
        
        G.add_node(node, data = switch_obj)

        for neighbor in attributes["neighbors"]:
            G.add_edge(node, neighbor, weight = 1, flows = {})
            G.add_edge(neighbor, node, weight = 1, flows = {})

        # Add host x to switch x
        G.add_node(f"10.0.1.{node}")

        G.add_edge(node, f"10.0.1.{node}", weight = 1, flows = {})
        G.add_edge(f"10.0.1.{node}", node, weight = 1, flows = {})



    return G

def traffic_leaving_mm1k(incoming_flows, service_rate, queue_capacity):
    aggregated_traffic = 0
    for edge_flows in incoming_flows.values():
        aggregated_traffic += sum(edge_flows.values())

    # Jeśli nie ma ruchu, zwracamy puste/zerowe flows bez liczenia
    if aggregated_traffic == 0:
        return incoming_flows

    # 2. Obliczenie obciążenia (rho)
    ro = aggregated_traffic / service_rate

    if ro == 1:
        ro = 0.999

    # Unikanie overflow
    if ro < 1.0:
        leaving_probability = (1 - ro**queue_capacity) / (1 - ro**(queue_capacity + 1))
            
    else:
        numerator = (1.0 / ro) - (ro ** -(queue_capacity + 1))
        denominator = 1.0 - (ro ** -(queue_capacity + 1))
        leaving_probability = numerator / denominator    
    
    outgoing_flows = {}
    
    for edge, flows_dict in incoming_flows.items():
        new_flows_dict = {}
        for flow_id, traffic in flows_dict.items():
            new_flows_dict[flow_id] = traffic * leaving_probability
    
        outgoing_flows[edge] = new_flows_dict

    return outgoing_flows