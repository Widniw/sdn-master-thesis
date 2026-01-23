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
        G.add_node(f"10.0.0.{node}")

        G.add_edge(node, f"10.0.0.{node}", weight = 1, flows = {})
        G.add_edge(f"10.0.0.{node}", node, weight = 1, flows = {})



    return G
