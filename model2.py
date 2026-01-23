from utils import json2networkx
from switch import Switch
import matplotlib.pyplot as plt
import networkx as nx
import copy
from traffic_leaving_mm1k import traffic_leaving_mm1k


G = json2networkx("topologies/mesh3x3.json")

flows = {("10.0.0.1", "10.0.0.2"): 3,
         ("10.0.0.2", "10.0.0.1"): 2}

for flow_name, traffic in flows.items():
    dijkstra_path = nx.dijkstra_path(G, source = flow_name[0], target = flow_name[1], weight = "weight")

    for u, v in zip(dijkstra_path, dijkstra_path[1:]):
        G[u][v]['flows'][flow_name] = traffic


for i in range(8):

    G_next = copy.deepcopy(G)

    # Create list of switches
    switches = []
    for node, attributes in G.nodes(data=True):
        obj = attributes.get('data')

        if isinstance(obj, Switch):
            switches.append((node, attributes))

    for switch, attributes in switches:
        incoming_flows = {}

        for u, v, data in G.in_edges(switch, data=True):
            flows_on_edge = data.get('flows', {})
            
            if flows_on_edge:
                incoming_flows[(u, v)] = flows_on_edge

        leaving_flows_values = traffic_leaving_mm1k(incoming_flows, attributes["data"].service_rate, attributes["data"].queue_capacity)

        for incoming_edge, flows_dict in leaving_flows_values.items():
            # incoming_edge to (poprzedni_węzeł, obecny_switch)
            
            for flow_id, new_traffic_amount in flows_dict.items():
                # flow_id to np. ("10.0.0.1", "10.0.0.2")
                
                    current_path = nx.dijkstra_path(G, source=flow_id[0], target=flow_id[1], weight="weight")
                    
                    # Znajdujemy indeks obecnego switcha na ścieżce przepływu
                    curr_index = current_path.index(switch)
                    
                    # Sprawdzamy, czy to nie jest koniec ścieżki
                    if curr_index + 1 < len(current_path):
                        next_hop = current_path[curr_index + 1]
                        
                        # 2. Aktualizujemy krawędź WYCHODZĄCĄ w grafie G_next
                        # Krawędź to (switch -> next_hop)
                                                
                        # Nadpisujemy wartość przepływu nową, mniejszą wartością obliczoną przez MM1K
                        G_next[switch][next_hop]['flows'][flow_id] = new_traffic_amount

        
    G = G_next

# Print DiGraph for debbuging purposes
print("G as text:")
for node in G.nodes(data=True):
    node_id, node_attr = node
    print(f"Node {node_id}: {node_attr}")
    for neighbor, edge_attr in G[node_id].items():
        print(f"  -> {neighbor}: {edge_attr}")


### CALCULATING DELAY AND PACKET LOSS FOR EACH FLOW

for flow_name, traffic in flows.items():
    print(f"For flow {flow_name} with traffic volume: {traffic}")
    dijkstra_path = nx.dijkstra_path(G, source = flow_name[0], target = flow_name[1], weight = "weight")
    print(f"{dijkstra_path = }")

    total_delay = 0

    for switch in dijkstra_path[1:-1]:

        switch_obj = G.nodes[switch]["data"]

        service_rate = switch_obj.service_rate
        queue_capacity = switch_obj.queue_capacity

        total_incoming = 0
        for u, v, data in G.in_edges(switch, data=True):
            total_incoming += sum(data.get('flows', {}).values())

        total_outgoing = 0
        for u, v, data in G.out_edges(switch, data=True):
            total_outgoing += sum(data.get('flows', {}).values())
        
        ro = total_incoming / service_rate

        exp_delay_at_switch = (ro / (1 - ro)) - ((queue_capacity+1)*ro**(queue_capacity+1)/(1-ro**(queue_capacity+1)))

        total_delay += exp_delay_at_switch
    
    print(f"{total_delay = }")

print("---------------------")
print("Packet loss for each switch")
print("---------------------")

for switch, attributes in switches:
    total_incoming = 0
    for u, v, data in G.in_edges(switch, data=True):
        total_incoming += sum(data.get('flows', {}).values())

    total_outgoing = 0
    for u, v, data in G.out_edges(switch, data=True):
        total_outgoing += sum(data.get('flows', {}).values())

    packet_loss = total_incoming - total_outgoing

    print(f"Packet loss for switch {switch} = {packet_loss}")





# TODO: Policzyc dlugosc kolejki, policzyc delay na kazdym switchu i dodac do total,
# podac go dla flow. Straty pakietow moge dac total_straty, for each switch inc - outgoing
#Wyprintowac i tyle


# pos = nx.kamada_kawai_layout(G)
# # 4. Draw the Graph
# plt.figure(figsize=(8, 6))
# nx.draw(
#     G,
#     with_labels=True,
#     node_size=2000,
#     node_color="skyblue",
#     font_size=15,
#     font_weight="bold",
#     arrows=True,       # Ensures arrows are drawn
#     arrowstyle='-|>',  # Fancy arrow style
#     arrowsize=20,
#     connectionstyle="arc3,rad=0.1",
# )

# plt.title("Basic DiGraph Visualization")
# plt.show()
