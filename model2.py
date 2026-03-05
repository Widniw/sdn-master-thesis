from utils import json2networkx
from switch import Switch
import matplotlib.pyplot as plt
import networkx as nx
from traffic_leaving_mm1k import traffic_leaving_mm1k
import random
import numpy as np


G = json2networkx("topologies/mesh5x5.json")

# flows = {("10.0.0.1", "10.0.0.2"): 3,
#          ("10.0.0.2", "10.0.0.1"): 2}

# flows = {("10.0.0.5", "10.0.0.1"): 15,
#          ("10.0.0.2", "10.0.0.1"): 3,
#          ("10.0.0.8", "10.0.0.7"): 6,
#          ("10.0.0.3", "10.0.0.6"): 8,
#          ("10.0.0.7", "10.0.0.8"): 11,}

# Article accurate flows for grid 5x5
flows = {}
no_of_flows = 150

for flow in range(no_of_flows):
    random_hosts = random.sample(range(1, 26), 2)
    random_traffic_rate = random.randint(10, 300)
    flows[(f"10.0.0.{random_hosts[0]}",f"10.0.0.{random_hosts[1]}")] = random_traffic_rate

flows_paths = {}

for flow_name, traffic in flows.items():
    dijkstra_path = nx.dijkstra_path(G, source = flow_name[0], target = flow_name[1], weight = "weight")

    flows_paths[flow_name] = dijkstra_path

    for u, v in zip(dijkstra_path, dijkstra_path[1:]):
        G[u][v]['flows'][flow_name] = traffic

# Create list of switches
switches = []
for node, attributes in G.nodes(data=True):
    obj = attributes.get('data')

    if isinstance(obj, Switch):
        switches.append((node, attributes))


for i in range(18):
    temp_flows_on_edges = {}

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
            G[u][v]['flows'][_flow_id] = _traffic

        

# Print DiGraph for debbuging purposes
print("G as text:")
for node in G.nodes(data=True):
    node_id, node_attr = node
    print(f"Node {node_id}:")
    for neighbor, edge_attr in G[node_id].items():
        print(f"  -> {neighbor}: {edge_attr}")

switches_delay = {}

for switch, attributes in switches:
    total_incoming = 0
    for u, v, data in G.in_edges(switch, data=True):
        total_incoming += sum(data.get('flows', {}).values())

    if total_incoming == 0:
        switches_delay[switch] = 0
        continue

    total_outgoing = 0
    for u, v, data in G.out_edges(switch, data=True):
        total_outgoing += sum(data.get('flows', {}).values())

    switch_obj = G.nodes[switch]["data"]

    service_rate = switch_obj.service_rate
    queue_capacity = switch_obj.queue_capacity

    ro = total_incoming / service_rate

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

    switches_delay[switch] = exp_delay_at_switch

### CALCULATING DELAY AND PACKET LOSS FOR EACH FLOW
print("------------------------------------------")

delays = []

for flow_name, traffic in flows.items():
    print(f"For flow {flow_name} with traffic volume: {traffic}")
    dijkstra_path = nx.dijkstra_path(G, source = flow_name[0], target = flow_name[1], weight = "weight")
    print(f"{dijkstra_path = }")

    total_flow_delay = 0

    for switch in dijkstra_path[1:-1]:
        total_flow_delay += switches_delay[switch]
    
    print(f"{round(total_flow_delay,2) = }s")

    delays.append(total_flow_delay)

average_delay_in_network = sum(delays) / len(delays)

print(f"Average delay = {round(average_delay_in_network,2)}s")

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

    print(f"Packet loss for switch {switch} = {round(packet_loss, 2)}pkt/s")


# --- WIZUALIZACJA ---

# 1. Obliczanie szerokości krawędzi na podstawie ruchu
widths = []
max_traffic = 0

# Iterujemy po krawędziach w takiej samej kolejności, w jakiej nx.draw je rysuje
for u, v, data in G.edges(data=True):
    flows_on_edge = data.get('flows', {})
    traffic_sum = sum(flows_on_edge.values())
    
    # Możesz dodać minimalną grubość (np. 1.0), żeby krawędzie z ruchem 0 były widoczne
    # lub skalować wartości, jeśli ruch jest bardzo duży
    widths.append(traffic_sum) 
    
    if traffic_sum > max_traffic:
        max_traffic = traffic_sum

# Opcjonalnie: Normalizacja szerokości, jeśli wartości ruchu są bardzo duże
widths = [1 + (w / max_traffic) * 3 for w in widths]

pos = nx.kamada_kawai_layout(G)

plt.figure(figsize=(10, 8))

# Rysowanie grafu z dynamiczną szerokością
nx.draw(
    G,
    pos=pos,
    with_labels=True,
    node_size=2000,
    node_color="skyblue",
    font_size=10,
    font_weight="bold",
    arrows=True,
    arrowstyle='-|>',
    arrowsize=20,
    connectionstyle="arc3,rad=0.1",
    width=widths,  # <--- TU WSTAWIAMY LISTĘ SZEROKOŚCI
)

# Dodanie legendy lub tytułu z informacją o skali
plt.title(f"Wizualizacja obciążenia sieci (Max traffic: {max_traffic:.2f})")
plt.show()

no_of_switches = len(switches)

AVTM_matrix = np.zeros((no_of_switches, no_of_switches))

for switches_src, attributes_src in switches:
    for switches_dst, attributes_dst in switches:
        switch_src_index = int(switches_src) - 1
        switch_dst_index = int(switches_dst) - 1 

        if switches_src == switches_dst:
            continue

        if G.has_edge(switches_src, switches_dst):
            
            edge_data_flows = G[switches_src][switches_dst]['flows']

            sum_of_flows = 0

            for flow, value in edge_data_flows.items():
                sum_of_flows += value
            
            AVTM_matrix[switch_src_index][switch_dst_index] = round(sum_of_flows,2)

plt.figure(figsize=(12, 12))

# We use imshow just to set the limits and aspect ratio, but make it invisible or very light background
plt.imshow(AVTM_matrix, cmap='Greys', alpha=0.1)

# Loop over data dimensions and create text annotations.
rows, cols = AVTM_matrix.shape
for i in range(rows):
    for j in range(cols):
        text = plt.text(j, i, str(AVTM_matrix[i, j]),
                       ha="center", va="center", color="black", fontsize=8)

plt.title('AVTM Matrix')
# We can keep the ticks to show row/col indices
# plt.xticks(np.arange(0, 25, 1))
# plt.yticks(np.arange(0, 25, 1))
plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Remove the frame to look more like a raw printed matrix? 
# Usually figures have frames. I'll keep the frame but make it look clean.
plt.tight_layout()    
plt.show()