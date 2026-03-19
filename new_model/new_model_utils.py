import networkx as nx
import json
from switch import Switch
from matplotlib import pyplot as plt
from pathlib import Path


def json2networkx(json_data):
    with open(json_data, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph() 

    for node, attributes in data.items():
        switch_obj = Switch(service_rate = attributes["service_rate"],
                queue_capacity = attributes["queue_capacity"],
                aggr_arrival_rate = 1)
        
        G.add_node(node)

        for neighbor in attributes["neighbors"]:
            queue_out = f"{node}_{neighbor}"
            queue_in = f"{neighbor}_{node}"

            G.add_node(queue_out, data = switch_obj)
            G.add_node(queue_in, data = switch_obj)

            G.add_edge(node, queue_out, weight = 1, flows = {})
            G.add_edge(queue_in, node, weight = 1, flows = {})

        # Add host x to switch x
        host = f"10.0.1.{node}"
        G.add_node(host)

        queue_out = f"{node}_10.0.1.{node}"
        G.add_node(queue_out, data = switch_obj)

        G.add_edge(node, queue_out, weight = 1, flows = {})
        G.add_edge(queue_out, host, weight = 1, flows = {})
        G.add_edge(host, node, weight = 1, flows = {})

    return G


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    topology_path = base_dir / "topologies" / "mesh5x5.json"

    G = json2networkx(topology_path)
    
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
    # widths = [1 + (w / max_traffic) * 3 for w in widths]

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
        width=1,  # <--- TU WSTAWIAMY LISTĘ SZEROKOŚCI
    )

    # Dodanie legendy lub tytułu z informacją o skali
    plt.title(f"Wizualizacja obciążenia sieci (Max traffic: {max_traffic:.2f})")
    plt.show()