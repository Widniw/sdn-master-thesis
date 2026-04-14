import numpy as np
import matplotlib.pyplot as plt
from network_env import NetworkEnv
from stable_baselines3 import DDPG, PPO 
import networkx as nx
from utils import json2networkx
from pathlib import Path
from switch import Switch
from matplotlib.colors import LinearSegmentedColormap


def main():
    print("Loading environment and model...")
    ddpg_env = NetworkEnv()
    seed = 412158
    
    # Load your best trained DDPG model
    best_model_path = "./models/article_dijkstra/ddpg_sdn_routing_200000_steps" 
    article_model = DDPG.load(best_model_path, env = ddpg_env)

    # 1. Reset the environment to generate a new random traffic matrix
    obs, _ = ddpg_env.reset(seed=seed) # Set seed early to be safe

    action, _states = article_model.predict(obs, deterministic=True)

    # --- TEST 1: THE NAIVE METHOD ---
    # action = np.ones(ddpg_env.action_space.shape)
    
    # Capture the REWARD (the second variable returned by step)
    flatten_AVTM_matrix, reward, _, _, info = ddpg_env.step(action)

    switch_AVTM_matrix = flatten_AVTM_matrix.reshape((25, 25))

    switch_ro = {}
    for switch in range(25):
        switch_ro[switch] = switch_AVTM_matrix[:, switch].sum()    

    print(f"{switch_AVTM_matrix = }")
    print(f"{switch_ro = }")
    print(f"average_ro = {np.average(list(switch_ro.values()))}")
    print(f"avg_delay = {info['avg_delay']}")
    print(f"packet loss = {info['packet_loss']}")
    print(f"{reward = }")

    base_dir = Path(__file__).resolve().parent
    topology_path = base_dir / "topologies" / "mesh5x5.json"

    G = json2networkx(topology_path)

    nodes_to_remove = []

    for node, attr in G.nodes(data = True):
        if not isinstance(attr.get('data'), Switch):
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        G.remove_node(node)

    node_traffic_values = []
    for node in G.nodes():
        node_traffic_values.append(switch_ro[int(node)])

    min_node_traffic = 0
    max_node_traffic = max(max(node_traffic_values), 2)

    pos = {node: (int(node) % 5, (int(node) // 5)) for node in G.nodes()}

    custom_colors = ["green", "yellow", "red", "purple"]
    traffic_cmap = LinearSegmentedColormap.from_list("traffic_heatmap", custom_colors)

    # 1. Extract the specific traffic value for every edge in the graph
    edge_traffic_values = []
    for u, v in G.edges():
        # Make sure the nodes can be mapped to your 0-24 matrix indices
        traffic_load = switch_AVTM_matrix[int(u)][int(v)]
        edge_traffic_values.append(traffic_load)

    edge_widths = []
    for load in edge_traffic_values:
        # Assuming your load is generally between 0 and 1
        # Adjust the multiplier (4.0) if you want them even thicker
        visual_thickness = 1.0 + (load * 4.0) 
        edge_widths.append(visual_thickness)

    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Rysowanie grafu z dynamiczną szerokością
    nx.draw(
        G,
        pos=pos,
        ax = ax,
        with_labels=True,
        node_size=1000,
        node_color=node_traffic_values, 
        cmap=traffic_cmap,              
        vmin=min_node_traffic,          
        vmax=max_node_traffic,
        font_size=10,
        font_weight="bold",
        font_color="white",
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        width=edge_widths,  
        edge_color="black", 
    )

    sm = plt.cm.ScalarMappable(cmap=traffic_cmap, norm=plt.Normalize(vmin=min_node_traffic, vmax=max_node_traffic))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('Utilization Factor ρ', fontsize=12, fontweight='bold')  
    cbar.ax.set_navigate(False)

    plt.title("Network Traffic Heatmap (Dijkstra Routing)", fontsize=16, fontweight='bold')

    plt.show()

if __name__ == "__main__":
    main()