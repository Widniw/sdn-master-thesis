import numpy as np
import matplotlib.pyplot as plt
from network_env_elephant import NetworkEnv
from stable_baselines3 import DDPG, PPO 
import networkx as nx
from utils import json2networkx
from pathlib import Path
from switch import Switch
from matplotlib.colors import LinearSegmentedColormap


def main():
    print("Loading environment and model...")
    ddpg_env = NetworkEnv()
    seed = 1003
    
    # Load your best trained DDPG model
    best_model_path = "./models/elephant/5x5_grid/dijkstra-based/ddpg_sdn_elephant_routing_300000_steps" 
    article_model = DDPG.load(best_model_path, env = ddpg_env)

    # 1. Reset the environment to generate a new random traffic matrix
    obs, _ = ddpg_env.reset(seed=seed) # Set seed early to be safe

    action, _states = article_model.predict(obs, deterministic=True)
    
    # Capture the REWARD (the second variable returned by step)
    flatten_AVTM_matrix, reward, _, _, info = ddpg_env.step(action)

    print(f"{reward = }")
    
    G = ddpg_env.model.G

    nodes_to_remove = []

    for node, attr in G.nodes(data = True):
        if not isinstance(attr.get('data'), Switch):
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        G.remove_node(node)


    elephant_flow_nodes = ddpg_env.flows_paths[ddpg_env.elephant_flow]
    elephant_flow_nodes = [node for node in elephant_flow_nodes if node in G.nodes()]    

    elephant_flow_edges = []
    for (u,v) in zip(elephant_flow_nodes, elephant_flow_nodes[1:]):
        elephant_flow_edges.append((u,v))

    pos = {node: (int(node) % 5, (int(node) // 5)) for node in G.nodes()}

    elephant_flow_color = "#89CFF0"

    node_colors = [elephant_flow_color if node in elephant_flow_nodes else "black" for node in G.nodes()]

    switch_AVTM_matrix = flatten_AVTM_matrix.reshape((25, 25))

    edge_traffic_values = []
    for u, v in G.edges():
        # Make sure the nodes can be mapped to your 0-24 matrix indices
        traffic_load = switch_AVTM_matrix[int(u)][int(v)]
        if (u, v) in elephant_flow_edges:
            print(f"{(u, v)} = {traffic_load}")
        edge_traffic_values.append(traffic_load)

    edge_widths = []
    for load in edge_traffic_values:
        # Assuming your load is generally between 0 and 1
        # Adjust the multiplier (4.0) if you want them even thicker
        visual_thickness = 1.0 + (load * 4.0) 
        edge_widths.append(visual_thickness)


    edge_colors = []
    for u, v in G.edges():
        if (u, v) in elephant_flow_edges:
            edge_colors.append(elephant_flow_color)
        else:
            edge_colors.append("black")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Rysowanie grafu z dynamiczną szerokością
    nx.draw(
        G,
        pos=pos,
        ax = ax,
        with_labels=True,
        node_size=1000,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        font_color="white",
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        edge_color=edge_colors,
        width=edge_widths,  
    )

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    nx.draw_networkx_edge_labels(
            G,
            pos=pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="darkred", 
            ax=ax,
            label_pos=0.7,  
            rotate=False  
        )

    plt.show()

if __name__ == "__main__":
    main()