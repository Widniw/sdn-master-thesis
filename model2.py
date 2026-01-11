from utils import json2networkx
import matplotlib.pyplot as plt
import networkx as nx


G = json2networkx("topologies/mesh3x3.json")
pos = nx.spring_layout(G)

# 4. Draw the Graph
plt.figure(figsize=(8, 6))
nx.draw(
    G,
    with_labels=True,
    node_size=2000,
    node_color="skyblue",
    font_size=15,
    font_weight="bold",
    arrows=True,       # Ensures arrows are drawn
    arrowstyle='-|>',  # Fancy arrow style
    arrowsize=20,
    connectionstyle="arc3,rad=0.1",
)

plt.title("Basic DiGraph Visualization")
plt.show()
