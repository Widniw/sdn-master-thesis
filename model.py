import networkx as nx
import matplotlib.pyplot as plt

CAPACITY = 3000

G = nx.Graph()

switches = {1: {"service_rate": 10,
                "capacity": 30,
                "queue_legth": 15}}

edges = {(1,2): {"weight": 3}}

G.add_nodes_from([(1, {"capacity": CAPACITY}), (2, {"capacity": CAPACITY}),
                   (3, {"capacity": CAPACITY})])

G.add_edges_from([(1, 2, {"weight": 3}), (1, 3, {"weight": 3}),
                   (2, 3, {"weight": 3})])

# print(list(G.nodes))
# print(list(G.edges))

path = nx.shortest_path(G, source=1, target=3, weight=None, method='dijkstra')
print(path)