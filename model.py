import networkx as nx
from switch import Switch
QUEUE_CAPACITY = 10000
SERVICE_RATE = 3000

G = nx.Graph()

switches_info = {1: {"service_rate": SERVICE_RATE,
                "queue_capacity": QUEUE_CAPACITY}
            }

switches = [Switch(**switches_info[i]) for i in switches_info]

G.add_nodes_from(switches)

print(list(G.nodes))

