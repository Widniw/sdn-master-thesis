import networkx as nx
from switch import Switch
QUEUE_CAPACITY = 10000
SERVICE_RATE = 3000

G = nx.Graph()

flows_info = {1: {"path": (1),
                  "packet_rate": 2500}}

switches_info = {1: {"service_rate": SERVICE_RATE,
                "queue_capacity": QUEUE_CAPACITY,
                "aggr_arrival_rate": 2999}
            }

switches = [Switch(**switches_info[i]) for i in switches_info]

for i, switch in enumerate(switches):
    G.add_node(i, data = switch)



G.nodes[0]['data'].aggr_arrival_rate = flows_info[1]["packet_rate"]
print(f"Switch 1 Aggr arrival rate: {G.nodes[0]['data'].aggr_arrival_rate}")
print(f"Switch 1 Packet Loss Probability: {G.nodes[0]['data'].packet_loss_probabilty}")
print(f"Switch 1 Expected Queue Occupation: {G.nodes[0]['data'].exp_queue_occupation}")
print(f"Switch 1 Expected Delay [s]: {G.nodes[0]['data'].exp_delay}")



