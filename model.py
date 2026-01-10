import networkx as nx
from switch import Switch
import matplotlib.pyplot as plt
QUEUE_CAPACITY = 10000
SERVICE_RATE = 3000

packet_rates = [i * 5 for i in range (1,600)]
packet_losses = []
queue_occupations = []
expected_delays = []

for packet_rate in packet_rates:
    G = nx.Graph()

    switches_info = {1: {"service_rate": SERVICE_RATE,
                    "queue_capacity": QUEUE_CAPACITY,
                    "aggr_arrival_rate": packet_rate}
                }

    switches = [Switch(**switches_info[i]) for i in switches_info]

    for i, switch in enumerate(switches):
        G.add_node(i, data = switch) 

    packet_losses.append(G.nodes[0]['data'].packet_loss_probabilty)
    queue_occupations.append(G.nodes[0]['data'].exp_queue_occupation)
    expected_delays.append(G.nodes[0]['data'].exp_delay)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Figure 1
axes[0].plot(packet_rates, packet_losses)
axes[0].set_title("Packet Loss vs Packet Rate")
axes[0].set_xlabel("Packet Rate [pkt/s]")
axes[0].set_ylabel("Packet Loss")
axes[0].grid(True)

# Figure 2
axes[1].plot(packet_rates, queue_occupations)
axes[1].set_title("Queue Occupation vs Packet Rate")
axes[1].set_xlabel("Packet Rate [pkt/s]")
axes[1].set_ylabel("Queue Occupation [pkt]")
axes[1].grid(True)

# Figure 3
axes[2].plot(packet_rates, expected_delays)
axes[2].set_title("Expected Delays vs Packet Rate")
axes[2].set_xlabel("Packet Rate [pkt/s]")
axes[2].set_ylabel("Expected Delays [s]")
axes[2].grid(True)

plt.tight_layout()
plt.show()
