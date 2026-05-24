import matplotlib.pyplot as plt

traffic_rate = {
    1: 1567.6016,
    6: 844.8232,
    11: 844.8211,
    16: 566.09143,
    17: 383.6731,
    18: 306.07492,
    23: 306.0671
}

# Convert keys to strings so they are treated as categorical labels on X-axis
switches = [str(k) for k in traffic_rate.keys()]
rates = [round(v, 2) for v in traffic_rate.values()]

plt.figure(figsize=(10, 6))
bars = plt.bar(switches, rates, color='red', edgecolor='black', alpha=0.85)

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(rates) * 0.02), f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel("Switch", fontsize=12, fontweight='bold')
plt.ylabel("Elephant Flow Outgoing Traffic Rate", fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(rates) + max(rates) * 0.1) # Add a 10% padding on top for the labels

plt.tight_layout()
plt.show()
