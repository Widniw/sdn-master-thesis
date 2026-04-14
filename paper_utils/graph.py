import numpy as np
import matplotlib.pyplot as plt

# Enable/Disable LaTeX rendering based on your system setup
plt.rcParams.update({
    "text.usetex": False, # Change to True if you installed texlive!
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Parameters
N_values = [1, 5, 10, 100, 1000]
mu = 1.0  # Fixed service rate (e.g., router can handle 10 packets/ms)

# Generate arrival rates (lambda) from 0 to 25
# This pushes lambda well past the service rate (mu) to show congestion
lambdas = np.linspace(0.0, 2.0, 1000)

plt.figure(figsize=(8, 5))

# Calculate and plot the data
with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
    for N in N_values:
        
        # Calculate utilization (rho) dynamically
        rho = lambdas / mu
        
        # 1. Calculate the acceptance ratio: (1 - P_B)
        ratio = (1 - rho**N) / (1 - rho**(N+1))
        
        # Handle the mathematical limits (just like before)
        ratio = np.where(rho == 1.0, N / (N + 1), ratio)
        ratio = np.where(np.isnan(ratio), 1.0 / rho, ratio)
        
        # 2. Multiply by lambda to get the Effective Arrival Rate (lambda_+)
        lambda_plus = lambdas * ratio
        
        # Plot the curve
        plt.plot(lambdas, lambda_plus, label=rf'$N = {N}$')

plt.axhline(y=mu, color='red', linestyle='--', alpha=0.5, 
            label=rf'Max Router Capacity ($\mu={mu}$)')

# Formatting the graph
plt.xlabel(r'Arrival Rate ($\lambda$)')
plt.ylabel(r'Effective Arrival Rate ($\lambda_{+}$)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title=r'Buffer Capacity ($N$)')

# Set axis limits
plt.xlim(0, max(lambdas))
plt.ylim(0, mu * 1.4) # Show just a bit above the max capacity

# Save and display
plt.tight_layout()
plt.savefig('throughput_vs_load.pdf', format='pdf', bbox_inches='tight')
plt.show()