import os
import numpy as np
import matplotlib.pyplot as plt

# Create the folder 'lbm_plots' if it doesn't exist
output_folder = "lbm_plots"
os.makedirs(output_folder, exist_ok=True)

# Parameters for LBM and grid
width, height = 100, 100
tau = 0.6  # Relaxation time, controls viscosity

# D2Q9 model directions and weights
DIRECTIONS = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
WEIGHTS = np.array([4/9] + [1/9] * 4 + [1/36] * 4)

# Helper function to calculate the equilibrium distribution with non-linearity
def equilibrium_distribution(rho, ux, uy, non_linear_factor=0.02):
    cu = np.einsum('ij,xyj->xyi', DIRECTIONS, np.stack([ux, uy], axis=-1))
    usqr = ux**2 + uy**2
    non_linear_term = non_linear_factor * rho[..., None] * usqr[..., None] * (1 + cu)
    feq = WEIGHTS * rho[..., None] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usqr[..., None]) + non_linear_term
    return feq

# Function to run the simulation for a specific tau and steps
def run_simulation(tau, max_steps=1000, instability_threshold=1e3):
    # Initialize variables
    rho = np.ones((width, height))
    ux = np.zeros((width, height))
    uy = np.zeros((width, height))
    f = np.ones((width, height, 9)) * WEIGHTS
    
    # Define particles as regions with higher density and initial velocity
    particles = [(30, 50, 1.5, 0.1, 0.2), (70, 50, 1.5, -0.1, -0.2)]
    for x, y, density, vx, vy in particles:
        rho[x-1:x+2, y-1:y+2] = density
        ux[x-1:x+2, y-1:y+2] = vx
        uy[x-1:x+2, y-1:y+2] = vy
        f[x-1:x+2, y-1:y+2] = equilibrium_distribution(rho[x-1:x+2, y-1:y+2], ux[x-1:x+2, y-1:y+2], uy[x-1:x+2, y-1:y+2])
    
    max_rho_list = []
    for step in range(max_steps):
        feq = equilibrium_distribution(rho, ux, uy)
        f = f - (f - feq) / tau

        # Streaming step
        for i, d in enumerate(DIRECTIONS):
            f[..., i] = np.roll(np.roll(f[..., i], d[0], axis=0), d[1], axis=1)

        # Update macroscopic variables
        rho = np.sum(f, axis=-1)
        ux = np.sum(f * DIRECTIONS[:, 0], axis=-1) / rho
        uy = np.sum(f * DIRECTIONS[:, 1], axis=-1) / rho

        # Record the maximum density
        max_rho_list.append(np.max(rho))

        # Detect instability
        if np.max(rho) > instability_threshold or not np.all(np.isfinite(rho)):
            print(f"Instability detected at step {step+1} for tau = {tau:.2f}")
            break
    
    return max_rho_list

# Parameters for convergence test
taus = np.linspace(0.6, 1.8, 8)  # Relaxation times to test
max_steps = 1000                 # Maximum steps to simulate
instability_threshold = 1e3      # Threshold for detecting instability in rho

# Store results
convergence_data = []

for tau in taus:
    print(f"Testing tau = {tau:.2f}...")
    max_rho_list = run_simulation(tau, max_steps, instability_threshold)
    convergence_data.append({
        "tau": tau,
        "steps": np.arange(len(max_rho_list)),
        "max_rho": max_rho_list
    })

# 1. Enhanced 2D Logarithmic Plot
fig, ax = plt.subplots(figsize=(10, 6))
for data in convergence_data:
    ax.plot(data["steps"], data["max_rho"], label=f"Tau = {data['tau']:.2f}")

ax.set_title("Convergence Test: Max Density (rho) vs Steps (Log Scale)")
ax.set_xlabel("Steps")
ax.set_ylabel("Max Density (rho)")
ax.set_yscale("log")  # Use logarithmic scale to observe divergence
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()

# Save the plot
convergence_plot_path = os.path.join(output_folder, "convergence_test_log_scale.png")
plt.savefig(convergence_plot_path)
plt.show()
