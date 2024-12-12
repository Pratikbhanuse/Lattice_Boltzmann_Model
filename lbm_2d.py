import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create the folder 'lbm_plots' if it doesn't exist
output_folder = "lbm_plots"
os.makedirs(output_folder, exist_ok=True)

# Parameters for LBM and grid
# width, height = 100, 100
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

# Initialize the distribution function
f = np.ones((width, height, 9)) * WEIGHTS
rho = np.ones((width, height))
ux = np.zeros((width, height))
uy = np.zeros((width, height))

# Define particles as regions with higher density and initial velocity
particles = [(30, 50, 1.5, 0.1, 0.2), (70, 50, 1.5, -0.1, -0.2)]
for x, y, density, vx, vy in particles:
    rho[x-1:x+2, y-1:y+2] = density
    ux[x-1:x+2, y-1:y+2] = vx
    uy[x-1:x+2, y-1:y+2] = vy
    f[x-1:x+2, y-1:y+2] = equilibrium_distribution(rho[x-1:x+2, y-1:y+2], ux[x-1:x+2, y-1:y+2], uy[x-1:x+2, y-1:y+2])

# Main simulation loop
num_steps = 300
positions_history = []

for step in range(num_steps):
    feq = equilibrium_distribution(rho, ux, uy)
    f = f - (f - feq) / tau

    for i, d in enumerate(DIRECTIONS):
        f[..., i] = np.roll(np.roll(f[..., i], d[0], axis=0), d[1], axis=1)

    # Update macroscopic variables
    rho = np.sum(f, axis=-1)
    ux = np.sum(f * DIRECTIONS[:, 0], axis=-1) / rho
    uy = np.sum(f * DIRECTIONS[:, 1], axis=-1) / rho

    # Store positions of particles for visualization (density peaks)
    positions_history.append((rho.copy(), ux.copy(), uy.copy()))

# Create a single GIF for the entire simulation
fig, ax = plt.subplots(figsize=(8, 6))
density_plot = ax.imshow(rho.T, cmap="viridis", origin="lower")
ax.set_title("Fluid Simulation")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def animate(frame):
    rho, ux, uy = positions_history[frame]
    density_plot.set_data(rho.T)
    return density_plot,

ani = FuncAnimation(fig, animate, frames=np.arange(num_steps), interval=50, blit=True)
gif_path = os.path.join(output_folder, "simulation.gif")
ani.save(gif_path, writer='pillow', fps=20)
plt.show()

# Create a single plot with 6 panels (3x2 grid) for snapshots
snapshot_steps = np.linspace(0, num_steps - 1, 6, dtype=int)  # Select 6 evenly spaced steps
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

for ax, step in zip(axes.flat, snapshot_steps):
    rho, ux, uy = positions_history[step]
    im = ax.imshow(rho.T, cmap="viridis", origin="lower")
    ax.set_title(f"Step {step+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()

# plt.tight_layout()
# Adjust layout and add a colorbar
fig.colorbar(im, ax=axes, location="right", shrink=0.6, label="Density (rho)")
plt.suptitle("Snapshots of Simulation at Different Steps", fontsize=16)
snapshots_path = os.path.join(output_folder, "simulation_snapshots.png")
plt.savefig(snapshots_path)
plt.show()

# Create a 3D scatter plot for the final state
fig_3d = plt.figure(figsize=(8, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Scatter plot with velocities and density
rho_flat = rho.flatten()
ux_flat = ux.flatten()
uy_flat = uy.flatten()
x_flat = np.repeat(np.arange(width), height)
y_flat = np.tile(np.arange(height), width)

scatter = ax_3d.scatter(x_flat, y_flat, rho_flat, c=rho_flat, cmap='viridis', s=2)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Density (rho)')
ax_3d.set_title('3D Scatter Plot: Density (rho) vs X, Y')
fig_3d.colorbar(scatter)
plt.tight_layout()

# Save and show the 3D scatter plot
scatter_plot_path = os.path.join(output_folder, "3d_scatter_plot.png")
plt.savefig(scatter_plot_path)
plt.show()