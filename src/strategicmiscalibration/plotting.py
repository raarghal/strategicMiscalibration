from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(["science", "ieee"])

granularity = 1000

# Parameters
r = 1.0  # reward for task success
e = 0.5  # effort to self complete task
c = 0.1  # cost to delegate task
p_plus = 0.8  # success rate on easy task
p_minus = 0.2  # success rate on hard task
theta_H = 0.8  # probability of easy task for high ability agent
theta_L = 0.2  # probability of easy task for low ability agent

# Prior belief grids
h_grid = np.linspace(0.0, 1.0, granularity)  # values for honesty prior belief
mu_grid = np.linspace(0.0, 1.0, granularity)  # values for ability prior belief

# Derived parameters
p_star = (r * (1.0 - p_minus) - e + c) / (r * (p_plus - p_minus))
p_ratio = p_star / (1.0 - p_star)

# Create meshgrid for plotting
h_mesh, mu_mesh = np.meshgrid(h_grid, mu_grid)

# Compute t_bar on the mesh
t_bar_mesh = mu_mesh * theta_H + (1.0 - mu_mesh) * theta_L
t_ratio = t_bar_mesh / (1.0 - t_bar_mesh)

# Compute the right-hand sides of the inequalities
with np.errstate(divide="ignore", invalid="ignore"):
    denom_1 = 1.0 - h_mesh
    denom_1 = np.where(denom_1 <= 0.0, np.nan, denom_1)
    rhs_1 = t_ratio / denom_1

    ratio_term = np.where(h_mesh > 0.0, (1.0 - h_mesh) / h_mesh, np.inf)
    cap_factor = np.minimum(1.0, ratio_term)
    rhs_2 = t_ratio * cap_factor

# Regions satisfying the inequalities
region_1_mask = np.isfinite(rhs_1) & (p_ratio <= rhs_1)
region_2_mask = np.isfinite(rhs_2) & (p_ratio <= rhs_2)

region_1 = np.where(region_1_mask, 1.0, np.nan)
region_2 = np.where(region_2_mask, 1.0, np.nan)
neither_region = np.where(~(region_1_mask | region_2_mask), 1.0, np.nan)

# Prepare figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel(r"$h$", fontsize=14)
ax.set_ylabel(r"$\mu$", fontsize=14)
ax.tick_params(axis="both", labelsize=14)
# ax.set_title("Final Period Trust Regions")

# Shade regions
neither_set = ax.contourf(
    h_mesh,
    mu_mesh,
    neither_region,
    levels=[0.5, 1.5],
    colors=["#7F2704"],
    alpha=0.2,
    antialiased=True,
)

region_1_set = ax.contourf(
    h_mesh,
    mu_mesh,
    region_1,
    levels=[0.5, 1.5],
    colors=["#4C78A8"],
    alpha=0.35,
    antialiased=True,
)

region_2_set = ax.contourf(
    h_mesh,
    mu_mesh,
    region_2,
    levels=[0.5, 1.5],
    colors=["#54A24B"],
    alpha=0.35,
    antialiased=True,
)

# Plot boundaries of the inequalities
boundary_1 = rhs_1 - p_ratio
boundary_2 = rhs_2 - p_ratio

cs1 = ax.contour(
    h_mesh,
    mu_mesh,
    boundary_1,
    levels=[0.0],
    colors="#7F2704",
    linestyles="-",
    linewidths=2.0,
)

cs2 = ax.contour(
    h_mesh,
    mu_mesh,
    boundary_2,
    levels=[0.0],
    colors="#1B3A4B",
    linestyles="--",
    linewidths=2.0,
)

# Legend
legend_handles = [
    Patch(
        facecolor="#54A24B",
        alpha=0.35,
        label=r"$\tau_B$",
    ),
    Patch(
        facecolor="#4C78A8",
        alpha=0.35,
        label=r"$\tau_C$",
    ),
    Patch(
        facecolor="#7F2704",
        alpha=0.2,
        label=r"$\tau_D$",
    ),
    Line2D([], [], color="#7F2704", linewidth=2.0, label="Delegation boundary"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=14)

# Annotate singular points
ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.3)
ax.axvline(1.0, color="black", linewidth=0.5, alpha=0.3)
ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.3)

# ax.set_aspect("auto")
# ax.grid(False)

# Save figure
output_dir = Path(__file__).resolve().parents[2] / "figures"
print(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "trust_regions.pdf", format="pdf", dpi=1200, bbox_inches="tight")

plt.close(fig)
