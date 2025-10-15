"""
Complete plotting script for protein experiments (Matplotlib)
----------------------------------------------------------

•  Protein generation experiments (FoldFlow)
    – TM-score vs. compute budget for different methods
    – Same modern styling as image experiments
    – TM-score is the primary metric (higher is better)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
import os
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# MODERN STYLING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Set modern style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Configure matplotlib for high-quality plots
rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 22,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

# Create output directory
OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modern color palette (identical to image plotting script)
colors = {
    # Noise schedules
    "SDE": "#E74C3C",
    "EDM-SDE": "#3498DB",
    "Score-SDE": "#2ECC71",
    "DivFree-ODE": "#9B59B6",
    "DMFM-ODE": "#F39C12",
    # Inference algorithms
    "RS": "#E67E22",
    "PE–DivFree": "#9B59B6",
    "PE–SDE": "#E74C3C",
    "PE–ScoreSDE": "#2ECC71",
    "RS+PE–DivFree": "#1ABC9C",
    "NS–DMFM-ODE": "#F39C12",
    "NS–SDE": "#34495E",
    "RS+NS–DMFM-ODE": "#8E44AD",
    # Back-compat keys used in loaded data
    "ODE-divfree": "#9B59B6",
    "ODE-divfree-max": "#F39C12",
    # Legacy key kept for back-compat; label mapped to RS in legends
    "Best-of-N": "#E67E22",
    "Path-DivFree": "#9B59B6",
    "Path-SDE": "#E74C3C",
    "Path-ScoreSDE": "#2ECC71",
    "BestN+Path-DivFree": "#1ABC9C",
    "NoiseSearch-3R-DivFree": "#F39C12",
    "BestN+NoiseSearch-3R-DivFree": "#8E44AD",
    "Random Search": "#E67E22",
    "ODE-divfree explore": "#9B59B6",
    "RS → Divfree explore": "#1ABC9C",
    "Score-SDE explore": "#2ECC71",
    "SDE explore": "#E74C3C",
    "baseline": "#7F8C8D",
}

# Modern markers (identical to image plotting script)
markers = {
    "SDE": "o",
    "EDM-SDE": "s",
    "Score-SDE": "^",
    "DivFree-ODE": "D",
    "DMFM-ODE": "v",
    # Back-compat
    "ODE-divfree": "D",
    "ODE-divfree-max": "v",
}
scaling_markers = ["o", "s", "^", "v", "D"]

# ══════════════════════════════════════════════════════════════════════════════
# PROTEIN EXPERIMENT DATA
# ══════════════════════════════════════════════════════════════════════════════

# New experimental data from latest protein experiments (multi-GPU scaling)
protein_data = {
    "method": [
        # BEST_OF_N method data → RS (Random Search)
        "RS",  # 1 branch
        "RS",  # 2 branches
        "RS",  # 4 branches
        "RS",  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX method data → NS–DMFM-ODE
        "NS–DMFM-ODE",  # 2 branches
        "NS–DMFM-ODE",  # 4 branches
        "NS–DMFM-ODE",  # 8 branches
        # NOISE_SEARCH_SDE method data → NS–SDE
        "NS–SDE",  # 2 branches
        "NS–SDE",  # 4 branches
        "NS–SDE",  # 8 branches
        # RANDOM_SEARCH_NOISE method data → RS+NS–DMFM-ODE
        "RS+NS–DMFM-ODE",  # 2 branches
        "RS+NS–DMFM-ODE",  # 4 branches
        "RS+NS–DMFM-ODE",  # 8 branches
    ],
    "num_branches": [1, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8],
    "mean_score": [
        # BEST_OF_N scores
        0.7926,  # 1 branch (baseline)
        0.8692,  # 2 branches
        0.8900,  # 4 branches
        0.8954,  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX scores
        0.8752,  # 2 branches
        0.8839,  # 4 branches
        0.8883,  # 8 branches
        # NOISE_SEARCH_SDE scores
        0.8797,  # 2 branches
        0.8740,  # 4 branches
        0.8864,  # 8 branches
        # RANDOM_SEARCH_NOISE scores
        0.8880,  # 2 branches
        0.8975,  # 4 branches
        0.9020,  # 8 branches
    ],
    "rmsd": [
        # BEST_OF_N RMSD values
        2.665,  # 1 branch
        2.203,  # 2 branches
        1.984,  # 4 branches
        1.922,  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX RMSD values
        2.098,  # 2 branches
        2.043,  # 4 branches
        2.022,  # 8 branches
        # NOISE_SEARCH_SDE RMSD values
        2.060,  # 2 branches
        2.124,  # 4 branches
        2.001,  # 8 branches
        # RANDOM_SEARCH_NOISE RMSD values
        1.968,  # 2 branches
        1.879,  # 4 branches
        1.849,  # 8 branches
    ],
    "under_2a_percent": [
        # BEST_OF_N <2Å percentages
        78.6,  # 1 branch
        84.7,  # 2 branches
        87.5,  # 4 branches
        86.1,  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX <2Å percentages
        84.0,  # 2 branches
        84.7,  # 4 branches
        86.1,  # 8 branches
        # NOISE_SEARCH_SDE <2Å percentages
        84.4,  # 2 branches
        82.1,  # 4 branches
        84.4,  # 8 branches
        # RANDOM_SEARCH_NOISE <2Å percentages
        84.9,  # 2 branches
        88.0,  # 4 branches
        88.9,  # 8 branches
    ],
}

# Convert to DataFrame for easier processing
df = pd.DataFrame(protein_data)

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC SCORING EXPERIMENT DATA
# ══════════════════════════════════════════════════════════════════════════════

# New experimental data using geometric scoring function
geometric_data = {
    "method": [
        # BEST_OF_N method data → RS (Random Search)
        "RS",  # 1 branch
        "RS",  # 2 branches
        "RS",  # 4 branches
        "RS",  # 8 branches
        # NOISE_SEARCH_SDE method data → NS–SDE
        "NS–SDE",  # 2 branches
        "NS–SDE",  # 4 branches
        "NS–SDE",  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX method data → NS–DMFM-ODE
        "NS–DMFM-ODE",  # 2 branches
        "NS–DMFM-ODE",  # 4 branches
        "NS–DMFM-ODE",  # 8 branches
        # RANDOM_SEARCH_NOISE method data → RS+NS–DMFM-ODE
        "RS+NS–DMFM-ODE",  # 2 branches
        "RS+NS–DMFM-ODE",  # 4 branches
        "RS+NS–DMFM-ODE",  # 8 branches
    ],
    "num_branches": [1, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8],
    "geometric_score": [
        # BEST_OF_N geometric scores
        3.6801,  # 1 branch
        3.9867,  # 2 branches
        4.1169,  # 4 branches
        4.2850,  # 8 branches
        # NOISE_SEARCH_SDE geometric scores
        4.3343,  # 2 branches
        4.4135,  # 4 branches
        4.5122,  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX geometric scores
        4.0715,  # 2 branches
        4.2472,  # 4 branches
        4.2227,  # 8 branches
        # RANDOM_SEARCH_NOISE geometric scores
        4.2099,  # 2 branches
        4.2886,  # 4 branches
        4.3509,  # 8 branches
    ],
    "tm_score": [
        # BEST_OF_N TM scores
        0.8384,  # 1 branch
        0.8414,  # 2 branches
        0.8222,  # 4 branches
        0.8608,  # 8 branches
        # NOISE_SEARCH_SDE TM scores
        0.8028,  # 2 branches
        0.7903,  # 4 branches
        0.7785,  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX TM scores
        0.8229,  # 2 branches
        0.8407,  # 4 branches
        0.8444,  # 8 branches
        # RANDOM_SEARCH_NOISE TM scores
        0.8459,  # 2 branches
        0.8387,  # 4 branches
        0.8519,  # 8 branches
    ],
    "rmsd": [
        # BEST_OF_N RMSD values
        2.865,  # 1 branch
        2.713,  # 2 branches
        3.071,  # 4 branches
        2.417,  # 8 branches
        # NOISE_SEARCH_SDE RMSD values
        3.171,  # 2 branches
        3.242,  # 4 branches
        3.712,  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX RMSD values
        3.133,  # 2 branches
        2.609,  # 4 branches
        2.650,  # 8 branches
        # RANDOM_SEARCH_NOISE RMSD values
        2.563,  # 2 branches
        2.839,  # 4 branches
        2.546,  # 8 branches
    ],
    "under_2a_percent": [
        # BEST_OF_N <2Å percentages
        78.8,  # 1 branch
        78.5,  # 2 branches
        74.5,  # 4 branches
        82.8,  # 8 branches
        # NOISE_SEARCH_SDE <2Å percentages
        73.4,  # 2 branches
        69.6,  # 4 branches
        71.0,  # 8 branches
        # NOISE_SEARCH_DIVFREE_MAX <2Å percentages
        75.7,  # 2 branches
        75.3,  # 4 branches
        77.8,  # 8 branches
        # RANDOM_SEARCH_NOISE <2Å percentages
        78.0,  # 2 branches
        76.7,  # 4 branches
        81.6,  # 8 branches
    ],
}

# Convert geometric data to DataFrame
df_geometric = pd.DataFrame(geometric_data)

# Organize data by method and compute budget
compute_budgets = [1, 2, 4, 8]

protein_scaling = {}

# Process each method
for method in df["method"].unique():
    method_data = df[df["method"] == method]
    protein_scaling[method] = {
        "mean_score": [],
        "rmsd": [],
        "under_2a_percent": [],
    }

    # Get the RS method's 1× baseline values (baseline for all methods)
    baseline_1x = df[(df["method"] == "RS") & (df["num_branches"] == 1)]
    if len(baseline_1x) > 0:
        baseline_score = baseline_1x["mean_score"].iloc[0]
        baseline_rmsd = baseline_1x["rmsd"].iloc[0]
        baseline_2a = baseline_1x["under_2a_percent"].iloc[0]
    else:
        baseline_score = 0.7926  # Keep original baseline TM-score
        baseline_rmsd = 2.665
        baseline_2a = 78.6

    for budget in compute_budgets:
        budget_data = method_data[method_data["num_branches"] == budget]
        if len(budget_data) > 0:
            protein_scaling[method]["mean_score"].append(
                budget_data["mean_score"].iloc[0]
            )
            protein_scaling[method]["rmsd"].append(budget_data["rmsd"].iloc[0])
            protein_scaling[method]["under_2a_percent"].append(
                budget_data["under_2a_percent"].iloc[0]
            )
        else:
            # Fill with baseline values if no data for this budget
            if budget == 1:
                protein_scaling[method]["mean_score"].append(baseline_score)
                protein_scaling[method]["rmsd"].append(baseline_rmsd)
                protein_scaling[method]["under_2a_percent"].append(baseline_2a)
            else:
                protein_scaling[method]["mean_score"].append(np.nan)
                protein_scaling[method]["rmsd"].append(np.nan)
                protein_scaling[method]["under_2a_percent"].append(np.nan)

# Process geometric scoring data
geometric_scaling = {}

for method in df_geometric["method"].unique():
    method_data = df_geometric[df_geometric["method"] == method]
    geometric_scaling[method] = {
        "geometric_score": [],
        "tm_score": [],
        "rmsd": [],
        "under_2a_percent": [],
    }

    # Get the RS method's 1× baseline values for geometric scoring experiments
    baseline_1x = df_geometric[
        (df_geometric["method"] == "RS") & (df_geometric["num_branches"] == 1)
    ]
    if len(baseline_1x) > 0:
        baseline_geometric = baseline_1x["geometric_score"].iloc[0]
        baseline_tm = baseline_1x["tm_score"].iloc[0]
        baseline_rmsd = baseline_1x["rmsd"].iloc[0]
        baseline_2a = baseline_1x["under_2a_percent"].iloc[0]
    else:
        baseline_geometric = 3.6801  # Fallback to known baselines
        baseline_tm = 0.8384
        baseline_rmsd = 2.865
        baseline_2a = 78.8

    for budget in compute_budgets:
        budget_data = method_data[method_data["num_branches"] == budget]
        if len(budget_data) > 0:
            geometric_scaling[method]["geometric_score"].append(
                budget_data["geometric_score"].iloc[0]
            )
            geometric_scaling[method]["tm_score"].append(
                budget_data["tm_score"].iloc[0]
            )
            geometric_scaling[method]["rmsd"].append(budget_data["rmsd"].iloc[0])
            geometric_scaling[method]["under_2a_percent"].append(
                budget_data["under_2a_percent"].iloc[0]
            )
        else:
            # Fill with baseline values if no data for this budget
            if budget == 1:
                geometric_scaling[method]["geometric_score"].append(baseline_geometric)
                geometric_scaling[method]["tm_score"].append(baseline_tm)
                geometric_scaling[method]["rmsd"].append(baseline_rmsd)
                geometric_scaling[method]["under_2a_percent"].append(baseline_2a)
            else:
                geometric_scaling[method]["geometric_score"].append(np.nan)
                geometric_scaling[method]["tm_score"].append(np.nan)
                geometric_scaling[method]["rmsd"].append(np.nan)
                geometric_scaling[method]["under_2a_percent"].append(np.nan)

# ══════════════════════════════════════════════════════════════════════════════
#                     P L O T T I N G   F U N C T I O N S
# ══════════════════════════════════════════════════════════════════════════════


def plot_protein_scaling():
    """Plot protein scaling results with modern styling."""

    # Plot mean TM-score vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(protein_scaling.items()):
        # Filter out NaN values
        valid_indices = [
            j for j, score in enumerate(data["mean_score"]) if not np.isnan(score)
        ]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_scores = [data["mean_score"][j] for j in valid_indices]

        if valid_scores:  # Only plot if we have valid data
            color = colors.get(method, colors["baseline"])
            marker = scaling_markers[i % len(scaling_markers)]

            ax.plot(
                valid_budgets,
                valid_scores,
                marker=marker,
                label=method,  # Use standardized method names directly
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

    ax.set_xlabel("Inference Compute Budget")
    ax.set_xticks(compute_budgets)
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.set_ylabel("Mean TM-Score\n(higher is better)")
    ax.set_title("Protein Structure Quality vs. Compute Budget\nTM Verifier", pad=20)

    # Improve legend
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        loc="lower right",
    )
    legend.get_frame().set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Set y-axis limits to accommodate new data range
    ax.set_ylim(0.78, 0.91)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_tm_score.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


def plot_protein_rmsd():
    """Plot protein RMSD scaling results with modern styling."""

    # Plot RMSD vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(protein_scaling.items()):
        # Filter out NaN values
        valid_indices = [j for j, rmsd in enumerate(data["rmsd"]) if not np.isnan(rmsd)]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_rmsd = [data["rmsd"][j] for j in valid_indices]

        if valid_rmsd:  # Only plot if we have valid data
            color = colors.get(method, colors["baseline"])
            marker = scaling_markers[i % len(scaling_markers)]

            ax.plot(
                valid_budgets,
                valid_rmsd,
                marker=marker,
                label=method,  # Use standardized method names directly
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

    ax.set_xlabel("Inference Compute Budget")
    ax.set_xticks(compute_budgets)
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.set_ylabel("RMSD (Å)\n(lower is better)")
    ax.set_title("Protein Structure RMSD vs. Compute Budget\nTM Verifier", pad=20)

    # Improve legend
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Set y-axis limits to accommodate RMSD data range
    ax.set_ylim(1.8, 2.8)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_rmsd.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


def plot_protein_2a_percent():
    """Plot protein <2Å percentage scaling results with modern styling."""

    # Plot <2Å percentage vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(protein_scaling.items()):
        # Filter out NaN values
        valid_indices = [
            j
            for j, percent in enumerate(data["under_2a_percent"])
            if not np.isnan(percent)
        ]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_percent = [data["under_2a_percent"][j] for j in valid_indices]

        if valid_percent:  # Only plot if we have valid data
            color = colors.get(method, colors["baseline"])
            marker = scaling_markers[i % len(scaling_markers)]

            ax.plot(
                valid_budgets,
                valid_percent,
                marker=marker,
                label=method,  # Use standardized method names directly
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

    ax.set_xlabel("Inference Compute Budget")
    ax.set_xticks(compute_budgets)
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.set_ylabel("Structures <2Å (%)\n(higher is better)")
    ax.set_title("Protein Structure Accuracy vs. Compute Budget\nTM Verifier", pad=20)

    # Improve legend
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        loc="lower right",
    )
    legend.get_frame().set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Set y-axis limits to accommodate percentage data range
    ax.set_ylim(76, 92)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_2a_percent.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


def plot_geometric_scaling():
    """Plot geometric scoring scaling results with modern styling."""

    # Plot geometric score vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(geometric_scaling.items()):
        # Filter out NaN values
        valid_indices = [
            j for j, score in enumerate(data["geometric_score"]) if not np.isnan(score)
        ]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_scores = [data["geometric_score"][j] for j in valid_indices]

        if valid_scores:  # Only plot if we have valid data
            color = colors.get(method, colors["baseline"])
            marker = scaling_markers[i % len(scaling_markers)]

            ax.plot(
                valid_budgets,
                valid_scores,
                marker=marker,
                label=method,  # Use standardized method names directly
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

    ax.set_xlabel("Inference Compute Budget")
    ax.set_xticks(compute_budgets)
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.set_ylabel("Geometric Score\n(higher is better)")
    ax.set_title(
        "Protein Structure Quality vs. Compute Budget\nGeometric Verifier",
        pad=20,
    )

    # Improve legend
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        loc="lower right",
    )
    legend.get_frame().set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Set y-axis limits to accommodate geometric score data range
    ax.set_ylim(3.6, 4.6)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_geometric.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


def plot_geometric_tm_score():
    """Plot TM-score scaling results from geometric scoring experiments."""

    # Plot TM-score vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(geometric_scaling.items()):
        # Filter out NaN values
        valid_indices = [
            j for j, score in enumerate(data["tm_score"]) if not np.isnan(score)
        ]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_scores = [data["tm_score"][j] for j in valid_indices]

        if valid_scores:  # Only plot if we have valid data
            color = colors.get(method, colors["baseline"])
            marker = scaling_markers[i % len(scaling_markers)]

            ax.plot(
                valid_budgets,
                valid_scores,
                marker=marker,
                label=method,
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

    ax.set_xlabel("Inference Compute Budget")
    ax.set_xticks(compute_budgets)
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.set_ylabel("TM-Score\n(higher is better)")
    ax.set_title(
        "Protein TM-Score vs. Compute Budget\nGeometric Verifier",
        pad=20,
    )

    # Improve legend
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        loc="lower left",
    )
    legend.get_frame().set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Set y-axis limits to accommodate TM-score data range
    ax.set_ylim(0.76, 0.87)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_geometric_tm.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


def plot_geometric_rmsd():
    """Plot RMSD scaling results from geometric scoring experiments."""

    # Plot RMSD vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(geometric_scaling.items()):
        # Filter out NaN values
        valid_indices = [j for j, rmsd in enumerate(data["rmsd"]) if not np.isnan(rmsd)]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_rmsd = [data["rmsd"][j] for j in valid_indices]

        if valid_rmsd:  # Only plot if we have valid data
            color = colors.get(method, colors["baseline"])
            marker = scaling_markers[i % len(scaling_markers)]

            ax.plot(
                valid_budgets,
                valid_rmsd,
                marker=marker,
                label=method,
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

    ax.set_xlabel("Inference Compute Budget")
    ax.set_xticks(compute_budgets)
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.set_ylabel("RMSD (Å)\n(lower is better)")
    ax.set_title(
        "Protein RMSD vs. Compute Budget\nGeometric Verifier",
        pad=20,
    )

    # Improve legend
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        loc="upper left",
    )
    legend.get_frame().set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Set y-axis limits to accommodate RMSD data range
    ax.set_ylim(2.3, 3.8)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_geometric_rmsd.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


def plot_geometric_2a_percent():
    """Plot <2Å percentage scaling results from geometric scoring experiments."""

    # Plot <2Å percentage vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(geometric_scaling.items()):
        # Filter out NaN values
        valid_indices = [
            j
            for j, percent in enumerate(data["under_2a_percent"])
            if not np.isnan(percent)
        ]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_percent = [data["under_2a_percent"][j] for j in valid_indices]

        if valid_percent:  # Only plot if we have valid data
            color = colors.get(method, colors["baseline"])
            marker = scaling_markers[i % len(scaling_markers)]

            ax.plot(
                valid_budgets,
                valid_percent,
                marker=marker,
                label=method,
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

    ax.set_xlabel("Inference Compute Budget")
    ax.set_xticks(compute_budgets)
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.set_ylabel("Structures <2Å (%)\n(higher is better)")
    ax.set_title(
        "Protein Structure Accuracy vs. Compute Budget\nGeometric Verifier",
        pad=20,
    )

    # Improve legend
    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        bbox_to_anchor=(0.25, 1.0),
        loc="upper left",
    )
    legend.get_frame().set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Set y-axis limits to accommodate percentage data range
    ax.set_ylim(68, 84)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_geometric_2a.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
#                                 MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating modern, publication-ready protein plots...")
    print(f"Saving all plots as PDF files in: {OUTPUT_DIR}/")
    print()

    # TM-score experiments (original data)
    plot_protein_scaling()
    plot_protein_rmsd()
    plot_protein_2a_percent()

    # Geometric scoring experiments
    plot_geometric_scaling()
    plot_geometric_tm_score()
    plot_geometric_rmsd()
    plot_geometric_2a_percent()

    print()
    print("All protein plots generated and saved as PDF files!")
    print(f"Files saved in {OUTPUT_DIR}/:")
    print("TM-score experiments:")
    print("- protein_scaling_tm_score.pdf")
    print("- protein_scaling_rmsd.pdf")
    print("- protein_scaling_2a_percent.pdf")
    print("Geometric scoring experiments:")
    print("- protein_scaling_geometric.pdf")
    print("- protein_scaling_geometric_tm.pdf")
    print("- protein_scaling_geometric_rmsd.pdf")
    print("- protein_scaling_geometric_2a.pdf")

    # Optionally show plots as well
    plt.show()
