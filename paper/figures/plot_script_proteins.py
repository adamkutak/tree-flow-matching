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
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 20,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "axes.spines.top": False,
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

# Modern color palette for protein methods
colors = {
    "standard": "#7F8C8D",  # Gray (baseline)
    "Best-of-N": "#E67E22",  # Orange
    "Path-DivFree": "#9B59B6",  # Purple
    "Path-SDE": "#E74C3C",  # Red
    "BestN+Path-DivFree": "#1ABC9C",  # Teal (two-stage)
    # Keep old names for backward compatibility
    "best_of_n": "#E67E22",  # Orange (Random Search)
    "divergence_free_ode": "#9B59B6",  # Purple
    "sde_path_exploration": "#E74C3C",  # Red
    "random_search_divfree": "#1ABC9C",  # Teal (two-stage)
}

# Modern markers
markers = ["o", "s", "^", "v", "D"]

# ══════════════════════════════════════════════════════════════════════════════
# PROTEIN EXPERIMENT DATA
# ══════════════════════════════════════════════════════════════════════════════

# Raw data from CSV (with updated naming conventions)
protein_data = {
    "method": [
        "standard",
        "Path-SDE",
        "BestN+Path-DivFree",
        "BestN+Path-DivFree",
        "Path-SDE",
        "Path-DivFree",
        "BestN+Path-DivFree",
        "Best-of-N",
        "Best-of-N",
        "Path-DivFree",
        "Path-SDE",
        "Best-of-N",
        "Path-DivFree",
    ],
    "num_branches": [1, 2, 2, 4, 4, 2, 8, 2, 4, 4, 8, 8, 8],
    "mean_score": [
        0.7432176647340222,
        0.7031350449163078,
        0.8298524341413012,
        0.8576552256607168,
        0.7177283341351568,
        0.7364550524971643,
        0.8680097959523632,
        0.803324141059262,
        0.8480285199567403,
        0.7717239957891483,
        0.7306474231318094,
        0.8639875398116925,
        0.7864018947859729,
    ],
    "std_score": [
        0.10818327826169723,
        0.10578621455787038,
        0.04672668552036438,
        0.027181748126285342,
        0.11069071276024542,
        0.12046299444177064,
        0.020120420627146345,
        0.059058327620999725,
        0.03966702700529788,
        0.1056420740338263,
        0.12575645273489347,
        0.018707223787889877,
        0.1047866075281783,
    ],
    "max_score": [
        0.9032795844357923,
        0.8721609019118262,
        0.9019136337952417,
        0.8984608030507292,
        0.8787435853366183,
        0.8923708418046031,
        0.904651026988976,
        0.8996845734758369,
        0.8928743092175654,
        0.8981364295692996,
        0.8895478715020056,
        0.907945252483447,
        0.8957684231299327,
    ],
    "min_score": [
        0.46252091441562126,
        0.45468767220402806,
        0.7186318999737344,
        0.789763226518424,
        0.43463616636361774,
        0.44927238163324595,
        0.8060360718637778,
        0.6070269212688462,
        0.6983093211256493,
        0.4665923573941706,
        0.4386196057960319,
        0.8053360624118574,
        0.45065893037649524,
    ],
}

# Convert to DataFrame for easier processing
df = pd.DataFrame(protein_data)

# Organize data by method and compute budget
compute_budgets = [1, 2, 4, 8]

protein_scaling = {}

# Process each method
for method in df["method"].unique():
    method_data = df[df["method"] == method]
    protein_scaling[method] = {
        "mean_score": [],
        "std_score": [],
        "max_score": [],
        "min_score": [],
    }

    # Get the standard method's 1× baseline value (same for all methods)
    standard_1x = df[df["method"] == "standard"]
    standard_baseline = standard_1x[standard_1x["num_branches"] == 1][
        "mean_score"
    ].iloc[0]

    for budget in compute_budgets:
        budget_data = method_data[method_data["num_branches"] == budget]
        if len(budget_data) > 0:
            protein_scaling[method]["mean_score"].append(
                budget_data["mean_score"].iloc[0]
            )
            protein_scaling[method]["std_score"].append(
                budget_data["std_score"].iloc[0]
            )
            protein_scaling[method]["max_score"].append(
                budget_data["max_score"].iloc[0]
            )
            protein_scaling[method]["min_score"].append(
                budget_data["min_score"].iloc[0]
            )
        else:
            # Fill with standard baseline value if no data for this budget
            protein_scaling[method]["mean_score"].append(standard_baseline)
            protein_scaling[method]["std_score"].append(np.nan)
            protein_scaling[method]["max_score"].append(np.nan)
            protein_scaling[method]["min_score"].append(np.nan)

# ══════════════════════════════════════════════════════════════════════════════
#                     P L O T T I N G   F U N C T I O N S
# ══════════════════════════════════════════════════════════════════════════════


def plot_protein_scaling():
    """Plot protein scaling results with modern styling."""

    # Plot mean TM-score vs compute budget
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (method, data) in enumerate(protein_scaling.items()):
        # Skip the standard method as it's just the baseline
        if method == "standard":
            continue

        # Filter out NaN values
        valid_indices = [
            j for j, score in enumerate(data["mean_score"]) if not np.isnan(score)
        ]
        valid_budgets = [compute_budgets[j] for j in valid_indices]
        valid_scores = [data["mean_score"][j] for j in valid_indices]

        if valid_scores:  # Only plot if we have valid data
            color = colors.get(method, colors["standard"])
            marker = markers[i % len(markers)]

            # Create cleaner labels for display
            display_labels = {
                "Best-of-N": "Best-of-N",
                "Path-DivFree": "Path-DivFree",
                "Path-SDE": "Path-SDE",
                "BestN+Path-DivFree": "BestN+Path-DivFree",
            }
            display_label = display_labels.get(method, method.replace("_", " ").title())

            ax.plot(
                valid_budgets,
                valid_scores,
                marker=marker,
                label=display_label,
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
    ax.set_title(
        "Protein Structure Quality vs. Compute Budget\nFoldFlow Experiments", pad=20
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

    # Set y-axis limits starting at 0.7
    ax.set_ylim(0.7, 0.95)

    plt.tight_layout()

    # Save as PDF
    filename = "protein_scaling_tm_score.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
#                                 MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating modern, publication-ready protein plots...")
    print(f"Saving all plots as PDF files in: {OUTPUT_DIR}/")
    print()

    plot_protein_scaling()

    print()
    print("All protein plots generated and saved as PDF files!")
    print(f"Files saved in {OUTPUT_DIR}/:")
    print("- protein_scaling_tm_score.pdf")

    # Optionally show plots as well
    plt.show()
