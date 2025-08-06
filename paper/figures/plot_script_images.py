"""
Complete plotting script (Matplotlib)
-------------------------------------

•  Noise & diversity study
    – Diversity vs. average-noise
    – FID vs. average-noise
    – Inception Score vs. average-noise
    – Pareto scatter (Diversity vs. –FID)
    – Pareto scatter (Diversity vs. Inception Score)

•  Inference-time-scaling experiments
    – Four metrics (FID, IS, DINO-Top-1, DINO-Top-5) under
      * Inception-scored branching
      * DINO-scored branching  (with corrected ODE-divfree run)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
import os

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

# Modern color palette
colors = {
    "SDE": "#E74C3C",  # Red
    "EDM-SDE": "#3498DB",  # Blue
    "Score-SDE": "#2ECC71",  # Green
    "ODE-divfree": "#9B59B6",  # Purple
    "Random Search": "#E67E22",  # Orange
    "ODE-divfree explore": "#9B59B6",
    "RS → Divfree explore": "#1ABC9C",  # Teal
    "Score-SDE explore": "#2ECC71",
    "SDE explore": "#E74C3C",
    "baseline": "#7F8C8D",  # Gray
}

# Modern markers
markers = {"SDE": "o", "EDM-SDE": "s", "Score-SDE": "^", "ODE-divfree": "D"}
scaling_markers = ["o", "s", "^", "v", "D"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NOISE & DIVERSITY STUDY DATA
# ══════════════════════════════════════════════════════════════════════════════

noise_data = {
    "SDE": {
        "ratio": [
            0.026495412989849292,
            0.03459538257986627,
            0.042616321834589986,
            0.053149880542785834,
            0.10766627382872786,
            0.16100460925205573,
        ],
        "fid": [
            89.8433609008789,
            89.70419311523438,
            91.51818084716797,
            95.59479522705078,
            143.19992065429688,
            226.57449340820312,
        ],
        "is": [
            35.07704544067383,
            34.13245391845703,
            34.4217529296875,
            31.621051788330078,
            13.075517654418945,
            3.8153419494628906,
        ],
        "div": [0.2637, 0.3348, 0.3947, 0.4857, 0.9015, 1.0024],
    },
    "EDM-SDE": {
        "ratio": [
            0.04140599493948965,
            0.057315890755118136,
            0.0766857133526424,
            0.08767313190019974,
            0.0964763010033051,
        ],
        "fid": [
            97.92323303222656,
            108.22187042236328,
            139.69790649414062,
            165.0597686767578,
            185.19456481933594,
        ],
        "is": [
            29.722951889038086,
            24.604198455810547,
            13.012596130371094,
            9.770713806152344,
            6.278625011444092,
        ],
        "div": [0.2661, 0.3693, 0.4974, 0.5432, 0.5810],
    },
    "Score-SDE": {
        "ratio": [
            0.008866081100947863,
            0.01771476173718701,
            0.0262853484629103,
        ],
        "fid": [86.58537292480469, 85.50243377685547, 90.74525451660156],
        "is": [35.74665832519531, 36.71392822265625, 33.67108154296875],
        "div": [0.0675, 0.1218, 0.1698],
    },
    "ODE-divfree": {
        "ratio": [
            0.11505767067015196,
            0.40381881547083676,
            0.46182786848423735,
            0.5216402821562226,
            0.5816642603221643,
            0.6949972820713578,
            2.334298904952883,
        ],
        "fid": [
            89.63546752929688,
            90.77523040771484,
            88.00147247314453,
            90.05825805664062,
            92.40699768066406,
            90.43621826171875,
            177.32891845703125,
        ],
        "is": [
            34.97759246826172,
            34.92412567138672,
            34.55693817138672,
            34.444786071777344,
            34.935997009277344,
            34.254276275634766,
            8.825911521911621,
        ],
        "div": [0.0682, 0.2176, 0.2455, 0.2665, 0.2939, 0.3421, 0.9455],
    },
}

baseline_fid = 90.45425415039062
baseline_is = 38.18083572387695
ode_random_div = 2.5682


# ══════════════════════════════════════════════════════════════════════════════
# 2.  INFERENCE-TIME-SCALING EXPERIMENTS  (INCEPTION-scored)
# ══════════════════════════════════════════════════════════════════════════════

compute = [1, 2, 4, 8]  # branches (plotted as 1× … 8×)

# common baseline for 1× in ALL curves
incp_baseline = {
    "fid": 54.3839645386,
    "is": 36.2285728455,
    "top1": 66.40625,
    "top5": 83.203125,
}

incp_scaling = {
    "Random Search": {
        "fid": [incp_baseline["fid"], 54.7576789856, 52.5880737305, 50.7637405396],
        "is": [incp_baseline["is"], 47.9309883118, 66.1153640747, 75.6240463257],
        "top1": [incp_baseline["top1"], 67.28515625, 75.9765625, 80.17578125],
        "top5": [incp_baseline["top5"], 84.765625, 89.74609375, 92.87109375],
    },
    "ODE-divfree explore": {
        "fid": [incp_baseline["fid"], 52.6444015503, 58.6198463440, 61.1252670288],
        "is": [incp_baseline["is"], 65.5407104492, 78.7351913452, 85.6708831787],
        "top1": [incp_baseline["top1"], 68.1640625, 63.28125, 67.67578125],
        "top5": [incp_baseline["top5"], 85.83984375, 81.54296875, 83.3984375],
    },
    "RS → Divfree explore": {
        "fid": [incp_baseline["fid"], 49.6474685669, 54.0792846680, 61.8381042480],
        "is": [incp_baseline["is"], 70.3865203857, 86.1988906860, 92.1974639893],
        "top1": [incp_baseline["top1"], 73.4375, 80.17578125, 84.66796875],
        "top5": [incp_baseline["top5"], 89.453125, 92.48046875, 92.67578125],
    },
    "Score-SDE explore": {
        "fid": [incp_baseline["fid"], 53.9133262634, 54.7942695618, 54.3388023376],
        "is": [incp_baseline["is"], 47.0511093140, 55.4215469360, 64.1331634521],
        "top1": [incp_baseline["top1"], 66.89453125, 62.6953125, 64.74609375],
        "top5": [incp_baseline["top5"], 82.71484375, 82.8125, 83.3984375],
    },
    "SDE explore": {
        "fid": [incp_baseline["fid"], 54.1602745056, 58.5168266296, 62.2201728821],
        "is": [incp_baseline["is"], 64.1173782349, 80.1871185303, 87.8954238892],
        "top1": [incp_baseline["top1"], 65.4296875, 68.75, 67.96875],
        "top5": [incp_baseline["top5"], 83.30078125, 84.9609375, 85.7421875],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  INFERENCE-TIME-SCALING EXPERIMENTS  (DINO-scored, *corrected* divfree)
# ══════════════════════════════════════════════════════════════════════════════

dino_baseline = {  # Random-Search 1× under DINO-scoring
    "fid": 52.6667022705,
    "is": 36.0008239746,
    "top1": 63.76953125,
    "top5": 83.7890625,
}

dino_scaling = {
    "Random Search": {
        "fid": [dino_baseline["fid"], 51.6026611328, 46.9245452881, 44.7447700500],
        "is": [dino_baseline["is"], 39.0032386780, 50.9636688232, 56.4447441101],
        "top1": [dino_baseline["top1"], 71.09375, 86.71875, 93.1640625],
        "top5": [dino_baseline["top5"], 89.84375, 97.55859375, 99.70703125],
    },
    "ODE-divfree explore": {  # corrected run
        "fid": [dino_baseline["fid"], 50.1412963867, 49.0300178528, 47.7467346191],
        "is": [dino_baseline["is"], 40.6086311340, 41.4910354614, 42.9033813477],
        "top1": [dino_baseline["top1"], 79.6875, 89.453125, 91.9921875],
        "top5": [dino_baseline["top5"], 94.43359375, 97.0703125, 98.4375],
    },
    "RS → Divfree explore": {
        "fid": [dino_baseline["fid"], 45.0442619324, 44.4675903320, 45.2783470154],
        "is": [dino_baseline["is"], 48.6486282349, 59.8598442078, 61.2256698608],
        "top1": [dino_baseline["top1"], 88.671875, 97.4609375, 98.92578125],
        "top5": [dino_baseline["top5"], 97.75390625, 100.0, 100.0],
    },
    "Score-SDE explore": {
        "fid": [dino_baseline["fid"], 51.0865936279, 52.2667312622, 50.8811492920],
        "is": [dino_baseline["is"], 38.8347549438, 39.3962249756, 39.6080360413],
        "top1": [dino_baseline["top1"], 71.09375, 75.68359375, 79.6875],
        "top5": [dino_baseline["top5"], 88.671875, 90.13671875, 93.45703125],
    },
    "SDE explore": {
        "fid": [dino_baseline["fid"], 50.9161262512, 48.6213989258, 47.5634880066],
        "is": [dino_baseline["is"], 39.5118789673, 43.5768890381, 46.3124923706],
        "top1": [dino_baseline["top1"], 78.80859375, 89.6484375, 93.5546875],
        "top5": [dino_baseline["top5"], 93.1640625, 97.8515625, 98.92578125],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#                     P L O T T I N G   F U N C T I O N S
# ══════════════════════════════════════════════════════════════════════════════


def plot_noise_study():
    """Plot noise study results with modern styling."""
    # 1A: Diversity • 1B: FID • 1C: IS
    for metric, ylabel, title, better in [
        ("div", "Sample Diversity", "Sample Diversity vs. Average Noise", "higher"),
        ("fid", "FID Score", "FID vs. Average Noise", "lower"),
        ("is", "Inception Score", "Inception Score vs. Average Noise", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for label, m in noise_data.items():
            ax.plot(
                m["ratio"],
                m[metric],
                marker=markers[label],
                label=label,
                color=colors[label],
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        # Add baseline lines with improved styling
        if metric == "fid":
            ax.axhline(
                baseline_fid,
                color=colors["baseline"],
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                label="Baseline ODE",
            )
        if metric == "is":
            ax.axhline(
                baseline_is,
                color=colors["baseline"],
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                label="Baseline ODE",
            )
        if metric == "div":
            ax.axhline(
                ode_random_div,
                color=colors["baseline"],
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                label="ODE-random (max)",
            )

        ax.set_xlabel("Average Noise Magnitude per Step")
        ax.set_ylabel(f"{ylabel}\n({better} is better)")
        ax.set_title(title, pad=20)

        # Improve legend
        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(1.2)

        # Add subtle background
        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        # Save as PDF
        filename = f"noise_study_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")


def plot_pareto():
    """Plot Pareto scatter plots with modern styling."""

    # Pareto 1: Diversity vs –FID
    fig, ax = plt.subplots(figsize=(10, 8))

    for label, m in noise_data.items():
        inv_fid = [-v for v in m["fid"]]  # ↑ is better now
        scatter = ax.scatter(
            m["div"],
            inv_fid,
            label=label,
            c=colors[label],
            s=120,
            alpha=0.8,
            marker=markers[label],
            edgecolors="white",
            linewidth=2,
        )

    # Add baseline line for FID
    ax.axhline(
        -baseline_fid,
        color=colors["baseline"],
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label="Baseline ODE",
    )

    ax.set_xlabel("Sample Diversity\n(higher is better)")
    ax.set_ylabel("Negative FID Score\n(higher is better)")
    ax.set_title("Pareto Frontier: Sample Diversity vs. Image Quality", pad=20)

    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
    )
    legend.get_frame().set_linewidth(1.2)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()

    # Save as PDF
    filename = "pareto_diversity_vs_fid.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")

    # Pareto 2: Diversity vs IS
    fig, ax = plt.subplots(figsize=(10, 8))

    for label, m in noise_data.items():
        scatter = ax.scatter(
            m["div"],
            m["is"],
            label=label,
            c=colors[label],
            s=120,
            alpha=0.8,
            marker=markers[label],
            edgecolors="white",
            linewidth=2,
        )

    # Add baseline line for Inception Score
    ax.axhline(
        baseline_is,
        color=colors["baseline"],
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label="Baseline ODE",
    )

    ax.set_xlabel("Sample Diversity\n(higher is better)")
    ax.set_ylabel("Inception Score\n(higher is better)")
    ax.set_title(
        "Pareto Frontier: Sample Diversity vs. Inception Score",
        pad=20,
    )

    legend = ax.legend(
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
    )
    legend.get_frame().set_linewidth(1.2)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()

    # Save as PDF
    filename = "pareto_diversity_vs_inception.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")


def plot_scaling(curves, title_suffix):
    """Plot inference scaling results with modern styling."""

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for (method, data), mk in zip(curves.items(), scaling_markers):
            color = colors.get(method, colors["baseline"])
            ax.plot(
                compute,
                data[metric],
                marker=mk,
                label=method,
                color=color,
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Inference Compute Budget")
        ax.set_xticks(compute)
        ax.set_xticklabels(["1×", "2×", "4×", "8×"])
        ax.set_ylabel(f"{lab}\n({better} is better)")
        ax.set_title(f"{lab} vs. Compute Budget\n{title_suffix}", pad=20)

        # Improve legend
        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(1.2)

        # Add subtle background
        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        # Save as PDF
        scoring_type = "inception" if "Inception" in title_suffix else "dino"
        filename = f"scaling_{scoring_type}_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
#                                 MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating modern, publication-ready plots...")
    print(f"Saving all plots as PDF files in: {OUTPUT_DIR}/")
    print()

    plot_noise_study()
    plot_pareto()

    plot_scaling(incp_scaling, "Inception-scored branching")
    plot_scaling(dino_scaling, "DINO-scored branching (corrected)")

    print()
    print("All plots generated and saved as PDF files!")
    print(f"Files saved in {OUTPUT_DIR}/:")
    print("- noise_study_div.pdf")
    print("- noise_study_fid.pdf")
    print("- noise_study_is.pdf")
    print("- pareto_diversity_vs_fid.pdf")
    print("- pareto_diversity_vs_inception.pdf")
    print("- scaling_inception_fid.pdf")
    print("- scaling_inception_is.pdf")
    print("- scaling_inception_top1.pdf")
    print("- scaling_inception_top5.pdf")
    print("- scaling_dino_fid.pdf")
    print("- scaling_dino_is.pdf")
    print("- scaling_dino_top1.pdf")
    print("- scaling_dino_top5.pdf")

    # Optionally show plots as well
    plt.show()
