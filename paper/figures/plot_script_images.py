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
    "Best-of-N": "#E67E22",  # Orange
    "Path-DivFree": "#9B59B6",  # Purple
    "Path-SDE": "#E74C3C",  # Red
    "Path-ScoreSDE": "#2ECC71",  # Green
    "BestN+Path-DivFree": "#1ABC9C",  # Teal
    "NoiseSearch-3R-DivFree": "#F39C12",  # Gold
    "BestN+NoiseSearch-3R-DivFree": "#8E44AD",  # Dark Purple
    # Keep old names for noise study compatibility
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
    "Best-of-N": {
        "fid": [incp_baseline["fid"], 53.2056465149, 50.8040390015, 52.0526847839],
        "is": [incp_baseline["is"], 50.1982078552, 63.9881401062, 74.6413650513],
        "top1": [incp_baseline["top1"], 67.48046875, 73.33984375, 80.46875],
        "top5": [incp_baseline["top5"], 84.86328125, 91.2109375, 91.6015625],
    },
    "Path-DivFree": {
        "fid": [incp_baseline["fid"], 54.8920936584, 58.4799156189, 68.3677062988],
        "is": [incp_baseline["is"], 63.2807121277, 83.5469818115, 89.4558868408],
        "top1": [incp_baseline["top1"], 63.0859375, 69.140625, 66.015625],
        "top5": [incp_baseline["top5"], 82.71484375, 83.88671875, 83.49609375],
    },
    "BestN+Path-DivFree": {
        "fid": [incp_baseline["fid"], 51.0203018188, 56.5868797302, 68.6828994751],
        "is": [incp_baseline["is"], 71.8548355103, 87.8421020508, 91.9832916260],
        "top1": [incp_baseline["top1"], 74.12109375, 81.15234375, 84.375],
        "top5": [incp_baseline["top5"], 87.98828125, 92.48046875, 95.1171875],
    },
    "Path-ScoreSDE": {
        "fid": [incp_baseline["fid"], 53.4748840332, 52.7677459717, 55.4086990356],
        "is": [incp_baseline["is"], 49.7877578735, 61.4951171875, 67.5772705078],
        "top1": [incp_baseline["top1"], 67.3828125, 65.52734375, 63.0859375],
        "top5": [incp_baseline["top5"], 84.1796875, 83.10546875, 84.27734375],
    },
    "Path-SDE": {
        "fid": [incp_baseline["fid"], 56.4615898132, 60.6950569153, 67.4587478638],
        "is": [incp_baseline["is"], 64.1559677124, 81.0101165771, 88.5352249146],
        "top1": [incp_baseline["top1"], 62.6953125, 65.33203125, 68.65234375],
        "top5": [incp_baseline["top5"], 81.640625, 84.1796875, 84.765625],
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
    "Best-of-N": {
        "fid": [dino_baseline["fid"], 51.8314666748, 46.8698577881, 46.2298698425],
        "is": [dino_baseline["is"], 40.1176834106, 52.1529541016, 56.6535110474],
        "top1": [dino_baseline["top1"], 75.1953125, 84.9609375, 93.45703125],
        "top5": [dino_baseline["top5"], 91.69921875, 97.36328125, 99.609375],
    },
    "Path-DivFree": {  # corrected run
        "fid": [dino_baseline["fid"], 51.2642211914, 48.0354537964, 46.9875450134],
        "is": [dino_baseline["is"], 41.1047897339, 45.5917053223, 45.5746231079],
        "top1": [dino_baseline["top1"], 83.203125, 91.11328125, 96.38671875],
        "top5": [dino_baseline["top5"], 94.62890625, 97.94921875, 99.21875],
    },
    "BestN+Path-DivFree": {
        "fid": [dino_baseline["fid"], 45.4849014282, 44.1251449585, 46.7672691345],
        "is": [dino_baseline["is"], 49.0139312744, 59.3470230103, 63.7213439941],
        "top1": [dino_baseline["top1"], 89.74609375, 98.14453125, 99.51171875],
        "top5": [dino_baseline["top5"], 98.53515625, 100.0, 100.0],
    },
    "Path-ScoreSDE": {
        "fid": [dino_baseline["fid"], 53.7896003723, 52.7023506165, 51.4293212891],
        "is": [dino_baseline["is"], 37.2301712036, 36.2029876709, 38.8587951660],
        "top1": [dino_baseline["top1"], 72.265625, 77.05078125, 81.0546875],
        "top5": [dino_baseline["top5"], 88.671875, 92.578125, 94.04296875],
    },
    "Path-SDE": {
        "fid": [dino_baseline["fid"], 50.4010772705, 49.0975723267, 47.7479972839],
        "is": [dino_baseline["is"], 42.1980285645, 47.1010398865, 47.0342483521],
        "top1": [dino_baseline["top1"], 83.3984375, 92.1875, 94.62890625],
        "top5": [dino_baseline["top5"], 95.21484375, 99.0234375, 99.609375],
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
