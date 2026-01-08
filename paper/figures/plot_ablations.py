"""
Ablation Study Plots (Matplotlib)
---------------------------------

Plots for appendix ablation studies:
1. Branching Schedule Ablation (trajectory count)
2. Non-Uniform Branching Ablation (more branches at later timesteps)
3. Classifier-Free Guidance (CFG) Ablation
4. CFG Search Ablation (using CFG variation for branching instead of noise)
5. Coarse Simulate Forward Ablation (fewer timesteps for reward simulation, dt=0.05)
6. Coarse Simulate Forward Fine Ablation (dt=0.01 with coarser simulate forward)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
import os

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
        "legend.fontsize": 16,
        "figure.titlesize": 22,
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

OUTPUT_DIR = "paper/figures/ablations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette for ablations
ablation_colors = {
    "random_search": "#E67E22",
    "schedule_3.55": "#3498DB",
    "schedule_2.15": "#2ECC71",
    "schedule_1.85": "#9B59B6",
    "schedule_1.55": "#E74C3C",
}

ablation_markers = {
    "random_search": "o",
    "schedule_3.55": "s",
    "schedule_2.15": "^",
    "schedule_1.85": "D",
    "schedule_1.55": "v",
}

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 1: BRANCHING SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

compute = [1, 2, 4, 8]

schedule_ablation_data = {
    "random_search": {
        "fid": [135.4319, 132.4169, 131.2982, 124.1121],
        "is": [15.0104, 15.9707, 17.5176, 18.4058],
        "top1": [62.890625, 72.265625, 86.328125, 91.40625],
        "top5": [80.46875, 93.75, 98.4375, 98.4375],
    },
    "schedule_3.55": {
        "fid": [135.4319, 135.9585, 128.4843, 129.1102],
        "is": [15.0104, 16.1210, 18.4147, 19.1162],
        "top1": [62.890625, 87.109375, 94.921875, 99.21875],
        "top5": [80.46875, 97.265625, 100.0, 100.0],
    },
    "schedule_2.15": {
        "fid": [135.4319, 134.0955, 130.3274, 128.3825],
        "is": [15.0104, 16.6459, 17.6494, 18.4822],
        "top1": [62.890625, 84.765625, 92.1875, 94.53125],
        "top5": [80.46875, 95.703125, 99.609375, 100.0],
    },
    "schedule_1.85": {
        "fid": [135.4319, 135.2272, 126.7962, 126.9620],
        "is": [15.0104, 16.3930, 17.8783, 19.2952],
        "top1": [62.890625, 85.15625, 93.359375, 97.265625],
        "top5": [80.46875, 96.875, 100.0, 99.609375],
    },
    "schedule_1.55": {
        "fid": [135.4319, 135.7198, 126.6267, 128.4208],
        "is": [15.0104, 15.7874, 18.0136, 18.8659],
        "top1": [62.890625, 79.296875, 92.578125, 96.09375],
        "top5": [80.46875, 95.3125, 98.828125, 100.0],
    },
}

schedule_labels = {
    "random_search": "Random Search",
    "schedule_3.55": "Schedule ~3.55 traj",
    "schedule_2.15": "Schedule ~2.15 traj",
    "schedule_1.85": "Schedule ~1.85 traj",
    "schedule_1.55": "Schedule ~1.55 traj",
}


def plot_schedule_ablation():
    """Plot branching schedule ablation results."""

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for method, data in schedule_ablation_data.items():
            ax.plot(
                compute,
                data[metric],
                marker=ablation_markers[method],
                label=schedule_labels[method],
                color=ablation_colors[method],
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Inference Compute Budget")
        ax.set_xticks(compute)
        ax.set_xticklabels(["1×", "2×", "4×", "8×"])
        ax.set_ylabel(f"{lab}\n({better} is better)")
        ax.set_title(f"Branching Schedule Ablation: {lab}", pad=20)

        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            loc="best",
        )
        legend.get_frame().set_linewidth(1.2)

        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        filename = f"ablation_schedule_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 2: NON-UNIFORM BRANCHING (more branches at later timesteps)
# ══════════════════════════════════════════════════════════════════════════════

nonuniform_colors = {
    "uniform": "#3498DB",
    "custom_a": "#2ECC71",
    "custom_b": "#9B59B6",
}

nonuniform_markers = {
    "uniform": "s",
    "custom_a": "^",
    "custom_b": "D",
}

nonuniform_ablation_data = {
    "uniform": {
        "fid": [135.4319, 125.6560, 128.3572, 126.2330],
        "is": [15.0104, 16.3510, 18.0409, 18.6289],
        "top1": [62.890625, 89.453125, 92.96875, 97.65625],
        "top5": [80.46875, 98.4375, 100.0, 100.0],
    },
    "custom_a": {
        "fid": [135.4319, 136.3976, 128.9374, 126.0756],
        "is": [15.0104, 15.2330, 17.0828, 18.6125],
        "top1": [62.890625, 77.734375, 88.28125, 95.3125],
        "top5": [80.46875, 92.578125, 98.828125, 100.0],
    },
    "custom_b": {
        "fid": [135.4319, 138.5179, 129.0016, 127.4319],
        "is": [15.0104, 13.7604, 15.9224, 18.6529],
        "top1": [62.890625, 75.0, 87.109375, 98.4375],
        "top5": [80.46875, 91.40625, 98.046875, 100.0],
    },
}

nonuniform_labels = {
    "uniform": "Uniform",
    "custom_a": "Custom A (gradual)",
    "custom_b": "Custom B (aggressive)",
}


def plot_nonuniform_branching_ablation():
    """Plot non-uniform branching ablation results."""

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for method, data in nonuniform_ablation_data.items():
            ax.plot(
                compute,
                data[metric],
                marker=nonuniform_markers[method],
                label=nonuniform_labels[method],
                color=nonuniform_colors[method],
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Inference Compute Budget")
        ax.set_xticks(compute)
        ax.set_xticklabels(["1×", "2×", "4×", "8×"])
        ax.set_ylabel(f"{lab}\n({better} is better)")
        ax.set_title(f"Non-Uniform Branching Ablation: {lab}", pad=20)

        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            loc="best",
        )
        legend.get_frame().set_linewidth(1.2)

        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        filename = f"ablation_nonuniform_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 3: CLASSIFIER-FREE GUIDANCE (CFG)
# ══════════════════════════════════════════════════════════════════════════════

cfg_colors = {
    "random_search": "#E67E22",
    "noise_search": "#3498DB",
    "two_stage": "#2ECC71",
}

cfg_markers = {
    "random_search": "o",
    "noise_search": "s",
    "two_stage": "^",
}

cfg_ablation_data = {
    "random_search": {
        "fid": [136.0240, 128.2897, 133.5634, 137.0680],
        "is": [14.6174, 19.7943, 20.2208, 20.6845],
        "top1": [65.234375, 90.625, 96.09375, 98.4375],
        "top5": [84.375, 99.21875, 100.0, 100.0],
    },
    "noise_search": {
        "fid": [136.0240, 130.2682, 134.3960, 135.6706],
        "is": [14.6174, 19.8201, 20.7385, 21.0627],
        "top1": [65.234375, 97.265625, 97.65625, 99.21875],
        "top5": [84.375, 100.0, 100.0, 100.0],
    },
    "two_stage": {
        "fid": [136.0240, 131.1297, 131.5594, 135.1045],
        "is": [14.6174, 19.6498, 20.4126, 20.7149],
        "top1": [65.234375, 96.875, 98.828125, 98.828125],
        "top5": [84.375, 99.609375, 100.0, 100.0],
    },
}

cfg_labels = {
    "random_search": "Random Search + CFG",
    "noise_search": "NS–DMFM-ODE + CFG",
    "two_stage": "RS+NS–DMFM-ODE + CFG",
}


def plot_cfg_ablation():
    """Plot CFG ablation results."""

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for method, data in cfg_ablation_data.items():
            ax.plot(
                compute,
                data[metric],
                marker=cfg_markers[method],
                label=cfg_labels[method],
                color=cfg_colors[method],
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Inference Compute Budget")
        ax.set_xticks(compute)
        ax.set_xticklabels(["1×", "2×", "4×", "8×"])
        ax.set_ylabel(f"{lab}\n({better} is better)")
        ax.set_title(f"CFG Ablation (scale=1.5): {lab}", pad=20)

        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            loc="best",
        )
        legend.get_frame().set_linewidth(1.2)

        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        filename = f"ablation_cfg_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 4: CFG SEARCH (using CFG variation for branching instead of noise)
# ══════════════════════════════════════════════════════════════════════════════

cfg_search_colors = {
    "random_search": "#E67E22",
    "cfg_search": "#3498DB",
}

cfg_search_markers = {
    "random_search": "o",
    "cfg_search": "s",
}

cfg_search_ablation_data = {
    "random_search": {
        "fid": [133.8048, 129.1111, 131.6277, 135.0231],
        "is": [15.1541, 19.2226, 20.3728, 21.0759],
        "top1": [70.3125, 92.1875, 97.265625, 98.4375],
        "top5": [89.0625, 98.828125, 100.0, 100.0],
    },
    "cfg_search": {
        "fid": [133.8048, 131.3993, 132.1015, 137.9055],
        "is": [15.1541, 19.9692, 21.1566, 20.7995],
        "top1": [70.3125, 96.484375, 98.828125, 98.828125],
        "top5": [89.0625, 99.609375, 100.0, 100.0],
    },
}

cfg_search_labels = {
    "random_search": "Random Search + CFG",
    "cfg_search": "CFG Search (CFG branching)",
}


def plot_cfg_search_ablation():
    """Plot CFG search ablation results (CFG variation for branching)."""

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for method, data in cfg_search_ablation_data.items():
            ax.plot(
                compute,
                data[metric],
                marker=cfg_search_markers[method],
                label=cfg_search_labels[method],
                color=cfg_search_colors[method],
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Inference Compute Budget")
        ax.set_xticks(compute)
        ax.set_xticklabels(["1×", "2×", "4×", "8×"])
        ax.set_ylabel(f"{lab}\n({better} is better)")
        ax.set_title(f"CFG Search Ablation: {lab}", pad=20)

        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            loc="best",
        )
        legend.get_frame().set_linewidth(1.2)

        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        filename = f"ablation_cfg_search_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 5: COARSE SIMULATE FORWARD (fewer timesteps for reward simulation)
# ══════════════════════════════════════════════════════════════════════════════

coarse_colors = {
    "random_search": "#E67E22",
    "noise_search": "#3498DB",
    "noise_search_coarse": "#2ECC71",
}

coarse_markers = {
    "random_search": "o",
    "noise_search": "s",
    "noise_search_coarse": "^",
}

coarse_ablation_data = {
    "random_search": {
        "fid": [136.8648, 134.4275, 129.2695, 126.6169],
        "is": [14.4085, 15.0617, 17.4542, 18.4819],
        "top1": [66.015625, 76.953125, 87.890625, 93.359375],
        "top5": [83.203125, 90.625, 97.65625, 99.609375],
    },
    "noise_search": {
        "fid": [136.8648, 134.5541, 127.9526, 124.6068],
        "is": [14.4085, 15.9361, 18.2196, 18.8544],
        "top1": [66.015625, 87.5, 93.359375, 99.21875],
        "top5": [83.203125, 97.65625, 100.0, 100.0],
    },
    "noise_search_coarse": {
        "fid": [136.8648, 141.3871, 130.7433, 131.1442],
        "is": [14.4085, 14.5918, 16.3387, 18.0113],
        "top1": [66.015625, 77.34375, 91.40625, 94.53125],
        "top5": [83.203125, 94.921875, 98.046875, 100.0],
    },
}

coarse_labels = {
    "random_search": "Random Search",
    "noise_search": "NS–DMFM-ODE (dt=0.05)",
    "noise_search_coarse": "NS–DMFM-ODE Coarse (dt=0.1)",
}


def plot_coarse_simulate_ablation():
    """Plot coarse simulate forward ablation results."""

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for method, data in coarse_ablation_data.items():
            ax.plot(
                compute,
                data[metric],
                marker=coarse_markers[method],
                label=coarse_labels[method],
                color=coarse_colors[method],
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Inference Compute Budget")
        ax.set_xticks(compute)
        ax.set_xticklabels(["1×", "2×", "4×", "8×"])
        ax.set_ylabel(f"{lab}\n({better} is better)")
        ax.set_title(f"Coarse Simulate Forward Ablation: {lab}", pad=20)

        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            loc="best",
        )
        legend.get_frame().set_linewidth(1.2)

        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        filename = f"ablation_coarse_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION 6: COARSE SIMULATE FORWARD FINE (dt=0.01 with coarser simulate forward)
# ══════════════════════════════════════════════════════════════════════════════

coarse_fine_colors = {
    "random_search": "#E67E22",
    "noise_search": "#3498DB",
    "noise_search_coarse_005": "#2ECC71",
    "noise_search_coarse_01": "#9B59B6",
}

coarse_fine_markers = {
    "random_search": "o",
    "noise_search": "s",
    "noise_search_coarse_005": "^",
    "noise_search_coarse_01": "D",
}

coarse_fine_ablation_data = {
    "random_search": {
        "fid": [177.8471, 179.0142, 175.5562, 173.2914],
        "is": [8.6753, 9.2717, 9.8290, 10.0503],
        "top1": [71.09375, 85.9375, 88.28125, 96.875],
        "top5": [85.15625, 96.875, 99.21875, 100.0],
    },
    "noise_search": {
        "fid": [177.8471, 174.7242, 172.0357, 177.0195],
        "is": [8.6753, 9.4281, 9.9984, 10.3640],
        "top1": [71.09375, 86.71875, 96.875, 96.875],
        "top5": [85.15625, 96.875, 100.0, 100.0],
    },
    "noise_search_coarse_005": {
        "fid": [177.8471, 182.2120, 170.6835, 175.5361],
        "is": [8.6753, 8.8248, 9.3774, 10.3476],
        "top1": [71.09375, 78.125, 95.3125, 97.65625],
        "top5": [85.15625, 97.65625, 99.21875, 100.0],
    },
    "noise_search_coarse_01": {
        "fid": [177.8471, 181.2575, 177.9640, 176.8070],
        "is": [8.6753, 8.4008, 9.0213, 9.3965],
        "top1": [71.09375, 83.59375, 93.75, 92.96875],
        "top5": [85.15625, 92.1875, 97.65625, 100.0],
    },
}

coarse_fine_labels = {
    "random_search": "Random Search (dt=0.01)",
    "noise_search": "NS–DMFM-ODE (dt=0.01)",
    "noise_search_coarse_005": "NS–DMFM-ODE Coarse (sim_dt=0.05)",
    "noise_search_coarse_01": "NS–DMFM-ODE Coarse (sim_dt=0.1)",
}


def plot_coarse_simulate_fine_ablation():
    """Plot coarse simulate forward fine ablation results (dt=0.01)."""

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for method, data in coarse_fine_ablation_data.items():
            ax.plot(
                compute,
                data[metric],
                marker=coarse_fine_markers[method],
                label=coarse_fine_labels[method],
                color=coarse_fine_colors[method],
                linewidth=3,
                markersize=10,
                markeredgewidth=2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Inference Compute Budget")
        ax.set_xticks(compute)
        ax.set_xticklabels(["1×", "2×", "4×", "8×"])
        ax.set_ylabel(f"{lab}\n({better} is better)")
        ax.set_title(f"Coarse Simulate Forward (dt=0.01): {lab}", pad=20)

        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.95,
            loc="best",
        )
        legend.get_frame().set_linewidth(1.2)

        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        filename = f"ablation_coarse_fine_{metric}.pdf"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#                                 MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating ablation study plots...")
    print(f"Saving all plots as PDF files in: {OUTPUT_DIR}/")
    print()

    plot_schedule_ablation()
    plot_nonuniform_branching_ablation()
    plot_cfg_ablation()
    plot_cfg_search_ablation()
    plot_coarse_simulate_ablation()
    plot_coarse_simulate_fine_ablation()

    print()
    print("All ablation plots generated!")
    print(f"Files saved in {OUTPUT_DIR}/:")
    print("- ablation_schedule_fid.pdf")
    print("- ablation_schedule_is.pdf")
    print("- ablation_schedule_top1.pdf")
    print("- ablation_schedule_top5.pdf")
    print("- ablation_nonuniform_fid.pdf")
    print("- ablation_nonuniform_is.pdf")
    print("- ablation_nonuniform_top1.pdf")
    print("- ablation_nonuniform_top5.pdf")
    print("- ablation_cfg_fid.pdf")
    print("- ablation_cfg_is.pdf")
    print("- ablation_cfg_top1.pdf")
    print("- ablation_cfg_top5.pdf")
    print("- ablation_cfg_search_fid.pdf")
    print("- ablation_cfg_search_is.pdf")
    print("- ablation_cfg_search_top1.pdf")
    print("- ablation_cfg_search_top5.pdf")
    print("- ablation_coarse_fid.pdf")
    print("- ablation_coarse_is.pdf")
    print("- ablation_coarse_top1.pdf")
    print("- ablation_coarse_top5.pdf")
    print("- ablation_coarse_fine_fid.pdf")
    print("- ablation_coarse_fine_is.pdf")
    print("- ablation_coarse_fine_top1.pdf")
    print("- ablation_coarse_fine_top5.pdf")
