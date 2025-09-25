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
      * DINO-scored branching
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
    # Noise schedules
    "SDE": "#E74C3C",
    "EDM-SDE": "#3498DB",
    "Score-SDE": "#2ECC71",
    "ODE-score-orth": "#9B59B6",
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

# Modern markers
markers = {
    "SDE": "o",
    "EDM-SDE": "s",
    "Score-SDE": "^",
    "ODE-score-orth": "D",
    "DMFM-ODE": "v",
    # Back-compat
    "ODE-divfree": "D",
    "ODE-divfree-max": "v",
}
scaling_markers = ["o", "s", "^", "v", "D"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NOISE & DIVERSITY STUDY DATA
# ══════════════════════════════════════════════════════════════════════════════

noise_data = {
    "SDE": {
        "ratio": [
            0.2504,
            0.3779,
            0.5046,
            0.6047,
            0.7106,
            0.8083,
            0.9143,
            1.0043,
            1.5169,
        ],
        "fid": [
            87.4612,
            92.0295,
            90.1710,
            96.3101,
            89.7057,
            95.7654,
            97.2009,
            103.6972,
            115.9218,
        ],
        "is": [
            21.2874,
            22.2206,
            21.2379,
            20.6732,
            21.0428,
            20.3009,
            21.1246,
            20.1875,
            17.2379,
        ],
        "div": [0.1374, 0.1985, 0.2486, 0.2953, 0.3357, 0.3876, 0.4362, 0.4683, 0.6789],
    },
    "EDM-SDE": {
        "ratio": [
            0.3572,
            0.4946,
            0.7059,
            0.7806,
            0.9273,
            0.9818,
            1.0881,
        ],
        "fid": [
            89.5376,
            92.9453,
            96.9788,
            97.4907,
            102.8617,
            108.8214,
            109.3022,
        ],
        "is": [
            21.3140,
            20.9175,
            20.5108,
            20.2692,
            19.5113,
            18.3376,
            18.5853,
        ],
        "div": [0.1301, 0.1723, 0.2367, 0.2592, 0.2998, 0.3246, 0.3501],
    },
    "Score-SDE": {
        "ratio": [
            0.1691,
            0.3338,
            0.5042,
            0.6591,
            0.9584,
            1.2206,
            1.4300,
            1.5729,
            1.7341,
        ],
        "fid": [
            94.0001,
            95.8718,
            95.6708,
            100.9059,
            114.9202,
            141.0572,
            169.4787,
            190.0741,
            226.5534,
        ],
        "is": [
            20.9604,
            20.7198,
            20.9081,
            19.6011,
            15.8803,
            11.3901,
            7.3832,
            5.5096,
            3.2606,
        ],
        "div": [0.0655, 0.1169, 0.1641, 0.2122, 0.3030, 0.3946, 0.4575, 0.4992, 0.5278],
    },
    "ODE-score-orth": {
        "ratio": [
            0.1153,
            0.2304,
            0.4027,
            0.4568,
            0.5190,
            0.5771,
            0.6917,
            0.8168,
            0.9327,
            1.1643,
            1.2811,
            1.3974,
            1.5330,
            1.6450,
            1.7578,
            2.3443,
        ],
        "fid": [
            89.2248,
            92.8932,
            92.8959,
            95.6153,
            93.8957,
            94.2256,
            95.2705,
            100.3158,
            95.6301,
            103.0657,
            106.3383,
            114.6040,
            113.2349,
            120.2541,
            117.5642,
            176.3332,
        ],
        "is": [
            20.7068,
            21.0029,
            21.2203,
            20.7766,
            20.5317,
            21.6945,
            21.2670,
            20.3107,
            19.4477,
            18.9992,
            18.9095,
            17.8703,
            16.6821,
            15.7843,
            14.4547,
            8.1700,
        ],
        "div": [
            0.0652,
            0.1254,
            0.2024,
            0.2292,
            0.2765,
            0.3246,
            0.3731,
            0.4315,
            0.4761,
            0.5183,
            0.5748,
            0.6126,
            0.6680,
            0.7114,
            0.7665,
            0.9564,
        ],
    },
    "DMFM-ODE": {
        "ratio": [
            0.09929307632389195,
            0.19834390933787727,
            0.344532514058439,
            0.39285259674509776,
            0.4931031221170705,
            0.5932488506566697,
            0.693311592008597,
            0.7895202505434171,
            0.8904068763052171,
            0.9960110432036678,
            1.0920357629487611,
            1.197683009865562,
            1.2903616733391603,
            1.3942665065449282,
            1.4941754460427292,
            1.9976494568660446,
        ],
        "fid": [
            92.95391082763672,
            95.35587310791016,
            90.66570281982422,
            90.08175659179688,
            98.75457763671875,
            96.22267150878906,
            91.76654815673828,
            93.44522094726562,
            99.04783630371094,
            107.61477661132812,
            105.08833312988281,
            107.0402603149414,
            105.17344665527344,
            110.95592498779297,
            108.82089233398438,
            145.07579040527344,
        ],
        "is": [
            21.531343460083008,
            20.743478775024414,
            21.00519561767578,
            21.251829147338867,
            20.93368911743164,
            20.74422836303711,
            20.333261489868164,
            20.087936401367188,
            20.264972686767578,
            19.205795288085938,
            19.717721939086914,
            18.57908058166504,
            18.027433395385742,
            16.99750518798828,
            16.81011962890625,
            11.994451522827148,
        ],
        "div": [
            0.1118,
            0.1634,
            0.2442,
            0.2645,
            0.3179,
            0.3660,
            0.4068,
            0.4684,
            0.5170,
            0.5679,
            0.6155,
            0.6536,
            0.7055,
            0.7595,
            0.7903,
            1.0174,
        ],
    },
}

baseline_fid = 98.3675
baseline_is = 21.5742
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
    # New data (Inception-scored)
    "noise_search_ode_divfree_max": {
        "fid": [incp_baseline["fid"], 52.4187316895, 54.6759490967, 58.9808082581],
        "is": [incp_baseline["is"], 71.8423843384, 82.3894119263, 88.78175354],
        "top1": [incp_baseline["top1"], 74.4140625, 76.7578125, 83.49609375],
        "top5": [incp_baseline["top5"], 87.890625, 89.55078125, 92.3828125],
    },
    "random_search_then_noise_search_ode_divfree_max": {
        "fid": [incp_baseline["fid"], 53.4427490234, 55.4165077209, 59.9713478088],
        "is": [incp_baseline["is"], 73.3276367188, 84.2112045288, 91.0559158325],
        "top1": [incp_baseline["top1"], 71.97265625, 79.78515625, 84.1796875],
        "top5": [incp_baseline["top5"], 87.40234375, 92.67578125, 93.45703125],
    },
    "noise_search_sde": {
        "fid": [incp_baseline["fid"], 51.3460083008, 51.3397254944, 55.8850822449],
        "is": [incp_baseline["is"], 72.7688751221, 82.5968170166, 89.6324462891],
        "top1": [incp_baseline["top1"], 74.4140625, 83.203125, 85.15625],
        "top5": [incp_baseline["top5"], 90.234375, 93.84765625, 94.3359375],
    },
}


# Path exploration (legacy) curves kept for appendix plots
incp_scaling_path = {
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
# 3.  INFERENCE-TIME-SCALING EXPERIMENTS  (DINO-scored)
# ══════════════════════════════════════════════════════════════════════════════

dino_baseline = {  # Random-Search 1× under DINO-scoring
    "fid": 52.6667022705,
    "is": 36.0008239746,
    "top1": 63.76953125,
    "top5": 83.7890625,
}

dino_scaling = {
    # New data (DINO-scored)
    "random_search": {
        "fid": [dino_baseline["fid"], 52.9817390442, 46.3392791748, 45.913066864],
        "is": [dino_baseline["is"], 43.3471069336, 50.3034744263, 55.2461967468],
        "top1": [dino_baseline["top1"], 76.5625, 85.64453125, 91.89453125],
        "top5": [dino_baseline["top5"], 90.33203125, 97.8515625, 99.609375],
    },
    "noise_search_sde": {
        "fid": [dino_baseline["fid"], 46.4305610657, 44.5015678406, 46.4238471985],
        "is": [dino_baseline["is"], 48.7623405457, 53.3355484009, 62.2763252258],
        "top1": [dino_baseline["top1"], 90.8203125, 94.62890625, 97.65625],
        "top5": [dino_baseline["top5"], 98.4375, 99.51171875, 100.0],
    },
    "noise_search_ode_divfree_max": {
        "fid": [dino_baseline["fid"], 46.0671424866, 43.3942871094, 45.1816902161],
        "is": [dino_baseline["is"], 48.1601676941, 56.3383789063, 61.5347709656],
        "top1": [dino_baseline["top1"], 88.96484375, 95.21484375, 98.14453125],
        "top5": [dino_baseline["top5"], 98.73046875, 99.8046875, 100.0],
    },
    "random_search_then_noise_search_ode_divfree_max": {
        "fid": [dino_baseline["fid"], 46.3289794922, 43.7386779785, 45.6046981812],
        "is": [dino_baseline["is"], 49.7810020447, 58.9731864929, 64.1715240479],
        "top1": [dino_baseline["top1"], 87.890625, 95.01953125, 97.8515625],
        "top5": [dino_baseline["top5"], 98.4375, 99.8046875, 100.0],
    },
}

# Path exploration (legacy) curves kept for appendix plots
dino_scaling_path = {
    "Best-of-N": {
        "fid": [dino_baseline["fid"], 51.8314666748, 46.8698577881, 46.2298698425],
        "is": [dino_baseline["is"], 40.1176834106, 52.1529541016, 56.6535110474],
        "top1": [dino_baseline["top1"], 75.1953125, 84.9609375, 93.45703125],
        "top5": [dino_baseline["top5"], 91.69921875, 97.36328125, 99.609375],
    },
    "Path-DivFree": {
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
            display_label = {
                "ODE-score-orth": "ODE-score-orth",
                "DMFM-ODE": "DMFM-ODE",
                # Back-compat for old keys
                "ODE-divfree": "ODE-score-orth",
                "ODE-divfree-max": "DMFM-ODE",
            }.get(label, label)
            ax.plot(
                m["ratio"],
                m[metric],
                marker=markers.get(display_label, markers.get(label, "o")),
                label=display_label,
                color=colors.get(display_label, colors.get(label, colors["baseline"])),
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

    # Temporarily revert to original smaller text sizes for pareto plots
    original_rcParams = rcParams.copy()
    rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.titlesize": 20,
        }
    )

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

    # Restore original rcParams
    rcParams.update(original_rcParams)


def plot_scaling(curves, title_suffix):
    """Plot inference scaling results with modern styling."""

    # Ensure RS baseline is present if available from legacy path data
    try:
        if (
            "random_search" not in curves
            and "Best-of-N" in incp_scaling_path
            and "Inception" in title_suffix
        ):
            curves = {**curves, "random_search": incp_scaling_path["Best-of-N"]}
    except NameError:
        pass
    try:
        if (
            "random_search" not in curves
            and "Best-of-N" in dino_scaling_path
            and "DINO" in title_suffix
        ):
            curves = {**curves, "random_search": dino_scaling_path["Best-of-N"]}
    except NameError:
        pass

    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Map standardized display names to legacy path-style keys for consistent styling
        path_style_map = {
            "RS": "Best-of-N",
            "PE–DivFree": "Path-DivFree",
            "PE–SDE": "Path-SDE",
            "PE–ScoreSDE": "Path-ScoreSDE",
            "RS+PE–DivFree": "BestN+Path-DivFree",
            # Map NS variants to closest PE styles to reuse exact colors/markers
            "NS–DMFM-ODE": "Path-DivFree",
            "RS+NS–DMFM-ODE": "BestN+Path-DivFree",
            "NS–SDE": "Path-SDE",
            # Raw method keys
            "random_search": "Best-of-N",
            "noise_search_ode_divfree_max": "Path-DivFree",
            "random_search_then_noise_search_ode_divfree_max": "BestN+Path-DivFree",
            "noise_search_sde": "Path-SDE",
        }
        path_markers = {
            "Best-of-N": "o",
            "Path-DivFree": "D",
            "Path-SDE": "s",
            "Path-ScoreSDE": "^",
            "BestN+Path-DivFree": "v",
        }

        for (method, data), mk in zip(curves.items(), scaling_markers):
            display_method = {
                "Best-of-N": "RS",
                "Path-DivFree": "PE–DivFree",
                "Path-SDE": "PE–SDE",
                "Path-ScoreSDE": "PE–ScoreSDE",
                "BestN+Path-DivFree": "RS+PE–DivFree",
                "NoiseSearch-3R-DivFree": "NS–DMFM-ODE",
                "BestN+NoiseSearch-3R-DivFree": "RS+NS–DMFM-ODE",
                # New data keys
                "random_search": "RS",
                "noise_search_ode_divfree_max": "NS–DMFM-ODE",
                "random_search_then_noise_search_ode_divfree_max": "RS+NS–DMFM-ODE",
                "noise_search_sde": "NS–SDE",
            }.get(method, method)
            style_key = path_style_map.get(display_method, display_method)
            color = colors.get(
                style_key,
                colors.get(display_method, colors.get(method, colors["baseline"])),
            )
            marker_choice = path_markers.get(style_key, mk)
            ax.plot(
                compute,
                data[metric],
                marker=marker_choice,
                label=display_method,
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
        # Convert title_suffix to use "Verifier" terminology
        verifier_suffix = title_suffix.replace(
            "Inception-scored branching", "Inception Verifier"
        ).replace("DINO-scored branching", "DINO Verifier")
        ax.set_title(f"{lab} vs. Compute Budget\n{verifier_suffix}", pad=20)

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


def plot_scaling_appendix(curves, title_suffix):
    # Use original colors/markers as in legacy path graphs
    appendix_markers = {
        "Best-of-N": "o",
        "Path-DivFree": "D",
        "Path-SDE": "s",
        "Path-ScoreSDE": "^",
        "BestN+Path-DivFree": "v",
    }
    for metric, lab, better in [
        ("fid", "FID Score", "lower"),
        ("is", "Inception Score", "higher"),
        ("top1", "DINO Top-1 Accuracy (%)", "higher"),
        ("top5", "DINO Top-5 Accuracy (%)", "higher"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))
        for method, data in curves.items():
            display_label = {
                "Best-of-N": "RS",
                "Path-DivFree": "PE–DivFree",
                "Path-SDE": "PE–SDE",
                "Path-ScoreSDE": "PE–ScoreSDE",
                "BestN+Path-DivFree": "RS+PE–DivFree",
            }.get(method, method)
            color = colors.get(method, colors["baseline"])
            ax.plot(
                compute,
                data[metric],
                marker=appendix_markers.get(method, "o"),
                label=display_label,
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
        # Convert title_suffix to use "Verifier" terminology
        verifier_suffix = title_suffix.replace(
            "Inception-scored branching", "Inception Verifier"
        ).replace("DINO-scored branching", "DINO Verifier")
        ax.set_title(f"{lab} vs. Compute Budget\n{verifier_suffix}", pad=20)

        # Special positioning for DINO FID plot
        if metric == "fid" and "DINO" in title_suffix:
            legend = ax.legend(
                frameon=True,
                fancybox=True,
                shadow=True,
                facecolor="white",
                edgecolor="gray",
                framealpha=0.95,
                bbox_to_anchor=(0.85, 1.0),
                loc="upper right",
            )
        else:
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
        scoring_type = "inception" if "Inception" in title_suffix else "dino"
        filename = f"scaling_{scoring_type}_{metric}_path.pdf"
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
    plot_scaling(dino_scaling, "DINO-scored branching")

    # Appendix: legacy path-exploration curves
    plot_scaling_appendix(incp_scaling_path, "Inception-scored branching")
    plot_scaling_appendix(dino_scaling_path, "DINO-scored branching")

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
