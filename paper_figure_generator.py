import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from mcts_single_flow import MCTSFlowSampler


def generate_class_comparison_figure(
    target_class_labels,
    sample_method,
    dataset="imagenet256",
    device="cuda",
    branch_dt=0.05,
    branch_start_time=0,
    scoring_function="dino_score",
    noise_scale=0.14,
    lambda_div=0.9,
    noise_schedule_end_factor=0.7,
    deterministic_rollout=0,
    repulsion_disable_until_time=0.0,
    rounds=9,
    output_dir="./paper_figures",
    samples_per_config=4,
):
    """
    Generate a comparison figure showing class labels at different computation levels.

    Args:
        target_class_labels: List of class labels to generate (e.g., [281, 207, 285] for ImageNet)
        sample_method: The sampling method to use
        dataset: Dataset name
        device: Device to use for computation
        branch_dt: Time step size for branching
        branch_start_time: Time to start branching
        scoring_function: Scoring function for sample selection
        noise_scale: Noise scale for SDE sampling
        lambda_div: Lambda for divergence-free sampling
        noise_schedule_end_factor: End factor for time-dependent noise scaling
        deterministic_rollout: Use deterministic rollout (1) or stochastic (0)
        repulsion_disable_until_time: Disable repulsion forces until this time
        rounds: Number of rounds for noise search methods
        output_dir: Directory to save the figure
        samples_per_config: Total number of samples to generate per branch configuration (distributed across classes)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Handle single class input (backward compatibility)
    if isinstance(target_class_labels, int):
        target_class_labels = [target_class_labels]

    num_classes = len(target_class_labels)
    samples_per_class = max(1, samples_per_config // num_classes)

    print(f"Generating {samples_per_class} samples per class for {num_classes} classes")
    print(f"Classes: {target_class_labels}")

    # Dataset parameters
    if dataset.lower() == "imagenet256":
        total_classes = 1000
        image_size = 32
        channels = 4
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Flow model
    flow_model_name = f"flow_model_{dataset}.pt"
    num_timesteps = int(1 / branch_dt)

    # Initialize sampler
    sampler = MCTSFlowSampler(
        image_size=image_size,
        channels=channels,
        device=device,
        num_timesteps=num_timesteps,
        num_classes=total_classes,
        buffer_size=10,
        load_models=True,
        flow_model=flow_model_name,
        num_channels=256,
        inception_layer=0,
        dataset=dataset,
        load_dino=True,
    )

    # Branch configurations: 1x, 2x, 4x, 8x computation
    branch_configs = [
        (1, 1),  # 1x computation (baseline)
        (2, 1),  # 2x computation
        (4, 1),  # 4x computation
        (8, 1),  # 8x computation
    ]

    # Generate samples for each configuration and class
    all_samples = {}

    print(
        f"Generating samples for classes {target_class_labels} using method {sample_method}"
    )

    for num_branches, num_keep in branch_configs:
        print(f"\nGenerating with branches={num_branches}, keep={num_keep}")

        # Store samples for this computation level
        computation_samples = []
        computation_labels = []

        for class_label in target_class_labels:
            print(f"  Generating {samples_per_class} samples for class {class_label}")

            # Create class label tensor
            class_labels = torch.full((samples_per_class,), class_label, device=device)

            # Generate samples using the specified method
            if sample_method == "ode":
                samples = sampler.batch_sample_ode(
                    class_label=class_labels,
                    batch_size=samples_per_class,
                )
            elif sample_method == "ode_divfree":
                samples = sampler.batch_sample_ode_divfree(
                    class_label=class_labels,
                    batch_size=samples_per_class,
                    lambda_div=lambda_div,
                )
            elif sample_method == "sde":
                samples = sampler.batch_sample_sde(
                    class_label=class_labels,
                    batch_size=samples_per_class,
                    noise_scale=noise_scale,
                )
            elif sample_method == "random_search":
                samples = sampler.batch_sample_with_random_search(
                    class_label=class_labels,
                    batch_size=samples_per_class,
                    num_branches=num_branches,
                    selector=scoring_function,
                    use_global=True,
                )
            elif sample_method == "noise_search_sde":
                samples = sampler.batch_sample_noise_search_sde(
                    class_label=class_labels,
                    batch_size=samples_per_class,
                    num_branches=num_branches,
                    num_keep=num_keep,
                    rounds=rounds,
                    noise_scale=noise_scale,
                    selector=scoring_function,
                    use_global=True,
                )
            elif sample_method == "noise_search_ode_divfree_max":
                samples = sampler.batch_sample_noise_search_ode_divfree_max(
                    class_label=class_labels,
                    batch_size=samples_per_class,
                    num_branches=num_branches,
                    num_keep=num_keep,
                    rounds=rounds,
                    lambda_div=lambda_div,
                    noise_schedule_end_factor=noise_schedule_end_factor,
                    selector=scoring_function,
                    use_global=True,
                    deterministic_rollout=bool(deterministic_rollout),
                    repulsion_disable_until_time=repulsion_disable_until_time,
                )
            elif sample_method == "random_search_then_noise_search_ode_divfree_max":
                samples = sampler.batch_sample_random_search_then_noise_search_ode_divfree_max(
                    class_label=class_labels,
                    batch_size=samples_per_class,
                    num_branches=num_branches,
                    num_keep=num_keep,
                    rounds=rounds,
                    lambda_div=lambda_div,
                    noise_schedule_end_factor=noise_schedule_end_factor,
                    selector=scoring_function,
                    use_global=True,
                    deterministic_rollout=bool(deterministic_rollout),
                    repulsion_disable_until_time=repulsion_disable_until_time,
                )
            else:
                raise ValueError(f"Unsupported sample method: {sample_method}")

            computation_samples.append(samples.cpu())
            computation_labels.extend([class_label] * samples_per_class)

        # Combine all samples for this computation level
        all_computation_samples = torch.cat(computation_samples, dim=0)
        all_samples[f"{num_branches}x"] = {
            "samples": all_computation_samples,
            "labels": computation_labels,
        }
        print(
            f"Generated {len(all_computation_samples)} total samples for {num_branches}x computation"
        )

    # Create the comparison figure
    create_multi_class_comparison_figure(
        all_samples=all_samples,
        target_class_labels=target_class_labels,
        sample_method=sample_method,
        dataset=dataset,
        output_dir=output_dir,
        samples_per_class=samples_per_class,
    )

    print(f"\nFigure generation completed! Check {output_dir} for results.")


def create_multi_class_comparison_figure(
    all_samples,
    target_class_labels,
    sample_method,
    dataset,
    output_dir,
    samples_per_class,
):
    """Create and save the multi-class comparison figure."""

    num_classes = len(target_class_labels)
    total_samples_per_config = num_classes * samples_per_class

    # Create figure: rows = total samples, columns = 4 computation levels
    fig, axes = plt.subplots(
        total_samples_per_config, 4, figsize=(16, 3 * total_samples_per_config)
    )

    # Handle single sample case
    if total_samples_per_config == 1:
        axes = axes.reshape(1, -1)

    # Column titles
    computation_levels = ["1x", "2x", "4x", "8x"]
    for col, level in enumerate(computation_levels):
        axes[0, col].set_title(f"{level} Computation", fontsize=14, fontweight="bold")

    # Plot samples organized by class
    row_idx = 0
    for class_idx, class_label in enumerate(target_class_labels):
        for sample_idx in range(samples_per_class):
            for col, level in enumerate(computation_levels):
                ax = axes[row_idx, col]

                # Get the sample for this class and computation level
                samples_data = all_samples[level]
                samples = samples_data["samples"]
                labels = samples_data["labels"]

                # Find the sample index for this class and sample number
                class_sample_indices = [
                    i for i, label in enumerate(labels) if label == class_label
                ]
                if sample_idx < len(class_sample_indices):
                    sample_global_idx = class_sample_indices[sample_idx]
                    img = samples[sample_global_idx].permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)

                    # Display the image
                    ax.imshow(img)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Add class and sample labels on the left
                    if col == 0:
                        if samples_per_class == 1:
                            ax.set_ylabel(
                                f"Class {class_label}", fontsize=12, fontweight="bold"
                            )
                        else:
                            ax.set_ylabel(
                                f"Class {class_label}\nSample {sample_idx + 1}",
                                fontsize=10,
                            )
                else:
                    # No sample available, show empty plot
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(
                        0.5,
                        0.5,
                        "No sample",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

            row_idx += 1

    # Main title
    if num_classes == 1:
        title = (
            f"Class {target_class_labels[0]} Generation - {sample_method} - {dataset}"
        )
    else:
        title = f"Multi-Class Generation ({num_classes} classes) - {sample_method} - {dataset}"

    plt.suptitle(
        f"{title}\nComparison across computation levels",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if num_classes == 1:
        filename = f"class_{target_class_labels[0]}_{sample_method}_{dataset}_comparison_{timestamp}.png"
    else:
        class_str = "_".join(map(str, target_class_labels))
        filename = f"multiclass_{class_str}_{sample_method}_{dataset}_comparison_{timestamp}.png"

    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison figure saved to: {filepath}")

    # Also save individual samples for paper use
    save_individual_samples_multi_class(
        all_samples,
        target_class_labels,
        sample_method,
        dataset,
        output_dir,
        timestamp,
        samples_per_class,
    )


def save_individual_samples_multi_class(
    all_samples,
    target_class_labels,
    sample_method,
    dataset,
    output_dir,
    timestamp,
    samples_per_class,
):
    """Save individual sample images for paper use (multi-class version)."""

    individual_dir = os.path.join(output_dir, f"individual_samples_{timestamp}")
    os.makedirs(individual_dir, exist_ok=True)

    for level, samples_data in all_samples.items():
        level_dir = os.path.join(individual_dir, f"{level}_computation")
        os.makedirs(level_dir, exist_ok=True)

        samples = samples_data["samples"]
        labels = samples_data["labels"]

        # Save samples organized by class
        for class_label in target_class_labels:
            class_dir = os.path.join(level_dir, f"class_{class_label}")
            os.makedirs(class_dir, exist_ok=True)

            # Find samples for this class
            class_sample_indices = [
                i for i, label in enumerate(labels) if label == class_label
            ]

            for sample_idx, global_idx in enumerate(class_sample_indices):
                img = samples[global_idx].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)

                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis("off")
                plt.title(
                    f"Class {class_label} - {level} - Sample {sample_idx + 1}",
                    fontsize=10,
                )

                sample_filename = (
                    f"class_{class_label}_{level}_sample_{sample_idx + 1}.png"
                )
                sample_filepath = os.path.join(class_dir, sample_filename)
                plt.savefig(sample_filepath, dpi=300, bbox_inches="tight")
                plt.close()

    print(f"Individual samples saved to: {individual_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate class comparison figure for paper"
    )

    parser.add_argument(
        "--target_class_labels",
        type=str,
        required=True,
        help="Comma-separated list of class labels to generate (e.g., '281,207,285' for ImageNet)",
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        required=True,
        choices=[
            "ode",
            "ode_divfree",
            "sde",
            "random_search",
            "noise_search_sde",
            "noise_search_ode_divfree_max",
            "random_search_then_noise_search_ode_divfree_max",
        ],
        help="Sampling method to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet256",
        choices=["imagenet256"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_figures",
        help="Directory to save the figure",
    )
    parser.add_argument(
        "--samples_per_config",
        type=int,
        default=4,
        help="Total number of samples to generate per branch configuration (distributed across classes)",
    )
    parser.add_argument(
        "--scoring_function",
        type=str,
        default="inception_score",
        choices=["inception_score", "dino_score", "global_mean_difference"],
        help="Scoring function for sample selection",
    )

    # Additional sampling parameters
    parser.add_argument("--branch_dt", type=float, default=0.05)
    parser.add_argument("--branch_start_time", type=float, default=0)
    parser.add_argument("--noise_scale", type=float, default=0.14)
    parser.add_argument("--lambda_div", type=float, default=0.9)
    parser.add_argument("--noise_schedule_end_factor", type=float, default=0.7)
    parser.add_argument("--deterministic_rollout", type=int, default=0)
    parser.add_argument("--repulsion_disable_until_time", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=9)

    args = parser.parse_args()

    # Parse class labels
    target_class_labels = [int(x.strip()) for x in args.target_class_labels.split(",")]

    generate_class_comparison_figure(
        target_class_labels=target_class_labels,
        sample_method=args.sample_method,
        dataset=args.dataset,
        device=args.device,
        branch_dt=args.branch_dt,
        branch_start_time=args.branch_start_time,
        scoring_function=args.scoring_function,
        noise_scale=args.noise_scale,
        lambda_div=args.lambda_div,
        noise_schedule_end_factor=args.noise_schedule_end_factor,
        deterministic_rollout=args.deterministic_rollout,
        repulsion_disable_until_time=args.repulsion_disable_until_time,
        rounds=args.rounds,
        output_dir=args.output_dir,
        samples_per_config=args.samples_per_config,
    )
