import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import argparse
import json
from datetime import datetime
import torchmetrics.image.fid as FID

from mcts_single_flow import MCTSFlowSampler
from imagenet_dataset import ImageNet32Dataset
from run_mcts_flow import calculate_inception_score

DEFAULT_DATASET = "imagenet256"
DEFAULT_DEVICE = "cuda:1"
DEFAULT_REAL_SAMPLES = 1000

# Evaluation mode defaults
DEFAULT_EVAL_MODE = "single_samples"

# Sample generation defaults
DEFAULT_N_SAMPLES = 128
DEFAULT_BRANCH_PAIRS = "1:1,2:1,4:1,8:1"

# Time step defaults
DEFAULT_BRANCH_DT = 0.1
DEFAULT_BRANCH_START_TIME = 0.5
DEFAULT_DT_STD = 0.7
DEFAULT_WARP_SCALE = 0.5

# Sampling method defaults
DEFAULT_SAMPLE_METHOD = "random_search"
DEFAULT_SCORING_FUNCTION = "inception_score"

# Batch optimization defaults
DEFAULT_REFINEMENT_BATCH_SIZE = 32
DEFAULT_NUM_ITERATIONS = 1

# Output defaults
DEFAULT_OUTPUT_DIR = "./results"


def evaluate_sampler(args):
    """
    Evaluate a flow sampler using specified configuration parameters.

    Args:
        args: Command line arguments containing configuration

    Returns:
        Dictionary of evaluation results
    """
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine dataset parameters
    if args.dataset.lower() == "cifar10":
        num_classes = 10
        print("Using CIFAR-10 dataset")
    elif args.dataset.lower() == "imagenet32" or args.dataset.lower() == "imagenet256":
        num_classes = 1000
        print(f"Using {args.dataset} dataset")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Common parameters
    image_size = 32
    channels = 4 if args.dataset.lower() == "imagenet256" else 3

    if args.dataset.lower() == "imagenet256":
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # Resize the smaller edge to 256
                transforms.CenterCrop(256),  # Crop the center to get a square image
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    # Load the appropriate dataset
    if args.dataset.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset.lower() == "imagenet32":
        dataset = ImageNet32Dataset(root_dir="./data", train=False, transform=transform)
    elif args.dataset.lower() == "imagenet256":
        dataset = datasets.ImageNet(root="./data", split="val", transform=transform)

    flow_model_name = f"flow_model_{args.dataset}.pt"

    num_timesteps = int(1 / args.branch_dt)

    # Initialize sampler with configurable parameters
    sampler = MCTSFlowSampler(
        image_size=image_size,
        channels=channels,
        device=device,
        num_timesteps=num_timesteps,
        num_classes=num_classes,
        buffer_size=10,
        load_models=True,
        flow_model=flow_model_name,
        num_channels=256,
        inception_layer=0,
        dataset=args.dataset,
        flow_model_config=(
            {
                "num_res_blocks": 3,
                "attention_resolutions": "16,8",
            }
            if args.dataset.lower() == "imagenet32"
            else None
        ),
        load_dino=True,
    )

    # Initialize FID for metrics
    fid = FID.FrechetInceptionDistance(normalize=True, reset_real_features=False).to(
        device
    )

    # Load real images for FID calculation
    print(f"Loading dataset for FID calculation...")
    sample_size = min(args.real_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    real_images = torch.stack([dataset[i][0] for i in indices]).to(device)

    # Process real images in batches
    real_batch_size = 100
    print(f"Processing {sample_size} real images from {args.dataset}...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid.update(batch, real=True)

    # Create output directory for results if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the appropriate evaluation
    if args.eval_mode == "single_samples":
        results = evaluate_single_samples(
            sampler=sampler,
            device=device,
            n_samples=args.n_samples,
            branch_pairs=args.branch_pairs,
            scoring_function=args.scoring_function,
            sample_method=args.sample_method,
            branch_dt=args.branch_dt,
            branch_start_time=args.branch_start_time,
            fid=fid,
            dataset=dataset,
            args=args,
        )
    elif args.eval_mode == "batch_optimization":
        results = evaluate_batch_optimization(
            sampler=sampler,
            device=device,
            n_samples=args.n_samples,
            branch_pairs=args.branch_pairs,
            refinement_batch_size=args.refinement_batch_size,
            num_iterations=args.num_iterations,
            fid=fid,
            sample_method=args.sample_method,
            branch_dt=args.branch_dt,
            branch_start_time=args.branch_start_time,
            dt_std=args.dt_std,
            warp_scale=args.warp_scale,
            dataset=dataset,
            args=args,
        )
    else:
        raise ValueError(f"Unsupported evaluation mode: {args.eval_mode}")

    # Add configuration to results
    results["config"] = vars(args)
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Convert results to JSON-serializable format
    json_results = convert_for_json(results)

    # Save results to file
    result_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.eval_mode}_{args.sample_method}_{results['timestamp']}.json",
    )
    with open(result_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {result_file}")
    return results


def evaluate_single_samples(
    sampler,
    device,
    n_samples,
    branch_pairs,
    scoring_function,
    sample_method,
    branch_dt,
    branch_start_time,
    fid,
    dataset,
    args,
):
    """
    Evaluate sampling methods that select individual samples
    """
    results = {
        "branch_pairs": {},
        "scoring_function": scoring_function,
        "sample_method": sample_method,
        "n_samples": n_samples,
        "branch_dt": branch_dt,
        "branch_start_time": branch_start_time,
    }

    for num_branches, num_keep in branch_pairs:
        print(f"\nEvaluating with branches={num_branches}, keep={num_keep}")

        metrics = generate_and_compute_metrics(
            sampler=sampler,
            num_branches=num_branches,
            num_keep=num_keep,
            device=device,
            n_samples=n_samples,
            scoring_function=scoring_function,
            sample_method=sample_method,
            branch_dt=branch_dt,
            branch_start_time=branch_start_time,
            fid=fid,
            dataset=dataset,
        )

        # Store results for this branch/keep pair
        pair_key = f"{num_branches}_{num_keep}"
        results["branch_pairs"][pair_key] = {
            "fid_score": metrics["fid_score"],
            "inception_score": metrics["inception_score"],
            "inception_std": metrics["inception_std"],
            "avg_mahalanobis": metrics["avg_mahalanobis"],
        }

        # Print metrics for this configuration
        print(f"\nResults for branches={num_branches}, keep={num_keep}:")
        print(f"FID Score: {metrics['fid_score']:.4f}")
        print(
            f"Inception Score: {metrics['inception_score']:.4f} ± {metrics['inception_std']:.4f}"
        )
        print(f"Average Mahalanobis Distance: {metrics['avg_mahalanobis']:.4f}")

        # Create and save sample comparison plot
        save_sample_comparison_plot(
            generated_samples=metrics["samples"],
            dataset=dataset,
            class_labels=metrics["class_labels"],
            args=args,
            metrics=metrics,
            branch_keep_pair=(num_branches, num_keep),
            num_display=16,
        )

    return results


def evaluate_batch_optimization(
    sampler,
    device,
    n_samples,
    branch_pairs,
    refinement_batch_size,
    num_iterations,
    fid,
    sample_method,
    branch_dt=0.1,
    branch_start_time=0.5,
    dt_std=0.7,
    warp_scale=0.5,
    dataset=None,
    args=None,
):
    """
    Evaluate batch optimization methods
    """
    results = {
        "branch_pairs": {},
        "refinement_batch_size": refinement_batch_size,
        "num_iterations": num_iterations,
        "n_samples": n_samples,
        "sample_method": sample_method,
        "branch_dt": branch_dt,
        "branch_start_time": branch_start_time,
        "dt_std": dt_std,
        "warp_scale": warp_scale,
    }

    for num_branches, num_batches in branch_pairs:
        print(f"\nEvaluating with branches={num_branches}, batches={num_batches}")

        fid.reset()
        random_class_labels = torch.randint(
            0, sampler.num_classes, (n_samples,), device=device
        )

        # Map the sample_method to the appropriate batch optimization function
        if sample_method == "random_search":
            final_samples = sampler.batch_sample_refine_global_fid_random(
                n_samples=n_samples,
                refinement_batch_size=refinement_batch_size,
                num_branches=num_branches,
                num_batches=num_batches,
                num_iterations=num_iterations,
                use_global=True,
            )
        elif sample_method == "path_exploration":
            final_samples = sampler.batch_sample_refine_global_fid_path_explore(
                n_samples=n_samples,
                refinement_batch_size=refinement_batch_size,
                num_branches=num_branches,
                num_batches=num_batches,
                branch_start_time=branch_start_time,
                branch_dt=branch_dt,
                dt_std=dt_std,
                num_iterations=num_iterations,
                use_global=True,
            )
        elif sample_method == "path_exploration_timewarp":
            final_samples = sampler.batch_sample_refine_global_fid_timewarp(
                n_samples=n_samples,
                refinement_batch_size=refinement_batch_size,
                num_branches=num_branches,
                num_batches=num_batches,
                branch_dt=branch_dt,
                warp_scale=warp_scale,
                num_iterations=num_iterations,
                use_global=True,
                branch_start_time=branch_start_time,
            )
        else:
            raise ValueError(
                f"Unsupported sample method for batch optimization: {sample_method}"
            )

        # Process final samples to compute metrics
        metrics = process_batch_samples(
            sampler, final_samples, random_class_labels, device, fid
        )

        # Store results for this branch/batch pair
        pair_key = f"{num_branches}_{num_batches}"
        results["branch_pairs"][pair_key] = metrics

        # Print metrics for this configuration
        print(f"\nResults for branches={num_branches}, batches={num_batches}:")
        print(f"FID Score: {metrics['fid_score']:.4f}")
        print(
            f"Inception Score: {metrics['inception_score']:.4f} ± {metrics['inception_std']:.4f}"
        )
        print(f"Average Mahalanobis Distance: {metrics['avg_mahalanobis']:.4f}")

        # Create and save sample comparison plot
        save_sample_comparison_plot(
            generated_samples=final_samples.cpu(),
            dataset=dataset,
            class_labels=random_class_labels,
            args=args,
            metrics=metrics,
            branch_keep_pair=(num_branches, num_batches),
            num_display=16,
        )

    return results


def process_batch_samples(sampler, samples, class_labels, device, fid):
    """
    Process batch optimization samples and compute metrics

    Args:
        sampler: The MCTSFlowSampler instance
        samples: Generated samples
        class_labels: Class labels for the samples
        device: Device to use for computation
        fid: FID instance for calculation

    Returns:
        Dictionary of metrics
    """
    metric_batch_size = 64

    # Compute mahalanobis distances
    mahalanobis_dist = sampler.batch_compute_global_mean_difference(samples)
    avg_mahalanobis = mahalanobis_dist.mean().item()

    # Process samples for FID
    for i in range(0, len(samples), metric_batch_size):
        batch = samples[i : i + metric_batch_size]
        fid.update(batch, real=False)

    # Compute FID score
    fid_score = fid.compute().item()

    # Compute Inception Score
    inception_score, inception_std = calculate_inception_score(
        samples, device=device, batch_size=metric_batch_size, splits=10
    )

    return {
        "fid_score": fid_score,
        "inception_score": inception_score,
        "inception_std": inception_std,
        "avg_mahalanobis": avg_mahalanobis,
    }


def generate_and_compute_metrics(
    sampler,
    num_branches,
    num_keep,
    device,
    n_samples,
    scoring_function,
    sample_method,
    branch_dt,
    branch_start_time,
    fid,
    dataset=None,  # Add dataset parameter
):
    """
    Generate samples and compute metrics

    Args:
        sampler: The MCTSFlowSampler instance
        num_branches: Number of branches to use
        num_keep: Number of samples to keep per branch
        device: Device to use for computation
        n_samples: Number of samples to generate
        scoring_function: Scoring function for sample selection
        sample_method: Sampling method to use
        branch_dt: Time step size for branching
        branch_start_time: Time to start branching
        fid: FID instance for calculation
        dataset: Dataset containing real samples

    Returns:
        Dictionary of metrics
    """
    fid.reset()

    generation_batch_size = 64
    metric_batch_size = 64
    generated_samples = []
    mahalanobis_distances = []
    all_class_labels = []

    # Calculate number of batches to generate
    num_batches = n_samples // generation_batch_size
    if n_samples % generation_batch_size != 0:
        num_batches += 1

    print(f"Generating {n_samples} samples...")

    # Generate samples
    for batch_idx in range(num_batches):
        # Adjust batch size for the last batch if needed
        current_batch_size = min(
            generation_batch_size, n_samples - batch_idx * generation_batch_size
        )

        # Randomly sample class labels
        random_class_labels = torch.randint(
            0, sampler.num_classes, (current_batch_size,), device=device
        )
        all_class_labels.append(random_class_labels)

        # Generate samples using the specified method
        if sample_method == "regular":
            sample = sampler.regular_batch_sample(
                class_label=random_class_labels, batch_size=current_batch_size
            )
        elif sample_method == "path_exploration_timewarp":
            sample = sampler.batch_sample_with_path_exploration_timewarp(
                class_label=random_class_labels,
                batch_size=current_batch_size,
                num_branches=num_branches,
                num_keep=num_keep,
                warp_scale=0.5,
                selector=scoring_function,
                use_global=True,
                branch_start_time=branch_start_time,
                branch_dt=branch_dt,
            )
        elif sample_method == "path_exploration_timewarp_shifted":
            sample = sampler.batch_sample_with_path_exploration_timewarp_shifted(
                class_label=random_class_labels,
                batch_size=current_batch_size,
                num_branches=num_branches,
                num_keep=num_keep,
                warp_scale=0.5,
                selector=scoring_function,
                use_global=True,
                branch_start_time=branch_start_time,
                branch_dt=branch_dt,
            )
        elif sample_method == "path_exploration":
            sample = sampler.batch_sample_with_path_exploration(
                class_label=random_class_labels,
                batch_size=current_batch_size,
                num_branches=num_branches,
                num_keep=num_keep,
                dt_std=0.1,
                selector=scoring_function,
                use_global=True,
                branch_start_time=branch_start_time,
                branch_dt=branch_dt,
            )
        elif sample_method == "random_search":
            sample = sampler.batch_sample_with_random_search(
                class_label=random_class_labels,
                batch_size=current_batch_size,
                num_branches=num_branches,
                selector=scoring_function,
                use_global=True,
            )
        else:
            raise ValueError(f"Unsupported sample method: {sample_method}")

        # Compute metrics
        mahalanobis_dist = sampler.batch_compute_global_mean_difference(sample)
        mahalanobis_distances.extend(mahalanobis_dist.cpu().tolist())
        generated_samples.extend(sample.cpu())

    # Process generated samples for FID
    generated_tensor = torch.stack(generated_samples)
    for i in range(0, len(generated_tensor), metric_batch_size):
        batch = generated_tensor[i : i + metric_batch_size].to(device)
        fid.update(batch, real=False)
        batch.cpu()

    # Compute FID score
    fid_score = fid.compute().item()

    # Compute Inception Score
    generated_tensor_device = torch.stack(generated_samples).to(device)
    inception_score, inception_std = calculate_inception_score(
        generated_tensor_device, device=device, batch_size=metric_batch_size, splits=10
    )

    # Compute average Mahalanobis distance
    avg_mahalanobis = sum(mahalanobis_distances) / len(mahalanobis_distances)

    # Combine all class labels
    all_class_labels = torch.cat(all_class_labels, dim=0)

    # Return results including samples and labels for plotting
    results = {
        "fid_score": fid_score,
        "inception_score": inception_score,
        "inception_std": inception_std,
        "avg_mahalanobis": avg_mahalanobis,
        "samples": generated_samples,
        "class_labels": all_class_labels,
    }

    # Clean up memory
    generated_tensor = None
    generated_tensor_device = None
    torch.cuda.empty_cache()

    return results


def save_sample_comparison_plot(
    generated_samples,
    dataset,
    class_labels,
    args,
    metrics,
    branch_keep_pair,
    num_display=16,
):
    """
    Save a plot comparing generated samples with random real samples from the dataset.

    Args:
        generated_samples: List of generated tensor samples
        dataset: The dataset containing real samples
        class_labels: Class labels for the generated samples
        args: Command line arguments
        metrics: Dictionary of metrics for this configuration
        branch_keep_pair: Tuple of (num_branches, num_keep) for this plot
        num_display: Number of samples to display (must be a perfect square)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import math

    # Convert num_display to the nearest perfect square if it's not already
    grid_size = int(math.sqrt(num_display))
    num_display = grid_size * grid_size

    # Only display a subset of the samples
    subset_indices = np.random.choice(
        len(generated_samples), num_display, replace=False
    )
    subset_samples = [generated_samples[i] for i in subset_indices]
    subset_labels = class_labels[subset_indices].cpu().numpy()

    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(grid_size * 2, grid_size * 2))
    plt.suptitle(
        f"Sample Comparison - {args.dataset} - {args.sample_method} (b={branch_keep_pair[0]}, k={branch_keep_pair[1]})\n"
        f"FID: {metrics['fid_score']:.2f}, IS: {metrics['inception_score']:.2f}±{metrics['inception_std']:.2f}"
    )

    # Configure the grid
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.2, width_ratios=[1, 1])

    # Left subplot for generated samples
    gen_grid = gridspec.GridSpecFromSubplotSpec(
        grid_size, grid_size, subplot_spec=outer_grid[0], wspace=0.1, hspace=0.1
    )

    # Right subplot for real samples
    real_grid = gridspec.GridSpecFromSubplotSpec(
        grid_size, grid_size, subplot_spec=outer_grid[1], wspace=0.1, hspace=0.1
    )

    # Plot generated samples
    for i in range(num_display):
        ax = plt.Subplot(fig, gen_grid[i])
        img = subset_samples[i].permute(1, 2, 0).cpu().numpy()
        # Normalize to [0, 1] range
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        # Show class label in the title
        ax.set_title(f"Gen (c={subset_labels[i]})", fontsize=8)
        fig.add_subplot(ax)

    # Get random real samples
    real_indices = np.random.choice(len(dataset), num_display, replace=False)
    real_samples = []
    real_labels = []

    for idx in real_indices:
        sample, label = dataset[idx]
        real_samples.append(sample)
        real_labels.append(label)

    # Plot real samples
    for i in range(num_display):
        ax = plt.Subplot(fig, real_grid[i])
        img = real_samples[i].permute(1, 2, 0).cpu().numpy()
        # Normalize to [0, 1] range
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        # Show class label in the title
        ax.set_title(f"Real (c={real_labels[i]})", fontsize=8)
        fig.add_subplot(ax)

    # Save the figure
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(
        plot_dir,
        f"{args.dataset}_{args.sample_method}_b{branch_keep_pair[0]}_k{branch_keep_pair[1]}_{timestamp}.png",
    )
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Sample comparison plot saved to {filepath}")


def convert_for_json(obj):
    """
    Convert objects to JSON serializable formats
    """
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_for_json(item) for item in obj)
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate flow sampler methods")

    # Dataset and device
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=["cifar10", "imagenet32", "imagenet256"],
        help="Dataset to use (cifar10 or imagenet32 or imagenet256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to use for computation",
    )

    # Evaluation mode
    parser.add_argument(
        "--eval_mode",
        type=str,
        default=DEFAULT_EVAL_MODE,
        choices=["single_samples", "batch_optimization"],
        help="Evaluation mode (single_samples or batch_optimization)",
    )

    # Sample generation parameters
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--real_samples",
        type=int,
        default=DEFAULT_REAL_SAMPLES,
        help="Number of real samples to use for FID calculation",
    )

    # Branching parameters
    parser.add_argument(
        "--branch_pairs",
        type=str,
        default=DEFAULT_BRANCH_PAIRS,
        help="Comma-separated list of branch:keep pairs (or branch:batch pairs for batch optimization)",
    )
    parser.add_argument(
        "--branch_dt",
        type=float,
        default=DEFAULT_BRANCH_DT,
        help="Time step size for branching",
    )
    parser.add_argument(
        "--branch_start_time",
        type=float,
        default=DEFAULT_BRANCH_START_TIME,
        help="Time to start branching",
    )

    # Sampling method
    parser.add_argument(
        "--sample_method",
        type=str,
        default=DEFAULT_SAMPLE_METHOD,
        choices=[
            "regular",
            "random_search",
            "path_exploration",
            "path_exploration_timewarp",
            "path_exploration_timewarp_shifted",
        ],
        help="Sampling method to use (for both single samples and batch optimization)",
    )

    # Scoring function
    parser.add_argument(
        "--scoring_function",
        type=str,
        default=DEFAULT_SCORING_FUNCTION,
        choices=[
            "inception_score",
            "dino_score",
            "global_mean_difference",
            "inception_classifier_score",
        ],
        help="Scoring function for sample selection",
    )

    # Batch optimization parameters
    parser.add_argument(
        "--refinement_batch_size",
        type=int,
        default=DEFAULT_REFINEMENT_BATCH_SIZE,
        help="Batch size for refinement",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="Number of refinement iterations",
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )

    # Additional parameters
    parser.add_argument(
        "--dt_std",
        type=float,
        default=DEFAULT_DT_STD,
        help="Standard deviation for time steps in path exploration",
    )
    parser.add_argument(
        "--warp_scale",
        type=float,
        default=DEFAULT_WARP_SCALE,
        help="Scale factor for time warping",
    )

    args = parser.parse_args()

    # Parse branch pairs
    args.branch_pairs = [
        tuple(map(int, pair.split(":"))) for pair in args.branch_pairs.split(",")
    ]

    # Run evaluation
    evaluate_sampler(args)
