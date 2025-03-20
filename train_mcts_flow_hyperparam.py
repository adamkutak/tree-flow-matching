import os
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mcts_single_flow import MCTSFlowSampler
import numpy as np
from tqdm import tqdm
import torchmetrics.image.fid as FID
import json
from datetime import datetime


def calculate_metrics(
    sampler,
    num_branches,
    num_keep,
    selector,
    use_global,
    branch_start_time,
    device,
    n_samples=2000,
    sigma=0.1,
    fid=None,
):
    """
    Calculate FID metrics for a specific configuration across all classes.
    """
    fid.reset()

    # Generate samples evenly across all classes
    samples_per_class = n_samples // sampler.num_classes
    generation_batch_size = 32
    metric_batch_size = 64
    generated_samples = []

    print(
        f"\nGenerating {n_samples} samples for:"
        f"\n  branches={num_branches}, keep={num_keep}"
        f"\n  selector={selector}, global={use_global}"
        f"\n  branch_start_time={branch_start_time:.1f}"
    )

    # Generate samples for each class
    for class_label in range(sampler.num_classes):
        num_batches = samples_per_class // generation_batch_size

        # Generate full batches
        for _ in range(num_batches):
            sample = sampler.batch_sample_wdt_with_selector(
                class_label=class_label,
                batch_size=generation_batch_size,
                num_branches=num_branches,
                num_keep=num_keep,
                dt_std=sigma,
                selector=selector,
                use_global=use_global,
                branch_start_time=branch_start_time,
            )
            generated_samples.extend(sample.cpu())

    # Process generated samples in batches for metrics
    generated_tensor = torch.stack(generated_samples)
    for i in range(0, len(generated_tensor), metric_batch_size):
        batch = generated_tensor[i : i + metric_batch_size].to(device)
        fid.update(batch, real=False)
        batch.cpu()
        torch.cuda.empty_cache()

    # Compute final scores
    fid_score = fid.compute()

    return fid_score


def main():
    # Setup device and paths
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparam_search_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # CIFAR-10 setup
    image_size = 32
    channels = 3
    num_classes = 10

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize sampler
    sampler = MCTSFlowSampler(
        image_size=image_size,
        channels=channels,
        device=device,
        num_timesteps=10,
        num_classes=num_classes,
        buffer_size=200,
        load_models=True,
        flow_model="large_flow_model.pt",
        value_model=None,
        num_channels=256,
        inception_layer=0,
    )

    # Hyperparameter search space
    branch_keep_pairs = [(1, 1), (4, 1), (8, 1), (16, 2)]
    selectors = ["fid", "mahalanobis", "mean"]
    global_options = [False, True]
    branch_start_times = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Initialize metrics
    fid = FID.FrechetInceptionDistance(normalize=True, reset_real_features=False).to(
        device
    )
    cifar10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Process real images for FID calculation
    print("Processing real images...")
    indices = np.random.choice(len(cifar10), 5000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    real_batch_size = 100
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid.update(batch, real=True)

    # Results dictionary
    results = {}

    # Hyperparameter search
    for selector in selectors:
        for use_global in global_options:
            for branch_start_time in branch_start_times:
                # Skip global option for FID as it's not meaningful
                if selector == "fid" and use_global:
                    continue

                print(f"\nTesting configurations for:")
                print(f"  selector={selector}, global={use_global}")
                print(f"  branch_start_time={branch_start_time:.1f}")

                for num_branches, num_keep in branch_keep_pairs:
                    config_key = (
                        f"b{num_branches}_k{num_keep}_"
                        f"{selector}_{'global' if use_global else 'class'}_"
                        f"start{branch_start_time}"
                    )

                    try:
                        fid_score = calculate_metrics(
                            sampler=sampler,
                            num_branches=num_branches,
                            num_keep=num_keep,
                            selector=selector,
                            use_global=use_global,
                            branch_start_time=branch_start_time,
                            device=device,
                            sigma=0.1,
                            n_samples=500,
                            fid=fid,
                        )

                        # Store results
                        results[config_key] = {
                            "branches": num_branches,
                            "keep": num_keep,
                            "selector": selector,
                            "global": use_global,
                            "branch_start_time": branch_start_time,
                            "fid_score": float(fid_score),
                        }

                        print(
                            f"  branches={num_branches}, keep={num_keep}: FID={fid_score:.4f}"
                        )

                        # Save results after each configuration
                        with open(f"{results_dir}/results.json", "w") as f:
                            json.dump(results, f, indent=2)

                    except Exception as e:
                        print(f"Error with configuration {config_key}: {str(e)}")
                        continue

    # Find and print best configurations
    sorted_results = sorted(results.items(), key=lambda x: x[1]["fid_score"])

    print("\nTop 10 Configurations:")
    for config_name, config_results in sorted_results[:10]:
        print(f"\nConfiguration: {config_name}")
        print(f"FID Score: {config_results['fid_score']:.4f}")
        print(f"Parameters:")
        print(f"  Branches: {config_results['branches']}")
        print(f"  Keep: {config_results['keep']}")
        print(f"  Selector: {config_results['selector']}")
        print(f"  Global: {config_results['global']}")
        print(f"  Branch Start Time: {config_results['branch_start_time']}")

    # Save final summary
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write("Top 10 Configurations:\n")
        for config_name, config_results in sorted_results[:10]:
            f.write(f"\nConfiguration: {config_name}\n")
            f.write(f"FID Score: {config_results['fid_score']:.4f}\n")
            f.write("Parameters:\n")
            f.write(f"  Branches: {config_results['branches']}\n")
            f.write(f"  Keep: {config_results['keep']}\n")
            f.write(f"  Selector: {config_results['selector']}\n")
            f.write(f"  Global: {config_results['global']}\n")
            f.write(f"  Branch Start Time: {config_results['branch_start_time']}\n")


if __name__ == "__main__":
    main()
