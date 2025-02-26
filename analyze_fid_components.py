import torch
import numpy as np
import os
from tqdm import tqdm

from mcts_single_flow import MCTSFlowSampler
from train_mcts_flow import calculate_metrics


def analyze_fid_components(
    sampler,
    device,
    branch_keep_configs=[(1, 1), (4, 2), (8, 2), (16, 4), (32, 8)],
    n_samples=1000,
):
    """
    Analyze FID components across different branch/keep configurations.
    Simply prints the metrics for each configuration.
    """
    print("\n===== FID Component Analysis =====")
    print(f"Testing {len(branch_keep_configs)} branch/keep configurations")
    print(f"Using {n_samples} samples per configuration")

    # Run 10 loops to get more stable results
    for loop in range(10):
        print(f"\n----- Loop {loop+1}/10 -----")

        # Test each branch/keep configuration
        for num_branches, num_keep in branch_keep_configs:
            print(f"\nTesting branches={num_branches}, keep={num_keep}")

            # Calculate metrics
            fid_score, is_mean, is_std = calculate_metrics(
                sampler,
                num_branches=num_branches,
                num_keep=num_keep,
                device=device,
                n_samples=n_samples,
                sigma=0.0,  # No additional noise
            )

            print(f"   FID Score: {fid_score:.4f}")
            print(f"   Inception Score: {is_mean:.4f} ± {is_std:.4f}")

            # If we have the analyze_fid_components method in the sampler
            if hasattr(sampler, "analyze_fid_components"):
                # Generate samples for one class to analyze components
                samples = sampler.batch_sample_wdt(
                    class_label=0,  # Just use class 0 for component analysis
                    batch_size=100,
                    num_branches=num_branches,
                    num_keep=num_keep,
                    dt_std=0.1,  # Fixed dt_std value
                )

                # Analyze FID components
                components = sampler.analyze_fid_components(samples, 0)

                # Print component breakdown
                print(f"   FID Components (Class 0):")
                print(
                    f"     Mean Distance: {components['mean_distance']:.4f} ({components['mean_distance_percent']:.2f}%)"
                )
                print(
                    f"     Covariance: {components['covariance_distance'] - components['covmean_term']:.4f} ({components['covariance_percent']:.2f}%)"
                )


def main():
    # Use the specified GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize sampler
    sampler = MCTSFlowSampler(
        image_size=32,
        channels=3,
        device=device,
        num_timesteps=50,  # Use 50 timesteps to match the large model
        num_classes=10,
        buffer_size=10,
    )

    # Run analysis
    analyze_fid_components(
        sampler=sampler,
        device=device,
        branch_keep_configs=[(1, 1), (4, 2), (8, 2), (16, 4), (32, 8)],
        n_samples=1000,  # Use fewer samples for faster evaluation
    )


if __name__ == "__main__":
    main()
