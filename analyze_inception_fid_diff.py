import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchmetrics.image.fid as FID
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from mcts_single_flow import MCTSFlowSampler


def analyze_fid_correlation(
    sampler,
    device,
    num_timesteps_list=[2, 4, 6, 8, 10, 12, 14, 16, 20],
    samples_per_timestep=500,
    batch_size=16,
):
    """
    Analyze correlation between our simplified FID calculation and the standard FID metric.

    Args:
        sampler: The MCTSFlowSampler instance
        device: The device to run computations on
        num_timesteps_list: List of different timestep counts to test
        samples_per_timestep: Number of samples to generate for each timestep count
        batch_size: Batch size for generation and evaluation

    Returns:
        Dictionary containing correlation metrics and raw data for plotting
    """
    print("\n===== FID Correlation Analysis =====")

    # Initialize standard FID metric
    fid_metric = FID.FrechetInceptionDistance(
        normalize=True, reset_real_features=False
    ).to(device)

    # Load real images for FID calculation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Randomly sample real images
    indices = np.random.choice(len(cifar10), 5000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    # Process real images in batches for standard FID
    real_batch_size = 100
    print("Processing real images for standard FID calculation...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid_metric.update(batch, real=True)

    # Data collection
    results = []

    # Set models to eval mode
    sampler.flow_model.eval()
    if sampler.value_model:
        sampler.value_model.eval()

    # Store original number of timesteps to restore later
    original_num_timesteps = sampler.num_timesteps

    with torch.no_grad():
        for num_timesteps in num_timesteps_list:
            print(f"\nTesting with {num_timesteps} timesteps")

            # Update the sampler with the new number of timesteps
            sampler.set_timesteps(num_timesteps)

            # Reset the standard FID metric for fake images
            fid_metric.reset_fake_features()

            # Generate samples evenly across all classes
            samples_per_class = samples_per_timestep // sampler.num_classes
            all_simplified_fid_changes = []

            # Generate samples for each class
            for class_label in range(sampler.num_classes):
                print(f"Generating samples for class {class_label}...")

                # Process in batches
                num_batches = (samples_per_class + batch_size - 1) // batch_size
                for batch_idx in tqdm(range(num_batches)):
                    # Determine actual batch size (might be smaller for last batch)
                    actual_batch_size = min(
                        batch_size, samples_per_class - batch_idx * batch_size
                    )
                    if actual_batch_size <= 0:
                        break

                    # Generate samples with num_branches=1, num_keep=1 (standard flow matching)
                    samples = sampler.batch_sample(
                        class_label=class_label,
                        batch_size=actual_batch_size,
                        num_branches=1,
                        num_keep=1,
                    )

                    # Update standard FID metric
                    fid_metric.update(samples, real=False)

                    # Calculate simplified FID change for each sample
                    for i in range(actual_batch_size):
                        simplified_fid_change = sampler.compute_fid_change(
                            samples[i : i + 1], class_label
                        )
                        all_simplified_fid_changes.append(simplified_fid_change)

            # Compute standard FID score
            standard_fid_score = fid_metric.compute().item()

            # Calculate average simplified FID change
            avg_simplified_fid_change = np.mean(all_simplified_fid_changes)
            std_simplified_fid_change = np.std(all_simplified_fid_changes)

            print(f"Standard FID score: {standard_fid_score:.4f}")
            print(
                f"Average simplified FID change: {avg_simplified_fid_change:.4f} ± {std_simplified_fid_change:.4f}"
            )

            # Store results
            results.append(
                {
                    "num_timesteps": num_timesteps,
                    "standard_fid": standard_fid_score,
                    "simplified_fid_change": avg_simplified_fid_change,
                    "simplified_fid_std": std_simplified_fid_change,
                }
            )

    # Restore original number of timesteps
    sampler.set_timesteps(original_num_timesteps)

    # Extract data for correlation analysis
    standard_fids = [result["standard_fid"] for result in results]
    simplified_fid_changes = [result["simplified_fid_change"] for result in results]

    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(simplified_fid_changes, standard_fids)
    spearman_corr, spearman_p = spearmanr(simplified_fid_changes, standard_fids)

    print("\n===== Correlation Results =====")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman rank correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(simplified_fid_changes, standard_fids)

    # Add labels for each point
    for i, result in enumerate(results):
        plt.annotate(
            f"{result['num_timesteps']}",
            (simplified_fid_changes[i], standard_fids[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Add trend line
    z = np.polyfit(simplified_fid_changes, standard_fids, 1)
    p = np.poly1d(z)
    plt.plot(simplified_fid_changes, p(simplified_fid_changes), "r--", alpha=0.8)

    plt.xlabel("Simplified FID Change (64-dim)")
    plt.ylabel("Standard FID Score (2048-dim)")
    plt.title(
        f"Correlation between Simplified and Standard FID\nPearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}"
    )
    plt.grid(True, alpha=0.3)
    plt.savefig("fid_correlation.png")
    plt.show()

    return {
        "results": results,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "standard_fids": standard_fids,
        "simplified_fid_changes": simplified_fid_changes,
    }


def main():
    # Use the specified GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize sampler
    sampler = MCTSFlowSampler(
        image_size=32,
        channels=3,
        device=device,
        num_timesteps=10,
        num_classes=10,
        buffer_size=100,
        flow_model="large_flow_model.pt",
        value_model=None,
        num_channels=256,
        inception_layer=0,  # Use the lower layer (64-dim) for simplified FID
    )

    # Run FID correlation analysis
    correlation_results = analyze_fid_correlation(
        sampler=sampler,
        device=device,
        num_timesteps_list=[2, 4, 6, 8, 10, 12, 14, 16, 20],
        samples_per_timestep=500,
        batch_size=16,
    )


if __name__ == "__main__":
    main()
