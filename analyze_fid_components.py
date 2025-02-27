import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchmetrics.image.fid as FID
from scipy.linalg import sqrtm

from mcts_single_flow import MCTSFlowSampler


def calculate_metrics_with_components(
    sampler, num_branches, num_keep, device, n_samples=1000, dt_std=0.1
):
    """
    Calculate FID metrics for a specific branch/keep configuration across all classes.
    Also returns the components of the FID score.
    """
    # Initialize metrics
    fid = FID.FrechetInceptionDistance(normalize=True, feature=2048).to(device)

    # Get random real images from CIFAR-10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    cifar10 = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Randomly sample real images
    indices = np.random.choice(len(cifar10), n_samples, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    # Process real images in batches
    real_batch_size = 100
    print("Processing real images...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid.update(batch, real=True)

    # Generate samples evenly across all classes
    samples_per_class = n_samples // sampler.num_classes
    generation_batch_size = 64
    metric_batch_size = 64
    generated_samples = []

    print(
        f"\nGenerating {n_samples} samples for branches={num_branches}, keep={num_keep}"
    )

    # Generate samples for each class
    for class_label in range(sampler.num_classes):
        num_batches = samples_per_class // generation_batch_size

        # Generate full batches
        for _ in range(num_batches):
            sample = sampler.batch_sample_wdt(
                class_label=class_label,
                batch_size=generation_batch_size,
                num_branches=num_branches,
                num_keep=num_keep,
                dt_std=dt_std,
            )
            generated_samples.append(sample)

    # Stack all generated samples
    generated_tensor = torch.cat(generated_samples, dim=0)

    # Process generated samples in batches for metrics
    for i in range(0, len(generated_tensor), metric_batch_size):
        batch = generated_tensor[i : i + metric_batch_size].to(device)
        fid.update(batch, real=False)
        torch.cuda.empty_cache()

    # Compute final scores
    fid_score = fid.compute()

    # Extract FID components from the internal state
    # Calculate means
    mu_real = fid.real_features_sum / fid.real_features_num_samples
    mu_fake = fid.fake_features_sum / fid.fake_features_num_samples

    # Calculate covariances
    cov_real_num = (
        fid.real_features_cov_sum
        - fid.real_features_num_samples
        * mu_real.unsqueeze(0).t()
        @ mu_real.unsqueeze(0)
    )
    cov_real = cov_real_num / (fid.real_features_num_samples - 1)

    cov_fake_num = (
        fid.fake_features_cov_sum
        - fid.fake_features_num_samples
        * mu_fake.unsqueeze(0).t()
        @ mu_fake.unsqueeze(0)
    )
    cov_fake = cov_fake_num / (fid.fake_features_num_samples - 1)

    # Calculate FID components
    mean_distance = torch.sum((mu_real - mu_fake) ** 2).item()
    trace_term = cov_real.trace().item() + cov_fake.trace().item()

    # Calculate sqrt(cov_real @ cov_fake)
    cov_real_np = cov_real.cpu().numpy()
    cov_fake_np = cov_fake.cpu().numpy()
    covmean = sqrtm(cov_real_np @ cov_fake_np)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean_term = 2 * np.trace(covmean)

    # Calculate total FID (should match fid_score)
    total_fid = mean_distance + trace_term - covmean_term

    # Calculate percentages
    mean_distance_percent = 100 * mean_distance / total_fid if total_fid > 0 else 0
    covariance_percent = (
        100 * (trace_term - covmean_term) / total_fid if total_fid > 0 else 0
    )

    components = {
        "fid_total": total_fid,
        "mean_distance": mean_distance,
        "covariance_distance": trace_term,
        "covmean_term": covmean_term,
        "mean_distance_percent": mean_distance_percent,
        "covariance_percent": covariance_percent,
    }

    return fid_score, components


def analyze_fid_components(
    sampler,
    device,
    branch_keep_configs=[(1, 1), (4, 2), (8, 2), (16, 4)],
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

            # Calculate metrics with components
            fid_score, components = calculate_metrics_with_components(
                sampler,
                num_branches=num_branches,
                num_keep=num_keep,
                device=device,
                n_samples=n_samples,
                dt_std=0.05,
            )

            print(f"   FID Score: {fid_score:.4f}")

            # Print component breakdown
            print(f"   FID Components:")
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

    # Initialize sampler
    sampler = MCTSFlowSampler(
        image_size=32,
        channels=3,
        device=device,
        num_timesteps=25,
        num_classes=10,
        buffer_size=10,
    )

    # Run analysis
    analyze_fid_components(
        sampler=sampler,
        device=device,
        branch_keep_configs=[(1, 1), (4, 2), (8, 4), (16, 8), (32, 16)],
        n_samples=2000,
    )


if __name__ == "__main__":
    main()
