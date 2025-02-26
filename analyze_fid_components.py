import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchmetrics.image.fid as FID
import torchmetrics.image.inception as IS
from scipy.linalg import sqrtm

from mcts_single_flow import MCTSFlowSampler


def calculate_metrics_with_components(
    sampler, num_branches, num_keep, device, n_samples=1000, dt_std=0.1
):
    """
    Calculate FID and IS metrics for a specific branch/keep configuration across all classes.
    Also returns the components of the FID score.
    """
    # Initialize metrics
    fid = FID.FrechetInceptionDistance(normalize=True, feature=64).to(device)
    inception_score = IS.InceptionScore(normalize=True).to(device)

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
            generated_samples.extend(sample.cpu())

    # Process generated samples in batches for metrics
    generated_tensor = torch.stack(generated_samples)
    for i in range(0, len(generated_tensor), metric_batch_size):
        batch = generated_tensor[i : i + metric_batch_size].to(device)
        fid.update(batch, real=False)
        inception_score.update(batch)
        batch.cpu()
        torch.cuda.empty_cache()

    # Compute final scores
    fid_score = fid.compute()
    is_mean, is_std = inception_score.compute()

    # Get the FID components
    # Access the internal features from the FID metric
    real_features = fid.real_features.cpu().numpy()
    fake_features = fid.fake_features.cpu().numpy()

    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID components
    mean_distance = np.sum((mu_real - mu_fake) ** 2)

    trace_sigma_real = np.trace(sigma_real)
    trace_sigma_fake = np.trace(sigma_fake)

    # Calculate sqrt(Σ1*Σ2)
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    covmean_term = 2 * np.trace(covmean)
    covariance_distance = trace_sigma_real + trace_sigma_fake

    # Calculate percentages
    total_fid = mean_distance + covariance_distance - covmean_term
    mean_distance_percent = 100 * mean_distance / total_fid if total_fid > 0 else 0
    covariance_percent = (
        100 * (covariance_distance - covmean_term) / total_fid if total_fid > 0 else 0
    )

    components = {
        "fid_total": total_fid,
        "mean_distance": mean_distance,
        "covariance_distance": covariance_distance,
        "covmean_term": covmean_term,
        "mean_distance_percent": mean_distance_percent,
        "covariance_percent": covariance_percent,
    }

    return fid_score, is_mean, is_std, components


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

            # Calculate metrics with components
            fid_score, is_mean, is_std, components = calculate_metrics_with_components(
                sampler,
                num_branches=num_branches,
                num_keep=num_keep,
                device=device,
                n_samples=n_samples,
                dt_std=0.1,
            )

            print(f"   FID Score: {fid_score:.4f}")
            print(f"   Inception Score: {is_mean:.4f} ± {is_std:.4f}")

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

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize sampler
    sampler = MCTSFlowSampler(
        image_size=32,
        channels=3,
        device=device,
        num_timesteps=25,
        num_classes=10,
        buffer_size=10,
    )

    # Try to load the large flow model if available
    try:
        checkpoint = torch.load("saved_models/large_flow_model.pt", map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            sampler.flow_model.load_state_dict(checkpoint["model"])
            print(f"Loaded large flow model from checkpoint")
        else:
            sampler.flow_model.load_state_dict(checkpoint)
            print(f"Loaded large flow model")
    except Exception as e:
        print(f"Could not load large flow model: {e}")

    # Run analysis
    analyze_fid_components(
        sampler=sampler,
        device=device,
        branch_keep_configs=[(1, 1), (4, 2), (8, 4), (16, 8), (32, 16)],
        n_samples=2000,  # Use fewer samples for faster evaluation
    )


if __name__ == "__main__":
    main()
