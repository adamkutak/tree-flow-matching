import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchmetrics.image.fid as FID

from mcts_single_flow import MCTSFlowSampler


def analyze_early_quality_prediction(
    sampler,
    device,
    num_samples_per_class=100,
    evaluation_times=[0.5, 0.7, 0.9, 1.0],
    num_groups=4,
):
    """
    Analyze how early quality metrics correlate with final quality.

    This function generates samples with regular flow matching, measuring the global mean
    difference at specified timesteps. It then groups samples by their quality at each
    timestep and evaluates the final FID of each group.
    """
    print("\n===== Early Quality Prediction Analysis =====")

    # Check if global stats are available
    if not hasattr(sampler, "has_global_stats") or not sampler.has_global_stats:
        raise ValueError(
            "Global statistics not available in the sampler. Please initialize with global statistics."
        )

    sampler.flow_model.eval()

    # Total number of samples
    total_samples = num_samples_per_class * sampler.num_classes

    # Store all samples and their metrics - use rounded evaluation times as keys
    rounded_eval_times = [round(t, 1) for t in evaluation_times]
    quality_metrics = {t: [] for t in rounded_eval_times}
    all_samples = []
    class_labels = []

    # Generate samples
    with torch.no_grad():
        for class_idx in range(sampler.num_classes):
            print(f"Generating samples for class {class_idx}...")

            for sample_idx in tqdm(
                range(num_samples_per_class), desc=f"Class {class_idx} samples"
            ):
                # Start with random noise
                x = torch.randn(
                    1,
                    sampler.channels,
                    sampler.image_size,
                    sampler.image_size,
                    device=device,
                )
                label = torch.full((1,), class_idx, device=device)

                # Current time
                t = 0.0

                # Base timestep
                base_dt = 1.0 / sampler.num_timesteps

                # Regular flow matching sampling
                while t < 1.0 - 1e-6:
                    t_batch = torch.full((1,), t, device=device)
                    velocity = sampler.flow_model(t_batch, x, label)
                    x = x + velocity * base_dt

                    # Round the time to 1 decimal place for comparison
                    rounded_t = round(t + base_dt, 1)

                    # Check if this is an evaluation time
                    if rounded_t in rounded_eval_times:
                        # Calculate global mean difference
                        class_idx_tensor = torch.full(
                            (x.size(0),), class_idx, device=device
                        )
                        mean_diff = sampler.batch_compute_global_mean_difference(
                            x
                        ).item()
                        quality_metrics[rounded_t].append(mean_diff)

                    t += base_dt

                # Store the final sample
                all_samples.append(x.cpu())
                class_labels.append(class_idx)

    # Convert lists to tensors
    all_samples_tensor = torch.cat(all_samples, dim=0)
    class_labels_tensor = torch.tensor(class_labels, device=device)

    # Group samples by quality at each evaluation time
    results = {}

    # Setup CIFAR-10 dataset for real FID calculation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Initialize FID metric
    fid_metric = FID.FrechetInceptionDistance(
        feature=64, normalize=True, reset_real_features=False
    ).to(device)
    # Process real images
    real_batch_size = 100
    indices = np.random.choice(len(cifar10), 50000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)
    print("Processing real images for FID calculation...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid_metric.update(batch, real=True)
        torch.cuda.empty_cache()
    # Print and compare the global means
    print("\n===== Comparing Global Means =====")
    # Get the FID module's computed mean
    fid_real_mean = fid_metric.real_features_sum / fid_metric.real_features_num_samples
    # Get the sampler's global mean
    sampler_global_mean = torch.from_numpy(sampler.global_fid["mu"]).to(device)
    print(f"FID module real mean shape: {fid_real_mean.shape}")
    print(f"Sampler global mean shape: {sampler_global_mean.shape}")
    # Calculate the difference between the means
    if fid_real_mean.shape == sampler_global_mean.shape:
        mean_diff_norm = torch.norm(fid_real_mean - sampler_global_mean).item()
        print(f"L2 norm of difference between means: {mean_diff_norm:.6f}")
        # Print the first few elements of each mean for comparison
        num_elements = min(10, fid_real_mean.shape[0])
        print(f"\nFirst {num_elements} elements of FID module mean:")
        print(fid_real_mean[:num_elements].cpu().numpy())
        print(f"\nFirst {num_elements} elements of sampler global mean:")
        print(sampler_global_mean[:num_elements].cpu().numpy())
    else:
        print(
            f"WARNING: Mean shapes don't match - FID: {fid_real_mean.shape}, Sampler: {sampler_global_mean.shape}"
        )

    for eval_time in evaluation_times:
        print(f"\n===== Analyzing samples grouped by quality at t={eval_time} =====")

        # Skip if we don't have metrics for this time (shouldn't happen with our setup)
        if not quality_metrics.get(eval_time):
            print(f"No metrics available for t={eval_time}, skipping...")
            continue

        # Group samples by class and quality
        class_grouped_samples = {c: [] for c in range(sampler.num_classes)}
        class_grouped_metrics = {c: [] for c in range(sampler.num_classes)}

        for i, (sample, label, metric) in enumerate(
            zip(all_samples, class_labels, quality_metrics[eval_time])
        ):
            class_grouped_samples[label].append((i, sample))
            class_grouped_metrics[label].append((i, metric))

        # Create quality groups for each class
        quality_groups = {g: [] for g in range(num_groups)}

        for class_idx in range(sampler.num_classes):
            # Sort samples by quality metric (higher is better)
            sorted_metrics = sorted(
                class_grouped_metrics[class_idx], key=lambda x: x[1], reverse=True
            )

            # Divide into groups
            samples_per_group = len(sorted_metrics) // num_groups

            for g in range(num_groups):
                start_idx = g * samples_per_group
                end_idx = (
                    (g + 1) * samples_per_group
                    if g < num_groups - 1
                    else len(sorted_metrics)
                )

                for i in range(start_idx, end_idx):
                    sample_idx = sorted_metrics[i][0]
                    quality_groups[g].append(sample_idx)

        # Calculate FID for each quality group
        group_fid_scores = {}

        # Calculate FID for each quality group
        for group_idx in range(num_groups):
            # Reset FID for fake images
            fid_metric.reset()

            # Get samples for this group
            group_samples = [all_samples[i] for i in quality_groups[group_idx]]
            group_samples_tensor = torch.cat(group_samples, dim=0)

            # Calculate average quality metric for this group
            group_metrics = [
                quality_metrics[eval_time][i] for i in quality_groups[group_idx]
            ]
            avg_metric = np.mean(group_metrics)

            # Process in batches
            batch_size = 64
            print(
                f"Processing quality group {group_idx+1}/{num_groups} (avg metric: {avg_metric:.4f})..."
            )

            for i in range(0, len(group_samples_tensor), batch_size):
                batch = group_samples_tensor[i : i + batch_size].to(device)
                fid_metric.update(batch, real=False)
                torch.cuda.empty_cache()

            # Compute FID score
            fid_score = fid_metric.compute().item()

            # Get the real statistics from the FID metric
            real_mean = (
                fid_metric.real_features_sum / fid_metric.real_features_num_samples
            )
            real_cov = (
                fid_metric.real_features_cov_sum / fid_metric.real_features_num_samples
            )

            # Get fake statistics
            fake_mean = (
                fid_metric.fake_features_sum / fid_metric.fake_features_num_samples
            )
            fake_cov = (
                fid_metric.fake_features_cov_sum / fid_metric.fake_features_num_samples
            )

            # Calculate mean component: ||μ_real - μ_fake||²
            mean_diff = torch.norm(real_mean - fake_mean) ** 2

            real_cov_np = real_cov.cpu().numpy()
            fake_cov_np = fake_cov.cpu().numpy()

            # Calculate sqrt(Σ_real·Σ_fake) using scipy
            from scipy import linalg

            sqrt_cov_product = linalg.sqrtm(np.matmul(real_cov_np, fake_cov_np))

            # Handle potential complex numbers (should be real but might have small imaginary parts due to numerical issues)
            if np.iscomplexobj(sqrt_cov_product):
                sqrt_cov_product = sqrt_cov_product.real

            # Convert back to tensor and calculate trace
            sqrt_cov_product_tensor = torch.from_numpy(sqrt_cov_product).to(device)
            cov_component = torch.trace(
                real_cov + fake_cov - 2 * sqrt_cov_product_tensor
            ).item()

            # Store components along with the FID score
            group_fid_scores[group_idx] = {
                "fid": fid_score,
                "mean_component": mean_diff.item(),
                "cov_component": cov_component,
                "avg_metric": avg_metric,
                "num_samples": len(group_samples),
            }

            print(
                f"  Group {group_idx+1} (top {(group_idx+1)*100/num_groups:.0f}%): FID = {fid_score:.4f}, "
                f"Mean Component = {mean_diff.item():.4f}, Covariance Component = {cov_component:.4f}"
            )

        # Calculate correlation between group quality and FID
        group_metrics = [group_fid_scores[g]["avg_metric"] for g in range(num_groups)]
        group_fids = [group_fid_scores[g]["fid"] for g in range(num_groups)]

        if len(group_metrics) > 1:
            correlation, p_value = spearmanr(group_metrics, group_fids)
        else:
            correlation, p_value = float("nan"), float("nan")

        print(
            f"\nCorrelation between t={eval_time} quality metric and final FID: {correlation:.4f} (p-value: {p_value:.4f})"
        )

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(group_metrics, group_fids, s=100)

        # Add group labels
        for i in range(num_groups):
            plt.annotate(
                f"Group {i+1}",
                (group_metrics[i], group_fids[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.xlabel(f"Average Quality Metric at t={eval_time} (higher is better)")
        plt.ylabel("FID Score (lower is better)")
        plt.title(f"Relationship Between t={eval_time} Quality and Final FID")
        plt.grid(True)
        plt.savefig(f"quality_prediction_t{eval_time:.1f}.png")
        plt.close()

        # Store results
        results[eval_time] = {
            "group_fid_scores": group_fid_scores,
            "correlation": correlation,
            "p_value": p_value,
        }

    # Compare correlations across different evaluation times
    eval_times_list = list(results.keys())
    correlations = [results[t]["correlation"] for t in eval_times_list]

    plt.figure(figsize=(10, 6))
    plt.plot(eval_times_list, correlations, "o-", linewidth=2)
    plt.xlabel("Evaluation Time")
    plt.ylabel("Correlation with Final FID")
    plt.title("Predictive Power of Quality Metric at Different Times")
    plt.grid(True)
    plt.savefig("quality_prediction_correlation_by_time.png")
    plt.close()

    print("\n===== Summary of Quality Prediction Analysis =====")
    print("Correlation between quality metric and final FID at different times:")
    for t in eval_times_list:
        print(
            f"  t={t:.1f}: {results[t]['correlation']:.4f} (p-value: {results[t]['p_value']:.4f})"
        )

    return results


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
        buffer_size=10,
        flow_model="large_flow_model.pt",
        value_model="value_model.pt",  # Still needed for sampler initialization
        num_channels=256,
        inception_layer=0,
        pca_dim=None,
    )

    # Run early quality prediction analysis
    early_quality_results = analyze_early_quality_prediction(
        sampler=sampler,
        device=device,
        num_samples_per_class=100,
        evaluation_times=[0.5, 0.7, 0.9, 1.0],
        num_groups=4,
    )


if __name__ == "__main__":
    main()
