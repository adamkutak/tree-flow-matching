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


def analyze_batch_fid_rank_consistency(
    sampler,
    device,
    num_samples=256,
    num_branches=8,
    dt_std=0.1,
    batch_size=32,
    class_label=None,
):
    """
    Analyze how consistently selecting the nth-ranked branch by batch FID affects final FID.

    This function generates trajectories where at each possible branching point, we consistently
    select the branch ranked n (where n=1 is best, n=2 is second-best, etc.) according
    to the intermediate batch FID calculation. We then measure the final FID of these trajectories.

    Args:
        sampler: The MCTSFlowSampler instance
        device: The device to run computations on
        num_samples: Total number of samples to analyze
        num_branches: Number of branches to create at each branch point (also determines max rank)
        dt_std: Standard deviation for dt variation when branching
        batch_size: Size of batches for batch FID calculation
        class_label: Specific class to analyze (if None, samples across all classes)

    Returns:
        Dictionary containing correlation metrics and raw data for plotting
    """
    print("\n===== Batch FID Rank Consistency Analysis =====")

    sampler.flow_model.eval()

    # Ensure num_samples is divisible by batch_size
    num_samples = (num_samples // batch_size) * batch_size

    # Sample across all classes or specific class
    if class_label is None:
        # Distribute samples across classes
        samples_per_class = num_samples // sampler.num_classes
        class_range = range(sampler.num_classes)
    else:
        samples_per_class = num_samples
        class_range = [class_label]

    base_dt = 1.0 / sampler.num_timesteps

    # Store results for each rank
    rank_results = {
        rank: {
            "batch_fid_scores": [],
            "final_samples": [],  # Store final samples for real FID calculation
            "running_mean": None,
            "running_cov": None,
            "num_accumulated": 0,
        }
        for rank in range(1, num_branches + 1)
    }

    def extract_features(images):
        with torch.no_grad():
            features = sampler.extract_inception_features(images)
            # Convert to tensor if it's a numpy array
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).to(device)
            return features

    # Function to compute batch statistics using sampler's inception model
    def compute_batch_statistics(samples):
        features = extract_features(samples)
        mean = torch.mean(features, dim=0)
        # Center the features before computing covariance
        centered_features = features - mean.unsqueeze(0)
        # Compute covariance matrix
        cov = torch.mm(centered_features.t(), centered_features) / (
            features.size(0) - 1
        )
        return mean, cov, features.size(0)

    # Function to update running statistics with dynamic alpha based on sample sizes
    def update_running_statistics(
        running_mean, running_cov, num_accumulated, new_mean, new_cov, new_samples
    ):
        if running_mean is None:
            return new_mean, new_cov, new_samples

        # Compute alpha based on relative sizes
        total_samples = num_accumulated + new_samples
        alpha = num_accumulated / total_samples  # Weight for old statistics
        beta = new_samples / total_samples  # Weight for new statistics

        updated_mean = alpha * running_mean + beta * new_mean
        updated_cov = alpha * running_cov + beta * new_cov

        return updated_mean, updated_cov, total_samples

    # Function to compute FID between two sets of statistics
    def compute_fid(mean1, cov1, mean2, cov2):
        # Calculate squared distance between means
        mean_diff = torch.sum((mean1 - mean2) ** 2)

        # Calculate sqrt of product of covariances
        import numpy as np
        from scipy import linalg

        cov1_np = cov1.cpu().numpy()
        cov2_np = cov2.cpu().numpy()

        # Calculate sqrt(cov1 * cov2)
        covmean = linalg.sqrtm(np.matmul(cov1_np, cov2_np))

        # Check for numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Convert back to torch tensor
        covmean_tensor = torch.from_numpy(covmean).to(device)

        # Calculate trace term
        trace_term = torch.trace(cov1 + cov2 - 2 * covmean_tensor)

        # FID = mean_diff + trace_term
        fid = mean_diff + trace_term

        return fid.item()

    # Get reference statistics from real data
    print("Computing reference statistics from real data...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Sample a subset of real images
    indices = np.random.choice(len(cifar10), 10000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    # Compute reference statistics in batches
    ref_features_list = []
    batch_size_ref = 100
    for i in range(0, len(real_images), batch_size_ref):
        batch = real_images[i : i + batch_size_ref]
        with torch.no_grad():
            features = extract_features(batch)
            ref_features_list.append(features)

    ref_features = torch.cat(ref_features_list, dim=0)
    ref_mean = torch.mean(ref_features, dim=0)
    centered_features = ref_features - ref_mean.unsqueeze(0)
    ref_cov = torch.mm(centered_features.t(), centered_features) / (
        ref_features.size(0) - 1
    )

    print(f"Reference statistics computed from {ref_features.size(0)} real images")

    # Process each rank separately
    with torch.no_grad():
        for rank in range(1, num_branches + 1):
            print(f"\nProcessing rank {rank}...")

            # Initialize with random noise samples
            initial_noises = []
            labels = []

            # Create initial noise and labels
            for class_idx in class_range:
                samples_this_class = (
                    samples_per_class if class_label is None else num_samples
                )
                for _ in range(samples_this_class // sampler.num_classes):
                    initial_noises.append(
                        torch.randn(
                            1,
                            sampler.channels,
                            sampler.image_size,
                            sampler.image_size,
                            device=device,
                        )
                    )
                    labels.append(torch.full((1,), class_idx, device=device))

            # Process in batches
            num_batches = len(initial_noises) // batch_size

            for batch_idx in range(num_batches):
                print(f"Processing batch {batch_idx+1}/{num_batches} for rank {rank}")

                # Get batch of initial noises and labels
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size

                batch_noises = initial_noises[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]

                # Initialize batch samples
                batch_samples = [noise.clone() for noise in batch_noises]
                batch_times = [0.0] * batch_size

                # Track if each sample is still being processed
                active_samples = [True] * batch_size

                # Process until all samples reach t=1.0
                while any(active_samples):
                    # Process each sample in the batch
                    for i in range(batch_size):
                        if not active_samples[i]:
                            continue

                        # Current sample and time
                        x = batch_samples[i]
                        t = batch_times[i]
                        label = batch_labels[i]

                        # Check if we can branch (at least 2 more steps remaining)
                        can_branch = t < 1.0 - 2 * base_dt

                        if can_branch:
                            # Create branches
                            branches = x.repeat(num_branches, 1, 1, 1)
                            branch_labels = label.repeat(num_branches)
                            branch_times = torch.full((num_branches,), t, device=device)

                            # Sample different dt values for each branch
                            dts = torch.normal(
                                mean=base_dt,
                                std=dt_std * base_dt,
                                size=(num_branches,),
                                device=device,
                            )
                            dts = torch.clamp(
                                dts,
                                min=torch.tensor(0.0, device=device),
                                max=1.0 - branch_times,
                            )

                            # Get velocity for all branches
                            velocity = sampler.flow_model(
                                branch_times, branches, branch_labels
                            )

                            # Apply different step sizes to create branches
                            branched_samples = branches + velocity * dts.view(
                                -1, 1, 1, 1
                            )
                            new_times = branch_times + dts

                            # Now simulate all branches forward one more step to a common time point
                            next_timestep = t + 2 * base_dt  # Always advance 2 steps

                            # Calculate dt to reach the next common timestep for each branch
                            dt_to_next = next_timestep - new_times

                            # Get velocity for all branches
                            velocity = sampler.flow_model(
                                new_times, branched_samples, branch_labels
                            )

                            # Apply the step to all branches
                            aligned_samples = (
                                branched_samples
                                + velocity * dt_to_next.view(-1, 1, 1, 1)
                            )

                            # Calculate intermediate FID for all aligned branches
                            # For batch-wise selection, we'll collect features for each branch
                            branch_features_list = []

                            for branch_idx in range(num_branches):
                                branch_sample = aligned_samples[
                                    branch_idx : branch_idx + 1
                                ]
                                branch_features = extract_features(branch_sample)
                                branch_features_list.append(branch_features)

                            # Calculate FID for each branch compared to reference
                            branch_fids = []

                            for branch_idx in range(num_branches):
                                branch_features = branch_features_list[branch_idx]
                                branch_mean = torch.mean(branch_features, dim=0)

                                # For single sample, covariance is not well-defined
                                # We'll use a small identity matrix as a placeholder
                                branch_cov = (
                                    torch.eye(branch_mean.size(0), device=device) * 1e-6
                                )

                                # Compute FID with reference statistics
                                branch_fid = compute_fid(
                                    branch_mean, branch_cov, ref_mean, ref_cov
                                )
                                branch_fids.append(branch_fid)

                            # Sort branches by FID (ascending, as lower FID is better)
                            sorted_indices = torch.argsort(torch.tensor(branch_fids))

                            # Select the branch with the specified rank (rank 1 = best, rank num_branches = worst)
                            selected_idx = sorted_indices[rank - 1]
                            batch_samples[i] = aligned_samples[
                                selected_idx : selected_idx + 1
                            ]

                            # Update time
                            batch_times[i] = next_timestep

                        else:
                            # Regular step (no branching) for the final steps
                            t_batch = torch.full((1,), t, device=device)
                            velocity = sampler.flow_model(t_batch, x, label)
                            batch_samples[i] = x + velocity * base_dt
                            batch_times[i] += base_dt

                        # Check if this sample is done
                        if batch_times[i] >= 1.0 - 1e-6:
                            active_samples[i] = False

                # Collect final samples for this batch
                final_batch_samples = torch.cat(batch_samples, dim=0)

                # Compute batch statistics
                batch_mean, batch_cov, batch_count = compute_batch_statistics(
                    final_batch_samples
                )

                # Update running statistics with dynamic alpha based on sample counts
                (
                    rank_results[rank]["running_mean"],
                    rank_results[rank]["running_cov"],
                    rank_results[rank]["num_accumulated"],
                ) = update_running_statistics(
                    rank_results[rank]["running_mean"],
                    rank_results[rank]["running_cov"],
                    rank_results[rank]["num_accumulated"],
                    batch_mean,
                    batch_cov,
                    batch_count,
                )

                # Compute batch FID with reference
                batch_fid = compute_fid(batch_mean, batch_cov, ref_mean, ref_cov)
                rank_results[rank]["batch_fid_scores"].append(batch_fid)

                # Store final samples for later analysis
                rank_results[rank]["final_samples"].append(final_batch_samples.cpu())

                print(f"  Batch {batch_idx+1} FID: {batch_fid:.4f}")

            # Compute overall FID for this rank using running statistics
            overall_fid = compute_fid(
                rank_results[rank]["running_mean"],
                rank_results[rank]["running_cov"],
                ref_mean,
                ref_cov,
            )

            print(f"Rank {rank} overall FID: {overall_fid:.4f}")

    # Calculate statistics for batch FID scores
    rank_avg_batch_fid = {}
    rank_std_batch_fid = {}
    rank_overall_fid = {}

    for rank in range(1, num_branches + 1):
        batch_fids = rank_results[rank]["batch_fid_scores"]
        rank_avg_batch_fid[rank] = np.mean(batch_fids)
        rank_std_batch_fid[rank] = np.std(batch_fids)

        # Calculate overall FID using running statistics
        overall_fid = compute_fid(
            rank_results[rank]["running_mean"],
            rank_results[rank]["running_cov"],
            ref_mean,
            ref_cov,
        )
        rank_overall_fid[rank] = overall_fid

    # Calculate correlation between rank and average batch FID
    ranks = list(range(1, num_branches + 1))
    avg_batch_fids = [rank_avg_batch_fid[r] for r in ranks]
    overall_fids = [rank_overall_fid[r] for r in ranks]

    if len(ranks) > 1:
        rank_batch_fid_correlation, batch_p_value = spearmanr(ranks, avg_batch_fids)
        rank_overall_fid_correlation, overall_p_value = spearmanr(ranks, overall_fids)
    else:
        rank_batch_fid_correlation, batch_p_value = float("nan"), float("nan")
        rank_overall_fid_correlation, overall_p_value = float("nan"), float("nan")

    # Print results
    print("\n===== Batch FID Rank Consistency Results =====")
    print(
        f"Correlation between rank and average batch FID: {rank_batch_fid_correlation:.4f} (p-value: {batch_p_value:.4f})"
    )
    print(
        f"Correlation between rank and overall FID: {rank_overall_fid_correlation:.4f} (p-value: {overall_p_value:.4f})"
    )

    print("\nAverage batch FID by rank:")
    for rank in range(1, num_branches + 1):
        print(
            f"  Rank {rank}: {rank_avg_batch_fid[rank]:.4f} ± {rank_std_batch_fid[rank]:.4f}"
        )

    print("\nOverall FID by rank:")
    for rank in range(1, num_branches + 1):
        print(f"  Rank {rank}: {rank_overall_fid[rank]:.4f}")

    # Plot results for batch FID
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        ranks,
        avg_batch_fids,
        yerr=[rank_std_batch_fid[r] for r in ranks],
        fmt="o-",
        capsize=5,
    )
    plt.xlabel("Branch Rank by Intermediate FID (1 = best)")
    plt.ylabel("Average Batch FID Score (lower is better)")
    plt.title("Batch FID by Consistently Selected Branch Rank")
    plt.grid(True)
    plt.savefig("batch_fid_rank_consistency.png")
    plt.close()

    # Plot results for overall FID
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, overall_fids, "o-")
    plt.xlabel("Branch Rank by Intermediate FID (1 = best)")
    plt.ylabel("Overall FID Score (lower is better)")
    plt.title("Overall FID by Consistently Selected Branch Rank")
    plt.grid(True)
    plt.savefig("overall_fid_rank_consistency.png")
    plt.close()

    # Return results
    return {
        "rank_avg_batch_fid": rank_avg_batch_fid,
        "rank_std_batch_fid": rank_std_batch_fid,
        "rank_overall_fid": rank_overall_fid,
        "batch_fid_correlation": rank_batch_fid_correlation,
        "batch_fid_p_value": batch_p_value,
        "overall_fid_correlation": rank_overall_fid_correlation,
        "overall_fid_p_value": overall_p_value,
        "rank_results": rank_results,
    }


# Update the main function to include the new analysis
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

    # Run batch FID rank consistency analysis
    batch_fid_analysis_results = analyze_batch_fid_rank_consistency(
        sampler=sampler,
        device=device,
        num_samples=256,  # Use a larger number of samples for batch analysis
        num_branches=8,
        dt_std=0.25,
        batch_size=32,
    )


if __name__ == "__main__":
    main()
