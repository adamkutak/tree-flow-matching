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
    base_dt=0.1,
    branch_start_t=0.0,  # When to start branching (0.0 = start immediately, 0.5 = start halfway)
    class_label=None,
):
    """
    Analyze how consistently selecting the nth-ranked branch by batch FID affects final FID.

    This function generates trajectories where at each possible branching point, we create
    num_branches different versions of the entire batch, compute FID for each branched batch,
    and consistently select the branch ranked n (where n=1 is best, n=2 is second-best, etc.)
    according to the batch FID calculation.

    Args:
        sampler: The MCTSFlowSampler instance
        device: The device to run computations on
        num_samples: Total number of samples to analyze
        num_branches: Number of branches to create at each branch point
        dt_std: Standard deviation for dt variation when branching
        batch_size: Size of batches for batch FID calculation
        base_dt: Base step size for flow matching
        branch_start_t: Time value at which to start branching (0.0-1.0)
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

    # Function to extract inception features and ensure they're tensors
    def extract_features(images):
        with torch.no_grad():
            features = sampler.extract_inception_features(images)
            # Convert to tensor if it's a numpy array
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).to(device)
            return features

    # Function to compute batch statistics
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
    indices = np.random.choice(len(cifar10), 5000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    # Compute reference statistics in batches
    ref_features_list = []
    batch_size_ref = 100
    for i in range(0, len(real_images), batch_size_ref):
        batch = real_images[i : i + batch_size_ref]
        with torch.no_grad():
            features = extract_features(batch)
            ref_features_list.append(features.cpu())  # Move to CPU to save GPU memory

        # Clear GPU memory
        del batch, features
        torch.cuda.empty_cache()

    # Compute reference statistics
    ref_features = torch.cat([f.to(device) for f in ref_features_list], dim=0)
    ref_mean = torch.mean(ref_features, dim=0)
    centered_features = ref_features - ref_mean.unsqueeze(0)
    ref_cov = torch.mm(centered_features.t(), centered_features) / (
        ref_features.size(0) - 1
    )

    # Clear GPU memory
    del ref_features, centered_features, ref_features_list
    torch.cuda.empty_cache()

    print(f"Reference statistics computed from {real_images.size(0)} real images")

    # Initialize running statistics with 1024 samples for each rank
    print("\nInitializing running statistics with 1024 samples...")

    # Generate 1024 samples using standard flow matching up to branch_start_t
    init_samples_count = 1024
    init_batch_size = 32  # Process in smaller batches

    # Create initial noise and labels for initialization
    init_noises = []
    init_labels = []

    for class_idx in class_range:
        samples_per_init_class = init_samples_count // len(class_range)
        for _ in range(samples_per_init_class):
            init_noises.append(
                torch.randn(
                    1,
                    sampler.channels,
                    sampler.image_size,
                    sampler.image_size,
                    device=device,
                )
            )
            init_labels.append(torch.full((1,), class_idx, device=device))

    # Process initialization samples in batches
    init_features_list = []  # Store features instead of samples

    for i in tqdm(range(0, len(init_noises), init_batch_size), desc="Initializing"):
        batch_noises = torch.cat(init_noises[i : i + init_batch_size], dim=0)
        batch_labels = torch.cat(init_labels[i : i + init_batch_size], dim=0)

        # Standard flow matching until branch_start_t
        x = batch_noises
        t = 0.0

        while t < branch_start_t - 1e-6:
            t_batch = torch.full((batch_noises.size(0),), t, device=device)
            velocity = sampler.flow_model(t_batch, x, batch_labels)
            dt = min(base_dt, branch_start_t - t)
            x = x + velocity * dt
            t += dt

        # Extract features directly and move to CPU
        features = extract_features(x)
        init_features_list.append(features.cpu())

        # Clear GPU memory
        del x, batch_noises, batch_labels, features
        torch.cuda.empty_cache()

    # Compute statistics from features
    all_features = torch.cat([f.to(device) for f in init_features_list], dim=0)
    init_mean = torch.mean(all_features, dim=0)
    centered_features = all_features - init_mean.unsqueeze(0)
    init_cov = torch.mm(centered_features.t(), centered_features) / (
        all_features.size(0) - 1
    )
    init_count = all_features.size(0)

    # Clear more GPU memory
    del all_features, centered_features, init_features_list
    torch.cuda.empty_cache()

    # Initialize running statistics for each rank
    for rank in range(1, num_branches + 1):
        rank_results[rank]["running_mean"] = init_mean.clone()
        rank_results[rank]["running_cov"] = init_cov.clone()
        rank_results[rank]["num_accumulated"] = init_count

    print(f"Running statistics initialized with {init_count} samples")

    # Process each rank separately
    with torch.no_grad():
        for rank in range(1, num_branches + 1):
            print(f"\nProcessing rank {rank}...")

            # Number of batches to process for this rank
            num_batches = num_samples // batch_size

            for batch_idx in range(num_batches):
                print(f"Processing batch {batch_idx+1}/{num_batches} for rank {rank}")

                # Create a batch of random noise
                batch_noise = torch.randn(
                    batch_size,
                    sampler.channels,
                    sampler.image_size,
                    sampler.image_size,
                    device=device,
                )

                # Create labels (randomly sampled classes)
                if class_label is None:
                    # Randomly sample classes
                    batch_labels = torch.randint(
                        0, sampler.num_classes, (batch_size,), device=device
                    )
                else:
                    # Use specified class
                    batch_labels = torch.full((batch_size,), class_label, device=device)

                # Start with noise
                x = batch_noise
                t = 0.0

                # Standard flow matching until branch_start_t
                while t < branch_start_t - 1e-6:
                    t_batch = torch.full((batch_size,), t, device=device)
                    velocity = sampler.flow_model(t_batch, x, batch_labels)
                    dt = min(base_dt, branch_start_t - t)
                    x = x + velocity * dt
                    t += dt

                # Now do branching flow matching until t=1.0
                while t < 1.0 - 1e-6:
                    # Check if we can branch (at least one more step remaining)
                    can_branch = t < 1.0 - base_dt

                    if can_branch:
                        # Create num_branches different versions of the entire batch
                        branched_batches = []
                        branched_times = []

                        for branch_idx in range(num_branches):
                            # Clone the current batch
                            branched_batch = x.clone()

                            # Sample different dt for this branch
                            branch_dt = torch.normal(
                                mean=base_dt,
                                std=dt_std * base_dt,
                                size=(1,),
                                device=device,
                            ).item()

                            # Clamp dt to ensure we don't go beyond t=1.0
                            branch_dt = min(branch_dt, 1.0 - t)

                            # Apply flow model to get velocity
                            t_batch = torch.full((batch_size,), t, device=device)
                            velocity = sampler.flow_model(
                                t_batch, branched_batch, batch_labels
                            )

                            # Apply step
                            branched_batch = branched_batch + velocity * branch_dt
                            branch_time = t + branch_dt

                            branched_batches.append(branched_batch)
                            branched_times.append(branch_time)

                        # Compute FID for each branched batch
                        branch_fids = []

                        for branch_idx in range(num_branches):
                            branch_mean, branch_cov, _ = compute_batch_statistics(
                                branched_batches[branch_idx]
                            )
                            branch_fid = compute_fid(
                                branch_mean, branch_cov, ref_mean, ref_cov
                            )
                            branch_fids.append(branch_fid)

                        # Sort branches by FID (ascending, as lower FID is better)
                        sorted_indices = torch.argsort(torch.tensor(branch_fids))
                        sorted_fids = [branch_fids[i] for i in sorted_indices]

                        # Select the branch with the specified rank
                        selected_idx = sorted_indices[rank - 1]
                        selected_fid = branch_fids[selected_idx]

                        # Format all FIDs for printing, highlighting the selected one
                        all_fids_str = (
                            ", ".join(
                                [
                                    f"[{i+1}: {fid:.4f}{'*' if i == rank-1 else ''}"
                                    for i, fid in enumerate(sorted_fids)
                                ]
                            )
                            + "]"
                        )

                        print(
                            f"  Branch point at t={t:.4f}, selected rank {rank} with FID {selected_fid:.4f} (all FIDs: {all_fids_str})"
                        )

                        # Continue with the selected branch
                        x = branched_batches[selected_idx]
                        t = branched_times[selected_idx]
                    else:
                        # Regular step (no branching) for the final step
                        t_batch = torch.full((batch_size,), t, device=device)
                        velocity = sampler.flow_model(t_batch, x, batch_labels)
                        dt = min(base_dt, 1.0 - t)
                        x = x + velocity * dt
                        t += dt

                # Compute batch statistics for final samples
                batch_mean, batch_cov, batch_count = compute_batch_statistics(x)

                # Update running statistics
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
                rank_results[rank]["final_samples"].append(x.cpu())

                print(f"  Batch {batch_idx+1} final FID: {batch_fid:.4f}")

                # Clear GPU memory
                del x
                torch.cuda.empty_cache()

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


def analyze_batch_fid_correlation(
    sampler,
    device,
    num_samples=100,
    num_branches=8,
    dt_std=0.1,
    batch_size=16,
    base_dt=0.1,
    branch_start_t=0.5,
    class_label=None,
):
    """
    Analyze how well batch FID selection correlates with final FID outcomes.

    This function generates trajectories where at each branch point, we create
    num_branches different versions of the entire batch, compute batch FID for each,
    and analyze how well these batch FID scores correlate with the final FID scores.

    Args:
        sampler: The MCTSFlowSampler instance
        device: The device to run computations on
        num_samples: Number of samples to analyze
        num_branches: Number of branches to create at each branch point
        dt_std: Standard deviation for dt variation when branching
        batch_size: Size of batches for batch FID calculation
        base_dt: Base step size for flow matching
        branch_start_t: Time value at which to start branching (0.0-1.0)
        class_label: Specific class to analyze (if None, samples across all classes)

    Returns:
        Dictionary containing correlation metrics and raw data for plotting
    """
    print("\n===== Batch FID Correlation Analysis =====")

    sampler.flow_model.eval()

    # Ensure num_samples is divisible by batch_size
    num_samples = (num_samples // batch_size) * batch_size

    # Data collection for per-branch-point analysis
    branch_point_data = []
    timestep_data = []

    # Import at the top level
    from scipy.stats import spearmanr

    # Sample across all classes or specific class
    if class_label is None:
        # Distribute samples across classes
        samples_per_class = num_samples // sampler.num_classes
        class_range = range(sampler.num_classes)
    else:
        samples_per_class = num_samples
        class_range = [class_label]

    # Function to extract inception features and ensure they're tensors
    def extract_features(images):
        with torch.no_grad():
            features = sampler.extract_inception_features(images)
            # Convert to tensor if it's a numpy array
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).to(device)
            return features

    # Function to compute batch statistics
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
    indices = np.random.choice(len(cifar10), 5000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    # Compute reference statistics in batches
    ref_features_list = []
    batch_size_ref = 100
    for i in range(0, len(real_images), batch_size_ref):
        batch = real_images[i : i + batch_size_ref]
        with torch.no_grad():
            features = extract_features(batch)
            ref_features_list.append(features.cpu())  # Move to CPU to save GPU memory

        # Clear GPU memory
        del batch, features
        torch.cuda.empty_cache()

    # Compute reference statistics
    ref_features = torch.cat([f.to(device) for f in ref_features_list], dim=0)
    ref_mean = torch.mean(ref_features, dim=0)
    centered_features = ref_features - ref_mean.unsqueeze(0)
    ref_cov = torch.mm(centered_features.t(), centered_features) / (
        ref_features.size(0) - 1
    )

    # Clear GPU memory
    del ref_features, centered_features, ref_features_list
    torch.cuda.empty_cache()

    print(f"Reference statistics computed from {real_images.size(0)} real images")

    # Counters for top branch matching
    batch_fid_top_match_count = 0
    total_branch_points = 0

    # Process batches
    num_batches = num_samples // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            print(f"Processing batch {batch_idx+1}/{num_batches}")

            # Create a batch of random noise
            batch_noise = torch.randn(
                batch_size,
                sampler.channels,
                sampler.image_size,
                sampler.image_size,
                device=device,
            )

            # Create labels (randomly sampled classes)
            if class_label is None:
                # Randomly sample classes
                batch_labels = torch.randint(
                    0, sampler.num_classes, (batch_size,), device=device
                )
            else:
                # Use specified class
                batch_labels = torch.full((batch_size,), class_label, device=device)

            # Start with noise
            x = batch_noise
            t = 0.0

            # Standard flow matching until branch_start_t
            while t < branch_start_t - 1e-6:
                t_batch = torch.full((batch_size,), t, device=device)
                velocity = sampler.flow_model(t_batch, x, batch_labels)
                dt = min(base_dt, branch_start_t - t)
                x = x + velocity * dt
                t += dt

            # Now do branching flow matching until t=1.0
            while t < 1.0 - 1e-6:
                # Check if we can branch (at least one more step remaining)
                can_branch = t < 1.0 - base_dt

                if can_branch:
                    # Create num_branches different versions of the entire batch
                    branched_batches = []
                    branched_times = []

                    for branch_idx in range(num_branches):
                        # Clone the current batch
                        branched_batch = x.clone()

                        # Sample different dt for this branch
                        branch_dt = torch.normal(
                            mean=base_dt,
                            std=dt_std * base_dt,
                            size=(1,),
                            device=device,
                        ).item()

                        # Clamp dt to ensure we don't go beyond t=1.0
                        branch_dt = min(branch_dt, 1.0 - t)

                        # Apply flow model to get velocity
                        t_batch = torch.full((batch_size,), t, device=device)
                        velocity = sampler.flow_model(
                            t_batch, branched_batch, batch_labels
                        )

                        # Apply step
                        branched_batch = branched_batch + velocity * branch_dt
                        branch_time = t + branch_dt

                        branched_batches.append(branched_batch)
                        branched_times.append(branch_time)

                    # Compute batch FID for each branched batch
                    batch_fids = []

                    for branch_idx in range(num_branches):
                        branch_mean, branch_cov, _ = compute_batch_statistics(
                            branched_batches[branch_idx]
                        )
                        branch_fid = compute_fid(
                            branch_mean, branch_cov, ref_mean, ref_cov
                        )
                        batch_fids.append(branch_fid)

                    # Now, instead of selecting a specific rank, we'll simulate all branches to completion
                    # to analyze correlation between batch FID and final FID

                    # For each branch, simulate to completion
                    final_batches = []

                    for branch_idx in range(num_branches):
                        branch_x = branched_batches[branch_idx]
                        branch_t = branched_times[branch_idx]

                        # Continue simulation until t=1.0
                        while branch_t < 1.0 - 1e-6:
                            branch_t_batch = torch.full(
                                (batch_size,), branch_t, device=device
                            )
                            branch_velocity = sampler.flow_model(
                                branch_t_batch, branch_x, batch_labels
                            )
                            branch_dt = min(base_dt, 1.0 - branch_t)
                            branch_x = branch_x + branch_velocity * branch_dt
                            branch_t += branch_dt

                        final_batches.append(branch_x)

                    # Compute final FID for each completed branch
                    final_fids = []

                    for branch_idx in range(num_branches):
                        final_mean, final_cov, _ = compute_batch_statistics(
                            final_batches[branch_idx]
                        )
                        final_fid = compute_fid(
                            final_mean, final_cov, ref_mean, ref_cov
                        )
                        final_fids.append(final_fid)

                    # Calculate correlation between batch FID and final FID
                    if len(batch_fids) > 1:
                        batch_final_corr, batch_final_p = spearmanr(
                            batch_fids, final_fids
                        )
                    else:
                        batch_final_corr, batch_final_p = float("nan"), float("nan")

                    # Check if top branch matches
                    if len(batch_fids) > 1 and len(final_fids) > 1:
                        total_branch_points += 1

                        # For FID, lower is better, so we need to find the minimum
                        batch_best_idx = np.argmin(batch_fids)
                        final_best_idx = np.argmin(final_fids)

                        if batch_best_idx == final_best_idx:
                            batch_fid_top_match_count += 1

                    # Store data for this branch point
                    branch_data = {
                        "time": t,
                        "batch_fids": np.array(batch_fids),
                        "final_fids": np.array(final_fids),
                        "batch_final_corr": batch_final_corr,
                        "batch_final_p": batch_final_p,
                    }

                    branch_point_data.append(branch_data)

                    # Find the timestep index for this branch point
                    timestep_idx = int(t / base_dt + 0.5)  # Round to nearest timestep

                    # Ensure timestep_data has enough elements
                    while len(timestep_data) <= timestep_idx:
                        timestep_data.append([])

                    timestep_data[timestep_idx].append(branch_data)

                    # Format all FIDs for printing
                    batch_fids_str = ", ".join([f"{fid:.4f}" for fid in batch_fids])
                    final_fids_str = ", ".join([f"{fid:.4f}" for fid in final_fids])

                    print(
                        f"  Branch point at t={t:.4f}, correlation={batch_final_corr:.4f}"
                    )
                    print(f"    Batch FIDs: [{batch_fids_str}]")
                    print(f"    Final FIDs: [{final_fids_str}]")

                    # Continue with the original batch (no selection)
                    # This is just to move forward to the next branch point
                    t_batch = torch.full((batch_size,), t, device=device)
                    velocity = sampler.flow_model(t_batch, x, batch_labels)
                    dt = min(base_dt, 1.0 - t)
                    x = x + velocity * dt
                    t += dt
                else:
                    # Regular step (no branching) for the final step
                    t_batch = torch.full((batch_size,), t, device=device)
                    velocity = sampler.flow_model(t_batch, x, batch_labels)
                    dt = min(base_dt, 1.0 - t)
                    x = x + velocity * dt
                    t += dt

    # Calculate overall correlation statistics
    batch_final_correlations = [
        data["batch_final_corr"]
        for data in branch_point_data
        if not np.isnan(data["batch_final_corr"])
    ]

    avg_batch_final_corr = (
        np.mean(batch_final_correlations) if batch_final_correlations else float("nan")
    )

    # Calculate top branch match percentage
    batch_fid_top_match_pct = (
        (batch_fid_top_match_count / total_branch_points * 100)
        if total_branch_points > 0
        else 0
    )

    # Print results
    print("\n===== Batch FID Correlation Results =====")
    print(f"Average batch-final FID correlation: {avg_batch_final_corr:.4f}")
    print(
        f"Batch FID top branch match percentage: {batch_fid_top_match_pct:.2f}% ({batch_fid_top_match_count}/{total_branch_points})"
    )

    # Calculate per-timestep correlations
    print("\n===== Per-Timestep Correlation Analysis =====")
    timestep_correlations = []
    timestep_match_counts = []
    timestep_branch_counts = []

    for step, step_data in enumerate(timestep_data):
        if not step_data:
            timestep_correlations.append(float("nan"))
            timestep_match_counts.append(0)
            timestep_branch_counts.append(0)
            continue

        step_correlations = [
            data["batch_final_corr"]
            for data in step_data
            if not np.isnan(data["batch_final_corr"])
        ]

        if step_correlations:
            avg_step_corr = np.mean(step_correlations)
            timestep_correlations.append(avg_step_corr)

            # Count matches for this timestep
            step_match_count = 0
            step_branch_count = 0

            for data in step_data:
                if len(data["batch_fids"]) > 1 and len(data["final_fids"]) > 1:
                    step_branch_count += 1
                    batch_best_idx = np.argmin(data["batch_fids"])
                    final_best_idx = np.argmin(data["final_fids"])

                    if batch_best_idx == final_best_idx:
                        step_match_count += 1

            timestep_match_counts.append(step_match_count)
            timestep_branch_counts.append(step_branch_count)

            # Calculate match percentage for this timestep
            match_pct = (
                (step_match_count / step_branch_count * 100)
                if step_branch_count > 0
                else 0
            )

            print(
                f"  Timestep {step}: correlation={avg_step_corr:.4f}, top match={match_pct:.1f}% ({step_match_count}/{step_branch_count})"
            )
        else:
            timestep_correlations.append(float("nan"))
            timestep_match_counts.append(0)
            timestep_branch_counts.append(0)
            print(f"  Timestep {step}: insufficient data")

    # Plot correlation by timestep
    plt.figure(figsize=(10, 6))
    valid_steps = [
        i for i, corr in enumerate(timestep_correlations) if not np.isnan(corr)
    ]
    valid_corrs = [timestep_correlations[i] for i in valid_steps]

    if valid_steps:
        plt.plot(valid_steps, valid_corrs, "o-")
        plt.xlabel("Timestep")
        plt.ylabel("Batch-Final FID Correlation")
        plt.title("Batch FID Correlation with Final FID by Timestep")
        plt.grid(True)
        plt.savefig("batch_fid_correlation_by_timestep.png")
    plt.close()

    # Plot match percentage by timestep
    plt.figure(figsize=(10, 6))
    valid_steps = [i for i, count in enumerate(timestep_branch_counts) if count > 0]
    valid_match_pcts = [
        timestep_match_counts[i] / timestep_branch_counts[i] * 100 for i in valid_steps
    ]

    if valid_steps:
        plt.plot(valid_steps, valid_match_pcts, "o-")
        plt.xlabel("Timestep")
        plt.ylabel("Top Branch Match Percentage (%)")
        plt.title("Batch FID Top Branch Match Percentage by Timestep")
        plt.grid(True)
        plt.savefig("batch_fid_match_percentage_by_timestep.png")
    plt.close()

    # Return results
    return {
        "overall_batch_final_correlation": avg_batch_final_corr,
        "batch_fid_top_match_percentage": batch_fid_top_match_pct,
        "timestep_correlations": timestep_correlations,
        "timestep_match_counts": timestep_match_counts,
        "timestep_branch_counts": timestep_branch_counts,
        "branch_point_data": branch_point_data,
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

    # Run batch FID correlation analysis
    batch_fid_correlation_results = analyze_batch_fid_correlation(
        sampler=sampler,
        device=device,
        num_samples=128,
        num_branches=8,
        dt_std=0.25,
        batch_size=64,
        base_dt=0.1,
        branch_start_t=0.5,
    )

    # # Run batch FID rank consistency analysis
    # batch_fid_analysis_results = analyze_batch_fid_rank_consistency(
    #     sampler=sampler,
    #     device=device,
    #     num_samples=256,
    #     num_branches=4,
    #     dt_std=0.25,
    #     batch_size=16,
    #     base_dt=0.1,
    #     branch_start_t=0.5,
    # )


if __name__ == "__main__":
    main()
