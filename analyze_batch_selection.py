import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from scipy.linalg import sqrtm

from mcts_single_flow import MCTSFlowSampler


def analyze_batch_selection_methods(
    sampler,
    device,
    num_samples=256,
    num_branches=8,
    dt_std=0.1,
    batch_size=32,
    base_dt=0.1,
    branch_start_t=0.5,  # When to start branching (0.0 = start immediately, 0.5 = start halfway)
    class_label=None,
    selection_methods=[
        "batch_fid",
        "value_model",
        "lookahead",
        "mahalanobis",
        "lookahead_mahalanobis",
    ],
):
    """
    Analyze different batch selection methods and compare their effectiveness.

    This function generates trajectories where at each branching point, we create
    num_branches different versions of the entire batch, then select the best branch
    according to different selection methods, and evaluate the final FID.

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
        selection_methods: List of methods to use for branch selection
            - "batch_fid": Select based on batch FID with reference statistics
            - "value_model": Select based on value model predictions
            - "lookahead": Select based on lookahead FID (extrapolate to t=1)
            - "mahalanobis": Select based on Mahalanobis distance
            - "lookahead_mahalanobis": Select based on lookahead Mahalanobis distance

    Returns:
        Dictionary containing FID scores and other metrics for each selection method
    """
    print("\n===== Batch Selection Methods Analysis =====")

    sampler.flow_model.eval()
    if hasattr(sampler, "value_model") and sampler.value_model is not None:
        sampler.value_model.eval()
    else:
        if "value_model" in selection_methods:
            print("Warning: Value model not available, removing from selection methods")
            selection_methods.remove("value_model")

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

    # Store results for each selection method
    method_results = {
        method: {
            "final_samples": [],  # Store final samples for real FID calculation
            "running_mean": None,
            "running_cov": None,
            "num_accumulated": 0,
            "branch_fids": [],  # Store FID at each branch point
            "final_fids": [],  # Store final FID for each batch
        }
        for method in selection_methods
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
        cov1_np = cov1.cpu().numpy()
        cov2_np = cov2.cpu().numpy()

        # Calculate sqrt(cov1 * cov2)
        covmean = sqrtm(np.matmul(cov1_np, cov2_np))

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

    # Initialize running statistics with 1024 samples for each method
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

    # Initialize running statistics for each method
    for method in selection_methods:
        method_results[method]["running_mean"] = init_mean.clone()
        method_results[method]["running_cov"] = init_cov.clone()
        method_results[method]["num_accumulated"] = init_count

    print(f"Running statistics initialized with {init_count} samples")

    # Process batches
    num_batches = num_samples // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            print(f"\nProcessing batch {batch_idx+1}/{num_batches}")

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

            # Create separate trajectories for each selection method
            method_trajectories = {}

            for method in selection_methods:
                # Start with noise
                method_trajectories[method] = batch_noise.clone()

            # Standard flow matching until branch_start_t (same for all methods)
            t = 0.0
            while t < branch_start_t - 1e-6:
                t_batch = torch.full((batch_size,), t, device=device)

                # Apply the same update to all method trajectories
                for method in selection_methods:
                    velocity = sampler.flow_model(
                        t_batch, method_trajectories[method], batch_labels
                    )
                    dt = min(base_dt, branch_start_t - t)
                    method_trajectories[method] = (
                        method_trajectories[method] + velocity * dt
                    )

                t += dt

            # Now do branching flow matching until t=1.0
            while t < 1.0 - 1e-6:
                # Check if we can branch (at least one more step remaining)
                can_branch = t < 1.0 - base_dt

                if can_branch:
                    # For each method, create branches and select the best one
                    for method in selection_methods:
                        # Create num_branches different versions of the entire batch
                        branched_batches = []
                        branched_times = []

                        for branch_idx in range(num_branches):
                            # Clone the current batch
                            branched_batch = method_trajectories[method].clone()

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

                        # Evaluate branches according to the selection method
                        if method == "batch_fid":
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

                            # Select branch with lowest FID (best quality)
                            selected_idx = np.argmin(branch_fids)
                            selected_score = branch_fids[selected_idx]
                            method_results[method]["branch_fids"].append(selected_score)

                        elif method == "value_model":
                            # Use value model to predict quality
                            branch_values = []

                            for branch_idx in range(num_branches):
                                t_batch = torch.full(
                                    (batch_size,),
                                    branched_times[branch_idx],
                                    device=device,
                                )
                                batch_values = sampler.value_model(
                                    t_batch, branched_batches[branch_idx], batch_labels
                                )
                                # Average the values across the batch
                                branch_value = torch.mean(batch_values).item()
                                branch_values.append(branch_value)

                            # Select branch with highest value (best predicted quality)
                            selected_idx = np.argmax(branch_values)
                            selected_score = branch_values[selected_idx]
                            method_results[method]["branch_fids"].append(selected_score)

                        elif method == "lookahead":
                            # Extrapolate to t=1.0 and compute FID
                            branch_lookahead_fids = []

                            for branch_idx in range(num_branches):
                                # Get current state and time
                                current_batch = branched_batches[branch_idx]
                                current_time = branched_times[branch_idx]

                                # Compute velocity at current state
                                t_batch = torch.full(
                                    (batch_size,), current_time, device=device
                                )
                                velocity = sampler.flow_model(
                                    t_batch, current_batch, batch_labels
                                )

                                # Extrapolate to t=1.0
                                dt_to_end = 1.0 - current_time
                                lookahead_batch = current_batch + velocity * dt_to_end

                                # Compute FID for lookahead batch
                                lookahead_mean, lookahead_cov, _ = (
                                    compute_batch_statistics(lookahead_batch)
                                )
                                lookahead_fid = compute_fid(
                                    lookahead_mean, lookahead_cov, ref_mean, ref_cov
                                )
                                branch_lookahead_fids.append(lookahead_fid)

                            # Select branch with lowest lookahead FID
                            selected_idx = np.argmin(branch_lookahead_fids)
                            selected_score = branch_lookahead_fids[selected_idx]
                            method_results[method]["branch_fids"].append(selected_score)

                        elif method == "mahalanobis":
                            # Compute Mahalanobis distance for each branch
                            branch_mahalanobis = []

                            for branch_idx in range(num_branches):
                                # Use sampler's batch_compute_mean_difference method
                                mahalanobis_scores = (
                                    sampler.batch_compute_mean_difference(
                                        branched_batches[branch_idx], batch_labels
                                    )
                                )
                                # Average across batch
                                avg_mahalanobis = torch.mean(mahalanobis_scores).item()
                                branch_mahalanobis.append(avg_mahalanobis)

                            # Select branch with highest Mahalanobis score (best quality)
                            selected_idx = np.argmax(branch_mahalanobis)
                            selected_score = branch_mahalanobis[selected_idx]
                            method_results[method]["branch_fids"].append(selected_score)

                        elif method == "lookahead_mahalanobis":
                            # Extrapolate to t=1.0 and compute Mahalanobis distance
                            branch_lookahead_mahalanobis = []

                            for branch_idx in range(num_branches):
                                # Get current state and time
                                current_batch = branched_batches[branch_idx]
                                current_time = branched_times[branch_idx]

                                # Compute velocity at current state
                                t_batch = torch.full(
                                    (batch_size,), current_time, device=device
                                )
                                velocity = sampler.flow_model(
                                    t_batch, current_batch, batch_labels
                                )

                                # Extrapolate to t=1.0
                                dt_to_end = 1.0 - current_time
                                lookahead_batch = current_batch + velocity * dt_to_end

                                # Compute Mahalanobis distance for lookahead batch
                                mahalanobis_scores = (
                                    sampler.batch_compute_mean_difference(
                                        lookahead_batch, batch_labels
                                    )
                                )
                                # Average across batch
                                avg_mahalanobis = torch.mean(mahalanobis_scores).item()
                                branch_lookahead_mahalanobis.append(avg_mahalanobis)

                            # Select branch with highest lookahead Mahalanobis score
                            selected_idx = np.argmax(branch_lookahead_mahalanobis)
                            selected_score = branch_lookahead_mahalanobis[selected_idx]
                            method_results[method]["branch_fids"].append(selected_score)

                        # Continue with the selected branch
                        method_trajectories[method] = branched_batches[selected_idx]
                        t_next = branched_times[selected_idx]

                        # Print selection results
                        print(
                            f"  Method {method}: selected branch {selected_idx+1}/{num_branches} with score {selected_score:.4f}"
                        )

                    # Update t to the minimum of all method's next times
                    # This ensures all methods progress together
                    t = min([t_next for method in selection_methods])

                else:
                    # Regular step (no branching) for the final step
                    t_batch = torch.full((batch_size,), t, device=device)

                    for method in selection_methods:
                        velocity = sampler.flow_model(
                            t_batch, method_trajectories[method], batch_labels
                        )
                        dt = min(base_dt, 1.0 - t)
                        method_trajectories[method] = (
                            method_trajectories[method] + velocity * dt
                        )

                    t += dt

            # Compute final FID for each method
            for method in selection_methods:
                # Compute batch statistics for final samples
                batch_mean, batch_cov, batch_count = compute_batch_statistics(
                    method_trajectories[method]
                )

                # Update running statistics
                (
                    method_results[method]["running_mean"],
                    method_results[method]["running_cov"],
                    method_results[method]["num_accumulated"],
                ) = update_running_statistics(
                    method_results[method]["running_mean"],
                    method_results[method]["running_cov"],
                    method_results[method]["num_accumulated"],
                    batch_mean,
                    batch_cov,
                    batch_count,
                )

                # Compute batch FID with reference
                batch_fid = compute_fid(batch_mean, batch_cov, ref_mean, ref_cov)
                method_results[method]["final_fids"].append(batch_fid)

                # Store final samples for later analysis
                method_results[method]["final_samples"].append(
                    method_trajectories[method].cpu()
                )

                print(f"  Method {method} final batch FID: {batch_fid:.4f}")

                # Clear GPU memory
                del method_trajectories[method]
                torch.cuda.empty_cache()

    # Calculate overall FID for each method using running statistics
    method_overall_fid = {}
    method_avg_batch_fid = {}
    method_std_batch_fid = {}

    for method in selection_methods:
        # Calculate overall FID using running statistics
        overall_fid = compute_fid(
            method_results[method]["running_mean"],
            method_results[method]["running_cov"],
            ref_mean,
            ref_cov,
        )
        method_overall_fid[method] = overall_fid

        # Calculate average and std of batch FIDs
        batch_fids = method_results[method]["final_fids"]
        method_avg_batch_fid[method] = np.mean(batch_fids)
        method_std_batch_fid[method] = np.std(batch_fids)

    # Print results
    print("\n===== Batch Selection Methods Results =====")
    print("\nOverall FID by method:")
    for method in selection_methods:
        print(f"  {method}: {method_overall_fid[method]:.4f}")

    print("\nAverage batch FID by method:")
    for method in selection_methods:
        print(
            f"  {method}: {method_avg_batch_fid[method]:.4f} ± {method_std_batch_fid[method]:.4f}"
        )

    # Plot results for overall FID
    plt.figure(figsize=(12, 6))
    plt.bar(selection_methods, [method_overall_fid[m] for m in selection_methods])
    plt.xlabel("Selection Method")
    plt.ylabel("Overall FID Score (lower is better)")
    plt.title("Overall FID by Selection Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("batch_selection_overall_fid.png")
    plt.close()

    # Plot results for average batch FID
    plt.figure(figsize=(12, 6))
    plt.bar(
        selection_methods,
        [method_avg_batch_fid[m] for m in selection_methods],
        yerr=[method_std_batch_fid[m] for m in selection_methods],
        capsize=5,
    )
    plt.xlabel("Selection Method")
    plt.ylabel("Average Batch FID Score (lower is better)")
    plt.title("Average Batch FID by Selection Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("batch_selection_avg_fid.png")
    plt.close()

    # Plot batch FIDs over time for each method
    plt.figure(figsize=(12, 8))
    for method in selection_methods:
        plt.plot(method_results[method]["final_fids"], label=method)
    plt.xlabel("Batch Index")
    plt.ylabel("Batch FID Score (lower is better)")
    plt.title("Batch FID Progression by Selection Method")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("batch_selection_fid_progression.png")
    plt.close()

    # Return results
    return {
        "method_overall_fid": method_overall_fid,
        "method_avg_batch_fid": method_avg_batch_fid,
        "method_std_batch_fid": method_std_batch_fid,
        "method_results": method_results,
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
        buffer_size=10,
        flow_model="large_flow_model.pt",
        value_model="value_model.pt",  # Needed for value_model selection method
        num_channels=256,
        inception_layer=0,
        pca_dim=None,
    )

    # Run batch selection methods analysis
    selection_methods = [
        "batch_fid",
        "value_model",
        "lookahead",
        "mahalanobis",
        "lookahead_mahalanobis",
    ]

    batch_selection_results = analyze_batch_selection_methods(
        sampler=sampler,
        device=device,
        num_samples=256,
        num_branches=4,
        dt_std=0.25,
        batch_size=16,
        base_dt=0.1,
        branch_start_t=0.5,
        selection_methods=["batch_fid"],
    )


if __name__ == "__main__":
    main()
