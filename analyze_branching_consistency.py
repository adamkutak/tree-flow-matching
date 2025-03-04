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


def analyze_fid_rank_consistency(
    sampler,
    device,
    num_samples=10,
    num_branches=8,
    dt_std=0.1,
    class_label=None,
):
    """
    Analyze how consistently selecting the nth-ranked branch by FID affects final FID.

    This function generates trajectories where at each possible branching point, we consistently
    select the branch ranked n (where n=1 is best, n=2 is second-best, etc.) according
    to the intermediate FID calculation. We then measure the final FID of these trajectories.

    Args:
        sampler: The MCTSFlowSampler instance
        device: The device to run computations on
        num_samples: Number of samples to analyze per class
        num_branches: Number of branches to create at each branch point (also determines max rank)
        dt_std: Standard deviation for dt variation when branching
        class_label: Specific class to analyze (if None, samples across all classes)

    Returns:
        Dictionary containing correlation metrics and raw data for plotting
    """
    print("\n===== FID Rank Consistency Analysis =====")

    sampler.flow_model.eval()

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
            "fid_scores": [],
            "trajectories": [],
            "final_samples": [],  # Store final samples for real FID calculation
        }
        for rank in range(1, num_branches + 1)
    }

    with torch.no_grad():
        for class_idx in class_range:
            print(f"Processing class {class_idx}...")

            for sample_idx in tqdm(
                range(samples_per_class), desc=f"Class {class_idx} samples"
            ):
                # For each sample, we'll create num_branches different trajectories,
                # each consistently selecting a different rank

                # Start with the same random noise for all rank trajectories
                initial_noise = torch.randn(
                    1,
                    sampler.channels,
                    sampler.image_size,
                    sampler.image_size,
                    device=device,
                )

                # Create a trajectory for each rank
                for rank in range(1, num_branches + 1):
                    # Start with the initial noise
                    x = initial_noise.clone()
                    label = torch.full((1,), class_idx, device=device)

                    # Track the trajectory for visualization
                    trajectory = [x.cpu().numpy()]

                    # Current time
                    t = 0.0

                    # Simulate the trajectory with branching at every possible step
                    step = 0
                    while t < 1.0 - 1e-6:
                        # We can branch if we have at least 2 more steps remaining
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
                            aligned_times = torch.full(
                                (num_branches,), next_timestep, device=device
                            )

                            # Calculate intermediate FID for all aligned branches
                            intermediate_fid_changes = sampler.batch_compute_fid_change(
                                aligned_samples, branch_labels
                            )

                            # Sort branches by FID (descending, as higher FID is better in this context)
                            sorted_indices = torch.argsort(
                                intermediate_fid_changes, descending=True
                            )

                            # Select the branch with the specified rank (rank 1 = best, rank num_branches = worst)
                            selected_idx = sorted_indices[rank - 1]
                            x = aligned_samples[selected_idx : selected_idx + 1]

                            # Update time
                            t = next_timestep
                            step += 2  # We advanced 2 steps

                        else:
                            # Regular step (no branching) for the final steps
                            t_batch = torch.full((1,), t, device=device)
                            velocity = sampler.flow_model(t_batch, x, label)
                            x = x + velocity * base_dt
                            t += base_dt
                            step += 1

                        # Add to trajectory
                        trajectory.append(x.cpu().numpy())

                    # Calculate FID for the final sample
                    fid_score = sampler.batch_compute_fid_change(x, label).item()

                    # Store results
                    rank_results[rank]["fid_scores"].append(fid_score)
                    rank_results[rank]["trajectories"].append(trajectory)
                    rank_results[rank]["final_samples"].append(
                        x.cpu()
                    )  # Store for real FID calculation

    # Calculate statistics for intermediate FID scores
    rank_avg_fid = {}
    rank_std_fid = {}

    for rank in range(1, num_branches + 1):
        fid_scores = rank_results[rank]["fid_scores"]
        rank_avg_fid[rank] = np.mean(fid_scores)
        rank_std_fid[rank] = np.std(fid_scores)

    # Calculate correlation between rank and average intermediate FID
    ranks = list(range(1, num_branches + 1))
    avg_fids = [rank_avg_fid[r] for r in ranks]

    if len(ranks) > 1:
        rank_fid_correlation, p_value = spearmanr(ranks, avg_fids)
    else:
        rank_fid_correlation, p_value = float("nan"), float("nan")

    # Print results for intermediate FID
    print("\n===== Intermediate FID Rank Consistency Results =====")
    print(
        f"Correlation between rank and average intermediate FID: {rank_fid_correlation:.4f} (p-value: {p_value:.4f})"
    )
    print("\nAverage intermediate FID by rank:")
    for rank in range(1, num_branches + 1):
        print(f"  Rank {rank}: {rank_avg_fid[rank]:.4f} ± {rank_std_fid[rank]:.4f}")

    # Plot results for intermediate FID
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        ranks, avg_fids, yerr=[rank_std_fid[r] for r in ranks], fmt="o-", capsize=5
    )
    plt.xlabel("Branch Rank by Intermediate FID (1 = best)")
    plt.ylabel("Average Intermediate FID Score")
    plt.title("Intermediate FID Score by Consistently Selected Branch Rank")
    plt.grid(True)
    plt.savefig("intermediate_fid_rank_consistency.png")
    plt.close()

    # Now calculate real FID scores for each rank
    print("\n===== Computing Real FID Scores =====")

    # Setup CIFAR-10 dataset with appropriate transforms for real FID calculation
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
        normalize=True, reset_real_features=False
    ).to(device)

    # Process real images
    real_batch_size = 100
    indices = np.random.choice(len(cifar10), 5000, replace=False)
    real_images = torch.stack([cifar10[i][0] for i in indices]).to(device)

    print("Processing real images...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid_metric.update(batch, real=True)
        torch.cuda.empty_cache()

    # Calculate real FID for each rank
    real_fid_scores = {}

    for rank in range(1, num_branches + 1):
        # Reset FID for fake images
        fid_metric.reset_fake_features()

        # Stack all final samples for this rank
        final_samples = torch.cat(rank_results[rank]["final_samples"], dim=0)

        # Process in batches
        batch_size = 64
        print(f"Processing rank {rank} samples for real FID...")
        for i in range(0, len(final_samples), batch_size):
            batch = final_samples[i : i + batch_size].to(device)
            fid_metric.update(batch, real=False)
            torch.cuda.empty_cache()

        # Compute FID score
        real_fid_score = fid_metric.compute().item()
        real_fid_scores[rank] = real_fid_score
        print(f"  Rank {rank} real FID: {real_fid_score:.4f}")

    # Calculate correlation between rank and real FID
    real_fid_values = [real_fid_scores[r] for r in ranks]

    if len(ranks) > 1:
        real_fid_correlation, real_p_value = spearmanr(ranks, real_fid_values)
    else:
        real_fid_correlation, real_p_value = float("nan"), float("nan")

    print("\n===== Real FID Rank Correlation =====")
    print(
        f"Correlation between rank and real FID: {real_fid_correlation:.4f} (p-value: {real_p_value:.4f})"
    )

    # Plot real FID results
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, real_fid_values, "o-")
    plt.xlabel("Branch Rank by Intermediate FID (1 = best)")
    plt.ylabel("Real FID Score (lower is better)")
    plt.title("Real FID Score by Consistently Selected Branch Rank")
    plt.grid(True)
    plt.savefig("real_fid_rank_consistency.png")
    plt.close()

    # Return results
    return {
        "intermediate_fid_correlation": rank_fid_correlation,
        "intermediate_fid_p_value": p_value,
        "rank_avg_intermediate_fid": rank_avg_fid,
        "rank_std_intermediate_fid": rank_std_fid,
        "real_fid_scores": real_fid_scores,
        "real_fid_correlation": real_fid_correlation,
        "real_fid_p_value": real_p_value,
        "rank_results": rank_results,
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
        value_model="value_model.pt",  # Still needed for sampler initialization
        num_channels=256,
        inception_layer=0,
    )

    # Run FID rank consistency analysis
    analysis_results = analyze_fid_rank_consistency(
        sampler=sampler,
        device=device,
        num_samples=100,  # Samples per class
        num_branches=8,
        dt_std=0.1,
    )


if __name__ == "__main__":
    main()
