import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchmetrics.image.fid as FID
from scipy.linalg import sqrtm

from mcts_single_flow import MCTSFlowSampler


def analyze_value_model_predictions(
    sampler,
    device,
    num_samples=100,
    num_branches=8,
    dt_std=0.1,
    class_label=None,
    batch_size=10,
):
    """
    Analyze how well the value model predicts actual FID differences.

    Args:
        sampler: The MCTSFlowSampler instance
        device: The device to run computations on
        num_samples: Number of samples to analyze
        num_branches: Number of branches to create at each test point
        dt_std: Standard deviation for dt variation when branching
        class_label: Specific class to analyze (if None, samples across all classes)
        batch_size: Number of samples to process in parallel

    Returns:
        Dictionary containing correlation metrics and raw data for plotting
    """
    print("\n===== Value Model Prediction Analysis =====")

    sampler.flow_model.eval()
    sampler.value_model.eval()

    # Data collection for per-branch-point analysis
    branch_point_data = []
    timestep_data = [[] for _ in range(sampler.num_timesteps)]

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

    # Print timestep schedule only once
    print(f"Timestep schedule: {sampler.timesteps.cpu().numpy()}")

    # Counters for top branch matching
    value_top_match_count = 0
    intermediate_top_match_count = 0
    lookahead_top_match_count = 0
    total_branch_points = 0

    with torch.no_grad():
        for class_idx in class_range:
            print(f"Processing class {class_idx}...")

            # Process in batches
            num_batches = (
                samples_per_class + batch_size - 1
            ) // batch_size  # Ceiling division
            for batch_idx in tqdm(
                range(num_batches), desc=f"Class {class_idx} batches"
            ):
                # Determine actual batch size (might be smaller for last batch)
                actual_batch_size = min(
                    batch_size, samples_per_class - batch_idx * batch_size
                )
                if actual_batch_size <= 0:
                    break

                # Start with random noise
                x = torch.randn(
                    actual_batch_size,
                    sampler.channels,
                    sampler.image_size,
                    sampler.image_size,
                    device=device,
                )
                label = torch.full((actual_batch_size,), class_idx, device=device)

                # Choose random timesteps to branch at (between 0 and num_timesteps-3)
                # We need at least 2 more steps after branching
                branch_steps = torch.randint(
                    0, sampler.num_timesteps - 2, (actual_batch_size,)
                ).to(device)

                # Track current time for each sample
                current_times = torch.zeros(actual_batch_size, device=device)

                # Simulate each sample up to its branch point
                for step in range(sampler.num_timesteps - 1):
                    # Find which samples need to be updated at this step
                    active_mask = branch_steps >= step
                    if not active_mask.any():
                        break

                    active_x = x[active_mask]
                    active_label = label[active_mask]

                    t = sampler.timesteps[step]
                    dt = sampler.timesteps[step + 1] - t
                    t_batch = torch.full((active_mask.sum(),), t.item(), device=device)

                    velocity = sampler.flow_model(t_batch, active_x, active_label)
                    active_x = active_x + velocity * dt

                    # Update the samples and times
                    x[active_mask] = active_x
                    current_times[active_mask] = sampler.timesteps[step + 1].item()

                # Process each sample's branches separately to maintain branch point grouping
                for i in range(actual_batch_size):
                    branch_step = branch_steps[i].item()

                    # Create branches for this sample
                    branches = x[i : i + 1].repeat(num_branches, 1, 1, 1)
                    branch_labels = label[i : i + 1].repeat(num_branches)
                    branch_times = torch.full(
                        (num_branches,), current_times[i].item(), device=device
                    )

                    # Sample different dt values for each branch
                    base_dt = 1.0 / sampler.num_timesteps
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

                    # Get velocity for all branches in a single batched calculation
                    velocity = sampler.flow_model(branch_times, branches, branch_labels)

                    # Apply different step sizes to create branches
                    branched_samples = branches + velocity * dts.view(-1, 1, 1, 1)
                    new_times = branch_times + dts

                    # Now simulate all branches forward one more step to a common time point
                    # Use the next timestep in the schedule after the branch point
                    next_timestep = sampler.timesteps[branch_step + 1].item()

                    # Calculate dt to reach the next common timestep for each branch
                    dt_to_next = next_timestep - new_times

                    # Get velocity for all branches in a single batched calculation
                    velocity = sampler.flow_model(
                        new_times, branched_samples, branch_labels
                    )

                    # Apply the step to all branches at once
                    aligned_samples = branched_samples + velocity * dt_to_next.view(
                        -1, 1, 1, 1
                    )
                    aligned_times = torch.full(
                        (num_branches,), next_timestep, device=device
                    )

                    # Verify all branches are at the same time
                    time_diff = torch.max(torch.abs(aligned_times - next_timestep))
                    if time_diff > 1e-8:
                        print(
                            f"WARNING: Branches not at same time. Max difference: {time_diff.item():.8f}"
                        )

                    # Create look-ahead samples by extrapolating linearly to t=1
                    dt_to_end = 1.0 - aligned_times
                    lookahead_samples = aligned_samples + velocity * dt_to_end.view(
                        -1, 1, 1, 1
                    )

                    # Get value model predictions for all aligned branches
                    value_preds = sampler.value_model(
                        aligned_times, aligned_samples, branch_labels
                    )

                    # Calculate FID for intermediate aligned samples
                    intermediate_fid_changes = []
                    for j in range(num_branches):
                        intermediate_fid = sampler.compute_fid_change(
                            aligned_samples[j : j + 1], class_idx
                        )
                        intermediate_fid_changes.append(intermediate_fid)

                    lookahead_times = torch.full((num_branches,), 1.0, device=device)
                    time_diffs = torch.abs(lookahead_times - 1.0)
                    if torch.any(time_diffs > 1e-8):
                        print(
                            f"WARNING: Look-ahead samples not all at t=1. Max difference: {torch.max(time_diffs).item():.6f}"
                        )

                    # Calculate FID for look-ahead samples
                    lookahead_fid_changes = []
                    for j in range(num_branches):
                        lookahead_fid = -1 * sampler.compute_fid_change(
                            lookahead_samples[j : j + 1], class_idx
                        )
                        lookahead_fid_changes.append(lookahead_fid)

                    # Simulate all branches to completion with batched operations
                    current_samples = aligned_samples
                    current_time = next_timestep  # All samples are at the same time now
                    next_step = branch_step + 1  # We know this is the correct index

                    # Simulate until completion with batched operations
                    for step in range(next_step + 1, sampler.num_timesteps - 1):
                        t = sampler.timesteps[step]
                        dt = sampler.timesteps[step + 1] - t
                        t_batch = torch.full((num_branches,), t.item(), device=device)

                        velocity = sampler.flow_model(
                            t_batch, current_samples, branch_labels
                        )
                        current_samples = current_samples + velocity * dt

                    # Final samples are now all in current_samples
                    final_samples = current_samples

                    # Verify all branches reached t=1
                    final_time = sampler.timesteps[-1].item()
                    if (
                        abs(final_time - 1.0) > 1e-5
                    ):  # Check if time is not approximately 1.0
                        print(
                            f"WARNING: Branches ended at time {final_time:.4f}, not t=1.0"
                        )

                    # Calculate actual FID change for each final sample
                    actual_fid_changes = []
                    for j in range(num_branches):
                        fid_change = sampler.compute_fid_change(
                            final_samples[j : j + 1], class_idx
                        )
                        actual_fid_changes.append(fid_change)

                    # Calculate correlation between intermediate FID and final FID
                    if len(intermediate_fid_changes) > 1:
                        intermediate_final_corr, _ = spearmanr(
                            intermediate_fid_changes, actual_fid_changes
                        )
                    else:
                        intermediate_final_corr = float("nan")

                    # Calculate correlation between look-ahead FID and final FID
                    if len(lookahead_fid_changes) > 1:
                        lookahead_final_corr, _ = spearmanr(
                            lookahead_fid_changes, actual_fid_changes
                        )
                    else:
                        lookahead_final_corr = float("nan")

                    # Check if top branch matches
                    if len(value_preds) > 1 and len(actual_fid_changes) > 1:
                        total_branch_points += 1

                        # For value model predictions (lower is better for FID)
                        value_best_idx = torch.argmin(value_preds).item()
                        final_best_idx = np.argmin(actual_fid_changes)
                        if value_best_idx == final_best_idx:
                            value_top_match_count += 1

                        # For intermediate FID (lower is better for FID)
                        intermediate_best_idx = np.argmin(intermediate_fid_changes)
                        if intermediate_best_idx == final_best_idx:
                            intermediate_top_match_count += 1

                        # For look-ahead FID (lower is better for FID)
                        lookahead_best_idx = np.argmin(lookahead_fid_changes)
                        if lookahead_best_idx == final_best_idx:
                            lookahead_top_match_count += 1

                    # Store data for this branch point
                    branch_data = {
                        "value_predictions": value_preds.cpu().numpy(),
                        "intermediate_fid_changes": np.array(intermediate_fid_changes),
                        "lookahead_fid_changes": np.array(lookahead_fid_changes),
                        "actual_fid_changes": np.array(actual_fid_changes),
                        "timestep": branch_step,
                        "intermediate_final_corr": intermediate_final_corr,
                        "lookahead_final_corr": lookahead_final_corr,
                    }
                    branch_point_data.append(branch_data)
                    timestep_data[branch_step].append(branch_data)

    # Calculate rank correlations for each branch point
    branch_correlations = []
    intermediate_final_correlations = []
    lookahead_final_correlations = []

    for branch_data in branch_point_data:
        # Only calculate if we have enough branches
        if len(branch_data["value_predictions"]) > 1:
            # Value model vs final FID correlation
            corr, _ = spearmanr(
                branch_data["value_predictions"], branch_data["actual_fid_changes"]
            )
            branch_correlations.append(corr)

            # Intermediate FID vs final FID correlation
            if not np.isnan(branch_data["intermediate_final_corr"]):
                intermediate_final_correlations.append(
                    branch_data["intermediate_final_corr"]
                )

            # Look-ahead FID vs final FID correlation
            if not np.isnan(branch_data["lookahead_final_corr"]):
                lookahead_final_correlations.append(branch_data["lookahead_final_corr"])

    # Calculate average correlations
    avg_correlation = (
        np.mean(branch_correlations) if branch_correlations else float("nan")
    )
    avg_intermediate_final_corr = (
        np.mean(intermediate_final_correlations)
        if intermediate_final_correlations
        else float("nan")
    )
    avg_lookahead_final_corr = (
        np.mean(lookahead_final_correlations)
        if lookahead_final_correlations
        else float("nan")
    )

    # Calculate top branch match percentages
    value_top_match_pct = (
        (value_top_match_count / total_branch_points * 100)
        if total_branch_points > 0
        else 0
    )
    intermediate_top_match_pct = (
        (intermediate_top_match_count / total_branch_points * 100)
        if total_branch_points > 0
        else 0
    )
    lookahead_top_match_pct = (
        (lookahead_top_match_count / total_branch_points * 100)
        if total_branch_points > 0
        else 0
    )

    # Print Value Model correlation results
    print("\n===== Value Model Correlation Results =====")
    print(
        f"Average value-final correlation across all branch points: {avg_correlation:.4f}"
    )
    print(
        f"Value model top branch match percentage: {value_top_match_pct:.2f}% ({value_top_match_count}/{total_branch_points})"
    )

    # Print Intermediate FID correlation results
    print("\n===== Intermediate FID Correlation Results =====")
    print(
        f"Average intermediate-final correlation across all branch points: {avg_intermediate_final_corr:.4f}"
    )
    print(
        f"Intermediate FID top branch match percentage: {intermediate_top_match_pct:.2f}% ({intermediate_top_match_count}/{total_branch_points})"
    )

    # Print Look-ahead FID correlation results
    print("\n===== Look-ahead FID Correlation Results =====")
    print(
        f"Average look-ahead-final correlation across all branch points: {avg_lookahead_final_corr:.4f}"
    )
    print(
        f"Look-ahead FID top branch match percentage: {lookahead_top_match_pct:.2f}% ({lookahead_top_match_count}/{total_branch_points})"
    )

    # Calculate per-timestep average correlations
    print("\n===== Per-Timestep Correlation Analysis =====")
    print("\nValue Model correlations by timestep:")
    timestep_correlations = []
    timestep_intermediate_final_correlations = []
    timestep_lookahead_final_correlations = []
    timestep_value_match_counts = [0] * sampler.num_timesteps
    timestep_intermediate_match_counts = [0] * sampler.num_timesteps
    timestep_lookahead_match_counts = [0] * sampler.num_timesteps
    timestep_branch_counts = [0] * sampler.num_timesteps

    for step in range(sampler.num_timesteps):
        step_correlations = []
        step_intermediate_final_correlations = []
        step_lookahead_final_correlations = []

        for branch_data in timestep_data[step]:
            if len(branch_data["value_predictions"]) > 1:
                # Value model vs final FID correlation
                corr, _ = spearmanr(
                    branch_data["value_predictions"], branch_data["actual_fid_changes"]
                )
                step_correlations.append(corr)

                # Intermediate FID vs final FID correlation
                if not np.isnan(branch_data["intermediate_final_corr"]):
                    step_intermediate_final_correlations.append(
                        branch_data["intermediate_final_corr"]
                    )

                # Look-ahead FID vs final FID correlation
                if not np.isnan(branch_data["lookahead_final_corr"]):
                    step_lookahead_final_correlations.append(
                        branch_data["lookahead_final_corr"]
                    )

                # Count top matches for this timestep
                timestep_branch_counts[step] += 1

                # Check top branch matches
                value_best_idx = np.argmin(branch_data["value_predictions"])
                final_best_idx = np.argmin(branch_data["actual_fid_changes"])
                intermediate_best_idx = np.argmin(
                    branch_data["intermediate_fid_changes"]
                )
                lookahead_best_idx = np.argmin(branch_data["lookahead_fid_changes"])

                if value_best_idx == final_best_idx:
                    timestep_value_match_counts[step] += 1

                if intermediate_best_idx == final_best_idx:
                    timestep_intermediate_match_counts[step] += 1

                if lookahead_best_idx == final_best_idx:
                    timestep_lookahead_match_counts[step] += 1

        if step_correlations:
            avg_step_corr = np.mean(step_correlations)
            timestep_correlations.append(avg_step_corr)

            # Calculate match percentage for this timestep
            match_pct = (
                (timestep_value_match_counts[step] / timestep_branch_counts[step] * 100)
                if timestep_branch_counts[step] > 0
                else 0
            )

            print(
                f"  Timestep {step}: correlation={avg_step_corr:.4f}, top match={match_pct:.1f}% ({timestep_value_match_counts[step]}/{timestep_branch_counts[step]})"
            )
        else:
            timestep_correlations.append(np.nan)
            print(f"  Timestep {step}: insufficient data")

    print("\nIntermediate FID correlations by timestep:")
    for step in range(sampler.num_timesteps):
        step_intermediate_final_correlations = []

        for branch_data in timestep_data[step]:
            if not np.isnan(branch_data["intermediate_final_corr"]):
                step_intermediate_final_correlations.append(
                    branch_data["intermediate_final_corr"]
                )

        if step_intermediate_final_correlations:
            avg_step_intermediate_final_corr = np.mean(
                step_intermediate_final_correlations
            )
            timestep_intermediate_final_correlations.append(
                avg_step_intermediate_final_corr
            )

            # Calculate match percentage for this timestep
            match_pct = (
                (
                    timestep_intermediate_match_counts[step]
                    / timestep_branch_counts[step]
                    * 100
                )
                if timestep_branch_counts[step] > 0
                else 0
            )

            print(
                f"  Timestep {step}: correlation={avg_step_intermediate_final_corr:.4f}, top match={match_pct:.1f}% ({timestep_intermediate_match_counts[step]}/{timestep_branch_counts[step]})"
            )
        else:
            timestep_intermediate_final_correlations.append(np.nan)
            print(f"  Timestep {step}: insufficient data")

    print("\nLook-ahead FID correlations by timestep:")
    for step in range(sampler.num_timesteps):
        step_lookahead_final_correlations = []

        for branch_data in timestep_data[step]:
            if not np.isnan(branch_data["lookahead_final_corr"]):
                step_lookahead_final_correlations.append(
                    branch_data["lookahead_final_corr"]
                )

        if step_lookahead_final_correlations:
            avg_step_lookahead_final_corr = np.mean(step_lookahead_final_correlations)
            timestep_lookahead_final_correlations.append(avg_step_lookahead_final_corr)

            # Calculate match percentage for this timestep
            match_pct = (
                (
                    timestep_lookahead_match_counts[step]
                    / timestep_branch_counts[step]
                    * 100
                )
                if timestep_branch_counts[step] > 0
                else 0
            )

            print(
                f"  Timestep {step}: correlation={avg_step_lookahead_final_corr:.4f}, top match={match_pct:.1f}% ({timestep_lookahead_match_counts[step]}/{timestep_branch_counts[step]})"
            )
        else:
            timestep_lookahead_final_correlations.append(np.nan)
            print(f"  Timestep {step}: insufficient data")

    # Return data for potential plotting
    return {
        "overall_correlation": avg_correlation,
        "overall_intermediate_final_correlation": avg_intermediate_final_corr,
        "overall_lookahead_final_correlation": avg_lookahead_final_corr,
        "timestep_correlations": timestep_correlations,
        "timestep_intermediate_final_correlations": timestep_intermediate_final_correlations,
        "timestep_lookahead_final_correlations": timestep_lookahead_final_correlations,
        "value_top_match_percentage": value_top_match_pct,
        "intermediate_top_match_percentage": intermediate_top_match_pct,
        "lookahead_top_match_percentage": lookahead_top_match_pct,
        "branch_point_data": branch_point_data,
    }


def analyze_intermediate_fid_correlation(
    sampler,
    device,
    num_samples=50,
    num_branches=8,
    dt_std=0.1,
    class_label=None,
    batch_size=10,
):
    """
    Analyze how well the intermediate FID rankings correlate with actual final FID scores.

    Args:
        sampler: The MCTSFlowSampler instance
        device: The device to run computations on
        num_samples: Number of samples to analyze
        num_branches: Number of branches to create at each test point
        dt_std: Standard deviation for dt variation when branching
        class_label: Specific class to analyze (if None, samples across all classes)
        batch_size: Number of samples to process in parallel

    Returns:
        Dictionary containing correlation metrics between intermediate FID rankings and actual FID
    """
    print("\n===== Intermediate FID Correlation Analysis =====")

    # Import necessary libraries
    from scipy.stats import spearmanr
    import torchmetrics.image.fid as FID

    # Initialize FID metric for actual FID calculation
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

    # Process real images in batches
    real_batch_size = 100
    print("Processing real images for FID calculation...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid_metric.update(batch, real=True)

    sampler.flow_model.eval()
    if sampler.value_model:
        sampler.value_model.eval()

    # Data collection
    branch_point_data = []
    timestep_data = [[] for _ in range(sampler.num_timesteps)]

    # Sample across all classes or specific class
    if class_label is None:
        # Distribute samples across classes
        samples_per_class = num_samples // sampler.num_classes
        class_range = range(sampler.num_classes)
    else:
        samples_per_class = num_samples
        class_range = [class_label]

    # Print timestep schedule
    print(f"Timestep schedule: {sampler.timesteps.cpu().numpy()}")

    # Counters for rank correlation
    total_branch_points = 0
    total_rank_correlation = 0
    top1_matches = 0
    top2_matches = 0
    top3_matches = 0

    with torch.no_grad():
        for class_idx in class_range:
            print(f"Processing class {class_idx}...")

            # Process in batches
            num_batches = (
                samples_per_class + batch_size - 1
            ) // batch_size  # Ceiling division
            for batch_idx in tqdm(
                range(num_batches), desc=f"Class {class_idx} batches"
            ):
                # Determine actual batch size (might be smaller for last batch)
                actual_batch_size = min(
                    batch_size, samples_per_class - batch_idx * batch_size
                )
                if actual_batch_size <= 0:
                    break

                # Start with random noise
                x = torch.randn(
                    actual_batch_size,
                    sampler.channels,
                    sampler.image_size,
                    sampler.image_size,
                    device=device,
                )
                label = torch.full((actual_batch_size,), class_idx, device=device)

                # Choose random timesteps to branch at (between 0 and num_timesteps-3)
                # We need at least 2 more steps after branching
                branch_steps = torch.randint(
                    0, sampler.num_timesteps - 2, (actual_batch_size,)
                ).to(device)

                # Track current time for each sample
                current_times = torch.zeros(actual_batch_size, device=device)

                # Simulate each sample up to its branch point
                for step in range(sampler.num_timesteps - 1):
                    # Find which samples need to be updated at this step
                    active_mask = branch_steps >= step
                    if not active_mask.any():
                        break

                    active_x = x[active_mask]
                    active_label = label[active_mask]

                    t = sampler.timesteps[step]
                    dt = sampler.timesteps[step + 1] - t
                    t_batch = torch.full((active_mask.sum(),), t.item(), device=device)

                    velocity = sampler.flow_model(t_batch, active_x, active_label)
                    active_x = active_x + velocity * dt

                    # Update the samples and times
                    x[active_mask] = active_x
                    current_times[active_mask] = sampler.timesteps[step + 1].item()

                # Process each sample's branches separately
                for i in range(actual_batch_size):
                    branch_step = branch_steps[i].item()

                    # Create branches for this sample
                    branches = x[i : i + 1].repeat(num_branches, 1, 1, 1)
                    branch_labels = label[i : i + 1].repeat(num_branches)
                    branch_times = torch.full(
                        (num_branches,), current_times[i].item(), device=device
                    )

                    # Sample different dt values for each branch
                    base_dt = 1.0 / sampler.num_timesteps
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

                    # Get velocity for all branches in a single batched calculation
                    velocity = sampler.flow_model(branch_times, branches, branch_labels)

                    # Apply different step sizes to create branches
                    branched_samples = branches + velocity * dts.view(-1, 1, 1, 1)
                    new_times = branch_times + dts

                    # Now simulate all branches forward one more step to a common time point
                    next_timestep = sampler.timesteps[branch_step + 1].item()

                    # Calculate dt to reach the next common timestep for each branch
                    dt_to_next = next_timestep - new_times

                    # Get velocity for all branches in a single batched calculation
                    velocity = sampler.flow_model(
                        new_times, branched_samples, branch_labels
                    )

                    # Apply the step to all branches at once
                    aligned_samples = branched_samples + velocity * dt_to_next.view(
                        -1, 1, 1, 1
                    )
                    aligned_times = torch.full(
                        (num_branches,), next_timestep, device=device
                    )

                    # Calculate intermediate FID for aligned samples
                    intermediate_fid_changes = []
                    for j in range(num_branches):
                        intermediate_fid = sampler.compute_fid_change(
                            aligned_samples[j : j + 1], class_idx
                        )
                        intermediate_fid_changes.append(intermediate_fid)

                    # Simulate all branches to completion
                    final_samples = []
                    for j in range(num_branches):
                        # Start from aligned sample
                        current_sample = aligned_samples[j : j + 1]
                        current_time = next_timestep

                        # Simulate until completion
                        for step in range(branch_step + 1, sampler.num_timesteps - 1):
                            t = sampler.timesteps[step]
                            dt = sampler.timesteps[step + 1] - t
                            t_batch = torch.full((1,), t.item(), device=device)

                            velocity = sampler.flow_model(
                                t_batch, current_sample, branch_labels[j : j + 1]
                            )
                            current_sample = current_sample + velocity * dt

                        final_samples.append(current_sample)

                    # Stack final samples
                    final_samples_tensor = torch.cat(final_samples, dim=0)

                    # Create FID metrics for each branch
                    branch_fids = []
                    for j in range(num_branches):
                        branch_fid = FID.FrechetInceptionDistance(
                            normalize=True, reset_real_features=False
                        ).to(device)
                        # Copy real features from main FID metric
                        branch_fid.real_features = fid_metric.real_features.clone()
                        branch_fid.real_mean = fid_metric.real_mean.clone()
                        branch_fid.real_cov = fid_metric.real_cov.clone()
                        branch_fid.real_features_num = fid_metric.real_features_num
                        branch_fids.append(branch_fid)

                    # Generate samples for each branch
                    num_gen_samples = 100  # Number of samples to generate per branch
                    gen_batch_size = 10

                    # Generate samples for all branches
                    for j in range(num_branches):
                        # Generate samples in smaller batches
                        for k in range(0, num_gen_samples, gen_batch_size):
                            actual_gen_size = min(gen_batch_size, num_gen_samples - k)

                            # Generate samples across all classes
                            all_class_samples = []
                            for gen_class in range(sampler.num_classes):
                                # Start from noise
                                gen_x = torch.randn(
                                    actual_gen_size,
                                    sampler.channels,
                                    sampler.image_size,
                                    sampler.image_size,
                                    device=device,
                                )
                                gen_label = torch.full(
                                    (actual_gen_size,), gen_class, device=device
                                )

                                # Simulate up to branch point using standard trajectory
                                gen_time = torch.zeros(actual_gen_size, device=device)

                                for step in range(branch_step + 1):
                                    t = sampler.timesteps[step]
                                    dt = sampler.timesteps[step + 1] - t
                                    t_batch = torch.full(
                                        (actual_gen_size,), t.item(), device=device
                                    )

                                    velocity = sampler.flow_model(
                                        t_batch, gen_x, gen_label
                                    )
                                    gen_x = gen_x + velocity * dt
                                    gen_time = sampler.timesteps[step + 1].item()

                                # Apply the specific branch's dt
                                t_batch = torch.full(
                                    (actual_gen_size,), gen_time, device=device
                                )
                                velocity = sampler.flow_model(t_batch, gen_x, gen_label)
                                gen_x = gen_x + velocity * dts[j]
                                gen_time = gen_time + dts[j]

                                # Continue simulation to completion
                                for step in range(
                                    branch_step + 1, sampler.num_timesteps - 1
                                ):
                                    t = sampler.timesteps[step]
                                    dt = sampler.timesteps[step + 1] - t
                                    t_batch = torch.full(
                                        (actual_gen_size,), t.item(), device=device
                                    )

                                    velocity = sampler.flow_model(
                                        t_batch, gen_x, gen_label
                                    )
                                    gen_x = gen_x + velocity * dt

                                all_class_samples.append(gen_x)

                            # Combine samples from all classes
                            combined_samples = torch.cat(all_class_samples, dim=0)

                            # Update FID for this branch
                            branch_fids[j].update(combined_samples, real=False)

                    # Compute FID scores for all branches
                    actual_fid_scores = [fid.compute().item() for fid in branch_fids]

                    # Rank branches by intermediate FID (lower is better)
                    intermediate_ranks = np.argsort(intermediate_fid_changes)

                    # Rank branches by actual FID (lower is better)
                    actual_ranks = np.argsort(actual_fid_scores)

                    # Store branch data
                    branch_data = {
                        "intermediate_fid_changes": np.array(intermediate_fid_changes),
                        "intermediate_ranks": intermediate_ranks,
                        "actual_fid_scores": np.array(actual_fid_scores),
                        "actual_ranks": actual_ranks,
                        "timestep": branch_step,
                    }

                    branch_point_data.append(branch_data)
                    timestep_data[branch_step].append(branch_data)

                    # Update correlation statistics
                    total_branch_points += 1

                    # Calculate rank correlation between intermediate FID and actual FID
                    if len(intermediate_fid_changes) > 1:
                        corr, _ = spearmanr(intermediate_fid_changes, actual_fid_scores)
                        total_rank_correlation += corr

                        # Check top-k matches
                        if intermediate_ranks[0] == actual_ranks[0]:
                            top1_matches += 1

                        if intermediate_ranks[0] in actual_ranks[:2]:
                            top2_matches += 1

                        if intermediate_ranks[0] in actual_ranks[:3]:
                            top3_matches += 1

    # Calculate average rank correlation
    avg_rank_correlation = (
        total_rank_correlation / total_branch_points if total_branch_points > 0 else 0
    )

    print(
        f"\nAverage rank correlation between intermediate FID and actual FID: {avg_rank_correlation:.4f}"
    )

    # Calculate top-k accuracy
    top1_accuracy = top1_matches / total_branch_points if total_branch_points > 0 else 0
    top2_accuracy = top2_matches / total_branch_points if total_branch_points > 0 else 0
    top3_accuracy = top3_matches / total_branch_points if total_branch_points > 0 else 0

    print(f"Top-1 accuracy: {top1_accuracy:.4f} ({top1_matches}/{total_branch_points})")
    print(f"Top-2 accuracy: {top2_accuracy:.4f} ({top2_matches}/{total_branch_points})")
    print(f"Top-3 accuracy: {top3_accuracy:.4f} ({top3_matches}/{total_branch_points})")

    # Calculate per-timestep correlations
    print("\nRank correlation by timestep:")
    timestep_correlations = []
    timestep_top1_accuracy = []

    for step in range(sampler.num_timesteps):
        step_correlations = []
        step_top1_matches = 0
        step_branch_points = 0

        for branch_data in timestep_data[step]:
            if len(branch_data["intermediate_fid_changes"]) > 1:
                corr, _ = spearmanr(
                    branch_data["intermediate_fid_changes"],
                    branch_data["actual_fid_scores"],
                )
                step_correlations.append(corr)

                # Check top-1 match for this timestep
                if (
                    branch_data["intermediate_ranks"][0]
                    == branch_data["actual_ranks"][0]
                ):
                    step_top1_matches += 1

                step_branch_points += 1

        if step_correlations:
            avg_step_corr = np.mean(step_correlations)
            timestep_correlations.append(avg_step_corr)

            step_top1_acc = (
                step_top1_matches / step_branch_points if step_branch_points > 0 else 0
            )
            timestep_top1_accuracy.append(step_top1_acc)

            print(
                f"  Timestep {step}: correlation={avg_step_corr:.4f}, top-1 accuracy={step_top1_acc:.4f} ({step_top1_matches}/{step_branch_points})"
            )
        else:
            timestep_correlations.append(np.nan)
            timestep_top1_accuracy.append(np.nan)
            print(f"  Timestep {step}: insufficient data")

    # Return analysis results
    return {
        "avg_rank_correlation": avg_rank_correlation,
        "top1_accuracy": top1_accuracy,
        "top2_accuracy": top2_accuracy,
    }


def main():
    # Use the specified GPU device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
    )
    # # Run value model prediction analysis
    # analysis_results = analyze_value_model_predictions(
    #     sampler=sampler,
    #     device=device,
    #     num_samples=1000,
    #     num_branches=8,
    #     dt_std=0.1,
    # )
    analyze_intermediate_fid_correlation(
        sampler=sampler,
        device=device,
        num_samples=1000,
        num_branches=8,
        dt_std=0.1,
    )


if __name__ == "__main__":
    main()
