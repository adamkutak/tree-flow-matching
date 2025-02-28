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

    # Debug: Print timestep schedule
    print(f"Timestep schedule: {sampler.timesteps.cpu().numpy()}")

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

                # Choose random timesteps to branch at (between 0 and num_timesteps-2)
                branch_steps = torch.randint(
                    0, sampler.num_timesteps - 1, (actual_batch_size,)
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

                    # Get velocity for all branches
                    velocity = sampler.flow_model(branch_times, branches, branch_labels)

                    # Apply different step sizes to create branches
                    branched_samples = branches + velocity * dts.view(-1, 1, 1, 1)
                    new_times = branch_times + dts

                    # Get value model predictions for all branches
                    value_preds = sampler.value_model(
                        new_times, branched_samples, branch_labels
                    )

                    # Calculate FID for intermediate branched samples
                    intermediate_fid_changes = []
                    for j in range(num_branches):
                        intermediate_fid = sampler.compute_fid_change(
                            branched_samples[j : j + 1], class_idx
                        )
                        intermediate_fid_changes.append(intermediate_fid)

                    # Debug: Print branch times
                    print(
                        f"\nBranch point at timestep {branch_step}, time {branch_times[0].item():.4f}"
                    )
                    for j in range(num_branches):
                        print(
                            f"  Branch {j}: time after step = {new_times[j].item():.4f}"
                        )

                    # Simulate each branch to completion
                    final_samples = []
                    final_times = []
                    for j in range(num_branches):
                        # Start with the branched sample
                        current_sample = branched_samples[j : j + 1].clone()
                        current_time = new_times[j].item()

                        # Find the next timestep index
                        next_step = None
                        for s in range(len(sampler.timesteps)):
                            if sampler.timesteps[s] >= current_time:
                                next_step = s
                                break

                        # Debug: Check if next_step was found
                        if next_step is None:
                            print(
                                f"WARNING: Could not find next timestep for time {current_time:.4f}"
                            )
                            # Default to the last timestep
                            next_step = len(sampler.timesteps) - 1

                        print(
                            f"  Branch {j}: current_time={current_time:.4f}, next_step={next_step}, next_time={sampler.timesteps[next_step].item():.4f}"
                        )

                        # Simulate until completion
                        for step in range(next_step, sampler.num_timesteps - 1):
                            t = sampler.timesteps[step]
                            dt = sampler.timesteps[step + 1] - t
                            t_batch = torch.full((1,), t.item(), device=device)

                            velocity = sampler.flow_model(
                                t_batch, current_sample, branch_labels[j : j + 1]
                            )
                            current_sample = current_sample + velocity * dt

                            # Track the final time for verification
                            if step == sampler.num_timesteps - 2:
                                final_times.append(sampler.timesteps[step + 1].item())

                        final_samples.append(current_sample)

                    # Debug: Verify all branches reached t=1
                    print("  Final times for all branches:")
                    for j, time in enumerate(final_times):
                        print(f"    Branch {j}: final time = {time:.4f}")

                    # Concatenate all final samples for this branch point
                    final_samples = torch.cat(final_samples, dim=0)

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
                        print(
                            f"  Correlation between intermediate FID and final FID: {intermediate_final_corr:.4f}"
                        )
                    else:
                        intermediate_final_corr = float("nan")
                        print(
                            "  Insufficient data for intermediate-final FID correlation"
                        )

                    # Store data for this branch point
                    branch_data = {
                        "value_predictions": value_preds.cpu().numpy(),
                        "intermediate_fid_changes": np.array(intermediate_fid_changes),
                        "actual_fid_changes": np.array(actual_fid_changes),
                        "timestep": branch_step,
                        "intermediate_final_corr": intermediate_final_corr,
                    }
                    branch_point_data.append(branch_data)
                    timestep_data[branch_step].append(branch_data)

    # Calculate rank correlations for each branch point
    branch_correlations = []
    intermediate_final_correlations = []
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

    # Calculate average correlations
    avg_correlation = (
        np.mean(branch_correlations) if branch_correlations else float("nan")
    )
    avg_intermediate_final_corr = (
        np.mean(intermediate_final_correlations)
        if intermediate_final_correlations
        else float("nan")
    )

    print(f"Average rank correlation across all branch points: {avg_correlation:.4f}")
    print(
        f"Average correlation between intermediate FID and final FID: {avg_intermediate_final_corr:.4f}"
    )

    # Calculate per-timestep average correlations
    timestep_correlations = []
    timestep_intermediate_final_correlations = []

    for step in range(sampler.num_timesteps):
        step_correlations = []
        step_intermediate_final_correlations = []

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

        if step_correlations:
            avg_step_corr = np.mean(step_correlations)
            timestep_correlations.append(avg_step_corr)
            print(
                f"  Timestep {step}: Value-Final correlation: {avg_step_corr:.4f} (from {len(step_correlations)} branch points)"
            )
        else:
            timestep_correlations.append(np.nan)
            print(f"  Timestep {step}: insufficient data for Value-Final correlation")

        if step_intermediate_final_correlations:
            avg_step_intermediate_final_corr = np.mean(
                step_intermediate_final_correlations
            )
            timestep_intermediate_final_correlations.append(
                avg_step_intermediate_final_corr
            )
            print(
                f"  Timestep {step}: Intermediate-Final correlation: {avg_step_intermediate_final_corr:.4f} (from {len(step_intermediate_final_correlations)} branch points)"
            )
        else:
            timestep_intermediate_final_correlations.append(np.nan)
            print(
                f"  Timestep {step}: insufficient data for Intermediate-Final correlation"
            )

    # Return data for potential plotting
    return {
        "overall_correlation": avg_correlation,
        "overall_intermediate_final_correlation": avg_intermediate_final_corr,
        "timestep_correlations": timestep_correlations,
        "timestep_intermediate_final_correlations": timestep_intermediate_final_correlations,
        "branch_point_data": branch_point_data,
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
    )
    # Run value model prediction analysis
    analysis_results = analyze_value_model_predictions(
        sampler=sampler,
        device=device,
        num_samples=2000,
        num_branches=4,
        dt_std=0.1,
    )


if __name__ == "__main__":
    main()
