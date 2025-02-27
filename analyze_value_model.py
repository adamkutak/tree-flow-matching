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

    # Data collection
    value_predictions = []
    actual_fid_changes = []
    timestep_indices = []

    # Sample across all classes or specific class
    if class_label is None:
        # Distribute samples across classes
        samples_per_class = num_samples // sampler.num_classes
        class_range = range(sampler.num_classes)
    else:
        samples_per_class = num_samples
        class_range = [class_label]

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

                # Create branches for all samples
                all_branches = []
                all_branch_labels = []
                all_branch_times = []
                all_branch_indices = (
                    []
                )  # To track which original sample each branch belongs to

                for i in range(actual_batch_size):
                    branches = x[i : i + 1].repeat(num_branches, 1, 1, 1)
                    branch_labels = label[i : i + 1].repeat(num_branches)
                    branch_times = torch.full(
                        (num_branches,), current_times[i].item(), device=device
                    )

                    all_branches.append(branches)
                    all_branch_labels.append(branch_labels)
                    all_branch_times.append(branch_times)
                    all_branch_indices.append(
                        torch.full((num_branches,), i, device=device)
                    )

                # Concatenate all branches
                all_branches = torch.cat(all_branches, dim=0)
                all_branch_labels = torch.cat(all_branch_labels, dim=0)
                all_branch_times = torch.cat(all_branch_times, dim=0)
                all_branch_indices = torch.cat(all_branch_indices, dim=0)

                # Sample different dt values for each branch
                base_dt = 1.0 / sampler.num_timesteps
                dts = torch.normal(
                    mean=base_dt,
                    std=dt_std * base_dt,
                    size=(len(all_branches),),
                    device=device,
                )
                dts = torch.clamp(dts, min=0.0, max=1.0 - all_branch_times)

                # Get velocity for all branches at once
                velocity = sampler.flow_model(
                    all_branch_times, all_branches, all_branch_labels
                )

                # Apply different step sizes to create branches
                branched_samples = all_branches + velocity * dts.view(-1, 1, 1, 1)
                new_times = all_branch_times + dts

                # Get value model predictions for all branches at once
                value_preds = sampler.value_model(
                    new_times, branched_samples, all_branch_labels
                )

                # Now simulate each branch to completion without further branching
                # We'll process this in sub-batches to avoid memory issues
                sub_batch_size = 50  # Adjust based on your GPU memory
                num_sub_batches = (
                    len(branched_samples) + sub_batch_size - 1
                ) // sub_batch_size

                all_final_samples = []

                for sub_idx in range(num_sub_batches):
                    start_idx = sub_idx * sub_batch_size
                    end_idx = min((sub_idx + 1) * sub_batch_size, len(branched_samples))

                    sub_samples = branched_samples[start_idx:end_idx]
                    sub_times = new_times[start_idx:end_idx]
                    sub_labels = all_branch_labels[start_idx:end_idx]

                    # Continue simulation to the end for each sample in the sub-batch
                    current_sub_samples = sub_samples.clone()
                    current_sub_times = sub_times.clone()

                    # Find the next timestep index for each sample
                    next_steps = torch.zeros(
                        len(current_sub_times), dtype=torch.long, device=device
                    )
                    for i, time in enumerate(current_sub_times):
                        for j in range(len(sampler.timesteps)):
                            if sampler.timesteps[j] >= time:
                                next_steps[i] = j
                                break

                    # Simulate until all samples reach the end
                    for step in range(sampler.num_timesteps - 1):
                        # Find which samples need to be updated at this step
                        active_mask = next_steps <= step
                        if not active_mask.any():
                            continue

                        active_samples = current_sub_samples[active_mask]
                        active_labels = sub_labels[active_mask]

                        t = sampler.timesteps[step]
                        dt = sampler.timesteps[step + 1] - t
                        t_batch = torch.full(
                            (active_mask.sum(),), t.item(), device=device
                        )

                        velocity = sampler.flow_model(
                            t_batch, active_samples, active_labels
                        )
                        active_samples = active_samples + velocity * dt

                        # Update the samples
                        current_sub_samples[active_mask] = active_samples

                    all_final_samples.append(current_sub_samples)

                # Concatenate all final samples
                all_final_samples = torch.cat(all_final_samples, dim=0)

                # Calculate actual FID change for each final sample
                for i in range(len(all_final_samples)):
                    sample_idx = all_branch_indices[i].item()
                    branch_step = branch_steps[sample_idx].item()

                    actual_fid_change = sampler.compute_fid_change(
                        all_final_samples[i : i + 1], class_idx
                    )

                    # Store data
                    value_predictions.append(value_preds[i].item())
                    actual_fid_changes.append(actual_fid_change)
                    timestep_indices.append(branch_step)

    # Convert to numpy arrays for analysis
    value_predictions = np.array(value_predictions)
    actual_fid_changes = np.array(actual_fid_changes)
    timestep_indices = np.array(timestep_indices)

    # Calculate overall correlation
    overall_corr = np.corrcoef(value_predictions, actual_fid_changes)[0, 1]
    print(f"Overall correlation: {overall_corr:.4f}")

    # Calculate per-timestep correlations
    timestep_correlations = []
    for step in range(sampler.num_timesteps):
        mask = timestep_indices == step
        if np.sum(mask) > 1:  # Need at least 2 points for correlation
            step_corr = np.corrcoef(value_predictions[mask], actual_fid_changes[mask])[
                0, 1
            ]
            timestep_correlations.append(step_corr)
        else:
            timestep_correlations.append(np.nan)

    # Print per-timestep correlations
    print("\nPer-timestep correlations:")
    for step, corr in enumerate(timestep_correlations):
        if not np.isnan(corr):
            print(f"  Timestep {step}: {corr:.4f}")
        else:
            print(f"  Timestep {step}: insufficient data")

    # Return data for potential plotting
    return {
        "overall_correlation": overall_corr,
        "timestep_correlations": timestep_correlations,
        "value_predictions": value_predictions,
        "actual_fid_changes": actual_fid_changes,
        "timestep_indices": timestep_indices,
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
        num_samples=5000,
        num_branches=8,
        dt_std=0.1,
    )


if __name__ == "__main__":
    main()
