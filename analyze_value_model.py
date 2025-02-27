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
    sampler, device, num_samples=100, num_branches=8, dt_std=0.1, class_label=None
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

            for _ in tqdm(range(samples_per_class), desc=f"Class {class_idx} samples"):
                # Start with random noise
                x = torch.randn(
                    1,
                    sampler.channels,
                    sampler.image_size,
                    sampler.image_size,
                    device=device,
                )
                label = torch.tensor([class_idx], device=device)

                # Choose random timestep to branch at (between 0 and num_timesteps-2)
                branch_step = torch.randint(0, sampler.num_timesteps - 1, (1,)).item()

                # Simulate up to branch point
                current_time = 0.0
                for step in range(branch_step):
                    t = sampler.timesteps[step]
                    dt = sampler.timesteps[step + 1] - t
                    t_batch = torch.full((1,), t.item(), device=device)
                    velocity = sampler.flow_model(t_batch, x, label)
                    x = x + velocity * dt
                    current_time = sampler.timesteps[step + 1].item()

                # Create branches at this point
                branches = x.repeat(num_branches, 1, 1, 1)
                branch_labels = label.repeat(num_branches)
                branch_times = torch.full((num_branches,), current_time, device=device)

                # Sample different dt values for each branch
                base_dt = 1.0 / sampler.num_timesteps
                dts = torch.normal(
                    mean=base_dt,
                    std=dt_std * base_dt,
                    size=(num_branches,),
                    device=device,
                )
                dts = torch.clamp(dts, min=0.0, max=1.0 - current_time)

                # Get velocity for all branches
                velocity = sampler.flow_model(branch_times, branches, branch_labels)

                # Apply different step sizes to create branches
                branched_samples = branches + velocity * dts.view(-1, 1, 1, 1)
                new_times = branch_times + dts

                # Get value model predictions for all branches
                value_preds = sampler.value_model(
                    new_times, branched_samples, branch_labels
                )

                # Now simulate each branch to completion without further branching
                final_samples = []
                for i in range(num_branches):
                    sample = branched_samples[i : i + 1]  # Keep batch dimension
                    time = new_times[i].item()

                    # Find the closest timestep index
                    next_step = 0
                    for j in range(len(sampler.timesteps)):
                        if sampler.timesteps[j] >= time:
                            next_step = j
                            break

                    # Continue simulation to the end
                    for step in range(next_step, sampler.num_timesteps - 1):
                        t = sampler.timesteps[step]
                        dt = sampler.timesteps[step + 1] - t
                        t_batch = torch.full((1,), t.item(), device=device)
                        velocity = sampler.flow_model(t_batch, sample, label)
                        sample = sample + velocity * dt

                    final_samples.append(sample)

                # Calculate actual FID change for each final sample
                for i in range(num_branches):
                    actual_fid_change = sampler.compute_fid_change(
                        final_samples[i], class_idx
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
