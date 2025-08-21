import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchmetrics.image.fid as FID
import os
import numpy as np
import argparse
import json
from datetime import datetime
from tqdm import tqdm

from mcts_single_flow import MCTSFlowSampler
from imagenet_dataset import ImageNet32Dataset
from run_mcts_flow import calculate_inception_score, compute_dino_accuracy
from utils import (
    divfree_swirl_si,
    score_si_linear,
    divergence_free_particle_guidance,
    get_alpha_schedule_flow_matching,
    sample_spherical_simplex_gaussian,
    make_divergence_free,
)


def batch_sample_ode_with_metrics(sampler, class_label, batch_size=16):
    """
    Regular flow matching sampling without branching, with velocity magnitude tracking.
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes = []
    with torch.no_grad():
        current_samples = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        if is_tensor:
            current_label = class_label
        else:
            current_label = torch.full(
                (batch_size,), class_label, device=sampler.device
            )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            # Flow step
            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            # Track velocity magnitude
            dims = tuple(range(1, velocity.ndim))
            vel_mag = torch.linalg.vector_norm(velocity, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            current_samples = current_samples + velocity * dt

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        return (
            sampler.unnormalize_images(current_samples),
            avg_velocity_magnitude,
            velocity_magnitudes,
        )


def batch_sample_sde_with_metrics(
    sampler, class_label, batch_size=16, noise_scale=0.05
):
    """
    Flow matching sampling with added noise (SDE sampling), with noise magnitude tracking.
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes = []
    noise_magnitudes = []
    noise_to_velocity_ratios = []

    with torch.no_grad():
        current_samples = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        if is_tensor:
            current_label = class_label
        else:
            current_label = torch.full(
                (batch_size,), class_label, device=sampler.device
            )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            # Flow step with SDE noise term
            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            # Add noise scaled by dt and noise_scale
            noise = torch.randn_like(current_samples) * torch.sqrt(dt) * noise_scale

            # Calculate what actually gets applied
            applied_velocity = velocity * dt
            applied_noise = noise

            # Track magnitudes of what's actually applied
            dims = tuple(range(1, velocity.ndim))
            vel_mag = torch.linalg.vector_norm(applied_velocity, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            noise_mag = torch.linalg.vector_norm(applied_noise, dim=dims).mean().item()
            noise_magnitudes.append(noise_mag)

            # Track ratio
            ratio = noise_mag / vel_mag if vel_mag > 0 else 0
            noise_to_velocity_ratios.append(ratio)

            # Euler-Maruyama update
            current_samples = current_samples + applied_velocity + applied_noise

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        avg_noise_magnitude = sum(noise_magnitudes) / len(noise_magnitudes)
        avg_noise_to_velocity_ratio = sum(noise_to_velocity_ratios) / len(
            noise_to_velocity_ratios
        )

        return (
            sampler.unnormalize_images(current_samples),
            avg_velocity_magnitude,
            avg_noise_magnitude,
            avg_noise_to_velocity_ratio,
            velocity_magnitudes,
            noise_magnitudes,
            noise_to_velocity_ratios,
        )


def batch_sample_sde_divfree_with_metrics(
    sampler,
    class_label,
    batch_size=16,
    lambda_div=0.2,
):
    """
    SDE sampling with divergence-free field as noise, with divergence-free field magnitude tracking.
    Uses Euler-Maruyama with the divergence-free field as the stochastic noise term.
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes = []
    divfree_magnitudes = []
    divfree_to_velocity_ratios = []

    with torch.no_grad():
        x = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        if is_tensor:
            y = class_label
        else:
            y = torch.full(
                (batch_size,), class_label, device=sampler.device, dtype=torch.long
            )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            u_t = sampler.flow_model(t_batch, x, y)  # drift
            w_unscaled = divfree_swirl_si(x, t_batch, y, u_t)

            # Use divergence-free field as noise term in Euler-Maruyama scheme
            divfree_noise = w_unscaled * torch.sqrt(dt) * lambda_div

            # Calculate what actually gets applied
            applied_velocity = u_t * dt
            applied_divfree = divfree_noise

            # Track magnitudes of what's actually applied
            dims = tuple(range(1, u_t.ndim))
            vel_mag = torch.linalg.vector_norm(applied_velocity, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            div_mag = torch.linalg.vector_norm(applied_divfree, dim=dims).mean().item()
            divfree_magnitudes.append(div_mag)

            # Track ratio
            ratio = div_mag / vel_mag if vel_mag > 0 else 0
            divfree_to_velocity_ratios.append(ratio)

            # Euler-Maruyama update: drift + noise
            x = x + applied_velocity + applied_divfree

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        avg_divfree_magnitude = sum(divfree_magnitudes) / len(divfree_magnitudes)
        avg_divfree_to_velocity_ratio = sum(divfree_to_velocity_ratios) / len(
            divfree_to_velocity_ratios
        )

        return (
            sampler.unnormalize_images(x),
            avg_velocity_magnitude,
            avg_divfree_magnitude,
            avg_divfree_to_velocity_ratio,
            velocity_magnitudes,
            divfree_magnitudes,
            divfree_to_velocity_ratios,
        )


def batch_sample_ode_divfree_with_metrics(
    sampler,
    class_label,
    batch_size=16,
    lambda_div=0.2,
):
    """
    ODE sampling with divergence-free term, with divergence-free field magnitude tracking.
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes = []
    divfree_magnitudes = []
    divfree_to_velocity_ratios = []

    with torch.no_grad():
        x = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        if is_tensor:
            y = class_label
        else:
            y = torch.full(
                (batch_size,), class_label, device=sampler.device, dtype=torch.long
            )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            u_t = sampler.flow_model(t_batch, x, y)  # drift
            w_unscaled = divfree_swirl_si(x, t_batch, y, u_t)
            w = lambda_div * w_unscaled

            # Calculate what actually gets applied
            applied_velocity = u_t * dt
            applied_divfree = w * dt

            # Track magnitudes of what's actually applied
            dims = tuple(range(1, u_t.ndim))
            vel_mag = torch.linalg.vector_norm(applied_velocity, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            div_mag = torch.linalg.vector_norm(applied_divfree, dim=dims).mean().item()
            divfree_magnitudes.append(div_mag)

            # Track ratio
            ratio = div_mag / vel_mag if vel_mag > 0 else 0
            divfree_to_velocity_ratios.append(ratio)

            x = x + applied_velocity + applied_divfree  # Euler ODE step

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        avg_divfree_magnitude = sum(divfree_magnitudes) / len(divfree_magnitudes)
        avg_divfree_to_velocity_ratio = sum(divfree_to_velocity_ratios) / len(
            divfree_to_velocity_ratios
        )

        return (
            sampler.unnormalize_images(x),
            avg_velocity_magnitude,
            avg_divfree_magnitude,
            avg_divfree_to_velocity_ratio,
            velocity_magnitudes,
            divfree_magnitudes,
            divfree_to_velocity_ratios,
        )


def batch_sample_edm_sde_with_metrics(sampler, class_label, batch_size=16, beta=0.05):
    """
    EDM Section-4 SDE sampler:
        dX = [u_t  − beta * sigma(t)^2 * s_t] dt
             + sqrt(2 * beta) * sigma(t) * dW.
    Tracks velocity and Brownian-noise magnitudes per step.
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes, noise_magnitudes, noise_to_velocity_ratios = [], [], []

    with torch.no_grad():
        current_samples = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )
        current_label = (
            class_label.to(sampler.device)
            if is_tensor
            else torch.full((batch_size,), class_label, device=sampler.device)
        )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            score = score_si_linear(current_samples, t_batch, velocity)

            sigma_t = t_batch.view(-1, *([1] * (current_samples.ndim - 1)))

            drift_corr = -beta * (sigma_t**2) * score

            # Total effective velocity is the sum of flow velocity and drift correction
            total_velocity = velocity + drift_corr

            brownian = (
                torch.randn_like(current_samples)
                * torch.sqrt(dt)
                * torch.sqrt(torch.tensor(2.0 * beta, device=sampler.device))
                * sigma_t
            )

            dims = tuple(range(1, velocity.ndim))
            # Track the magnitude of the total effective velocity, not just the raw velocity
            vel_mag = torch.linalg.vector_norm(total_velocity, dim=dims).mean().item()
            noise_mag = torch.linalg.vector_norm(brownian, dim=dims).mean().item()

            velocity_magnitudes.append(vel_mag)
            noise_magnitudes.append(noise_mag)
            noise_to_velocity_ratios.append(noise_mag / vel_mag if vel_mag > 0 else 0)

            current_samples = current_samples + total_velocity * dt + brownian

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        avg_noise_magnitude = sum(noise_magnitudes) / len(noise_magnitudes)
        avg_ratio = sum(noise_to_velocity_ratios) / len(noise_to_velocity_ratios)

        return (
            sampler.unnormalize_images(current_samples),
            avg_velocity_magnitude,
            avg_noise_magnitude,
            avg_ratio,
            velocity_magnitudes,
            noise_magnitudes,
            noise_to_velocity_ratios,
        )


def batch_sample_score_sde_with_metrics(
    sampler, class_label, batch_size=16, noise_scale_factor=1.0
):
    """
    Score SDE conversion from Section 4.2:
        dx_t = f_t(x_t)dt + g_t dw
    where f_t(x_t) = u_t(x_t) - (g_t^2/2) * ∇ log p_t(x_t)
    and g_t = t^2 scaled by noise_scale_factor.
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes = []
    noise_magnitudes = []
    noise_to_velocity_ratios = []

    with torch.no_grad():
        current_samples = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        if is_tensor:
            current_label = class_label
        else:
            current_label = torch.full(
                (batch_size,), class_label, device=sampler.device
            )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            # Get velocity from flow model
            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            # Compute score using existing score_si_linear function
            score = score_si_linear(current_samples, t_batch, velocity)

            # Compute noise schedule g_t = t^2 scaled by factor
            g_t = (t_batch**2) * noise_scale_factor
            g_t = g_t.view(-1, *([1] * (current_samples.ndim - 1)))

            # Compute drift coefficient: f_t(x_t) = u_t(x_t) - (g_t^2/2) * score
            drift_correction = -(g_t**2) / 2.0 * score
            drift = velocity + drift_correction

            # Generate noise term: g_t * dW
            noise = torch.randn_like(current_samples) * g_t * torch.sqrt(dt)

            # Track magnitudes
            dims = tuple(range(1, velocity.ndim))
            vel_mag = torch.linalg.vector_norm(drift, dim=dims).mean().item()
            noise_mag = torch.linalg.vector_norm(noise, dim=dims).mean().item()

            velocity_magnitudes.append(vel_mag)
            noise_magnitudes.append(noise_mag)
            noise_to_velocity_ratios.append(noise_mag / vel_mag if vel_mag > 0 else 0)

            # Euler-Maruyama update
            current_samples = current_samples + drift * dt + noise

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        avg_noise_magnitude = sum(noise_magnitudes) / len(noise_magnitudes)
        avg_noise_to_velocity_ratio = sum(noise_to_velocity_ratios) / len(
            noise_to_velocity_ratios
        )

        return (
            sampler.unnormalize_images(current_samples),
            avg_velocity_magnitude,
            avg_noise_magnitude,
            avg_noise_to_velocity_ratio,
            velocity_magnitudes,
            noise_magnitudes,
            noise_to_velocity_ratios,
        )


def batch_sample_ode_divfree_max_with_metrics(
    sampler,
    class_label,
    batch_size=16,
    lambda_div=0.2,
    sub_batch_size=4,
    repulsion_strength=0.05,
    noise_schedule_end_factor=0.1,
):
    """
    ODE sampling with divergence-free term using normal Gaussian noise plus repulsion
    within virtual batches, with divergence-free field magnitude tracking.
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes = []
    divfree_magnitudes = []
    divfree_to_velocity_ratios = []

    with torch.no_grad():
        x = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        if is_tensor:
            y = class_label
        else:
            y = torch.full(
                (batch_size,), class_label, device=sampler.device, dtype=torch.long
            )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            u_t = sampler.flow_model(t_batch, x, y)  # drift

            # Apply divfree + repulsion in sub-batches
            w_total = torch.zeros_like(x)
            num_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size

            for sb in range(num_sub_batches):
                start_idx = sb * sub_batch_size
                end_idx = min((sb + 1) * sub_batch_size, batch_size)
                current_sub_batch_size = end_idx - start_idx

                # Get sub-batch data
                sub_x = x[start_idx:end_idx]
                sub_t = t_batch[start_idx:end_idx]
                sub_y = y[start_idx:end_idx]
                sub_u_t = u_t[start_idx:end_idx]

                # Standard divfree term - normal Gaussian noise projected to be divergence-free
                w_divfree_unscaled = divfree_swirl_si(sub_x, sub_t, sub_y, sub_u_t)
                w_divfree = lambda_div * w_divfree_unscaled

                # Add repulsion term within this sub-batch using vectorized approach
                from utils import particle_guidance_forces

                raw_repulsion_forces = particle_guidance_forces(
                    sub_x, 0.0, alpha_t=1.0, kernel_type="euclidean"
                )

                # Regularize repulsion magnitude to match Gaussian magnitude
                dims = tuple(range(1, w_divfree_unscaled.ndim))
                gaussian_magnitude = torch.linalg.vector_norm(
                    w_divfree_unscaled, dim=dims
                ).mean()
                repulsion_magnitude = torch.linalg.vector_norm(
                    raw_repulsion_forces, dim=dims
                ).mean()

                if repulsion_magnitude > 1e-8:  # Avoid division by zero
                    # Scale forces so their average magnitude equals gaussian magnitude
                    regularization_factor = gaussian_magnitude / repulsion_magnitude
                    regularized_repulsion = raw_repulsion_forces * regularization_factor
                else:
                    regularized_repulsion = raw_repulsion_forces

                # Apply user-specified repulsion strength AFTER regularization
                scaled_repulsion = regularized_repulsion * repulsion_strength

                # Make divergence-free
                repulsion_divfree = make_divergence_free(
                    scaled_repulsion, sub_x, sub_t, sub_u_t
                )

                # Combine divfree and repulsion
                combined_perturbation = w_divfree + repulsion_divfree

                # Apply time-dependent scaling to combined perturbation
                t_scalar = t_batch[0].item()
                noise_scale_factor = (
                    1.0 + (noise_schedule_end_factor - 1.0) * t_scalar
                )  # Linear interpolation
                scaled_perturbation = combined_perturbation * noise_scale_factor

                w_total[start_idx:end_idx] = scaled_perturbation

            # Calculate what actually gets applied
            applied_velocity = u_t * dt
            applied_divfree = w_total * dt

            # Track magnitudes of what's actually applied
            dims = tuple(range(1, u_t.ndim))
            vel_mag = torch.linalg.vector_norm(applied_velocity, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            div_mag = torch.linalg.vector_norm(applied_divfree, dim=dims).mean().item()
            divfree_magnitudes.append(div_mag)

            # Track ratio
            ratio = div_mag / vel_mag if vel_mag > 0 else 0
            divfree_to_velocity_ratios.append(ratio)

            x = x + applied_velocity + applied_divfree  # Euler ODE step

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        avg_divfree_magnitude = sum(divfree_magnitudes) / len(divfree_magnitudes)
        avg_divfree_to_velocity_ratio = sum(divfree_to_velocity_ratios) / len(
            divfree_to_velocity_ratios
        )

        return (
            sampler.unnormalize_images(x),
            avg_velocity_magnitude,
            avg_divfree_magnitude,
            avg_divfree_to_velocity_ratio,
            velocity_magnitudes,
            divfree_magnitudes,
            divfree_to_velocity_ratios,
        )


def batch_sample_particle_guidance_with_metrics(
    sampler,
    class_label,
    batch_size=16,
    alpha_0=2.0,
    alpha_1=0.1,
    kernel_type="euclidean",
    schedule_type="linear",
    sub_batch_size=16,
):
    """
    Flow matching with divergence-free particle guidance for diversity.
    Handles large batches by splitting into sub-batches for particle guidance.

    Args:
        sampler: Flow sampler object
        class_label: Class label(s) for conditioning
        batch_size: Total number of samples to generate
        alpha_0: HIGH guidance strength at t=0 (start of sampling, noise)
        alpha_1: LOW guidance strength at t=1 (end of sampling, data)
        kernel_type: 'rbf' or 'euclidean' for repulsive forces
        schedule_type: 'linear', 'exponential' for time-dependent guidance
        sub_batch_size: Size of sub-batches for particle guidance (default 16)
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    velocity_magnitudes = []
    guidance_magnitudes = []
    guidance_to_velocity_ratios = []

    with torch.no_grad():
        x = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        if is_tensor:
            y = class_label
        else:
            y = torch.full(
                (batch_size,), class_label, device=sampler.device, dtype=torch.long
            )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            # Get velocity field from flow model
            u_t = sampler.flow_model(t_batch, x, y)

            # Get time-dependent guidance strength
            alpha_t = get_alpha_schedule_flow_matching(
                t.item(), alpha_0, alpha_1, schedule_type
            )

            # Apply particle guidance in sub-batches
            guidance_forces = torch.zeros_like(x)
            num_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size

            for sb in range(num_sub_batches):
                start_idx = sb * sub_batch_size
                end_idx = min((sb + 1) * sub_batch_size, batch_size)

                if end_idx - start_idx > 1:  # Need at least 2 samples for repulsion
                    sub_x = x[start_idx:end_idx]
                    sub_t = t_batch[start_idx:end_idx]
                    sub_y = y[start_idx:end_idx]
                    sub_u_t = u_t[start_idx:end_idx]

                    # Get divergence-free particle guidance for this sub-batch
                    sub_guidance = divergence_free_particle_guidance(
                        sub_x, sub_t, sub_y, sub_u_t, alpha_t, kernel_type
                    )
                    guidance_forces[start_idx:end_idx] = sub_guidance

            # Calculate what actually gets applied
            applied_velocity = u_t * dt
            applied_guidance = guidance_forces * dt

            # Track magnitudes of what's actually applied
            dims = tuple(range(1, u_t.ndim))
            vel_mag = torch.linalg.vector_norm(applied_velocity, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            guidance_mag = (
                torch.linalg.vector_norm(applied_guidance, dim=dims).mean().item()
            )
            guidance_magnitudes.append(guidance_mag)

            # Track ratio
            ratio = guidance_mag / vel_mag if vel_mag > 0 else 0
            guidance_to_velocity_ratios.append(ratio)

            # ODE update with particle guidance
            x = x + applied_velocity + applied_guidance

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)
        avg_guidance_magnitude = sum(guidance_magnitudes) / len(guidance_magnitudes)
        avg_guidance_to_velocity_ratio = sum(guidance_to_velocity_ratios) / len(
            guidance_to_velocity_ratios
        )

        return (
            sampler.unnormalize_images(x),
            avg_velocity_magnitude,
            avg_guidance_magnitude,
            avg_guidance_to_velocity_ratio,
            velocity_magnitudes,
            guidance_magnitudes,
            guidance_to_velocity_ratios,
        )


def run_sampling_experiment(
    sampler,
    device,
    fid,
    n_samples,
    batch_size,
    sampling_func,
    sampling_params,
    experiment_name,
):
    """Generic experiment runner that works with any sampling function."""
    fid.reset()

    generated_samples = []
    total_metrics = {}
    all_metrics = {}
    class_labels_all = []

    num_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        num_batches += 1

    print(f"Generating {n_samples} samples using {experiment_name}...")

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, n_samples - i * batch_size)

        # Generate diverse classes but in groups of 16 for particle guidance
        # For batch_size=256, this creates 16 groups of 16 samples each with same class
        sub_batch_size = 4
        num_sub_batches = (current_batch_size + sub_batch_size - 1) // sub_batch_size

        # Generate random class for each sub-batch, then expand each to sub_batch_size
        sub_batch_classes = torch.randint(
            0, sampler.num_classes, (num_sub_batches,), device=device
        )
        class_labels = sub_batch_classes.repeat_interleave(sub_batch_size)[
            :current_batch_size
        ]
        class_labels_all.append(class_labels.cpu())

        batch_results = sampling_func(
            sampler, class_labels, current_batch_size, **sampling_params
        )

        samples = batch_results[0]
        generated_samples.extend(samples.cpu())
        fid.update(samples.to(device), real=False)

        for idx, key in enumerate(
            ["avg_velocity_magnitude", "avg_secondary_magnitude", "avg_ratio"]
        ):
            if idx + 1 < len(batch_results) and isinstance(
                batch_results[idx + 1], (int, float)
            ):
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += batch_results[idx + 1] * current_batch_size

        for idx, key in enumerate(
            ["velocity_magnitudes", "secondary_magnitudes", "ratios"]
        ):
            list_idx = len(batch_results) - 3 + idx
            if list_idx < len(batch_results) and isinstance(
                batch_results[list_idx], list
            ):
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].extend(batch_results[list_idx])

    for key in total_metrics:
        total_metrics[key] /= n_samples

    fid_score = fid.compute().item()

    generated_tensor = torch.stack(generated_samples).to(device)
    class_labels_tensor = torch.cat(class_labels_all).to(device)

    inception_score, inception_std = calculate_inception_score(
        generated_tensor, device=device, batch_size=64, splits=10
    )

    dino_accuracy = compute_dino_accuracy(
        sampler, generated_tensor, class_labels_tensor, batch_size=64
    )

    metrics = {
        "fid_score": fid_score,
        "inception_score": inception_score,
        "inception_std": inception_std,
        "dino_top1_accuracy": dino_accuracy["top1_accuracy"],
        "dino_top5_accuracy": dino_accuracy["top5_accuracy"],
        **total_metrics,
        **sampling_params,
    }

    return metrics


def get_methods_to_run(args):
    """
    Determine which methods to run based on user input.

    Args:
        args: Command line arguments

    Returns:
        set: Set of method names to run
    """
    if "all" in args.methods:
        return {
            "ode_baseline",
            "particle_guidance",
            "ode_divfree",
            "ode_divfree_max",
            "sde",
            "sde_divfree",
        }
    else:
        return set(args.methods)


def run_experiment(args):
    """
    Run experiments to measure the effect of different noise levels on sampling quality.
    """
    # Set up device
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Determine dataset parameters
    if args.dataset.lower() == "cifar10":
        num_classes = 10
        print("Using CIFAR-10 dataset")
    elif args.dataset.lower() == "imagenet32" or args.dataset.lower() == "imagenet256":
        num_classes = 1000
        print(f"Using {args.dataset} dataset")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Common parameters
    image_size = 32
    channels = 4 if args.dataset.lower() == "imagenet256" else 3

    if args.dataset.lower() == "imagenet256":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    # Load the appropriate dataset
    if args.dataset.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset.lower() == "imagenet32":
        dataset = ImageNet32Dataset(root_dir="./data", train=False, transform=transform)
    elif args.dataset.lower() == "imagenet256":
        dataset = datasets.ImageNet(root="./data", split="val", transform=transform)

    flow_model_name = f"flow_model_{args.dataset}.pt"

    num_timesteps = 20  # Using fixed 20 timesteps for all experiments

    # Initialize sampler
    sampler = MCTSFlowSampler(
        image_size=image_size,
        channels=channels,
        device=device,
        num_timesteps=num_timesteps,
        num_classes=num_classes,
        buffer_size=10,
        load_models=True,
        flow_model=flow_model_name,
        num_channels=256,
        inception_layer=0,
        dataset=args.dataset,
        flow_model_config=(
            {
                "num_res_blocks": 3,
                "attention_resolutions": "16,8",
            }
            if args.dataset.lower() == "imagenet32"
            else None
        ),
        load_dino=True,
    )

    # Initialize FID for metrics
    fid = FID.FrechetInceptionDistance(normalize=True, reset_real_features=False).to(
        device
    )

    # Load real images for FID calculation
    print(f"Loading dataset for FID calculation...")
    fid_real_samples = 10000
    sample_size = min(args.real_samples, fid_real_samples)
    indices = np.random.choice(fid_real_samples, sample_size, replace=False)
    real_images = torch.stack([dataset[i][0] for i in indices]).to(device)

    # Process real images in batches
    real_batch_size = 100
    print(f"Processing {sample_size} real images from {args.dataset}...")
    for i in range(0, len(real_images), real_batch_size):
        batch = real_images[i : i + real_batch_size]
        fid.update(batch, real=True)

    # Create output directory for results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.output_dir, f"noise_study_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "timesteps": num_timesteps,
        "experiments": [],
    }

    # Get methods to run
    methods_to_run = get_methods_to_run(args)
    print(f"Methods to test: {', '.join(sorted(methods_to_run))}")

    if "particle_guidance" in methods_to_run:
        print("\n\n===== Running Particle Guidance experiments =====")
        # Test different alpha ranges for particle guidance (HIGH at t=0 → LOW at t=1)
        alpha_ranges = [(0.3, 0.1), (0.1, 0.1), (0.05, 0.025), (0.025, 0.0125)]

        for alpha_0, alpha_1 in alpha_ranges:
            print(
                f"\nTesting Particle Guidance with alpha_0={alpha_0}, alpha_1={alpha_1}"
            )
            pg_metrics = run_sampling_experiment(
                sampler,
                device,
                fid,
                args.num_samples,
                args.batch_size,
                batch_sample_particle_guidance_with_metrics,
                {
                    "alpha_0": alpha_0,  # HIGH guidance at t=0 (start, noise)
                    "alpha_1": alpha_1,  # LOW guidance at t=1 (end, data)
                    "kernel_type": "euclidean",
                    "schedule_type": "linear",
                    "sub_batch_size": 4,  # Match the sub_batch_size used in class label generation
                },
                f"Particle Guidance sampling with α₀={alpha_0} (t=0, high) → α₁={alpha_1} (t=1, low)",
            )

            results["experiments"].append(
                {
                    "type": "particle_guidance",
                    "alpha_0": alpha_0,
                    "alpha_1": alpha_1,
                    "kernel_type": "euclidean",
                    "schedule_type": "linear",
                    "metrics": pg_metrics,
                }
            )

            print(
                f"Particle Guidance (α₀={alpha_0}, α₁={alpha_1}) - FID: {pg_metrics['fid_score']:.4f}, IS: {pg_metrics['inception_score']:.4f}±{pg_metrics['inception_std']:.4f}, DINO Top-1: {pg_metrics['dino_top1_accuracy']:.2f}%, Top-5: {pg_metrics['dino_top5_accuracy']:.2f}%"
            )
            print(f"Average guidance/velocity ratio: {pg_metrics['avg_ratio']:.4f}")

    if "ode_baseline" in methods_to_run:
        print("\n\n===== Running baseline ODE experiment =====")
        ode_metrics = run_sampling_experiment(
            sampler,
            device,
            fid,
            args.num_samples,
            args.batch_size,
            batch_sample_ode_with_metrics,
            {},
            "regular ODE sampling",
        )
        results["ode_baseline"] = ode_metrics
        print(
            f"Baseline ODE - FID: {ode_metrics['fid_score']:.4f}, IS: {ode_metrics['inception_score']:.4f}±{ode_metrics['inception_std']:.4f}, DINO Top-1: {ode_metrics['dino_top1_accuracy']:.2f}%, Top-5: {ode_metrics['dino_top5_accuracy']:.2f}%"
        )

    if "ode_divfree" in methods_to_run:
        print("\n\n===== Running ODE-divfree experiments =====")
        for lambda_div in args.lambda_divs:
            print(f"\nTesting ODE-divfree with lambda_div={lambda_div}")
            divfree_metrics = run_sampling_experiment(
                sampler,
                device,
                fid,
                args.num_samples,
                args.batch_size,
                batch_sample_ode_divfree_with_metrics,
                {"lambda_div": lambda_div},
                f"ODE-divfree sampling with lambda_div={lambda_div}",
            )

            results["experiments"].append(
                {
                    "type": "ode_divfree",
                    "lambda_div": lambda_div,
                    "metrics": divfree_metrics,
                }
            )

            print(
                f"ODE-divfree (lambda_div={lambda_div}) - FID: {divfree_metrics['fid_score']:.4f}, IS: {divfree_metrics['inception_score']:.4f}±{divfree_metrics['inception_std']:.4f}, DINO Top-1: {divfree_metrics['dino_top1_accuracy']:.2f}%, Top-5: {divfree_metrics['dino_top5_accuracy']:.2f}%"
            )
            print(f"Average divfree/velocity ratio: {divfree_metrics['avg_ratio']:.4f}")

    if "ode_divfree_max" in methods_to_run:
        print("\n\n===== Running ODE-divfree-max experiments =====")
        for lambda_div in args.lambda_divs:
            print(f"\nTesting ODE-divfree-max with lambda_div={lambda_div}")
            divfree_max_metrics = run_sampling_experiment(
                sampler,
                device,
                fid,
                args.num_samples,
                args.batch_size,
                batch_sample_ode_divfree_max_with_metrics,
                {
                    "lambda_div": lambda_div,
                    "sub_batch_size": 4,
                    "repulsion_strength": 0.05,
                    "noise_schedule_end_factor": 0.1,
                },
                f"ODE-divfree-max sampling with lambda_div={lambda_div}",
            )

            results["experiments"].append(
                {
                    "type": "ode_divfree_max",
                    "lambda_div": lambda_div,
                    "metrics": divfree_max_metrics,
                }
            )

            print(
                f"ODE-divfree-max (lambda_div={lambda_div}) - FID: {divfree_max_metrics['fid_score']:.4f}, IS: {divfree_max_metrics['inception_score']:.4f}±{divfree_max_metrics['inception_std']:.4f}, DINO Top-1: {divfree_max_metrics['dino_top1_accuracy']:.2f}%, Top-5: {divfree_max_metrics['dino_top5_accuracy']:.2f}%"
            )
            print(
                f"Average divfree/velocity ratio: {divfree_max_metrics['avg_ratio']:.4f}"
            )

    if "sde" in methods_to_run:
        print("\n\n===== Running SDE experiments =====")
        for noise_scale in args.noise_scales:
            print(f"\nTesting SDE with noise_scale={noise_scale}")
            sde_metrics = run_sampling_experiment(
                sampler,
                device,
                fid,
                args.num_samples,
                args.batch_size,
                batch_sample_sde_with_metrics,
                {"noise_scale": noise_scale},
                f"SDE sampling with noise_scale={noise_scale}",
            )

            results["experiments"].append(
                {"type": "sde", "noise_scale": noise_scale, "metrics": sde_metrics}
            )

            print(
                f"SDE (noise_scale={noise_scale}) - FID: {sde_metrics['fid_score']:.4f}, IS: {sde_metrics['inception_score']:.4f}±{sde_metrics['inception_std']:.4f}, DINO Top-1: {sde_metrics['dino_top1_accuracy']:.2f}%, Top-5: {sde_metrics['dino_top5_accuracy']:.2f}%"
            )
            print(f"Average noise/velocity ratio: {sde_metrics['avg_ratio']:.4f}")

    if "sde_divfree" in methods_to_run:
        print("\n\n===== Running SDE-divfree experiments =====")
        for lambda_div in args.lambda_divs:
            print(f"\nTesting SDE-divfree with lambda_div={lambda_div}")
            sde_divfree_metrics = run_sampling_experiment(
                sampler,
                device,
                fid,
                args.num_samples,
                args.batch_size,
                batch_sample_sde_divfree_with_metrics,
                {"lambda_div": lambda_div},
                f"SDE-divfree sampling with lambda_div={lambda_div}",
            )

            results["experiments"].append(
                {
                    "type": "sde_divfree",
                    "lambda_div": lambda_div,
                    "metrics": sde_divfree_metrics,
                }
            )

            print(
                f"SDE-divfree (lambda_div={lambda_div}) - FID: {sde_divfree_metrics['fid_score']:.4f}, IS: {sde_divfree_metrics['inception_score']:.4f}±{sde_divfree_metrics['inception_std']:.4f}, DINO Top-1: {sde_divfree_metrics['dino_top1_accuracy']:.2f}%, Top-5: {sde_divfree_metrics['dino_top5_accuracy']:.2f}%"
            )
            print(
                f"Average divfree/velocity ratio: {sde_divfree_metrics['avg_ratio']:.4f}"
            )

    # Save results
    result_file = os.path.join(results_dir, f"noise_study_results_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to {result_file}")
    return results


def convert_to_serializable(obj):
    """
    Convert numpy float32s and other non-serializable types to Python native types.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow matching noise study")

    # Dataset and device parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet256",
        choices=["cifar10", "imagenet32", "imagenet256"],
        help="Dataset to use (cifar10, imagenet32, or imagenet256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation",
    )

    # Sample generation parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Number of samples to generate for each experiment",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for sample generation",
    )
    parser.add_argument(
        "--real_samples",
        type=int,
        default=50000,
        help="Number of real samples to use for FID calculation",
    )

    # Method selection parameters
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["ode_baseline", "ode_divfree_max", "ode_divfree"],
        choices=[
            "all",
            "ode_baseline",
            "particle_guidance",
            "ode_divfree",
            "ode_divfree_max",
            "sde",
            "sde_divfree",
        ],
        help="Methods to test (default: all methods)",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    # Noise parameters
    parser.add_argument(
        "--noise_scales",
        type=float,
        nargs="+",
        default=[0.1, 0.13, 0.16, 0.2],
        help="Noise scale values to test for SDE sampling",
    )
    parser.add_argument(
        "--lambda_divs",
        type=float,
        nargs="+",
        # default=[0.1, 0.35, 0.4, 0.45, 0.5, 0.6, 2.0],
        default=[0.1, 0.4, 0.5, 0.6],
        help="Lambda values for divergence-free flow to test",
    )
    parser.add_argument(
        "--beta_values",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.4],
        help="Beta values for EDM SDE sampling",
    )
    parser.add_argument(
        "--score_sde_factors",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3],
        help="Noise scale factors for Score SDE sampling",
    )

    args = parser.parse_args()

    # Run the experiments
    run_experiment(args)
