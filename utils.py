import torch
import numpy as np


def score_si_linear(x, t_batch, u_t):
    t_scalar = t_batch[0].item()

    if t_scalar == 0.0:
        # limit t → 0  gives score = −x
        return -x
    else:
        one_minus_t = 1.0 - t_batch.view(-1, *([1] * (x.ndim - 1)))
        t = t_batch.view(-1, *([1] * (x.ndim - 1)))
        return -((one_minus_t * u_t + x) / t)


def make_divergence_free(noise, x, t_batch, u_t, eps=1e-8):
    """
    Project noise to be divergence-free using score projection method.
    """
    score = score_si_linear(x, t_batch, u_t)

    # Project out component parallel to score
    dims = tuple(range(1, len(noise.shape)))  # All dims except batch
    dot = (noise * score).sum(dim=dims, keepdim=True)
    s_norm2 = torch.linalg.vector_norm(score, dim=dims, keepdim=True).pow(2) + eps
    proj = dot / s_norm2

    divergence_free_noise = noise - proj * score

    return divergence_free_noise


def divfree_swirl_si(x, t_batch, y, u_t, eps=1e-8):
    """Generate divergence-free noise by projecting Gaussian noise."""
    eps_raw = torch.randn_like(x)
    return make_divergence_free(eps_raw, x, t_batch, u_t, eps)


def particle_guidance_forces(x_batch, t, alpha_t=1.0, kernel_type="euclidean"):
    """
    Extract repulsive forces from particle guidance potential.
    Only operates within same-class batches.

    Args:
        x_batch: [batch_size, channels, height, width] - current samples at time t
        t: current time step (scalar)
        alpha_t: guidance strength (time-dependent)
        kernel_type: 'rbf' or 'euclidean'

    Returns:
        forces: [batch_size, channels, height, width] - repulsive forces for each sample
    """
    batch_size = x_batch.shape[0]
    if batch_size == 1:
        return torch.zeros_like(x_batch)

    # Flatten spatial dimensions for distance computation
    x_flat = x_batch.flatten(1)  # [batch_size, flattened_dims]

    if kernel_type == "rbf":
        forces = _rbf_repulsive_forces(x_batch, x_flat, alpha_t)
    elif kernel_type == "euclidean":
        forces = _euclidean_repulsive_forces(x_batch, x_flat, alpha_t)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return forces


def _rbf_repulsive_forces(x_batch, x_flat, alpha_t):
    """RBF kernel implementation from particle guidance paper - vectorized."""
    batch_size = x_batch.shape[0]

    # Compute pairwise distances
    distances = torch.cdist(x_flat, x_flat)  # [batch_size, batch_size]

    # RBF kernel bandwidth (median heuristic)
    with torch.no_grad():
        median_dist = torch.median(distances[distances > 0])
        h_t = median_dist**2 / torch.log(
            torch.tensor(float(batch_size), device=x_batch.device)
        )

    # # DEBUG: Print RBF kernel information
    # print(f"  RBF Debug - batch_size: {batch_size}")
    # print(
    #     f"  Distances - min: {distances.min():.6f}, max: {distances.max():.6f}, median: {median_dist:.6f}"
    # )
    # print(f"  Bandwidth h_t: {h_t:.6f}")

    # Compute RBF kernel values k(xᵢ, xⱼ) = exp(-||xᵢ - xⱼ||² / h_t)
    kernel_values = torch.exp(-(distances**2) / h_t)  # [batch_size, batch_size]
    # print(
    #     f"  Kernel values - min: {kernel_values.min():.6f}, max: {kernel_values.max():.6f}"
    # )
    # print(f"  Kernel diagonal should be 1.0: {kernel_values.diag().mean():.6f}")

    # Vectorized computation of pairwise differences
    # x_flat: [batch_size, features] -> [batch_size, 1, features] - [1, batch_size, features]
    diff = x_flat.unsqueeze(1) - x_flat.unsqueeze(
        0
    )  # [batch_size, batch_size, features]

    # Compute gradients: ∇ᵢ k(xᵢ, xⱼ) = -2(xᵢ - xⱼ) * k(xᵢ, xⱼ) / h_t
    # kernel_values: [batch_size, batch_size] -> [batch_size, batch_size, 1]
    kernel_grad = -2 * diff * kernel_values.unsqueeze(2) / h_t

    # Sum over j for each i, excluding diagonal (i=j)
    # Create mask to exclude diagonal elements
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=x_batch.device).unsqueeze(2)
    kernel_grad = kernel_grad * mask

    # Sum forces for each particle
    forces_flat = kernel_grad.sum(dim=1)  # [batch_size, features]

    # Reshape back to original dimensions and apply guidance strength
    forces = forces_flat.view_as(x_batch)

    # # DEBUG: Print force statistics before alpha scaling
    # print(f"  Forces before alpha - min: {forces.min():.6f}, max: {forces.max():.6f}")
    # print(f"  Forces before alpha - mean: {forces.mean():.6f}, std: {forces.std():.6f}")
    # print(f"  Alpha_t being applied: {alpha_t:.6f}")

    final_forces = -alpha_t * forces  # Negative for repulsion
    # print(
    #     f"  Final forces - min: {final_forces.min():.6f}, max: {final_forces.max():.6f}"
    # )

    return final_forces


def _euclidean_repulsive_forces(x_batch, x_flat, alpha_t):
    """Simple Euclidean distance repulsion - vectorized."""
    batch_size = x_batch.shape[0]

    # Vectorized computation of pairwise differences
    # x_flat: [batch_size, features] -> [batch_size, 1, features] - [1, batch_size, features]
    diff = x_flat.unsqueeze(1) - x_flat.unsqueeze(
        0
    )  # [batch_size, batch_size, features]

    # Compute pairwise distances
    distances = (
        torch.norm(diff, dim=2, keepdim=True) + 1e-6
    )  # [batch_size, batch_size, 1] - larger epsilon for identical starts

    # Force proportional to 1/distance³ (since force = diff/distance³)
    forces = diff / (distances**3)  # [batch_size, batch_size, features]

    # Exclude diagonal elements (i=j) by setting them to zero
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=x_batch.device).unsqueeze(2)
    forces = forces * mask

    # Sum forces for each particle
    forces_flat = forces.sum(dim=1)  # [batch_size, features]

    forces = forces_flat.view_as(x_batch)

    return alpha_t * forces


def get_alpha_schedule_flow_matching(
    t, alpha_0=2.0, alpha_1=0.1, schedule_type="linear"
):
    if schedule_type == "linear":
        return alpha_0 + (alpha_1 - alpha_0) * t
    elif schedule_type == "exponential":
        return torch.exp(
            t * torch.log(torch.tensor(alpha_1))
            + (1 - t) * torch.log(torch.tensor(alpha_0))
        )
    else:
        return alpha_0  # constant


def divergence_free_particle_guidance(
    x_batch, t_batch, y, u_t, alpha_t=1.0, kernel_type="euclidean"
):
    """
    Combine particle guidance with divergence-free constraint.
    Forces are automatically regularized to match velocity field magnitude.

    Args:
        x_batch: current samples
        t_batch: time batch
        y: conditioning (if any)
        u_t: velocity field from flow model
        alpha_t: particle guidance strength (relative to velocity magnitude)
        kernel_type: kernel for repulsion

    Returns:
        divergence_free_repulsion: clean repulsive forces that preserve continuity equation
    """
    # Get raw repulsive forces from particle guidance (without alpha scaling)
    t_scalar = t_batch[0].item() if torch.is_tensor(t_batch) else t_batch
    raw_repulsive_forces = particle_guidance_forces(x_batch, t_scalar, 1.0, kernel_type)

    # Regularize forces to match velocity field magnitude
    dims = tuple(range(1, u_t.ndim))
    velocity_magnitude = torch.linalg.vector_norm(u_t, dim=dims).mean()
    force_magnitude = torch.linalg.vector_norm(raw_repulsive_forces, dim=dims).mean()

    # # DEBUG: Print detailed information about the forces
    # print(f"\n=== PARTICLE GUIDANCE DEBUG (t={t_scalar:.3f}) ===")
    # print(f"Batch shape: {x_batch.shape}")
    # print(f"Velocity magnitude: {velocity_magnitude:.6f}")
    # print(f"Raw force magnitude: {force_magnitude:.6f}")

    # # # Check for NaN/inf values
    # if torch.isnan(raw_repulsive_forces).any():
    #     print("WARNING: NaN values in raw_repulsive_forces!")
    # if torch.isinf(raw_repulsive_forces).any():
    #     print("WARNING: Inf values in raw_repulsive_forces!")

    # # Print some statistics about the raw forces
    # print(
    #     f"Raw forces - min: {raw_repulsive_forces.min():.6f}, max: {raw_repulsive_forces.max():.6f}"
    # )
    # print(
    #     f"Raw forces - mean: {raw_repulsive_forces.mean():.6f}, std: {raw_repulsive_forces.std():.6f}"
    # )

    if force_magnitude > 1e-8:  # Avoid division by zero
        # Scale forces so their average magnitude equals velocity magnitude
        regularization_factor = velocity_magnitude / force_magnitude
        regularized_forces = raw_repulsive_forces * regularization_factor
        # print(f"Regularization factor: {regularization_factor:.6f}")
    else:
        regularized_forces = raw_repulsive_forces
        # print("WARNING: Force magnitude too small, no regularization applied!")

    # Apply user-specified alpha scaling AFTER regularization
    scaled_forces = regularized_forces * alpha_t
    scaled_magnitude = torch.linalg.vector_norm(scaled_forces, dim=dims).mean()
    # print(f"Alpha: {alpha_t:.3f}")
    # print(f"Scaled force magnitude: {scaled_magnitude:.6f}")
    # print(
    #     f"Velocity/scaled_force ratio: {velocity_magnitude/scaled_magnitude:.3f}"
    #     if scaled_magnitude > 1e-8
    #     else "inf"
    # )

    # Apply divergence-free projection
    divergence_free_repulsion = make_divergence_free(
        scaled_forces, x_batch, t_batch, u_t
    )

    # # DEBUG: Compare before and after divergence-free projection
    divfree_magnitude = torch.linalg.vector_norm(
        divergence_free_repulsion, dim=dims
    ).mean()
    # print(f"Divergence-free force magnitude: {divfree_magnitude:.6f}")
    # print(
    #     f"Projection preservation ratio: {divfree_magnitude/scaled_magnitude:.3f}"
    #     if scaled_magnitude > 1e-8
    #     else "inf"
    # )
    # print("=" * 50)

    return divergence_free_repulsion
