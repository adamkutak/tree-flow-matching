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
from utils import divfree_swirl_si, score_si_linear


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

            # Track velocity magnitude
            dims = tuple(range(1, velocity.ndim))
            vel_mag = torch.linalg.vector_norm(velocity, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            # Add noise scaled by dt and noise_scale
            noise = torch.randn_like(current_samples) * torch.sqrt(dt) * noise_scale

            # Track noise magnitude
            noise_mag = torch.linalg.vector_norm(noise, dim=dims).mean().item()
            noise_magnitudes.append(noise_mag)

            # Track ratio
            ratio = noise_mag / vel_mag if vel_mag > 0 else 0
            noise_to_velocity_ratios.append(ratio)

            # Euler-Maruyama update
            current_samples = current_samples + velocity * dt + noise

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

            # Track magnitudes
            dims = tuple(range(1, u_t.ndim))
            vel_mag = torch.linalg.vector_norm(u_t, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            div_mag = torch.linalg.vector_norm(w, dim=dims).mean().item()
            divfree_magnitudes.append(div_mag)

            # Track ratio
            ratio = div_mag / vel_mag if vel_mag > 0 else 0
            divfree_to_velocity_ratios.append(ratio)

            x = x + (u_t + w) * dt  # Euler ODE step

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


def batch_sample_inference_time_sde_with_metrics(
    sampler, class_label, batch_size=16, noise_scale_factor=1.0
):
    """
    Inference-time SDE conversion from Section 4.2:
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


def batch_sample_vp_sde_with_metrics(
    sampler, class_label, batch_size=16, beta_min=0.1, beta_max=20.0
):
    """
    Complete VP-SDE sampler using proper velocity transformation.

    This implements the repository's approach:
    1. SNR matching between CondOT (flow matching) and VP schedulers
    2. Proper velocity field transformation
    3. VP-SDE sampling with transformed velocities

    Args:
        sampler: Your flow matching sampler with timesteps in [0,1] (0=noise, 1=clean)
        class_label: Class label for conditional sampling
        batch_size: Number of samples to generate
        beta_min, beta_max: VP scheduler parameters
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    # Scheduler implementations matching the repository
    class CondOTScheduler:
        """Flow matching scheduler: linear interpolation between noise and data"""

        def __call__(self, t):
            return {
                "alpha_t": 1 - t,
                "sigma_t": t,
                "d_alpha_t": -torch.ones_like(t),
                "d_sigma_t": torch.ones_like(t),
            }

        def snr_inverse(self, snr):
            # For CondOT: snr = alpha/sigma = (1-t)/t
            # Solving: snr = (1-t)/t  =>  t = 1/(1+snr)
            return 1.0 / (1.0 + snr)

    class VPScheduler:
        """Variance Preserving scheduler"""

        def __init__(self, beta_min=0.1, beta_max=20.0):
            self.beta_min = beta_min
            self.beta_max = beta_max

        def __call__(self, t):
            b = self.beta_min
            B = self.beta_max
            T = 0.5 * t**2 * (B - b) + t * b
            dT = t * (B - b) + b

            alpha_t = torch.exp(-0.5 * T)
            sigma_t = torch.sqrt(1 - torch.exp(-T))
            d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T)
            d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T))

            return {
                "alpha_t": alpha_t,
                "sigma_t": sigma_t,
                "d_alpha_t": d_alpha_t,
                "d_sigma_t": d_sigma_t,
            }

    original_scheduler = CondOTScheduler()
    vp_scheduler = VPScheduler(beta_min, beta_max)

    def compute_velocity_transform(x, t_fm, label):
        """
        Transform velocity from CondOT to VP scheduler.
        This implements the repository's compute_velocity_transform_scheduler logic.
        """
        # Convert flow matching time to VP time
        # Flow matching: t=0 (noise) → t=1 (clean)
        # VP diffusion: t=1 (noise) → t=0 (clean)
        t_vp = 1.0 - t_fm

        # Get VP scheduler coefficients
        vp_coeffs = vp_scheduler(t_vp)
        alpha_r = vp_coeffs["alpha_t"]
        sigma_r = vp_coeffs["sigma_t"]
        d_alpha_r = vp_coeffs["d_alpha_t"]
        d_sigma_r = vp_coeffs["d_sigma_t"]

        # Find equivalent CondOT time using SNR matching
        snr = alpha_r / sigma_r
        t_equiv = original_scheduler.snr_inverse(snr)

        # Get CondOT coefficients at equivalent time
        ot_coeffs = original_scheduler(t_equiv)
        alpha_t = ot_coeffs["alpha_t"]
        sigma_t = ot_coeffs["sigma_t"]
        d_alpha_t = ot_coeffs["d_alpha_t"]
        d_sigma_t = ot_coeffs["d_sigma_t"]

        # Compute scaling factor
        s_r = sigma_r / sigma_t

        # Compute time derivative transformation
        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )

        # Compute space derivative transformation
        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        # Debug first few calls
        if not hasattr(compute_velocity_transform, "debug_count"):
            compute_velocity_transform.debug_count = 0

        if compute_velocity_transform.debug_count < 3:
            print(
                f"\n=== Velocity Transform Debug (call {compute_velocity_transform.debug_count}) ==="
            )
            print(f"t_fm: {t_fm:.6f}, t_vp: {t_vp:.6f}, t_equiv: {t_equiv:.6f}")
            print(f"VP coeffs - alpha_r: {alpha_r:.6f}, sigma_r: {sigma_r:.6f}")
            print(f"CondOT coeffs - alpha_t: {alpha_t:.6f}, sigma_t: {sigma_t:.6f}")
            print(f"SNR: {snr:.6f}")
            print(
                f"VP derivatives - d_alpha_r: {d_alpha_r:.6f}, d_sigma_r: {d_sigma_r:.6f}"
            )
            print(
                f"CondOT derivatives - d_alpha_t: {d_alpha_t:.6f}, d_sigma_t: {d_sigma_t:.6f}"
            )

            # Debug dt_r calculation step by step
            numerator_dt = (
                sigma_t * sigma_t * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            )
            denominator_dt = (
                sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t)
            )
            print(
                f"dt_r numerator: {numerator_dt:.6f} = {sigma_t:.6f}^2 * ({sigma_r:.6f} * {d_alpha_r:.6f} - {alpha_r:.6f} * {d_sigma_r:.6f})"
            )
            print(
                f"dt_r denominator: {denominator_dt:.6f} = {sigma_r:.6f}^2 * ({sigma_t:.6f} * {d_alpha_t:.6f} - {alpha_t:.6f} * {d_sigma_t:.6f})"
            )
            print(f"dt_r = {numerator_dt:.6f} / {denominator_dt:.6f} = {dt_r:.6f}")

            # Debug ds_r calculation step by step
            numerator_ds = sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r
            denominator_ds = sigma_t * sigma_t
            print(
                f"ds_r numerator: {numerator_ds:.6f} = {sigma_t:.6f} * {d_sigma_r:.6f} - {sigma_r:.6f} * {d_sigma_t:.6f} * {dt_r:.6f}"
            )
            print(f"ds_r denominator: {denominator_ds:.6f} = {sigma_t:.6f}^2")
            print(f"ds_r = {numerator_ds:.6f} / {denominator_ds:.6f} = {ds_r:.6f}")

            print(
                f"Scaling factors - s_r: {s_r:.6f}, dt_r: {dt_r:.6f}, ds_r: {ds_r:.6f}"
            )

        # Call original model with scaled input at equivalent time
        t_batch = torch.full((x.shape[0],), t_equiv.item(), device=x.device)
        u_t = sampler.flow_model(t_batch, x / s_r, label)

        # Transform velocity to VP space
        u_r = ds_r * x / s_r + dt_r * s_r * u_t

        if compute_velocity_transform.debug_count < 3:
            print(f"Original velocity range: [{u_t.min():.6f}, {u_t.max():.6f}]")
            print(f"Transformed velocity range: [{u_r.min():.6f}, {u_r.max():.6f}]")
            print(
                f"Transform magnitude ratio: {torch.linalg.vector_norm(u_r) / torch.linalg.vector_norm(u_t):.6f}"
            )
            compute_velocity_transform.debug_count += 1

        return u_r, sigma_r

    # Initialize metrics tracking
    velocity_magnitudes = []
    noise_magnitudes = []
    noise_to_velocity_ratios = []

    with torch.no_grad():
        # Initialize samples with pure noise
        current_samples = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        # Prepare class labels
        if is_tensor:
            current_label = class_label
        else:
            current_label = torch.full(
                (batch_size,), class_label, device=sampler.device
            )

        # Get timesteps (flow matching notation: 0→1)
        timesteps_fm = sampler.timesteps

        # VP-SDE sampling loop
        for step, t_curr_fm in enumerate(timesteps_fm[:-1]):
            t_next_fm = timesteps_fm[step + 1]
            dt_fm = t_next_fm - t_curr_fm  # Time step size

            # Transform velocity from CondOT to VP
            transformed_velocity, diffusion_coeff = compute_velocity_transform(
                current_samples, t_curr_fm, current_label
            )

            # VP-SDE drift: just negative velocity (no additional score conversion needed)
            drift = -transformed_velocity

            # VP-SDE diffusion: use σ(t) from VP scheduler
            diffusion_coeff_expanded = diffusion_coeff.view(
                -1, *([1] * (current_samples.ndim - 1))
            )
            noise = (
                torch.randn_like(current_samples)
                * diffusion_coeff_expanded
                * torch.sqrt(torch.abs(dt_fm))
            )

            # Debug info for first few steps
            print(f"\nStep {step}, t_fm={t_curr_fm:.4f}")
            print(f"  diffusion_coeff: {diffusion_coeff:.6f}")
            print(f"  dt_fm: {dt_fm:.6f}")
            print(f"  sqrt(dt_fm): {torch.sqrt(torch.abs(dt_fm)):.6f}")
            print(
                f"  transformed_velocity range: [{transformed_velocity.min():.6f}, {transformed_velocity.max():.6f}]"
            )
            print(f"  drift range: [{drift.min():.6f}, {drift.max():.6f}]")
            print(
                f"  noise coeff: {diffusion_coeff * torch.sqrt(torch.abs(dt_fm)):.6f}"
            )
            print(f"  noise range: [{noise.min():.6f}, {noise.max():.6f}]")

            # Track magnitudes for monitoring
            dims = tuple(range(1, current_samples.ndim))
            vel_mag = torch.linalg.vector_norm(drift, dim=dims).mean().item()
            noise_mag = torch.linalg.vector_norm(noise, dim=dims).mean().item()

            velocity_magnitudes.append(vel_mag)
            noise_magnitudes.append(noise_mag)
            noise_to_velocity_ratios.append(noise_mag / vel_mag if vel_mag > 0 else 0)

            # Update samples using Euler-Maruyama
            current_samples = current_samples + drift * dt_fm + noise

        # Compute average metrics
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


def batch_sample_vp_ode_with_metrics(
    sampler, class_label, batch_size=16, beta_min=0.1, beta_max=20.0
):
    """
    Deterministic VP-ODE sampler using proper time conventions:
    - VP-SDE timesteps go 1→0 (diffusion convention)
    - Flow model called with 1-t_vp (converts to flow matching 0→1)
    - No confusing time conversions or backwards mappings
    """
    is_tensor = torch.is_tensor(class_label)
    sampler.flow_model.eval()

    # CondOT scheduler now uses diffusion convention (1→0) to match VP
    class CondOTScheduler:
        def __call__(self, t):
            # t goes from 1 (noise) to 0 (clean) in diffusion convention
            return {
                "alpha_t": t,  # t=1: alpha=1 (noise), t=0: alpha=0 (clean)
                "sigma_t": 1 - t,  # t=1: sigma=0 (noise), t=0: sigma=1 (clean)
                "d_alpha_t": torch.ones_like(t),
                "d_sigma_t": -torch.ones_like(t),
            }

        def snr_inverse(self, snr):
            # For diffusion CondOT: snr = alpha/sigma = t/(1-t)
            # Solving: snr = t/(1-t) => t = snr/(1+snr)
            return snr / (1.0 + snr)

    class VPScheduler:
        def __init__(self, beta_min=0.1, beta_max=20.0):
            self.beta_min = beta_min
            self.beta_max = beta_max

        def __call__(self, t):
            # t goes from 1 (noise) to 0 (clean) - standard diffusion convention
            b = self.beta_min
            B = self.beta_max
            T = 0.5 * t**2 * (B - b) + t * b
            dT = t * (B - b) + b

            alpha_t = torch.exp(-0.5 * T)
            sigma_t = torch.sqrt(1 - torch.exp(-T))
            d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T)
            d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T))

            return {
                "alpha_t": alpha_t,
                "sigma_t": sigma_t,
                "d_alpha_t": d_alpha_t,
                "d_sigma_t": d_sigma_t,
            }

    original_scheduler = CondOTScheduler()
    vp_scheduler = VPScheduler(beta_min, beta_max)

    def compute_velocity_transform(x, t_vp, label):
        # t_vp is in diffusion convention (1→0)
        vp_coeffs = vp_scheduler(t_vp)
        alpha_r = vp_coeffs["alpha_t"]
        sigma_r = vp_coeffs["sigma_t"]
        d_alpha_r = vp_coeffs["d_alpha_t"]
        d_sigma_r = vp_coeffs["d_sigma_t"]

        # Find equivalent CondOT time using SNR matching
        snr = alpha_r / sigma_r
        t_equiv = original_scheduler.snr_inverse(snr)

        ot_coeffs = original_scheduler(t_equiv)
        alpha_t = ot_coeffs["alpha_t"]
        sigma_t = ot_coeffs["sigma_t"]
        d_alpha_t = ot_coeffs["d_alpha_t"]
        d_sigma_t = ot_coeffs["d_sigma_t"]

        s_r = sigma_r / sigma_t
        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )
        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        # Debug output for every step
        if not hasattr(compute_velocity_transform, "debug_count"):
            compute_velocity_transform.debug_count = 0

        step = compute_velocity_transform.debug_count
        print(f"\n=== VP-ODE Transform Debug Step {step} ===")
        print(f"t_vp (diffusion): {t_vp:.6f} -> t_equiv (diffusion): {t_equiv:.6f}")
        print(f"VP coeffs: alpha_r={alpha_r:.6f}, sigma_r={sigma_r:.6f}, SNR={snr:.6f}")
        print(f"CondOT coeffs: alpha_t={alpha_t:.6f}, sigma_t={sigma_t:.6f}")
        print(f"Scaling: s_r={s_r:.6f}, dt_r={dt_r:.6f}, ds_r={ds_r:.6f}")

        # Convert to flow matching convention ONLY when calling the model
        t_flow_matching = 1.0 - t_equiv  # Convert diffusion→flow matching
        t_batch = torch.full((x.shape[0],), t_flow_matching, device=x.device)

        # Debug input to model
        x_scaled = x / s_r
        print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Scaled input range: [{x_scaled.min():.4f}, {x_scaled.max():.4f}]")
        print(
            f"Using flow model time: {t_flow_matching:.6f} (converted from diffusion {t_equiv:.6f})"
        )

        u_t = sampler.flow_model(t_batch, x_scaled, label)
        u_r = ds_r * x / s_r + dt_r * s_r * u_t

        print(f"Original velocity range: [{u_t.min():.6f}, {u_t.max():.6f}]")
        print(f"Transformed velocity range: [{u_r.min():.6f}, {u_r.max():.6f}]")
        print(
            f"Velocity magnitude ratio: {torch.linalg.vector_norm(u_r) / torch.linalg.vector_norm(u_t):.6f}"
        )

        # Check if transformation is reasonable
        if torch.linalg.vector_norm(u_r) / torch.linalg.vector_norm(u_t) < 0.01:
            print("WARNING: Velocity magnitude reduced by >99%!")
        if torch.linalg.vector_norm(u_r) / torch.linalg.vector_norm(u_t) > 100:
            print("WARNING: Velocity magnitude increased by >100x!")

        compute_velocity_transform.debug_count += 1

        return u_r

    # Initialize metrics tracking
    velocity_magnitudes = []

    with torch.no_grad():
        # Initialize samples with pure noise
        current_samples = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )

        # Prepare class labels
        if is_tensor:
            current_label = class_label
        else:
            current_label = torch.full(
                (batch_size,), class_label, device=sampler.device
            )

        # Create diffusion timesteps: 1→0 (1=noise, 0=clean)
        timesteps_diffusion = torch.linspace(
            1.0, 0.0, len(sampler.timesteps), device=sampler.device
        )

        # VP-ODE sampling loop using diffusion convention
        for step, t_curr_vp in enumerate(timesteps_diffusion[:-1]):
            t_next_vp = timesteps_diffusion[step + 1]
            dt_vp = t_next_vp - t_curr_vp  # Negative dt (going 1→0)

            # Debug sample evolution
            sample_norm_before = (
                torch.linalg.vector_norm(current_samples, dim=(1, 2, 3)).mean().item()
            )
            print(
                f"\n--- VP-ODE Step {step}: t={t_curr_vp:.4f} -> {t_next_vp:.4f} (dt={dt_vp:.4f}) ---"
            )
            print(f"Sample norm before: {sample_norm_before:.4f}")

            # Transform velocity using diffusion time
            transformed_velocity = compute_velocity_transform(
                current_samples, t_curr_vp, current_label
            )

            # VP-ODE drift: just negative velocity
            drift = -transformed_velocity

            # Track velocity magnitude
            dims = tuple(range(1, current_samples.ndim))
            vel_mag = torch.linalg.vector_norm(drift, dim=dims).mean().item()
            velocity_magnitudes.append(vel_mag)

            print(f"Drift magnitude: {vel_mag:.4f}")
            print(f"Drift range: [{drift.min():.4f}, {drift.max():.4f}]")

            # DETERMINISTIC UPDATE: Only drift term, NO noise!
            current_samples = current_samples + drift * dt_vp

            # Debug sample evolution after update
            sample_norm_after = (
                torch.linalg.vector_norm(current_samples, dim=(1, 2, 3)).mean().item()
            )
            print(f"Sample norm after: {sample_norm_after:.4f}")
            print(f"Sample change: {sample_norm_after - sample_norm_before:.4f}")

            # Check for problematic behavior
            if torch.any(torch.isnan(current_samples)):
                print("ERROR: NaN detected in samples!")
                break
            if torch.any(torch.isinf(current_samples)):
                print("ERROR: Inf detected in samples!")
                break
            if sample_norm_after > 100:
                print("WARNING: Samples are exploding!")
            if abs(sample_norm_after - sample_norm_before) > 10:
                print("WARNING: Large sample change!")

            # Only show first few steps to avoid spam
            if step >= 5:
                # Reset debug counter to stop per-step debugging
                if hasattr(compute_velocity_transform, "debug_count"):
                    compute_velocity_transform.debug_count = (
                        1000  # Stop detailed debugging
                    )

        avg_velocity_magnitude = sum(velocity_magnitudes) / len(velocity_magnitudes)

        return (
            sampler.unnormalize_images(current_samples),
            avg_velocity_magnitude,
            0.0,  # No noise magnitude
            0.0,  # No noise-to-velocity ratio
            velocity_magnitudes,
            [0.0] * len(velocity_magnitudes),  # No noise magnitudes
            [0.0] * len(velocity_magnitudes),  # No noise ratios
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

        class_labels = torch.randint(
            0, sampler.num_classes, (current_batch_size,), device=device
        )
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
        **all_metrics,
        **sampling_params,
    }

    return metrics


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

    vp_sde_configs = [
        (0.0001, 0.02),  # Stable Diffusion standard
        (0.0001, 0.01),  # More conservative
        (0.0001, 0.03),  # Slightly more aggressive
        (0.0005, 0.02),  # Higher start, same end
        (0.0001, 0.015),  # In between
    ]

    print("\n\n===== Running VP-ODE experiments (deterministic) =====")
    for beta_min, beta_max in vp_sde_configs:
        print(
            f"\nTesting VP-ODE (deterministic) with beta_min={beta_min}, beta_max={beta_max}"
        )
        vp_ode_metrics = run_sampling_experiment(
            sampler,
            device,
            fid,
            args.num_samples,
            args.batch_size,
            batch_sample_vp_ode_with_metrics,
            {"beta_min": beta_min, "beta_max": beta_max},
            f"VP-ODE sampling with beta_min={beta_min}, beta_max={beta_max}",
        )

        results["experiments"].append(
            {
                "type": "vp_ode",
                "beta_min": beta_min,
                "beta_max": beta_max,
                "metrics": vp_ode_metrics,
            }
        )

        print(
            f"VP-ODE (beta_min={beta_min}, beta_max={beta_max}) - FID: {vp_ode_metrics['fid_score']:.4f}, IS: {vp_ode_metrics['inception_score']:.4f}±{vp_ode_metrics['inception_std']:.4f}, DINO Top-1: {vp_ode_metrics['dino_top1_accuracy']:.2f}%, Top-5: {vp_ode_metrics['dino_top5_accuracy']:.2f}%"
        )
        print(f"Velocity magnitude: {vp_ode_metrics['avg_velocity_magnitude']:.4f}")

    print("\n\n===== Running VP-SDE experiments =====")
    for beta_min, beta_max in vp_sde_configs:
        print(f"\nTesting VP-SDE with beta_min={beta_min}, beta_max={beta_max}")
        vp_sde_metrics = run_sampling_experiment(
            sampler,
            device,
            fid,
            args.num_samples,
            args.batch_size,
            batch_sample_vp_sde_with_metrics,
            {"beta_min": beta_min, "beta_max": beta_max},
            f"VP-SDE sampling with beta_min={beta_min}, beta_max={beta_max}",
        )

        results["experiments"].append(
            {
                "type": "vp_sde",
                "beta_min": beta_min,
                "beta_max": beta_max,
                "metrics": vp_sde_metrics,
            }
        )

        print(
            f"VP-SDE (beta_min={beta_min}, beta_max={beta_max}) - FID: {vp_sde_metrics['fid_score']:.4f}, IS: {vp_sde_metrics['inception_score']:.4f}±{vp_sde_metrics['inception_std']:.4f}, DINO Top-1: {vp_sde_metrics['dino_top1_accuracy']:.2f}%, Top-5: {vp_sde_metrics['dino_top5_accuracy']:.2f}%"
        )
        print(f"Average noise/velocity ratio: {vp_sde_metrics['avg_ratio']:.4f}")

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

    print("\n\n===== Running EDM SDE experiments =====")
    for beta in args.beta_values:
        print(f"\nTesting EDM SDE with beta={beta}")
        edm_sde_metrics = run_sampling_experiment(
            sampler,
            device,
            fid,
            args.num_samples,
            args.batch_size,
            batch_sample_edm_sde_with_metrics,
            {"beta": beta},
            f"EDM SDE sampling with beta={beta}",
        )

        results["experiments"].append(
            {"type": "edm_sde", "beta": beta, "metrics": edm_sde_metrics}
        )

        print(
            f"EDM SDE (beta={beta}) - FID: {edm_sde_metrics['fid_score']:.4f}, IS: {edm_sde_metrics['inception_score']:.4f}±{edm_sde_metrics['inception_std']:.4f}, DINO Top-1: {edm_sde_metrics['dino_top1_accuracy']:.2f}%, Top-5: {edm_sde_metrics['dino_top5_accuracy']:.2f}%"
        )
        print(f"Average noise/velocity ratio: {edm_sde_metrics['avg_ratio']:.4f}")

    print("\n\n===== Running score SDE experiments =====")
    for noise_scale_factor in args.inference_sde_factors:
        print(
            f"\nTesting Inference-Time SDE with noise_scale_factor={noise_scale_factor}"
        )
        inference_sde_metrics = run_sampling_experiment(
            sampler,
            device,
            fid,
            args.num_samples,
            args.batch_size,
            batch_sample_inference_time_sde_with_metrics,
            {"noise_scale_factor": noise_scale_factor},
            f"Inference-Time SDE sampling with noise_scale_factor={noise_scale_factor}",
        )

        results["experiments"].append(
            {
                "type": "inference_time_sde",
                "noise_scale_factor": noise_scale_factor,
                "metrics": inference_sde_metrics,
            }
        )

        print(
            f"Inference-Time SDE (factor={noise_scale_factor}) - FID: {inference_sde_metrics['fid_score']:.4f}, IS: {inference_sde_metrics['inception_score']:.4f}±{inference_sde_metrics['inception_std']:.4f}, DINO Top-1: {inference_sde_metrics['dino_top1_accuracy']:.2f}%, Top-5: {inference_sde_metrics['dino_top5_accuracy']:.2f}%"
        )
        print(f"Average noise/velocity ratio: {inference_sde_metrics['avg_ratio']:.4f}")

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
        default=128,
        help="Number of samples to generate for each experiment",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for sample generation",
    )
    parser.add_argument(
        "--real_samples",
        type=int,
        default=100,
        help="Number of real samples to use for FID calculation",
    )

    # Noise parameters
    parser.add_argument(
        "--noise_scales",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.6, 1.0],
        help="Noise scale values to test for SDE sampling",
    )
    parser.add_argument(
        "--lambda_divs",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.6, 1.0, 2.0],
        help="Lambda values for divergence-free flow to test",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--beta_values",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4, 0.6],
        help="Beta values for EDM SDE sampling",
    )

    parser.add_argument(
        "--inference_sde_factors",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.6, 1.0],
        help="Noise scale factors for inference-time SDE sampling",
    )

    args = parser.parse_args()

    # Run the experiments
    run_experiment(args)
