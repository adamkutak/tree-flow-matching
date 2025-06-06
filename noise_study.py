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


def score_vp_converted_corrected(x, alpha_t, sigma_t, d_alpha_t, d_sigma_t, u_t):
    """
    Convert velocity to score using repository's exact formula.
    Added numerical stability checks.
    """
    # Debug info (only for first call to avoid spam)
    if not hasattr(score_vp_converted_corrected, "debug_called"):
        print(f"\n=== Score Conversion Debug ===")
        print(f"Input shapes - x: {x.shape}, u_t: {u_t.shape}")
        print(f"VP coeffs before expansion - alpha_t: {alpha_t}, sigma_t: {sigma_t}")
        print(
            f"VP derivs before expansion - d_alpha_t: {d_alpha_t}, d_sigma_t: {d_sigma_t}"
        )
        score_vp_converted_corrected.debug_called = True

    # Expand dimensions to match x
    alpha_t = alpha_t.view(-1, *([1] * (x.ndim - 1)))
    sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))
    d_alpha_t = d_alpha_t.view(-1, *([1] * (x.ndim - 1)))
    d_sigma_t = d_sigma_t.view(-1, *([1] * (x.ndim - 1)))

    # Repository's exact formula with numerical stability
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t

    # Debug the intermediate calculations
    if not hasattr(score_vp_converted_corrected, "debug_intermediate"):
        print(f"After expansion - alpha_t shape: {alpha_t.shape}")
        print(
            f"reverse_alpha_ratio range: [{reverse_alpha_ratio.min():.4f}, {reverse_alpha_ratio.max():.4f}]"
        )
        print(f"var range before epsilon: [{var.min():.6f}, {var.max():.6f}]")

        # Check for problematic values in intermediate calculations
        if torch.isnan(reverse_alpha_ratio).any():
            print("WARNING: reverse_alpha_ratio has NaN!")
        if torch.isinf(reverse_alpha_ratio).any():
            print("WARNING: reverse_alpha_ratio has Inf!")
        if (var <= 0).any():
            print(f"WARNING: var has non-positive values! Min: {var.min():.6f}")

        score_vp_converted_corrected.debug_intermediate = True

    # Add small epsilon for numerical stability
    var = var + 1e-8

    score = (reverse_alpha_ratio * u_t - x) / var

    return score


def batch_sample_vp_sde_with_metrics(
    sampler, class_label, batch_size=16, beta_min=0.1, beta_max=20.0
):
    """
    VP-SDE conversion for flow matching models (0=noise, 1=clean).
    Key fix: Convert between flow matching and diffusion time notations!

    Flow matching: t=0 (noise) → t=1 (clean)
    Diffusion: t=1 (noise) → t=0 (clean)
    Conversion: t_diffusion = 1 - t_flow_matching
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

        # Your timesteps go 0→1 (noise→clean in flow matching notation)
        timesteps_fm = sampler.timesteps  # Flow matching notation

        print(f"\n=== VP-SDE Debug Info ===")
        print(f"Beta range: {beta_min} to {beta_max}")
        print(f"Flow matching timesteps: {timesteps_fm[:5]}...{timesteps_fm[-5:]}")
        print(f"Sample shape: {current_samples.shape}")

        for step, t_curr_fm in enumerate(timesteps_fm[:-1]):
            t_next_fm = timesteps_fm[step + 1]

            # dt in flow matching notation (positive, going 0→1)
            dt_fm = t_next_fm - t_curr_fm

            t_batch_fm = torch.full(
                (batch_size,), t_curr_fm.item(), device=sampler.device
            )

            # Get velocity from flow model (expects flow matching timesteps 0→1)
            velocity = sampler.flow_model(t_batch_fm, current_samples, current_label)

            # CRITICAL: Convert to diffusion notation for VP scheduler
            # Flow matching t=0 (noise) corresponds to diffusion t=1 (noise)
            # Flow matching t=1 (clean) corresponds to diffusion t=0 (clean)
            t_diffusion = 1.0 - t_curr_fm

            # Create VP scheduler coefficients using diffusion notation
            b = beta_min
            B = beta_max
            T = 0.5 * t_diffusion**2 * (B - b) + t_diffusion * b
            dT = t_diffusion * (B - b) + b

            alpha_t = torch.exp(-0.5 * T)
            sigma_t = torch.sqrt(1 - torch.exp(-T))
            d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T)
            d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T))

            # Debug prints for first few steps
            if step < 3:
                print(f"\nStep {step}:")
                print(f"  t_flow_matching: {t_curr_fm:.4f}")
                print(f"  t_diffusion: {t_diffusion:.4f}")
                print(f"  T: {T:.4f}")
                print(f"  alpha_t: {alpha_t:.4f}")
                print(f"  sigma_t: {sigma_t:.4f}")
                print(f"  d_alpha_t: {d_alpha_t:.4f}")
                print(f"  d_sigma_t: {d_sigma_t:.4f}")
                print(f"  velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")

                # Check for problematic values
                if torch.isnan(alpha_t) or torch.isinf(alpha_t):
                    print(f"  WARNING: alpha_t has NaN/Inf!")
                if torch.isnan(sigma_t) or torch.isinf(sigma_t):
                    print(f"  WARNING: sigma_t has NaN/Inf!")
                if torch.isnan(d_sigma_t) or torch.isinf(d_sigma_t):
                    print(f"  WARNING: d_sigma_t has NaN/Inf!")

            # Convert velocity to score using repository's formula
            score = score_vp_converted_corrected(
                current_samples, alpha_t, sigma_t, d_alpha_t, d_sigma_t, velocity
            )

            # Use sigma_t as diffusion coefficient
            diffuse = sigma_t

            # Calculate drift (repository formula)
            drift = -velocity + (0.5 * diffuse**2) * score

            # Noise term - use dt in flow matching notation
            noise = (
                torch.randn_like(current_samples)
                * diffuse
                * torch.sqrt(torch.abs(dt_fm))
            )

            # More debug info for first few steps
            if step < 3:
                print(f"  score range: [{score.min():.4f}, {score.max():.4f}]")
                print(f"  diffuse: {diffuse:.4f}")
                print(f"  drift range: [{drift.min():.4f}, {drift.max():.4f}]")
                print(f"  noise range: [{noise.min():.4f}, {noise.max():.4f}]")
                print(f"  dt_fm: {dt_fm:.4f}")

                # Check for problematic values
                if torch.isnan(score).any() or torch.isinf(score).any():
                    print(f"  WARNING: score has NaN/Inf!")
                if torch.isnan(drift).any() or torch.isinf(drift).any():
                    print(f"  WARNING: drift has NaN/Inf!")

            # Track magnitudes
            dims = tuple(range(1, velocity.ndim))
            vel_mag = torch.linalg.vector_norm(drift, dim=dims).mean().item()
            noise_mag = torch.linalg.vector_norm(noise, dim=dims).mean().item()

            velocity_magnitudes.append(vel_mag)
            noise_magnitudes.append(noise_mag)
            noise_to_velocity_ratios.append(noise_mag / vel_mag if vel_mag > 0 else 0)

            # Update samples - use dt in flow matching notation
            current_samples = current_samples + drift * dt_fm + noise

            # Debug sample evolution for first few steps
            if step < 3:
                print(
                    f"  sample range after update: [{current_samples.min():.4f}, {current_samples.max():.4f}]"
                )

        print(
            f"Final sample range: [{current_samples.min():.4f}, {current_samples.max():.4f}]"
        )
        print(f"=== VP-SDE Debug End ===\n")

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

    print("\n\n===== Running VP-SDE experiments =====")
    for beta_schedule in args.vp_sde_factors:
        print(f"\nTesting VP-SDE with beta_schedule={beta_schedule}")
        vp_sde_metrics = run_sampling_experiment(
            sampler,
            device,
            fid,
            args.num_samples,
            args.batch_size,
            batch_sample_vp_sde_with_metrics,
            {"beta_min": beta_schedule, "beta_max": beta_schedule},
            f"VP-SDE sampling with beta_schedule={beta_schedule}",
        )

        results["experiments"].append(
            {
                "type": "vp_sde",
                "beta_min": beta_schedule,
                "beta_max": beta_schedule,
                "metrics": vp_sde_metrics,
            }
        )

        print(
            f"VP-SDE (beta_min={beta_schedule}, beta_max={beta_schedule}) - FID: {vp_sde_metrics['fid_score']:.4f}, IS: {vp_sde_metrics['inception_score']:.4f}±{vp_sde_metrics['inception_std']:.4f}, DINO Top-1: {vp_sde_metrics['dino_top1_accuracy']:.2f}%, Top-5: {vp_sde_metrics['dino_top5_accuracy']:.2f}%"
        )
        print(f"Average noise/velocity ratio: {vp_sde_metrics['avg_ratio']:.4f}")

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

    parser.add_argument(
        "--vp_sde_factors",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
        help="Beta schedule values for VP-SDE sampling",
    )

    args = parser.parse_args()

    # Run the experiments
    run_experiment(args)


def test_vp_sde_debug():
    """
    Simple test function to debug VP-SDE implementation
    """
    import torch

    # Create a mock sampler with minimal required attributes
    class MockSampler:
        def __init__(self):
            self.device = torch.device("cpu")
            self.channels = 3
            self.image_size = 32
            self.timesteps = torch.linspace(0, 1, 5)  # Just 5 steps for debugging

        def flow_model(self, t, x, y):
            # Return a simple velocity field for testing
            return torch.randn_like(x) * 0.1

        def unnormalize_images(self, x):
            return x

    print("=== Testing VP-SDE Implementation ===")

    # Test with small batch size and simple parameters
    sampler = MockSampler()
    class_label = 0
    batch_size = 2
    beta_min = 0.1
    beta_max = 1.0  # Start with smaller range

    try:
        result = batch_sample_vp_sde_with_metrics(
            sampler, class_label, batch_size, beta_min, beta_max
        )
        print("VP-SDE test completed successfully!")
        print(f"Result shapes: {[type(r) for r in result]}")

    except Exception as e:
        print(f"VP-SDE test failed with error: {e}")
        import traceback

        traceback.print_exc()


# Uncomment the line below to run the test
# test_vp_sde_debug()
