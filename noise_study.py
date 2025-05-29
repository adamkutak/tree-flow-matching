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

            brownian = (
                torch.randn_like(current_samples)
                * torch.sqrt(dt)
                * torch.sqrt(torch.tensor(2.0 * beta, device=sampler.device))
                * sigma_t
            )

            dims = tuple(range(1, velocity.ndim))
            vel_mag = torch.linalg.vector_norm(velocity, dim=dims).mean().item()
            noise_mag = torch.linalg.vector_norm(brownian, dim=dims).mean().item()

            velocity_magnitudes.append(vel_mag)
            noise_magnitudes.append(noise_mag)
            noise_to_velocity_ratios.append(noise_mag / vel_mag if vel_mag > 0 else 0)

            current_samples = current_samples + (velocity + drift_corr) * dt + brownian

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
    ode_metrics = run_ode_experiment(
        sampler, device, fid, args.num_samples, args.batch_size
    )
    results["ode_baseline"] = ode_metrics
    print(
        f"Baseline ODE - FID: {ode_metrics['fid_score']:.4f}, IS: {ode_metrics['inception_score']:.4f}±{ode_metrics['inception_std']:.4f}, DINO Top-1: {ode_metrics['dino_top1_accuracy']:.2f}%, Top-5: {ode_metrics['dino_top5_accuracy']:.2f}%"
    )

    # Run SDE experiments
    print("\n\n===== Running SDE experiments =====")
    for noise_scale in args.noise_scales:
        print(f"\nTesting SDE with noise_scale={noise_scale}")
        sde_metrics = run_sde_experiment(
            sampler, device, fid, args.num_samples, args.batch_size, noise_scale
        )

        results["experiments"].append(
            {"type": "sde", "noise_scale": noise_scale, "metrics": sde_metrics}
        )

        print(
            f"SDE (noise_scale={noise_scale}) - FID: {sde_metrics['fid_score']:.4f}, IS: {sde_metrics['inception_score']:.4f}±{sde_metrics['inception_std']:.4f}, DINO Top-1: {sde_metrics['dino_top1_accuracy']:.2f}%, Top-5: {sde_metrics['dino_top5_accuracy']:.2f}%"
        )
        print(
            f"Average noise/velocity ratio: {sde_metrics['avg_noise_to_velocity_ratio']:.4f}"
        )

    # Run ODE-divfree experiments
    print("\n\n===== Running ODE-divfree experiments =====")
    for lambda_div in args.lambda_divs:
        print(f"\nTesting ODE-divfree with lambda_div={lambda_div}")
        divfree_metrics = run_ode_divfree_experiment(
            sampler, device, fid, args.num_samples, args.batch_size, lambda_div
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
        print(
            f"Average divfree/velocity ratio: {divfree_metrics['avg_divfree_to_velocity_ratio']:.4f}"
        )

    # Run EDM SDE experiments
    print("\n\n===== Running EDM SDE experiments =====")
    for beta in args.beta_values:
        print(f"\nTesting EDM SDE with beta={beta}")
        edm_sde_metrics = run_edm_sde_experiment(
            sampler, device, fid, args.num_samples, args.batch_size, beta
        )

        results["experiments"].append(
            {"type": "edm_sde", "beta": beta, "metrics": edm_sde_metrics}
        )

        print(
            f"EDM SDE (beta={beta}) - FID: {edm_sde_metrics['fid_score']:.4f}, IS: {edm_sde_metrics['inception_score']:.4f}±{edm_sde_metrics['inception_std']:.4f}, DINO Top-1: {edm_sde_metrics['dino_top1_accuracy']:.2f}%, Top-5: {edm_sde_metrics['dino_top5_accuracy']:.2f}%"
        )
        print(
            f"Average noise/velocity ratio: {edm_sde_metrics['avg_noise_to_velocity_ratio']:.4f}"
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


def run_ode_experiment(sampler, device, fid, n_samples, batch_size):
    """Run baseline ODE (regular flow matching) experiment."""
    fid.reset()

    generated_samples = []
    total_velocity_magnitude = 0.0
    velocity_magnitudes_all = []
    class_labels_all = []

    # Calculate number of batches
    num_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        num_batches += 1

    print(f"Generating {n_samples} samples using regular ODE sampling...")

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, n_samples - i * batch_size)

        # Random class labels
        class_labels = torch.randint(
            0, sampler.num_classes, (current_batch_size,), device=device
        )
        class_labels_all.append(class_labels.cpu())

        # Generate samples
        samples, avg_velocity, velocity_magnitudes = batch_sample_ode_with_metrics(
            sampler, class_labels, current_batch_size
        )

        # Update metrics
        generated_samples.extend(samples.cpu())
        total_velocity_magnitude += avg_velocity * current_batch_size
        velocity_magnitudes_all.extend(velocity_magnitudes)

        # Update FID
        fid.update(samples.to(device), real=False)

    # Compute average velocity magnitude
    avg_velocity_magnitude = total_velocity_magnitude / n_samples

    # Compute FID
    fid_score = fid.compute().item()

    # Stack generated samples and class labels
    generated_tensor = torch.stack(generated_samples).to(device)
    class_labels_tensor = torch.cat(class_labels_all).to(device)

    # Compute Inception Score
    inception_score, inception_std = calculate_inception_score(
        generated_tensor, device=device, batch_size=64, splits=10
    )

    # Compute DINO accuracy
    dino_accuracy = compute_dino_accuracy(
        sampler, generated_tensor, class_labels_tensor, batch_size=64
    )

    # Return metrics
    metrics = {
        "fid_score": fid_score,
        "inception_score": inception_score,
        "inception_std": inception_std,
        "avg_velocity_magnitude": avg_velocity_magnitude,
        "velocity_magnitudes": velocity_magnitudes_all,
        "dino_top1_accuracy": dino_accuracy["top1_accuracy"],
        "dino_top5_accuracy": dino_accuracy["top5_accuracy"],
    }

    return metrics


def run_sde_experiment(sampler, device, fid, n_samples, batch_size, noise_scale):
    """Run SDE experiment with the given noise scale."""
    fid.reset()

    generated_samples = []
    total_velocity_magnitude = 0.0
    total_noise_magnitude = 0.0
    total_noise_to_velocity_ratio = 0.0
    velocity_magnitudes_all = []
    noise_magnitudes_all = []
    noise_to_velocity_ratios_all = []
    class_labels_all = []

    # Calculate number of batches
    num_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        num_batches += 1

    print(
        f"Generating {n_samples} samples using SDE sampling with noise_scale={noise_scale}..."
    )

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, n_samples - i * batch_size)

        # Random class labels
        class_labels = torch.randint(
            0, sampler.num_classes, (current_batch_size,), device=device
        )
        class_labels_all.append(class_labels.cpu())

        # Generate samples
        (
            samples,
            avg_velocity,
            avg_noise,
            avg_ratio,
            velocity_magnitudes,
            noise_magnitudes,
            noise_to_velocity_ratios,
        ) = batch_sample_sde_with_metrics(
            sampler, class_labels, current_batch_size, noise_scale
        )

        # Update metrics
        generated_samples.extend(samples.cpu())
        total_velocity_magnitude += avg_velocity * current_batch_size
        total_noise_magnitude += avg_noise * current_batch_size
        total_noise_to_velocity_ratio += avg_ratio * current_batch_size
        velocity_magnitudes_all.extend(velocity_magnitudes)
        noise_magnitudes_all.extend(noise_magnitudes)
        noise_to_velocity_ratios_all.extend(noise_to_velocity_ratios)

        # Update FID
        fid.update(samples.to(device), real=False)

    # Compute averages
    avg_velocity_magnitude = total_velocity_magnitude / n_samples
    avg_noise_magnitude = total_noise_magnitude / n_samples
    avg_noise_to_velocity_ratio = total_noise_to_velocity_ratio / n_samples

    # Compute FID
    fid_score = fid.compute().item()

    # Stack generated samples and class labels
    generated_tensor = torch.stack(generated_samples).to(device)
    class_labels_tensor = torch.cat(class_labels_all).to(device)

    # Compute Inception Score
    inception_score, inception_std = calculate_inception_score(
        generated_tensor, device=device, batch_size=64, splits=10
    )

    # Compute DINO accuracy
    dino_accuracy = compute_dino_accuracy(
        sampler, generated_tensor, class_labels_tensor, batch_size=64
    )

    # Return metrics
    metrics = {
        "fid_score": fid_score,
        "inception_score": inception_score,
        "inception_std": inception_std,
        "noise_scale": noise_scale,
        "avg_velocity_magnitude": avg_velocity_magnitude,
        "avg_noise_magnitude": avg_noise_magnitude,
        "avg_noise_to_velocity_ratio": avg_noise_to_velocity_ratio,
        "velocity_magnitudes": velocity_magnitudes_all,
        "noise_magnitudes": noise_magnitudes_all,
        "noise_to_velocity_ratios": noise_to_velocity_ratios_all,
        "dino_top1_accuracy": dino_accuracy["top1_accuracy"],
        "dino_top5_accuracy": dino_accuracy["top5_accuracy"],
    }

    return metrics


def run_ode_divfree_experiment(sampler, device, fid, n_samples, batch_size, lambda_div):
    """Run ODE-divfree experiment with the given lambda_div value."""
    fid.reset()

    generated_samples = []
    total_velocity_magnitude = 0.0
    total_divfree_magnitude = 0.0
    total_divfree_to_velocity_ratio = 0.0
    velocity_magnitudes_all = []
    divfree_magnitudes_all = []
    divfree_to_velocity_ratios_all = []
    class_labels_all = []

    # Calculate number of batches
    num_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        num_batches += 1

    print(
        f"Generating {n_samples} samples using ODE-divfree sampling with lambda_div={lambda_div}..."
    )

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, n_samples - i * batch_size)

        # Random class labels
        class_labels = torch.randint(
            0, sampler.num_classes, (current_batch_size,), device=device
        )
        class_labels_all.append(class_labels.cpu())

        # Generate samples
        (
            samples,
            avg_velocity,
            avg_divfree,
            avg_ratio,
            velocity_magnitudes,
            divfree_magnitudes,
            divfree_to_velocity_ratios,
        ) = batch_sample_ode_divfree_with_metrics(
            sampler, class_labels, current_batch_size, lambda_div
        )

        # Update metrics
        generated_samples.extend(samples.cpu())
        total_velocity_magnitude += avg_velocity * current_batch_size
        total_divfree_magnitude += avg_divfree * current_batch_size
        total_divfree_to_velocity_ratio += avg_ratio * current_batch_size
        velocity_magnitudes_all.extend(velocity_magnitudes)
        divfree_magnitudes_all.extend(divfree_magnitudes)
        divfree_to_velocity_ratios_all.extend(divfree_to_velocity_ratios)

        # Update FID
        fid.update(samples.to(device), real=False)

    # Compute averages
    avg_velocity_magnitude = total_velocity_magnitude / n_samples
    avg_divfree_magnitude = total_divfree_magnitude / n_samples
    avg_divfree_to_velocity_ratio = total_divfree_to_velocity_ratio / n_samples

    # Compute FID
    fid_score = fid.compute().item()

    # Stack generated samples and class labels
    generated_tensor = torch.stack(generated_samples).to(device)
    class_labels_tensor = torch.cat(class_labels_all).to(device)

    # Compute Inception Score
    inception_score, inception_std = calculate_inception_score(
        generated_tensor, device=device, batch_size=64, splits=10
    )

    # Compute DINO accuracy
    dino_accuracy = compute_dino_accuracy(
        sampler, generated_tensor, class_labels_tensor, batch_size=64
    )

    # Return metrics
    metrics = {
        "fid_score": fid_score,
        "inception_score": inception_score,
        "inception_std": inception_std,
        "lambda_div": lambda_div,
        "avg_velocity_magnitude": avg_velocity_magnitude,
        "avg_divfree_magnitude": avg_divfree_magnitude,
        "avg_divfree_to_velocity_ratio": avg_divfree_to_velocity_ratio,
        "velocity_magnitudes": velocity_magnitudes_all,
        "divfree_magnitudes": divfree_magnitudes_all,
        "divfree_to_velocity_ratios": divfree_to_velocity_ratios_all,
        "dino_top1_accuracy": dino_accuracy["top1_accuracy"],
        "dino_top5_accuracy": dino_accuracy["top5_accuracy"],
    }

    return metrics


def run_edm_sde_experiment(sampler, device, fid, n_samples, batch_size, beta):
    """Run EDM SDE experiment with the given beta value."""
    fid.reset()

    generated_samples = []
    total_velocity_magnitude = 0.0
    total_noise_magnitude = 0.0
    total_noise_to_velocity_ratio = 0.0
    velocity_magnitudes_all = []
    noise_magnitudes_all = []
    noise_to_velocity_ratios_all = []
    class_labels_all = []

    num_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        num_batches += 1

    print(f"Generating {n_samples} samples using EDM SDE sampling with beta={beta}...")

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, n_samples - i * batch_size)

        class_labels = torch.randint(
            0, sampler.num_classes, (current_batch_size,), device=device
        )
        class_labels_all.append(class_labels.cpu())

        (
            samples,
            avg_velocity,
            avg_noise,
            avg_ratio,
            velocity_magnitudes,
            noise_magnitudes,
            noise_to_velocity_ratios,
        ) = batch_sample_edm_sde_with_metrics(
            sampler, class_labels, current_batch_size, beta
        )

        generated_samples.extend(samples.cpu())
        total_velocity_magnitude += avg_velocity * current_batch_size
        total_noise_magnitude += avg_noise * current_batch_size
        total_noise_to_velocity_ratio += avg_ratio * current_batch_size
        velocity_magnitudes_all.extend(velocity_magnitudes)
        noise_magnitudes_all.extend(noise_magnitudes)
        noise_to_velocity_ratios_all.extend(noise_to_velocity_ratios)

        fid.update(samples.to(device), real=False)

    avg_velocity_magnitude = total_velocity_magnitude / n_samples
    avg_noise_magnitude = total_noise_magnitude / n_samples
    avg_noise_to_velocity_ratio = total_noise_to_velocity_ratio / n_samples

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
        "beta": beta,
        "avg_velocity_magnitude": avg_velocity_magnitude,
        "avg_noise_magnitude": avg_noise_magnitude,
        "avg_noise_to_velocity_ratio": avg_noise_to_velocity_ratio,
        "velocity_magnitudes": velocity_magnitudes_all,
        "noise_magnitudes": noise_magnitudes_all,
        "noise_to_velocity_ratios": noise_to_velocity_ratios_all,
        "dino_top1_accuracy": dino_accuracy["top1_accuracy"],
        "dino_top5_accuracy": dino_accuracy["top5_accuracy"],
    }

    return metrics


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
        default=5000,
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
        default=20000,
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

    args = parser.parse_args()

    # Run the experiments
    run_experiment(args)
