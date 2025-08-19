import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import itertools

from mcts_single_flow import MCTSFlowSampler
from imagenet_dataset import ImageNet32Dataset
from utils import (
    divfree_swirl_si,
    score_si_linear,
    divergence_free_particle_guidance,
    get_alpha_schedule_flow_matching,
)


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


def batch_sample_ode_random_start(sampler, class_label, batch_size):
    """
    Regular flow matching sampling where each sample starts from different random noise.
    This serves as a baseline for maximum expected diversity from random initialization.
    """
    sampler.flow_model.eval()

    with torch.no_grad():
        # Each sample starts from different random noise
        current_samples = torch.randn(
            batch_size,
            sampler.channels,
            sampler.image_size,
            sampler.image_size,
            device=sampler.device,
        )
        current_label = torch.full((batch_size,), class_label, device=sampler.device)

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            velocity = sampler.flow_model(t_batch, current_samples, current_label)
            current_samples = current_samples + velocity * dt

        return sampler.unnormalize_images(current_samples)


def batch_sample_sde_identical_start(
    sampler, class_label, single_noise, batch_size, noise_scale=0.05
):
    """
    SDE sampling where all samples start from identical noise.
    Should produce different outputs due to stochastic noise.
    """
    sampler.flow_model.eval()

    with torch.no_grad():
        # All samples start identical
        current_samples = single_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        current_label = torch.full((batch_size,), class_label, device=sampler.device)

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            # Add different random noise to each sample
            noise = torch.randn_like(current_samples) * torch.sqrt(dt) * noise_scale
            current_samples = current_samples + velocity * dt + noise

        return sampler.unnormalize_images(current_samples)


def batch_sample_ode_divfree_identical_start(
    sampler, class_label, single_noise, batch_size, lambda_div=0.2
):
    """
    ODE-divfree sampling where all samples start from identical noise.
    Should produce different outputs due to divergence-free field, even though deterministic.
    """
    sampler.flow_model.eval()

    with torch.no_grad():
        # All samples start identical
        x = single_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        y = torch.full(
            (batch_size,), class_label, device=sampler.device, dtype=torch.long
        )

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            u_t = sampler.flow_model(t_batch, x, y)
            w_unscaled = divfree_swirl_si(x, t_batch, y, u_t)
            w = lambda_div * w_unscaled

            x = x + (u_t + w) * dt

        return sampler.unnormalize_images(x)


def batch_sample_edm_sde_identical_start(
    sampler, class_label, single_noise, batch_size, beta=0.05
):
    """
    EDM SDE sampling where all samples start from identical noise.
    Should produce different outputs due to stochastic noise with score-based drift correction.
    """
    sampler.flow_model.eval()

    with torch.no_grad():
        # All samples start identical
        current_samples = single_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        current_label = torch.full((batch_size,), class_label, device=sampler.device)

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            score = score_si_linear(current_samples, t_batch, velocity)

            sigma_t = t_batch.view(-1, *([1] * (current_samples.ndim - 1)))

            drift_corr = -beta * (sigma_t**2) * score

            total_velocity = velocity + drift_corr

            brownian = (
                torch.randn_like(current_samples)
                * torch.sqrt(dt)
                * torch.sqrt(torch.tensor(2.0 * beta, device=sampler.device))
                * sigma_t
            )

            current_samples = current_samples + total_velocity * dt + brownian

        return sampler.unnormalize_images(current_samples)


def batch_sample_score_sde_identical_start(
    sampler, class_label, single_noise, batch_size, noise_scale_factor=1.0
):
    """
    Score SDE sampling where all samples start from identical noise.
    Should produce different outputs due to stochastic noise with score-based diffusion.
    """
    sampler.flow_model.eval()

    with torch.no_grad():
        # All samples start identical
        current_samples = single_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        current_label = torch.full((batch_size,), class_label, device=sampler.device)

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

            # Euler-Maruyama update
            current_samples = current_samples + drift * dt + noise

        return sampler.unnormalize_images(current_samples)


def batch_sample_sde_divfree_identical_start(
    sampler, class_label, single_noise, batch_size, lambda_div=0.2
):
    """
    SDE sampling with divergence-free field as noise where all samples start from identical noise.
    Uses Euler-Maruyama with the divergence-free field as the stochastic noise term.
    """
    sampler.flow_model.eval()

    with torch.no_grad():
        # All samples start identical
        current_samples = single_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        current_label = torch.full((batch_size,), class_label, device=sampler.device)

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            # Compute divergence-free field
            w_unscaled = divfree_swirl_si(
                current_samples, t_batch, current_label, velocity
            )

            # Use divergence-free field as noise term in Euler-Maruyama scheme
            divfree_noise = w_unscaled * torch.sqrt(dt) * lambda_div

            # Euler-Maruyama update: drift + noise
            current_samples = current_samples + velocity * dt + divfree_noise

        return sampler.unnormalize_images(current_samples)


def batch_sample_particle_guidance_identical_start(
    sampler,
    class_label,
    single_noise,
    batch_size,
    alpha_0=2.0,
    alpha_1=0.1,
    kernel_type="rbf",
    schedule_type="linear",
):
    """
    Particle guidance sampling where all samples start from identical noise.
    Should produce diverse outputs due to repulsive forces between particles.
    """
    sampler.flow_model.eval()

    with torch.no_grad():
        # All samples start identical
        current_samples = single_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        current_label = torch.full((batch_size,), class_label, device=sampler.device)

        for step, t in enumerate(sampler.timesteps[:-1]):
            dt = sampler.timesteps[step + 1] - t
            t_batch = torch.full((batch_size,), t.item(), device=sampler.device)

            # Get velocity field from flow model
            velocity = sampler.flow_model(t_batch, current_samples, current_label)

            # Get time-dependent guidance strength
            alpha_t = get_alpha_schedule_flow_matching(
                t.item(), alpha_0, alpha_1, schedule_type
            )

            # Get divergence-free particle guidance
            guidance_forces = divergence_free_particle_guidance(
                current_samples, t_batch, current_label, velocity, alpha_t, kernel_type
            )

            # ODE update with particle guidance
            current_samples = current_samples + velocity * dt + guidance_forces * dt

        return sampler.unnormalize_images(current_samples)


def calculate_batch_diversity(features):
    """
    Calculate average pairwise distance between features in a batch.

    Args:
        features: numpy array of shape (batch_size, feature_dim)

    Returns:
        float: average pairwise distance
    """
    batch_size = features.shape[0]
    if batch_size < 2:
        return 0.0

    total_distance = 0.0
    num_pairs = 0

    # Calculate pairwise distances
    for i, j in itertools.combinations(range(batch_size), 2):
        distance = np.linalg.norm(features[i] - features[j])
        total_distance += distance
        num_pairs += 1

    return total_distance / num_pairs


def run_diversity_experiment(args):
    """
    Run diversity experiments testing how much samples diverge when starting from identical initial noise.
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
    flow_model_name = f"flow_model_{args.dataset}.pt"
    num_timesteps = 20

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
        load_dino=False,
    )

    # Create output directory for results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.output_dir, f"diversity_study_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "num_trials": args.num_trials,
        "timesteps": num_timesteps,
        "experiments": [],
    }

    print(
        f"\nRunning diversity experiments with {args.num_trials} trials of batch size {args.batch_size}"
    )
    print(
        "Testing how much samples diverge when starting from IDENTICAL initial noise..."
    )

    # For each trial, test all methods starting from the same single noise sample
    for trial_idx in tqdm(range(args.num_trials), desc="Processing trials"):
        # Generate a single noise sample that all batch samples will start from
        single_noise = torch.randn(channels, image_size, image_size, device=device)

        # Use a random class for this trial
        class_label = torch.randint(0, num_classes, (1,)).item()

        trial_results = {
            "trial_idx": trial_idx,
            "class_label": class_label,
            "methods": {},
        }

        # Test ODE baseline with random different starting noises (maximum diversity baseline)
        ode_samples = batch_sample_ode_random_start(
            sampler, class_label, args.batch_size
        )
        ode_features = sampler.extract_inception_features(ode_samples)
        ode_diversity = calculate_batch_diversity(ode_features)
        trial_results["methods"]["ode_random"] = {"diversity": ode_diversity}

        # Test SDE with different noise scales (all starting from same noise)
        for noise_scale in args.noise_scales:
            sde_samples = batch_sample_sde_identical_start(
                sampler, class_label, single_noise, args.batch_size, noise_scale
            )
            sde_features = sampler.extract_inception_features(sde_samples)
            sde_diversity = calculate_batch_diversity(sde_features)
            trial_results["methods"][f"sde_{noise_scale}"] = {
                "diversity": sde_diversity
            }

        # Test EDM SDE with different beta values (all starting from same noise)
        for beta in args.beta_values:
            edm_sde_samples = batch_sample_edm_sde_identical_start(
                sampler, class_label, single_noise, args.batch_size, beta
            )
            edm_sde_features = sampler.extract_inception_features(edm_sde_samples)
            edm_sde_diversity = calculate_batch_diversity(edm_sde_features)
            trial_results["methods"][f"edm_sde_{beta}"] = {
                "diversity": edm_sde_diversity
            }

        # Test Score SDE with different noise scale factors (all starting from same noise)
        for noise_scale_factor in args.score_sde_factors:
            score_sde_samples = batch_sample_score_sde_identical_start(
                sampler, class_label, single_noise, args.batch_size, noise_scale_factor
            )
            score_sde_features = sampler.extract_inception_features(score_sde_samples)
            score_sde_diversity = calculate_batch_diversity(score_sde_features)
            trial_results["methods"][f"score_sde_{noise_scale_factor}"] = {
                "diversity": score_sde_diversity
            }

        # Test ODE-divfree with different lambda values (all starting from same noise)
        for lambda_div in args.lambda_divs:
            divfree_samples = batch_sample_ode_divfree_identical_start(
                sampler, class_label, single_noise, args.batch_size, lambda_div
            )
            divfree_features = sampler.extract_inception_features(divfree_samples)
            divfree_diversity = calculate_batch_diversity(divfree_features)
            trial_results["methods"][f"divfree_{lambda_div}"] = {
                "diversity": divfree_diversity
            }

        # Test SDE-divfree with different lambda values (all starting from same noise)
        for lambda_div in args.lambda_divs:
            sde_divfree_samples = batch_sample_sde_divfree_identical_start(
                sampler, class_label, single_noise, args.batch_size, lambda_div
            )
            sde_divfree_features = sampler.extract_inception_features(
                sde_divfree_samples
            )
            sde_divfree_diversity = calculate_batch_diversity(sde_divfree_features)
            trial_results["methods"][f"sde_divfree_{lambda_div}"] = {
                "diversity": sde_divfree_diversity
            }

        # Test Particle Guidance with different alpha ranges (all starting from same noise)
        alpha_ranges = [(2.0, 0.1), (1.0, 0.1), (0.5, 0.1), (1.0, 0.2)]
        for alpha_0, alpha_1 in alpha_ranges:
            pg_samples = batch_sample_particle_guidance_identical_start(
                sampler,
                class_label,
                single_noise,
                args.batch_size,
                alpha_0,
                alpha_1,
                "rbf",
                "linear",
            )
            pg_features = sampler.extract_inception_features(pg_samples)
            pg_diversity = calculate_batch_diversity(pg_features)
            trial_results["methods"][f"particle_guidance_{alpha_0}_{alpha_1}"] = {
                "diversity": pg_diversity
            }

        results["experiments"].append(trial_results)

    # Calculate summary statistics
    print("\n" + "=" * 60)
    print("DIVERSITY RESULTS SUMMARY")
    print("(How much do samples diverge when starting from identical noise?)")
    print("=" * 60)

    summary = {}

    # Collect all diversity scores by method
    method_diversities = {}
    for experiment in results["experiments"]:
        for method_name, method_data in experiment["methods"].items():
            if method_name not in method_diversities:
                method_diversities[method_name] = []
            method_diversities[method_name].append(method_data["diversity"])

    # Calculate mean and std for each method
    for method_name, diversities in method_diversities.items():
        mean_diversity = np.mean(diversities)
        std_diversity = np.std(diversities)
        summary[method_name] = {
            "mean_diversity": float(mean_diversity),
            "std_diversity": float(std_diversity),
        }

        if method_name == "ode_random":
            print(
                f"{method_name:15s}: {mean_diversity:.4f} ± {std_diversity:.4f} (random baseline - max diversity)"
            )
        else:
            print(f"{method_name:15s}: {mean_diversity:.4f} ± {std_diversity:.4f}")

    results["summary"] = summary

    # Save results with proper serialization
    result_file = os.path.join(results_dir, f"diversity_study_results_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to {result_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flow matching diversity study - identical start test"
    )

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

    # Experiment parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for diversity measurement",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of trials to test",
    )

    # Noise parameters
    parser.add_argument(
        "--noise_scales",
        type=float,
        nargs="+",
        default=[0.1, 0.13, 0.16, 0.2, 0.4, 0.6],
        help="Noise scale values to test for SDE sampling",
    )
    parser.add_argument(
        "--lambda_divs",
        type=float,
        nargs="+",
        default=[0.1, 0.35, 0.4, 0.45, 0.5, 0.6, 2.0],
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

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Run the experiments
    run_diversity_experiment(args)
