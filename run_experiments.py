import subprocess
import os
import json
from datetime import datetime

# Configuration constants for experiments
DATASET = "imagenet256"
SAMPLE_METHODS = [
    "sde_path_exploration",
    "score_sde_path_exploration",
    "ode_divfree_path_exploration",
    "random_search_then_divfree_path_exploration",
    "random_search",
]
TIMESTEP_CONFIGS = [
    (20, 0.05, 0),
]  # (num_timesteps, branch_dt, branch_start_time)
SAMPLE_SIZES = [1024]
BRANCH_PAIRS = "1:1,2:1,4:1,8:1"  # Always use these branch pairs
SCORING_FUNCTION = "dino_score"  # Default scoring function
DT_STD = 0.7  # Path exploration time step standard deviation
WARP_SCALE = 0.5  # Time warp scale factor
DEVICE = "cuda"  # Default device
NOISE_SCALE = 0.14
LAMBDA_DIV = 0.55


def run_experiment(cmd):
    """Run a single experiment with the given command."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Error running command: {' '.join(cmd)}")
        return False


def main():
    """Run a predefined set of experiments."""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"./results_sweep_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Base command
    base_cmd = [
        "python",
        "eval_sampler.py",
        "--output_dir",
        results_dir,
        "--dataset",
        DATASET,
    ]

    # Configuration for logging
    config = {
        "timestamp": timestamp,
        "results_dir": results_dir,
        "dataset": DATASET,
        "sample_methods": SAMPLE_METHODS,
        "timestep_configs": TIMESTEP_CONFIGS,
        "sample_sizes": SAMPLE_SIZES,
        "branch_pairs": BRANCH_PAIRS,
        "scoring_function": SCORING_FUNCTION,
        "dt_std": DT_STD,
        "warp_scale": WARP_SCALE,
        "noise_scale": NOISE_SCALE,
        "lambda_div": LAMBDA_DIV,
        "device": DEVICE,
    }

    # Save configuration
    with open(f"{results_dir}/sweep_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Track experiments
    total_experiments = len(SAMPLE_METHODS) * len(TIMESTEP_CONFIGS) * len(SAMPLE_SIZES)
    completed_experiments = 0

    print(f"\n=== Running {total_experiments} experiments ===\n")

    # Run single sample optimization experiments
    print("\n=== Running single sample optimization experiments ===")
    for sample_method in SAMPLE_METHODS:
        for num_timesteps, branch_dt, branch_start_time in TIMESTEP_CONFIGS:
            for n_samples in SAMPLE_SIZES:
                # Build command
                cmd = base_cmd + [
                    "--eval_mode",
                    "single_samples",
                    "--sample_method",
                    sample_method,
                    "--scoring_function",
                    SCORING_FUNCTION,
                    "--n_samples",
                    str(n_samples),
                    "--branch_pairs",
                    BRANCH_PAIRS,
                    "--branch_dt",
                    str(branch_dt),
                    "--branch_start_time",
                    str(branch_start_time),
                    "--dt_std",
                    str(DT_STD),
                    "--warp_scale",
                    str(WARP_SCALE),
                    "--device",
                    DEVICE,
                    "--noise_scale",
                    str(NOISE_SCALE),
                    "--lambda_div",
                    str(LAMBDA_DIV),
                ]

                # Run the experiment
                print(f"\nRunning experiment with:")
                print(f"  Sample method: {sample_method}")
                print(f"  Time steps: {num_timesteps} (dt={branch_dt})")
                print(f"  Samples: {n_samples}")

                if run_experiment(cmd):
                    completed_experiments += 1

    print(
        f"\nSweep completed! {completed_experiments}/{total_experiments} experiments ran successfully."
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
