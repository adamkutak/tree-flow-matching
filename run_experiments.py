import subprocess
import os
import json
from datetime import datetime

# Configuration constants for experiments
DATASET = "imagenet32"
SAMPLE_METHODS = ["random_search", "path_exploration", "path_exploration_timewarp"]
TIMESTEP_CONFIGS = [
    (10, 0.1, 0.5),
    (20, 0.05, 0.5),
]  # (num_timesteps, branch_dt, branch_start_time)
SAMPLE_SIZES = [1000, 5000]
BRANCH_PAIRS = "1:1,2:1,4:1,8:1"  # Always use these branch pairs
BRANCH_PAIRS_BATCH = "2:4,4:8,8:16"  # Use different pairs for batch optimization
SCORING_FUNCTION = "inception_score, inception_classifier"  # Default scoring function
DT_STD = 0.7  # Path exploration time step standard deviation
WARP_SCALE = 0.5  # Time warp scale factor
DEVICE = "cuda:0"  # Default device


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
                ]

                # Run the experiment
                print(f"\nRunning experiment with:")
                print(f"  Sample method: {sample_method}")
                print(f"  Time steps: {num_timesteps} (dt={branch_dt})")
                print(f"  Samples: {n_samples}")

                if run_experiment(cmd):
                    completed_experiments += 1

    # Run batch optimization experiments
    print("\n=== Running batch optimization experiments ===")
    for sample_method in SAMPLE_METHODS:
        for num_timesteps, branch_dt, branch_start_time in TIMESTEP_CONFIGS:
            for n_samples in SAMPLE_SIZES:
                # Build command
                cmd = base_cmd + [
                    "--eval_mode",
                    "batch_optimization",
                    "--sample_method",
                    sample_method,
                    "--n_samples",
                    str(n_samples),
                    "--branch_pairs",
                    BRANCH_PAIRS_BATCH,
                    "--refinement_batch_size",
                    "32",
                    "--num_iterations",
                    "1",
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
                ]

                # Run the experiment
                print(f"\nRunning batch experiment with:")
                print(f"  Sample method: {sample_method}")
                print(f"  Time steps: {num_timesteps} (dt={branch_dt})")
                print(f"  Samples: {n_samples}")

                if run_experiment(cmd):
                    completed_experiments += 1

    print(
        f"\nSweep completed! {completed_experiments}/{total_experiments*2} experiments ran successfully."
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
