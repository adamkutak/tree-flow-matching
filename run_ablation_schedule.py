import subprocess
import os
import json
from datetime import datetime
import time

# Configuration constants for ablation
DATASET = "imagenet256"
# Schedules to test
SCHEDULES = [
    # Default/Original
    [0.0, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95],
    # Tighter 1
    [0.0, 0.4, 0.75, 0.85, 0.9, 0.95],
    # Tighter 2
    [0.0, 0.5, 0.8, 0.9, 0.95],
    # Most efficient
    [0.0, 0.6, 0.9, 0.95],
]

TIMESTEP_CONFIGS = [
    (20, 0.05, 0),
]  # (num_timesteps, branch_dt, branch_start_time)
SAMPLE_SIZES = [256]
BRANCH_PAIRS = "2:1,4:1,8:1"  # Always use these branch pairs
SCORING_FUNCTION = "dino_score"
DT_STD = 0.7
WARP_SCALE = 0.5
DEVICE = "cuda"
NOISE_SCALE = 0.14
LAMBDA_DIV = 0.9
NOISE_SCHEDULE_END_FACTOR = 0.7
DETERMINISTIC_ROLLOUT = 0
REPULSION_DISABLE_UNTIL_TIME = 0.0


def run_experiment(cmd):
    """Run a single experiment with the given command."""
    print(f"Running: {' '.join(cmd)}")
    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Error running command: {' '.join(cmd)}")
        return False
    finally:
        duration = time.perf_counter() - start
        total_seconds = int(duration)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            print(f"Duration: {hours}h {minutes}m")
        elif minutes > 0:
            print(f"Duration: {minutes}m {seconds}s")
        else:
            print(f"Duration: {seconds}s")


def main():
    """Run schedule ablation experiments."""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"./ablations/schedule_ablation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Base command
    base_cmd = [
        "python3",
        "eval_sampler.py",
        "--output_dir",
        results_dir,
        "--dataset",
        DATASET,
        "--device",
        DEVICE,
    ]

    # Configuration for logging
    config = {
        "timestamp": timestamp,
        "results_dir": results_dir,
        "dataset": DATASET,
        "schedules": SCHEDULES,
        "timestep_configs": TIMESTEP_CONFIGS,
        "sample_sizes": SAMPLE_SIZES,
        "branch_pairs": BRANCH_PAIRS,
        "scoring_function": SCORING_FUNCTION,
        "dt_std": DT_STD,
        "warp_scale": WARP_SCALE,
        "noise_scale": NOISE_SCALE,
        "lambda_div": LAMBDA_DIV,
        "noise_schedule_end_factor": NOISE_SCHEDULE_END_FACTOR,
        "deterministic_rollout": DETERMINISTIC_ROLLOUT,
        "repulsion_disable_until_time": REPULSION_DISABLE_UNTIL_TIME,
        "device": DEVICE,
    }

    # Save configuration
    with open(f"{results_dir}/ablation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    completed_experiments = 0

    print(f"\n=== Running Ablation Experiments ===\n")

    # 1. Run Random Search Baseline
    print("\n=== Running Random Search Baseline ===")
    for num_timesteps, branch_dt, branch_start_time in TIMESTEP_CONFIGS:
        for n_samples in SAMPLE_SIZES:
            cmd = base_cmd + [
                "--eval_mode",
                "single_samples",
                "--sample_method",
                "random_search",
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
            ]

            print(f"\nRunning Random Search Baseline")
            if run_experiment(cmd):
                completed_experiments += 1

    # 2. Run Noise Search with different schedules
    print("\n=== Running Noise Search Schedule Ablation ===")
    sample_method = "noise_search_ode_divfree_max"

    for schedule in SCHEDULES:
        schedule_str = ",".join(map(str, schedule))
        for num_timesteps, branch_dt, branch_start_time in TIMESTEP_CONFIGS:
            for n_samples in SAMPLE_SIZES:
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
                    "--noise_scale",
                    str(NOISE_SCALE),
                    "--lambda_div",
                    str(LAMBDA_DIV),
                    "--noise_schedule_end_factor",
                    str(NOISE_SCHEDULE_END_FACTOR),
                    "--deterministic_rollout",
                    str(DETERMINISTIC_ROLLOUT),
                    "--repulsion_disable_until_time",
                    str(REPULSION_DISABLE_UNTIL_TIME),
                    "--round_start_times",
                    schedule_str,
                ]

                print(f"\nRunning experiment with schedule: {schedule}")
                if run_experiment(cmd):
                    completed_experiments += 1

    print(
        f"\nAblation sweep completed! {completed_experiments} experiments ran successfully."
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
