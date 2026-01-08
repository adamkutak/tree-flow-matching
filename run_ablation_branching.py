import subprocess
import os
import json
from datetime import datetime
import time

# Configuration constants for ablation
DATASET = "imagenet256"

# Branch schedules for ablation
# Note: num_keep is always 1 for these experiments
BRANCH_SCHEDULES = {
    # 2x Compute
    "2x_uniform": [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Uniform 2x
    "2x_custom_a": [1, 2, 1, 2, 2, 4, 4, 8, 8],  # Custom A
    "2x_custom_b": [1, 1, 1, 2, 2, 4, 8, 12, 16],  # Custom B (Extreme)
    # 4x Compute
    "4x_uniform": [4, 4, 4, 4, 4, 4, 4, 4, 4],  # Uniform 4x
    "4x_custom_a": [2, 4, 2, 4, 4, 8, 8, 16, 16],  # Custom A
    "4x_custom_b": [2, 2, 2, 4, 4, 8, 16, 24, 32],  # Custom B
    # 8x Compute
    "8x_uniform": [8, 8, 8, 8, 8, 8, 8, 8, 8],  # Uniform 8x
    "8x_custom_a": [4, 8, 4, 8, 8, 16, 16, 32, 32],  # Custom A
    "8x_custom_b": [4, 4, 4, 8, 8, 16, 32, 48, 64],  # Custom B
}

TIMESTEP_CONFIGS = [
    (20, 0.05, 0),
]
SAMPLE_SIZES = [256]
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
    """Run branching schedule ablation experiments."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"./ablations/branching_ablation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

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

    config = {
        "timestamp": timestamp,
        "results_dir": results_dir,
        "dataset": DATASET,
        "branch_schedules": BRANCH_SCHEDULES,
        "timestep_configs": TIMESTEP_CONFIGS,
        "sample_sizes": SAMPLE_SIZES,
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

    with open(f"{results_dir}/ablation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    completed_experiments = 0

    print(f"\n=== Running Branching Ablation Experiments ===\n")

    # 1. Run Random Search Baseline (1x)
    print("\n=== Running Random Search Baseline (1x) ===")
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
                "1:1",  # 1x compute
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

    # 2. Run Noise Search with different branching schedules
    print("\n=== Running Noise Search Branching Ablation ===")
    sample_method = "noise_search_ode_divfree_max"

    for schedule_name, schedule in BRANCH_SCHEDULES.items():
        schedule_str = ",".join(map(str, schedule))

        # Determine effective "branch_pair" label for logging consistency
        # e.g. if average is 2, label as 2:1. If average is 4, label as 4:1.
        avg_branches = sum(schedule) / len(schedule)
        approx_branch_label = f"{int(round(avg_branches))}:1"

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
                    approx_branch_label,  # Label for logging
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
                    "--branch_schedule",
                    schedule_str,  # Custom schedule
                ]

                print(
                    f"\nRunning experiment with schedule: {schedule_name} ({schedule})"
                )
                if run_experiment(cmd):
                    completed_experiments += 1

    print(
        f"\nAblation sweep completed! {completed_experiments} experiments ran successfully."
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
