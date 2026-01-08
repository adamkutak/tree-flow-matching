import subprocess
import os
import json
from datetime import datetime
import time

DATASET = "imagenet256"
SAMPLE_SIZES = [256]
SCORING_FUNCTION = "dino_score"
DEVICE = "cuda"

BRANCH_PAIRS_LIST = ["1:1", "2:1", "4:1", "8:1"]
NUM_TIMESTEPS = 20
DT = 0.05
SIMULATE_FORWARD_DT = 0.1
FINE_DT_THRESHOLD = 0.7

LAMBDA_DIV = 0.9
NOISE_SCHEDULE_END_FACTOR = 0.7
DETERMINISTIC_ROLLOUT = 0
REPULSION_DISABLE_UNTIL_TIME = 0.0


def run_experiment(cmd):
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"./ablations/coarse_simulate_ablation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "results_dir": results_dir,
        "dataset": DATASET,
        "sample_sizes": SAMPLE_SIZES,
        "scoring_function": SCORING_FUNCTION,
        "device": DEVICE,
        "branch_pairs_list": BRANCH_PAIRS_LIST,
        "num_timesteps": NUM_TIMESTEPS,
        "dt": DT,
        "simulate_forward_dt": SIMULATE_FORWARD_DT,
        "fine_dt_threshold": FINE_DT_THRESHOLD,
        "lambda_div": LAMBDA_DIV,
        "noise_schedule_end_factor": NOISE_SCHEDULE_END_FACTOR,
        "deterministic_rollout": DETERMINISTIC_ROLLOUT,
        "repulsion_disable_until_time": REPULSION_DISABLE_UNTIL_TIME,
    }

    with open(f"{results_dir}/ablation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    completed_experiments = 0

    print(
        f"\n=== Running Coarse Simulate Ablation (dt={DT}, simulate_forward_dt={SIMULATE_FORWARD_DT}) ===\n"
    )

    base_cmd = [
        "python3",
        "eval_sampler.py",
        "--output_dir",
        results_dir,
        "--dataset",
        DATASET,
        "--device",
        DEVICE,
        "--scoring_function",
        SCORING_FUNCTION,
        "--n_samples",
        str(SAMPLE_SIZES[0]),
        "--lambda_div",
        str(LAMBDA_DIV),
        "--noise_schedule_end_factor",
        str(NOISE_SCHEDULE_END_FACTOR),
        "--deterministic_rollout",
        str(DETERMINISTIC_ROLLOUT),
        "--repulsion_disable_until_time",
        str(REPULSION_DISABLE_UNTIL_TIME),
        "--branch_dt",
        str(DT),
    ]

    for branch_pairs in BRANCH_PAIRS_LIST:
        print(f"\n=== Running Random Search ({branch_pairs}) ===")
        cmd = base_cmd + [
            "--eval_mode",
            "single_samples",
            "--sample_method",
            "random_search",
            "--branch_pairs",
            branch_pairs,
        ]
        if run_experiment(cmd):
            completed_experiments += 1

    for branch_pairs in BRANCH_PAIRS_LIST:
        print(
            f"\n=== Running Standard Noise Search ODE Divfree Max ({branch_pairs}) ==="
        )
        cmd = base_cmd + [
            "--eval_mode",
            "single_samples",
            "--sample_method",
            "noise_search_ode_divfree_max",
            "--branch_pairs",
            branch_pairs,
        ]
        if run_experiment(cmd):
            completed_experiments += 1

    for branch_pairs in BRANCH_PAIRS_LIST:
        print(
            f"\n=== Running Coarse Simulate Noise Search ({branch_pairs}, simulate_forward_dt={SIMULATE_FORWARD_DT}) ==="
        )
        cmd = base_cmd + [
            "--eval_mode",
            "single_samples",
            "--sample_method",
            "noise_search_ode_divfree_max_coarse",
            "--branch_pairs",
            branch_pairs,
            "--simulate_forward_dt",
            str(SIMULATE_FORWARD_DT),
            "--fine_dt_threshold",
            str(FINE_DT_THRESHOLD),
        ]
        if run_experiment(cmd):
            completed_experiments += 1

    print(
        f"\nAblation sweep completed! {completed_experiments} experiments ran successfully."
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
