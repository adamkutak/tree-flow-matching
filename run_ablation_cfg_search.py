import subprocess
import os
import json
from datetime import datetime
import time

DATASET = "imagenet256"
SAMPLE_SIZES = [256]
BRANCH_PAIRS = "1:1,2:1,4:1,8:1"
SCORING_FUNCTION = "dino_score"
DEVICE = "cuda"

TIMESTEP_CONFIGS = [
    (20, 0.05, 0),
]


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
    results_dir = f"./ablations/cfg_search_ablation_{timestamp}"
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
        "timestep_configs": TIMESTEP_CONFIGS,
        "sample_sizes": SAMPLE_SIZES,
        "branch_pairs": BRANCH_PAIRS,
        "scoring_function": SCORING_FUNCTION,
        "device": DEVICE,
    }

    with open(f"{results_dir}/ablation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    completed_experiments = 0

    print(f"\n=== Running CFG Search Ablation Experiments ===\n")
    print("This ablation tests using CFG scale variation as the branching mechanism")
    print("instead of noise injection.\n")

    for num_timesteps, branch_dt, branch_start_time in TIMESTEP_CONFIGS:
        for n_samples in SAMPLE_SIZES:
            cmd = base_cmd + [
                "--eval_mode",
                "single_samples",
                "--sample_method",
                "cfg_search",
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
            ]

            print(f"\nRunning CFG Search")
            if run_experiment(cmd):
                completed_experiments += 1

    print(
        f"\nCFG search ablation sweep completed! {completed_experiments} experiments ran successfully."
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
