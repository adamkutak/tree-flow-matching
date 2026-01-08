import subprocess
import os
import json
from datetime import datetime
import time

DATASET = "imagenet256"
SAMPLE_SIZES = [256]
BRANCH_PAIRS = "1:1,2:1,4:1,8:1"
SCORING_FUNCTION = "dino_score"
DT_STD = 0.7
WARP_SCALE = 0.5
DEVICE = "cuda"
NOISE_SCALE = 0.14
LAMBDA_DIV = 0.9
NOISE_SCHEDULE_END_FACTOR = 0.7
DETERMINISTIC_ROLLOUT = 0
REPULSION_DISABLE_UNTIL_TIME = 0.0
CFG_SCALE = 1.5

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
    results_dir = f"./ablations/cfg_ablation_{timestamp}"
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
        "dt_std": DT_STD,
        "warp_scale": WARP_SCALE,
        "noise_scale": NOISE_SCALE,
        "lambda_div": LAMBDA_DIV,
        "noise_schedule_end_factor": NOISE_SCHEDULE_END_FACTOR,
        "deterministic_rollout": DETERMINISTIC_ROLLOUT,
        "repulsion_disable_until_time": REPULSION_DISABLE_UNTIL_TIME,
        "cfg_scale": CFG_SCALE,
        "device": DEVICE,
    }

    with open(f"{results_dir}/ablation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    completed_experiments = 0

    print(f"\n=== Running CFG Ablation Experiments (CFG scale={CFG_SCALE}) ===\n")

    sample_methods = [
        "random_search",
        "noise_search_ode_divfree_max",
        "random_search_then_noise_search_ode_divfree_max",
    ]

    for sample_method in sample_methods:
        print(f"\n=== Running {sample_method} with CFG ===")

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
                    "--cfg_scale",
                    str(CFG_SCALE),
                ]

                print(f"\nRunning {sample_method}")
                if run_experiment(cmd):
                    completed_experiments += 1

    print(
        f"\nCFG ablation sweep completed! {completed_experiments} experiments ran successfully."
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
