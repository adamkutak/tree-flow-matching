import subprocess
import time
from datetime import datetime


def run_command(cmd):
    """Run a command and print its output."""
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

    duration = time.time() - start_time
    print(f"Command completed in {duration:.2f}s")
    return True


def main():
    target_class_labels = ["291", "309", "582", "11"]
    branch_dt = "0.01"

    # Configurations to run
    configs = [
        {
            "sample_method": "random_search",
            "scoring_function": "dino_score",
            "description": "DINO score, random search",
        },
        {
            "sample_method": "random_search",
            "scoring_function": "inception_score",
            "description": "inception score, random search",
        },
        {
            "sample_method": "random_search_then_noise_search_ode_divfree_max",
            "scoring_function": "dino_score",
            "description": "DINO score, RS+NS two stage",
        },
        {
            "sample_method": "random_search_then_noise_search_ode_divfree_max",
            "scoring_function": "inception_score",
            "description": "inception score, RS+NS two stage",
        },
    ]

    print(f"Starting generation of {len(configs)} paper figures...")
    print(f"Classes: {target_class_labels}")
    print(f"Branch dt: {branch_dt}")

    for config in configs:
        print(f"\n=== Generating: {config['description']} ===")

        cmd = [
            "python3",
            "paper_figure_generator.py",
            "--target_class_labels",
            *target_class_labels,
            "--sample_method",
            config["sample_method"],
            "--scoring_function",
            config["scoring_function"],
            "--branch_dt",
            branch_dt,
            "--dataset",
            "imagenet256",
            "--device",
            "cuda",  # assuming cuda is available, script handles fallback
            "--output_dir",
            "./paper_figures_ablation",
        ]

        success = run_command(cmd)
        if not success:
            print(f"Failed to generate figure for {config['description']}")

    print("\nAll requested figures have been processed.")


if __name__ == "__main__":
    main()
