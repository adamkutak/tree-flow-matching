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
    target_class_labels = ["742", "95", "995", "200"]
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

    print(f"Starting generation of individual single-row paper figures (PDFs)...")
    print(f"Classes: {target_class_labels}")
    print(f"Branch dt: {branch_dt}")

    for config in configs:
        print(f"\n=== Configuration: {config['description']} ===")

        # For each class, generate a single row figure (1x4)
        for class_label in target_class_labels:
            print(f"  Generating figure for Class {class_label}...")

            cmd = [
                "python3",
                "paper_figure_generator.py",
                "--target_class_label",
                class_label,
                "--samples_per_config",
                "1",
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
                "./paper_figures_individual_rows",
            ]

            success = run_command(cmd)
            if not success:
                print(
                    f"Failed to generate figure for {config['description']} Class {class_label}"
                )

    print("\nAll requested individual row figures have been processed.")


if __name__ == "__main__":
    main()
