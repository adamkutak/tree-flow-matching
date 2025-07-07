import json
import argparse


def clean_noise_json(input_file, output_file=None):
    """
    Clean noise study JSON by removing individual value lists and keeping only averages.
    """
    if output_file is None:
        output_file = input_file  # Overwrite the original file

    # Read the JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Keys to remove (individual lists)
    keys_to_remove = [
        "velocity_magnitudes",
        "secondary_magnitudes",
        "ratios",
        "noise_magnitudes",
        "divfree_magnitudes",
        "noise_to_velocity_ratios",
        "divfree_to_velocity_ratios",
    ]

    def clean_metrics(metrics_dict):
        """Remove individual value lists from a metrics dictionary."""
        if not isinstance(metrics_dict, dict):
            return metrics_dict

        cleaned = {}
        for key, value in metrics_dict.items():
            if key not in keys_to_remove:
                cleaned[key] = value
        return cleaned

    # Clean the data structure
    if "ode_baseline" in data:
        data["ode_baseline"] = clean_metrics(data["ode_baseline"])

    if "experiments" in data:
        for experiment in data["experiments"]:
            if "metrics" in experiment:
                experiment["metrics"] = clean_metrics(experiment["metrics"])

    # Write the cleaned data back
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Cleaned JSON data written to {output_file}")

    # Report what was removed
    total_removed = 0
    for key in keys_to_remove:
        count = count_occurrences(data, key)
        if count > 0:
            print(f"Removed {count} instances of '{key}' lists")
            total_removed += count

    if total_removed == 0:
        print("No individual value lists found to remove (data may already be clean)")
    else:
        print(f"Total removed: {total_removed} individual value lists")


def count_occurrences(obj, key):
    """Count how many times a key would have been removed."""
    count = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                count += 1
            else:
                count += count_occurrences(v, key)
    elif isinstance(obj, list):
        for item in obj:
            count += count_occurrences(item, key)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean noise study JSON by removing individual value lists"
    )

    parser.add_argument(
        "input_file", type=str, help="Input JSON file to clean (e.g., noise.json)"
    )
    parser.add_argument(
        "--output", type=str, help="Output file (defaults to overwriting input file)"
    )

    args = parser.parse_args()

    clean_noise_json(args.input_file, args.output)
