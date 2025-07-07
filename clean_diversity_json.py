import json
import argparse


def clean_diversity_json(input_file, output_file=None):
    """
    Clean diversity study JSON by removing individual diversity value lists and keeping only averages.
    """
    if output_file is None:
        output_file = input_file  # Overwrite the original file

    # Read the JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Keys to remove from summary section (individual lists)
    keys_to_remove = ["all_diversities"]

    def clean_summary_metrics(summary_dict):
        """Remove individual value lists from summary metrics."""
        if not isinstance(summary_dict, dict):
            return summary_dict

        cleaned = {}
        for method_name, method_data in summary_dict.items():
            if isinstance(method_data, dict):
                cleaned_method = {}
                for key, value in method_data.items():
                    if key not in keys_to_remove:
                        cleaned_method[key] = value
                cleaned[method_name] = cleaned_method
            else:
                cleaned[method_name] = method_data
        return cleaned

    # Clean the summary section
    if "summary" in data:
        data["summary"] = clean_summary_metrics(data["summary"])

    # Write the cleaned data back
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Cleaned diversity JSON data written to {output_file}")

    # Report what was removed
    total_removed = 0
    removed_count = count_summary_occurrences(data, keys_to_remove)

    for key, count in removed_count.items():
        if count > 0:
            print(f"Removed {count} instances of '{key}' lists from summary")
            total_removed += count

    if total_removed == 0:
        print(
            "No individual diversity lists found to remove (data may already be clean)"
        )
    else:
        print(f"Total removed: {total_removed} individual diversity lists")


def count_summary_occurrences(data, keys_to_remove):
    """Count how many times keys would have been removed from summary."""
    counts = {key: 0 for key in keys_to_remove}

    if "summary" in data and isinstance(data["summary"], dict):
        for method_name, method_data in data["summary"].items():
            if isinstance(method_data, dict):
                for key in keys_to_remove:
                    if key in method_data:
                        counts[key] += 1

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean diversity study JSON by removing individual diversity lists"
    )

    parser.add_argument(
        "input_file", type=str, help="Input JSON file to clean (e.g., diversity.json)"
    )
    parser.add_argument(
        "--output", type=str, help="Output file (defaults to overwriting input file)"
    )

    args = parser.parse_args()

    clean_diversity_json(args.input_file, args.output)
