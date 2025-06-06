import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


def parse_folder_name(folder_name):
    """
    Parse folder name to extract dataset, retrieval model, score function, and top_k
    Expected format: dataset-retrieval_model-score_function-top_k-extract_val
    """
    parts = folder_name.split("-")
    if len(parts) >= 4:
        dataset = parts[0]
        retrieval_model = parts[2]
        score_function = parts[3]
        top_k = parts[4]
        if score_function == "ms":
            retrieval_model = f"{parts[2]}-{parts[3]}"
            score_function = parts[4]
            top_k = parts[5]
        return dataset, retrieval_model, score_function, top_k
    return None, None, None, None


def calculate_quantiles(data, quantiles=[0.95, 0.90]):
    """
    Calculate specified quantiles (default: 95th and 90th percentiles)
    """
    return [np.quantile(data, q) for q in quantiles]


def process_json_files(root_directory):
    """
    Process all JSON files in the directory structure and extract quantile data
    """
    results = []

    # Walk through all directories
    for root, dirs, files in os.walk(root_directory):
        folder_name = os.path.basename(root)
        if "extract_val" not in folder_name:
            continue
        # Parse folder name to get components
        dataset, retrieval_model, score_function, top_k = parse_folder_name(folder_name)

        # Skip if we can't parse the folder name
        if not all([dataset, retrieval_model, score_function, top_k]):
            continue

        # Look for JSON files with specified suffixes
        target_files = []
        for file in files:
            if file.endswith(".json") and (
                "diff_end_all" in file or "diff_start_all" in file
            ):
                target_files.append(file)

        # Process each target file
        for filename in target_files:
            filepath = os.path.join(root, filename)

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Ensure data is a list of numbers
                if isinstance(data, list) and all(
                    isinstance(x, (int, float)) for x in data
                ):
                    # Calculate 5th and 10th highest quantiles (95th and 90th percentiles)
                    quantiles = calculate_quantiles(data, [0.95, 0.90])

                    # Determine file type
                    adv_pos = "end" if "diff_end_all" in filename else "start"

                    results.append(
                        {
                            "dataset": dataset,
                            "retrieval_model": retrieval_model,
                            "score_function": score_function,
                            "top_k": top_k,
                            "adv_pos": adv_pos,
                            "data_points": len(data),
                            "5th_highest_quantile_95th_percentile": quantiles[0],
                            "10th_highest_quantile_90th_percentile": quantiles[1],
                        }
                    )

                    print(f"Processed: {filepath}")

            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")

    return results


def main():
    # Set the root directory path (change this to your actual path)
    root_directory = (
        "./data_cache/outputs"  # Current directory - change this to your data directory
    )

    print("Starting JSON file analysis...")
    print(f"Root directory: {os.path.abspath(root_directory)}")

    # Process all JSON files
    results = process_json_files(root_directory)

    if not results:
        print("No matching JSON files found!")
        print("Make sure:")
        print("1. The root directory path is correct")
        print(
            "2. JSON files exist with 'diff_end_all' or 'diff_start_all' in their names"
        )
        print("3. Folder names follow the expected format")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by dataset, retrieval_model, score_function, top_k for better organization
    df = df.sort_values(
        ["dataset", "retrieval_model", "score_function", "top_k", "adv_pos"]
    )

    # Save to CSV
    output_file = "quantile_analysis_results.csv"
    df.to_csv(output_file, index=False)

    print("\nAnalysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total files processed: {len(results)}")

    # Display summary
    print("\nSummary by dataset:")
    summary = df.groupby(["dataset", "adv_pos"]).size().reset_index(name="file_count")
    print(summary.to_string(index=False))

    # Display first few rows
    print("\nFirst 5 rows of results:")
    print(
        df[
            [
                "adv_pos",
                "5th_highest_quantile_95th_percentile",
                "10th_highest_quantile_90th_percentile",
            ]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
