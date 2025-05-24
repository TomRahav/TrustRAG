import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np  # For NaN values


def parse_log_file(file_path):
    """
    Extracts data from a single log file.
    """
    desired_patterns = {
        "success_injection_rate": r"Success injection rate in top 5 contents: (\d+\.\d+%)",
        "inference_time": r"Total run time for inference, removal and defense: (\d+\.\d+)",
        "correct_answer_pct": r"Correct Answer Percentage: (\d+\.\d+%)",
        "incorrect_answer_pct": r"Incorrect Answer Percentage: (\d+\.\d+%)",
        "false_removal_rate": r"False removal rate in true content contents: (\d+\.\d+%)",
        "success_removal_rate": r"Success removal rate in adv contents: (\d+\.\d+%)",
    }

    data = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        content = "".join(lines)

        # Extract experiment name
        experiment_name = None
        for line in lines:
            if "Experiment Name --" in line:
                match = re.search(r"Experiment Name --\s*(.*)", line)
                if match:
                    experiment_name = match.group(1).strip()
                break

        data["experiment_full_name"] = (
            experiment_name  # Keep full name for debugging/context
        )

        # Extract components from the experiment name
        if experiment_name:
            patterns = {
                "dataset": r"dataset_([^-\s]+)",
                "eval_model": r"retriever_([^-\s]+)",  # Corrected from retriver to retriever
                "model": r"model_(?:true_|false_)?([^-\s]+)",
                "attack": r"attack_([^-\s]+)",
                "removal": r"removal_([^-\s]+)",
                "adv_a_position": r"adv_a_position_([^-\s]+)",
                "score": r"-(dot|cos)",
                "adv_per_query": r"adv_per_query(\d+)",
                "M": r"-M(\d+)",  # Added M back based on folder name
                "repeat": r"Repeat(\d+)",  # Added Repeat back based on folder name
                "no_questions": r"no_questions",
                "both_sides": r"both_sides",
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, experiment_name)
                if key in ["no_questions", "both_sides"]:
                    data[key] = 1 if match else 0
                else:
                    data[key] = match.group(1) if match else None
        else:
            # If experiment name is not found, set components to None
            for key in [
                "dataset",
                "eval_model",
                "model",
                "score",
                "attack",
                "adv_a_position",
                "removal",
                "adv_per_query",
                "M",
                "repeat",
                "no_questions",
                "both_sides",
            ]:
                data[key] = None

        # Extract other desired patterns
        for key, pattern in desired_patterns.items():
            match = re.search(pattern, content)
            data[key] = match.group(1) if match else None

    return data


def clean_and_convert_data(df):
    """
    Cleans the DataFrame and converts percentage strings to floats,
    multiplying by 100 if the header contains 'rate' or 'pct'.
    """
    # Columns that represent percentages (and will be multiplied by 100)
    percentage_cols = [
        "success_injection_rate",
        "correct_answer_pct",
        "incorrect_answer_pct",
        "false_removal_rate",
        "success_removal_rate",
    ]

    for col in percentage_cols:
        if col in df.columns:  # Check if column exists in the DataFrame
            # Fill None/empty strings with NaN first, then apply conversion
            df[col] = df[col].replace("", np.nan).str.replace("%", "", regex=False)
            # Convert to float, coercing errors (e.g., if still non-numeric)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Multiply by 100
            if "rate" in col:
                df[col] = df[col] * 100.0

    # Convert inference_time to float
    if "inference_time" in df.columns:
        df["inference_time"] = pd.to_numeric(df["inference_time"], errors="coerce")

    # Convert numeric columns that might have None
    for col in ["adv_per_query", "M", "repeat", "no_questions", "both_sides"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def analyze_and_visualize(df, output_dir="plots"):  # Added output_dir argument
    """
    Performs analysis and generates visualizations.
    """
    print("\n--- Basic Statistics ---")
    print(df.describe().to_string())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Focus on 'hotflip' attack for comparison
    df_hotflip = df[df["attack"] == "hotflip"].copy()

    if df_hotflip.empty:
        print("\nNo 'hotflip' attack data found for comparison.")
        return

    # Group by relevant columns to compare removal methods
    comparison_keys = [
        "dataset",
        "eval_model",
        "model",
        "score",
        "adv_a_position",
        "adv_per_query",
        "M",
        "repeat",
        "no_questions",
        "both_sides",
    ]

    # Filter for only 'drift' and 'kmeans_ngram' removal methods
    df_filtered_removal = df_hotflip[
        df_hotflip["removal"].isin(["drift", "kmeans_ngram"])
    ].copy()

    if df_filtered_removal.empty:
        print("\nNo 'drift' or 'kmeans_ngram' removal data for 'hotflip' attack found.")
        return

    # Identify common experiment setups to ensure fair comparison
    # This creates a unique identifier for each experiment setup *excluding* the removal method
    # Use fillna('') to treat NaN as empty string for id creation, ensuring consistent grouping
    df_filtered_removal["experiment_setup_id"] = (
        df_filtered_removal[comparison_keys]
        .astype(str)
        .fillna("NA")
        .agg("_".join, axis=1)
    )

    # Count how many removal methods exist for each setup ID
    setup_counts = df_filtered_removal.groupby("experiment_setup_id")[
        "removal"
    ].nunique()

    # Filter for setups that have both 'drift' and 'kmeans_ngram'
    complete_setups = setup_counts[setup_counts == 2].index
    df_comparable = df_filtered_removal[
        df_filtered_removal["experiment_setup_id"].isin(complete_setups)
    ].copy()

    if df_comparable.empty:
        print(
            "\nNo complete pairs of 'drift' and 'kmeans_ngram' experiments found for comparison."
        )
        print(
            "Consider checking if all relevant columns (M, repeat, no_questions, both_sides) are correctly extracted and present."
        )
        return

    print("\n--- Comparison of Drift vs. Kmeans_Ngram (Hotflip Attack) ---")
    # Group by the defining characteristics and the removal method
    comparison_metrics = [
        "success_injection_rate",
        "correct_answer_pct",
        "false_removal_rate",
        "success_removal_rate",
        "inference_time",
    ]

    # Aggregate and then pivot for clear comparison
    # Drop rows where all comparison metrics are NaN for a given group to avoid entirely empty comparisons
    comparison_summary = (
        df_comparable.groupby(comparison_keys + ["removal"])[comparison_metrics]
        .mean()
        .dropna(how="all")
        .unstack(level="removal")
    )

    print(comparison_summary.to_string())

    # --- Visualizations ---
    print("\n--- Generating Visualizations ---")
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it doesn't exist
    sns.set_style("whitegrid")

    # Plotting loop for key metrics
    for metric in comparison_metrics:
        # Check if the metric column has any non-NaN data before plotting
        if df_comparable[metric].dropna().empty:
            print(
                f"Skipping plot for '{metric}' as all values are missing in comparable data."
            )
            continue

        plt.figure(figsize=(14, 7))
        # Ensure a unique identifier for each x-axis group, e.g., combining dataset and model
        df_comparable["display_group"] = (
            df_comparable["dataset"]
            + "-"
            + df_comparable["model"]
            + "-"
            + df_comparable["score"]
            + "-"
            + df_comparable["adv_a_position"]
            + "-"
            + df_comparable["adv_per_query"].astype(str)
            + (
                ("-M" + df_comparable["M"].astype(str))
                if "M" in df_comparable.columns and df_comparable["M"].notna().any()
                else ""
            )
            + (
                ("-R" + df_comparable["repeat"].astype(str))
                if "repeat" in df_comparable.columns
                and df_comparable["repeat"].notna().any()
                else ""
            )
        )

        # Sort for better visualization
        df_comparable_sorted = df_comparable.sort_values(
            by=["dataset", "model", "score", "adv_a_position"]
        )

        sns.barplot(
            x="display_group",
            y=metric,
            hue="removal",
            data=df_comparable_sorted,
            palette={"drift": "skyblue", "kmeans_ngram": "salmon"},
        )
        plt.title(
            f'Comparison of Mean {metric.replace("_", " ").title()} by Removal Method (Hotflip Attack)'
        )
        plt.xlabel("Experiment Group (Dataset-Model-Score-AdvPosition-AdvPerQuery...)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=90, fontsize=8)
        plt.legend(title="Removal Method")
        plt.tight_layout()

        # Save the plot instead of showing it
        plot_filename = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(plot_filename)
        print(f"Saved plot to {plot_filename}")
        plt.close()  # Close the plot to free up memory


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze RAG experiment log data."
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="/home/tom.rahav/TrustRAG/logs",
        help="Path to the directory containing log files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/tom.rahav/TrustRAG/notebooks/extracted_logs_data.csv",
        help="Path to save the extracted CSV data.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="/home/tom.rahav/TrustRAG/plots",
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    extracted_data = []
    for root, dirs, files in os.walk(args.logs_dir):
        for file in files:
            if file.endswith(".log"):
                file_path = os.path.join(root, file)
                # print(f"Processing: {file_path}") # Suppress this print to reduce console clutter
                data = parse_log_file(file_path)
                if data:  # Only append if data was successfully extracted
                    extracted_data.append(data)

    if not extracted_data:
        print("No log files found or no data extracted.")
        return

    # Dynamically get headers, ensuring 'experiment_full_name' is included
    # Also ensure consistent order for DictWriter
    all_keys = set()
    for row in extracted_data:
        all_keys.update(row.keys())
    headers = sorted(list(all_keys))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in extracted_data:
            writer.writerow(row)

    print(f"Data extraction complete. Results saved to '{args.output_csv}'.")

    # --- Start Analysis ---
    print("\n--- Starting Data Analysis ---")
    df = pd.read_csv(args.output_csv)
    df = clean_and_convert_data(df)

    # Pass the plot directory to the analysis function
    analyze_and_visualize(df, args.plot_dir)


if __name__ == "__main__":
    main()
