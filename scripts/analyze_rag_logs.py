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


def analyze_and_visualize(df, output_dir="plots"):
    """
    Performs analysis and generates visualizations.
    Plots are generated per metric, with one plot for 'dot' score_function and
    one for 'cos' score_function. Within each plot, experiment groups are
    ordered by adv_per_query (1,1,1..., then 3,3,3..., then 5,5,5...).
    X-axis labels are formatted for improved readability.
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
        "score",  # This corresponds to score_function
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
    existing_comparison_keys = [
        key for key in comparison_keys if key in df_filtered_removal.columns
    ]
    if not all(key in existing_comparison_keys for key in ["score", "adv_per_query"]):
        print(
            "\nCritical error: 'score' or 'adv_per_query' columns are missing from the DataFrame after filtering."
        )
        print(
            "Cannot proceed with experiment_setup_id creation or plotting as specified."
        )
        return

    df_filtered_removal["experiment_setup_id"] = (
        df_filtered_removal[existing_comparison_keys]
        .astype(str)
        .fillna("NA")
        .agg("_".join, axis=1)
    )

    setup_counts = df_filtered_removal.groupby("experiment_setup_id")[
        "removal"
    ].nunique()

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
    comparison_metrics = [
        "success_injection_rate",
        "correct_answer_pct",
        "false_removal_rate",
        "success_removal_rate",
        "inference_time",
    ]

    final_comparison_keys = [
        key for key in comparison_keys if key in df_comparable.columns
    ]

    comparison_summary = (
        df_comparable.groupby(final_comparison_keys + ["removal"])[comparison_metrics]
        .mean()
        .dropna(how="all")
        .unstack(level="removal")
    )
    print(comparison_summary.to_string())

    # --- Visualizations ---
    print("\n--- Generating Visualizations ---")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    if "score" not in df_comparable.columns:
        print(
            "Error: 'score' column (for score_function) not found in df_comparable. Cannot generate split plots."
        )
        return
    if "adv_per_query" not in df_comparable.columns:
        print(
            "Error: 'adv_per_query' column not found in df_comparable. Cannot generate split plots."
        )
        return

    score_function_values_to_iterate = ["dot", "cos"]

    # Define the sort keys. 'adv_per_query' is now prioritized to achieve 1,1,..,3,3,..,5,5.. ordering.
    # 'score' is also included because plots are filtered by it later.
    sort_keys = ["adv_per_query", "score", "dataset", "model", "adv_a_position"]
    existing_sort_keys = [key for key in sort_keys if key in df_comparable.columns]

    if not existing_sort_keys or "adv_per_query" not in existing_sort_keys:
        print(
            "Error: Not all necessary sort keys (especially 'adv_per_query') exist in df_comparable. Cannot proceed with sorting for plots."
        )
        return
    if len(existing_sort_keys) != len(sort_keys):
        print(
            f"Warning: Not all specified sort keys ({sort_keys}) exist in df_comparable. Using existing: {existing_sort_keys}"
        )

    # Sort df_comparable once. This sorted version will be filtered for each plot.
    # The sort order is crucial for the desired x-axis grouping.
    df_comparable_sorted = df_comparable.sort_values(by=existing_sort_keys)

    # Helper function to create formatted labels for x-axis
    def create_display_label(row):
        # Ensure all expected columns for labeling are present in the row's index
        # This is important because df_plot_data is a slice and might not always have M or repeat if they were all NaN
        label_parts = []
        if "dataset" in row.index:
            label_parts.append(f"D:{row['dataset']}")
        if "model" in row.index:
            label_parts.append(f"M:{row['model']}")
        if "adv_a_position" in row.index:
            label_parts.append(f"P:{row['adv_a_position']}")

        # Conditional parts: Mval and R
        # Check if column exists in the row's Series index AND if the value is not NaN
        # if "M" in row.index and pd.notna(row["M"]):
        #     label_parts.append(f"Mval:{row['M']}")
        # if "repeat" in row.index and pd.notna(row["repeat"]):
        #     label_parts.append(f"R:{row['repeat']}")

        if "adv_per_query" in row.index:
            label_parts.append(f"APQ:{row['adv_per_query']}")

        return " ".join(label_parts)

    for metric in comparison_metrics:
        if (
            metric not in df_comparable_sorted.columns
            or df_comparable_sorted[metric].dropna().empty
        ):
            print(
                f"Skipping plots for '{metric}' as the column is missing or all values are missing in comparable data."
            )
            continue

        for sf_value in score_function_values_to_iterate:
            # Filter data for the current score_function
            # The data is already sorted correctly by adv_per_query and then other keys.
            df_plot_data = df_comparable_sorted[
                df_comparable_sorted["score"] == sf_value
            ].copy()

            if df_plot_data.empty:
                # print(f"No data for metric '{metric}', score_function '{sf_value}'. Skipping plot.")
                continue

            plt.figure(
                figsize=(15, 7)
            )  # Adjusted figsize for potentially more x-axis groups

            # Create a display group for the x-axis using the helper function.
            df_plot_data["plot_specific_display_group"] = df_plot_data.apply(
                create_display_label, axis=1
            )

            # sns.barplot will use the order from df_plot_data, which is already sorted
            # by existing_sort_keys (adv_per_query first, then score, then others).

            sns.barplot(
                x="plot_specific_display_group",
                y=metric,
                hue="removal",
                data=df_plot_data,  # Use the filtered and pre-sorted data
                palette={"drift": "skyblue", "kmeans_ngram": "salmon"},
            )

            title_metric_name = metric.replace("_", " ").title()
            plt.title(
                f"{title_metric_name} (Hotflip) for Score Func: {sf_value}\n(X-axis groups ordered by Adv per Query: 1s, then 3s, then 5s)"
            )
            plt.xlabel(
                "Experiment Group (Dataset-Model-AdvPos-Mval-R-AdvPerQuery)"
            )  # Updated to reflect label components
            plt.ylabel(title_metric_name)
            plt.xticks(rotation=75, ha="right", fontsize=8)
            plt.legend(title="Removal Method")
            plt.tight_layout()

            plot_filename = os.path.join(
                output_dir, f"{metric}_score_{sf_value}.png"
            )  # Filename reflects the score_function
            try:
                plt.savefig(plot_filename)
                print(f"Saved plot to {plot_filename}")
            except Exception as e:
                print(f"Error saving plot {plot_filename}: {e}")
            plt.close()
    print("\n--- Finished Generating Visualizations ---")


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
