#!/usr/bin/env python3
"""
Adversarial Filter Comparison Analysis Script

This script compares adversarial removal methods (drift vs kmeans_ngram) across datasets
to identify common questions and adversarial sentences that pass both filters.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np


def load_adv_passed_data(json_path):
    """Load adversarial passed removal data from JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return {}


def extract_experiment_info(folder_name):
    """Extract experiment information from folder name."""
    parts = folder_name.split("_")

    # Find removal method
    removal_method = None
    if "drift-defend" in folder_name:
        removal_method = "drift"
    elif "kmeans_ngram-defend" in folder_name:
        removal_method = "kmeans_ngram"

    # Extract other parameters
    info = {
        "removal_method": removal_method,
        "model": None,
        "attack_type": None,
        "seed": None,
    }

    # Extract model
    if "gpt-4o" in folder_name:
        info["model"] = "gpt-4o"
    elif "Llama-3.1-8B-Instruct" in folder_name:
        info["model"] = "meta-llama"
    elif "Mistral-Nemo-Instruct" in folder_name:
        info["model"] = "mistralai"

    # Extract attack type
    if "hotflip" in folder_name:
        info["attack_type"] = "hotflip"

    # Extract seed
    for part in parts:
        if "Seed_" in part:
            try:
                info["seed"] = int(part.split("_")[1])
            except:
                pass

    return info


def find_matching_experiments(outputs_dir):
    """Find pairs of experiments that differ only in removal method."""
    experiments = defaultdict(list)

    for dataset_folder in os.listdir(outputs_dir):
        if not dataset_folder.startswith("dataset_"):
            continue

        dataset_path = os.path.join(outputs_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        dataset_name = dataset_folder.replace("dataset_", "").split("-")[0]

        for exp_folder in os.listdir(dataset_path):
            exp_path = os.path.join(dataset_path, exp_folder)
            json_path = os.path.join(exp_path, "adv_passed_removal.json")

            if os.path.exists(json_path):
                exp_info = extract_experiment_info(exp_folder)
                if exp_info["removal_method"]:
                    # Create a key without removal method to group similar experiments
                    key_parts = [
                        dataset_name,
                        exp_info["model"],
                        exp_info["attack_type"],
                        str(exp_info["seed"]),
                    ]
                    key = "_".join([p for p in key_parts if p is not None])

                    experiments[key].append(
                        {
                            "dataset": dataset_name,
                            "path": json_path,
                            "info": exp_info,
                            "full_name": exp_folder,
                        }
                    )

    # Filter to only experiments that have both drift and kmeans_ngram
    matched_pairs = {}
    for key, exp_list in experiments.items():
        methods = {exp["info"]["removal_method"] for exp in exp_list}
        if "drift" in methods and "kmeans_ngram" in methods:
            matched_pairs[key] = exp_list

    return matched_pairs


def analyze_common_questions(drift_data, kmeans_data):
    """Analyze common questions between two removal methods."""
    drift_qids = set(drift_data.keys())
    kmeans_qids = set(kmeans_data.keys())

    common_qids = drift_qids.intersection(kmeans_qids)

    results = {
        "drift_total": len(drift_qids),
        "kmeans_total": len(kmeans_qids),
        "common_total": len(common_qids),
        "drift_only": len(drift_qids - kmeans_qids),
        "kmeans_only": len(kmeans_qids - kmeans_qids),
        "common_details": {},
    }

    # Analyze common questions in detail
    for qid in common_qids:
        drift_sentences = drift_data[qid]
        kmeans_sentences = kmeans_data[qid]

        # Convert to sets for comparison (excluding the original question which is first)
        drift_adv = set(drift_sentences[1:]) if len(drift_sentences) > 1 else set()
        kmeans_adv = set(kmeans_sentences[1:]) if len(kmeans_sentences) > 1 else set()

        common_adv = drift_adv.intersection(kmeans_adv)

        results["common_details"][qid] = {
            "original_question": drift_sentences[0] if drift_sentences else "",
            "drift_adv_count": len(drift_adv),
            "kmeans_adv_count": len(kmeans_adv),
            "common_adv_count": len(common_adv),
            "common_adv_sentences": list(common_adv),
        }

    return results


def create_summary_statistics(all_results):
    """Create summary statistics across all experiment pairs."""
    summary_data = []

    for exp_key, result in all_results.items():
        parts = exp_key.split("_")
        summary_data.append(
            {
                "experiment": exp_key,
                "dataset": parts[0] if parts else "unknown",
                "model": parts[1] if len(parts) > 1 else "unknown",
                "drift_total": result["drift_total"],
                "kmeans_total": result["kmeans_total"],
                "common_total": result["common_total"],
                "drift_only": result["drift_only"],
                "kmeans_only": result["kmeans_only"],
                "common_percentage": (
                    (
                        result["common_total"]
                        / max(result["drift_total"], result["kmeans_total"])
                        * 100
                    )
                    if max(result["drift_total"], result["kmeans_total"]) > 0
                    else 0
                ),
                "overlap_jaccard": (
                    result["common_total"]
                    / (
                        result["drift_total"]
                        + result["kmeans_total"]
                        - result["common_total"]
                    )
                    if (
                        result["drift_total"]
                        + result["kmeans_total"]
                        - result["common_total"]
                    )
                    > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(summary_data)


def create_visualizations(summary_df, output_dir="analysis_output"):
    """Create visualizations for the analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Bar plot of common questions by dataset
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Common questions by dataset
    dataset_summary = (
        summary_df.groupby("dataset")
        .agg({"drift_total": "mean", "kmeans_total": "mean", "common_total": "mean"})
        .reset_index()
    )

    ax1 = axes[0, 0]
    x = np.arange(len(dataset_summary))
    width = 0.25

    ax1.bar(
        x - width, dataset_summary["drift_total"], width, label="Drift Only", alpha=0.8
    )
    ax1.bar(x, dataset_summary["kmeans_total"], width, label="K-means Only", alpha=0.8)
    ax1.bar(
        x + width, dataset_summary["common_total"], width, label="Common", alpha=0.8
    )

    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Average Number of Questions")
    ax1.set_title("Average Questions Passed by Removal Method")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_summary["dataset"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Overlap percentage by dataset
    ax2 = axes[0, 1]
    dataset_overlap = summary_df.groupby("dataset")["common_percentage"].mean()
    bars = ax2.bar(dataset_overlap.index, dataset_overlap.values, alpha=0.8)
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Average Overlap Percentage (%)")
    ax2.set_title("Average Question Overlap Between Methods")
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    # 3. Jaccard similarity by model
    ax3 = axes[1, 0]
    if "model" in summary_df.columns:
        model_jaccard = summary_df.groupby("model")["overlap_jaccard"].mean()
        bars = ax3.bar(model_jaccard.index, model_jaccard.values, alpha=0.8)
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Average Jaccard Similarity")
        ax3.set_title("Method Similarity by Model")
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    # 4. Distribution of overlap percentages
    ax4 = axes[1, 1]
    ax4.hist(summary_df["common_percentage"], bins=15, alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Overlap Percentage (%)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Question Overlap Percentages")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "adversarial_comparison_overview.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Create detailed heatmap
    if len(summary_df) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_data = summary_df.pivot_table(
            values="common_percentage", index="dataset", columns="model", aggfunc="mean"
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Overlap Percentage (%)"},
        )
        ax.set_title("Question Overlap Heatmap: Dataset vs Model")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "overlap_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


def save_detailed_analysis(all_results, output_dir="analysis_output"):
    """Save detailed analysis to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed common questions analysis
    detailed_output = {}
    for exp_key, result in all_results.items():
        detailed_output[exp_key] = {
            "summary": {
                "drift_total": result["drift_total"],
                "kmeans_total": result["kmeans_total"],
                "common_total": result["common_total"],
                "overlap_percentage": (
                    (
                        result["common_total"]
                        / max(result["drift_total"], result["kmeans_total"])
                        * 100
                    )
                    if max(result["drift_total"], result["kmeans_total"]) > 0
                    else 0
                ),
            },
            "common_questions_sample": {
                qid: details
                for qid, details in list(result["common_details"].items())[
                    :5
                ]  # First 5 for brevity
            },
        }

    with open(
        os.path.join(output_dir, "detailed_analysis.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(detailed_output, f, indent=2, ensure_ascii=False)


def main():
    """Main analysis function."""
    outputs_dir = "data_cache/outputs"  # Adjust path as needed

    if not os.path.exists(outputs_dir):
        print(f"Error: {outputs_dir} directory not found!")
        print(
            "Please ensure the script is run from the correct directory or update the outputs_dir path."
        )
        return

    print("🔍 Finding matching experiment pairs...")
    matched_experiments = find_matching_experiments(outputs_dir)

    if not matched_experiments:
        print("❌ No matching experiment pairs found!")
        print(
            "Make sure you have experiments with both 'drift-defend' and 'kmeans_ngram-defend' methods."
        )
        return

    print(f"✅ Found {len(matched_experiments)} experiment pairs to compare")

    all_results = {}

    for exp_key, exp_list in matched_experiments.items():
        print(f"\n📊 Analyzing experiment group: {exp_key}")

        # Find drift and kmeans experiments
        drift_exp = next(
            (exp for exp in exp_list if exp["info"]["removal_method"] == "drift"), None
        )
        kmeans_exp = next(
            (
                exp
                for exp in exp_list
                if exp["info"]["removal_method"] == "kmeans_ngram"
            ),
            None,
        )

        if not drift_exp or not kmeans_exp:
            print(f"⚠️  Skipping {exp_key}: missing drift or kmeans experiment")
            continue

        # Load data
        drift_data = load_adv_passed_data(drift_exp["path"])
        kmeans_data = load_adv_passed_data(kmeans_exp["path"])

        if not drift_data or not kmeans_data:
            print(f"⚠️  Skipping {exp_key}: failed to load data")
            continue

        # Analyze common questions
        results = analyze_common_questions(drift_data, kmeans_data)
        all_results[exp_key] = results

        # Print summary
        print(f"   Drift method: {results['drift_total']} questions passed")
        print(f"   K-means method: {results['kmeans_total']} questions passed")
        print(f"   Common questions: {results['common_total']}")
        overlap_pct = (
            (
                results["common_total"]
                / max(results["drift_total"], results["kmeans_total"])
                * 100
            )
            if max(results["drift_total"], results["kmeans_total"]) > 0
            else 0
        )
        print(f"   Overlap percentage: {overlap_pct:.1f}%")

    if not all_results:
        print("❌ No valid results to analyze!")
        return

    print("\n📈 Creating summary statistics and visualizations...")

    # Create summary statistics
    summary_df = create_summary_statistics(all_results)

    # Save summary to CSV
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(
        os.path.join(output_dir, "experiment_comparison_summary.csv"), index=False
    )

    # Create visualizations
    create_visualizations(summary_df, output_dir)

    # Save detailed analysis
    save_detailed_analysis(all_results, output_dir)

    # Print final summary
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved to '{output_dir}' directory:")
    print(f"   - experiment_comparison_summary.csv: Summary statistics")
    print(f"   - detailed_analysis.json: Detailed common questions analysis")
    print(f"   - adversarial_comparison_overview.png: Overview visualizations")
    print(f"   - overlap_heatmap.png: Heatmap visualization")

    # Print key insights
    print(f"\n🔑 Key Insights:")
    avg_overlap = summary_df["common_percentage"].mean()
    print(f"   - Average overlap percentage: {avg_overlap:.1f}%")

    best_overlap = summary_df.loc[summary_df["common_percentage"].idxmax()]
    print(
        f"   - Highest overlap: {best_overlap['experiment']} ({best_overlap['common_percentage']:.1f}%)"
    )

    by_dataset = (
        summary_df.groupby("dataset")["common_percentage"]
        .mean()
        .sort_values(ascending=False)
    )
    print(
        f"   - Dataset with highest avg overlap: {by_dataset.index[0]} ({by_dataset.iloc[0]:.1f}%)"
    )


if __name__ == "__main__":
    main()
