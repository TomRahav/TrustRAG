import argparse
import json
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

from src.utils import load_beir_datasets, load_cached_data, load_models, save_outputs
from src.defend_module import drift_filtering_question_aware


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run drift filtering on a BEIR dataset"
    )
    parser.add_argument(
        "--eval_dataset", type=str, required=True, help="Dataset name, e.g., nq"
    )
    parser.add_argument(
        "--eval_model_code",
        type=str,
        required=True,
        help="Model code, e.g., contriever",
    )
    parser.add_argument(
        "--score_function",
        type=str,
        choices=["dot", "cos_sim"],
        default="dot",
        help="Scoring function",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Top-k passages to consider"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split = "test"
    device = "cuda"

    # Load model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.to(device).eval()

    # Load dataset
    corpus, queries, qrels = load_cached_data(
        f"data_cache/{args.eval_dataset}_{split}.pkl",
        load_beir_datasets,
        args.eval_dataset,
        split,
    )

    log_args = SimpleNamespace(
        log_name=f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-extract_val",
        score_function=args.score_function,
    )

    # Load retrieval results
    result_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}"
    if args.score_function == "cos_sim":
        result_path += "-cos"
    result_path += ".json"

    with open(result_path, "r") as f:
        results = json.load(f)

    # Run drift filtering
    total_removed = 0
    total_queries = 0
    drift_diffs = []
    diff_start_all = []
    diff_end_all = []

    for query_id, topk_dict in tqdm(results.items(), desc="Evaluating drift impact"):
        question = queries[query_id]
        topk_indices = list(topk_dict.keys())[: args.top_k]
        original_docs = [corpus[doc_id]["text"] for doc_id in topk_indices]

        try:
            filtered_docs, diff_start, diff_end = drift_filtering_question_aware(
                args=log_args,
                tokenizer=tokenizer,
                model=model,
                get_emb=get_emb,
                question=question,
                contents=original_docs,
                score=args.score_function,
            )
        except Exception as e:
            print(f"Skipping {query_id} due to error: {e}")
            continue

        removed = len(set(original_docs) - set(filtered_docs))
        drift_diffs.append(removed)
        diff_start_all = np.concatenate((diff_start_all, diff_start))
        diff_end_all = np.concatenate((diff_end_all, diff_end))
        total_removed += removed
        total_queries += 1

    # Report
    avg_removed = total_removed / total_queries if total_queries > 0 else 0
    print("\n[Contriever Drift Filter Summary]")
    print(f"Total queries processed: {total_queries}")
    print(
        f"Average documents removed by drift: {avg_removed:.2f} out of top-{args.top_k}"
    )

    save_outputs(
        diff_start_all,
        log_args.log_name,
        f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-diff_start_all",
    )
    save_outputs(
        diff_end_all,
        log_args.log_name,
        f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-diff_end_all",
    )


if __name__ == "__main__":
    main()
