import logging
import os
import json
import torch
import transformers
import pandas as pd
import time
from typing import Dict, Tuple
from datasets import load_dataset
import wikipedia
from urllib.parse import unquote
import re

from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from src.contriever_src.contriever import Contriever
from src.contriever_src.beir_utils import DenseEncoderModel
from src.utils import model_code_to_cmodel_name

import argparse

parser = argparse.ArgumentParser(
    description="Evaluate retrieval models on MIRAGE dataset"
)

parser.add_argument(
    "--model_code",
    type=str,
    default="contriever",
    help="Model code (contriever, ance, minilm, mpnet, roberta, etc.)",
)
parser.add_argument(
    "--score_function",
    type=str,
    default="dot",
    choices=["dot", "cos_sim"],
    help="Scoring function for similarity computation",
)
parser.add_argument(
    "--top_k", type=int, default=100, help="Number of top results to retrieve"
)

# MIRAGE dataset arguments
parser.add_argument(
    "--mirage_split",
    type=str,
    default="train",
    choices=["train"],
    help="MIRAGE dataset split to evaluate",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="Directory to cache the downloaded MIRAGE dataset",
)
parser.add_argument(
    "--wikipedia_cache",
    type=str,
    default="./wikipedia_cache",
    help="Directory to cache Wikipedia content",
)
parser.add_argument(
    "--max_doc_length",
    type=int,
    default=512,
    help="Maximum length of document text (in words)",
)
parser.add_argument(
    "--fetch_delay",
    type=float,
    default=0.1,
    help="Delay between Wikipedia API calls (seconds)",
)

parser.add_argument(
    "--result_output",
    default="results/mirage_results/debug.json",
    type=str,
    help="Output file for results",
)

parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument(
    "--per_gpu_batch_size",
    default=64,
    type=int,
    help="Batch size per GPU/CPU for indexing.",
)
parser.add_argument("--max_length", type=int, default=128)

args = parser.parse_args()


class WikipediaFetcher:
    """Helper class to fetch Wikipedia content from URLs"""

    def __init__(
        self,
        cache_dir: str = "./wikipedia_cache",
        max_length: int = 512,
        delay: float = 0.1,
    ):
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.delay = delay
        os.makedirs(cache_dir, exist_ok=True)

        # Set up Wikipedia API
        wikipedia.set_rate_limiting(True, min_wait=delay)

    def url_to_title(self, url: str) -> str:
        """Extract Wikipedia page title from URL"""
        if "wikipedia.org/wiki/" in url:
            title = url.split("/wiki/")[-1]
            title = unquote(title)  # Decode URL encoding
            title = title.replace("_", " ")
            return title
        return url

    def get_cached_content(self, title: str) -> str:
        """Get content from cache if available"""
        cache_file = os.path.join(self.cache_dir, f"{title.replace('/', '_')}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def save_to_cache(self, title: str, content: str):
        """Save content to cache"""
        cache_file = os.path.join(self.cache_dir, f"{title.replace('/', '_')}.txt")
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(content)

    def fetch_wikipedia_content(self, url_or_title: str) -> Dict[str, str]:
        """
        Fetch Wikipedia content from URL or title
        Returns dict with 'title' and 'text' keys
        """
        title = self.url_to_title(url_or_title)

        # Check cache first
        cached_content = self.get_cached_content(title)
        if cached_content:
            return {"title": title, "text": cached_content}

        try:
            # Try to fetch the page
            time.sleep(self.delay)  # Rate limiting
            page = wikipedia.page(title, auto_suggest=True)

            # Get summary and first few sections for better coverage
            content = page.summary

            # Add first section if available and not too long
            if hasattr(page, "content") and len(content.split()) < self.max_length:
                full_content = page.content
                # Take first part of full content
                words = full_content.split()
                if len(words) > len(content.split()):
                    remaining_words = self.max_length - len(content.split())
                    if remaining_words > 0:
                        additional_content = " ".join(
                            words[
                                len(content.split()) : len(content.split())
                                + remaining_words
                            ]
                        )
                        content = content + " " + additional_content

            # Truncate if too long
            words = content.split()
            if len(words) > self.max_length:
                content = " ".join(words[: self.max_length])

            # Clean up content
            content = re.sub(r"\n+", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

            # Cache the content
            self.save_to_cache(title, content)

            logging.debug(f"Fetched Wikipedia content for: {title}")
            return {"title": page.title, "text": content}

        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first option
            try:
                time.sleep(self.delay)
                page = wikipedia.page(e.options[0])
                content = page.summary
                words = content.split()
                if len(words) > self.max_length:
                    content = " ".join(words[: self.max_length])

                content = re.sub(r"\n+", " ", content)
                content = re.sub(r"\s+", " ", content).strip()

                self.save_to_cache(title, content)
                logging.debug(f"Fetched Wikipedia content (disambiguated) for: {title}")
                return {"title": page.title, "text": content}
            except Exception as e2:
                logging.warning(f"Failed to fetch disambiguated page for {title}: {e2}")
                return {"title": title, "text": f"Wikipedia page not found: {title}"}

        except wikipedia.exceptions.PageError:
            logging.warning(f"Wikipedia page not found: {title}")
            # Cache the failure to avoid repeated attempts
            self.save_to_cache(title, f"Wikipedia page not found: {title}")
            return {"title": title, "text": f"Wikipedia page not found: {title}"}

        except Exception as e:
            logging.warning(f"Error fetching Wikipedia content for {title}: {e}")
            return {
                "title": title,
                "text": f"Error fetching Wikipedia content: {str(e)}",
            }


class MirageHFDataLoader:
    """Data loader for MIRAGE dataset from Hugging Face with Wikipedia content fetching"""

    def __init__(
        self,
        cache_dir: str = None,
        wikipedia_cache: str = "./wikipedia_cache",
        max_doc_length: int = 512,
        fetch_delay: float = 0.1,
    ):
        self.cache_dir = cache_dir
        self.wikipedia_fetcher = WikipediaFetcher(
            wikipedia_cache, max_doc_length, fetch_delay
        )

    def load(self, split: str = "train") -> Tuple[Dict, Dict, Dict]:
        """
        Load MIRAGE dataset from Hugging Face and fetch Wikipedia content
        Returns:
            corpus: Dict[doc_id, {"title": title, "text": text}]
            queries: Dict[query_id, query_text]
            qrels: Dict[query_id, Dict[doc_id, relevance_score]]
        """
        logging.info(f"Loading MIRAGE {split} split from Hugging Face")

        # Load dataset from Hugging Face
        try:
            dataset = load_dataset(
                "nlpai-lab/mirage", cache_dir=self.cache_dir, trust_remote_code=True
            )
        except Exception as e:
            logging.error(f"Failed to load dataset from Hugging Face: {e}")
            raise

        # Get the appropriate split
        if split not in dataset:
            available_splits = list(dataset.keys())
            logging.error(
                f"Split '{split}' not found. Available splits: {available_splits}"
            )
            raise ValueError(f"Split '{split}' not found in MIRAGE dataset")

        data_split = dataset[split]
        logging.info(f"Loaded {len(data_split)} examples from {split} split")

        # Initialize containers
        corpus = {}
        queries = {}
        qrels = {}

        # Track unique URLs to avoid duplicate fetching
        unique_urls = set()

        # First pass: collect all unique URLs
        for example in data_split:
            positive_passages = example.get("positive_passages", [])
            negative_passages = example.get("negative_passages", [])

            for passage in positive_passages + negative_passages:
                url = passage.get("url", "") or passage.get("docid", "")
                if url:
                    unique_urls.add(url)

        logging.info(f"Found {len(unique_urls)} unique Wikipedia URLs to fetch")

        # Fetch Wikipedia content for all unique URLs
        url_to_content = {}
        for i, url in enumerate(unique_urls):
            if i % 50 == 0:
                logging.info(f"Fetching Wikipedia content: {i}/{len(unique_urls)}")

            content = self.wikipedia_fetcher.fetch_wikipedia_content(url)
            url_to_content[url] = content

        logging.info("Finished fetching Wikipedia content")

        # Second pass: process examples with fetched content
        for example in data_split:
            # Extract query information
            query_id = str(example["query_id"])
            query_text = example["query"]
            queries[query_id] = query_text

            # Extract positive and negative passages
            positive_passages = example.get("positive_passages", [])
            negative_passages = example.get("negative_passages", [])

            # Initialize qrels for this query
            if query_id not in qrels:
                qrels[query_id] = {}

            # Process positive passages
            for passage in positive_passages:
                url = passage.get("url", "") or passage.get("docid", "")
                doc_id = str(passage.get("docid", url))

                if url in url_to_content:
                    content = url_to_content[url]
                    corpus[doc_id] = {
                        "title": content["title"],
                        "text": content["text"],
                    }
                else:
                    # Fallback if URL not found
                    corpus[doc_id] = {
                        "title": passage.get("title", ""),
                        "text": passage.get(
                            "text", f"Wikipedia content not available for: {url}"
                        ),
                    }

                # Add positive relevance (score = 1)
                qrels[query_id][doc_id] = 1

            # Process negative passages
            for passage in negative_passages:
                url = passage.get("url", "") or passage.get("docid", "")
                doc_id = str(passage.get("docid", url))

                if url in url_to_content:
                    content = url_to_content[url]
                    corpus[doc_id] = {
                        "title": content["title"],
                        "text": content["text"],
                    }
                else:
                    # Fallback if URL not found
                    corpus[doc_id] = {
                        "title": passage.get("title", ""),
                        "text": passage.get(
                            "text", f"Wikipedia content not available for: {url}"
                        ),
                    }

                # Add negative relevance (score = 0)
                qrels[query_id][doc_id] = 0

        logging.info(
            f"Processed: {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels"
        )

        # Validate the data
        total_qrels = sum(len(qrels[qid]) for qid in qrels)
        logging.info(f"Total query-document pairs: {total_qrels}")

        return corpus, queries, qrels


def compress(results: Dict, top_k: int = 2000) -> Dict:
    """Compress results to top-k for each query"""
    for y in results:
        k_old = len(results[y])
        break

    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:top_k]:
            sub_results[query_id][c_id] = s

    for y in sub_results:
        k_new = len(sub_results[y])
        break

    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results


def load_retrieval_model(args):
    """Load the specified retrieval model"""
    logging.info(f"Loading model: {args.model_code}")

    if "contriever" in args.model_code:
        encoder = Contriever.from_pretrained(
            model_code_to_cmodel_name[args.model_code]
        ).cuda()
        tokenizer = transformers.BertTokenizerFast.from_pretrained(
            model_code_to_cmodel_name[args.model_code]
        )
        model = DRES(
            DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer),
            batch_size=args.per_gpu_batch_size,
        )
    elif "dpr" in args.model_code:
        model = DRES(
            models.SentenceBERT(
                (
                    "facebook-dpr-question_encoder-multiset-base",
                    "facebook-dpr-ctx_encoder-multiset-base",
                    " [SEP] ",
                ),
                batch_size=args.per_gpu_batch_size,
            )
        )
    elif any(
        substring in args.model_code
        for substring in ["ance", "minilm", "mpnet", "roberta"]
    ):
        model = DRES(
            models.SentenceBERT(model_code_to_cmodel_name[args.model_code]),
            batch_size=args.per_gpu_batch_size,
            corpus_chunk_size=10000,
        )
    else:
        raise NotImplementedError(f"Model {args.model_code} not implemented")

    logging.info(f"Model loaded: {model.model}")
    return model


def main():
    #### Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    logging.info(f"Arguments: {args}")

    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load MIRAGE dataset from Hugging Face with Wikipedia content fetching
    data_loader = MirageHFDataLoader(
        cache_dir=args.cache_dir,
        wikipedia_cache=args.wikipedia_cache,
        max_doc_length=args.max_doc_length,
        fetch_delay=args.fetch_delay,
    )
    corpus, queries, qrels = data_loader.load(split=args.mirage_split)

    # Load retrieval model
    model = load_retrieval_model(args)
    k_values = [20]
    # Setup retrieval evaluator
    retriever = EvaluateRetrieval(
        model, score_function=args.score_function, k_values=k_values
    )

    # Perform retrieval
    logging.info("Starting retrieval...")
    results = retriever.retrieve(corpus, queries)

    # Evaluate results
    logging.info("Evaluating results...")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

    # Print evaluation metrics
    logging.info("=== EVALUATION RESULTS ===")
    for k in k_values:
        logging.info(f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
        logging.info(f"MAP@{k}: {_map[f'MAP@{k}']:.4f}")
        logging.info(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")
        logging.info(f"P@{k}: {precision[f'P@{k}']:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.result_output), exist_ok=True)

    # Compress and save retrieval results
    sub_results = compress(results, top_k=2000)
    with open(args.result_output, "w") as f:
        json.dump(sub_results, f, indent=2)
    output_data = {
        "model": args.model_code,
        "score_function": args.score_function,
        "dataset": "mirage",
        "split": args.mirage_split,
        "metrics": {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
        },
    }

    with open(args.result_output.replace(".json", "_summary.json"), "w") as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"Results saved to: {args.result_output}")

    # Also save a summary CSV for easy analysis
    summary_file = args.result_output.replace(".json", "_summary.csv")
    summary_data = []
    for k in k_values:
        summary_data.append(
            {
                "model": args.model_code,
                "score_function": args.score_function,
                "k": k,
                "NDCG": ndcg[f"NDCG@{k}"],
                "MAP": _map[f"MAP@{k}"],
                "Recall": recall[f"Recall@{k}"],
                "Precision": precision[f"P@{k}"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
