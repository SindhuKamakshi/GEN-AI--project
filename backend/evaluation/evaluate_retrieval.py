"""
Evaluate CodeLens AI retrieval quality.

This evaluates the existing SearchService without modifying it.

Metrics:
- Hit Rate@K
- Recall@K
- Precision@K
- MRR
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from main import search_service

from evaluation.dataset import EVALUATION_DATASET
from evaluation.metrics import (
    hit_rate_at_k,
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
)


REPOSITORY_ID = "419c6d657edc"
TOP_K = 5


def get_metadata(result: dict) -> dict:
    return result.get("metadata", {})


def result_to_file_identifier(result: dict) -> str:
    metadata = get_metadata(result)

    return metadata.get("file_path", "")


def result_to_entity_identifier(result: dict) -> str:
    metadata = get_metadata(result)

    file_path = metadata.get("file_path", "")
    name = metadata.get("name", "")

    if file_path and name:
        return f"{file_path}::{name}"

    return file_path


def normalize_results(
    results: list[dict],
    evaluation_type: str,
) -> list[str]:
    """
    Convert SearchService results into evaluation identifiers.

    File-level:
        Multiple chunks from the same file count once.

    Entity-level:
        Entity identifiers are preserved.

    Mixed-level:
        Entity identifiers are preserved.
    """

    identifiers = []

    for result in results:

        if evaluation_type == "file":

            identifier = result_to_file_identifier(
                result
            )

        else:

            identifier = result_to_entity_identifier(
                result
            )

        if identifier:
            identifiers.append(identifier)

    # Remove duplicate files for file-level evaluation.
    if evaluation_type == "file":

        unique_identifiers = []

        for identifier in identifiers:

            if identifier not in unique_identifiers:
                unique_identifiers.append(identifier)

        return unique_identifiers

    return identifiers


def evaluate_case(test_case: dict) -> dict:
    query = test_case["query"]

    relevant_items = test_case["relevant_items"]

    evaluation_type = test_case.get(
        "evaluation_type",
        "entity",
    )

    results = search_service.search(
        repository_id=REPOSITORY_ID,
        query=query,
        top_k=TOP_K,
    )

    retrieved_items = normalize_results(
        results=results,
        evaluation_type=evaluation_type,
    )

    return {
        "id": test_case["id"],
        "query": query,
        "evaluation_type": evaluation_type,
        "relevant_items": relevant_items,
        "retrieved_items": retrieved_items,
        "hit_rate": hit_rate_at_k(
            retrieved_items,
            relevant_items,
            TOP_K,
        ),
        "recall": recall_at_k(
            retrieved_items,
            relevant_items,
            TOP_K,
        ),
        "precision": precision_at_k(
            retrieved_items,
            relevant_items,
            TOP_K,
        ),
        "reciprocal_rank": reciprocal_rank(
            retrieved_items,
            relevant_items,
        ),
    }


def average_metric(
    results: list[dict],
    metric_name: str,
) -> float:

    if not results:
        return 0.0

    return sum(
        result[metric_name]
        for result in results
    ) / len(results)


def main():

    print("=" * 70)
    print("CodeLens AI - Retrieval Evaluation")
    print("=" * 70)

    print(
        f"Evaluation cases: {len(EVALUATION_DATASET)}"
    )

    print(
        f"Repository: {REPOSITORY_ID}"
    )

    print(
        f"Top K: {TOP_K}"
    )

    results = []

    for test_case in EVALUATION_DATASET:

        print()
        print("-" * 70)

        print(
            f"Query: {test_case['query']}"
        )

        print(
            f"Type: {test_case['evaluation_type']}"
        )

        result = evaluate_case(test_case)

        results.append(result)

        print(
            "Relevant:",
            result["relevant_items"],
        )

        print(
            "Retrieved:",
            result["retrieved_items"],
        )

        print(
            f"Hit Rate@{TOP_K}: "
            f"{result['hit_rate']}"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{result['recall']}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{result['precision']}"
        )

        print(
            "Reciprocal Rank:",
            result["reciprocal_rank"],
        )

    if not results:

        print("\nNo evaluation results.")

        return

    # =========================================================
    # Overall metrics
    # =========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 70)

    print(
        f"Hit Rate@{TOP_K}:  "
        f"{average_metric(results, 'hit_rate'):.4f}"
    )

    print(
        f"Recall@{TOP_K}:    "
        f"{average_metric(results, 'recall'):.4f}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{average_metric(results, 'precision'):.4f}"
    )

    print(
        f"MRR@{TOP_K}:       "
        f"{average_metric(results, 'reciprocal_rank'):.4f}"
    )

    # =========================================================
    # Entity-level metrics
    # =========================================================

    entity_results = [
        result
        for result in results
        if result["evaluation_type"] == "entity"
    ]

    print()
    print("=" * 70)
    print("ENTITY-LEVEL METRICS")
    print("=" * 70)

    print(
        f"Cases:             {len(entity_results)}"
    )

    print(
        f"Hit Rate@{TOP_K}:   "
        f"{average_metric(entity_results, 'hit_rate'):.4f}"
    )

    print(
        f"Recall@{TOP_K}:     "
        f"{average_metric(entity_results, 'recall'):.4f}"
    )

    print(
        f"Precision@{TOP_K}:  "
        f"{average_metric(entity_results, 'precision'):.4f}"
    )

    print(
        f"MRR@{TOP_K}:        "
        f"{average_metric(entity_results, 'reciprocal_rank'):.4f}"
    )

    # =========================================================
    # File-level metrics
    # =========================================================

    file_results = [
        result
        for result in results
        if result["evaluation_type"] == "file"
    ]

    print()
    print("=" * 70)
    print("FILE-LEVEL METRICS")
    print("=" * 70)

    print(
        f"Cases:             {len(file_results)}"
    )

    print(
        f"Hit Rate@{TOP_K}:   "
        f"{average_metric(file_results, 'hit_rate'):.4f}"
    )

    print(
        f"Recall@{TOP_K}:     "
        f"{average_metric(file_results, 'recall'):.4f}"
    )

    print(
        f"Precision@{TOP_K}:  "
        f"{average_metric(file_results, 'precision'):.4f}"
    )

    print(
        f"MRR@{TOP_K}:        "
        f"{average_metric(file_results, 'reciprocal_rank'):.4f}"
    )

    # =========================================================
    # Mixed-level metrics
    # =========================================================

    mixed_results = [
        result
        for result in results
        if result["evaluation_type"] == "mixed"
    ]

    if mixed_results:

        print()
        print("=" * 70)
        print("MIXED-LEVEL METRICS")
        print("=" * 70)

        print(
            f"Cases:             {len(mixed_results)}"
        )

        print(
            f"Hit Rate@{TOP_K}:   "
            f"{average_metric(mixed_results, 'hit_rate'):.4f}"
        )

        print(
            f"Recall@{TOP_K}:     "
            f"{average_metric(mixed_results, 'recall'):.4f}"
        )

        print(
            f"Precision@{TOP_K}:  "
            f"{average_metric(mixed_results, 'precision'):.4f}"
        )

        print(
            f"MRR@{TOP_K}:        "
            f"{average_metric(mixed_results, 'reciprocal_rank'):.4f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()