"""
Retrieval evaluation metrics for CodeLens AI.

Metrics:
- Hit Rate@K
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
"""


def hit_rate_at_k(
    retrieved_items: list[str],
    relevant_items: set[str],
    k: int,
) -> float:
    """
    Return 1.0 if at least one relevant item
    appears in the top-k results, otherwise 0.0.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if not relevant_items:
        return 0.0

    top_k = retrieved_items[:k]

    return 1.0 if any(
        item in relevant_items
        for item in top_k
    ) else 0.0


def recall_at_k(
    retrieved_items: list[str],
    relevant_items: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K =
    relevant retrieved items in top-k /
    total relevant items
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if not relevant_items:
        return 0.0

    top_k = retrieved_items[:k]

    retrieved_relevant = sum(
        1
        for item in top_k
        if item in relevant_items
    )

    return retrieved_relevant / len(
        relevant_items
    )


def precision_at_k(
    retrieved_items: list[str],
    relevant_items: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K =
    relevant retrieved items in top-k /
    number of retrieved items in top-k
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    top_k = retrieved_items[:k]

    if not top_k:
        return 0.0

    retrieved_relevant = sum(
        1
        for item in top_k
        if item in relevant_items
    )

    return retrieved_relevant / len(top_k)


def reciprocal_rank(
    retrieved_items: list[str],
    relevant_items: set[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    Returns:
        1 / rank of the first relevant result.

    Returns 0.0 if no relevant result is found.
    """

    if not relevant_items:
        return 0.0

    for rank, item in enumerate(
        retrieved_items,
        start=1,
    ):
        if item in relevant_items:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    results: list[list[str]],
    relevant_items: list[set[str]],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    across multiple queries.
    """

    if len(results) != len(relevant_items):
        raise ValueError(
            "results and relevant_items "
            "must have the same length."
        )

    if not results:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(
            retrieved,
            relevant,
        )
        for retrieved, relevant
        in zip(results, relevant_items)
    ]

    return sum(
        reciprocal_ranks
    ) / len(reciprocal_ranks)