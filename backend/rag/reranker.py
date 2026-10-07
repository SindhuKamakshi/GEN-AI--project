from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker for CodeLens AI.

    The reranker takes the query and the candidate chunks
    produced by the hybrid RRF search, then scores how
    relevant each candidate is to the query.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        print(f"Loading reranker model: {self.MODEL_NAME}")

        self.model = CrossEncoder(
            self.MODEL_NAME
        )

        print("Reranker model loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5
    ) -> list[dict]:
        """
        Rerank candidate search results using a cross-encoder.

        Args:
            query:
                User's search question.

            candidates:
                Candidate results produced by hybrid search.

            top_k:
                Number of final results to return.

        Returns:
            Reranked list of candidates.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        pairs = [
            [
                query,
                candidate.get("content", "")
            ]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        reranked = []

        for candidate, score in zip(candidates, scores):
            result = dict(candidate)

            result["reranker_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True
        )

        return reranked[:top_k]