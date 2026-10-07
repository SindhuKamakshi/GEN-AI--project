from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of documents embedded in one batch.
# This prevents very large repositories from creating
# one enormous embedding request.
EMBEDDING_BATCH_SIZE = 32


class EmbeddingService:
    """
    Create embeddings for CodeLens AI code chunks.

    The service uses a local SentenceTransformer model and
    processes documents in batches for better memory usage.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE
    ):
        if batch_size <= 0:
            raise ValueError(
                "Embedding batch size must be greater than 0."
            )

        self.model_name = model_name
        self.batch_size = batch_size

        print(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        print(
            "Embedding model loaded successfully."
        )

    # ========================================================
    # DOCUMENT EMBEDDINGS
    # ========================================================

    def embed_documents(
        self,
        documents: list[str]
    ) -> list[list[float]]:
        """
        Create embeddings for multiple documents.

        Documents are processed in batches so that large
        repositories do not need to be embedded in one
        enormous operation.
        """

        if not documents:
            return []

        all_embeddings = []

        total_documents = len(
            documents
        )

        # ----------------------------------------------------
        # Process documents batch-by-batch
        # ----------------------------------------------------

        for start in range(
            0,
            total_documents,
            self.batch_size
        ):

            end = min(
                start + self.batch_size,
                total_documents
            )

            batch = documents[
                start:end
            ]

            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            all_embeddings.extend(
                embeddings.tolist()
            )

        return all_embeddings

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        query: str
    ) -> list[float]:
        """
        Create an embedding for a user query.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()