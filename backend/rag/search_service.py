from pathlib import Path

from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore
from rag.keyword_store import KeywordStore


class SearchService:
    """
    Hybrid repository search service for CodeLens AI.

    Retrieval:
    - ChromaDB -> semantic search
    - SQLite FTS5 -> keyword/BM25 search
    - RRF -> rank-based result fusion

    Ranking:
    - Query intent detection
    - Repository-aware filtering
    - File-level relevance signals
    - Implementation-aware signals
    """

    # ---------------------------------------------------------
    # Retrieval configuration
    # ---------------------------------------------------------

    CANDIDATE_MULTIPLIER = 4

    # Number of candidates retrieved from each search system
    # before fusion.
    HYBRID_CANDIDATE_MULTIPLIER = 8

    # Standard RRF constant.
    RRF_K = 60

    # ---------------------------------------------------------
    # High-value repository files
    # ---------------------------------------------------------

    HIGH_VALUE_FILES = {
        "readme.md",
        "readme.txt",
        "readme",
        "pyproject.toml",
        "package.json",
        "requirements.txt",
        "pom.xml",
        "build.gradle",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
    }

    # ---------------------------------------------------------
    # Low-value repository files
    # ---------------------------------------------------------

    LOW_VALUE_FILES = {
        "license",
        "license.txt",
        "license.md",
        "copying",
        "copying.txt",
        "gitattributes",
        "gitattributes.txt",
        ".gitattributes",
        "gitignore",
        "gitignore.txt",
        ".gitignore",
    }

    # ---------------------------------------------------------
    # Documentation files
    # ---------------------------------------------------------

    DOCUMENTATION_EXTENSIONS = {
        ".md",
        ".rst",
        ".txt",
    }

    # ---------------------------------------------------------
    # Source-code files
    # ---------------------------------------------------------

    SOURCE_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".php",
        ".cs",
    }

    # ---------------------------------------------------------
    # Configuration / dependency files
    # ---------------------------------------------------------

    CONFIGURATION_FILES = {
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pom.xml",
        "build.gradle",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
    }

    # ---------------------------------------------------------
    # Files commonly used as tests
    # ---------------------------------------------------------

    TEST_FILE_INDICATORS = {
        "test_",
        "_test",
        "tests",
    }

    # ---------------------------------------------------------
    # File-name indicators for implementation modules
    # ---------------------------------------------------------

    IMPLEMENTATION_FILE_INDICATORS = {
        "service",
        "services",
        "manager",
        "controller",
        "controllers",
        "repository",
        "repositories",
        "api",
        "router",
        "routers",
        "handler",
        "handlers",
        "engine",
        "processor",
        "processing",
        "pipeline",
        "model",
        "models",
        "utils",
        "utility",
        "helper",
        "helpers",
    }

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.keyword_store = keyword_store

    # =========================================================
    # QUERY INTENT
    # =========================================================

    def detect_query_intent(self, query: str) -> str:
        """
        Detect the broad intent of a repository question.
        """

        normalized_query = query.lower().strip()

        overview_keywords = [
            "what is this repository",
            "what is this project",
            "what does this repository do",
            "what does this project do",
            "repository about",
            "project about",
            "overview",
            "purpose of this repository",
            "purpose of this project",
            "explain this repository",
            "explain this project",
        ]

        architecture_keywords = [
            "architecture",
            "structure",
            "modules",
            "components",
            "how is the project organized",
            "how is the repository organized",
        ]

        technology_keywords = [
            "technology",
            "technologies",
            "tech stack",
            "framework",
            "libraries",
            "dependencies",
            "tools used",
            "what technologies",
        ]

        implementation_keywords = [
            "how does",
            "how is",
            "how are",
            "implementation",
            "implemented",
            "where is",
            "where are",
            "defined",
            "definition",
            "function",
            "class",
            "method",
            "pipeline",
            "workflow",
            "logic",
            "code",
        ]

        code_keywords = [
            "function",
            "class",
            "method",
            "code",
            "implementation",
            "where is",
            "defined",
            "implemented",
            "how does",
        ]

        if any(
            keyword in normalized_query
            for keyword in overview_keywords
        ):
            return "overview"

        if any(
            keyword in normalized_query
            for keyword in architecture_keywords
        ):
            return "architecture"

        if any(
            keyword in normalized_query
            for keyword in technology_keywords
        ):
            return "technology"

        if any(
            keyword in normalized_query
            for keyword in implementation_keywords
        ):
            return "implementation"

        if any(
            keyword in normalized_query
            for keyword in code_keywords
        ):
            return "code"

        return "general"

    # =========================================================
    # METADATA HELPERS
    # =========================================================

    def get_file_name(self, metadata: dict) -> str:
        """
        Extract the file name from chunk metadata.
        """

        file_path = metadata.get("file_path", "")

        if not file_path:
            return ""

        return Path(file_path).name.lower()

    def get_file_extension(self, metadata: dict) -> str:
        """
        Extract the file extension.
        """

        file_name = self.get_file_name(metadata)

        if not file_name:
            return ""

        return Path(file_name).suffix.lower()

    def get_file_stem(self, metadata: dict) -> str:
        """
        Extract the file stem without extension.
        """

        file_name = self.get_file_name(metadata)

        if not file_name:
            return ""

        return Path(file_name).stem.lower()

    def is_low_value_file(self, metadata: dict) -> bool:
        """
        Determine whether the file is low-value for search.
        """

        file_name = self.get_file_name(metadata)

        return file_name in self.LOW_VALUE_FILES

    def is_test_file(self, metadata: dict) -> bool:
        """
        Determine whether a file looks like a test file.
        """

        file_name = self.get_file_name(metadata)
        file_stem = self.get_file_stem(metadata)

        if file_name.startswith("test_"):
            return True

        if file_stem.endswith("_test"):
            return True

        if file_stem.startswith("test"):
            return True

        file_path = metadata.get(
            "file_path",
            "",
        ).lower()

        path_parts = Path(file_path).parts

        if "tests" in path_parts or "test" in path_parts:
            return True

        return False

    def is_documentation_file(self, metadata: dict) -> bool:
        """
        Determine whether the file is documentation.
        """

        extension = self.get_file_extension(metadata)

        return extension in self.DOCUMENTATION_EXTENSIONS

    def is_source_file(self, metadata: dict) -> bool:
        """
        Determine whether the file contains source code.
        """

        extension = self.get_file_extension(metadata)

        return extension in self.SOURCE_EXTENSIONS

    def is_configuration_file(self, metadata: dict) -> bool:
        """
        Determine whether the file is a configuration/dependency file.
        """

        file_name = self.get_file_name(metadata)

        return file_name in self.CONFIGURATION_FILES

    def is_implementation_file(self, metadata: dict) -> bool:
        """
        Determine whether a file looks like an implementation module.
        """

        if not self.is_source_file(metadata):
            return False

        if self.is_test_file(metadata):
            return False

        file_stem = self.get_file_stem(metadata)

        if not file_stem:
            return False

        for indicator in self.IMPLEMENTATION_FILE_INDICATORS:
            if indicator in file_stem:
                return True

        return False

    def is_function_or_class_chunk(self, metadata: dict) -> bool:
        """
        Detect chunks produced from actual functions/classes.
        """

        chunk_type = str(
            metadata.get(
                "chunk_type",
                "",
            )
        ).lower()

        implementation_chunk_types = {
            "function",
            "class",
            "method",
            "async_function",
            "implementation",
        }

        return chunk_type in implementation_chunk_types

    # =========================================================
    # QUERY FILTERING
    # =========================================================

    def should_include_result(
        self,
        metadata: dict,
        query_intent: str,
    ) -> bool:
        """
        Decide whether a retrieved result should participate
        in final ranking.
        """

        if self.is_low_value_file(metadata):
            return False

        extension = self.get_file_extension(metadata)

        if query_intent in {
            "code",
            "implementation",
        }:
            return (
                self.is_source_file(metadata)
                or self.is_configuration_file(metadata)
            )

        if query_intent == "overview":
            return (
                self.is_documentation_file(metadata)
                or self.is_source_file(metadata)
                or self.is_configuration_file(metadata)
            )

        if query_intent == "technology":
            return (
                self.is_configuration_file(metadata)
                or self.is_documentation_file(metadata)
                or self.is_source_file(metadata)
            )

        if query_intent == "architecture":
            return (
                self.is_documentation_file(metadata)
                or self.is_source_file(metadata)
                or self.is_configuration_file(metadata)
            )

        return bool(extension)

    # =========================================================
    # FILE-LEVEL RANKING
    # =========================================================

    def calculate_file_boost(
        self,
        metadata: dict,
        query_intent: str,
    ) -> float:
        """
        Calculate a repository-aware ranking boost.
        """

        file_name = self.get_file_name(metadata)

        boost = 0.0

        if file_name in self.HIGH_VALUE_FILES:
            boost += 0.10

        if query_intent == "overview":

            if self.is_documentation_file(metadata):
                boost += 0.35

            if file_name in {
                "readme.md",
                "readme.txt",
                "readme",
            }:
                boost += 0.35

            if self.is_source_file(metadata):
                boost += 0.05

        elif query_intent == "technology":

            if self.is_configuration_file(metadata):
                boost += 0.45

            if self.is_documentation_file(metadata):
                boost += 0.20

            if self.is_source_file(metadata):
                boost += 0.05

        elif query_intent == "architecture":

            if self.is_documentation_file(metadata):
                boost += 0.30

            if self.is_source_file(metadata):
                boost += 0.20

            if self.is_configuration_file(metadata):
                boost += 0.15

            if self.is_test_file(metadata):
                boost -= 0.20

        elif query_intent in {
            "implementation",
            "code",
        }:

            if self.is_source_file(metadata):
                boost += 0.20

            if self.is_function_or_class_chunk(metadata):
                boost += 0.45

            if self.is_implementation_file(metadata):
                boost += 0.30

            if self.is_test_file(metadata):
                boost -= 0.35

            if self.is_documentation_file(metadata):
                boost -= 0.05

        else:

            if self.is_source_file(metadata):
                boost += 0.10

            if self.is_documentation_file(metadata):
                boost += 0.05

        return boost

    # =========================================================
    # SEMANTIC SCORE
    # =========================================================

    def calculate_semantic_score(
        self,
        distance: float,
    ) -> float:
        """
        Convert Chroma distance into a similarity-like score.
        """

        safe_distance = max(
            float(distance),
            0.0,
        )

        return 1.0 / (
            1.0 + safe_distance
        )

    # =========================================================
    # RRF
    # =========================================================

    def calculate_rrf_score(
        self,
        semantic_rank: int | None,
        keyword_rank: int | None,
    ) -> float:
        """
        Calculate Reciprocal Rank Fusion score.

        A chunk receives evidence from either or both retrieval
        systems.

        If a chunk appears in both rankings, the contributions
        are added together.
        """

        score = 0.0

        if semantic_rank is not None:
            score += 1.0 / (
                self.RRF_K + semantic_rank
            )

        if keyword_rank is not None:
            score += 1.0 / (
                self.RRF_K + keyword_rank
            )

        return score

    # =========================================================
    # HYBRID SEARCH
    # =========================================================

    def search(
        self,
        repository_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Perform hybrid repository search.

        Flow:

        Query
          ↓
        Intent detection
          ↓
        ┌─────────────────────┐
        │                     │
        ↓                     ↓
      ChromaDB             SQLite FTS5
      Semantic             Keyword/BM25
        │                     │
        └──────────┬──────────┘
                   ↓
                 RRF
                   ↓
          Query-intent filtering
                   ↓
          Repository-aware ranking
                   ↓
                  Top-K
        """

        if not repository_id:
            raise ValueError(
                "repository_id is required."
            )

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # -----------------------------------------------------
        # Detect query intent
        # -----------------------------------------------------

        query_intent = self.detect_query_intent(
            query
        )

        # -----------------------------------------------------
        # Candidate count
        # -----------------------------------------------------

        candidate_k = max(
            top_k * self.HYBRID_CANDIDATE_MULTIPLIER,
            20,
        )

        # =====================================================
        # 1. SEMANTIC SEARCH
        # =====================================================

        query_embedding = (
            self.embedding_service.embed_query(
                query
            )
        )

        semantic_response = (
            self.vector_store.search(
                repository_id=repository_id,
                query_embedding=query_embedding,
                top_k=candidate_k,
            )
        )

        semantic_documents = semantic_response.get(
            "documents",
            [[]],
        )

        semantic_metadatas = semantic_response.get(
            "metadatas",
            [[]],
        )

        semantic_distances = semantic_response.get(
            "distances",
            [[]],
        )

        semantic_ids = semantic_response.get(
            "ids",
            [[]],
        )

        semantic_documents = (
            semantic_documents[0]
            if semantic_documents
            else []
        )

        semantic_metadatas = (
            semantic_metadatas[0]
            if semantic_metadatas
            else []
        )

        semantic_distances = (
            semantic_distances[0]
            if semantic_distances
            else []
        )

        semantic_ids = (
            semantic_ids[0]
            if semantic_ids
            else []
        )

        # =====================================================
        # 2. KEYWORD / BM25 SEARCH
        # =====================================================

        keyword_results = self.keyword_store.search(
            repository_id=repository_id,
            query=query,
            top_k=candidate_k,
        )

        # =====================================================
        # 3. BUILD UNIFIED CANDIDATE MAP
        # =====================================================

        candidates = {}

        # -----------------------------------------------------
        # Semantic candidates
        # -----------------------------------------------------

        for index, content in enumerate(
            semantic_documents
        ):

            metadata = (
                semantic_metadatas[index]
                if index < len(semantic_metadatas)
                else {}
            )

            distance = (
                semantic_distances[index]
                if index < len(semantic_distances)
                else 0.0
            )

            chunk_id = (
                semantic_ids[index]
                if index < len(semantic_ids)
                else metadata.get("chunk_id")
            )

            if not chunk_id:
                chunk_id = (
                    f"semantic_{index}_{metadata.get('file_path', '')}"
                )

            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "semantic_rank": index + 1,
                "keyword_rank": None,
                "keyword_score": 0.0,
            }

        # -----------------------------------------------------
        # Keyword candidates
        # -----------------------------------------------------

        for index, result in enumerate(
            keyword_results
        ):

            chunk_id = result.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": result.get(
                        "content",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "distance": None,
                    "semantic_rank": None,
                    "keyword_rank": index + 1,
                    "keyword_score": result.get(
                        "keyword_score",
                        0.0,
                    ),
                }

            else:

                candidates[chunk_id][
                    "keyword_rank"
                ] = index + 1

                candidates[chunk_id][
                    "keyword_score"
                ] = result.get(
                    "keyword_score",
                    0.0,
                )

        # =====================================================
        # 4. RRF FUSION
        # =====================================================

        fused_results = []

        for candidate in candidates.values():

            rrf_score = self.calculate_rrf_score(
                semantic_rank=candidate[
                    "semantic_rank"
                ],
                keyword_rank=candidate[
                    "keyword_rank"
                ],
            )

            candidate["rrf_score"] = rrf_score

            fused_results.append(candidate)

        # -----------------------------------------------------
        # RRF ranking
        # -----------------------------------------------------

        fused_results.sort(
            key=lambda item: item[
                "rrf_score"
            ],
            reverse=True,
        )

        # =====================================================
        # 5. FILTER + REPOSITORY-AWARE RANKING
        # =====================================================

        ranked_results = []

        for candidate in fused_results:

            metadata = candidate.get(
                "metadata",
                {},
            )

            if not self.should_include_result(
                metadata=metadata,
                query_intent=query_intent,
            ):
                continue

            # -------------------------------------------------
            # Base hybrid score
            #
            # RRF determines retrieval agreement.
            # Semantic similarity remains available when
            # Chroma retrieved the candidate.
            # -------------------------------------------------

            rrf_score = candidate[
                "rrf_score"
            ]

            semantic_score = 0.0

            if candidate[
                "distance"
            ] is not None:

                semantic_score = (
                    self.calculate_semantic_score(
                        candidate["distance"]
                    )
                )

            file_boost = (
                self.calculate_file_boost(
                    metadata=metadata,
                    query_intent=query_intent,
                )
            )

            # -------------------------------------------------
            # Normalize the relative influence.
            #
            # RRF is the primary retrieval signal.
            # Semantic similarity provides additional
            # information for semantic-only / semantic-ranked
            # candidates.
            # -------------------------------------------------

            final_score = (
                rrf_score
                + (0.10 * semantic_score)
                + file_boost
            )

            candidate["score"] = final_score

            ranked_results.append(
                candidate
            )

        # =====================================================
        # 6. FINAL SORT
        # =====================================================

        ranked_results.sort(
            key=lambda item: item[
                "score"
            ],
            reverse=True,
        )

        # =====================================================
        # 7. RETURN TOP-K
        # =====================================================

        final_results = []

        for rank, result in enumerate(
            ranked_results[:top_k],
            start=1,
        ):

            final_results.append(
                {
                    "rank": rank,
                    "content": result[
                        "content"
                    ],
                    "metadata": result[
                        "metadata"
                    ],
                    "distance": (
                        result["distance"]
                        if result["distance"]
                        is not None
                        else 0.0
                    ),
                }
            )

        return final_results