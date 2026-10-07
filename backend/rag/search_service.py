from pathlib import Path

from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore


class SearchService:
    """
    Semantic search service for repository code.

    Responsibilities:
    - Detect the intent of a repository question
    - Retrieve semantic candidates from ChromaDB
    - Filter low-value files
    - Prefer relevant file types
    - Prefer actual implementation files for implementation questions
    - Rank results using semantic similarity + repository-aware signals
    """

    CANDIDATE_MULTIPLIER = 4

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
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

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
        Determine whether the file is low-value for semantic search.
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

        This is intentionally based on repository metadata and naming,
        not on assumptions about the actual code.
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
        Detect chunks produced from actual Python functions/classes
        or similar implementation-aware chunks.
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

        # -----------------------------------------------------
        # Code / implementation questions
        # -----------------------------------------------------

        if query_intent in {
            "code",
            "implementation",
        }:
            return (
                self.is_source_file(metadata)
                or self.is_configuration_file(metadata)
            )

        # -----------------------------------------------------
        # Overview questions
        # -----------------------------------------------------

        if query_intent == "overview":
            return (
                self.is_documentation_file(metadata)
                or self.is_source_file(metadata)
                or self.is_configuration_file(metadata)
            )

        # -----------------------------------------------------
        # Technology questions
        # -----------------------------------------------------

        if query_intent == "technology":
            return (
                self.is_configuration_file(metadata)
                or self.is_documentation_file(metadata)
                or self.is_source_file(metadata)
            )

        # -----------------------------------------------------
        # Architecture questions
        # -----------------------------------------------------

        if query_intent == "architecture":
            return (
                self.is_documentation_file(metadata)
                or self.is_source_file(metadata)
                or self.is_configuration_file(metadata)
            )

        # -----------------------------------------------------
        # General questions
        # -----------------------------------------------------

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

        The goal is not to replace semantic similarity.
        It adds contextual signals on top of semantic similarity.
        """

        file_name = self.get_file_name(metadata)

        boost = 0.0

        # -----------------------------------------------------
        # High-value repository files
        # -----------------------------------------------------

        if file_name in self.HIGH_VALUE_FILES:
            boost += 0.10

        # -----------------------------------------------------
        # Overview
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Technology
        # -----------------------------------------------------

        elif query_intent == "technology":

            if self.is_configuration_file(metadata):
                boost += 0.45

            if self.is_documentation_file(metadata):
                boost += 0.20

            if self.is_source_file(metadata):
                boost += 0.05

        # -----------------------------------------------------
        # Architecture
        # -----------------------------------------------------

        elif query_intent == "architecture":

            if self.is_documentation_file(metadata):
                boost += 0.30

            if self.is_source_file(metadata):
                boost += 0.20

            if self.is_configuration_file(metadata):
                boost += 0.15

            if self.is_test_file(metadata):
                boost -= 0.20

        # -----------------------------------------------------
        # Implementation / code
        # -----------------------------------------------------

        elif query_intent in {
            "implementation",
            "code",
        }:

            # Source code should be strongly preferred.
            if self.is_source_file(metadata):
                boost += 0.20

            # Actual function/class chunks are stronger evidence
            # than generic text blocks.
            if self.is_function_or_class_chunk(metadata):
                boost += 0.45

            # Implementation modules are preferred.
            if self.is_implementation_file(metadata):
                boost += 0.30

            # Tests and evaluation scripts usually consume
            # implementation rather than define it.
            if self.is_test_file(metadata):
                boost -= 0.35

            # Documentation can explain implementation, but
            # actual source should normally rank higher.
            if self.is_documentation_file(metadata):
                boost -= 0.05

        # -----------------------------------------------------
        # General query
        # -----------------------------------------------------

        else:

            if self.is_source_file(metadata):
                boost += 0.10

            if self.is_documentation_file(metadata):
                boost += 0.05

        return boost

    # =========================================================
    # SEMANTIC + REPOSITORY RANKING
    # =========================================================

    def calculate_ranking_score(
        self,
        distance: float,
        metadata: dict,
        query_intent: str,
    ) -> float:
        """
        Combine semantic similarity with repository-aware signals.

        Chroma distance is converted into a similarity-like score
        where smaller distance means stronger semantic relevance.
        """

        safe_distance = max(
            float(distance),
            0.0,
        )

        semantic_score = 1.0 / (
            1.0 + safe_distance
        )

        file_boost = self.calculate_file_boost(
            metadata=metadata,
            query_intent=query_intent,
        )

        return semantic_score + file_boost

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        repository_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Perform repository-aware semantic search.

        Flow:

        Query
          ↓
        Intent detection
          ↓
        Query embedding
          ↓
        Candidate retrieval
          ↓
        File filtering
          ↓
        Repository-aware ranking
          ↓
        Top-K results
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
        # Create query embedding
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_service.embed_query(
                query
            )
        )

        # -----------------------------------------------------
        # Retrieve more candidates than requested.
        #
        # This gives the ranking layer enough candidates to
        # remove low-value files and promote better matches.
        # -----------------------------------------------------

        candidate_k = max(
            top_k * self.CANDIDATE_MULTIPLIER,
            10,
        )

        search_results = (
            self.vector_store.search(
                repository_id=repository_id,
                query_embedding=query_embedding,
                top_k=candidate_k,
            )
        )

        if not search_results:
            return []

        # -----------------------------------------------------
        # Extract ChromaDB results
        # -----------------------------------------------------

        documents = search_results.get(
            "documents",
            [[]],
        )

        metadatas = search_results.get(
            "metadatas",
            [[]],
        )

        distances = search_results.get(
            "distances",
            [[]],
        )

        documents = (
            documents[0]
            if documents
            else []
        )

        metadatas = (
            metadatas[0]
            if metadatas
            else []
        )

        distances = (
            distances[0]
            if distances
            else []
        )

        # -----------------------------------------------------
        # Build ranked candidates
        # -----------------------------------------------------

        ranked_results = []

        for index, content in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else 0.0
            )

            # -------------------------------------------------
            # Apply query-intent filtering
            # -------------------------------------------------

            if not self.should_include_result(
                metadata=metadata,
                query_intent=query_intent,
            ):
                continue

            # -------------------------------------------------
            # Calculate final ranking score
            # -------------------------------------------------

            ranking_score = (
                self.calculate_ranking_score(
                    distance=distance,
                    metadata=metadata,
                    query_intent=query_intent,
                )
            )

            ranked_results.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "distance": distance,
                    "score": ranking_score,
                }
            )

        # -----------------------------------------------------
        # Sort highest score first
        # -----------------------------------------------------

        ranked_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # -----------------------------------------------------
        # Return requested number of results
        # -----------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            ranked_results[:top_k],
            start=1,
        ):

            final_results.append(
                {
                    "rank": rank,
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "distance": result["distance"],
                }
            )

        return final_results
    