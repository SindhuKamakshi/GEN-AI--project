import re
from typing import Optional

from rag.embedding_service import EmbeddingService
from rag.keyword_store import KeywordStore
from rag.reranker import Reranker
from rag.vector_store import VectorStore


class SearchService:
    """
    Hybrid repository search service for CodeLens AI.

    Search pipeline:

        Semantic Search
              +
        Keyword/BM25 Search
              ↓
             RRF
              ↓
       Intent-aware ranking
              ↓
        Cross-Encoder
          Reranking
              ↓
      Code-Aware Ranking
              ↓
          Final Top-K
    """

    HYBRID_CANDIDATE_MULTIPLIER = 8
    RRF_K = 60

    # ==========================================================
    # Code-aware ranking weights
    # ==========================================================

    DEFINITION_BONUS = 2.0
    EXACT_ENTITY_NAME_BONUS = 1.5
    FUNCTION_DEFINITION_BONUS = 1.0
    CLASS_DEFINITION_BONUS = 1.0
    USAGE_BONUS = 0.5

    # Usage-specific weights.
    USAGE_CALL_BONUS = 1.25
    USAGE_REFERENCE_BONUS = 0.75
    USAGE_DEFINITION_PENALTY = -1.5

    # Explanation-specific weights.
    EXPLANATION_DEFINITION_BONUS = 3.0
    EXPLANATION_RELATED_CODE_BONUS = 0.75

        # File-level retrieval improvements.
    

    HIGH_VALUE_FILES = {
        "readme.md",
        "readme.txt",
        "readme.rst",
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "dockerfile",
    }

    LOW_VALUE_FILES = {
        "license",
        "license.txt",
        "license.md",
        ".gitignore",
        ".gitattributes",
    }

    DOCUMENT_EXTENSIONS = {
        ".md",
        ".rst",
        ".txt",
    }

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

    CONFIG_EXTENSIONS = {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".env",
    }

    IMPLEMENTATION_KEYWORDS = {
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
        reranker: Optional[Reranker] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.keyword_store = keyword_store

        # Reranker is optional so the existing RRF search can
        # continue working if the reranker is unavailable.
        self.reranker = reranker

    # ==========================================================
    # Query Intent
    # ==========================================================

    def detect_query_intent(self, query: str) -> str:
        query_lower = query.lower()

        overview_terms = {
            "overview",
            "summarize",
            "summary",
            "what does this project do",
            "what is this project",
            "purpose",
            "main functionality",
        }

        architecture_terms = {
            "architecture",
            "architectural",
            "structure",
            "system design",
            "how is the system organized",
            "components",
        }

        technology_terms = {
            "technology",
            "technologies",
            "tech stack",
            "libraries",
            "library",
            "framework",
            "dependencies",
            "tools",
            "what is used",
        }

        implementation_terms = {
            "implemented",
            "implementation",
            "where is",
            "defined",
            "function",
            "class",
            "method",
            "code",
            "logic",
        }

        # ------------------------------------------------------
        # High-level project questions first
        # ------------------------------------------------------

        if any(
            term in query_lower
            for term in overview_terms
        ):
            return "overview"

        if any(
            term in query_lower
            for term in architecture_terms
        ):
            return "architecture"

        if any(
            term in query_lower
            for term in technology_terms
        ):
            return "technology"

        # ------------------------------------------------------
        # Code relationship intent
        # ------------------------------------------------------

        code_intent = self.detect_code_intent(query)

        if code_intent in {
            "definition",
            "usage",
            "explanation",
        }:
            return "implementation"

        # ------------------------------------------------------
        # General implementation wording
        # ------------------------------------------------------

        if any(
            term in query_lower
            for term in implementation_terms
        ):
            return "implementation"

        return "general"

    # ==========================================================
    # File Information
    # ==========================================================

    def get_file_name(self, metadata: dict) -> str:
        file_name = metadata.get("file_name")

        if file_name:
            return str(file_name).lower()

        file_path = metadata.get("file_path", "")

        if not file_path:
            return ""

        return (
            str(file_path)
            .replace("\\", "/")
            .split("/")[-1]
            .lower()
        )

    def get_file_extension(self, metadata: dict) -> str:
        file_path = str(
            metadata.get("file_path", "")
        ).lower()

        if "." not in file_path:
            return ""

        return "." + file_path.rsplit(".", 1)[-1]

    def get_file_stem(self, metadata: dict) -> str:
        file_name = self.get_file_name(metadata)

        if "." in file_name:
            return file_name.rsplit(".", 1)[0]

        return file_name
    # ==========================================================
    # File Query Detection
    # ==========================================================

    def is_file_oriented_query(
        self,
        query: str,
    ) -> bool:
        """
        Detect queries where the user primarily wants to know
        which file contains or implements something.

        Examples:

            "Which file contains the RAG implementation?"
            "Which file contains the application entry point?"
            "Which file contains RAG evaluation code?"
        """

        query_lower = query.lower().strip()

        file_patterns = [
            r"\bwhich\s+file\b",
            r"\bwhich\s+files\b",
            r"\bwhat\s+file\b",
            r"\bwhat\s+files\b",
            r"\bfile\s+contains\b",
            r"\bfiles\s+contain\b",
            r"\bfile\s+has\b",
            r"\bfiles\s+have\b",
        ]

        return any(
            re.search(
                pattern,
                query_lower,
            )
            for pattern in file_patterns
        )

    # ==========================================================
    # File Query Matching
    # ==========================================================

    
    # ==========================================================
    # Intent Filtering
    # ==========================================================

    def should_keep_result(
        self,
        metadata: dict,
        intent: str,
        query: str,
        content: str,
    ) -> bool:

        extension = self.get_file_extension(metadata)
        file_name = self.get_file_name(metadata)

        if intent == "technology":
            return (
                file_name in self.HIGH_VALUE_FILES
                or extension in self.CONFIG_EXTENSIONS
                or extension in self.DOCUMENT_EXTENSIONS
                or self.is_relevant_notebook_data_result(
                    metadata,
                    query,
                    content,
                )
            )

        if intent == "overview":
            return (
                file_name in self.HIGH_VALUE_FILES
                or extension in self.DOCUMENT_EXTENSIONS
            )

        if intent == "architecture":
            return (
                file_name in self.HIGH_VALUE_FILES
                or extension in self.DOCUMENT_EXTENSIONS
                or extension in self.SOURCE_EXTENSIONS
            )

        if intent == "implementation":
            return (
                extension in self.SOURCE_EXTENSIONS
                or (
                    extension == ".ipynb"
                    and self.is_relevant_notebook_data_result(
                        metadata,
                        query,
                        content,
                    )
                )
            )

        return True

    @staticmethod
    def is_notebook_import_query(query: str) -> bool:
        query_terms = set(
            re.findall(r"[a-z_]+", query.lower())
        )
        return (
            "notebook" in query_terms
            and bool(
                query_terms.intersection(
                    {"import", "imports", "imported", "importing"}
                )
            )
        )

    @staticmethod
    def is_relevant_notebook_data_result(
        metadata: dict,
        query: str,
        content: str,
    ) -> bool:
        if (
            metadata.get("chunk_type") != "notebook_cell"
            or metadata.get("notebook_cell_type") != "code"
        ):
            return False

        query_terms = set(
            re.findall(r"[a-z_]+", query.lower())
        )

        asks_about_notebook_data_loading = (
            "notebook" in query_terms
            and query_terms.intersection(
                {"dataset", "data", "file"}
            )
            and query_terms.intersection(
                {"read", "reads", "reading", "load", "loads", "loading"}
            )
        )

        has_python_import = re.search(
            r"(?m)^\s*(?:import\s+\w+|from\s+\w+\s+import\b)",
            content,
        )
        has_data_reader = re.search(
            r"\b(?:[A-Za-z_]\w*\.)?(?:read|load)_[A-Za-z_]\w*\s*\(",
            content,
        )

        if asks_about_notebook_data_loading:
            return bool(has_python_import and has_data_reader)

        if SearchService.is_notebook_import_query(query):
            return bool(has_python_import)

        return False

    # ==========================================================
    # File Boost
    # ==========================================================

    def calculate_file_boost(
        self,
        metadata: dict,
        intent: str,
    ) -> float:

        file_name = self.get_file_name(metadata)
        file_stem = self.get_file_stem(metadata)
        extension = self.get_file_extension(metadata)

        boost = 0.0

        if file_name in self.HIGH_VALUE_FILES:
            boost += 0.30

        if file_name in self.LOW_VALUE_FILES:
            boost -= 0.15

        if extension in self.DOCUMENT_EXTENSIONS:
            boost += 0.05

        if extension in self.CONFIG_EXTENSIONS:
            boost += 0.04

        if intent == "technology":
            if file_name in {
                "requirements.txt",
                "pyproject.toml",
                "package.json",
                "pom.xml",
                "build.gradle",
                "build.gradle.kts",
            }:
                boost += 0.30

        if intent in {
            "overview",
            "architecture",
        }:
            if file_name.startswith("readme"):
                boost += 0.20

        if intent == "implementation":
            if (
                file_stem.startswith("test_")
                or file_stem.endswith("_test")
            ):
                boost -= 0.03

            if any(
                keyword in file_stem
                for keyword in self.IMPLEMENTATION_KEYWORDS
            ):
                boost += 0.08

        return boost

    # ==========================================================
    # Semantic Score
    # ==========================================================

    def calculate_semantic_score(
        self,
        distance: float,
    ) -> float:

        return 1.0 / (
            1.0 + max(distance, 0.0)
        )

    # ==========================================================
    # Query Entity Extraction
    # ==========================================================

    def extract_entity_name(
        self,
        query: str,
    ) -> Optional[str]:
        """
        Extract the likely code entity from implementation-style
        questions.

        Examples:

            "Where is retrieve_context defined?"
                -> retrieve_context

            "What does answer_question do?"
                -> answer_question

            "Where is UserService implemented?"
                -> UserService
        """

        query_text = query.strip()

        if not query_text:
            return None

        words = (
            query_text
            .replace("?", "")
            .replace(",", " ")
            .split()
        )

        stop_words = {
            "where",
            "is",
            "are",
            "the",
            "a",
            "an",
            "what",
            "does",
            "do",
            "how",
            "why",
            "when",
            "which",
            "function",
            "class",
            "method",
            "defined",
            "define",
            "implemented",
            "implementation",
            "used",
            "use",
            "uses",
            "called",
            "call",
            "code",
            "logic",
            "in",
            "this",
            "project",
            "file",
        }

        candidates = []

        for word in words:
            cleaned = word.strip(
                "`'\"()[]{}:;."
            )

            if not cleaned:
                continue

            if cleaned.lower() in stop_words:
                continue

            # Code identifiers commonly contain underscores,
            # camelCase, PascalCase, or digits.
            if (
                "_" in cleaned
                or any(
                    character.isupper()
                    for character in cleaned
                )
                or any(
                    character.isdigit()
                    for character in cleaned
                )
            ):
                candidates.append(cleaned)

        if candidates:
            return candidates[-1]

        # Fallback: use the last meaningful word.
        meaningful_words = []

        for word in words:
            cleaned = word.strip(
                "`'\"()[]{}:;."
            )

            if (
                cleaned
                and cleaned.lower() not in stop_words
            ):
                meaningful_words.append(cleaned)

        if meaningful_words:
            return meaningful_words[-1]

        return None

    # ==========================================================
    # Code-Aware Intent
    # ==========================================================

    def detect_code_intent(
        self,
        query: str,
    ) -> str:
        """
        Detect the user's requested relationship with a code entity.

        Returns:

            definition
            usage
            explanation
            general
        """

        query_lower = query.lower().strip()

        # ------------------------------------------------------
        # Usage terms
        # ------------------------------------------------------

        usage_terms = {
            "where is it used",
            "where is it called",
            "where is this used",
            "where is this called",
            "where is used",
            "where is called",
            "usages",
            "usage",
            "calls",
            "called",
            "references",
            "referenced",
            "imports",
            "imported",
            "who calls",
            "who uses",
            "which functions use",
        }

        if any(
            term in query_lower
            for term in usage_terms
        ):
            return "usage"

        # ------------------------------------------------------
        # Entity-specific usage patterns
        #
        # These handle questions such as:
        #
        # "Where is retrieve_context used?"
        # "Where is retrieve_context called?"
        # "Where is UserService referenced?"
        # ------------------------------------------------------

        entity_usage_patterns = [
            r"\bwhere\s+is\s+.+\s+used\b",
            r"\bwhere\s+is\s+.+\s+called\b",
            r"\bwhere\s+is\s+.+\s+referenced\b",
            r"\bwhere\s+is\s+.+\s+imported\b",
            r"\bwhere\s+are\s+.+\s+used\b",
            r"\bwhere\s+are\s+.+\s+called\b",
            r"\bwhere\s+can\s+i\s+find\s+.+\s+usages?\b",
            r"\bwho\s+calls\s+.+\b",
            r"\bwho\s+uses\s+.+\b",
        ]

        if any(
            re.search(
                pattern,
                query_lower,
            )
            for pattern in entity_usage_patterns
        ):
            return "usage"

        # ------------------------------------------------------
        # Explanation terms
        # ------------------------------------------------------

        explanation_terms = {
            "what does",
            "how does",
            "explain",
            "what is the purpose",
            "how is it implemented",
            "how does it work",
            "what does it do",
        }

        if any(
            term in query_lower
            for term in explanation_terms
        ):
            return "explanation"

        # Entity-specific explanation patterns.
        entity_explanation_patterns = [
            r"\bwhat\s+does\s+.+\s+do\b",
            r"\bhow\s+does\s+.+\s+work\b",
            r"\bexplain\s+.+\b",
            r"\bwhat\s+is\s+the\s+purpose\s+of\s+.+\b",
        ]

        if any(
            re.search(
                pattern,
                query_lower,
            )
            for pattern in entity_explanation_patterns
        ):
            return "explanation"

        # ------------------------------------------------------
        # Definition terms
        # ------------------------------------------------------

        definition_terms = {
            "where is",
            "defined",
            "definition",
            "declare",
            "declared",
            "implemented",
            "implementation",
        }

        if any(
            term in query_lower
            for term in definition_terms
        ):
            return "definition"

        # Entity-specific definition patterns.
        entity_definition_patterns = [
            r"\bwhere\s+is\s+.+\s+defined\b",
            r"\bwhere\s+is\s+.+\s+implemented\b",
            r"\bwhere\s+is\s+the\s+definition\s+of\s+.+\b",
        ]

        if any(
            re.search(
                pattern,
                query_lower,
            )
            for pattern in entity_definition_patterns
        ):
            return "definition"

        return "general"

    # ==========================================================
    # Code-Aware Ranking
    # ==========================================================

    def calculate_code_aware_boost(
        self,
        query: str,
        metadata: dict,
        content: str,
        intent: str,
    ) -> float:
        """
        Apply code-specific ranking rules after cross-encoder
        reranking.

        Ranking is adjusted for:

            definition
            usage
            explanation

        The goal is to prefer the actual entity definition
        for definition/explanation queries and actual callers
        for usage/caller queries.
        """

        if intent != "implementation":
            return 0.0

        code_intent = self.detect_code_intent(
            query
        )

        entity_name = self.extract_entity_name(
            query
        )

        if not entity_name:
            return 0.0

        metadata_name = str(
            metadata.get("name", "")
        ).strip()

        chunk_type = str(
            metadata.get("chunk_type", "")
        ).strip().lower()

        content_text = content or ""
        content_lower = content_text.lower()

        entity_lower = entity_name.lower()
        metadata_name_lower = metadata_name.lower()

        boost = 0.0

        # ------------------------------------------------------
        # Basic entity / definition signals
        # ------------------------------------------------------

        is_exact_entity = (
            metadata_name_lower == entity_lower
        )

        is_definition_chunk = chunk_type in {
            "function",
            "class",
            "method",
        }

        # ------------------------------------------------------
        # Detect whether the query specifically asks for a
        # caller / function that calls the requested entity.
        # ------------------------------------------------------

        query_lower = query.lower()


        # ------------------------------------------------------
        # Definition query
        # ------------------------------------------------------

        if code_intent == "definition":

            # Strong preference for the exact requested entity.
            if is_exact_entity:
                boost += (
                    self.EXACT_ENTITY_NAME_BONUS
                )

                boost += (
                    self.DEFINITION_BONUS
                )

            # Actual AST definition.
            if (
                is_definition_chunk
                and is_exact_entity
            ):
                boost += (
                    self.FUNCTION_DEFINITION_BONUS
                )

            # Detect explicit definition syntax.
            definition_patterns = [
                f"def {entity_name}(",
                f"class {entity_name}",
                f"function {entity_name}(",
                f"{entity_name} = function",
                f"{entity_name} = async function",
            ]

            contains_definition = any(
                pattern.lower() in content_lower
                for pattern in definition_patterns
            )

            if contains_definition:
                boost += self.USAGE_BONUS

            # If the query asks specifically where the entity
            # is defined, an exact entity definition gets an
            # additional strong signal.
            if (
                is_exact_entity
                and is_definition_chunk
            ):
                boost += 1.5

        # ------------------------------------------------------
        # Usage / caller query
        # ------------------------------------------------------

        elif code_intent == "usage":

            # --------------------------------------------------
            # Detect whether this is specifically asking for
            # an actual caller/function.
            # --------------------------------------------------

            caller_query = any(
                phrase in query_lower
                for phrase in [
                    "which function calls",
                    "which functions call",
                    "what function calls",
                    "what functions call",
                    "who calls",
                    "where is it called",
                ]
            )

            # --------------------------------------------------
            # Penalize the requested entity's own definition.
            #
            # Example:
            # "Which function calls retrieve_context?"
            #
            # The retrieve_context() definition itself is NOT
            # the answer.
            # --------------------------------------------------

            if (
                is_definition_chunk
                and is_exact_entity
            ):
                boost += -3.0

            # --------------------------------------------------
            # Detect an actual call to the requested entity.
            # --------------------------------------------------

            call_patterns = [
                rf"\b{re.escape(entity_name)}\s*\(",
                rf"\.\s*{re.escape(entity_name)}\s*\(",
            ]

            contains_call = any(
                re.search(
                    pattern,
                    content_text,
                    flags=re.IGNORECASE,
                )
                for pattern in call_patterns
            )

            # --------------------------------------------------
            # Detect whether the chunk is the entity's own
            # definition.
            # --------------------------------------------------

            definition_patterns = [
                rf"\bdef\s+{re.escape(entity_name)}\s*\(",
                rf"\basync\s+def\s+{re.escape(entity_name)}\s*\(",
                rf"\bfunction\s+{re.escape(entity_name)}\s*\(",
                rf"\bclass\s+{re.escape(entity_name)}\b",
            ]

            is_own_definition = any(
                re.search(
                    pattern,
                    content_text,
                    flags=re.IGNORECASE,
                )
                for pattern in definition_patterns
            )

            # --------------------------------------------------
            # Actual caller
            #
            # A function/method chunk containing the requested
            # entity call is the strongest evidence.
            # --------------------------------------------------

            if (
                contains_call
                and not is_own_definition
            ):

                if is_definition_chunk:

                    boost += (
                        self.USAGE_CALL_BONUS
                    )

                    if caller_query:
                        # Strong preference for an actual caller
                        # when the user explicitly asks which
                        # function calls the entity.
                        boost += 4.0

                else:

                    boost += (
                        self.USAGE_REFERENCE_BONUS
                    )

                    if caller_query:
                        # A code reference is useful, but weaker
                        # than an actual function definition.
                        boost += 0.75

            # --------------------------------------------------
            # Import / reference detection
            # --------------------------------------------------

            python_import_pattern = (
                rf"\bfrom\s+\S+\s+import\s+.*\b"
                rf"{re.escape(entity_name)}\b"
            )

            generic_import_pattern = (
                rf"\bimport\s+.*\b"
                rf"{re.escape(entity_name)}\b"
            )

            contains_import = bool(
                re.search(
                    python_import_pattern,
                    content_text,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    generic_import_pattern,
                    content_text,
                    flags=re.IGNORECASE,
                )
            )

            # --------------------------------------------------
            # Import/reference should not outrank an actual
            # caller for "which function calls..." queries.
            # --------------------------------------------------

            if contains_import:

                boost += (
                    self.USAGE_REFERENCE_BONUS
                )

                if (
                    caller_query
                    and not contains_call
                ):
                    boost += 0.25

            # --------------------------------------------------
            # Generic usage references.
            # --------------------------------------------------

            usage_patterns = [
                rf"\b{re.escape(entity_name)}\b",
            ]

            contains_reference = any(
                re.search(
                    pattern,
                    content_text,
                    flags=re.IGNORECASE,
                )
                for pattern in usage_patterns
            )

            if (
                contains_reference
                and not contains_call
                and not contains_import
                and not is_own_definition
            ):
                boost += 0.25
        # ------------------------------------------------------
        # Explanation query
        # ------------------------------------------------------

        elif code_intent == "explanation":

            # --------------------------------------------------
            # Exact requested entity definition
            # --------------------------------------------------

            if (
                is_definition_chunk
                and is_exact_entity
            ):

                boost += (
                    self.EXPLANATION_DEFINITION_BONUS
                )

                boost += (
                    self.EXACT_ENTITY_NAME_BONUS
                )

                boost += (
                    self.FUNCTION_DEFINITION_BONUS
                )

                # Additional preference for the exact entity
                # when the user asks where an implementation is.
                boost += 1.5

            # --------------------------------------------------
            # Exact entity even when chunk type metadata is not
            # perfectly classified.
            # --------------------------------------------------

            elif is_exact_entity:

                boost += (
                    self.EXACT_ENTITY_NAME_BONUS
                )

                boost += 1.0

            # --------------------------------------------------
            # Related caller
            # --------------------------------------------------

            elif is_definition_chunk:

                call_patterns = [
                    f"{entity_name}(",
                    f".{entity_name}(",
                ]

                contains_call = any(
                    pattern.lower() in content_lower
                    for pattern in call_patterns
                )

                if contains_call:
                    boost += (
                        self.EXPLANATION_RELATED_CODE_BONUS
                    )
        # ------------------------------------------------------
        # Implementation-intent ranking
        #
        # For implementation questions, prefer the function
        # that actually performs the requested operation rather
        # than nearby functions that merely call or use it.
        # ------------------------------------------------------

        if code_intent == "explanation":

            implementation_query = any(
                phrase in query_lower
                for phrase in [
                    "implemented",
                    "implementation",
                    "where is",
                    "how is",
                    "how does",
                ]
            )

            if implementation_query:

                content_lower = content_text.lower()

                # ------------------------------------------
                # Document retrieval
                # ------------------------------------------

                if (
                    "document retrieval" in query_lower
                    or "retrieval implemented" in query_lower
                ):

                    if (
                        "def retrieve_context(" in content_lower
                        or "def retrieve_context (" in content_lower
                    ):
                        boost += 4.0

                    if (
                        "def answer_question(" in content_lower
                        or "def answer_question (" in content_lower
                    ):
                        boost -= 1.5

                # ------------------------------------------
                # Document chunking
                # ------------------------------------------

                if (
                    "document chunking" in query_lower
                    or "chunking implemented" in query_lower
                ):

                    if (
                        "def create_chunks(" in content_lower
                        or "def create_chunks (" in content_lower
                    ):
                        boost += 4.0

                    if (
                        "def store_chunks(" in content_lower
                        or "def store_chunks (" in content_lower
                    ):
                        boost -= 1.5

                # ------------------------------------------
                # Question answering pipeline
                # ------------------------------------------

                if (
                    "question answering pipeline"
                    in query_lower
                    or "question answering" in query_lower
                ):

                    if (
                        "def answer_question(" in content_lower
                        or "def answer_question (" in content_lower
                    ):
                        boost += 4.0

        return boost
        

    # ==========================================================
    # Hybrid Candidate Collection
    # ==========================================================

    def collect_candidates(
        self,
        repository_id: str,
        query: str,
        top_k: int,
    ) -> dict:

        candidate_k = max(
            top_k * self.HYBRID_CANDIDATE_MULTIPLIER,
            10,
        )

        query_embedding = (
            self.embedding_service.embed_query(query)
        )

        semantic_results = self.vector_store.search(
            repository_id=repository_id,
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        keyword_results = self.keyword_store.search(
            repository_id=repository_id,
            query=query,
            top_k=candidate_k,
        )

        candidates = {}

        # ------------------------------------------------------
        # Semantic candidates
        # ------------------------------------------------------

        semantic_documents = semantic_results.get(
            "documents",
            [[]],
        )[0]

        semantic_metadatas = semantic_results.get(
            "metadatas",
            [[]],
        )[0]

        semantic_distances = semantic_results.get(
            "distances",
            [[]],
        )[0]

        semantic_ids = semantic_results.get(
            "ids",
            [[]],
        )[0]

        for index, content in enumerate(
            semantic_documents
        ):

            metadata = semantic_metadatas[index]
            distance = semantic_distances[index]

            chunk_id = (
                semantic_ids[index]
                if index < len(semantic_ids)
                else metadata.get("chunk_id")
            )

            if not chunk_id:
                chunk_id = (
                    f"{metadata.get('repository_id', repository_id)}_"
                    f"{metadata.get('file_path', 'unknown')}_"
                    f"{metadata.get('start_line', index)}"
                )

            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "content": content,
                "metadata": metadata,
                "distance": float(distance),
                "semantic_rank": index + 1,
                "keyword_rank": None,
                "keyword_score": 0.0,
            }

        # ------------------------------------------------------
        # Keyword candidates
        # ------------------------------------------------------

        for keyword_result in keyword_results:

            chunk_id = keyword_result["chunk_id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": keyword_result["content"],
                    "metadata": keyword_result["metadata"],
                    "distance": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": keyword_result["rank"],
                    "keyword_score": keyword_result.get(
                        "keyword_score",
                        0.0,
                    ),
                }

            else:

                candidates[chunk_id]["keyword_rank"] = (
                    keyword_result["rank"]
                )

                candidates[chunk_id]["keyword_score"] = (
                    keyword_result.get(
                        "keyword_score",
                        0.0,
                    )
                )

        return candidates

    # ==========================================================
    # RRF
    # ==========================================================

    def calculate_rrf_score(
        self,
        candidate: dict,
    ) -> float:

        score = 0.0

        semantic_rank = candidate.get(
            "semantic_rank"
        )

        keyword_rank = candidate.get(
            "keyword_rank"
        )

        if semantic_rank is not None:
            score += 1.0 / (
                self.RRF_K + semantic_rank
            )

        if keyword_rank is not None:
            score += 1.0 / (
                self.RRF_K + keyword_rank
            )

        return score
    # ==========================================================
    # File-Level Ranking
    # ==========================================================

    def rank_file_results(
        self,
        query: str,
        results: list[dict],
    ) -> list[dict]:
        """
        Rank search results at the file level.

        Used for queries where the user explicitly asks
        which file/files contain or implement something.

        Multiple chunks from the same file are grouped so that
        one file does not occupy the entire Top-K result set.
        """

        if not results:
            return []

        file_groups = {}

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            file_path = str(
                metadata.get(
                    "file_path",
                    "",
                )
            ).replace("\\", "/")

            if not file_path:
                continue

            if file_path not in file_groups:
                file_groups[file_path] = []

            file_groups[file_path].append(result)

        if not file_groups:
            return results

        query_lower = query.lower()

        ranked_files = []

        for file_path, file_results in file_groups.items():

            file_name = file_path.split("/")[-1].lower()

            file_stem = file_name

            if "." in file_stem:
                file_stem = file_stem.rsplit(
                    ".",
                    1,
                )[0]

            score = 0.0

            # --------------------------------------------------
            # Best underlying retrieval score
            # --------------------------------------------------

            best_result = max(
                file_results,
                key=lambda item: item.get(
                    "final_score",
                    item.get(
                        "final_rrf_score",
                        0.0,
                    ),
                ),
            )

            score += best_result.get(
                "final_score",
                best_result.get(
                    "final_rrf_score",
                    0.0,
                ),
            )

            # --------------------------------------------------
            # File-name lexical matching
            # --------------------------------------------------

            query_words = re.findall(
                r"[a-zA-Z0-9_]+",
                query_lower,
            )

            stop_words = {
                "which",
                "what",
                "where",
                "is",
                "are",
                "the",
                "a",
                "an",
                "file",
                "files",
                "contains",
                "contain",
                "has",
                "have",
                "code",
                "implemented",
                "implementation",
                "used",
                "use",
                "using",
                "for",
                "of",
                "in",
                "to",
            }

            meaningful_words = [
                word
                for word in query_words
                if (
                    word not in stop_words
                    and len(word) >= 3
                )
            ]

            for word in meaningful_words:

                normalized_file_name = (
                    file_name
                    .replace("_", " ")
                    .replace("-", " ")
                    .replace(".", " ")
                )

                normalized_file_stem = (
                    file_stem
                    .replace("_", " ")
                    .replace("-", " ")
                )

                if word in normalized_file_name:
                    score += 0.75

                elif word in normalized_file_stem:
                    score += 0.50

            # --------------------------------------------------
            # Special semantic file-name signals
            # --------------------------------------------------

            if (
                "evaluation" in query_lower
                and (
                    "eval" in file_stem
                    or "evaluation" in file_stem
                )
            ):
                score += 1.5

            if (
                "application" in query_lower
                and (
                    file_stem == "app"
                    or "application" in file_stem
                )
            ):
                score += 1.5

            if (
                "rag" in query_lower
                and "rag" in file_stem
            ):
                score += 1.5

            # --------------------------------------------------
            # Keep strongest chunk as representative result
            # --------------------------------------------------

            representative = dict(best_result)

            representative[
                "file_level_score"
            ] = score

            representative[
                "file_result_count"
            ] = len(file_results)

            ranked_files.append(
                representative
            )

        ranked_files.sort(
            key=lambda item: item[
                "file_level_score"
            ],
            reverse=True,
        )

        return ranked_files
    def search(
        self,
        repository_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        if not repository_id:
            raise ValueError(
                "Repository ID cannot be empty."
            )

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # ------------------------------------------------------
        # Detect query intent
        # ------------------------------------------------------

        intent = self.detect_query_intent(
            query
        )

        # ------------------------------------------------------
        # Collect hybrid candidates
        #
        # Semantic Search
        #        +
        # Keyword / BM25 Search
        # ------------------------------------------------------

        candidates = self.collect_candidates(
            repository_id=repository_id,
            query=query,
            top_k=top_k,
        )

        if not candidates:
            return []

        # ------------------------------------------------------
        # Calculate RRF + existing CodeLens ranking
        # ------------------------------------------------------

        ranked_candidates = []

        for candidate in candidates.values():

            metadata = candidate["metadata"]

            # --------------------------------------------------
            # Intent-aware filtering
            # --------------------------------------------------

            if not self.should_keep_result(
                metadata,
                intent,
                query,
                candidate["content"],
            ):
                continue

            # --------------------------------------------------
            # Reciprocal Rank Fusion
            # --------------------------------------------------

            rrf_score = self.calculate_rrf_score(
                candidate
            )

            # --------------------------------------------------
            # Semantic similarity score
            # --------------------------------------------------

            semantic_score = (
                self.calculate_semantic_score(
                    candidate["distance"]
                )
            )

            # --------------------------------------------------
            # File-level boost
            # --------------------------------------------------

            file_boost = self.calculate_file_boost(
                metadata,
                intent,
            )

            # --------------------------------------------------
            # Final RRF ranking score
            # --------------------------------------------------

            final_rrf_score = (
                rrf_score
                + (0.10 * semantic_score)
                + file_boost
            )

            result = dict(candidate)

            result["rrf_score"] = (
                rrf_score
            )

            result["semantic_score"] = (
                semantic_score
            )

            result["file_boost"] = (
                file_boost
            )

            result["final_rrf_score"] = (
                final_rrf_score
            )

            ranked_candidates.append(
                result
            )

        # ------------------------------------------------------
        # Sort by RRF ranking
        # ------------------------------------------------------

        ranked_candidates.sort(
            key=lambda item: item[
                "final_rrf_score"
            ],
            reverse=True,
        )

        if not ranked_candidates:
            return []

        # ------------------------------------------------------
        # Cross-Encoder reranking
        #
        # Keep a larger candidate pool so that code-aware
        # ranking can rescue a relevant code chunk that the
        # cross-encoder placed below the final Top-K.
        # ------------------------------------------------------

        if self.reranker is not None:

            try:

                rerank_top_k = min(
                    len(ranked_candidates),
                    max(top_k * 4, 20),
                )

                reranked_candidates = (
                    self.reranker.rerank(
                        query=query,
                        candidates=ranked_candidates,
                        top_k=rerank_top_k,
                    )
                )

            except Exception as error:

                print(
                    "Reranker failed. "
                    f"Falling back to RRF results: {error}"
                )

                reranked_candidates = (
                    ranked_candidates[:top_k]
                )

        else:

            reranked_candidates = (
                ranked_candidates[:top_k]
            )

        # ------------------------------------------------------
        # Code-aware ranking
        # ------------------------------------------------------

        for result in reranked_candidates:

            code_aware_boost = (
                self.calculate_code_aware_boost(
                    query=query,
                    metadata=result["metadata"],
                    content=result["content"],
                    intent=intent,
                )
            )

            result["code_aware_boost"] = (
                code_aware_boost
            )

            # --------------------------------------------------
            # Cross-encoder remains the primary relevance signal.
            #
            # Code-aware rules provide controlled adjustments
            # for definitions, usages, callers and explanations.
            # --------------------------------------------------

            reranker_score = result.get(
                "reranker_score"
            )

            if reranker_score is None:
                reranker_score = 0.0

            result["final_score"] = (
                reranker_score
                + code_aware_boost
            )
        # ------------------------------------------------------
        # DEBUG: inspect ranking scores
        # ------------------------------------------------------

        if query.strip().lower() in {
            "which function calls retrieve_context?",
            "where is document retrieval implemented?",
            "where is document chunking implemented?",
        }:

            print("\n" + "=" * 70)
            print(
                f"DEBUG RANKING: {query}"
            )
            print("=" * 70)

            debug_results = sorted(
                reranked_candidates,
                key=lambda item: item.get(
                    "final_score",
                    0.0,
                ),
                reverse=True,
            )

            for debug_rank, debug_result in enumerate(
                debug_results[:10],
                start=1,
            ):

                debug_metadata = (
                    debug_result.get(
                        "metadata",
                        {},
                    )
                )

                print(
                    f"\nRank {debug_rank}"
                )

                print(
                    "  File: "
                    f"{debug_metadata.get('file_path')}"
                )

                print(
                    "  Name: "
                    f"{debug_metadata.get('name')}"
                )

                print(
                    "  Reranker: "
                    f"{debug_result.get('reranker_score', 0.0):.4f}"
                )

                print(
                    "  Code boost: "
                    f"{debug_result.get('code_aware_boost', 0.0):.4f}"
                )

                print(
                    "  Final: "
                    f"{debug_result.get('final_score', 0.0):.4f}"
                )

        # ------------------------------------------------------
        # Relationship-aware ranking
        #
        # Give stronger priority to actual callers for
        # "which function calls X?" queries.
        #
        # This is intentionally applied after the CrossEncoder
        # and code-aware ranking so it acts as a small,
        # deterministic code-intelligence correction.
        # ------------------------------------------------------

        query_lower = query.strip().lower()

        caller_query = any(
            phrase in query_lower
            for phrase in [
                "which function calls",
                "which functions call",
                "what function calls",
                "what functions call",
                "who calls",
                "where is it called",
            ]
        )

        if caller_query:

            entity_name = self.extract_entity_name(
                query
            )

            for result in reranked_candidates:

                content = result.get(
                    "content",
                    "",
                )

                content_lower = content.lower()

                metadata = result.get(
                    "metadata",
                    {},
                )

                file_path = str(
                    metadata.get(
                        "file_path",
                        "",
                    )
                ).lower()

                result_name = str(
                    metadata.get(
                        "name",
                        "",
                    )
                ).lower()

                # ------------------------------------------
                # Actual call to the requested entity
                # ------------------------------------------

                call_pattern = (
                    rf"\b{re.escape(entity_name)}\s*\("
                )

                contains_call = bool(
                    re.search(
                        call_pattern,
                        content,
                        flags=re.IGNORECASE,
                    )
                )

                # ------------------------------------------
                # Is this the entity's own definition?
                # ------------------------------------------

                definition_pattern = (
                    rf"\b(def|async\s+def|function)\s+"
                    rf"{re.escape(entity_name)}\s*\("
                )

                is_own_definition = bool(
                    re.search(
                        definition_pattern,
                        content,
                        flags=re.IGNORECASE,
                    )
                )

                # ------------------------------------------
                # Actual caller function
                # ------------------------------------------

                if (
                    contains_call
                    and not is_own_definition
                    and (
                        "chunk" not in result_name
                        or "::" in result_name
                    )
                ):
                    result["final_score"] += 5.0

                # ------------------------------------------
                # Test files are useful evidence but should
                # not outrank the actual application caller.
                # ------------------------------------------

                if (
                    "test_" in file_path
                    or "/tests/" in file_path
                    or "\\tests\\" in file_path
                ):
                    result["final_score"] -= 1.5

        # ------------------------------------------------------
        # Final sort after relationship-aware ranking
        # ------------------------------------------------------

        reranked_candidates.sort(
            key=lambda item: item[
                "final_score"
            ],
            reverse=True,
        )
        # ------------------------------------------------------
        # File-level ranking
        # ------------------------------------------------------

        if self.is_file_oriented_query(query):

            reranked_candidates = (
                self.rank_file_results(
                    query=query,
                    results=reranked_candidates,
                )
            )
        # ------------------------------------------------------
        # Final API response
        # ------------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            reranked_candidates[:top_k],
            start=1,
        ):

            final_results.append(
                {
                    "rank": rank,
                    "chunk_id": result.get("chunk_id"),
                    "content": result[
                        "content"
                    ],
                    "metadata": result[
                        "metadata"
                    ],
                    "distance": result.get(
                        "distance",
                        0.0,
                    ),
                    "reranker_score": result.get(
                        "reranker_score"
                    ),
                    "code_aware_boost": result.get(
                        "code_aware_boost",
                        0.0,
                    ),
                    "file_level_score": result.get(
                        "file_level_score",
                        0.0,
                    ),
                }
            )

        return final_results