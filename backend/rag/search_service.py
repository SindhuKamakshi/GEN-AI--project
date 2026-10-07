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
    # Intent Filtering
    # ==========================================================

    def should_keep_result(
        self,
        metadata: dict,
        intent: str,
    ) -> bool:

        extension = self.get_file_extension(metadata)
        file_name = self.get_file_name(metadata)

        if intent == "technology":
            return (
                file_name in self.HIGH_VALUE_FILES
                or extension in self.CONFIG_EXTENSIONS
                or extension in self.DOCUMENT_EXTENSIONS
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
            return extension in self.SOURCE_EXTENSIONS

        return True

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

        The ranking behavior is intentionally different for:

            definition
            usage
            explanation

        so that a caller does not outrank a definition for a
        definition query, and a definition does not outrank
        callers for a usage query.
        """

        if intent != "implementation":
            return 0.0

        code_intent = self.detect_code_intent(query)

        entity_name = self.extract_entity_name(query)

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

        is_exact_entity = (
            metadata_name_lower == entity_lower
        )

        is_definition_chunk = chunk_type in {
            "function",
            "class",
            "method",
        }

        # ------------------------------------------------------
        # Definition query
        # ------------------------------------------------------

        if code_intent == "definition":

            # Exact entity metadata match.
            if is_exact_entity:
                boost += (
                    self.EXACT_ENTITY_NAME_BONUS
                )

            # Actual AST/function/class definition.
            if is_definition_chunk:
                boost += (
                    self.FUNCTION_DEFINITION_BONUS
                )

            # Strong signal: metadata and definition both
            # identify the requested entity.
            if is_exact_entity:
                boost += self.DEFINITION_BONUS

            definition_patterns = [
                f"def {entity_name}(",
                f"class {entity_name}",
                f"function {entity_name}(",
                f"{entity_name} = function",
                f"{entity_name} = async function",
            ]

            if any(
                pattern.lower() in content_lower
                for pattern in definition_patterns
            ):
                boost += self.USAGE_BONUS

        # ------------------------------------------------------
        # Usage query
        # ------------------------------------------------------

        elif code_intent == "usage":

            # --------------------------------------------------
            # Definition penalty
            # --------------------------------------------------
            #
            # If this is the exact entity's own definition,
            # penalize it because the user asked for usages.
            #

            if (
                is_definition_chunk
                and is_exact_entity
            ):
                boost += (
                    self.USAGE_DEFINITION_PENALTY
                )

            # --------------------------------------------------
            # Actual caller detection
            # --------------------------------------------------

            if is_definition_chunk:

                call_patterns = [
                    f"{entity_name}(",
                    f".{entity_name}(",
                ]

                contains_call = any(
                    pattern.lower() in content_lower
                    for pattern in call_patterns
                )

                definition_patterns = [
                    f"def {entity_name}(",
                    f"function {entity_name}(",
                    f"class {entity_name}",
                ]

                is_own_definition = any(
                    pattern.lower() in content_lower
                    for pattern in definition_patterns
                )

                # A function containing the entity call is a
                # genuine usage only if it isn't the entity's
                # own definition.
                if (
                    contains_call
                    and not (
                        is_exact_entity
                        and is_own_definition
                    )
                ):
                    boost += (
                        self.USAGE_CALL_BONUS
                    )

            # --------------------------------------------------
            # Text/code references
            # --------------------------------------------------

            if chunk_type in {
                "text_block",
                "code_block",
            }:
                boost += (
                    self.USAGE_REFERENCE_BONUS
                )

            # --------------------------------------------------
            # Import / call / reference patterns
            # --------------------------------------------------

            usage_patterns = [
                f"import {entity_name}",
                f"from {entity_name}",
                f"{entity_name}(",
                f".{entity_name}(",
            ]

            contains_usage_pattern = any(
                pattern.lower() in content_lower
                for pattern in usage_patterns
            )

            # Python-style:
            #
            # from rag import retrieve_context
            #
            # This is not covered by "from retrieve_context".
            #
            python_import_pattern = (
                rf"\bfrom\s+\S+\s+import\s+.*\b"
                rf"{re.escape(entity_name)}\b"
            )

            generic_import_pattern = (
                rf"\bimport\s+.*\b"
                rf"{re.escape(entity_name)}\b"
            )

            if (
                contains_usage_pattern
                or re.search(
                    python_import_pattern,
                    content_text,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    generic_import_pattern,
                    content_text,
                    flags=re.IGNORECASE,
                )
            ):
                boost += (
                    self.USAGE_CALL_BONUS
                )

        # ------------------------------------------------------
        # Explanation query
        # ------------------------------------------------------

        elif code_intent == "explanation":

            # --------------------------------------------------
            # Exact entity definition
            # --------------------------------------------------
            #
            # Only boost a definition chunk strongly if it is
            # actually the requested entity.
            #
            # This prevents unrelated functions such as:
            #
            #     clean_text()
            #
            # from receiving a definition bonus.
            #

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

            # --------------------------------------------------
            # Related caller
            # --------------------------------------------------
            #
            # If another function contains a real reference to
            # the requested entity, give it a smaller bonus.
            #

            elif (
                is_definition_chunk
                and not is_exact_entity
            ):

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
    # Search
    # ==========================================================

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

        intent = self.detect_query_intent(
            query
        )

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

            if not self.should_keep_result(
                metadata,
                intent,
            ):
                continue

            rrf_score = self.calculate_rrf_score(
                candidate
            )

            semantic_score = (
                self.calculate_semantic_score(
                    candidate["distance"]
                )
            )

            file_boost = self.calculate_file_boost(
                metadata,
                intent,
            )

            final_rrf_score = (
                rrf_score
                + (0.10 * semantic_score)
                + file_boost
            )

            result = dict(candidate)

            result["rrf_score"] = rrf_score
            result["semantic_score"] = semantic_score
            result["file_boost"] = file_boost
            result["final_rrf_score"] = (
                final_rrf_score
            )

            ranked_candidates.append(result)

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

            # The reranker remains the main relevance signal.
            # Code-aware rules provide controlled adjustments
            # for repository-specific code relationships.
            reranker_score = result.get(
                "reranker_score"
            )

            if reranker_score is None:
                reranker_score = 0.0

            result["final_score"] = (
                reranker_score
                + code_aware_boost
            )

        reranked_candidates.sort(
            key=lambda item: item[
                "final_score"
            ],
            reverse=True,
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
                    "content": result["content"],
                    "metadata": result["metadata"],
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
                }
            )

        return final_results