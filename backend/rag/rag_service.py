import re

from rag.search_service import SearchService
from rag.llm_service import LLMService


class RAGService:
    """
    Retrieval-Augmented Generation service for CodeLens AI.

    Flow:

    User Question
          ↓
    Semantic Search
          ↓
    Ranked Repository Chunks
          ↓
    Context Builder
          ↓
    Gemini
          ↓
    Grounded Answer
          ↓
    Sources
    """

    # Maximum amount of repository context sent to Gemini.
    #
    # This prevents large repositories from producing
    # unnecessarily large prompts.
    MAX_CONTEXT_CHARACTERS = 24000
    COMPOUND_QUESTION_RETRIEVAL_FLOOR = 40
    MAX_COMPOUND_QUESTION_PARTS = 4

    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        dependency_service,
    ):
        self.search_service = search_service
        self.llm_service = llm_service
        self.dependency_service = dependency_service

    @classmethod
    def _split_compound_question(cls, question: str) -> list[str]:
        parts = [
            part.strip(" \t\r\n,;.")
            for part in re.split(
                r"\s*,?\s+(?:and|or)\s+"
                r"(?=(?:what|which|who|where|when|why|how|whose)\b)",
                question,
                flags=re.IGNORECASE,
            )
            if part.strip(" \t\r\n,;.")
        ]

        if not 1 < len(parts) <= cls.MAX_COMPOUND_QUESTION_PARTS:
            return [question]

        questions = []
        for part in parts:
            part = part[0].upper() + part[1:]
            if not part.endswith(("?", "!")):
                part += "?"
            questions.append(part)

        return [
            questions[0],
            *(
                f"{part} Context: {'; '.join(questions[:index])}"
                for index, part in enumerate(questions[1:], start=1)
            ),
        ]

    @classmethod
    def _workflow_search_queries(cls, question: str) -> list[str]:
        question_text = question.strip()
        if not question_text:
            return [question]

        lower_question = question_text.lower()
        workflow_markers = (
            "explain how",
            "describe how",
            "how is",
            "how does",
            "prepare",
            "preprocess",
            "before training",
            "feature engineering",
            "workflow",
            "pipeline",
        )

        if not any(marker in lower_question for marker in workflow_markers):
            return [question_text]

        subject = ""
        for phrase in (
            "input features",
            "features",
            "dataset",
            "data",
            "target variable",
            "engagement score",
            "project",
        ):
            if phrase in lower_question:
                subject = phrase
                break

        if not subject:
            tokens = re.findall(r"[a-z0-9_]+", lower_question)
            stop_words = {
                "explain",
                "describe",
                "how",
                "what",
                "which",
                "where",
                "when",
                "why",
                "who",
                "the",
                "this",
                "project",
                "before",
                "training",
                "does",
                "is",
                "are",
                "and",
                "or",
            }
            filtered_tokens = [
                token for token in tokens if token not in stop_words and len(token) > 2
            ]
            subject = " ".join(filtered_tokens[:4])

        queries = [question_text]
        suffixes = [
            "preprocessing",
            "feature engineering",
            "target variable",
        ]

        for suffix in suffixes:
            candidate = f"{subject} {suffix}" if subject else suffix
            if candidate not in queries:
                queries.append(candidate)

        if len(queries) > 4:
            queries = queries[:4]

        return queries

    @staticmethod
    def _result_identity(result: dict, repository_id: str):
        metadata = result.get("metadata", {})
        result_repository_id = metadata.get("repository_id") or repository_id
        chunk_id = result.get("chunk_id")
        if chunk_id is not None:
            return ("chunk_id", result_repository_id, chunk_id)

        file_path = metadata.get("file_path")
        if file_path is None:
            return None

        if metadata.get("chunk_type") == "notebook_cell":
            fields = (
                metadata.get("cell_number"),
                metadata.get("cell_part"),
                metadata.get("notebook_cell_type"),
            )
            if any(value is None for value in fields):
                return None
            return ("notebook_cell", result_repository_id, file_path, *fields)

        fields = (
            metadata.get("chunk_type"),
            metadata.get("name"),
            metadata.get("start_line"),
            metadata.get("end_line"),
        )
        if any(value is None for value in fields):
            return None
        return ("source_chunk", result_repository_id, file_path, *fields)

    @classmethod
    def _merge_compound_search_results(
        cls,
        result_sets: list[list[dict]],
        repository_id: str,
        limit: int,
    ) -> list[dict]:
        if limit <= 0:
            return []

        merged = []
        seen = set()
        for rank in range(max(map(len, result_sets), default=0)):
            for results in result_sets:
                if rank >= len(results):
                    continue

                result = results[rank]
                metadata_repository_id = result.get(
                    "metadata", {}
                ).get("repository_id")
                if (
                    metadata_repository_id is not None
                    and metadata_repository_id != repository_id
                ):
                    continue

                identity = cls._result_identity(result, repository_id)
                if identity is not None and identity in seen:
                    continue
                if identity is not None:
                    seen.add(identity)

                merged.append(result)
                if len(merged) >= limit:
                    return merged

        return merged

    # =========================================================
    # SOURCE LABEL
    # =========================================================

    def build_source_label(
        self,
        metadata: dict,
    ) -> str:
        """
        Create a human-readable source label.
        """

        file_path = metadata.get(
            "file_path",
            "Unknown file",
        )

        start_line = metadata.get(
            "start_line"
        )

        end_line = metadata.get(
            "end_line"
        )

        name = metadata.get(
            "name"
        )

        if start_line is not None and end_line is not None:

            line_info = (
                f"Lines {start_line}-{end_line}"
            )

        elif start_line is not None:

            line_info = (
                f"Line {start_line}"
            )

        else:

            line_info = "Line information unavailable"

        if name:

            return (
                f"{file_path} | "
                f"{name} | "
                f"{line_info}"
            )

        return (
            f"{file_path} | "
            f"{line_info}"
        )

    # =========================================================
    # CONTEXT BUILDER
    # =========================================================

    def build_context(
        self,
        search_results: list[dict],
    ) -> str:
        context, _ = self._build_context_details(search_results)
        return context

    def _build_context_details(
        self,
        search_results: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        Build bounded context and retain exactly the results represented in it.
        """

        if not search_results:
            return "", []

        context_parts = []
        context_results = []
        total_characters = 0

        for index, result in enumerate(
            search_results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            content = result.get(
                "content",
                "",
            )
            dependency_context = result.get(
    "dependency_context",
    {}
)

            dependencies = dependency_context.get(
    "dependencies",
    []
)

            dependents = dependency_context.get(
    "dependents",
    []
)
            if not content:
                continue

            file_path = metadata.get(
                "file_path",
                "Unknown file",
            )

            name = metadata.get(
                "name",
                "",
            )

            start_line = metadata.get(
                "start_line",
                "?",
            )

            end_line = metadata.get(
                "end_line",
                "?",
            )

            chunk_type = metadata.get(
                "chunk_type",
                "unknown",
            )
            dependency_lines = []

            for dependency in dependencies:
                dependency_lines.append(
                    f"- {dependency.get('from_file')}::"
                    f"{dependency.get('from_entity')} "
                    f"{dependency.get('relationship')} "
                    f"{dependency.get('to_file')}::"
                    f"{dependency.get('to_entity')}"
                )

            for dependent in dependents:
                dependency_lines.append(
                    f"- {dependent.get('from_file')}::"
                    f"{dependent.get('from_entity')} "
                    f"{dependent.get('relationship')} "
                    f"{dependent.get('to_file')}::"
                    f"{dependent.get('to_entity')}"
                )

            dependency_section = ""

            if dependency_lines:
                dependency_section = (
                    "Dependency relationships:\n"
                    + "\n".join(dependency_lines)
                    + "\n\n"
                )
            source_block = (
                f"SOURCE {index}\n"
                f"File: {file_path}\n"
                f"Name: {name or 'N/A'}\n"
                f"Lines: {start_line}-{end_line}\n"
                f"Chunk type: {chunk_type}\n"
                f"\n"
                f"{dependency_section}"
                f"Repository content:\n"
                f"{content}\n"
                f"\n"
                f"{'=' * 60}\n"
            )

            block_length = len(
                source_block
            )
            separator = "\n" if context_parts else ""

            # -------------------------------------------------
            # Stop before exceeding context limit.
            # -------------------------------------------------

            if (
                total_characters + len(separator) + block_length
                > self.MAX_CONTEXT_CHARACTERS
            ):

                remaining = (
                    self.MAX_CONTEXT_CHARACTERS
                    - total_characters
                    - len(separator)
                )

                # If there isn't enough room for a useful
                # additional source, stop.
                if remaining < 500:
                    break

                truncation_marker = "\n[Context truncated]"
                truncated_block = (
                    source_block[
                        :max(0, remaining - len(truncation_marker))
                    ]
                    + truncation_marker
                )

                context_parts.append(truncated_block)
                context_results.append(result)
                total_characters += len(separator) + len(truncated_block)

                break

            context_parts.append(source_block)
            context_results.append(result)
            total_characters += len(separator) + block_length

        return "\n".join(context_parts), context_results

    # =========================================================
    # SOURCE BUILDER
    # =========================================================

    def build_sources(
        self,
        search_results: list[dict],
    ) -> list[dict]:
        """
        Convert search metadata into clean source references.
        """

        sources = []

        seen_sources = set()

        for result in search_results:

            metadata = result.get(
                "metadata",
                {},
            )

            file_path = metadata.get(
                "file_path",
                "Unknown file",
            )

            start_line = metadata.get(
                "start_line"
            )

            end_line = metadata.get(
                "end_line"
            )

            chunk_type = metadata.get(
                "chunk_type"
            )

            name = metadata.get(
                "name"
            )

            distance = result.get(
                "distance"
            )

            source_key = (
                file_path,
                start_line,
                end_line,
                name,
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            sources.append(
                {
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_type": chunk_type,
                    "name": name,
                    "distance": distance,
                }
            )

        return sources

    # =========================================================
    # ANSWER QUESTION
    # =========================================================

    def answer_question(
        self,
        repository_id: str,
        question: str,
        top_k: int = 5,
        repository_path = None
    ) -> dict:
        """
        Complete RAG pipeline.

        1. Retrieve relevant repository chunks
        2. Build structured context
        3. Send context + question to Gemini
        4. Return answer + sources + retrieved chunks
        """

        if not repository_id:
            raise ValueError(
                "repository_id is required."
            )

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # -----------------------------------------------------
        # STEP 1: RETRIEVE
        # -----------------------------------------------------

        retrieval_queries = self._split_compound_question(question)
        if len(retrieval_queries) == 1:
            retrieval_queries = self._workflow_search_queries(question)

        if len(retrieval_queries) == 1:
            search_results = self.search_service.search(
                repository_id=repository_id,
                query=question,
                top_k=top_k,
            )
        else:
            search_top_k = max(
                top_k,
                self.COMPOUND_QUESTION_RETRIEVAL_FLOOR,
            )
            result_sets = [
                self.search_service.search(
                    repository_id=repository_id,
                    query=part,
                    top_k=search_top_k,
                )
                for part in retrieval_queries
            ]
            search_results = self._merge_compound_search_results(
                result_sets,
                repository_id,
                search_top_k,
            )

        search_results = [
            result
            for result in search_results
            if result.get("metadata", {}).get("repository_id")
            in (None, repository_id)
        ]

        search_results = (
            self.dependency_service.expand_search_results(
                repository_id=repository_id,
                search_results=search_results,
                repository_path=repository_path,
            )
        )

        # -----------------------------------------------------
        # STEP 2: NO RESULTS
        # -----------------------------------------------------

        if not search_results:

            return {
                "answer": (
                    "I couldn't find evidence of that "
                    "in the uploaded repository."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "model": "Not used",
            }

        # -----------------------------------------------------
        # STEP 3: BUILD CONTEXT
        # -----------------------------------------------------

        context, context_results = self._build_context_details(
            search_results
        )

        # -----------------------------------------------------
        # STEP 4: SAFETY CHECK
        # -----------------------------------------------------

        if not context.strip():

            return {
                "answer": (
                    "I couldn't find evidence of that "
                    "in the uploaded repository."
                ),
                "sources": [],
                "retrieved_chunks": search_results,
                "model": "Not used",
            }

        # -----------------------------------------------------
        # STEP 5: GENERATE ANSWER
        # -----------------------------------------------------

        generation_result = (
            self.llm_service.generate_answer(
                question=question,
                context=context,
            )
        )

        # -----------------------------------------------------
        # STEP 6: BUILD SOURCES
        # -----------------------------------------------------

        sources = self.build_sources(
            context_results
        )

        # -----------------------------------------------------
        # STEP 7: FINAL RESPONSE
        # -----------------------------------------------------

        return {
            "answer": generation_result[
                "answer"
            ],
            "sources": sources,
            "retrieved_chunks": search_results,
            "model": generation_result[
                "model"
            ],
        }