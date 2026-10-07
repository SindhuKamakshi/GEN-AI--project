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

    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
    ):
        self.search_service = search_service
        self.llm_service = llm_service

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
        """
        Convert retrieved repository chunks into
        structured context for Gemini.

        The context contains:
        - file path
        - function/class/chunk name
        - line numbers
        - chunk type
        - source code/content
        """

        if not search_results:
            return ""

        context_parts = []
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

            source_block = (
                f"SOURCE {index}\n"
                f"File: {file_path}\n"
                f"Name: {name or 'N/A'}\n"
                f"Lines: {start_line}-{end_line}\n"
                f"Chunk type: {chunk_type}\n"
                f"\n"
                f"Repository content:\n"
                f"{content}\n"
                f"\n"
                f"{'=' * 60}\n"
            )

            block_length = len(
                source_block
            )

            # -------------------------------------------------
            # Stop before exceeding context limit.
            # -------------------------------------------------

            if (
                total_characters + block_length
                > self.MAX_CONTEXT_CHARACTERS
            ):

                remaining = (
                    self.MAX_CONTEXT_CHARACTERS
                    - total_characters
                )

                # If there isn't enough room for a useful
                # additional source, stop.
                if remaining < 500:
                    break

                truncated_block = (
                    source_block[:remaining]
                    + "\n[Context truncated]"
                )

                context_parts.append(
                    truncated_block
                )

                total_characters += len(
                    truncated_block
                )

                break

            context_parts.append(
                source_block
            )

            total_characters += block_length

        return "\n".join(
            context_parts
        )

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

        search_results = (
            self.search_service.search(
                repository_id=repository_id,
                query=question,
                top_k=top_k,
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

        context = self.build_context(
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
            search_results
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