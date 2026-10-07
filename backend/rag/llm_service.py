from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time


load_dotenv()


class LLMService:
    """
    Handles Gemini generation for CodeLens AI using LangChain.

    The model is instructed to answer only from repository
    evidence and to distinguish actual implementation/evidence
    from mentions of questions, tests, documentation, or examples.
    """

    PRIMARY_MODEL = "gemini-3.8-flash"
    FALLBACK_MODEL = "gemini-3.5-flash-lite"

    MAX_RETRIES_PER_MODEL = 2
    RETRY_DELAY_SECONDS = 2

    TEMPERATURE = 0.2

    FALLBACK_ANSWER = (
        "I couldn't find evidence of that "
        "in the uploaded repository."
    )

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.models = [
            self.PRIMARY_MODEL,
            self.FALLBACK_MODEL,
        ]

        self.llms = {}

        for model_name in self.models:
            self.llms[model_name] = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=self.TEMPERATURE,
                max_retries=0,
            )

        print(
            "LLM service initialized successfully "
            "using LangChain."
        )

    # =========================================================
    # PROMPT BUILDER
    # =========================================================

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build a strongly grounded repository-analysis prompt.
        """

        return f"""
You are CodeLens AI, an AI-powered software repository
understanding assistant.

Your job is to answer the user's question using ONLY
actual evidence contained in the provided repository context.

============================================================
CRITICAL GROUNDING RULES
============================================================

1. Do NOT use outside knowledge.

2. Do NOT invent files, functions, classes, APIs,
   technologies, dependencies, behavior, or facts.

3. Repository context may contain:
   - source code
   - documentation
   - README files
   - test files
   - evaluation questions
   - examples
   - comments
   - configuration files

   These are not automatically evidence for the answer.

4. A question appearing inside a README, test file,
   evaluation list, example, comment, or documentation
   does NOT mean that the repository contains the answer
   to that question.

5. If the context only shows that the user's question is
   mentioned, tested, listed, or used as an example, but
   does NOT provide the actual answer, you MUST NOT answer
   the question using general knowledge.

6. For example, if the context contains:

   "The evaluation includes:
    What is the capital of France?"

   this is NOT evidence that the repository knows the
   capital of France.

   In that situation, respond exactly:

   "I couldn't find evidence of that in the uploaded repository."

7. Only answer a factual question when the repository
   context contains actual evidence supporting that answer.

8. For code questions, actual source code is strong evidence.

9. For implementation questions, distinguish between:
   - code that implements functionality
   - code that calls functionality
   - code that tests functionality
   - documentation describing functionality

10. If multiple files provide actual evidence, combine them
    and explain their relationship.

11. If the evidence is insufficient, incomplete, or only
    indirectly related to the question, use the fallback
    response instead of guessing.

12. Never answer from general knowledge simply because you
    recognize the question.

============================================================
REPOSITORY CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWERING INSTRUCTIONS
============================================================

First determine whether the repository context contains
actual evidence that can answer the question.

If YES:
- provide a clear answer
- mention relevant files
- mention functions/classes when applicable
- mention line ranges when available
- explain relationships between files when useful

If NO:
respond exactly:

"I couldn't find evidence of that in the uploaded repository."

Do not add outside knowledge to the fallback case.
"""

    # =========================================================
    # GEMINI GENERATION
    # =========================================================

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> dict:
        """
        Generate a grounded answer using Gemini through LangChain.

        Returns:

        {
            "answer": "...",
            "model": "..."
        }
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not context or not context.strip():
            return {
                "answer": self.FALLBACK_ANSWER,
                "model": "Not used",
            }

        prompt = self.build_prompt(
            question=question,
            context=context,
        )

        last_error = None

        # -----------------------------------------------------
        # Try Gemini models through LangChain
        # -----------------------------------------------------

        for model_name in self.models:

            llm = self.llms[model_name]

            for attempt in range(
                self.MAX_RETRIES_PER_MODEL
            ):

                try:
                    print(
                        f"Trying Gemini model through LangChain: "
                        f"{model_name} "
                        f"(attempt "
                        f"{attempt + 1}/"
                        f"{self.MAX_RETRIES_PER_MODEL})"
                    )

                    response = llm.invoke(prompt)

                    if response is None:
                        raise RuntimeError(
                            "Gemini returned an empty response."
                        )

                    answer = getattr(
    response,
    "content",
    None,
)

                    if isinstance(answer, list):
                        text_parts = []

                        for item in answer:
                            if isinstance(item, str):
                                text_parts.append(item)

                            elif isinstance(item, dict):
                                text = item.get("text")

                                if text:
                                    text_parts.append(str(text))

                        answer = "\n".join(text_parts)

                    elif isinstance(answer, dict):
                          answer = answer.get("text", "")

                    if not answer or not str(answer).strip():
                        raise RuntimeError(
        "Gemini returned empty text."
    )

                    answer = str(answer).strip()

                    print(
                        f"Gemini generation successful "
                        f"using LangChain + {model_name}"
                    )

                    return {
                        "answer": answer,
                        "model": model_name,
                    }

                except Exception as error:

                    last_error = error

                    print(
                        f"Gemini model {model_name} "
                        f"failed through LangChain: {error}"
                    )

                    if (
                        attempt
                        < self.MAX_RETRIES_PER_MODEL - 1
                    ):
                        time.sleep(
                            self.RETRY_DELAY_SECONDS
                        )

            print(
                f"Trying fallback Gemini model "
                f"after failure: {model_name}"
            )

        # -----------------------------------------------------
        # All models failed
        # -----------------------------------------------------

        raise RuntimeError(
            "All Gemini models failed. "
            f"Last error: {last_error}"
        )