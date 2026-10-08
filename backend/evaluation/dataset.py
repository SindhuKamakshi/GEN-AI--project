"""
Evaluation dataset for CodeLens AI retrieval.
"""

EVALUATION_DATASET = [

    # =========================================================
    # Definition — Entity level
    # =========================================================

    {
        "id": "definition_retrieve_context",
        "query": "Where is retrieve_context defined?",
        "evaluation_type": "entity",
        "relevant_items": {"rag.py::retrieve_context"},
    },

    {
        "id": "definition_answer_question",
        "query": "Where is answer_question defined?",
        "evaluation_type": "entity",
        "relevant_items": {"rag.py::answer_question"},
    },

    {
        "id": "definition_create_chunks",
        "query": "Where is create_chunks defined?",
        "evaluation_type": "entity",
        "relevant_items": {"rag.py::create_chunks"},
    },

    {
        "id": "definition_store_chunks",
        "query": "Where is store_chunks defined?",
        "evaluation_type": "entity",
        "relevant_items": {"rag.py::store_chunks"},
    },

    # =========================================================
    # Usage
    # =========================================================

    {
        "id": "usage_retrieve_context",
        "query": "Where is retrieve_context used?",
        "evaluation_type": "mixed",
        "relevant_items": {
            "rag.py::answer_question",
            "test_retrieval.py",
        },
    },

    {
        "id": "usage_answer_question",
        "query": "Where is answer_question used?",
        "evaluation_type": "file",
        "relevant_items": {
            "app.py",
            "evaluate_rag.py",
        },
    },

    {
        "id": "usage_create_chunks",
        "query": "Where is create_chunks used?",
        "evaluation_type": "file",
        "relevant_items": {
            "app.py",
            "test_ingestion.py",
        },
    },

    # =========================================================
    # Dependency
    # =========================================================

    {
        "id": "dependency_retrieve_context",
        "query": "Which function calls retrieve_context?",
        "evaluation_type": "entity",
        "relevant_items": {
            "rag.py::answer_question",
        },
    },

    {
        "id": "dependency_answer_question",
        "query": "Which files import answer_question?",
        "evaluation_type": "file",
        "relevant_items": {
            "app.py",
            "evaluate_rag.py",
        },
    },

    {
        "id": "dependency_retrieve_context_test",
        "query": "Which file imports retrieve_context?",
        "evaluation_type": "file",
        "relevant_items": {
            "test_retrieval.py",
        },
    },

    {
        "id": "dependency_create_chunks",
        "query": "Which files use create_chunks?",
        "evaluation_type": "file",
        "relevant_items": {
            "app.py",
            "test_ingestion.py",
        },
    },

    # =========================================================
    # File / Repository — File level
    # =========================================================

    {
        "id": "file_rag",
        "query": "Which file contains the RAG implementation?",
        "evaluation_type": "file",
        "relevant_items": {"rag.py"},
    },

    {
        "id": "file_application",
        "query": "Which file contains the application entry point?",
        "evaluation_type": "file",
        "relevant_items": {"app.py"},
    },

    {
        "id": "file_evaluation",
        "query": "Which file contains RAG evaluation code?",
        "evaluation_type": "file",
        "relevant_items": {"evaluate_rag.py"},
    },

    # =========================================================
    # Architecture — Entity level
    # =========================================================

    {
        "id": "architecture_retrieval",
        "query": "Where is document retrieval implemented?",
        "evaluation_type": "entity",
        "relevant_items": {"rag.py::retrieve_context"},
    },

    {
        "id": "architecture_answering",
        "query": "Where is the question answering pipeline implemented?",
        "evaluation_type": "entity",
        "relevant_items": {"rag.py::answer_question"},
    },

    {
        "id": "architecture_chunking",
        "query": "Where is document chunking implemented?",
        "evaluation_type": "entity",
        "relevant_items": {"rag.py::create_chunks"},
    },

    # =========================================================
    # Technology — File level
    # =========================================================

    {
        "id": "technology_embeddings",
        "query": "Where is the embedding model used?",
        "evaluation_type": "file",
        "relevant_items": {"rag.py"},
    },

    {
        "id": "technology_vector_database",
        "query": "Where is the vector database used?",
        "evaluation_type": "file",
        "relevant_items": {"rag.py"},
    },

    {
        "id": "technology_llm",
        "query": "Where is the language model used to generate answers?",
        "evaluation_type": "file",
        "relevant_items": {"rag.py"},
    },
]