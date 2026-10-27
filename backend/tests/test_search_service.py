from rag.search_service import SearchService


QUESTION = (
    "Which dataset file does the notebook load, "
    "and which library is used to read it?"
)
DATASET_READER_QUESTION = (
    "Which dataset does the notebook load, and which function reads it?"
)
REPOSITORY_ID = "repository-under-test"


class FakeEmbeddingService:
    def embed_query(self, query):
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self, candidates):
        self.candidates = candidates
        self.repository_ids = []

    def search(self, repository_id, query_embedding, top_k):
        self.repository_ids.append(repository_id)
        return {
            "documents": [[item["content"] for item in self.candidates]],
            "metadatas": [[item["metadata"] for item in self.candidates]],
            "distances": [[item["distance"] for item in self.candidates]],
            "ids": [[item["chunk_id"] for item in self.candidates]],
        }


class FakeKeywordStore:
    def __init__(self, candidates):
        self.candidates = candidates
        self.repository_ids = []

    def search(self, repository_id, query, top_k):
        self.repository_ids.append(repository_id)
        return [
            {
                "chunk_id": item["chunk_id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "rank": rank,
                "keyword_score": 1.0 / rank,
            }
            for rank, item in enumerate(self.candidates, start=1)
        ]


class FakeReranker:
    def rerank(self, query, candidates, top_k):
        ranked = sorted(
            candidates,
            key=lambda item: "pd.read_csv" in item["content"],
            reverse=True,
        )
        for item in ranked:
            item["reranker_score"] = (
                1.0 if "pd.read_csv" in item["content"] else 0.0
            )
        return ranked[:top_k]


def make_candidate(chunk_id, file_path, content, **metadata):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "distance": 0.1,
        "metadata": {
            "repository_id": REPOSITORY_ID,
            "file_path": file_path,
            **metadata,
        },
    }


def make_service(candidates):
    vector_store = FakeVectorStore(candidates)
    keyword_store = FakeKeywordStore(candidates)
    service = SearchService(
        FakeEmbeddingService(),
        vector_store,
        keyword_store,
        FakeReranker(),
    )
    return service, vector_store, keyword_store


def test_notebook_dataset_question_keeps_relevant_cell_ahead_of_distractors():
    notebook_cell = make_candidate(
        "notebook-cell-5",
        "saas_churn_project (2).ipynb",
        (
            "Notebook: saas_churn_project (2).ipynb\n"
            "Cell: 5\n"
            "Cell type: code\n"
            "Language: Python\n\n"
            "import pandas as pd\n"
            "df = pd.read_csv('ravenstack_subscriptions.csv')\n"
            "df.head()"
        ),
        chunk_type="notebook_cell",
        notebook_cell_type="code",
        cell_number=5,
        cell_language="Python",
    )
    candidates = [
        make_candidate(
            "readme",
            "README.md",
            "Project documentation and general description.",
        ),
        make_candidate(
            "gitignore",
            "gitignore.txt",
            "Ignore generated files and local data.",
        ),
        notebook_cell,
        make_candidate(
            "unrelated-notebook-cell",
            "unrelated.ipynb",
            "Notebook: unrelated.ipynb\nCell type: code\nimport numpy as np\n",
            chunk_type="notebook_cell",
            notebook_cell_type="code",
            cell_number=2,
            cell_language="Python",
        ),
    ]
    service, vector_store, keyword_store = make_service(candidates)

    results = service.search(
        repository_id=REPOSITORY_ID,
        query=QUESTION,
        top_k=5,
    )

    assert results
    assert results[0]["metadata"]["file_path"] == (
        "saas_churn_project (2).ipynb"
    )
    assert results[0]["chunk_id"] == notebook_cell["chunk_id"]
    assert results[0]["metadata"]["cell_number"] == 5
    assert results[0]["metadata"]["notebook_cell_type"] == "code"
    assert results[0]["metadata"]["repository_id"] == REPOSITORY_ID
    assert all(
        result["metadata"]["file_path"] != "unrelated.ipynb"
        for result in results
    )
    assert vector_store.repository_ids == [REPOSITORY_ID]
    assert keyword_store.repository_ids == [REPOSITORY_ID]


def test_notebook_technology_exception_requires_reader_code_and_matching_query():
    metadata = {
        "file_path": "notebook.ipynb",
        "chunk_type": "notebook_cell",
        "notebook_cell_type": "code",
    }

    assert SearchService.is_relevant_notebook_data_result(
        metadata,
        QUESTION,
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
    )
    assert not SearchService.is_relevant_notebook_data_result(
        metadata,
        QUESTION,
        "import numpy as np\nvalues = np.array([1, 2, 3])",
    )
    assert not SearchService.is_relevant_notebook_data_result(
        {**metadata, "notebook_cell_type": "markdown"},
        QUESTION,
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
    )
    assert not SearchService.is_relevant_notebook_data_result(
        metadata,
        "Which libraries does this project use?",
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
    )


def test_notebook_import_discovery_keeps_multiple_imports_without_reader():
    query = "Which Python libraries are imported in the notebook?"
    metadata = {
        "repository_id": REPOSITORY_ID,
        "file_path": "notebook.ipynb",
        "language": "Jupyter Notebook",
        "chunk_type": "notebook_cell",
        "name": "cell_1_part_1_code",
        "start_line": 1,
        "end_line": 1,
        "notebook_cell_type": "code",
        "cell_number": 1,
        "cell_part": 1,
        "cell_language": "Python",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(query)

    assert intent == "technology"
    assert service.should_keep_result(
        metadata,
        intent,
        query,
        "import os\nimport json\nfrom pathlib import Path\n",
    )


def test_notebook_import_discovery_keeps_import_only_cell():
    query = "Which Python libraries are imported in the notebook?"
    metadata = {
        "repository_id": REPOSITORY_ID,
        "file_path": "notebook.ipynb",
        "language": "Jupyter Notebook",
        "chunk_type": "notebook_cell",
        "name": "cell_2_part_1_code",
        "start_line": 2,
        "end_line": 2,
        "notebook_cell_type": "code",
        "cell_number": 2,
        "cell_part": 1,
        "cell_language": "Python",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(query)

    assert service.should_keep_result(
        metadata,
        intent,
        query,
        "from pathlib import Path\n",
    )


def test_notebook_import_discovery_rejects_cell_without_imports():
    query = "Which Python libraries are imported in the notebook?"
    metadata = {
        "repository_id": REPOSITORY_ID,
        "file_path": "notebook.ipynb",
        "language": "Jupyter Notebook",
        "chunk_type": "notebook_cell",
        "name": "cell_3_part_1_code",
        "start_line": 3,
        "end_line": 3,
        "notebook_cell_type": "code",
        "cell_number": 3,
        "cell_part": 1,
        "cell_language": "Python",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(query)

    assert not service.should_keep_result(
        metadata,
        intent,
        query,
        "values = [1, 2, 3]\n",
    )


def test_notebook_import_discovery_rejects_markdown_cell():
    query = "Which Python libraries are imported in the notebook?"
    metadata = {
        "repository_id": REPOSITORY_ID,
        "file_path": "notebook.ipynb",
        "language": "Jupyter Notebook",
        "chunk_type": "notebook_cell",
        "name": "cell_4_part_1_markdown",
        "start_line": 4,
        "end_line": 4,
        "notebook_cell_type": "markdown",
        "cell_number": 4,
        "cell_part": 1,
        "cell_language": "Markdown",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(query)

    assert not service.should_keep_result(
        metadata,
        intent,
        query,
        "The notebook imports os and json.",
    )


def test_unrelated_technology_query_does_not_admit_notebook_imports():
    query = "Which libraries does this project use?"
    metadata = {
        "repository_id": REPOSITORY_ID,
        "file_path": "notebook.ipynb",
        "language": "Jupyter Notebook",
        "chunk_type": "notebook_cell",
        "name": "cell_5_part_1_code",
        "start_line": 5,
        "end_line": 5,
        "notebook_cell_type": "code",
        "cell_number": 5,
        "cell_part": 1,
        "cell_language": "Python",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(query)

    assert intent == "technology"
    assert not service.should_keep_result(
        metadata,
        intent,
        query,
        "import os\n",
    )


def test_implementation_intent_notebook_import_query_keeps_import_cell():
    query = "Where are imports in the notebook?"
    metadata = {
        "repository_id": REPOSITORY_ID,
        "file_path": "notebook.ipynb",
        "language": "Jupyter Notebook",
        "chunk_type": "notebook_cell",
        "name": "cell_6_part_1_code",
        "start_line": 6,
        "end_line": 6,
        "notebook_cell_type": "code",
        "cell_number": 6,
        "cell_part": 1,
        "cell_language": "Python",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(query)

    assert intent == "implementation"
    assert service.should_keep_result(
        metadata,
        intent,
        query,
        "import os\nfrom pathlib import Path\n",
    )
    assert not service.should_keep_result(
        metadata,
        intent,
        query,
        "values = [1, 2, 3]\n",
    )


def test_dataset_loading_question_still_requires_reader_call():
    metadata = {
        "repository_id": REPOSITORY_ID,
        "file_path": "notebook.ipynb",
        "language": "Jupyter Notebook",
        "chunk_type": "notebook_cell",
        "name": "cell_5_part_1_code",
        "start_line": 5,
        "end_line": 5,
        "notebook_cell_type": "code",
        "cell_number": 5,
        "cell_part": 1,
        "cell_language": "Python",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(QUESTION)

    assert service.should_keep_result(
        metadata,
        intent,
        QUESTION,
        "import pandas as pd\n"
        "df = pd.read_csv('ravenstack_subscriptions.csv')\n",
    )
    assert not service.should_keep_result(
        metadata,
        intent,
        QUESTION,
        "import pandas as pd\n",
    )


def test_dataset_reader_question_keeps_notebook_reader_cell():
    loader_cell = make_candidate(
        "notebook-loader",
        "notebook.ipynb",
        (
            "Notebook: notebook.ipynb\n"
            "Cell: 1\n"
            "Cell type: code\n"
            "Language: Python\n\n"
            "import pandas as pd\n"
            "data = pd.read_csv(\"data.csv\")"
        ),
        chunk_type="notebook_cell",
        notebook_cell_type="code",
        cell_number=1,
        cell_language="Python",
    )
    unrelated_cell = make_candidate(
        "notebook-unrelated",
        "notebook.ipynb",
        (
            "Notebook: notebook.ipynb\n"
            "Cell: 2\n"
            "Cell type: code\n"
            "Language: Python\n\n"
            "import numpy as np\n"
            "values = np.array([1, 2, 3])"
        ),
        chunk_type="notebook_cell",
        notebook_cell_type="code",
        cell_number=2,
        cell_language="Python",
    )
    service, _, _ = make_service([loader_cell, unrelated_cell])
    intent = service.detect_query_intent(DATASET_READER_QUESTION)

    assert intent == "implementation"
    results = service.search(
        repository_id=REPOSITORY_ID,
        query=DATASET_READER_QUESTION,
        top_k=5,
    )

    assert any(
        result["metadata"]["cell_number"] == 1
        for result in results
    )
    assert all(
        result["metadata"]["cell_number"] != 2
        for result in results
    )


def test_dataset_reader_question_rejects_imports_without_reader_call():
    metadata = {
        "file_path": "notebook.ipynb",
        "chunk_type": "notebook_cell",
        "notebook_cell_type": "code",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(DATASET_READER_QUESTION)

    assert not service.should_keep_result(
        metadata,
        intent,
        DATASET_READER_QUESTION,
        "import pandas as pd\ndata = [1, 2, 3]\n",
    )


def test_dataset_reader_question_rejects_markdown_cell():
    metadata = {
        "file_path": "notebook.ipynb",
        "chunk_type": "notebook_cell",
        "notebook_cell_type": "markdown",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(DATASET_READER_QUESTION)

    assert not service.should_keep_result(
        metadata,
        intent,
        DATASET_READER_QUESTION,
        "The notebook loads a dataset using a reader.",
    )


def test_unrelated_implementation_query_rejects_notebook_cell():
    query = "Where is the model training logic implemented?"
    metadata = {
        "file_path": "notebook.ipynb",
        "chunk_type": "notebook_cell",
        "notebook_cell_type": "code",
    }
    service, _, _ = make_service([])
    intent = service.detect_query_intent(query)

    assert intent == "implementation"
    assert not service.should_keep_result(
        metadata,
        intent,
        query,
        "import pandas as pd\n"
        "data = pd.read_csv(\"data.csv\")\n",
    )
