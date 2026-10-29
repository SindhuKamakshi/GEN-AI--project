import json

from analysis.dependency_analyzer import DependencyAnalyzer
from analysis.dependency_service import DependencyService
from analysis.importance_analyzer import ImportantFileAnalyzer
from analysis.overview_analyzer import RepositoryOverviewAnalyzer
from analysis.tech_stack_detector import TechnologyStackDetector
from repository.chunker import chunk_file
from rag.keyword_store import KeywordStore
from rag.rag_service import RAGService
from rag.search_service import SearchService
from rag.vector_store import VectorStore


def write_notebook(path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cells": cells, "nbformat": 4, "nbformat_minor": 5}),
        encoding="utf-8",
    )
    return path


def code_cell(source, metadata=None, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": metadata or {},
        "outputs": outputs or [],
        "source": source,
    }


def test_notebook_chunks_include_source_metadata_and_exclude_outputs(tmp_path):
    notebook = write_notebook(
        tmp_path / "analysis.ipynb",
        [
            code_cell(
                ["import pandas as pd\n", "df = pd.read_csv('customers.csv')"],
                {"tags": ["important"]},
                [{"output_type": "stream", "text": ["output-only value"]}],
            ),
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Dataset\n", "The notebook reads customer data."],
            },
            code_cell(""),
        ],
    )

    chunks = chunk_file(notebook, tmp_path, "repository")

    code_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"]["notebook_cell_type"] == "code"
    ]
    assert len(code_chunks) == 1
    assert "pd.read_csv('customers.csv')" in code_chunks[0]["content"]
    assert "output-only value" not in code_chunks[0]["content"]
    assert code_chunks[0]["metadata"]["cell_number"] == 1
    assert json.loads(code_chunks[0]["metadata"]["cell_metadata_json"]) == {
        "tags": ["important"]
    }
    assert any(
        chunk["metadata"]["notebook_cell_type"] == "markdown"
        for chunk in chunks
    )


def test_notebook_chunk_metadata_is_accepted_by_vector_store(tmp_path):
    notebook = write_notebook(
        tmp_path / "analysis.ipynb",
        [code_cell("import pandas as pd", {"tags": ["setup"]})],
    )
    chunk = chunk_file(notebook, tmp_path, "repository")[0]
    store = VectorStore(tmp_path / "chroma")

    assert store.add_chunks("repository", [chunk], [[0.1, 0.2]]) == 1
    result = store.search("repository", [0.1, 0.2], top_k=1)

    assert json.loads(result["metadatas"][0][0]["cell_metadata_json"]) == {
        "tags": ["setup"]
    }


def test_dataset_loading_question_retrieves_notebook_code_and_cell_reference(
    tmp_path,
):
    notebook = write_notebook(
        tmp_path / "analysis.ipynb",
        [
            {"cell_type": "markdown", "metadata": {}, "source": "Overview"},
            code_cell("df = pd.read_csv('ravenstack_subscriptions.csv')"),
        ],
    )
    repository_id = "repository"
    chunks = chunk_file(notebook, tmp_path, repository_id)
    keyword_store = KeywordStore(tmp_path / "keyword.sqlite3")
    keyword_store.add_chunks(repository_id, chunks)

    class FixedEmbedding:
        @staticmethod
        def embed_query(_query):
            return [0.1, 0.2]

    class EmptyVectorStore:
        @staticmethod
        def search(**_kwargs):
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

    class FakeLLM:
        @staticmethod
        def generate_answer(question, context):
            assert "pd.read_csv('ravenstack_subscriptions.csv')" in context
            return {
                "answer": "The notebook loads ravenstack_subscriptions.csv.",
                "model": "test",
            }

    class NoDependencyExpansion:
        @staticmethod
        def expand_search_results(**kwargs):
            return kwargs["search_results"]

    search_service = SearchService(
        FixedEmbedding(),
        EmptyVectorStore(),
        keyword_store,
    )
    rag_service = RAGService(
        search_service,
        FakeLLM(),
        NoDependencyExpansion(),
    )

    result = rag_service.answer_question(
        repository_id=repository_id,
        question="What dataset is loaded by read_csv ravenstack_subscriptions csv?",
        repository_path=tmp_path,
    )

    assert result["retrieved_chunks"][0]["content"].find(
        "pd.read_csv('ravenstack_subscriptions.csv')"
    ) >= 0
    assert result["sources"] == [{
        "file_path": "analysis.ipynb",
        "start_line": 2,
        "end_line": 2,
        "chunk_type": "notebook_cell",
        "name": "cell_2_part_1_code",
        "distance": 0.0,
        "cell_number": 2,
    }]


def test_notebook_technology_detection_uses_imports_and_file_format(tmp_path):
    write_notebook(
        tmp_path / "analysis.ipynb",
        [
            code_cell(
                "import pandas as pd\n"
                "from sklearn.model_selection import train_test_split"
            )
        ],
    )

    stack = TechnologyStackDetector(tmp_path).detect()

    assert stack["languages"] == ["Jupyter Notebook"]
    assert "Pandas" in stack["data_science"]
    assert "Scikit-learn" in stack["ai_ml"]


def test_notebook_dependency_analysis_includes_imports_and_cell_entities(tmp_path):
    (tmp_path / "helper.py").write_text(
        "def transform(frame):\n    return frame.dropna()\n",
        encoding="utf-8",
    )
    write_notebook(
        tmp_path / "analysis.ipynb",
        [
            code_cell("import pandas as pd\nfrom helper import transform"),
            code_cell(
                "def prepare(frame):\n"
                "    return transform(frame)\n"
            ),
        ],
    )

    result = DependencyAnalyzer("repository", tmp_path).analyze()

    assert result["total_python_files"] == 1
    assert result["total_notebooks"] == 1
    assert result["analyzed_files"] == 2
    assert any(
        dependency["from"] == "analysis.ipynb"
        and dependency["to"] == "helper.py"
        for dependency in result["internal_dependencies"]
    )
    assert any(
        entity["file"] == "analysis.ipynb"
        and entity["entity"] == "prepare"
        and entity["cell_number"] == 2
        for entity in result["entity_dependency_summary"]
    )
    assert "pandas" in result["external_dependencies_by_file"]["analysis.ipynb"]


def test_notebook_overview_counts_notebook_format(tmp_path):
    write_notebook(tmp_path / "analysis.ipynb", [code_cell("x = 1")])

    overview = RepositoryOverviewAnalyzer(tmp_path).analyze()

    assert overview["total_files"] == 1
    assert overview["file_types"] == {".ipynb": 1}
    assert overview["languages"] == {"Jupyter Notebook": 1}


def test_notebook_with_implementation_code_is_ranked_as_source(tmp_path):
    notebook = write_notebook(
        tmp_path / "analysis.ipynb",
        [code_cell("import pandas as pd\nmodel = train_model(pd.read_csv('data.csv'))")],
    )

    ranked_files = ImportantFileAnalyzer(tmp_path).analyze()
    notebook_result = next(
        item for item in ranked_files if item["file_path"] == notebook.name
    )

    assert notebook_result["importance_score"] >= 15
    assert "Contains project source code." in notebook_result["reasons"]


def test_malformed_notebook_is_skipped_without_failing_other_analysis(tmp_path):
    (tmp_path / "broken.ipynb").write_text("{not json", encoding="utf-8")
    (tmp_path / "usable.py").write_text("value = 1\n", encoding="utf-8")

    chunks = chunk_file(
        tmp_path / "broken.ipynb",
        tmp_path,
        "repository",
    )
    result = DependencyAnalyzer("repository", tmp_path).analyze()

    assert chunks == []
    assert result["total_python_files"] == 1
    assert result["total_notebooks"] == 1
    assert result["analyzed_files"] == 1
    assert result["failed_files"] == [{
        "path": "broken.ipynb",
        "error": "Notebook is not valid readable JSON or has no cells list.",
    }]


def test_dependency_cache_refreshes_when_repository_source_changes(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("def first():\n    return 1\n", encoding="utf-8")
    service = DependencyService()

    before = service.analyze_repository("repository", tmp_path)
    source.write_text(
        "def first():\n    return 1\n\ndef second():\n    return 2\n",
        encoding="utf-8",
    )
    after = service.analyze_repository("repository", tmp_path)

    assert before["total_entities"] == 1
    assert after["total_entities"] == 2
    assert before is not after
