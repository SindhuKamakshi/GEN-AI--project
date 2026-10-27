from rag.rag_service import RAGService


REPOSITORY_ID = "test-repository"
FALLBACK_ANSWER = (
    "I couldn't find evidence of that "
    "in the uploaded repository."
)


class FakeSearchService:
    def __init__(self, results):
        self.results = results
        self.requested_top_k = []
        self.queries = []
        self.repository_ids = []
        self.results_by_query = {}

    def search(self, repository_id, query, top_k):
        self.requested_top_k.append(top_k)
        self.queries.append(query)
        self.repository_ids.append(repository_id)
        return self.results_by_query.get(query, self.results)[:top_k]


class FakeLLMService:
    def __init__(self):
        self.context = None

    def generate_answer(self, question, context):
        self.context = context
        return {"answer": "test answer", "model": "test model"}


class FakeDependencyService:
    def expand_search_results(
        self,
        repository_id,
        search_results,
        repository_path=None,
    ):
        return search_results


def make_result(content, index):
    return {
        "chunk_id": f"{REPOSITORY_ID}_chunk_{index}",
        "content": content,
        "metadata": {
            "repository_id": REPOSITORY_ID,
            "file_path": f"source-{index}.py",
            "name": f"chunk-{index}",
            "start_line": index,
            "end_line": index,
            "chunk_type": "code",
        },
        "distance": 0.0,
    }


def make_service(results):
    search_service = FakeSearchService(results)
    llm_service = FakeLLMService()
    service = RAGService(
        search_service,
        llm_service,
        FakeDependencyService(),
    )
    return service, search_service, llm_service


def test_compound_question_includes_both_late_evidence_chunks_in_llm_context():
    question = "What is the first fact, and what is the second fact?"
    parts = RAGService._split_compound_question(question)
    service, search_service, llm_service = make_service([])
    shared_result = make_result("Shared evidence.", 999)
    first_results = [
        make_result(f"First-part distractor {index}.", index)
        for index in range(40)
    ]
    second_results = [
        make_result(f"Second-part distractor {index}.", index + 100)
        for index in range(40)
    ]
    first_results[0] = shared_result
    second_results[0] = {
        **shared_result,
        "metadata": {**shared_result["metadata"]},
    }
    first_results[14] = make_result("Evidence for the first part.", 14)
    second_results[14] = make_result("Evidence for the second part.", 114)
    search_service.results_by_query = {
        parts[0]: first_results,
        parts[1]: second_results,
    }

    response = service.answer_question(
        repository_id=REPOSITORY_ID,
        question=question,
        top_k=5,
    )

    assert search_service.requested_top_k == [40, 40]
    assert search_service.queries == parts
    assert search_service.repository_ids == [
        REPOSITORY_ID,
        REPOSITORY_ID,
    ]
    assert "Evidence for the first part." in llm_service.context
    assert "Evidence for the second part." in llm_service.context
    assert len(response["retrieved_chunks"]) == 40
    chunk_ids = [
        result["chunk_id"]
        for result in response["retrieved_chunks"]
    ]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_workflow_question_uses_bounded_contextual_subqueries():
    question = "Explain how the project prepares its input features before training."
    queries = RAGService._workflow_search_queries(question)

    assert queries == [
        question,
        "input features preprocessing",
        "input features feature engineering",
        "input features target variable",
    ]

    service, search_service, llm_service = make_service([])
    search_service.results_by_query = {
        question: [make_result("Project overview distractor.", 1)],
        "input features preprocessing": [
            make_result("df = df.drop(columns=['unused'])", 2),
        ],
        "input features feature engineering": [
            make_result("engagement_score = seats * 0.2 + mrr_amount * 0.3", 3),
        ],
        "input features target variable": [
            make_result("X = df.drop('churn_flag', axis=1)\ny = df['churn_flag']", 4),
        ],
    }

    response = service.answer_question(
        repository_id=REPOSITORY_ID,
        question=question,
        top_k=5,
    )

    assert search_service.requested_top_k == [40, 40, 40, 40]
    assert search_service.queries == queries
    assert "df = df.drop(columns=['unused'])" in llm_service.context
    assert "engagement_score = seats * 0.2 + mrr_amount * 0.3" in llm_service.context
    assert "X = df.drop('churn_flag', axis=1)" in llm_service.context
    assert len(response["retrieved_chunks"]) == 4


def test_single_part_question_preserves_requested_top_k():
    service, search_service, _ = make_service(
        [make_result("Repository evidence.", 0)]
    )

    service.answer_question(
        repository_id=REPOSITORY_ID,
        question="What evidence is available?",
        top_k=5,
    )

    assert search_service.requested_top_k == [5]
    assert len(search_service.queries) == 1
    assert search_service.queries == ["What evidence is available?"]
    assert search_service.repository_ids == [REPOSITORY_ID]


def test_compound_question_preserves_requested_top_k_greater_than_floor():
    service, search_service, _ = make_service([])

    service.answer_question(
        repository_id=REPOSITORY_ID,
        question="What is one fact, and what is another fact?",
        top_k=45,
    )

    assert search_service.requested_top_k == [45, 45]


def test_empty_search_results_keep_existing_fallback():
    service, search_service, llm_service = make_service([])

    response = service.answer_question(
        repository_id=REPOSITORY_ID,
        question="What evidence is available?",
        top_k=5,
    )

    assert search_service.requested_top_k == [5]
    assert response["answer"] == FALLBACK_ANSWER
    assert response["sources"] == []
    assert response["retrieved_chunks"] == []
    assert response["model"] == "Not used"
    assert llm_service.context is None


def test_splitter_preserves_question_punctuation_and_context():
    cases = {
        (
            "How is the target variable separated from the input features, "
            "and which variable stores the target?"
        ): [
            "How is the target variable separated from the input features?",
            (
                "Which variable stores the target? Context: "
                "How is the target variable separated from the input features?"
            ),
        ],
        (
            "Which dataset does the notebook load, "
            "and which function reads it?"
        ): [
            "Which dataset does the notebook load?",
            (
                "Which function reads it? Context: "
                "Which dataset does the notebook load?"
            ),
        ],
    }

    for question, expected in cases.items():
        assert RAGService._split_compound_question(question) == expected


def test_splitter_keeps_ordinary_conjunctions_as_single_queries():
    for question in (
        "Which libraries are imported and used in the notebook?",
        "How does the loader read and transform data?",
    ):
        assert RAGService._split_compound_question(question) == [question]


def test_splitter_splits_four_parts_and_falls_back_above_limit():
    four_part_question = (
        "What is A, and what is B, and what is C, and what is D?"
    )
    four_parts = RAGService._split_compound_question(four_part_question)

    assert len(four_parts) == RAGService.MAX_COMPOUND_QUESTION_PARTS == 4
    service, search_service, _ = make_service([])
    service.answer_question(
        repository_id=REPOSITORY_ID,
        question=four_part_question,
        top_k=5,
    )
    assert search_service.queries == four_parts
    assert search_service.repository_ids == [REPOSITORY_ID] * 4

    five_part_question = (
        "What is A, and what is B, and what is C, "
        "and what is D, and what is E?"
    )
    assert RAGService._split_compound_question(five_part_question) == [
        five_part_question
    ]

    service, search_service, _ = make_service([])
    service.answer_question(
        repository_id=REPOSITORY_ID,
        question=five_part_question,
        top_k=5,
    )
    assert search_service.queries == [five_part_question]
    assert search_service.repository_ids == [REPOSITORY_ID]


def test_merge_limit_deduplicates_and_keeps_distinct_results_within_limit():
    shared = make_result("shared evidence", 1)
    duplicate = {**shared, "metadata": {**shared["metadata"]}}
    first_distinct = make_result("first distinct evidence", 2)
    second_distinct = make_result("second distinct evidence", 3)

    merged = RAGService._merge_compound_search_results(
        [
            [shared, first_distinct],
            [duplicate, second_distinct],
        ],
        REPOSITORY_ID,
        limit=2,
    )

    assert len(merged) == 2
    assert [result["content"] for result in merged] == [
        "shared evidence",
        "first distinct evidence",
    ]


def test_merge_keeps_distinct_notebook_cells_and_incomplete_identities():
    first = {
        "content": "first cell",
        "metadata": {
            "repository_id": REPOSITORY_ID,
            "file_path": "analysis.ipynb",
            "chunk_type": "notebook_cell",
            "cell_number": 1,
            "cell_part": 1,
            "notebook_cell_type": "code",
        },
    }
    second = {
        **first,
        "content": "second cell",
        "metadata": {**first["metadata"], "cell_number": 2},
    }
    incomplete = {
        "content": "incomplete identity",
        "metadata": {
            "repository_id": REPOSITORY_ID,
            "file_path": "analysis.ipynb",
            "chunk_type": "notebook_cell",
        },
    }

    merged = RAGService._merge_compound_search_results(
        [[first, incomplete], [second, {**incomplete}]],
        REPOSITORY_ID,
        10,
    )

    assert [result["content"] for result in merged] == [
        "first cell",
        "second cell",
        "incomplete identity",
        "incomplete identity",
    ]
    assert RAGService._merge_compound_search_results(
        [[first]],
        REPOSITORY_ID,
        0,
    ) == []


def test_merge_rejects_conflicting_repository_metadata():
    foreign = {
        "chunk_id": "other-repository_chunk_1",
        "content": "foreign evidence",
        "metadata": {
            "repository_id": "other-repository",
            "file_path": "foreign.py",
            "chunk_type": "function",
            "name": "foreign_function",
            "start_line": 1,
            "end_line": 2,
        },
    }

    assert RAGService._merge_compound_search_results(
        [[foreign]],
        REPOSITORY_ID,
        10,
    ) == []


def test_compound_subqueries_contribute_evidence_from_each_part():
    cases = (
        (
            "How is the target variable separated from the input features, "
            "and which variable stores the target?",
            "X = frame.drop('target', axis=1)",
            "y = frame['target']",
        ),
        (
            "Which dataset does the notebook load, "
            "and which function reads it?",
            "dataset.csv",
            "read_csv",
        ),
        (
            "What is the main objective of this notebook, "
            "and what is the target variable used for prediction?",
            "objective evidence",
            "target evidence",
        ),
    )

    for question, first_evidence, second_evidence in cases:
        parts = RAGService._split_compound_question(question)
        service, search_service, llm_service = make_service([])
        search_service.results_by_query = {
            parts[0]: [make_result(first_evidence, 1)],
            parts[1]: [make_result(second_evidence, 2)],
        }

        response = service.answer_question(
            repository_id=REPOSITORY_ID,
            question=question,
            top_k=5,
        )

        assert search_service.queries == parts
        assert search_service.repository_ids == [
            REPOSITORY_ID,
            REPOSITORY_ID,
        ]
        assert search_service.requested_top_k == [40, 40]
        assert first_evidence in llm_service.context
        assert second_evidence in llm_service.context
        assert len(response["retrieved_chunks"]) == 2
