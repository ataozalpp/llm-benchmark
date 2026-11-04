from dataclasses import asdict

import pytest

from llm_benchmark.rag.models import (
    RagAnswer,
    RagContext,
    RagGenerationRequest,
    RagScenario,
)
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery


def query():
    return RetrievalQuery(query_id="q1", text="what is the refund period?")


def test_answerable_scenario_requires_answers_and_evidence():
    with pytest.raises(ValueError):
        RagScenario(
            query=query(),
            reference_answers=(),
            relevant_chunk_ids=frozenset({"refund"}),
        )

    with pytest.raises(ValueError):
        RagScenario(
            query=query(),
            reference_answers=("30 days",),
            relevant_chunk_ids=frozenset(),
        )


def test_unanswerable_scenario_has_no_gold_answer():
    scenario = RagScenario(
        query=query(),
        reference_answers=(),
        relevant_chunk_ids=frozenset(),
        answerable=False,
    )
    assert scenario.answerable is False


def test_unanswerable_scenario_rejects_reference_answer():
    with pytest.raises(ValueError):
        RagScenario(
            query=query(),
            reference_answers=("30 days",),
            relevant_chunk_ids=frozenset(),
            answerable=False,
        )


def test_generation_request_does_not_contain_gold_fields():
    context = RagContext(
        query_id="q1",
        chunks=(
            Chunk(
                chunk_id="refund",
                document_id="policy",
                text="Refunds are available within 30 days.",
            ),
        ),
    )
    request = RagGenerationRequest(query=query(), context=context)

    assert set(asdict(request)) == {"query", "context"}


def test_request_rejects_mismatched_query_id():
    with pytest.raises(ValueError, match="IDs"):
        RagGenerationRequest(
            query=query(),
            context=RagContext(query_id="other", chunks=()),
        )


def test_valid_answer_and_abstention():
    answer = RagAnswer(answer="30 days", citations=("refund",))
    abstention = RagAnswer(answer="", citations=(), abstained=True)

    assert answer.abstained is False
    assert abstention.abstained is True


@pytest.mark.parametrize(
    "values",
    [
        {"answer": "", "citations": ()},
        {"answer": "30 days", "citations": ("a", "a")},
        {"answer": "30 days", "citations": ["a"]},
        {"answer": "30 days", "citations": (), "abstained": True},
        {"answer": "", "citations": ("a",), "abstained": True},
        {"answer": "30 days", "citations": (), "abstained": 1},
    ],
)
def test_invalid_answers_are_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        RagAnswer(**values)


@pytest.mark.parametrize("answers", ["30 days", ["30 days"], None, 30])
def test_scenario_rejects_non_tuple_reference_answers(answers):
    with pytest.raises(TypeError, match="Reference answers must be a tuple"):
        RagScenario(
            query=query(),
            reference_answers=answers,
            relevant_chunk_ids=frozenset({"refund"}),
        )
