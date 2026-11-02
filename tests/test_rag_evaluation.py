import pytest

from llm_benchmark.rag.evaluation import evaluate_rag_answer
from llm_benchmark.rag.models import RagAnswer, RagContext, RagScenario
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery


def make_scenario(
    *,
    references=("30 days", "thirty days"),
    gold=frozenset({"refund"}),
    answerable=True,
):
    return RagScenario(
        query=RetrievalQuery(query_id="q1", text="What is the refund period?"),
        reference_answers=references,
        relevant_chunk_ids=gold,
        answerable=answerable,
    )


def make_context(*chunk_ids):
    return RagContext(
        query_id="q1",
        chunks=tuple(
            Chunk(
                chunk_id=chunk_id,
                document_id=f"doc-{chunk_id}",
                text=f"Evidence for {chunk_id}.",
            )
            for chunk_id in chunk_ids
        ),
    )


def test_correct_answer_with_valid_evidence():
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context("refund"),
        answer=RagAnswer(answer="30 days", citations=("refund",)),
    )

    assert result.answer_exact_match is True
    assert result.abstention_correct is True
    assert result.context_evidence_recall == 1.0
    assert result.citation_validity == 1.0
    assert result.citation_evidence_recall == 1.0
    assert result.citation_count == 1


@pytest.mark.parametrize("text", ["  THIRTY   DAYS ", "３０ days"])
def test_reference_aliases_and_normalization(text):
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context("refund"),
        answer=RagAnswer(answer=text, citations=()),
    )

    assert result.answer_exact_match is True


@pytest.mark.parametrize(
    ("reference", "generated"),
    [
        ("1.0", "10"),
        ("30 days", "The answer might be 30 days or 60 days."),
    ],
)
def test_normalization_does_not_create_false_matches(reference, generated):
    result = evaluate_rag_answer(
        scenario=make_scenario(references=(reference,)),
        context=make_context("refund"),
        answer=RagAnswer(answer=generated, citations=()),
    )

    assert result.answer_exact_match is False


def test_unknown_citation_reduces_validity():
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context("refund"),
        answer=RagAnswer(
            answer="30 days",
            citations=("refund", "invented"),
        ),
    )

    assert result.citation_validity == 0.5
    assert result.citation_evidence_recall == 1.0


def test_gold_id_outside_context_does_not_count_as_valid_evidence():
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context("other"),
        answer=RagAnswer(answer="30 days", citations=("refund",)),
    )

    assert result.answer_exact_match is True
    assert result.context_evidence_recall == 0.0
    assert result.citation_validity == 0.0
    assert result.citation_evidence_recall == 0.0


def test_missing_citations_are_not_perfect_validity():
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context("refund"),
        answer=RagAnswer(answer="30 days", citations=()),
    )

    assert result.citation_count == 0
    assert result.citation_validity is None
    assert result.citation_evidence_recall == 0.0


def test_partial_evidence_recall():
    result = evaluate_rag_answer(
        scenario=make_scenario(gold=frozenset({"a", "b"})),
        context=make_context("a"),
        answer=RagAnswer(answer="30 days", citations=("a",)),
    )

    assert result.context_evidence_recall == 0.5
    assert result.citation_evidence_recall == 0.5


def test_valid_but_irrelevant_citation():
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context("refund", "other"),
        answer=RagAnswer(answer="30 days", citations=("other",)),
    )

    assert result.citation_validity == 1.0
    assert result.citation_evidence_recall == 0.0


def test_abstention_on_answerable_scenario():
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context(),
        answer=RagAnswer(answer="", citations=(), abstained=True),
    )

    assert result.answer_exact_match is False
    assert result.abstention_correct is False


@pytest.mark.parametrize("abstained", [True, False])
def test_unanswerable_scenario(abstained):
    result = evaluate_rag_answer(
        scenario=make_scenario(
            references=(),
            gold=frozenset(),
            answerable=False,
        ),
        context=make_context(),
        answer=RagAnswer(
            answer="" if abstained else "An unsupported answer",
            citations=(),
            abstained=abstained,
        ),
    )

    assert result.abstention_correct is abstained
    assert result.answer_exact_match is None
    assert result.context_evidence_recall is None
    assert result.citation_evidence_recall is None


def test_mismatched_query_ids_are_rejected():
    with pytest.raises(ValueError, match="IDs"):
        evaluate_rag_answer(
            scenario=make_scenario(),
            context=RagContext(query_id="other", chunks=()),
            answer=RagAnswer(answer="30 days", citations=()),
        )


@pytest.mark.parametrize("field", ["scenario", "context", "answer"])
def test_invalid_input_types_are_rejected(field):
    values = {
        "scenario": make_scenario(),
        "context": make_context("refund"),
        "answer": RagAnswer(answer="30 days", citations=("refund",)),
    }
    values[field] = None

    with pytest.raises(TypeError, match="Expected"):
        evaluate_rag_answer(**values)


def test_citation_ids_are_not_answer_normalized():
    result = evaluate_rag_answer(
        scenario=make_scenario(),
        context=make_context("refund"),
        answer=RagAnswer(answer="30 days", citations=("REFUND",)),
    )

    assert result.answer_exact_match is True
    assert result.citation_validity == 0.0
    assert result.citation_evidence_recall == 0.0


def test_context_recall_and_citation_recall_are_independent():
    result = evaluate_rag_answer(
        scenario=make_scenario(gold=frozenset({"a", "b"})),
        context=make_context("a", "b"),
        answer=RagAnswer(answer="30 days", citations=("a",)),
    )

    assert result.context_evidence_recall == 1.0
    assert result.citation_validity == 1.0
    assert result.citation_evidence_recall == 0.5
