import json
from dataclasses import asdict

import pytest

from llm_benchmark.rag.models import RagScenario
from llm_benchmark.rag.providers import RagGenerationResponse, RagProviderError
from llm_benchmark.rag.suite import run_rag_suite
from llm_benchmark.retrieval.models import Chunk, RankedRetrieval, RetrievalQuery


def scenario(query_id="q1", *, gold=frozenset({"refund"}), answerable=True):
    return RagScenario(
        query=RetrievalQuery(query_id=query_id, text="Refund period?"),
        reference_answers=("30 days",) if answerable else (),
        relevant_chunk_ids=gold,
        answerable=answerable,
    )


def response(text=None, reason="stop"):
    return RagGenerationResponse(
        text=text
        if text is not None
        else json.dumps(
            {
                "answer": "30 days",
                "citations": ["refund"],
                "abstained": False,
            }
        ),
        finish_reason=reason,
        input_tokens=20,
        output_tokens=10,
    )


class Retriever:
    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.calls = []

    def retrieve(self, query, *, top_k):
        self.calls.append((query.query_id, top_k))
        output = self.outputs.get(
            query.query_id, RankedRetrieval(query.query_id, ("refund",))
        )
        if isinstance(output, Exception):
            raise output
        return output


class Provider:
    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        output = self.outputs.get(request.query.query_id, response())
        if isinstance(output, Exception):
            raise output
        return output


@pytest.fixture
def inputs():
    return {
        "chunks": (
            Chunk(
                chunk_id="refund", document_id="policy", text="Refunds within 30 days."
            ),
            Chunk(chunk_id="other", document_id="other", text="Shipping takes 5 days."),
        ),
        "scenarios": (scenario(),),
        "retriever": Retriever(),
        "provider": Provider(),
        "k": 1,
        "max_context_chars": 1000,
    }


def test_success_preserves_ranking_context_response_usage_and_evaluation(inputs):
    result = run_rag_suite(**inputs)
    case = result.cases[0]
    assert inputs["retriever"].calls == [("q1", 1)]
    request = inputs["provider"].requests[0]
    assert set(asdict(request)) == {"query", "context"}
    assert request.context == case.context
    assert case.ranking.chunk_ids == case.context.chunk_ids == ("refund",)
    assert case.result.evaluation.answer_exact_match is True
    assert case.result.response.input_tokens == 20
    assert case.result.response.text == response().text


@pytest.mark.parametrize(
    "changes",
    [
        {"k": True},
        {"k": 0},
        {"k": 1.5},
        {"max_context_chars": True},
        {"max_context_chars": 1},
        {"chunks": []},
        {"chunks": ("bad",)},
        {"scenarios": []},
        {"scenarios": ()},
        {"scenarios": ("bad",)},
        {"scenarios": (scenario(), scenario())},
        {"scenarios": (scenario(), scenario("q2", gold=frozenset({"missing"})))},
        {"on_case_completed": 1},
    ],
)
def test_invalid_configuration_fails_before_any_work(inputs, changes):
    inputs.update(changes)
    with pytest.raises((ValueError, TypeError)):
        run_rag_suite(**inputs)
    assert inputs["retriever"].calls == []
    assert inputs["provider"].requests == []


def test_duplicate_corpus_ids_fail_before_work(inputs):
    inputs["chunks"] = (inputs["chunks"][0],) * 2
    with pytest.raises(ValueError, match="unique"):
        run_rag_suite(**inputs)
    assert inputs["retriever"].calls == []


@pytest.mark.parametrize(
    "ranking",
    [
        "bad",
        RankedRetrieval("wrong", ("refund",)),
        RankedRetrieval("q1", ("refund", "other")),
        RankedRetrieval("q1", ("missing",)),
    ],
)
def test_invalid_ranking_fails_before_generation(inputs, ranking):
    inputs["retriever"] = Retriever({"q1": ranking})
    with pytest.raises((ValueError, TypeError)):
        run_rag_suite(**inputs)
    assert inputs["provider"].requests == []


@pytest.mark.parametrize("ids", [(), ("other",)])
def test_missing_gold_in_ranking_is_evaluated_not_rejected(inputs, ids):
    inputs["retriever"] = Retriever({"q1": RankedRetrieval("q1", ids)})
    case = run_rag_suite(**inputs).cases[0]
    assert case.context.chunk_ids == ids
    assert case.result.evaluation.context_evidence_recall == 0.0
    assert case.result.evaluation.citation_validity == 0.0


def test_context_budget_preserves_ranking_but_sends_empty_context(inputs):
    inputs["max_context_chars"] = 2
    case = run_rag_suite(**inputs).cases[0]
    assert case.ranking.chunk_ids == ("refund",)
    assert case.context.render() == "[]"
    assert inputs["provider"].requests[0].context.chunks == ()


@pytest.mark.parametrize(
    ("output", "status"),
    [
        (RagProviderError("Unavailable"), "provider_failed"),
        (response("not json"), "invalid_response"),
        (response(reason="length"), "incomplete_response"),
    ],
)
def test_expected_failure_continues_in_order_and_emits_callback(inputs, output, status):
    inputs["scenarios"] = (scenario(), scenario("q2"))
    inputs["provider"] = Provider({"q1": output})
    emitted = []
    result = run_rag_suite(**inputs, on_case_completed=emitted.append)
    assert tuple(emitted) == result.cases
    assert [case.result.query_id for case in emitted] == ["q1", "q2"]
    assert [case.result.status for case in emitted] == [status, "completed"]


@pytest.mark.parametrize("stage", ["retriever", "provider"])
def test_prior_case_can_be_persisted_before_later_unexpected_failure(
    inputs, tmp_path, stage
):
    inputs["scenarios"] = (scenario(), scenario("q2"))
    cls = Retriever if stage == "retriever" else Provider
    inputs[stage] = cls({"q2": RuntimeError("Unexpected bug")})
    path = tmp_path / "results.jsonl"

    def persist(case):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(case), allow_nan=False) + "\n")

    with pytest.raises(RuntimeError, match="Unexpected bug"):
        run_rag_suite(**inputs, on_case_completed=persist)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["result"]["query_id"] == "q1"
    assert rows[0]["result"]["response"]["text"] == response().text


def test_callback_failure_stops_before_next_paid_call(inputs):
    inputs["scenarios"] = (scenario(), scenario("q2"))

    def fail(case):
        raise OSError("Disk full")

    with pytest.raises(OSError, match="Disk full"):
        run_rag_suite(**inputs, on_case_completed=fail)
    assert len(inputs["provider"].requests) == 1
    assert inputs["retriever"].calls == [("q1", 1)]


def test_unanswerable_case_with_related_but_insufficient_context(inputs):
    inputs["scenarios"] = (
        RagScenario(
            query=RetrievalQuery(
                query_id="q1", text="Are damaged items eligible for refunds?"
            ),
            reference_answers=(),
            relevant_chunk_ids=frozenset(),
            answerable=False,
        ),
    )
    inputs["provider"] = Provider(
        {"q1": response('{"answer":"","citations":[],"abstained":true}')}
    )
    case = run_rag_suite(**inputs).cases[0]
    assert case.context.chunk_ids == ("refund",)
    assert case.result.evaluation.abstention_correct is True
    assert case.result.evaluation.answer_exact_match is None
