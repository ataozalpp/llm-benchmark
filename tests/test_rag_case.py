from dataclasses import asdict

import pytest

from llm_benchmark.rag.case import run_rag_case
from llm_benchmark.rag.models import RagContext, RagScenario
from llm_benchmark.rag.providers import (
    RagGenerationResponse,
    RagProviderError,
)
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery

VALID_ANSWER = '{"answer":"30 days","citations":["refund"],"abstained":false}'


class StubProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def generate(self, request):
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return self.response


def make_scenario(answerable=True):
    return RagScenario(
        query=RetrievalQuery(query_id="q1", text="What is the refund period?"),
        reference_answers=("30 days",) if answerable else (),
        relevant_chunk_ids=(frozenset({"refund"}) if answerable else frozenset()),
        answerable=answerable,
    )


def make_context():
    return RagContext(
        query_id="q1",
        chunks=(
            Chunk(
                chunk_id="refund",
                document_id="policy",
                text="Refunds are available within 30 days.",
            ),
        ),
    )


def test_successful_case_passes_only_generation_data():
    provider = StubProvider(
        response=RagGenerationResponse(
            text=VALID_ANSWER,
            finish_reason="stop",
        )
    )

    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.status == "completed"
    assert result.evaluation.answer_exact_match is True
    assert result.evaluation.citation_validity == 1.0
    assert len(provider.requests) == 1
    assert set(asdict(provider.requests[0])) == {"query", "context"}


def test_valid_wrong_answer_is_still_evaluated():
    provider = StubProvider(
        response=RagGenerationResponse(
            text=VALID_ANSWER.replace("30 days", "60 days"),
            finish_reason="stop",
        )
    )

    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.status == "completed"
    assert result.evaluation.answer_exact_match is False


def test_parse_failure_is_not_an_abstention():
    provider = StubProvider(
        response=RagGenerationResponse(
            text="I dont know.",
            finish_reason="stop",
        )
    )

    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.status == "invalid_response"
    assert result.response.text == "I dont know."
    assert result.answer is None
    assert result.evaluation is None


def test_valid_abstention_is_evaluated():
    provider = StubProvider(
        response=RagGenerationResponse(
            text='{"answer":"","citations":[],"abstained":true}',
            finish_reason="stop",
        )
    )

    result = run_rag_case(
        scenario=make_scenario(answerable=False),
        context=RagContext(query_id="q1", chunks=()),
        provider=provider,
    )

    assert result.status == "completed"
    assert result.answer.abstained is True
    assert result.evaluation.abstention_correct is True


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls"])
def test_non_stop_finish_is_not_evaluated(reason):
    provider = StubProvider(
        response=RagGenerationResponse(
            text=VALID_ANSWER,
            finish_reason=reason,
        )
    )

    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.status == "incomplete_response"
    assert result.response is not None
    assert result.evaluation is None


def test_expected_provider_failure_is_recorded():
    provider = StubProvider(error=RagProviderError("Unavailable"))

    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.status == "provider_failed"
    assert result.response is None
    assert result.evaluation is None


def test_programming_errors_are_not_hidden():
    provider = StubProvider(error=RuntimeError("Implementation bug"))

    with pytest.raises(RuntimeError, match="Implementation bug"):
        run_rag_case(
            scenario=make_scenario(),
            context=make_context(),
            provider=provider,
        )


def test_wrong_provider_return_type_is_rejected():
    provider = StubProvider(response="plain string")

    with pytest.raises(TypeError, match="RagGenerationResponse"):
        run_rag_case(
            scenario=make_scenario(),
            context=make_context(),
            provider=provider,
        )


def test_mismatched_context_is_rejected_before_generation():
    provider = StubProvider()

    with pytest.raises(ValueError, match="IDs"):
        run_rag_case(
            scenario=make_scenario(),
            context=RagContext(query_id="other", chunks=()),
            provider=provider,
        )

    assert provider.requests == []


@pytest.mark.parametrize(
    "text,code",
    [
        ('{"answer": "No", "citations": [], "abstained": true}', "abstention_with_content"),
        ("not json", "invalid_json"),
        ('{"answer": "30 days"}', "wrong_fields"),
    ],
)
def test_invalid_response_records_parse_failure_code(text, code):
    provider = StubProvider(
        response=RagGenerationResponse(text=text, finish_reason="stop")
    )
    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.status == "invalid_response"
    assert result.failure_code == code
    assert result.response.text == text
    assert result.answer is None
    assert result.evaluation is None


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_non_parse_outcomes_have_no_failure_code(reason):
    provider = StubProvider(
        response=RagGenerationResponse(text=VALID_ANSWER, finish_reason=reason)
    )
    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.failure_code is None


def test_provider_failure_has_no_parse_failure_code():
    provider = StubProvider(error=RagProviderError("Unavailable"))
    result = run_rag_case(
        scenario=make_scenario(),
        context=make_context(),
        provider=provider,
    )

    assert result.status == "provider_failed"
    assert result.failure_code is None
