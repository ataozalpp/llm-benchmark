import pytest

from llm_benchmark.models import DatasetExample, ProviderResponse
from llm_benchmark.prompting import build_prompt, prompt_hash
from llm_benchmark.task_adapters import MultipleChoiceTaskAdapter


@pytest.fixture
def example() -> DatasetExample:
    return DatasetExample(
        sample_id="q01",
        question="What is 2 + 2?",
        options=["3", "4", "5", "6"],
        correct_answer="B",
        category="math",
    )


def provider_response(
    *,
    raw_response: str | None,
    request_status: str = "succeeded",
) -> ProviderResponse:
    return ProviderResponse(
        request_status=request_status,
        raw_response=raw_response,
        prompt_tokens=10,
        completion_tokens=1,
        total_tokens=11,
        latency_ms=5.0,
    )


def test_multiple_choice_adapter_preserves_existing_prompt(
        example: DatasetExample,
) -> None:
    adapter = MultipleChoiceTaskAdapter()

    assert adapter.build_prompt(example) == build_prompt(example)


def test_multiple_choice_adapter_evaluates_correct_answer(
        example: DatasetExample,
) -> None:
    outcome = MultipleChoiceTaskAdapter().evaluate_response(
        example,
        provider_response(raw_response="B"),
    )

    assert outcome.parsed_answer == "B"
    assert outcome.parse_status == "normalized_label"
    assert outcome.evaluation_status == "correct"
    assert outcome.is_correct is True


def test_multiple_choice_adapter_evaluates_incorrect_answer(
    example: DatasetExample,
) -> None:
    outcome = MultipleChoiceTaskAdapter().evaluate_response(
        example,
        provider_response(raw_response="A"),
    )

    assert outcome.parsed_answer == "A"
    assert outcome.evaluation_status == "incorrect"
    assert outcome.is_correct is False


def test_multiple_choice_adapter_preserves_unparseable_status(
    example: DatasetExample,
) -> None:
    outcome = MultipleChoiceTaskAdapter().evaluate_response(
        example,
        provider_response(
            raw_response="Mars is the correct planet."
        ),
    )

    assert outcome.parsed_answer is None
    assert outcome.parse_status == "no_answer_found"
    assert outcome.evaluation_status == "unparseable"
    assert outcome.is_correct is False


def test_multiple_choice_adapter_preserves_request_failure(
    example: DatasetExample,
) -> None:
    outcome = MultipleChoiceTaskAdapter().evaluate_response(
        example,
        provider_response(
            raw_response=None,
            request_status="failed",
        ),
    )

    assert outcome.parsed_answer is None
    assert outcome.parse_status == "request_failed"
    assert outcome.evaluation_status == "request_failed"
    assert outcome.is_correct is False


def test_multiple_choice_adapter_preserves_ambiguous_response(
        example: DatasetExample,
) -> None:
    outcome = MultipleChoiceTaskAdapter().evaluate_response(
        example,
        provider_response(raw_response="A and C are possible"),
    )

    assert outcome.parsed_answer is None
    assert outcome.parse_status == "ambiguous_multiple_answers"
    assert outcome.candidate_answers == ("A", "C")
    assert outcome.evaluation_status == "unparseable"


def test_multiple_choice_adapter_uses_actual_allowed_labels() -> None:
    example = DatasetExample(
        "q10",
        "Question?",
        [str(index) for index in range(10)],
        "J",
        "test",
    )

    outcome = MultipleChoiceTaskAdapter().evaluate_response(
        example,
        provider_response(raw_response="J"),
    )

    assert outcome.parsed_answer == "J"
    assert outcome.evaluation_status == "correct"


def test_multiple_choice_adapter_preserves_existing_prompt_hash() -> None:
    adapter = MultipleChoiceTaskAdapter()

    assert adapter.prompt_template_hash() == prompt_hash()