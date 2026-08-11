import pytest

from llm_benchmark.config import ModelConfig
from llm_benchmark.models import DatasetExample
from llm_benchmark.parser import parse_multiple_choice
from llm_benchmark.prompting import build_prompt
from llm_benchmark.providers import MockProvider


@pytest.fixture
def example() -> DatasetExample:
    return DatasetExample("q", "Question?", ["one", "two", "three", "four"], "B", "test")


def test_prompt_builder(example: DatasetExample) -> None:
    prompt = build_prompt(example)
    assert "FINAL ANSWER: <OPTION_LETTER>" in prompt
    assert "B. two" in prompt


@pytest.mark.parametrize(("raw", "answer", "status"), [
    ("b", "B", "normalized_label"),
    ("(B).", "B", "normalized_label"),
    ("Answer: B. brief reason", "B", "explicit_final_marker"),
    ("Final answer: B", "B", "explicit_final_marker"),
    ("B. brief reason", "B", "leading_label"),
    ("No explicit choice", None, "no_answer_found"),
    ("A and C are possible", None, "ambiguous_multiple_answers"),
])
def test_parser(raw: str, answer: str | None, status: str) -> None:
    parsed = parse_multiple_choice(raw, ["A", "B", "C", "D"])
    assert parsed.parsed_answer == answer
    assert parsed.parse_status == status


@pytest.mark.parametrize(("scenario", "status", "raw_is_none"), [
    ("correct", "succeeded", False),
    ("incorrect", "succeeded", False),
    ("unparseable", "succeeded", False),
    ("request_failed", "failed", True),
])
def test_mock_provider_is_deterministic(example: DatasetExample, scenario: str, status: str, raw_is_none: bool) -> None:
    config = ModelConfig(model_id="mock", scenario_cycle=[scenario], mock_latency_ms=5)
    provider = MockProvider(config)
    first = provider.generate("prompt", example)
    second = provider.generate("prompt", example)
    assert first == second
    assert first.request_status == status
    assert (first.raw_response is None) is raw_is_none
    assert first.latency_ms == 5


def test_missing_usage_is_null(example: DatasetExample) -> None:
    response = MockProvider(ModelConfig(model_id="mock", scenario_cycle=["missing_usage"])).generate("prompt", example)
    assert response.prompt_tokens is None
    assert response.total_tokens is None
