import pytest

from llm_benchmark.config import ModelConfig
from llm_benchmark.models import DatasetExample
from llm_benchmark.parser import parse_multiple_choice
from llm_benchmark.prompting import build_prompt
from llm_benchmark.providers import LMStudioProvider, MockProvider, create_provider


@pytest.fixture
def example() -> DatasetExample:
    return DatasetExample("q", "Question?", ["one", "two", "three", "four"], "B", "test")


def test_prompt_builder(example: DatasetExample) -> None:
    prompt = build_prompt(example)
    lines = prompt.splitlines()
    assert lines[1] == "Respond with exactly one uppercase option letter from these available choices: A, B, C, D"
    assert lines[2] == "Do not include words, punctuation, explanation, or any other text."
    assert lines[3] == "Example valid response:"
    assert lines[4] == "B"
    assert "B. two" in prompt


def test_prompt_uses_actual_available_labels() -> None:
    example = DatasetExample("q", "Question?", [str(i) for i in range(10)], "J", "test")
    prompt = build_prompt(example)
    assert "available choices: A, B, C, D, E, F, G, H, I, J" in prompt


@pytest.mark.parametrize(("raw", "answer", "status"), [
    ("b", "B", "normalized_label"),
    ("B", "B", "normalized_label"),
    ("(B).", "B", "normalized_label"),
    ("Answer: B. brief reason", "B", "explicit_final_marker"),
    ("Final answer: B", "B", "explicit_final_marker"),
    ("FINAL ANSWER: B", "B", "explicit_final_marker"),
    ("B. brief reason", "B", "leading_label"),
    ("Mars is commonly called the Red Planet.", None, "no_answer_found"),
    ("No explicit choice", None, "no_answer_found"),
    ("A and C are possible", None, "ambiguous_multiple_answers"),
    ("FINAL ANSW: B", None, "no_answer_found"),
])
def test_parser(raw: str, answer: str | None, status: str) -> None:
    parsed = parse_multiple_choice(raw, ["A", "B", "C", "D"])
    assert parsed.parsed_answer == answer
    assert parsed.parse_status == status


def test_parser_accepts_j_only_when_allowed() -> None:
    assert parse_multiple_choice("J", list("ABCDEFGHIJ")).parsed_answer == "J"
    rejected = parse_multiple_choice("J", list("ABCD"))
    assert rejected.parsed_answer is None
    assert rejected.parse_status == "no_answer_found"


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


class FakeTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response or {}
        self.error = error
        self.calls: list[tuple[str, dict, float]] = []

    def post_json(self, url: str, payload: dict, timeout_seconds: float) -> dict:
        self.calls.append((url, payload, timeout_seconds))
        if self.error:
            raise self.error
        return self.response


def lm_studio_config() -> ModelConfig:
    return ModelConfig(
        provider="lm_studio",
        endpoint_alias="lm-studio-localhost",
        base_url="http://127.0.0.1:1234",
        model_id="qwen3.5-0.8b",
        reasoning="off",
        temperature=0,
        max_output_tokens=64,
        timeout_seconds=120,
    )


def test_lm_studio_native_response_uses_messages_and_ignores_reasoning(example: DatasetExample) -> None:
    transport = FakeTransport({
        "model": "qwen3.5-0.8b",
        "output": [
            {"type": "reasoning", "content": "FINAL ANSWER: A"},
            {"type": "message", "content": [{"type": "output_text", "text": "FINAL ANSWER: B"}]},
        ],
        "usage": {"input_tokens": 21, "total_output_tokens": 9, "reasoning_output_tokens": 5, "total_tokens": 30},
        "stats": {"tokens_per_second": 12.5, "time_to_first_token_seconds": 0.25},
        "stop_reason": "completed",
    })
    provider = LMStudioProvider(lm_studio_config(), transport=transport)
    response = provider.generate("prompt", example)
    url, payload, timeout = transport.calls[0]
    assert url == "http://127.0.0.1:1234/api/v1/chat"
    assert payload == {
        "model": "qwen3.5-0.8b",
        "input": "prompt",
        "reasoning": "off",
        "temperature": 0.0,
        "max_output_tokens": 64,
        "store": False,
    }
    assert timeout == 120
    assert response.request_status == "succeeded"
    assert response.raw_response == "FINAL ANSWER: B"
    assert response.input_tokens == 21
    assert response.total_output_tokens == 9
    assert response.reasoning_output_tokens == 5
    assert response.total_tokens == 30
    assert response.tokens_per_second == 12.5
    assert response.time_to_first_token_ms == 250
    assert response.stop_reason == "completed"


def test_lm_studio_ignores_reasoning_when_message_is_missing(example: DatasetExample) -> None:
    transport = FakeTransport({"output": [{"type": "reasoning", "content": "FINAL ANSWER: B"}]})
    response = LMStudioProvider(lm_studio_config(), transport=transport).generate("prompt", example)
    assert response.request_status == "succeeded"
    assert response.raw_response == ""
    assert response.stop_reason is None


def test_lm_studio_network_failure_is_normalized(example: DatasetExample) -> None:
    transport = FakeTransport(error=ConnectionError("offline"))
    response = LMStudioProvider(lm_studio_config(), transport=transport).generate("prompt", example)
    assert response.request_status == "failed"
    assert response.error_type == "network_error"
    assert response.raw_response is None


def test_provider_factory_preserves_mock_and_isolates_lm_studio() -> None:
    assert isinstance(create_provider(ModelConfig(model_id="mock")), MockProvider)
    assert isinstance(create_provider(lm_studio_config(), transport=FakeTransport()), LMStudioProvider)
