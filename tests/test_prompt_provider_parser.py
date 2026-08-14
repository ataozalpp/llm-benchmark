import io
import json
import urllib.error
from pathlib import Path

import pytest

from llm_benchmark.config import ModelConfig, load_config
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
    assert response.final_output_tokens == 4
    assert response.reasoning_observed is True
    assert response.total_tokens == 30
    assert response.tokens_per_second == 12.5
    assert response.time_to_first_token_ms == 250
    assert response.stop_reason == "completed"


def test_lm_studio_omits_unset_sampling_fields(example: DatasetExample) -> None:
    transport = FakeTransport({"output": [{"type": "message", "content": "B"}]})
    LMStudioProvider(lm_studio_config(), transport=transport).generate("prompt", example)
    payload = transport.calls[0][1]
    assert not {"top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty"} & payload.keys()


def test_lm_studio_omits_max_output_tokens_when_not_configured(example: DatasetExample) -> None:
    config = ModelConfig(
        provider="lm_studio",
        base_url="http://127.0.0.1:1234",
        model_id="qwen3.5-0.8b",
        reasoning="on",
    )
    transport = FakeTransport({"output": [{"type": "message", "content": "B"}]})
    LMStudioProvider(config, transport=transport).generate("prompt", example)
    assert "max_output_tokens" not in transport.calls[0][1]


def test_lm_studio_omits_explicit_null_max_output_tokens(example: DatasetExample) -> None:
    config = ModelConfig(
        provider="lm_studio",
        base_url="http://127.0.0.1:1234",
        model_id="qwen3.5-0.8b",
        reasoning="on",
        max_output_tokens=None,
    )
    transport = FakeTransport({"output": [{"type": "message", "content": "B"}]})
    LMStudioProvider(config, transport=transport).generate("prompt", example)
    assert "max_output_tokens" not in transport.calls[0][1]


def test_mock_provider_is_unaffected_by_optional_output_budget(example: DatasetExample) -> None:
    without_limit = MockProvider(ModelConfig(model_id="mock", scenario_cycle=["correct"]))
    with_limit = MockProvider(ModelConfig(model_id="mock", scenario_cycle=["correct"], max_output_tokens=64))
    assert without_limit.generate("prompt", example) == with_limit.generate("prompt", example)


def test_context_bounded_mini_omits_output_limit_for_all_three_payloads() -> None:
    config = load_config(Path("configs/mmlu_pro_lm_studio_reasoning_on_context_bounded_mini.yaml"))
    model = config.models[0]
    transport = FakeTransport({"output": [{"type": "message", "content": "A"}]})
    provider = LMStudioProvider(model, transport=transport)
    for sample_id in config.dataset.sample_ids:
        provider.generate("prompt", DatasetExample(sample_id, "Question?", ["one", "two"], "A", "test"))

    assert len(transport.calls) == 3
    for _, payload, timeout in transport.calls:
        assert "max_output_tokens" not in payload
        assert "presence_penalty" not in payload
        assert payload["reasoning"] == "on"
        assert payload["store"] is False
        assert timeout == 660


def test_lm_studio_includes_supported_sampling_fields_when_set(example: DatasetExample) -> None:
    config = lm_studio_config().model_copy(update={
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    })
    transport = FakeTransport({"output": [{"type": "message", "content": "B"}]})
    LMStudioProvider(config, transport=transport).generate("prompt", example)
    payload = transport.calls[0][1]
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["repeat_penalty"] == 1.0
    assert "presence_penalty" not in payload


def test_lm_studio_reasoning_on_is_sent_and_only_message_is_scored(example: DatasetExample) -> None:
    config = lm_studio_config().model_copy(update={"reasoning": "on", "max_output_tokens": 1024})
    transport = FakeTransport({
        "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Option A is wrong."}]},
            {"type": "message", "content": "B"},
        ],
        "usage": {"input_tokens": 20, "total_output_tokens": 33, "reasoning_output_tokens": 32},
        "stats": {"tokens_per_second": 10.0, "time_to_first_token_ms": 100},
        "stop_reason": "completed",
    })
    response = LMStudioProvider(config, transport=transport).generate("prompt", example)
    _, payload, timeout = transport.calls[0]
    assert payload["reasoning"] == "on"
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 1024
    assert timeout == 120
    assert response.raw_response == "B"
    assert response.reasoning_observed is True
    assert response.final_output_tokens == 1


def test_final_output_tokens_are_null_when_not_safely_derivable(example: DatasetExample) -> None:
    transport = FakeTransport({
        "output": [{"type": "message", "content": "B"}],
        "usage": {"total_output_tokens": 3},
    })
    response = LMStudioProvider(lm_studio_config(), transport=transport).generate("prompt", example)
    assert response.final_output_tokens is None
    assert response.reasoning_observed is False


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


def http_error(status: int, body: bytes, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1:1234/api/v1/chat", status, "provider failure", headers or {}, io.BytesIO(body)
    )


def test_lm_studio_structured_internal_error_is_recorded_safely(example: DatasetExample) -> None:
    body = b'{"error":{"message":"Engine protocol predict request failed: fetch failed","type":"internal_error","code":"unknown","param":null}}'
    response = LMStudioProvider(
        lm_studio_config(), transport=FakeTransport(error=http_error(500, body, {"Authorization": "Bearer secret"}))
    ).generate("sensitive prompt", example)
    assert response.error_type == "server_error"
    assert response.http_status_code == 500
    assert response.provider_error_type == "internal_error"
    assert response.provider_error_code == "unknown"
    assert response.provider_error_message == "Engine protocol predict request failed: fetch failed"
    assert "secret" not in repr(response)
    assert "sensitive prompt" not in repr(response)


def test_lm_studio_non_json_and_empty_http_errors(example: DatasetExample) -> None:
    plain = LMStudioProvider(
        lm_studio_config(), transport=FakeTransport(error=http_error(500, b"temporary engine failure"))
    ).generate("prompt", example)
    assert plain.http_status_code == 500
    assert plain.provider_error_message == "temporary engine failure"

    empty = LMStudioProvider(
        lm_studio_config(), transport=FakeTransport(error=http_error(503, b""))
    ).generate("prompt", example)
    assert empty.error_type == "server_error"
    assert empty.http_status_code == 503
    assert empty.provider_error_message is None


def test_lm_studio_error_message_is_sanitized_limited_and_omits_html(example: DatasetExample) -> None:
    unsafe = "Bearer top-secret api_key=abc123 " + ("x" * 700)
    body = json.dumps({"error": {"message": unsafe}}).encode()
    response = LMStudioProvider(
        lm_studio_config(), transport=FakeTransport(error=http_error(500, body))
    ).generate("prompt", example)
    assert response.provider_error_message is not None
    assert "top-secret" not in response.provider_error_message
    assert "abc123" not in response.provider_error_message
    assert "[REDACTED]" in response.provider_error_message
    assert len(response.provider_error_message) <= 512
    assert response.provider_error_message.endswith("...")
    assert response.provider_error_message.startswith("Bearer [REDACTED]")

    html = LMStudioProvider(
        lm_studio_config(), transport=FakeTransport(error=http_error(502, b"<html><body>stack trace secret</body></html>"))
    ).generate("prompt", example)
    assert html.provider_error_message == "HTML provider error response omitted"


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "api_key=abc123",
        "api_key = abc123",
        "password: example-value",
        "password : example-value",
        "authorization = sensitive-value",
        "token: sensitive-value",
    ],
)
def test_lm_studio_error_message_redacts_secret_separator_variants(
    example: DatasetExample, unsafe_value: str
) -> None:
    body = json.dumps({"error": {"message": f"provider failed: {unsafe_value}"}}).encode()
    response = LMStudioProvider(
        lm_studio_config(), transport=FakeTransport(error=http_error(500, body))
    ).generate("prompt", example)
    assert response.provider_error_message is not None
    assert response.provider_error_message.endswith("=[REDACTED]")
    assert not any(
        value in response.provider_error_message
        for value in ("abc123", "example-value", "sensitive-value")
    )


def test_lm_studio_malformed_json_error_is_generic(example: DatasetExample) -> None:
    response = LMStudioProvider(
        lm_studio_config(), transport=FakeTransport(error=http_error(500, b'{"error":'))
    ).generate("prompt", example)
    assert response.provider_error_message == "Malformed provider error response"


def test_provider_factory_preserves_mock_and_isolates_lm_studio() -> None:
    assert isinstance(create_provider(ModelConfig(model_id="mock")), MockProvider)
    assert isinstance(create_provider(lm_studio_config(), transport=FakeTransport()), LMStudioProvider)
