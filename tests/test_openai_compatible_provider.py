from __future__ import annotations

import io
import json
import socket
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from llm_benchmark.config import DatasetConfig, ModelConfig, RunConfig
from llm_benchmark.models import DatasetExample
from llm_benchmark.providers import (
    MockProvider,
    OpenAICompatibleProvider,
    create_provider,
)
from llm_benchmark.runner import _evaluate
from llm_benchmark.storage import append_result
from llm_benchmark.trace import InMemoryTraceRecorder


@pytest.fixture
def example() -> DatasetExample:
    return DatasetExample("q", "Question?", ["one", "two", "three"], "B", "test")


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any], float, dict[str, str] | None]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((url, payload, timeout_seconds, headers))
        if self.error is not None:
            raise self.error
        return self.response


def config(**updates: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "provider": "openai_compatible",
        "endpoint_alias": "local-openai",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "configured-model",
        "temperature": 0,
        "timeout_seconds": 45,
    }
    values.update(updates)
    return ModelConfig(**values)


def success_body() -> dict[str, Any]:
    return {
        "model": "returned-model",
        "choices": [{"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 7,
            "total_tokens": 27,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }


def http_error(status: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1:1234/v1/chat/completions",
        status,
        "provider failure",
        {},
        io.BytesIO(body),
    )


def test_exact_request_and_success_normalization(example: DatasetExample) -> None:
    transport = FakeTransport(success_body())
    response = OpenAICompatibleProvider(config(max_output_tokens=128, top_p=0.9), transport).generate(
        "prompt text", example
    )

    assert transport.calls == [
        (
            "http://127.0.0.1:1234/v1/chat/completions",
            {
                "model": "configured-model",
                "messages": [{"role": "user", "content": "prompt text"}],
                "temperature": 0.0,
                "max_tokens": 128,
                "top_p": 0.9,
            },
            45.0,
            None,
        )
    ]
    assert response.request_status == "succeeded"
    assert response.raw_response == "B"
    assert response.returned_model == "returned-model"
    assert response.prompt_tokens == response.input_tokens == 20
    assert response.completion_tokens == response.total_output_tokens == 7
    assert response.reasoning_output_tokens == 5
    assert response.final_output_tokens == 2
    assert response.reasoning_observed is True
    assert response.total_tokens == 27
    assert response.stop_reason == "stop"
    assert response.time_to_first_token_ms is None
    assert response.tokens_per_second is None


def test_omits_optional_output_limit_top_p_authorization_and_native_fields(example: DatasetExample) -> None:
    transport = FakeTransport(success_body())
    model = config(reasoning="on", top_k=20, min_p=0.1, repeat_penalty=1.1)
    OpenAICompatibleProvider(model, transport).generate("prompt", example)
    payload = transport.calls[0][1]
    assert "max_tokens" not in payload
    assert "top_p" not in payload
    assert "reasoning" not in payload
    assert "top_k" not in payload
    assert "min_p" not in payload
    assert "repeat_penalty" not in payload
    assert transport.calls[0][3] is None


def test_credential_is_resolved_at_request_time_and_only_sent_in_header(
    example: DatasetExample, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "temporary-test-credential"
    model = config(credential_env_var="BENCHMARK_TEST_KEY")
    transport = FakeTransport(success_body())
    provider = OpenAICompatibleProvider(model, transport)
    monkeypatch.setenv("BENCHMARK_TEST_KEY", secret)

    response = provider.generate("prompt", example)

    assert transport.calls[0][3] == {"Authorization": f"Bearer {secret}"}
    assert secret not in repr(model)
    assert secret not in json.dumps(model.model_dump(mode="json"))
    assert secret not in repr(response)


def test_credential_value_never_reaches_resolved_config_or_result_artifact(
    example: DatasetExample, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "artifact-secret-value"
    monkeypatch.setenv("BENCHMARK_ARTIFACT_KEY", secret)
    model = config(credential_env_var="BENCHMARK_ARTIFACT_KEY")
    run_config = RunConfig(
        schema_version=1,
        experiment_name="openai-artifact-test",
        dataset=DatasetConfig(source="local", name="fixture", path="fixture.jsonl"),
        models=[model],
    )
    provider = OpenAICompatibleProvider(model, FakeTransport(success_body()))
    trace_recorder = InMemoryTraceRecorder(
        clock=lambda: "2026-09-01T10:00:00+00:00"
    )

    result = _evaluate(
        "run",
        run_config,
        example,
        model,
        provider,
        trace_recorder=trace_recorder,
    )
    artifact = tmp_path / "results.jsonl"
    append_result(artifact, result)

    assert secret not in json.dumps(run_config.model_dump(mode="json"))
    assert secret not in artifact.read_text(encoding="utf-8")

    serialized_trace = json.dumps(
        [event.to_dict() for event in trace_recorder.events()]
    )
    assert secret not in serialized_trace


def test_missing_credential_fails_before_transport_call(
    example: DatasetExample, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_BENCHMARK_KEY", raising=False)
    transport = FakeTransport(success_body())
    response = OpenAICompatibleProvider(
        config(credential_env_var="MISSING_BENCHMARK_KEY"), transport
    ).generate("prompt", example)
    assert transport.calls == []
    assert response.request_status == "failed"
    assert response.error_type == "authentication_error"
    assert response.provider_error_type == "missing_credential_environment_variable"
    assert "MISSING_BENCHMARK_KEY" not in repr(response)


@pytest.mark.parametrize(
    ("body", "provider_error_type"),
    [
        ({}, "missing_choices"),
        ({"choices": []}, "missing_choices"),
        ({"choices": [{"message": {"content": ""}}]}, "empty_message_content"),
        ({"choices": [{"message": {"content": None}}]}, "empty_message_content"),
    ],
)
def test_invalid_success_responses_are_provider_failures(
    example: DatasetExample, body: dict[str, Any], provider_error_type: str
) -> None:
    response = OpenAICompatibleProvider(config(), FakeTransport(body)).generate("prompt", example)
    assert response.request_status == "failed"
    assert response.error_type == "provider_error"
    assert response.provider_error_type == provider_error_type
    assert response.raw_response is None


def test_reasoning_content_is_never_used_as_final_message(example: DatasetExample) -> None:
    body = {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": "FINAL ANSWER: B"},
                "finish_reason": "length",
            }
        ]
    }
    response = OpenAICompatibleProvider(config(), FakeTransport(body)).generate("prompt", example)
    assert response.request_status == "failed"
    assert response.provider_error_type == "empty_message_content"
    assert response.raw_response is None


def test_malformed_json_is_normalized(example: DatasetExample) -> None:
    error = json.JSONDecodeError("invalid", "not-json", 0)
    response = OpenAICompatibleProvider(config(), FakeTransport(error=error)).generate("prompt", example)
    assert response.request_status == "failed"
    assert response.error_type == "provider_error"
    assert response.provider_error_type == "malformed_json"
    assert response.provider_error_message == "Provider returned malformed JSON"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, "invalid_request"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (429, "rate_limit"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_http_error_mapping_and_secret_redaction(
    example: DatasetExample, status: int, expected: str
) -> None:
    body = json.dumps(
        {"error": {"message": "Bearer private-token api_key=private-key", "type": "api_error", "code": status}}
    ).encode()
    response = OpenAICompatibleProvider(config(), FakeTransport(error=http_error(status, body))).generate(
        "private prompt", example
    )
    assert response.request_status == "failed"
    assert response.error_type == expected
    assert response.http_status_code == status
    assert response.provider_error_type == "api_error"
    assert response.provider_error_code == str(status)
    assert response.provider_error_message is not None
    assert "private-token" not in repr(response)
    assert "private-key" not in repr(response)
    assert "private prompt" not in repr(response)


@pytest.mark.parametrize(
    ("error", "expected"),
    [(socket.timeout("slow"), "timeout"), (urllib.error.URLError("offline"), "network_error")],
)
def test_transport_error_mapping(example: DatasetExample, error: Exception, expected: str) -> None:
    response = OpenAICompatibleProvider(config(), FakeTransport(error=error)).generate("prompt", example)
    assert response.request_status == "failed"
    assert response.error_type == expected


def test_factory_adds_openai_without_affecting_mock(example: DatasetExample) -> None:
    openai = create_provider(config(), transport=FakeTransport(success_body()))
    mock = create_provider(ModelConfig(model_id="mock"))
    assert isinstance(openai, OpenAICompatibleProvider)
    assert isinstance(mock, MockProvider)
    assert mock.generate("prompt", example).request_status == "succeeded"
