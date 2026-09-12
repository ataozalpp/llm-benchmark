from __future__ import annotations

import copy
import io
import json
import urllib.error
from dataclasses import asdict

import pytest

import llm_benchmark.providers as providers
from llm_benchmark.config import ModelConfig
from llm_benchmark.models import DatasetExample
from llm_benchmark.tool_calling import ToolCallNormalizationErrorCode
from llm_benchmark.tool_provider_models import ToolProviderErrorCode, ToolProviderStatus
from llm_benchmark.tool_requests import ToolTurnRequest, build_openai_tool_payload
from llm_benchmark.tool_runtime import ToolRuntime
from llm_benchmark.tools import create_example_tool_registry


class FakeTransport:
    def __init__(self, body: object = None, error: BaseException | None = None):
        self.body = body
        self.error = error
        self.calls = []

    def post_json(self, url, payload, timeout_seconds, headers=None):
        self.calls.append((url, copy.deepcopy(payload), timeout_seconds, headers))
        if self.error is not None:
            raise self.error
        return self.body


def config(**updates: object) -> ModelConfig:
    values = dict(
        provider="openai_compatible",
        model_id="synthetic",
        base_url="http://127.0.0.1:1234/v1/",
        timeout_seconds=45,
    )
    values.update(updates)
    return ModelConfig.model_validate(values)


def request() -> ToolTurnRequest:
    selected = create_example_tool_registry().get("calculator")
    assert selected is not None
    return ToolTurnRequest(user_content="Synthetic request", registrations=(selected,))


def response(content: str | None = None, calls: bool = True) -> dict:
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"operation":"add","left":2,"right":3}',
                },
            }
        ]
    return {
        "choices": [
            {"message": message, "finish_reason": "tool_calls" if calls else "stop"}
        ]
    }


@pytest.mark.parametrize(
    ("content", "calls"), [(None, True), ("Text", True), ("Text", False)]
)
def test_success_mapping_and_exact_request(monkeypatch, content, calls) -> None:
    body = response(content, calls)
    body["usage"] = {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    transport = FakeTransport(body)
    cfg = config()
    req = request()
    ticks = iter([1.0, 1.125])
    monkeypatch.setattr(providers.time, "perf_counter", lambda: next(ticks))
    result = providers.OpenAICompatibleProvider(cfg, transport).generate_tool_turn(req)
    assert result.status is ToolProviderStatus.SUCCEEDED
    assert result.turn.content == content
    assert len(result.turn.tool_calls) == int(calls)
    if calls:
        assert result.turn.tool_calls[0].call_id == "call_1"
        assert result.turn.tool_calls[0].arguments == {
            "operation": "add",
            "left": 2,
            "right": 3,
        }
    assert result.turn.finish_reason == ("tool_calls" if calls else "stop")
    assert transport.calls == [
        (
            "http://127.0.0.1:1234/v1/chat/completions",
            build_openai_tool_payload(cfg, req),
            45,
            None,
        )
    ]
    assert "max_tokens" not in transport.calls[0][1]
    assert "top_p" not in transport.calls[0][1]
    assert result.telemetry.latency_ms == 125
    assert result.telemetry.input_tokens == 12
    assert result.telemetry.output_tokens == 8
    assert result.telemetry.reasoning_tokens == 3
    assert result.telemetry.total_tokens == 20
    assert result.telemetry.ttft_ms is None
    assert result.telemetry.throughput_tokens_per_second is None


def test_explicit_budget_and_sampling() -> None:
    transport = FakeTransport(response())
    providers.OpenAICompatibleProvider(
        config(max_output_tokens=128, top_p=0.9, temperature=0.5), transport
    ).generate_tool_turn(request())
    payload = transport.calls[0][1]
    assert payload["max_tokens"] == 128
    assert payload["top_p"] == 0.9
    assert payload["temperature"] == 0.5
    assert payload["stream"] is False
    assert {
        "presence_penalty",
        "reasoning",
        "top_k",
        "min_p",
        "repeat_penalty",
        "max_output_tokens",
    }.isdisjoint(payload)


def test_credentials_read_at_request_time_only(monkeypatch) -> None:
    monkeypatch.delenv("TEST_TOOL_KEY", raising=False)
    transport = FakeTransport(response())
    cfg = config(credential_env_var="TEST_TOOL_KEY")
    provider = providers.OpenAICompatibleProvider(cfg, transport)
    req = request()
    for secret in ("SYNTHETIC_SECRET_ONE", "SYNTHETIC_SECRET_TWO"):
        monkeypatch.setenv("TEST_TOOL_KEY", secret)
        result = provider.generate_tool_turn(req)
        assert transport.calls[-1][3] == {"Authorization": f"Bearer {secret}"}
        for rendered in (
            repr(result),
            repr(provider),
            repr(cfg),
            json.dumps(asdict(result)),
            cfg.model_dump_json(),
            json.dumps(transport.calls[-1][1]),
        ):
            assert secret not in rendered
    assert len(transport.calls) == 2  # Two explicit invocations, one attempt each.


@pytest.mark.parametrize("value", [None, ""])
def test_missing_credential_has_no_transport_call(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("TEST_TOOL_KEY", raising=False)
    else:
        monkeypatch.setenv("TEST_TOOL_KEY", value)
    transport = FakeTransport(response())
    result = providers.OpenAICompatibleProvider(
        config(credential_env_var="TEST_TOOL_KEY"), transport
    ).generate_tool_turn(request())
    assert result.error_code is ToolProviderErrorCode.MISSING_CREDENTIAL
    assert result.status is ToolProviderStatus.REQUEST_FAILED
    assert result.telemetry.latency_ms == 0
    assert result.telemetry.input_tokens is None
    assert transport.calls == []


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "invalid_request"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (429, "rate_limit"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_http_errors_are_safe_and_never_retried(status, code) -> None:
    sentinel = "SYNTHETIC_PRIVATE_ERROR"
    body = io.BytesIO(sentinel.encode())
    error = urllib.error.HTTPError(
        "http://synthetic.invalid/" + sentinel, status, sentinel, {}, body
    )
    transport = FakeTransport(error=error)
    result = providers.OpenAICompatibleProvider(config(), transport).generate_tool_turn(
        request()
    )
    assert result.status is ToolProviderStatus.REQUEST_FAILED
    assert result.error_code.value == code
    assert result.turn is None
    assert result.normalization_error_code is None
    assert sentinel not in repr(result)
    assert sentinel not in json.dumps(asdict(result))
    assert body.tell() == 0  # Provider error body is not read or copied.
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TimeoutError("PRIVATE"), "timeout"),
        (urllib.error.URLError(TimeoutError("PRIVATE")), "timeout"),
        (urllib.error.URLError("PRIVATE"), "network_error"),
        (ConnectionError("PRIVATE"), "network_error"),
        (OSError("PRIVATE"), "network_error"),
        (json.JSONDecodeError("PRIVATE", "PRIVATE", 0), "malformed_json"),
        (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "PRIVATE"), "malformed_json"),
    ],
)
def test_transport_failure_mapping(error, code) -> None:
    transport = FakeTransport(error=error)
    result = providers.OpenAICompatibleProvider(config(), transport).generate_tool_turn(
        request()
    )
    assert result.error_code.value == code
    assert result.status is ToolProviderStatus.REQUEST_FAILED
    assert "PRIVATE" not in repr(result)
    assert result.telemetry.total_tokens is None
    assert len(transport.calls) == 1


@pytest.mark.parametrize("body", [None, [], {}, {"choices": []}, response("", False)])
def test_invalid_responses_are_not_transport_errors(body) -> None:
    transport = FakeTransport(body)
    result = providers.OpenAICompatibleProvider(config(), transport).generate_tool_turn(
        request()
    )
    assert result.status is ToolProviderStatus.RESPONSE_INVALID
    assert result.error_code is None
    assert result.normalization_error_code is not None
    assert result.turn is None
    assert len(transport.calls) == 1


def test_one_bad_call_rejects_whole_response_preserving_reported_usage() -> None:
    body = response("Some text")
    body["choices"][0]["message"]["tool_calls"].append({"id": "bad"})
    body["usage"] = {"input_tokens": 7}
    transport = FakeTransport(body)
    result = providers.OpenAICompatibleProvider(config(), transport).generate_tool_turn(
        request()
    )
    assert (
        result.normalization_error_code
        is ToolCallNormalizationErrorCode.INVALID_TOOL_CALL
    )
    assert result.turn is None
    assert result.telemetry.input_tokens == 7
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "usage",
    [
        None,
        [],
        {},
        {
            "prompt_tokens": True,
            "completion_tokens": -1,
            "total_tokens": "12",
            "reasoning_tokens": 1.5,
        },
    ],
)
def test_missing_or_invalid_usage_stays_null(usage) -> None:
    body = response()
    body["usage"] = usage
    result = providers.OpenAICompatibleProvider(
        config(), FakeTransport(body)
    ).generate_tool_turn(request())
    assert result.status is ToolProviderStatus.SUCCEEDED
    assert result.telemetry.input_tokens is None
    assert result.telemetry.output_tokens is None
    assert result.telemetry.reasoning_tokens is None
    assert result.telemetry.total_tokens is None


def test_usage_aliases_and_no_synthesized_total() -> None:
    body = response()
    body["usage"] = {"input_tokens": 0, "output_tokens": 4, "reasoning_tokens": 0}
    result = providers.OpenAICompatibleProvider(
        config(), FakeTransport(body)
    ).generate_tool_turn(request())
    assert result.telemetry.input_tokens == 0
    assert result.telemetry.output_tokens == 4
    assert result.telemetry.reasoning_tokens == 0
    assert result.telemetry.total_tokens is None


@pytest.mark.parametrize(
    "updates",
    [{"reasoning": "off"}, {"top_k": 20}, {"min_p": 0.0}, {"repeat_penalty": 1.0}],
)
def test_request_errors_precede_transport(updates) -> None:
    transport = FakeTransport(response())
    with pytest.raises(ValueError, match="Unsupported"):
        providers.OpenAICompatibleProvider(
            config(**updates), transport
        ).generate_tool_turn(request())
    assert transport.calls == []


class FatalSignal(BaseException):
    pass


@pytest.mark.parametrize(
    "error_type", [RuntimeError, TypeError, AssertionError, FatalSignal]
)
@pytest.mark.parametrize("boundary", ["transport", "normalizer"])
def test_programming_failures_propagate(monkeypatch, error_type, boundary) -> None:
    error = error_type("programming failure")
    transport = FakeTransport(response(), error if boundary == "transport" else None)
    if boundary == "normalizer":

        def fail(body):
            raise error

        monkeypatch.setattr(providers, "normalize_openai_tool_response", fail)
    with pytest.raises(error_type) as captured:
        providers.OpenAICompatibleProvider(config(), transport).generate_tool_turn(
            request()
        )
    assert captured.value is error
    assert len(transport.calls) == 1


def test_no_tool_execution_and_old_generate_preserved(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("Tool must not execute")

    monkeypatch.setattr(ToolRuntime, "execute", fail)
    transport = FakeTransport(response())
    provider = providers.OpenAICompatibleProvider(config(), transport)
    assert provider.generate_tool_turn(request()).status is ToolProviderStatus.SUCCEEDED
    example = DatasetExample("q", "Question", ["A", "B"], "A", "test")
    classic = provider.generate("Synthetic", example)
    assert classic.request_status == "failed"
    assert classic.provider_error_type == "empty_message_content"
    assert len(transport.calls) == 2
