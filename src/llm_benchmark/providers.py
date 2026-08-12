from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .config import ModelConfig
from .models import DatasetExample, ProviderResponse


class Provider(Protocol):
    name: str

    def generate(self, prompt: str, example: DatasetExample) -> ProviderResponse: ...


class JsonTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]: ...


class UrllibJsonTransport:
    def post_json(self, url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class MockProvider:
    name = "mock"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def _scenario(self, example: DatasetExample) -> str:
        if example.sample_id in self.config.scenario_overrides:
            return self.config.scenario_overrides[example.sample_id]
        checksum = sum(example.sample_id.encode("utf-8"))
        return self.config.scenario_cycle[checksum % len(self.config.scenario_cycle)]

    def generate(self, prompt: str, example: DatasetExample) -> ProviderResponse:
        scenario = self._scenario(example)
        prompt_tokens = max(1, len(prompt.split()))
        latency = self.config.mock_latency_ms
        if scenario == "request_failed":
            return ProviderResponse("failed", None, None, None, None, latency, "server_error", self.config.model_id)
        if scenario == "incorrect":
            answer = next(label for label in example.allowed_labels if label != example.correct_answer)
            raw = f"FINAL ANSWER: {answer}"
        elif scenario == "unparseable":
            raw = f"{example.allowed_labels[0]} and {example.allowed_labels[1]} are possible"
        else:
            raw = f"FINAL ANSWER: {example.correct_answer}"
        completion_tokens = len(raw.split())
        if scenario == "missing_usage":
            prompt_tokens_out = completion_tokens_out = total_tokens = None
        else:
            prompt_tokens_out = prompt_tokens
            completion_tokens_out = completion_tokens
            total_tokens = prompt_tokens + completion_tokens
        return ProviderResponse(
            "succeeded",
            raw,
            prompt_tokens_out,
            completion_tokens_out,
            total_tokens,
            latency,
            returned_model=self.config.model_id,
            input_tokens=prompt_tokens_out,
            total_output_tokens=completion_tokens_out,
        )


class LMStudioProvider:
    """LM Studio native `/api/v1/chat` provider; separate from OpenAI compatibility."""

    name = "lm_studio"

    def __init__(self, config: ModelConfig, transport: JsonTransport | None = None) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport()

    def generate(self, prompt: str, example: DatasetExample) -> ProviderResponse:
        del example
        assert self.config.base_url is not None
        url = f"{self.config.base_url.rstrip('/')}/api/v1/chat"
        payload = {
            "model": self.config.model_id,
            "input": prompt,
            "reasoning": self.config.reasoning,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
        }
        started = time.perf_counter()
        try:
            body = self.transport.post_json(url, payload, self.config.timeout_seconds)
            latency_ms = (time.perf_counter() - started) * 1000
            raw = _message_text(body.get("output"))
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            stats = body.get("stats") if isinstance(body.get("stats"), dict) else {}
            input_tokens = _integer(usage, stats, "input_tokens", "prompt_tokens")
            output_tokens = _integer(usage, stats, "total_output_tokens", "output_tokens", "completion_tokens")
            reasoning_tokens = _integer(usage, stats, "reasoning_output_tokens", "reasoning_tokens")
            total_tokens = _integer(usage, stats, "total_tokens")
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            return ProviderResponse(
                request_status="succeeded",
                raw_response=raw,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                returned_model=_string(body, "model", "model_id") or self.config.model_id,
                input_tokens=input_tokens,
                total_output_tokens=output_tokens,
                reasoning_output_tokens=reasoning_tokens,
                tokens_per_second=_number(stats, usage, "tokens_per_second", "output_tokens_per_second"),
                time_to_first_token_ms=_duration_ms(stats, usage),
                stop_reason=_stop_reason(body),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            return ProviderResponse(
                request_status="failed",
                raw_response=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_ms=latency_ms,
                error_type=_error_type(exc),
                returned_model=self.config.model_id,
            )


def create_provider(config: ModelConfig, transport: JsonTransport | None = None) -> Provider:
    if config.provider == "mock":
        return MockProvider(config)
    if config.provider == "lm_studio":
        return LMStudioProvider(config, transport=transport)
    raise ValueError(f"unsupported provider: {config.provider}")


def _message_text(output: Any) -> str:
    if not isinstance(output, list):
        return ""
    messages: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            messages.append(content)
        elif isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in {"text", "output_text"}]
            messages.append("".join(text_parts))
    return "\n".join(text for text in messages if text)


def _integer(*sources_and_keys: Any) -> int | None:
    sources: list[dict[str, Any]] = []
    keys: list[str] = []
    for value in sources_and_keys:
        (sources if isinstance(value, dict) else keys).append(value)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _number(*sources_and_keys: Any) -> float | None:
    sources: list[dict[str, Any]] = []
    keys: list[str] = []
    for value in sources_and_keys:
        (sources if isinstance(value, dict) else keys).append(value)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _duration_ms(*sources: dict[str, Any]) -> float | None:
    milliseconds = _number(*sources, "time_to_first_token_ms", "ttft_ms")
    if milliseconds is not None:
        return milliseconds
    seconds = _number(*sources, "time_to_first_token_seconds", "time_to_first_token")
    return seconds * 1000 if seconds is not None else None


def _string(source: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str):
            return value
    return None


def _stop_reason(body: dict[str, Any]) -> str | None:
    top_level = _string(body, "stop_reason", "finish_reason")
    if top_level is not None:
        return top_level
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                item_reason = _string(item, "stop_reason", "finish_reason")
                if item_reason is not None:
                    return item_reason
    return None


def _error_type(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return "rate_limit"
        if 400 <= exc.code < 500:
            return "invalid_request"
        if exc.code >= 500:
            return "server_error"
    if isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
        return "network_error"
    if isinstance(exc, (json.JSONDecodeError, KeyError, TypeError, ValueError)):
        return "provider_error"
    return "unknown_error"
