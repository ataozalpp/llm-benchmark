from __future__ import annotations

from typing import Protocol

from .config import ModelConfig
from .models import DatasetExample, ProviderResponse


class Provider(Protocol):
    name: str
    def generate(self, prompt: str, example: DatasetExample) -> ProviderResponse: ...


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
            choices = example.allowed_labels
            answer = next(label for label in choices if label != example.correct_answer)
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
        return ProviderResponse("succeeded", raw, prompt_tokens_out, completion_tokens_out, total_tokens, latency, returned_model=self.config.model_id)
