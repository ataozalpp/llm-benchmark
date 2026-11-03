"""Gold-free generation contracts for RAG providers."""

from dataclasses import dataclass
from typing import Protocol

from .models import RagGenerationRequest


class RagProviderError(RuntimeError):
    """An expected provider execution failure."""


@dataclass(frozen=True)
class RagGenerationResponse:
    text: str
    finish_reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Generation text must be a string.")

        if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
            raise ValueError("finish_reason must be a nonblank string.")

        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)

            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None.")


class RagProvider(Protocol):
    def generate(self, request: RagGenerationRequest) -> RagGenerationResponse: ...
