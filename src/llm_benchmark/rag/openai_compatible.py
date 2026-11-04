"""Non-streaming RAG generation through a compatible chat endpoint."""

import json
import math
from http.client import HTTPException
from urllib.parse import urlsplit

from ..providers import JsonTransport, UrllibJsonTransport
from .models import RagGenerationRequest
from .prompts import build_rag_messages
from .providers import RagGenerationResponse, RagProviderError


def _normalize_response(payload: object) -> RagGenerationResponse:
    if not isinstance(payload, dict):
        raise RagProviderError("Expected a JSON response object.")

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RagProviderError("Expected exactly one response choice.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RagProviderError("Invalid response choice.")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise RagProviderError("Missing assistant message.")

    if message.get("role") != "assistant":
        raise RagProviderError("Expected an assistant message.")

    finish_reason = choice.get("finish_reason")
    content = message.get("content")

    # Some incomplete responses have no text content.
    if content is None and finish_reason != "stop":
        content = ""

    if (
        message.get("tool_calls") or message.get("function_call")
    ) and finish_reason == "stop":
        raise RagProviderError("Unexpected tool calls in a completed answer.")

    usage = payload.get("usage")
    if usage is None:
        usage = {}

    if not isinstance(usage, dict):
        raise RagProviderError("Invalid token usage object.")

    try:
        return RagGenerationResponse(
            text=content,
            finish_reason=finish_reason,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
    except (TypeError, ValueError) as exc:
        raise RagProviderError("Invalid generation response fields.") from exc


class OpenAICompatibleRagProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        timeout_seconds: float = 60.0,
        max_tokens: int = 256,
        transport: JsonTransport | None = None,
    ) -> None:
        if not isinstance(base_url, str):
            raise ValueError("base_url must be a string.")

        if any(character.isspace() or ord(character) < 32 for character in base_url):
            raise ValueError(
                "base_url must not contain whitespace or control characters."
            )

        try:
            parts = urlsplit(base_url)
            # urlsplit defers port validation until this property is accessed.
            port = parts.port
        except ValueError as exc:
            raise ValueError("Invalid HTTP base URL.") from exc

        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.query
            or parts.fragment
            or parts.username is not None
            or parts.password is not None
            or port == 0
        ):
            raise ValueError("Expected an HTTP base URL without credentials.")

        if parts.path.rstrip("/").endswith("/chat/completions"):
            raise ValueError("Pass the base URL, not the chat completions endpoint.")

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonblank string.")

        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive.")

        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model_id = model_id
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = max_tokens
        self.transport = transport if transport is not None else UrllibJsonTransport()

    def generate(
        self,
        request: RagGenerationRequest,
    ) -> RagGenerationResponse:
        payload = {
            "model": self.model_id,
            "messages": build_rag_messages(request),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "n": 1,
        }

        try:
            response = self.transport.post_json(
                url=self.endpoint,
                payload=payload,
                timeout_seconds=self.timeout_seconds,
            )
        except (
            OSError,
            HTTPException,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RagProviderError("Generation request failed.") from exc

        return _normalize_response(response)
