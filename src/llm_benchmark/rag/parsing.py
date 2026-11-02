"""Parse structured RAG answers without repairing model output."""

import json
from typing import Any

from .models import RagAnswer

MAX_RESPONSE_CHARS = 65_536


class RagParseError(ValueError):
    """The generated response does not satisfy the answer contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise RagParseError(f"Duplicate JSON field: {key}")
        result[key] = value

    return result


def _reject_constant(value: str) -> Any:
    raise RagParseError(f"Invalid JSON constant: {value}")


def parse_rag_answer(raw: str) -> RagAnswer:
    if not isinstance(raw, str):
        raise RagParseError("Response must be a string.")

    if not raw.strip():
        raise RagParseError("Response must not be blank.")

    if len(raw) > MAX_RESPONSE_CHARS:
        raise RagParseError("Response exceeds the character limit.")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RagParseError:
        raise
    except (ValueError, RecursionError) as exc:
        raise RagParseError("Response must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RagParseError("Response must be a JSON object.")

    expected_fields = {"answer", "citations", "abstained"}

    if set(payload) != expected_fields:
        raise RagParseError(
            "Response must contain exactly answer, citations and abstained."
        )

    if not isinstance(payload["citations"], list):
        raise RagParseError("citations must be a JSON array.")

    try:
        return RagAnswer(
            answer=payload["answer"],
            citations=tuple(payload["citations"]),
            abstained=payload["abstained"],
        )
    except (TypeError, ValueError) as exc:
        raise RagParseError(f"Invalid answer fields: {exc}") from exc
