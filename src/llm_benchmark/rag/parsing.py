"""Parse structured RAG answers without repairing model output."""

import json
from typing import Any

from .models import RagAnswer

MAX_RESPONSE_CHARS = 65_536

PARSE_FAILURE_CODES = (
    "not_string",
    "blank_response",
    "too_long",
    "invalid_json",
    "duplicate_field",
    "invalid_constant",
    "not_object",
    "wrong_fields",
    "invalid_field_type",
    "invalid_citation_id",
    "duplicate_citation",
    "abstention_with_content",
    "blank_answer",
)


class RagParseError(ValueError):
    """The generated response does not satisfy the answer contract."""

    def __init__(self, message: str, *, code: str) -> None:
        if code not in PARSE_FAILURE_CODES:
            raise ValueError("Unknown parse failure code.")
        super().__init__(message)
        self.code = code


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise RagParseError(f"Duplicate JSON field: {key}", code="duplicate_field")
        result[key] = value

    return result


def _reject_constant(value: str) -> Any:
    raise RagParseError(f"Invalid JSON constant: {value}", code="invalid_constant")


def parse_rag_answer(raw: str) -> RagAnswer:
    if not isinstance(raw, str):
        raise RagParseError("Response must be a string.", code="not_string")

    if not raw.strip():
        raise RagParseError("Response must not be blank.", code="blank_response")

    if len(raw) > MAX_RESPONSE_CHARS:
        raise RagParseError("Response exceeds the character limit.", code="too_long")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RagParseError:
        raise
    except (ValueError, RecursionError) as exc:
        raise RagParseError("Response must be valid JSON.", code="invalid_json") from exc

    if not isinstance(payload, dict):
        raise RagParseError("Response must be a JSON object.", code="not_object")

    if set(payload) != {"answer", "citations", "abstained"}:
        raise RagParseError(
            "Response must contain exactly answer, citations and abstained.",
            code="wrong_fields",
        )

    answer = payload["answer"]
    citations = payload["citations"]
    abstained = payload["abstained"]

    if not isinstance(citations, list):
        raise RagParseError("citations must be a JSON array.", code="invalid_field_type")

    if not isinstance(answer, str) or type(abstained) is not bool:
        raise RagParseError(
            "Invalid answer fields: answer must be a string and abstained a boolean.",
            code="invalid_field_type",
        )

    if any(not isinstance(item, str) or not item.strip() for item in citations):
        raise RagParseError(
            "Invalid answer fields: citation IDs must be nonblank strings.",
            code="invalid_citation_id",
        )

    if len(citations) != len(set(citations)):
        raise RagParseError(
            "Invalid answer fields: citation IDs must be unique.",
            code="duplicate_citation",
        )

    if abstained and (answer or citations):
        raise RagParseError(
            "Invalid answer fields: an abstention must have an empty answer "
            "and no citations.",
            code="abstention_with_content",
        )

    if not abstained and not answer.strip():
        raise RagParseError(
            "Invalid answer fields: a non-abstaining answer must be nonblank.",
            code="blank_answer",
        )

    return RagAnswer(answer=answer, citations=tuple(citations), abstained=abstained)