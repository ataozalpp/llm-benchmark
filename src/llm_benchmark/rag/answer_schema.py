"""JSON Schema for schema-constrained RAG answer generation.

The schema mirrors the parser's answer contract, including the abstention
rule, so a server that enforces it cannot emit an abstention with content
or a non-abstaining blank answer. The parser still validates every response:
a server may ignore ``response_format`` without reporting it.

Citation IDs are deliberately not restricted to the supplied context, so
citation validity keeps measuring the model rather than the grammar.
"""

import copy

ANSWER_SCHEMA_VERSION = "rag-answer-schema-v1"
ANSWER_SCHEMA_NAME = "rag_answer"

_ANSWER_SCHEMA: dict[str, object] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "const": ""},
                "citations": {"type": "array", "maxItems": 0},
                "abstained": {"type": "boolean", "const": True},
            },
            "required": ["answer", "citations", "abstained"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "citations": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
                "abstained": {"type": "boolean", "const": False},
            },
            "required": ["answer", "citations", "abstained"],
            "additionalProperties": False,
        },
    ]
}


def answer_schema() -> dict[str, object]:
    """Return an independent copy of the answer schema."""
    return copy.deepcopy(_ANSWER_SCHEMA)


def response_format() -> dict[str, object]:
    """Return the OpenAI-compatible ``response_format`` request field."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": ANSWER_SCHEMA_NAME,
            "strict": True,
            "schema": answer_schema(),
        },
    }
