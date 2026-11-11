"""The constrained-decoding schema mirrors the parser's answer contract."""

import json

import pytest
from jsonschema import Draft202012Validator

from llm_benchmark.rag.answer_schema import (
    ANSWER_SCHEMA_NAME,
    ANSWER_SCHEMA_VERSION,
    answer_schema,
    response_format,
)
from llm_benchmark.rag.parsing import RagParseError, parse_rag_answer

VALIDATOR = Draft202012Validator(answer_schema())

ACCEPTED = [
    {"answer": "", "citations": [], "abstained": True},
    {"answer": "30 days", "citations": ["refund"], "abstained": False},
    {"answer": "Email", "citations": [], "abstained": False},
    {"answer": "Free", "citations": ["a", "b"], "abstained": False},
]

REJECTED = [
    # abstention_with_content
    {"answer": "No", "citations": [], "abstained": True},
    {"answer": "", "citations": ["refund"], "abstained": True},
    # blank_answer
    {"answer": "", "citations": [], "abstained": False},
    # duplicate_citation
    {"answer": "x", "citations": ["a", "a"], "abstained": False},
    # invalid_citation_id
    {"answer": "x", "citations": [""], "abstained": False},
    {"answer": "x", "citations": [1], "abstained": False},
    # wrong_fields
    {"answer": "x", "citations": [], "abstained": False, "note": "y"},
    {"answer": "x", "citations": []},
    # invalid_field_type
    {"answer": 1, "citations": [], "abstained": False},
    {"answer": "x", "citations": [], "abstained": "false"},
]


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(answer_schema())


@pytest.mark.parametrize("value", ACCEPTED)
def test_schema_and_parser_accept_valid_answers(value):
    assert VALIDATOR.is_valid(value)
    parse_rag_answer(json.dumps(value))


@pytest.mark.parametrize("value", REJECTED)
def test_schema_and_parser_reject_contract_violations(value):
    assert not VALIDATOR.is_valid(value)
    with pytest.raises(RagParseError):
        parse_rag_answer(json.dumps(value))


def test_response_format_wraps_schema():
    value = response_format()
    assert value == {
        "type": "json_schema",
        "json_schema": {
            "name": ANSWER_SCHEMA_NAME,
            "strict": True,
            "schema": answer_schema(),
        },
    }
    assert ANSWER_SCHEMA_VERSION == "rag-answer-schema-v1"


def test_returned_schema_is_an_independent_copy():
    first = answer_schema()
    first["anyOf"].clear()
    assert answer_schema()["anyOf"]
