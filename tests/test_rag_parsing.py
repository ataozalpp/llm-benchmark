import json

import pytest

from llm_benchmark.rag.parsing import (
    MAX_RESPONSE_CHARS,
    RagParseError,
    parse_rag_answer,
)


def test_parse_answer():
    answer = parse_rag_answer(
        '{"answer":"30 days","citations":["refund"], "abstained":false}'
    )

    assert answer.answer == "30 days"
    assert answer.citations == ("refund",)
    assert answer.abstained is False


def test_parse_abstention():
    answer = parse_rag_answer('{"answer":"","citations":[],"abstained":true}')

    assert answer.answer == ""
    assert answer.citations == ()
    assert answer.abstained is True


def test_unknown_citation_is_left_for_evaluation():
    answer = parse_rag_answer(
        '{"answer":"30 days","citations":["unknown"],"abstained":false}'
    )

    assert answer.citations == ("unknown",)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "[]",
        "null",
        "{}",
        '{"answer":"30 days","citations":[]}',
        ('{"answer":"30 days","citations":[],"abstained":false,"extra":1}'),
        ('{"answer":"30 days","answer":"60 days","citations":[],"abstained":false}'),
        '{"answer":NaN,"citations":[],"abstained":false}',
        '{"answer":"30 days","citations":"refund","abstained":false}',
        '{"answer":"30 days","citations":[1],"abstained":false}',
        '{"answer":"30 days","citations":["a","a"],"abstained":false}',
        '{"answer":"30 days","citations":[],"abstained":"false"}',
        '{"answer":"","citations":[],"abstained":false}',
        '{"answer":"30 days","citations":[],"abstained":true}',
        '```json\n{"answer":"","citations":[],"abstained":true}\n```',
    ],
)
def test_invalid_responses_are_rejected(raw):
    with pytest.raises(RagParseError):
        parse_rag_answer(raw)


def test_non_string_response_is_rejected():
    with pytest.raises(RagParseError, match="string"):
        parse_rag_answer(None)


def test_oversized_response_is_rejected():
    raw = json.dumps(
        {
            "answer": "x" * MAX_RESPONSE_CHARS,
            "citations": [],
            "abstained": False,
        }
    )

    with pytest.raises(RagParseError, match="character limit"):
        parse_rag_answer(raw)


def test_response_at_character_limit_is_accepted():
    raw = '{"answer":"30 days","citations":[],"abstained":false}'
    raw += " " * (MAX_RESPONSE_CHARS - len(raw))

    assert parse_rag_answer(raw).answer == "30 days"
    with pytest.raises(RagParseError, match="character limit"):
        parse_rag_answer(raw + " ")


def test_excessively_nested_json_is_a_parse_error():
    raw = "[" * 10_000 + "0" + "]" * 10_000

    with pytest.raises(RagParseError, match="valid JSON"):
        parse_rag_answer(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":Infinity,"citations":[],"abstained":false}',
        '{"answer":-Infinity,"citations":[],"abstained":false}',
        '{"answer":"30 days","citations":[null],"abstained":false}',
        '{"answer":"30 days","citations":[""],"abstained":false}',
        '{"answer":"30 days","citations":[{}],"abstained":false}',
        '{"answer":"30 days","citations":[],"abstained":0}',
        '{"answer":"","citations":["refund"],"abstained":true}',
        '{"answer":"30 days","citations":[],"abstained":false} trailing',
    ],
)
def test_invalid_field_values_and_trailing_text_are_rejected(raw):
    with pytest.raises(RagParseError):
        parse_rag_answer(raw)


@pytest.mark.parametrize(
    "raw,code",
    [
        ("   ", "blank_response"),
        ("{", "invalid_json"),
        ('{"answer": "a", "answer": "b"}', "duplicate_field"),
        ('{"answer": NaN, "citations": [], "abstained": false}', "invalid_constant"),
        ("[]", "not_object"),
        ('{"answer": "a", "citations": []}', "wrong_fields"),
        ('{"answer": "a", "citations": "x", "abstained": false}', "invalid_field_type"),
        ('{"answer": 1, "citations": [], "abstained": false}', "invalid_field_type"),
        ('{"answer": "a", "citations": [], "abstained": "no"}', "invalid_field_type"),
        ('{"answer": "a", "citations": [1], "abstained": false}', "invalid_citation_id"),
        ('{"answer": "a", "citations": [" "], "abstained": false}', "invalid_citation_id"),
        ('{"answer": "a", "citations": ["x", "x"], "abstained": false}', "duplicate_citation"),
        ('{"answer": "No", "citations": [], "abstained": true}', "abstention_with_content"),
        ('{"answer": "", "citations": ["x"], "abstained": true}', "abstention_with_content"),
        ('{"answer": " ", "citations": [], "abstained": false}', "blank_answer"),
    ],
)
def test_parse_failures_have_stable_codes(raw, code):
    with pytest.raises(RagParseError) as caught:
        parse_rag_answer(raw)
    assert caught.value.code == code


def test_non_string_and_oversized_codes():
    with pytest.raises(RagParseError) as caught:
        parse_rag_answer(None)
    assert caught.value.code == "not_string"

    with pytest.raises(RagParseError) as caught:
        parse_rag_answer("x" * (MAX_RESPONSE_CHARS + 1))
    assert caught.value.code == "too_long"


def test_unknown_parse_code_is_rejected():
    with pytest.raises(ValueError, match="Unknown parse failure code"):
        RagParseError("message", code="other")
