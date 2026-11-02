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
