import json
from pathlib import Path

import pytest

from llm_benchmark.rag import dataset
from llm_benchmark.rag.dataset import load_rag_dataset

EXAMPLE = Path(__file__).parents[1] / "examples/rag/policy_qa_smoke.json"
FAILURE_CASES = Path(__file__).parents[1] / "examples/rag/policy_qa_failure_cases.json"


def write(tmp_path, payload):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_example_suite_loads():
    loaded = load_rag_dataset(EXAMPLE)
    assert loaded.suite_id == "policy-qa-smoke-v1"
    assert len(loaded.scenarios) == 5
    last = loaded.scenarios[-1]
    assert not last.answerable
    assert last.reference_answers == ()
    assert last.relevant_chunk_ids == frozenset()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["scenarios"][0].update(relevant_chunk_ids=["missing"]),
        lambda p: p["scenarios"][0].update(reference_answers=[]),
        lambda p: p["scenarios"][0].update(reference_answers=["x", "x"]),
        lambda p: p["scenarios"][-1].update(reference_answers=["cash"]),
        lambda p: p["scenarios"][0].update(answerable="yes"),
        lambda p: p["scenarios"][0].update(extra=1),
        lambda p: p.update(schema_version=2),
        lambda p: p["chunks"].append(dict(p["chunks"][0])),
        lambda p: p["scenarios"].append(dict(p["scenarios"][0])),
    ],
)
def test_invalid_suite_is_rejected(tmp_path, mutate):
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mutate(payload)
    with pytest.raises(ValueError, match="^Invalid RAG suite dataset.$"):
        load_rag_dataset(write(tmp_path, payload))


@pytest.mark.parametrize("raw", [b'{"a": 1, "a": 2}', b"NaN", b"\xff", b"[]", b""])
def test_malformed_json_is_rejected(tmp_path, raw):
    path = tmp_path / "suite.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="^Invalid RAG suite dataset.$"):
        load_rag_dataset(path)


def test_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "MAX_DATASET_BYTES", 10)
    with pytest.raises(ValueError, match="^Invalid RAG suite dataset.$"):
        load_rag_dataset(EXAMPLE)

def test_failure_case_suite_loads():
    loaded = load_rag_dataset(FAILURE_CASES)
    assert loaded.suite_id == "policy-qa-failure-cases-v1"
    assert len(loaded.chunks) == 9
    answerable = [item for item in loaded.scenarios if item.answerable]
    unanswerable = [item for item in loaded.scenarios if not item.answerable]
    assert len(answerable) == 5
    assert len(unanswerable) == 4
    assert all(item.reference_answers == () for item in unanswerable)


def test_failure_case_suite_covers_its_scenarios():
    loaded = load_rag_dataset(FAILURE_CASES)
    by_query = {item.query.query_id: item for item in loaded.scenarios}
    chunks = {item.chunk_id: item for item in loaded.chunks}

    # Multi-chunk evidence: both chunks are required for the total.
    assert by_query["q-international-total"].relevant_chunk_ids == frozenset(
        {"order-processing", "international-shipping"}
    )
    # Conflict: only the current policy is gold; the previous one is a distractor.
    assert by_query["q-refund-conflict"].relevant_chunk_ids == frozenset(
        {"refund-current"}
    )
    assert "30 days" in chunks["refund-previous"].text
    # The over-budget evidence chunk is larger than the example configs' budget.
    assert len(chunks["warranty-terms"].text) > 1000
    assert "Accidental damage is not covered" in chunks["warranty-terms"].text
    # The embedded instruction is corpus data, never gold evidence.
    assert all(
        "support-notice" not in item.relevant_chunk_ids for item in loaded.scenarios
    )
