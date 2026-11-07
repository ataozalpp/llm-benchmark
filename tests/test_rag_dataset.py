import json
from pathlib import Path

import pytest

from llm_benchmark.rag import dataset
from llm_benchmark.rag.dataset import load_rag_dataset

EXAMPLE = Path(__file__).parents[1] / "examples/rag/policy_qa_smoke.json"


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