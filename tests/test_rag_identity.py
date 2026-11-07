from dataclasses import asdict, replace
from pathlib import Path

import pytest

from llm_benchmark.rag.dataset import load_rag_dataset
from llm_benchmark.rag.identity import build_rag_run_identity
from llm_benchmark.retrieval.profile import BM25_PROFILE, KEYWORD_PROFILE

EXAMPLE = Path(__file__).parents[1] / "examples/retrieval/rag/policy_qa_smoke.json"
DATASET = load_rag_dataset(EXAMPLE)


def identity(**overrides):
    arguments = {
        "chunks": DATASET.chunks,
        "scenarios": DATASET.scenarios,
        "retriever_profile": asdict(KEYWORD_PROFILE),
        "k": 2,
        "max_context_chars": 2000,
        "generation": {"provider": "synthetic"},
    }
    arguments.update(overrides)
    return build_rag_run_identity(**arguments)


def test_identity_is_deterministic():
    assert identity() == identity()


def test_chunk_order_does_not_change_corpus_hash():
    reordered = identity(chunks=tuple(reversed(DATASET.chunks)))
    assert reordered["corpus_hash"] == identity()["corpus_hash"]


def test_reference_answer_change_changes_suite_hash():
    first = DATASET.scenarios[0]
    changed = (replace(first, reference_answers=("thirty days,")),) + DATASET.scenarios[1:]
    assert identity(scenarios=changed)["suite_hash"] != identity()["suite_hash"]


@pytest.mark.parametrize(
    "key, value, field",
    [
        ("k", 3, "evaluation_config_hash"),
        ("max_context_chars", 500, "evaluation_config_hash"),
        ("generation", {"provider": "openai_compatible"}, "generation_config_hash"),
        ("retriever_profile", asdict(BM25_PROFILE), "retriever_profile_hash"),
    ],
)
def test_settings_change_identity(key, value, field):
    base, other = identity(), identity(**{key: value})
    assert base[field] != other[field]
    assert base["run_fingerprint"] != other["run_fingerprint"]


@pytest.mark.parametrize("key, value", [("k", 0), ("k",True), ("max_context_chars", 1)])
def test_invalid_settings(key, value):
    with pytest.raises(ValueError):
        identity(**{key: value})