"""Validated RAG run configuration and shipped example configs."""

from pathlib import Path

import pytest
import yaml

from llm_benchmark.rag import run_config
from llm_benchmark.rag.run_config import load_rag_run_config
from llm_benchmark.retrieval.run_config import (
    DenseEmbeddingConfig,
    HybridFusionConfig,
)

ROOT = Path(__file__).parents[1]
MESSAGE = "^Invalid RAG run configuration.$"
MODEL = {
    "provider": "openai_compatible",
    "endpoint_alias": "local",
    "model_id": "test-model",
    "base_url": "http://127.0.0.1:1234/v1",
    "temperature": 0,
    "max_output_tokens": 128,
    "timeout_seconds": 5,
}
EMBEDDING = {
    "model_id": "test/encoder",
    "model_revision": "a" * 40,
    "embedding_dimension": 2,
    "max_sequence_length": 256,
}


def base_payload():
    return {
        "schema_version": 1,
        "suite_path": "suite.json",
        "retriever": "bm25",
        "k": 2,
        "max_context_chars": 2000,
        "provider": "synthetic",
        "output_dir": "runs",
    }


def write(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_paths_resolve_relative_to_config_not_cwd(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = load_rag_run_config(write(tmp_path, base_payload()))
    assert loaded.suite_path == tmp_path / "suite.json"
    assert loaded.output_dir == tmp_path / "runs"
    assert not loaded.output_dir.exists()


def test_defaults(tmp_path):
    payload = base_payload()
    del payload["provider"], payload["output_dir"]
    loaded = load_rag_run_config(write(tmp_path, payload))
    assert loaded.config.provider == "synthetic"
    assert loaded.config.model is None
    assert loaded.output_dir == tmp_path / "outputs" / "rag"
    assert loaded.config.prompt_version == "rag-grounded-json-v1"
    assert loaded.config.structured_output == "none"


@pytest.mark.parametrize(
    "updates",
    [
        {"prompt_version": "rag-grounded-json-v3"},
        {
            "provider": "openai_compatible",
            "model": MODEL,
            "prompt_version": "rag-grounded-json-v2",
            "structured_output": "json_schema",
        },
    ],
)
def test_prompt_and_structured_output_are_accepted(tmp_path, updates):
    loaded = load_rag_run_config(write(tmp_path, {**base_payload(), **updates}))
    for key, value in updates.items():
        if key != "model":
            assert getattr(loaded.config, key) == value


def test_openai_compatible_config(tmp_path):
    payload = {**base_payload(), "provider": "openai_compatible", "model": MODEL}
    loaded = load_rag_run_config(write(tmp_path, payload))
    assert loaded.config.model.model_id == "test-model"
    assert loaded.config.model.max_output_tokens == 128


@pytest.mark.parametrize(
    "updates",
    [
        {"retriever": "dense", "embedding": EMBEDDING},
        {
            "retriever": "hybrid",
            "embedding": EMBEDDING,
            "hybrid": {"candidate_k": 5, "rank_constant": 60},
        },
        {"retriever": "keyword"},
    ],
)
def test_all_retrievers_are_accepted(tmp_path, updates):
    loaded = load_rag_run_config(write(tmp_path, {**base_payload(), **updates}))
    assert loaded.config.retriever == updates["retriever"]
    if "embedding" in updates:
        assert isinstance(loaded.config.embedding, DenseEmbeddingConfig)
    if "hybrid" in updates:
        assert isinstance(loaded.config.hybrid, HybridFusionConfig)


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"k": 0},
        {"k": True},
        {"k": "2"},
        {"max_context_chars": 1},
        {"max_context_chars": 2.5},
        {"retriever": "vector"},
        {"suite_path": " "},
        {"output_dir": "a\x00b"},
        {"unknown": 1},
        {"provider": "lm_studio"},
        {"model": MODEL},
        {"provider": "openai_compatible"},
        {"provider": "openai_compatible", "model": {**MODEL, "provider": "mock"}},
        {"provider": "openai_compatible", "model": {**MODEL, "temperature": 0.5}},
        {
            "provider": "openai_compatible",
            "model": {**MODEL, "credential_env_var": "API_KEY"},
        },
        {
            "provider": "openai_compatible",
            "model": {k: v for k, v in MODEL.items() if k != "max_output_tokens"},
        },
        {"provider": "openai_compatible", "model": {**MODEL, "base_url": "ftp://x"}},
        {"retriever": "dense"},
        {"retriever": "bm25", "embedding": EMBEDDING},
        {"retriever": "hybrid", "embedding": EMBEDDING},
        {"hybrid": {"candidate_k": 5}},
        {
            "retriever": "hybrid",
            "k": 6,
            "embedding": EMBEDDING,
            "hybrid": {"candidate_k": 5},
        },
        {"retriever": "dense", "embedding": {**EMBEDDING, "model_revision": "main"}},
        {"prompt_version": "rag-grounded-json-v9"},
        {"prompt_version": 1},
        {"structured_output": "json_schema"},
        {
            "provider": "openai_compatible",
            "model": MODEL,
            "structured_output": "json_object",
        },
    ],
)
def test_invalid_config(tmp_path, updates):
    with pytest.raises(ValueError, match=MESSAGE):
        load_rag_run_config(write(tmp_path, {**base_payload(), **updates}))


@pytest.mark.parametrize(
    "raw",
    [b"k: 1\nk: 2", b"!!python/object:secret {}", b"\xff", b"[]", b"", b"[broken"],
)
def test_parser_rejects_ambiguous_or_invalid_input(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=MESSAGE):
        load_rag_run_config(path)


def test_missing_and_oversized_files(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=MESSAGE):
        load_rag_run_config(tmp_path / "missing.yaml")
    path = write(tmp_path, base_payload())
    monkeypatch.setattr(run_config, "MAX_CONFIG_BYTES", 8)
    with pytest.raises(ValueError, match=MESSAGE):
        load_rag_run_config(path)


@pytest.mark.parametrize(
    "name",
    [
        "rag_synthetic_smoke.yaml",
        "rag_hybrid_synthetic_smoke.yaml",
        "rag_openai_compatible_smoke.yaml",
        "rag_hybrid_openai_compatible_smoke.yaml",
    ],
)
def test_example_configs_load_and_point_at_the_fixture(name):
    loaded = load_rag_run_config(ROOT / "configs" / name)
    assert loaded.suite_path == (ROOT / "examples/rag/policy_qa_smoke.json").resolve()
    assert loaded.suite_path.is_file()
    assert loaded.output_dir == (ROOT / "outputs/rag").resolve()


@pytest.mark.parametrize(
    ("name", "retriever", "provider"),
    [
        ("rag_failure_cases_synthetic.yaml", "bm25", "synthetic"),
        ("rag_failure_cases_hybrid_synthetic.yaml", "hybrid", "synthetic"),
        ("rag_failure_cases_openai_compatible.yaml", "bm25", "openai_compatible"),
        (
            "rag_failure_cases_hybrid_openai_compatible.yaml",
            "hybrid",
            "openai_compatible",
        ),
    ],
)
def test_failure_case_configs_point_at_their_fixture(name, retriever, provider):
    loaded = load_rag_run_config(ROOT / "configs" / name)
    fixture = ROOT / "examples/rag/policy_qa_failure_cases.json"
    assert loaded.suite_path == fixture.resolve()
    assert loaded.suite_path.is_file()
    assert loaded.output_dir == (ROOT / "outputs/rag").resolve()
    assert loaded.config.retriever == retriever
    assert loaded.config.provider == provider
    assert loaded.config.k == 2
    # Smaller than the warranty-terms chunk, so the budget case is exercised.
    assert loaded.config.max_context_chars == 1000


@pytest.mark.parametrize(
    "name",
    [
        "rag_failure_cases_openai_compatible.yaml",
        "rag_failure_cases_hybrid_openai_compatible.yaml",
    ],
)
def test_opt_in_failure_case_configs_keep_a_placeholder_model(name):
    loaded = load_rag_run_config(ROOT / "configs" / name)
    assert loaded.config.model.model_id == "replace-with-your-model-id"
