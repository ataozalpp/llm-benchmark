import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_benchmark.retrieval import run_service
from llm_benchmark.retrieval.profile import BM25_PROFILE
from llm_benchmark.retrieval.run_config import (
    RetrievalRunConfig,
    load_retrieval_run_config,
)
from scripts import review_retrieval_run
from scripts.compare_retrieval_runs import compare_runs
from scripts.review_retrieval_run import build_review

EXAMPLE = Path(__file__).parents[1] / "examples" / "retrieval" / "keyword_smoke.json"


def test_bm25_config_rejects_embedding_settings():
    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(
            {
                "schema_version": 1,
                "suite_path": "suite.json",
                "retriever": "bm25",
                "k": 2,
                "embedding": {
                    "model_id": "test/model",
                    "model_revision": "a" * 40,
                    "embedding_dimension": 2,
                    "max_sequence_length": 4,
                },
            }
        )


def test_bm25_run_review_and_comparison(tmp_path, monkeypatch):
    suite = tmp_path / "suite.json"
    suite.write_bytes(EXAMPLE.read_bytes())
    config = tmp_path / "config.yaml"

    def forbidden(*args, **kwargs):
        pytest.fail("Lexical runs must not load embedding models.")

    monkeypatch.setattr(run_service, "load_cpu_embedder", forbidden)

    def run(retriever):
        config.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "suite_path": "suite.json",
                    "retriever": retriever,
                    "k": 2,
                    "output_dir": "runs",
                }
            ),
            encoding="utf-8",
        )
        return run_service.run_retrieval_benchmark(load_retrieval_run_config(config))

    baseline, _ = run("keyword")
    directory, summary = run("bm25")

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    baseline_manifest = json.loads(
        (baseline / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "completed"
    assert manifest["completed_case_count"] == 4
    assert manifest["retriever_profile"] == asdict(BM25_PROFILE)
    assert "embedding_runtime" not in manifest
    assert "embedding_input_statistics" not in manifest
    assert summary["model_load_ms"] == 0.0
    assert summary["hit_rate_at_k"] == 0.75

    assert manifest["corpus_hash"] == baseline_manifest["corpus_hash"]
    assert manifest["run_fingerprint"] != baseline_manifest["run_fingerprint"]

    def forbidden_keyword_replay(*args, **kwargs):
        pytest.fail("BM25 replay must not construct KeywordRetriever.")

    monkeypatch.setattr(
        review_retrieval_run, "KeywordRetriever", forbidden_keyword_replay
    )

    review = build_review(suite, directory, verify_retrieval=True)
    assert review["retrieval_replay"] == "matched"

    comparison = compare_runs(suite, baseline, directory)
    assert comparison["query_count"] == 4
    assert comparison["hit_rate_delta"] == 0.0


def test_bm25_index_time_is_recorded_in_milliseconds(tmp_path, monkeypatch):
    suite = tmp_path / "suite.json"
    suite.write_bytes(EXAMPLE.read_bytes())

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "suite_path": "suite.json",
                "retriever": "bm25",
                "k": 2,
                "output_dir": "runs",
            }
        ),
        encoding="utf-8",
    )

    clock = {"seconds": 0.0}
    original = run_service.BM25Retriever

    class TimedBM25Retriever(original):
        def __init__(self, chunks):
            super().__init__(chunks)
            clock["seconds"] += 2.5

    monkeypatch.setattr(
        run_service.time,
        "perf_counter",
        lambda: clock["seconds"],
    )
    monkeypatch.setattr(
        run_service,
        "BM25Retriever",
        TimedBM25Retriever,
    )

    _, summary = run_service.run_retrieval_benchmark(load_retrieval_run_config(config))

    assert summary["index_build_ms"] == pytest.approx(2500.0)
    assert summary["model_load_ms"] == 0.0
