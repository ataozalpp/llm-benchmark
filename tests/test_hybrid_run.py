import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from llm_benchmark.retrieval import run_service
from llm_benchmark.retrieval.identity import fingerprint
from llm_benchmark.retrieval.run_config import load_retrieval_run_config
from scripts.review_retrieval_run import build_review

EXAMPLE = Path(__file__).parents[1] / "examples/retrieval/keyword_smoke.json"


class FixedEmbedder:
    def __init__(self):
        self.model = object()
        self.document_calls = 0
        self.document_count = 0
        self.query_count = 0

    def encode_documents(self, texts):
        self.document_calls += 1
        self.document_count += len(texts)
        return tuple((1.0, 0.0) for _ in texts)

    def encode_query(self, text):
        self.query_count += 1
        return (1.0, 0.0)

    def input_statistics(self):
        return {
            "schema_version": 1,
            "encoded_document_count": self.document_count,
            "over_limit_document_count": 0,
            "encoded_query_count": self.query_count,
            "over_limit_query_count": 0,
        }


def test_hybrid_run_loads_model_once_and_supports_recorded_review(
    tmp_path, monkeypatch
):
    suite = tmp_path / "suite.json"
    suite.write_bytes(EXAMPLE.read_bytes())

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "suite_path": "suite.json",
                "retriever": "hybrid",
                "k": 2,
                "output_dir": "runs",
                "embedding": {
                    "model_id": "test/model",
                    "model_revision": "a" * 40,
                    "embedding_dimension": 2,
                    "max_sequence_length": 256,
                },
                "hybrid": {
                    "candidate_k": 5,
                    "rank_constant": 60,
                },
            }
        ),
        encoding="utf-8",
    )

    embedder = FixedEmbedder()
    loads = []

    def load(profile, *, local_files_only):
        loads.append((profile, local_files_only))
        return embedder

    monkeypatch.setattr(run_service, "load_cpu_embedder", load)
    monkeypatch.setattr(
        run_service, "collect_embedding_runtime", lambda model: {"schema_version": 1}
    )

    loaded = load_retrieval_run_config(config)
    directory, summary = run_service.run_retrieval_benchmark(loaded)

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert len(loads) == 1
    assert loads[0] == (loaded.config.embedding.to_profile(), True)
    assert embedder.document_calls == 1
    assert embedder.document_count == 4
    assert embedder.query_count == 4
    assert manifest["status"] == "completed"
    assert manifest["completed_case_count"] == 4
    assert summary["query_count"] == 4

    profile = manifest["retriever_profile"]
    assert profile["name"] == "hybrid-bm25-dense-rrf"
    assert profile["candidate_k"] == 5
    assert profile["dense"] == asdict(loaded.config.embedding.to_profile())
    assert manifest["retriever_profile_hash"] == fingerprint(profile)

    def forbidden(*args, **kwargs):
        pytest.fail("Recorded review must not load an embedding model.")

    monkeypatch.setattr(run_service, "load_cpu_embedder", forbidden)

    review = build_review(suite, directory)
    assert review["retrieval_replay"] == "not_requested"

    with pytest.raises(ValueError, match="Hybrid replay"):
        build_review(suite, directory, verify_retrieval=True)


@pytest.mark.parametrize(
    ("stage", "completed"),
    [
        ("dense_index", 0),
        ("lexical_index", 0),
        ("dense_query", 1),
        ("lexical_query", 1),
    ],
)
def test_hybrid_child_failures_preserve_run_state(
    tmp_path, monkeypatch, stage, completed
):
    suite = tmp_path / "suite.json"
    suite.write_bytes(EXAMPLE.read_bytes())

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "suite_path": "suite.json",
                "retriever": "hybrid",
                "k": 2,
                "output_dir": "runs",
                "embedding": {
                    "model_id": "test/model",
                    "model_revision": "a" * 40,
                    "embedding_dimension": 2,
                    "max_sequence_length": 256,
                },
                "hybrid": {
                    "candidate_k": 5,
                    "rank_constant": 60,
                },
            }
        ),
        encoding="utf-8",
    )

    class FailingEmbedder(FixedEmbedder):
        def encode_documents(self, texts):
            if stage == "dense_index":
                raise RuntimeError("private backend detail")
            return super().encode_documents(texts)

        def encode_query(self, text):
            if stage == "dense_query" and self.query_count == 1:
                raise RuntimeError("private backend detail")
            return super().encode_query(text)

    original_bm25 = run_service.BM25Retriever

    class FailingBM25(original_bm25):
        def __init__(self, chunks):
            if stage == "lexical_index":
                raise RuntimeError("private backend detail")
            super().__init__(chunks)
            self.calls = 0

        def retrieve(self, query, *, top_k):
            self.calls += 1
            if stage == "lexical_query" and self.calls == 2:
                raise RuntimeError("private backend detail")
            return super().retrieve(query, top_k=top_k)

    embedder = FailingEmbedder()

    monkeypatch.setattr(
        run_service,
        "load_cpu_embedder",
        lambda *args, **kwargs: embedder,
    )
    monkeypatch.setattr(
        run_service,
        "collect_embedding_runtime",
        lambda model: {"schema_version": 1},
    )
    monkeypatch.setattr(run_service, "BM25Retriever", FailingBM25)

    loaded = load_retrieval_run_config(config)

    with pytest.raises(RuntimeError, match="Retrieval run failed") as caught:
        run_service.run_retrieval_benchmark(loaded)

    assert "private backend detail" not in str(caught.value)

    (directory,) = loaded.output_dir.iterdir()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "failed"
    assert manifest["completed_case_count"] == completed
    assert manifest["error_code"] == "retrieval_run_failed"
    assert not (directory / "summary.json").exists()
    assert "private backend detail" not in json.dumps(manifest)

    results = directory / "results.jsonl"
    lines = results.read_text(encoding="utf-8").splitlines() if results.exists() else []
    assert len(lines) == completed
