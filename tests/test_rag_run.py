"""RAG CLI and durable run lifecycle contracts."""

import json
from pathlib import Path

import pytest
import yaml

from llm_benchmark.cli import main
from llm_benchmark.rag import run_service
from llm_benchmark.rag.artifacts import RagArtifactWriter
from llm_benchmark.rag.run_config import load_rag_run_config
from llm_benchmark.retrieval import factory

EXAMPLE = Path(__file__).parents[1] / "examples/rag/policy_qa_smoke.json"
SCENARIO_IDS = ["q-refund", "q-shipping", "q-warranty", "q-password", "q-gift-card"]
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
    "embedding_dimension": 5,
    "max_sequence_length": 256,
    "local_files_only": True,
}
HYBRID = {"candidate_k": 5, "rank_constant": 60}
VOCABULARY = ("refund", "shipping", "warranty", "password", "gift")


def write_config(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "suite.json").write_bytes(EXAMPLE.read_bytes())
    payload = {
        "schema_version": 1,
        "suite_path": "suite.json",
        "retriever": "bm25",
        "k": 2,
        "max_context_chars": 2000,
        "provider": "synthetic",
        "output_dir": "runs",
    }
    payload.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def run(tmp_path, transport=None, **overrides):
    loaded = load_rag_run_config(write_config(tmp_path, **overrides))
    return run_service.run_rag_benchmark(loaded, transport=transport)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def only_run_dir(tmp_path):
    (directory,) = (tmp_path / "runs").iterdir()
    return directory


def chat_response(content, *, finish_reason="stop", usage=None):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage if usage is not None else {
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    }


ABSTAIN = json.dumps({"answer": "", "citations": [], "abstained": True})


class FakeTransport:
    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.calls = []

    def post_json(self, url, payload, timeout_seconds, headers=None):
        self.calls.append((url, payload, timeout_seconds))
        question = json.loads(payload["messages"][-1]["content"])["question"]
        output = self.outputs.get(question, chat_response(ABSTAIN))
        if isinstance(output, Exception):
            raise output
        return output


class VocabularyEmbedder:
    """Deterministic test vectors; a bias dimension avoids zero vectors."""

    def __init__(self):
        self.model = object()
        self.documents = 0
        self.queries = 0

    def _vector(self, text):
        lowered = text.casefold()
        return tuple(float(word in lowered) for word in VOCABULARY) + (0.01,)

    def encode_documents(self, texts):
        self.documents += len(texts)
        return tuple(self._vector(text) for text in texts)

    def encode_query(self, text):
        self.queries += 1
        return self._vector(text)

    def input_statistics(self):
        return {
            "schema_version": 1,
            "encoded_document_count": self.documents,
            "over_limit_document_count": 0,
            "encoded_query_count": self.queries,
            "over_limit_query_count": 0,
        }


@pytest.fixture(autouse=True)
def fake_embedding_runtime(monkeypatch):
    monkeypatch.setattr(
        run_service,
        "collect_embedding_runtime",
        lambda model: {"schema_version": 1, "runtime_versions": {"test": "1"}},
    )


@pytest.fixture
def embedder_loads(monkeypatch):
    calls = []

    def load(profile, *, local_files_only):
        calls.append((profile, local_files_only))
        return VocabularyEmbedder()

    monkeypatch.setattr(factory, "load_cpu_embedder", load)
    return calls


@pytest.fixture
def forbid_embedder(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Lexical runs must not load an embedding model.")

    monkeypatch.setattr(factory, "load_cpu_embedder", forbidden)


def test_synthetic_run_writes_complete_artifacts(tmp_path, forbid_embedder, capsys):
    run_dir, summary = run(tmp_path)

    manifest = read_json(run_dir / "manifest.json")
    rows = read_jsonl(run_dir / "results.jsonl")

    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name == manifest["run_id"] == summary["run_id"]
    assert manifest["status"] == "completed"
    assert manifest["completed_at"] >= manifest["started_at"]
    assert manifest["scenario_ids"] == SCENARIO_IDS
    assert [row["query_id"] for row in rows] == SCENARIO_IDS
    assert manifest["completed_case_count"] == len(rows) == summary["case_count"] == 5
    assert manifest["suite_id"] == "policy-qa-smoke-v1"
    assert manifest["prompt_version"] == "rag-grounded-json-v1"
    assert manifest["generation"] == {"provider": "synthetic"}
    assert manifest["retriever_profile"]["name"] == "bm25-rank-bm25-okapi"
    assert "embedding_runtime" not in manifest
    assert summary["status_counts"]["completed"] == 5
    assert summary["token_usage"]["input_tokens"]["measured_case_count"] == 0
    assert summary["model_load_ms"] == 0.0
    assert read_json(run_dir / "summary.json") == summary
    assert not list(run_dir.glob("*.tmp"))
    assert capsys.readouterr().err == "[rag] Completed 5/5 queries\n"


def test_rows_record_retrieval_context_and_latencies(tmp_path):
    run_dir, _ = run(tmp_path)
    rows = read_jsonl(run_dir / "results.jsonl")
    for row in rows:
        assert row["run_id"] == run_dir.name
        assert len(row["retrieved_chunk_ids"]) <= 2
        assert set(row["context_chunk_ids"]) <= set(row["retrieved_chunk_ids"])
        assert row["retrieval_latency_ms"] >= 0
        assert row["generation_latency_ms"] >= 0
    shipping = rows[1]
    assert shipping["retrieved_chunk_ids"] == ["shipping"]
    assert shipping["evaluation"]["context_evidence_recall"] == 1.0


def test_bm25_keyword_distractor_is_a_measured_retrieval_miss(tmp_path):
    # BM25 does not stem "Refunds"; the distractor matches the "refund" token.
    run_dir, summary = run(tmp_path)
    refund = read_jsonl(run_dir / "results.jsonl")[0]
    assert refund["query_id"] == "q-refund"
    assert refund["retrieved_chunk_ids"] == ["refund-history", "password"]
    assert refund["status"] == "completed"
    assert refund["evaluation"]["context_evidence_recall"] == 0.0
    assert refund["answer"]["citations"] == ["refund-history"]
    assert summary["context_evidence_recall"]["mean"] == 0.75


def test_small_context_budget_is_recorded(tmp_path):
    run_dir, summary = run(tmp_path, max_context_chars=2)
    rows = read_jsonl(run_dir / "results.jsonl")
    assert all(row["context_chunk_ids"] == [] for row in rows)
    assert all(row["answer"]["abstained"] for row in rows)
    assert summary["context_evidence_recall"]["mean"] == 0.0
    assert summary["unanswerable_abstention"]["rate"] == 1.0
    assert summary["answer_exact_match"]["rate"] == 0.0


def test_same_inputs_produce_same_fingerprint(tmp_path):
    first, _ = run(tmp_path)
    second, _ = run(tmp_path)
    one, two = read_json(first / "manifest.json"), read_json(second / "manifest.json")
    assert one["run_fingerprint"] == two["run_fingerprint"]
    assert one["run_id"] != two["run_id"]


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"k": 3}, "evaluation_config_hash"),
        ({"max_context_chars": 100}, "evaluation_config_hash"),
        ({"retriever": "keyword"}, "retriever_profile_hash"),
    ],
)
def test_settings_change_fingerprint(tmp_path, overrides, field):
    base, _ = run(tmp_path / "a")
    other, _ = run(tmp_path / "b", **overrides)
    one, two = read_json(base / "manifest.json"), read_json(other / "manifest.json")
    assert one[field] != two[field]
    assert one["run_fingerprint"] != two["run_fingerprint"]
    assert one["corpus_hash"] == two["corpus_hash"]
    assert one["suite_hash"] == two["suite_hash"]


def test_openai_compatible_run_uses_injected_transport(tmp_path):
    transport = FakeTransport(
        {
            "How long is the refund period?": chat_response(
                json.dumps(
                    {"answer": "30 days", "citations": ["refund"], "abstained": False}
                )
            )
        }
    )
    run_dir, summary = run(
        tmp_path, transport, provider="openai_compatible", model=MODEL
    )

    assert len(transport.calls) == 5
    url, payload, timeout = transport.calls[0]
    assert url == "http://127.0.0.1:1234/v1/chat/completions"
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.0
    assert timeout == 5.0
    assert summary["answer_exact_match"]["numerator"] == 1
    assert summary["unanswerable_abstention"]["numerator"] == 1
    assert summary["token_usage"]["input_tokens"]["observed_sum"] == 50
    assert summary["token_usage"]["output_tokens"]["observed_sum"] == 25

    manifest = read_json(run_dir / "manifest.json")
    assert manifest["generation"] == {
        "provider": "openai_compatible",
        "endpoint_alias": "local",
        "model_id": "test-model",
        "temperature": 0.0,
        "max_tokens": 128,
    }
    assert "127.0.0.1" not in json.dumps(manifest)


def test_requests_never_include_gold_labels(tmp_path):
    transport = FakeTransport()
    run(tmp_path, transport, provider="openai_compatible", model=MODEL)
    for _, payload, _ in transport.calls:
        user = json.loads(payload["messages"][-1]["content"])
        assert set(user) == {"question", "context"}
    encoded = json.dumps([payload for _, payload, _ in transport.calls])
    for forbidden in ("reference_answers", "relevant_chunk_ids", "answerable"):
        assert forbidden not in encoded


def test_case_level_failures_do_not_abort_the_run(tmp_path):
    transport = FakeTransport(
        {
            "How long is the refund period?": OSError("connection refused"),
            "How long does standard shipping take?": chat_response("not json"),
            "How long is the electronics warranty?": chat_response(
                "", finish_reason="length"
            ),
        }
    )
    run_dir, summary = run(
        tmp_path, transport, provider="openai_compatible", model=MODEL
    )
    rows = {row["query_id"]: row for row in read_jsonl(run_dir / "results.jsonl")}

    assert read_json(run_dir / "manifest.json")["status"] == "completed"
    assert rows["q-refund"]["status"] == "provider_failed"
    assert rows["q-refund"]["response"] is None
    assert rows["q-shipping"]["status"] == "invalid_response"
    assert rows["q-shipping"]["response"]["text"] == "not json"
    assert rows["q-warranty"]["status"] == "incomplete_response"
    assert summary["status_counts"] == {
        "completed": 2,
        "invalid_response": 1,
        "incomplete_response": 1,
        "provider_failed": 1,
    }
    assert summary["answer_exact_match"]["denominator"] == 4
    assert "connection refused" not in json.dumps(list(rows.values()))


def test_invalid_base_url_is_rejected_before_artifacts(tmp_path):
    with pytest.raises(ValueError):
        run(
            tmp_path,
            provider="openai_compatible",
            model={**MODEL, "base_url": "http://127.0.0.1:1234/v1/chat/completions"},
        )
    assert not (tmp_path / "runs").exists()


def test_dense_run_records_embedding_runtime(tmp_path, embedder_loads):
    run_dir, summary = run(tmp_path, retriever="dense", embedding=EMBEDDING)
    manifest = read_json(run_dir / "manifest.json")
    rows = read_jsonl(run_dir / "results.jsonl")

    assert len(embedder_loads) == 1
    assert embedder_loads[0][1] is True
    assert manifest["status"] == "completed"
    assert manifest["retriever_profile"]["model_id"] == "test/encoder"
    assert manifest["embedding_runtime"]["schema_version"] == 1
    assert manifest["embedding_load_policy"] == {"local_files_only": True}
    assert manifest["embedding_input_statistics"]["encoded_document_count"] == 5
    assert manifest["embedding_input_statistics"]["encoded_query_count"] == 5
    assert rows[0]["retrieved_chunk_ids"][0] == "refund"
    assert summary["model_load_ms"] >= 0


def test_hybrid_run_uses_fusion_profile(tmp_path, embedder_loads):
    run_dir, summary = run(
        tmp_path, retriever="hybrid", embedding=EMBEDDING, hybrid=HYBRID
    )
    manifest = read_json(run_dir / "manifest.json")
    rows = {row["query_id"]: row for row in read_jsonl(run_dir / "results.jsonl")}

    assert len(embedder_loads) == 1
    assert manifest["retriever_profile"]["name"] == "hybrid-bm25-dense-rrf"
    assert manifest["retriever_profile"]["candidate_k"] == 5
    assert manifest["retriever_profile"]["dense"]["model_id"] == "test/encoder"
    assert rows["q-warranty"]["retrieved_chunk_ids"][0] == "warranty"
    # The distractor appears in both rankings, so equal-weight fusion keeps it
    # ahead of the gold chunk that only the dense ranking found.
    assert rows["q-refund"]["retrieved_chunk_ids"] == ["refund-history", "password"]
    assert summary["context_evidence_recall"]["mean"] == 0.75


def test_dense_and_hybrid_have_distinct_fingerprints(tmp_path, embedder_loads):
    dense, _ = run(tmp_path / "dense", retriever="dense", embedding=EMBEDDING)
    hybrid, _ = run(
        tmp_path / "hybrid", retriever="hybrid", embedding=EMBEDDING, hybrid=HYBRID
    )
    one = read_json(dense / "manifest.json")
    two = read_json(hybrid / "manifest.json")
    assert one["retriever_profile_hash"] != two["retriever_profile_hash"]


def test_embedding_load_failure_marks_manifest_failed(tmp_path, monkeypatch):
    def fail(profile, *, local_files_only):
        raise OSError("model files are not cached")

    monkeypatch.setattr(factory, "load_cpu_embedder", fail)
    with pytest.raises(RuntimeError, match="^RAG run failed; run ID: [0-9a-f]{32}.$"):
        run(tmp_path, retriever="dense", embedding=EMBEDDING)

    run_dir = only_run_dir(tmp_path)
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["error_code"] == "rag_run_failed"
    assert manifest["completed_case_count"] == 0
    assert not (run_dir / "summary.json").exists()


def test_runtime_is_persisted_before_indexing_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        factory, "load_cpu_embedder", lambda *a, **k: VocabularyEmbedder()
    )

    def fail_index(*args, **kwargs):
        raise ValueError("index failed")

    monkeypatch.setattr(factory, "DenseRetriever", fail_index)
    with pytest.raises(RuntimeError):
        run(tmp_path, retriever="dense", embedding=EMBEDDING)

    manifest = read_json(only_run_dir(tmp_path) / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["embedding_runtime"]["schema_version"] == 1


def test_failed_suite_marks_manifest_failed(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise OSError("disk")

    monkeypatch.setattr(run_service, "run_rag_suite", broken)
    with pytest.raises(RuntimeError, match="^RAG run failed; run ID: [0-9a-f]{32}.$"):
        run(tmp_path)

    manifest = read_json(only_run_dir(tmp_path) / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["error_code"] == "rag_run_failed"


def test_result_write_failure_stops_further_generation(tmp_path, monkeypatch):
    transport = FakeTransport()

    def fail(self, value):
        raise OSError("disk full")

    monkeypatch.setattr(RagArtifactWriter, "append_result", fail)
    with pytest.raises(RuntimeError):
        run(tmp_path, transport, provider="openai_compatible", model=MODEL)

    assert len(transport.calls) == 1
    manifest = read_json(only_run_dir(tmp_path) / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["completed_case_count"] == 0


def test_partial_results_survive_a_later_failure(tmp_path, monkeypatch):
    original = RagArtifactWriter.append_result
    calls = []

    def fail_second(self, value):
        calls.append(value["query_id"])
        if len(calls) == 2:
            raise OSError("disk full")
        original(self, value)

    monkeypatch.setattr(RagArtifactWriter, "append_result", fail_second)
    with pytest.raises(RuntimeError):
        run(tmp_path)

    run_dir = only_run_dir(tmp_path)
    rows = read_jsonl(run_dir / "results.jsonl")
    assert [row["query_id"] for row in rows] == ["q-refund"]
    assert read_json(run_dir / "manifest.json")["completed_case_count"] == 1


def test_keyboard_interrupt_is_recorded_and_reraised(tmp_path, monkeypatch):
    def interrupted(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_service, "run_rag_suite", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path)

    manifest = read_json(only_run_dir(tmp_path) / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["error_code"] == "rag_run_interrupted"


def test_invalid_suite_creates_no_artifacts(tmp_path):
    path = write_config(tmp_path)
    (tmp_path / "suite.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="^Invalid RAG suite dataset.$"):
        run_service.run_rag_benchmark(load_rag_run_config(path))
    assert not (tmp_path / "runs").exists()


def test_cli(tmp_path, capsys):
    main(["rag-run", "--config", str(write_config(tmp_path))])
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert Path(response["output_dir"]).is_dir()
    assert response["overall"]["case_count"] == 5
    assert captured.err == "[rag] Completed 5/5 queries\n"


def test_cli_invalid_config(tmp_path, capsys):
    path = write_config(tmp_path)
    path.write_text("private: secret", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["rag-run", "--config", str(path)])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: Invalid RAG run configuration.\n"


def test_cli_run_failure_exits_with_two(tmp_path, capsys, monkeypatch):
    def broken(**kwargs):
        raise OSError("disk")

    monkeypatch.setattr(run_service, "run_rag_suite", broken)
    with pytest.raises(SystemExit) as caught:
        main(["rag-run", "--config", str(write_config(tmp_path))])
    assert caught.value.code == 2
    assert "error: RAG run failed; run ID:" in capsys.readouterr().err
