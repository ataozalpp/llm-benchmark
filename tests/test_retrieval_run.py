"""Local retrieval ingestion, CLI, and durable run lifecycle contracts."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llm_benchmark.cli import main
from llm_benchmark.retrieval import dataset, run_config, run_service
from llm_benchmark.retrieval.artifacts import RetrievalArtifactWriter
from llm_benchmark.retrieval.identity import fingerprint
from llm_benchmark.retrieval.local_embeddings import EmbeddingInputTooLongError

EXAMPLE = Path(__file__).parents[1] / "examples/retrieval/keyword_smoke.json"


@pytest.fixture
def config_path(tmp_path):
    (tmp_path / "suite.json").write_bytes(EXAMPLE.read_bytes())
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "suite_path": "suite.json",
                "retriever": "keyword",
                "k": 2,
                "output_dir": "runs",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_config_resolves_relative_to_file_not_cwd(config_path, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = run_config.load_retrieval_run_config(config_path)
    assert loaded.suite_path == tmp_path / "suite.json"
    assert loaded.output_dir == tmp_path / "runs"
    assert not loaded.output_dir.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("k", True),
        ("k", "2"),
        ("k", 0),
        ("k", -1),
        ("retriever", "dense"),
        ("suite_path", " "),
        ("output_dir", "a\x00b"),
        ("unknown", 1),
    ],
)
def test_invalid_config(config_path, key, value):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload[key] = value
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="^Invalid retrieval run configuration.$"):
        run_config.load_retrieval_run_config(config_path)


@pytest.mark.parametrize(
    "raw",
    [b"k: 1\nk: 2", b"!!python/object:secret {}", b"\xff", b"[]", b"", b"[broken"],
)
def test_config_parser_rejects_ambiguous_or_invalid_input(config_path, raw):
    config_path.write_bytes(raw)
    with pytest.raises(ValueError, match="Invalid retrieval run configuration"):
        run_config.load_retrieval_run_config(config_path)


def test_config_size_and_missing_file(config_path, monkeypatch):
    monkeypatch.setattr(run_config, "MAX_CONFIG_BYTES", 8)
    with pytest.raises(ValueError):
        run_config.load_retrieval_run_config(config_path)
    with pytest.raises(ValueError):
        run_config.load_retrieval_run_config(config_path.parent / "missing.yaml")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(schema_version=True),
        lambda p: p.update(schema_version=2),
        lambda p: p.update(k=2),
        lambda p: p.update(chunks=[]),
        lambda p: p.update(scenarios=[]),
        lambda p: p["chunks"].append(p["chunks"][0]),
        lambda p: p["scenarios"].append(p["scenarios"][0]),
        lambda p: p["chunks"][0].update(text=" "),
        lambda p: p["chunks"][0].update(document_id=123),
        lambda p: p["chunks"][0].update(unexpected="secret"),
        lambda p: p["chunks"][0].update(text="\ud800"),
        lambda p: p["scenarios"][0].update(relevant_chunk_ids=[]),
        lambda p: p["scenarios"][0].update(relevant_chunk_ids=["refund", "refund"]),
        lambda p: p["scenarios"][0].update(relevant_chunk_ids=["unknown"]),
        lambda p: p["scenarios"][0].update(relevant_chunk_ids=[True]),
    ],
)
def test_dataset_validation(config_path, mutation):
    path = config_path.parent / "suite.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="^Invalid retrieval suite dataset.$"):
        dataset.load_retrieval_dataset(path)


@pytest.mark.parametrize(
    "raw",
    [b'{"a":1,"a":2}', b'{"nested":{"a":1,"a":2}}', b'{"a":NaN}', b"\xff", b"[]", b"{"],
)
def test_dataset_parse_errors(config_path, raw):
    path = config_path.parent / "suite.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="Invalid retrieval suite dataset"):
        dataset.load_retrieval_dataset(path)


def test_dataset_limit(config_path, monkeypatch):
    monkeypatch.setattr(dataset, "MAX_DATASET_BYTES", 8)
    with pytest.raises(ValueError):
        dataset.load_retrieval_dataset(config_path.parent / "suite.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_artifacts_identity_and_metrics(config_path):
    loaded = run_config.load_retrieval_run_config(config_path)
    directory, summary = run_service.run_retrieval_benchmark(loaded)
    manifest = read_json(directory / "manifest.json")
    rows = [
        json.loads(line)
        for line in (directory / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert read_json(directory / "summary.json") == summary
    assert manifest["status"] == "completed"
    assert manifest["completed_case_count"] == summary["query_count"] == len(rows) == 4
    assert [r["query_id"] for r in rows] == manifest["scenario_ids"]
    assert [r["retrieved_chunk_ids"] for r in rows] == [
        ["refund"],
        ["account"],
        [],
        ["payment"],
    ]
    assert (
        summary["hit_rate_at_k"]
        == summary["mean_recall_at_k"]
        == summary["mrr_at_k"]
        == 0.75
    )
    assert summary["mean_precision_at_k"] == 0.375
    assert summary["retrieval_total_ms"] == sum(r["retrieval_latency_ms"] for r in rows)
    assert summary["index_build_ms"] >= 0
    assert summary["wall_time_ms"] >= summary["retrieval_total_ms"]
    again, _ = run_service.run_retrieval_benchmark(loaded)
    assert again != directory
    assert (
        read_json(again / "manifest.json")["run_fingerprint"]
        == manifest["run_fingerprint"]
    )
    for path in directory.iterdir():
        assert "refund period thirty days" not in path.read_text(encoding="utf-8")


def test_preflight_failure_creates_no_run(config_path):
    loaded = run_config.load_retrieval_run_config(config_path)
    loaded.suite_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        run_service.run_retrieval_benchmark(loaded)
    assert not loaded.output_dir.exists()


def test_partial_failure_preserves_completed_results(config_path, monkeypatch):
    original = run_service.KeywordRetriever

    class FailsOnSecond(original):
        def retrieve(self, query, *, top_k):
            if query.query_id == "q-account":
                raise RuntimeError("private query secret")
            return super().retrieve(query, top_k=top_k)

    monkeypatch.setattr(run_service, "KeywordRetriever", FailsOnSecond)
    loaded = run_config.load_retrieval_run_config(config_path)
    with pytest.raises(RuntimeError, match="Retrieval run failed") as caught:
        run_service.run_retrieval_benchmark(loaded)
    assert "secret" not in str(caught.value)
    (directory,) = loaded.output_dir.iterdir()
    manifest = read_json(directory / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["completed_case_count"] == 1
    assert not (directory / "summary.json").exists()
    assert (
        len((directory / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    )
    assert "secret" not in json.dumps(manifest)


def test_storage_failure_is_not_a_success(config_path, monkeypatch):
    def fail(self, value):
        raise OSError("private storage path")

    monkeypatch.setattr(RetrievalArtifactWriter, "append_result", fail)
    loaded = run_config.load_retrieval_run_config(config_path)
    with pytest.raises(RuntimeError, match="^Retrieval run failed; run ID:"):
        run_service.run_retrieval_benchmark(loaded)
    (directory,) = loaded.output_dir.iterdir()
    assert read_json(directory / "manifest.json")["status"] == "failed"
    assert not (directory / "summary.json").exists()


def test_artifact_writer_preserves_existing_json_and_lines(tmp_path):
    writer = RetrievalArtifactWriter(tmp_path / "run")
    writer.write_json("manifest.json", {"status": "running"})
    writer.append_result({"query_id": "one"})
    with pytest.raises(ValueError):
        writer.write_json("manifest.json", {"bad": float("nan")})
    with pytest.raises(ValueError):
        writer.append_result({"bad": float("nan")})
    assert read_json(writer.directory / "manifest.json") == {"status": "running"}
    assert (writer.directory / "results.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 1
    with pytest.raises(ValueError):
        writer.write_json("../escape.json", {})
    with pytest.raises(FileExistsError):
        RetrievalArtifactWriter(writer.directory)


def test_cli(config_path, capsys):
    main(["retrieval-run", "--config", str(config_path)])
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response["overall"]["hit_rate_at_k"] == 0.75
    assert Path(response["output_dir"]).is_dir()
    assert captured.err == "[retrieval] Completed 4/4 queries\n"


def test_cli_invalid_config(config_path, capsys):
    config_path.write_text("private: secret", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["retrieval-run", "--config", str(config_path)])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: Invalid retrieval run configuration.\n"


@pytest.fixture
def dense_config_path(config_path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload.update(
        retriever="dense",
        embedding={
            "model_id": "test/encoder",
            "model_revision": "a" * 40,
            "embedding_dimension": 2,
            "max_sequence_length": 256,
            "batch_size": 8,
            "local_files_only": True,
        },
    )
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


@pytest.fixture(autouse=True)
def fake_embedding_runtime(monkeypatch):
    calls = []
    metadata = {
        "schema_version": 1,
        "runtime_versions": {"test-backend": "1"},
    }

    def collect(model):
        calls.append(model)
        return metadata.copy()

    monkeypatch.setattr(run_service, "collect_embedding_runtime", collect)

    return calls


class FixedEmbedder:
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0
        self.model = object()

    def encode_documents(self, texts):
        self.document_calls += 1
        assert len(texts) == 4
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

    def encode_query(self, text):
        self.query_calls += 1
        return {
            "refund period": (1.0, 0.0),
            "password reset": (-1.0, 0.0),
            "delivery": (0.0, 1.0),
            "invoice": (0.0, -1.0),
        }[text]

    def input_statistics(self):
        return {
            "schema_version": 1,
            "encoded_document_count": self.document_calls * 4,
            "over_limit_document_count": 0,
            "encoded_query_count": self.query_calls,
            "over_limit_query_count": 0,
        }


def read_results(directory):
    path = directory / "results.jsonl"
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )


@pytest.mark.parametrize("offline", [True, False])
def test_dense_artifacts_use_dense_profile_and_load_once(
    dense_config_path, monkeypatch, offline, fake_embedding_runtime
):
    payload = yaml.safe_load(dense_config_path.read_text(encoding="utf-8"))
    payload["embedding"]["local_files_only"] = offline
    dense_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = run_config.load_retrieval_run_config(dense_config_path)
    embedder = FixedEmbedder()
    calls = []

    def load(profile, *, local_files_only):
        calls.append((profile, local_files_only))
        (directory,) = loaded.output_dir.iterdir()
        assert read_json(directory / "manifest.json")["status"] == "running"
        return embedder

    def forbid_keyword(*args, **kwargs):
        pytest.fail("Dense execution must not use KeywordRetriever.")

    monkeypatch.setattr(run_service, "load_cpu_embedder", load)
    monkeypatch.setattr(run_service, "KeywordRetriever", forbid_keyword)
    directory, summary = run_service.run_retrieval_benchmark(loaded)
    manifest = read_json(directory / "manifest.json")
    rows = read_results(directory)
    profile = loaded.config.embedding.to_profile()

    assert calls == [(profile, offline)]
    assert fake_embedding_runtime == [embedder.model]
    assert manifest["embedding_runtime"] == {
        "schema_version": 1,
        "runtime_versions": {"test-backend": "1"},
    }
    assert manifest["embedding_load_policy"] == {"local_files_only": offline}
    assert embedder.document_calls == 1
    assert embedder.query_calls == 4
    assert manifest["retriever_profile"] == asdict(profile)
    assert manifest["retriever_profile_hash"] == fingerprint(asdict(profile))
    assert manifest["status"] == "completed"
    assert manifest["completed_case_count"] == len(rows) == 4
    assert [row["query_id"] for row in rows] == manifest["scenario_ids"]
    assert [row["retrieved_chunk_ids"][0] for row in rows] == [
        "refund",
        "account",
        "shipping",
        "payment",
    ]
    assert all(row["run_id"] == manifest["run_id"] for row in rows)
    assert summary["hit_rate_at_k"] == summary["mrr_at_k"] == 1.0
    assert summary["mean_precision_at_k"] == 0.5
    assert read_json(directory / "summary.json") == summary
    assert manifest["embedding_input_statistics"] == {
        "schema_version": 1,
        "encoded_document_count": 4,
        "over_limit_document_count": 0,
        "encoded_query_count": 4,
        "over_limit_query_count": 0,
    }


def test_keyword_does_not_load_embedding_model(
    config_path, monkeypatch, fake_embedding_runtime
):
    def forbidden(*args, **kwargs):
        pytest.fail("Keyword runs must not load an embedding model.")

    monkeypatch.setattr(run_service, "load_cpu_embedder", forbidden)
    directory, summary = run_service.run_retrieval_benchmark(
        run_config.load_retrieval_run_config(config_path)
    )
    assert summary["model_load_ms"] == 0.0
    assert summary["hit_rate_at_k"] == 0.75
    assert fake_embedding_runtime == []
    manifest = read_json(directory / "manifest.json")
    assert "embedding_runtime" not in manifest
    assert "embedding_load_policy" not in manifest
    assert "embedding_input_statistics" not in manifest


@pytest.mark.parametrize("stage,completed", [("load", 0), ("index", 0), ("query", 1)])
def test_dense_failures_preserve_lifecycle(
    dense_config_path, monkeypatch, stage, completed
):
    class FailingEmbedder(FixedEmbedder):
        def encode_documents(self, texts):
            if stage == "index":
                raise EmbeddingInputTooLongError("private backend secret")
            return super().encode_documents(texts)

        def encode_query(self, text):
            if stage == "query" and self.query_calls == 1:
                raise RuntimeError("private backend secret")
            return super().encode_query(text)

    def load(*args, **kwargs):
        if stage == "load":
            raise OSError("private backend secret")
        return FailingEmbedder()

    monkeypatch.setattr(run_service, "load_cpu_embedder", load)
    loaded = run_config.load_retrieval_run_config(dense_config_path)
    with pytest.raises(RuntimeError, match="Retrieval run failed") as caught:
        run_service.run_retrieval_benchmark(loaded)

    (directory,) = loaded.output_dir.iterdir()
    manifest = read_json(directory / "manifest.json")
    if stage == "load":
        assert "embedding_input_statistics" not in manifest
    else:
        assert manifest["embedding_input_statistics"] == {
            "schema_version": 1,
            "encoded_document_count": 0 if stage == "index" else 4,
            "over_limit_document_count": 0,
            "encoded_query_count": completed,
            "over_limit_query_count": 0,
        }
    assert manifest["status"] == "failed"
    assert manifest["error_code"] == "retrieval_run_failed"
    assert manifest["completed_case_count"] == completed
    if stage == "load":
        assert "embedding_runtime" not in manifest
    else:
        assert manifest["embedding_runtime"]["runtime_versions"] == {
            "test-backend": "1"
        }
        assert manifest["embedding_load_policy"] == {"local_files_only": True}
    assert len(read_results(directory)) == completed
    assert not (directory / "summary.json").exists()
    assert "secret" not in str(caught.value)
    assert all(
        "secret" not in path.read_text(encoding="utf-8") for path in directory.iterdir()
    )


def test_dense_timing_separates_loading_index_and_queries(
    dense_config_path, monkeypatch
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        run_service, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )

    class TimedEmbedder(FixedEmbedder):
        def encode_documents(self, texts):
            clock.now += 3.0
            return super().encode_documents(texts)

        def encode_query(self, text):
            clock.now += 0.25
            return super().encode_query(text)

    def load(*args, **kwargs):
        clock.now += 2.0
        return TimedEmbedder()

    def collect(model):
        clock.now += 0.5
        return {"schema_version": 1}

    monkeypatch.setattr(run_service, "collect_embedding_runtime", collect)
    monkeypatch.setattr(run_service, "load_cpu_embedder", load)
    directory, summary = run_service.run_retrieval_benchmark(
        run_config.load_retrieval_run_config(dense_config_path)
    )
    assert summary["model_load_ms"] == 2000.0
    assert summary["index_build_ms"] == 3000.0
    assert summary["retrieval_total_ms"] == 1000.0
    assert summary["retrieval_mean_ms"] == 250.0
    assert summary["wall_time_ms"] == 6500.0
    assert [row["retrieval_latency_ms"] for row in read_results(directory)] == [
        250.0
    ] * 4


def test_dense_revision_changes_run_identity(dense_config_path, monkeypatch):
    monkeypatch.setattr(
        run_service, "load_cpu_embedder", lambda *a, **k: FixedEmbedder()
    )
    first, _ = run_service.run_retrieval_benchmark(
        run_config.load_retrieval_run_config(dense_config_path)
    )
    payload = yaml.safe_load(dense_config_path.read_text(encoding="utf-8"))
    payload["embedding"]["model_revision"] = "b" * 40
    dense_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    second, _ = run_service.run_retrieval_benchmark(
        run_config.load_retrieval_run_config(dense_config_path)
    )
    left, right = (
        read_json(first / "manifest.json"),
        read_json(second / "manifest.json"),
    )
    assert left["corpus_hash"] == right["corpus_hash"]
    assert left["suite_hash"] == right["suite_hash"]
    assert left["retriever_profile_hash"] != right["retriever_profile_hash"]
    assert left["run_fingerprint"] != right["run_fingerprint"]


def test_dense_cli_success(dense_config_path, monkeypatch, capsys):
    monkeypatch.setattr(
        run_service, "load_cpu_embedder", lambda *a, **k: FixedEmbedder()
    )
    main(["retrieval-run", "--config", str(dense_config_path)])
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    manifest = read_json(Path(response["output_dir"]) / "manifest.json")
    assert response["run_id"] == manifest["run_id"]
    assert response["overall"]["hit_rate_at_k"] == 1.0
    assert manifest["retriever_profile"]["name"] == "dense-exact-cosine"
    assert captured.err == "[retrieval] Completed 4/4 queries\n"
    assert manifest["embedding_runtime"] == {
        "schema_version": 1,
        "runtime_versions": {"test-backend": "1"},
    }
    assert manifest["embedding_load_policy"] == {
        "local_files_only": True,
    }


def test_dense_cli_load_failure_exits_two(dense_config_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("private cache secret")

    monkeypatch.setattr(run_service, "load_cpu_embedder", fail)
    with pytest.raises(SystemExit) as caught:
        main(["retrieval-run", "--config", str(dense_config_path)])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Retrieval run failed" in captured.err
    assert "secret" not in captured.err


def test_runtime_is_persisted_before_index_encoding(dense_config_path, monkeypatch):
    loaded = run_config.load_retrieval_run_config(dense_config_path)

    class InspectingEmbedder(FixedEmbedder):
        def encode_documents(self, texts):
            (directory,) = loaded.output_dir.iterdir()
            manifest = read_json(directory / "manifest.json")
            assert manifest["status"] == "running"
            assert manifest["completed_case_count"] == 0
            assert manifest["embedding_runtime"]["runtime_versions"] == {
                "test-backend": "1"
            }
            assert manifest["embedding_load_policy"] == {"local_files_only": True}
            return super().encode_documents(texts)

    monkeypatch.setattr(
        run_service, "load_cpu_embedder", lambda *a, **k: InspectingEmbedder()
    )
    run_service.run_retrieval_benchmark(loaded)


def test_runtime_collection_failure_stops_before_encoding(
    dense_config_path, monkeypatch
):
    embedder = FixedEmbedder()
    monkeypatch.setattr(run_service, "load_cpu_embedder", lambda *a, **k: embedder)

    def fail(model):
        assert model is embedder.model
        raise RuntimeError("private runtime secret")

    monkeypatch.setattr(run_service, "collect_embedding_runtime", fail)
    loaded = run_config.load_retrieval_run_config(dense_config_path)
    with pytest.raises(RuntimeError, match="Retrieval run failed") as caught:
        run_service.run_retrieval_benchmark(loaded)

    (directory,) = loaded.output_dir.iterdir()
    manifest = read_json(directory / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["completed_case_count"] == 0
    assert "embedding_runtime" not in manifest
    assert embedder.document_calls == embedder.query_calls == 0
    assert read_results(directory) == []
    assert not (directory / "summary.json").exists()
    assert "secret" not in str(caught.value)
    assert "secret" not in json.dumps(manifest)


def test_runtime_manifest_write_failure_stops_before_encoding(
    dense_config_path, monkeypatch
):
    embedder = FixedEmbedder()
    monkeypatch.setattr(run_service, "load_cpu_embedder", lambda *a, **k: embedder)
    original = RetrievalArtifactWriter.write_json
    failed_writes = []

    def fail_runtime_write(self, filename, value):
        if (
            filename == "manifest.json"
            and value.get("status") == "running"
            and "embedding_runtime" in value
        ):
            failed_writes.append(filename)
            raise OSError("private storage secret")
        return original(self, filename, value)

    monkeypatch.setattr(RetrievalArtifactWriter, "write_json", fail_runtime_write)
    loaded = run_config.load_retrieval_run_config(dense_config_path)
    with pytest.raises(RuntimeError, match="Retrieval run failed"):
        run_service.run_retrieval_benchmark(loaded)

    (directory,) = loaded.output_dir.iterdir()
    manifest = read_json(directory / "manifest.json")
    assert failed_writes == ["manifest.json"]
    assert manifest["status"] == "failed"
    assert manifest["completed_case_count"] == 0
    assert "embedding_runtime" in manifest
    assert embedder.document_calls == embedder.query_calls == 0
    assert not (directory / "summary.json").exists()
    assert "secret" not in json.dumps(manifest)


def test_truncate_policy_and_statistics_reach_manifest(
    dense_config_path, monkeypatch
):
    payload = yaml.safe_load(
        dense_config_path.read_text(encoding="utf-8")
    )
    payload["embedding"]["truncation_policy"] = "truncate"
    dense_config_path.write_text(
        yaml.safe_dump(payload),
        encoding="utf-8",
    )

    class TruncatingEmbedder(FixedEmbedder):
        def input_statistics(self):
            return {
                "schema_version": 1,
                "encoded_document_count": self. document_calls * 4,
                "over_limit_document_count": self.document_calls * 2,
                "encoded_query_count": self.query_calls,
                "over_limit_query_count": min(self.query_calls, 1),
            }

    def load(profile, *, local_files_only):
        assert profile.truncation_policy == "truncate"
        assert local_files_only is True
        return TruncatingEmbedder()

    monkeypatch.setattr(run_service, "load_cpu_embedder", load)

    directory, _ = run_service.run_retrieval_benchmark(
        run_config.load_retrieval_run_config(dense_config_path)
    )
    manifest = read_json(directory / "manifest.json")

    assert manifest["status"] == "completed"
    assert manifest["retriever_profile"]["truncation_policy"] == "truncate"
    assert manifest["embedding_input_statistics"] == {
        "schema_version": 1,
        "encoded_document_count": 4,
        "over_limit_document_count": 2,
        "encoded_query_count": 4,
        "over_limit_query_count": 1,
    }