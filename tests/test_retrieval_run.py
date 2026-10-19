"""Local retrieval ingestion, CLI, and durable run lifecycle contracts."""

import json
from pathlib import Path

import pytest
import yaml

from llm_benchmark.cli import main
from llm_benchmark.retrieval import dataset, run_config, run_service
from llm_benchmark.retrieval.artifacts import RetrievalArtifactWriter

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
    assert captured.err == ""


def test_cli_invalid_config(config_path, capsys):
    config_path.write_text("private: secret", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["retrieval-run", "--config", str(config_path)])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: Invalid retrieval run configuration.\n"
