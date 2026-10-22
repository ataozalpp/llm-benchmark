import json
import runpy
import sys
from pathlib import Path

import pytest

from llm_benchmark.retrieval.evaluation import evaluate_retrieval
from llm_benchmark.retrieval.models import (
    RankedRetrieval,
    RetrievalEvaluationCase,
)
from llm_benchmark.retrieval.review import select_review_queries
from llm_benchmark.retrieval.run_config import load_retrieval_run_config
from llm_benchmark.retrieval.run_service import run_retrieval_benchmark


def metric(query_id, retrieved_ids, *, k=10):
    return evaluate_retrieval(
        case=RetrievalEvaluationCase(
            query_id=query_id,
            relevant_chunk_ids=frozenset({"gold"}),
        ),
        result=RankedRetrieval(
            query_id=query_id,
            chunk_ids=retrieved_ids,
        ),
        k=k,
    )


def test_groups_and_sample_limits():
    evaluations = (
        [metric(f"first-{index}", ("gold",)) for index in range(8)]
        + [metric(f"later-{index}", ("other", "gold")) for index in range(7)]
        + [metric(f"miss-{index}", ("other",)) for index in range(12)]
    )

    result = select_review_queries(evaluations)
    selected = result["selected_query_ids"]

    assert result["population_counts"] == {
        "rank_1": 8,
        "rank_2_to_k": 7,
        "miss": 12,
    }

    assert len(selected["rank_1"]) == 5
    assert len(selected["rank_2_to_k"]) == 5
    assert len(selected["miss"]) == 10

    assert all(item.startswith("first-") for item in selected["rank_1"])
    assert all(item.startswith("later-") for item in selected["rank_2_to_k"])
    assert all(item.startswith("miss-") for item in selected["miss"])
    assert result == select_review_queries(list(reversed(evaluations)))


def test_small_groups_are_not_padded():
    result = select_review_queries(
        [
            metric("first", ("gold",)),
            metric("missing", ()),
        ]
    )

    assert result["selected_query_ids"] == {
        "rank_1": ["first"],
        "rank_2_to_k": [],
        "miss": ["missing"],
    }


def test_duplicate_queries_are_rejected():
    item = metric("same", ("gold",))

    with pytest.raises(ValueError, match="unique"):
        select_review_queries([item, item])


def test_mixed_k_is_rejected():
    with pytest.raises(ValueError, match="same K"):
        select_review_queries(
            [
                metric("one", ("gold",), k=1),
                metric("two", ("gold",), k=10),
            ]
        )


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        select_review_queries([])


def test_review_run_identity_and_metric(tmp_path):
    root = Path(__file__).resolve().parents[1]

    suite_path = tmp_path / "suite.json"
    suite_path.write_bytes(
        (root / "examples/retrieval/keyword_smoke.json").read_bytes()
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "schema_version: 1\n"
        "suite_path: suite.json\n"
        "retriever: keyword\n"
        "k: 2\n"
        "output_dir: runs\n",
        encoding="utf-8",
    )

    run_dir, _ = run_retrieval_benchmark(load_retrieval_run_config(config_path))

    namespace = runpy.run_path(str(root / "scripts/review_retrieval_run.py"))
    build_review = namespace["build_review"]

    report = build_review(suite_path, run_dir)

    assert report["population_counts"] == {
        "rank_1": 3,
        "rank_2_to_k": 0,
        "miss": 1,
    }

    original_suite = suite_path.read_bytes()
    changed = json.loads(original_suite)
    changed["chunks"][0]["text"] += "changed"
    suite_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        build_review(suite_path, run_dir)

    suite_path.write_bytes(original_suite)

    results_path = run_dir / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["metrics"]["hit_at_k"] = -1

    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metrics"):
        build_review(suite_path, run_dir)


@pytest.fixture
def review_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    suite = tmp_path / "suite.json"
    suite.write_bytes((root / "examples/retrieval/keyword_smoke.json").read_bytes())
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema_version: 1\nsuite_path: suite.json\nretriever: keyword\nk: 2\noutput_dir: runs\n",
        encoding="utf-8",
    )
    directory, _ = run_retrieval_benchmark(load_retrieval_run_config(config))
    namespace = runpy.run_path(str(root / "scripts/review_retrieval_run.py"))
    return suite, directory, namespace


def test_replay_and_fingerprint(review_run):
    suite, directory, api = review_run
    normal = api["build_review"](suite, directory)
    replay = api["build_review"](suite, directory, verify_retrieval=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert normal["run_fingerprint"] == manifest["run_fingerprint"]
    assert "run_fingerpint" not in normal
    assert normal["retrieval_replay"] == "not_requested"
    assert replay["retrieval_replay"] == "matched"
    assert normal["selected_query_ids"] == replay["selected_query_ids"]


@pytest.mark.parametrize(
    "change",
    [
        lambda value: [],
        lambda value: {key: item for key, item in value.items() if key != "k"},
        lambda value: {**value, "k": True},
        lambda value: {**value, "run_id": None},
        lambda value: {**value, "completed_case_count": True},
        lambda value: {**value, "artifact_schema_version": True},
        lambda value: {**value, "status": "running"},
        lambda value: {**value, "scenario_ids": []},
    ],
)
def test_invalid_manifest(review_run, change):
    suite, directory, api = review_run
    path = directory / "manifest.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(change(original)), encoding="utf-8")
    with pytest.raises(ValueError):
        api["build_review"](suite, directory)


@pytest.mark.parametrize(
    "change",
    [
        lambda row: [],
        lambda row: {key: value for key, value in row.items() if key != "query_id"},
        lambda row: {**row, "run_id": "other-run"},
        lambda row: {**row, "schema_version": True},
        lambda row: {**row, "retrieved_chunk_ids": "refund"},
        lambda row: {**row, "retrieved_chunk_ids": [False]},
        lambda row: {**row, "retrieved_chunk_ids": ["refund", "refund"]},
        lambda row: {**row, "retrieved_chunk_ids": ["missing"]},
        lambda row: {**row, "retrieved_chunk_ids": ["refund", "account", "payment"]},
        lambda row: {**row, "metrics": []},
        lambda row: {**row, "metrics": {**row["metrics"], "hit_at_k": True}},
        lambda row: {**row, "metrics": {**row["metrics"], "retrieved_count": 1.0}},
    ],
)
def test_invalid_result(review_run, change):
    suite, directory, api = review_run
    path = directory / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0] = change(rows[0])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        api["build_review"](suite, directory)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "reordered"])
def test_query_coverage(review_run, mode):
    suite, directory, api = review_run
    path = directory / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    if mode == "missing":
        lines.pop()
    elif mode == "duplicate":
        lines.append(lines[0])
    else:
        lines.reverse()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        api["build_review"](suite, directory)


def test_replay_detects_changed_behavior(review_run, monkeypatch):
    suite, directory, api = review_run

    class ChangedRetriever:
        def __init__(self, chunks):
            pass

        def retrieve(self, query, *, top_k):
            return RankedRetrieval(query.query_id, ())

    monkeypatch.setitem(
        api["build_review"].__globals__, "KeywordRetriever", ChangedRetriever
    )
    api["build_review"](suite, directory)
    with pytest.raises(ValueError, match="replay differs"):
        api["build_review"](suite, directory, verify_retrieval=True)


def test_partial_recall_is_visible_at_rank_one(review_run):
    suite, directory, api = review_run
    payload = json.loads(suite.read_text(encoding="utf-8"))
    payload["scenarios"][0]["relevant_chunk_ids"] = ["refund", "shipping"]
    suite.write_text(json.dumps(payload), encoding="utf-8")
    directory, _ = run_retrieval_benchmark(
        load_retrieval_run_config(suite.parent / "config.yaml")
    )
    report = api["build_review"](suite, directory)
    query_id = payload["scenarios"][0]["query_id"]
    assert query_id in report["selected_query_ids"]["rank_1"]
    assert report["recall_diagnostics"] == {
        "partial_recall_count": 1,
        "partial_recall_query_ids": [query_id],
    }


@pytest.mark.parametrize("raw", ['{"k":1,"k":2}', '{"k":NaN}', "[]", "{"])
def test_invalid_json(review_run, tmp_path, raw):
    _, _, api = review_run
    path = tmp_path / "bad.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        api["read_json"](path)


def test_input_limit(review_run, monkeypatch):
    suite, directory, api = review_run
    monkeypatch.setitem(api["build_review"].__globals__, "MAX_INPUT_BYTES", 8)
    with pytest.raises(ValueError, match="size limit"):
        api["build_review"](suite, directory)


def test_writer_preserves_existing_output(review_run, tmp_path):
    _, _, api = review_run
    path = tmp_path / "review.json"
    api["write_review"](path, {"value": 1})
    with pytest.raises(FileExistsError):
        api["write_review"](path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    assert not list(tmp_path.glob(".review-*.tmp"))


def test_writer_serialization_and_size_errors_leave_no_output(
    review_run, tmp_path, monkeypatch
):
    _, _, api = review_run
    path = tmp_path / "review.json"
    with pytest.raises(ValueError):
        api["write_review"](path, {"value": float("nan")})
    monkeypatch.setitem(api["write_review"].__globals__, "MAX_REVIEW_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        api["write_review"](path, {"value": 1})
    assert not path.exists()
    assert not list(tmp_path.glob(".review-*.tmp"))


def test_failed_publication_leaves_no_output(review_run, tmp_path, monkeypatch):
    _, _, api = review_run
    path = tmp_path / "review.json"

    def fail_link(*args):
        raise OSError("storage diagnostic")

    monkeypatch.setattr(api["write_review"].__globals__["os"], "link", fail_link)
    with pytest.raises(OSError):
        api["write_review"](path, {"value": 1})
    assert not path.exists()
    assert not list(tmp_path.glob(".review-*.tmp"))


def test_cli_success_and_existing_output(review_run, monkeypatch, capsys):
    suite, directory, api = review_run
    path = directory / "custom-review.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--suite",
            str(suite),
            "--run-dir",
            str(directory),
            "--verify-retrieval",
            "--output",
            str(path),
        ],
    )
    api["main"]()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["population_counts"]["rank_1"] == 3
    assert json.loads(path.read_text(encoding="utf-8"))["retrieval_replay"] == "matched"
    before = path.read_bytes()
    with pytest.raises(SystemExit) as caught:
        api["main"]()
    assert caught.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert path.read_bytes() == before


@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_cli_failure_is_sanitized(review_run, monkeypatch, capsys, error_type):
    suite, directory, api = review_run

    def fail(*args, **kwargs):
        raise error_type("private internal details")

    monkeypatch.setitem(api["main"].__globals__, "build_review", fail)
    monkeypatch.setattr(
        sys, "argv", ["review", "--suite", str(suite), "--run-dir", str(directory)]
    )
    with pytest.raises(SystemExit) as caught:
        api["main"]()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Review failed" in captured.err
    assert "private internal details" not in captured.err
    assert not (directory / "review.json").exists()
