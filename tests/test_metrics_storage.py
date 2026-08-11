import json
from pathlib import Path

import pytest

from llm_benchmark.metrics import percentile, summarize
from llm_benchmark.models import BenchmarkResult
from llm_benchmark.storage import append_result


def result(status: str, latency: float, tokens: int | None = 10) -> BenchmarkResult:
    request_status = "failed" if status == "request_failed" else "succeeded"
    parse_status = "no_answer_found" if status == "unparseable" else "normalized_label"
    return BenchmarkResult(
        "run", status, "fixture", "test", "cat", "mock", "local", "model", "model", "raw", "A" if status in {"correct", "incorrect"} else None,
        "A", status == "correct", status, parse_status, request_status, "server_error" if status == "request_failed" else None,
        tokens, tokens, tokens, None, 1, latency, latency, "start", "end", "v1", "v1"
    )


def test_metrics_accuracy_tokens_and_latency() -> None:
    results = [result("correct", 10), result("incorrect", 20), result("unparseable", 30, None), result("request_failed", 40, None)]
    summary = summarize(results, 99)
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["answered_accuracy"] == pytest.approx(0.5)
    assert summary["request_success_rate"] == pytest.approx(0.75)
    assert summary["parse_success_rate"] == pytest.approx(2 / 3)
    assert summary["format_failure_rate"] == pytest.approx(1 / 3)
    assert summary["total_tokens"] == 20
    assert summary["token_usage_missing_count"] == 2
    assert summary["tokens_per_correct_answer"] == 20
    assert summary["latency_p50_ms"] == 20
    assert summary["latency_p95_ms"] == pytest.approx(29)
    assert summary["failure_latency_mean_ms"] == 40


def test_tokens_per_correct_answer_handles_zero() -> None:
    assert summarize([result("incorrect", 1)], 1)["tokens_per_correct_answer"] is None


def test_percentile_empty() -> None:
    assert percentile([], 0.95) is None


def test_append_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    append_result(path, result("correct", 1))
    append_result(path, result("incorrect", 2))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["evaluation_status"] for row in rows] == ["correct", "incorrect"]
