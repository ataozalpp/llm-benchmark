"""Comparison of recorded RAG run artifacts."""

import json
from pathlib import Path

import pytest
import yaml

from llm_benchmark.rag.run_config import load_rag_run_config
from llm_benchmark.rag.run_service import run_rag_benchmark
from scripts.compare_rag_runs import compare_runs
from scripts.review_retrieval_run import write_review

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "examples/rag/policy_qa_smoke.json"
FAILURE_CASES = ROOT / "examples/rag/policy_qa_failure_cases.json"


@pytest.fixture
def make_run(tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_bytes(SMOKE.read_bytes())
    counter = iter(range(100))

    def create(**overrides):
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
        config_path = tmp_path / f"config-{next(counter)}.yaml"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        run_dir, _ = run_rag_benchmark(load_rag_run_config(config_path))
        return run_dir

    create.suite_path = suite_path
    return create


def test_identical_configuration_reports_no_variation(make_run):
    baseline, candidate = make_run(), make_run()
    report = compare_runs(make_run.suite_path, baseline, candidate)

    assert report["varied_fields"] == []
    assert "note" in report
    assert report["answer_exact_match"]["delta"] == 0
    assert report["answer_exact_match"]["denominator"] == 4
    assert report["unanswerable_abstention"]["denominator"] == 1
    assert report["answer_changed"] == []
    assert report["baseline_run_id"] != report["candidate_run_id"]
    assert report["recorded_summaries"]["baseline"]["answer_exact_match"] == (
        report["recorded_summaries"]["candidate"]["answer_exact_match"]
    )


def test_prompt_version_is_the_only_varied_field(make_run):
    report = compare_runs(
        make_run.suite_path,
        make_run(),
        make_run(prompt_version="rag-grounded-json-v3"),
    )
    assert report["varied_fields"] == ["prompt_version"]
    assert report["candidate_settings"]["prompt_version"] == "rag-grounded-json-v3"
    assert "note" not in report


def test_retriever_change_reports_context_changes(make_run):
    report = compare_runs(
        make_run.suite_path, make_run(), make_run(retriever="keyword")
    )
    assert report["varied_fields"] == ["retriever_profile_hash"]
    assert report["context_changed"]


@pytest.mark.parametrize(
    "overrides, field",
    [({"k": 3}, "evaluation_config_hash"), ({"max_context_chars": 100}, "evaluation_config_hash")],
)
def test_incompatible_runs_are_rejected(make_run, overrides, field):
    with pytest.raises(ValueError, match=f"Incompatible runs: {field} differs"):
        compare_runs(make_run.suite_path, make_run(), make_run(**overrides))


def test_wrong_suite_file_is_rejected(make_run, tmp_path):
    other = tmp_path / "other.json"
    other.write_bytes(FAILURE_CASES.read_bytes())
    with pytest.raises(ValueError, match="suite_hash"):
        compare_runs(other, make_run(), make_run())


def test_incomplete_run_is_rejected(make_run):
    candidate = make_run()
    results = candidate / "results.jsonl"
    lines = results.read_text(encoding="utf-8").splitlines()
    results.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or reordered"):
        compare_runs(make_run.suite_path, make_run(), candidate)


def test_rows_from_another_run_are_rejected(make_run):
    baseline, candidate = make_run(), make_run()
    (candidate / "results.jsonl").write_bytes(
        (baseline / "results.jsonl").read_bytes()
    )
    with pytest.raises(ValueError, match="belongs to another run"):
        compare_runs(make_run.suite_path, baseline, candidate)


def test_unfinished_run_is_rejected(make_run):
    candidate = make_run()
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        compare_runs(make_run.suite_path, make_run(), candidate)


def test_report_is_written_once(make_run, tmp_path):
    report = compare_runs(make_run.suite_path, make_run(), make_run())
    output = tmp_path / "comparison.json"
    write_review(output, report)
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    with pytest.raises(FileExistsError):
        write_review(output, report)
