import json
from pathlib import Path

import pytest
import yaml

from llm_benchmark.retrieval.run_config import load_retrieval_run_config
from llm_benchmark.retrieval.run_service import run_retrieval_benchmark
from scripts.compare_retrieval_runs import compare_runs
from scripts.review_retrieval_run import write_review

EXAMPLE = (
    Path(__file__).parents[1]
    / "examples"
    / "retrieval"
    / "keyword_smoke.json"
)


@pytest.fixture
def make_run(tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_bytes(EXAMPLE.read_bytes())

    def create(*, k=2):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "suite_path": "suite.json",
                    "retriever": "keyword",
                    "k": k,
                    "output_dir": "runs",
                }
            ),
            encoding="utf-8",
        )

        directory, _ = run_retrieval_benchmark(load_retrieval_run_config(config_path))
        return suite_path, directory

    return create


def test_matching_runs_compare_without_changing_source_files(make_run):
    suite, baseline = make_run()
    _, candidate = make_run()

    source_paths = [
        directory / filename
        for directory in (baseline, candidate)
        for filename in ("manifest.json", "results.jsonl", "summary.json")
    ]
    before = {path: path.read_bytes() for path in source_paths}

    report = compare_runs(suite, baseline, candidate)

    assert report["counts"] == {
        "both_hit": 3,
        "baseline_only": 0,
        "candidate_only": 0,
        "both_miss": 1,
    }
    assert report["hit_rate_delta"] == 0.0
    assert report["retrieval_replay"] == "not_requested"

    assert {
        path: path.read_bytes()
        for path in source_paths
    } == before


def test_different_k_is_rejected(make_run):
    suite, baseline = make_run(k=1)
    _, candidate = make_run(k=2)

    with pytest.raises(ValueError, match="Incompatible runs"):
        compare_runs(suite, baseline, candidate)


def test_changed_stored_metric_is_rejected(make_run):
    suite, baseline = make_run()
    _, candidate = make_run()

    results_path = candidate / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]

    original = rows[0]["metrics"]["hit_at_k"]
    rows[0]["metrics"]["hit_at_k"] = 1.0 - original

    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        compare_runs(suite, baseline, candidate)


def test_existing_comparison_output_is_preserved(make_run, tmp_path):
    suite, baseline = make_run()
    _, candidate = make_run()
    report = compare_runs(suite, baseline, candidate)

    output = tmp_path / "comparison.json"
    original = b'{existing: true}\n'
    output.write_bytes(original)

    with pytest.raises(FileExistsError):
        write_review(output, report)

    assert output.read_bytes() == original
