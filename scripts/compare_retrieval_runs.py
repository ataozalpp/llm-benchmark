"""Compares hit outcomes from two completed retrieval runs."""

import argparse
import json
import sys
from pathlib import Path

from llm_benchmark.retrieval.comparison import compare_hit_outcomes
from scripts.review_retrieval_run import (
    build_review,
    read_json,
    read_results,
    write_review,
)


def compare_runs(
    suite_path: Path,
    baseline_dir: Path,
    candidate_dir: Path,
) -> dict[str, object]:
    # Validate identities, completeness, rankings and stored metrics.
    # This does not replay either retriever.
    build_review(suite_path, baseline_dir)
    build_review(suite_path, candidate_dir)

    baseline_manifest = read_json(baseline_dir / "manifest.json")
    candidate_manifest = read_json(candidate_dir / "manifest.json")

    matching_fields = (
        "corpus_hash",
        "suite_hash",
        "evaluation_config_hash",
        "evaluator_version",
        "input_profile_version",
        "k",
    )
    for field in matching_fields:
        if baseline_manifest[field] != candidate_manifest[field]:
            raise ValueError(f"Incompatible runs: {field} differs.")

    baseline_rows = read_results(baseline_dir / "results.jsonl")
    candidate_rows = read_results(candidate_dir / "results.jsonl")

    baseline_hits = {
        row["query_id"]: row["metrics"]["hit_at_k"] == 1.0
        for row in baseline_rows
    }
    candidate_hits = {
        row["query_id"]: row["metrics"]["hit_at_k"] == 1.0
        for row in candidate_rows
    }

    report = compare_hit_outcomes(baseline_hits, candidate_hits)

    return {
        **report,
        "baseline_run_id": baseline_manifest["run_id"],
        "candidate_run_id": candidate_manifest["run_id"],
        "baseline_profile": baseline_manifest["retriever_profile"],
        "candidate_profile": candidate_manifest["retriever_profile"],
        "suite_hash": baseline_manifest["suite_hash"],
        "corpus_hash": baseline_manifest["corpus_hash"],
        "k": baseline_manifest["k"],
        "validation": "recorded_rankings_and_metrics",
        "retrieval_replay": "not_requested",
        "union_note": (
            "Union coverage may use up to 2*k candidates. "
            "It is not measured hybrid hit rate at k."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = compare_runs(
            args.suite,
            args.baseline,
            args.candidate,
        )
        write_review(args.output, report)
    except (OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output_path": str(args.output.resolve()),
                "counts": report["counts"],
                "hit_rate_delta": report["hit_rate_delta"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())