"""Compares query-level outcomes from two completed RAG runs."""

import argparse
import json
import sys
from pathlib import Path

from llm_benchmark.rag.comparison import compare_rag_outcomes
from llm_benchmark.rag.dataset import load_rag_dataset
from llm_benchmark.rag.identity import corpus_fingerprint, suite_fingerprint
from scripts.review_retrieval_run import (
    _parse_json,
    _read_text,
    read_json,
    write_review,
)

# Fields that define what was evaluated; both runs must agree on them.
MATCHING_FIELDS = (
    "corpus_hash",
    "suite_hash",
    "evaluation_config_hash",
    "evaluator_version",
    "context_policy_version",
    "k",
    "max_context_chars",
)

# Fields a comparison is meant to vary; reported, never required to match.
VARIABLE_FIELDS = (
    "prompt_version",
    "generation_config_hash",
    "retriever_profile_hash",
)

SUMMARY_FIELDS = (
    "status_counts",
    "invalid_response_reasons",
    "answer_exact_match",
    "answer_exact_match_given_full_context",
    "unanswerable_abstention",
    "context_evidence_recall",
    "citation_validity",
)


def read_rag_results(path: Path, run_id: str) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        row = _parse_json(line)
        if not isinstance(row, dict):
            raise ValueError(f"Result line {line_number} must be a JSON object.")
        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("Unsupported RAG result schema.")
        if row.get("run_id") != run_id:
            raise ValueError(f"Result line {line_number} belongs to another run.")
        rows.append(row)
    return rows


def load_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("artifact_schema_version") != 1:
        raise ValueError(f"Unsupported RAG manifest in {run_dir.name}.")
    if manifest.get("status") != "completed":
        raise ValueError(f"Run {run_dir.name} is not completed.")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Run {run_dir.name} has no run ID.")

    rows = read_rag_results(run_dir / "results.jsonl", run_id)
    scenario_ids = manifest.get("scenario_ids")
    if (
        not isinstance(scenario_ids, list)
        or manifest.get("completed_case_count") != len(scenario_ids)
        or len(rows) != len(scenario_ids)
        or [row.get("query_id") for row in rows] != scenario_ids
    ):
        raise ValueError(f"Run {run_id} results are incomplete or reordered.")

    summary = read_json(run_dir / "summary.json")
    if summary.get("run_id") != run_id:
        raise ValueError(f"Summary does not belong to run {run_id}.")
    return manifest, rows, summary


def compare_runs(
    suite_path: Path,
    baseline_dir: Path,
    candidate_dir: Path,
) -> dict[str, object]:
    # Validates recorded artifacts only; nothing is retrieved or generated.
    dataset = load_rag_dataset(suite_path)
    baseline_manifest, baseline_rows, baseline_summary = load_run(baseline_dir)
    candidate_manifest, candidate_rows, candidate_summary = load_run(candidate_dir)

    for field in MATCHING_FIELDS:
        if baseline_manifest.get(field) != candidate_manifest.get(field):
            raise ValueError(f"Incompatible runs: {field} differs.")

    if baseline_manifest["suite_hash"] != suite_fingerprint(dataset.scenarios):
        raise ValueError("Suite file does not match the runs' suite_hash.")
    if baseline_manifest["corpus_hash"] != corpus_fingerprint(dataset.chunks):
        raise ValueError("Suite file does not match the runs' corpus_hash.")

    report = compare_rag_outcomes(baseline_rows, candidate_rows, dataset.scenarios)
    varied = [
        field
        for field in VARIABLE_FIELDS
        if baseline_manifest.get(field) != candidate_manifest.get(field)
    ]

    result = {
        **report,
        "baseline_run_id": baseline_manifest["run_id"],
        "candidate_run_id": candidate_manifest["run_id"],
        "suite_id": dataset.suite_id,
        "suite_hash": baseline_manifest["suite_hash"],
        "corpus_hash": baseline_manifest["corpus_hash"],
        "k": baseline_manifest["k"],
        "max_context_chars": baseline_manifest["max_context_chars"],
        "varied_fields": varied,
        "baseline_settings": {
            "prompt_version": baseline_manifest.get("prompt_version"),
            "generation": baseline_manifest.get("generation"),
            "retriever_profile": baseline_manifest.get("retriever_profile"),
        },
        "candidate_settings": {
            "prompt_version": candidate_manifest.get("prompt_version"),
            "generation": candidate_manifest.get("generation"),
            "retriever_profile": candidate_manifest.get("retriever_profile"),
        },
        "recorded_summaries": {
            "baseline": {key: baseline_summary.get(key) for key in SUMMARY_FIELDS},
            "candidate": {key: candidate_summary.get(key) for key in SUMMARY_FIELDS},
        },
        "validation": "recorded_artifacts",
        "replay": "not_requested",
    }
    if not varied:
        result["note"] = (
            "Identical configuration; differences reflect generation "
            "nondeterminism or endpoint changes."
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = compare_runs(args.suite, args.baseline, args.candidate)
        write_review(args.output, report)
    except (OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output_path": str(args.output.resolve()),
                "varied_fields": report["varied_fields"],
                "answer_exact_match_delta": report["answer_exact_match"]["delta"],
                "unanswerable_abstention_delta": report["unanswerable_abstention"][
                    "delta"
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
