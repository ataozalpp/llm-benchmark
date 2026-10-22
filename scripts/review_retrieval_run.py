"""Create a local diagnostic sample from a completed keyword run."""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from llm_benchmark.retrieval.dataset import load_retrieval_dataset
from llm_benchmark.retrieval.evaluation import evaluate_retrieval
from llm_benchmark.retrieval.identity import build_keyword_run_identity
from llm_benchmark.retrieval.keyword import KeywordRetriever
from llm_benchmark.retrieval.models import RankedRetrieval
from llm_benchmark.retrieval.review import select_review_queries

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_REVIEW_BYTES = 16 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in review input.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON constant in review input.")


def _parse_json(text: str):
    try:
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except RecursionError:
        raise ValueError("Review input is nested too deeply.") from None


def _read_text(path: Path) -> str:
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Review input exceeds the size limit.")
    return raw.decode("utf-8")


def _metrics_match(actual: dict, expected: dict) -> bool:
    if actual.keys() != expected.keys():
        return False
    for key, value in expected.items():
        candidate = actual[key]
        if type(value) is float:
            if type(candidate) not in (int, float) or not math.isfinite(candidate):
                return False
        elif type(candidate) is not type(value):
            return False
        if candidate != value:
            return False
    return True


def require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def read_json(path: Path) -> dict[str, object]:
    value = _parse_json(_read_text(path))
    return require_object(value, label=path.name)


def read_results(path: Path) -> list[dict[str, object]]:
    rows = []

    for line_number, line in enumerate(
        _read_text(path).splitlines(),
        start=1,
    ):
        row = require_object(
            _parse_json(line),
            label=f"Result line {line_number}",
        )

        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Result query ID must be a nonblank string.")

        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("Result run ID must be a nonblank string.")

        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("Unsupported result schema.")

        retrieved_ids = row.get("retrieved_chunk_ids")
        if not isinstance(retrieved_ids, list):
            raise ValueError("Retrieved IDs must be a JSON array.")

        if any(not isinstance(item, str) or not item.strip() for item in retrieved_ids):
            raise ValueError("Retrieved IDs must be nonblank strings.")

        require_object(row.get("metrics"), label="Result metrics")
        rows.append(row)

    return rows


def build_review(
    suite_path: Path, run_dir: Path, *, verify_retrieval: bool = False
) -> dict[str, object]:
    dataset = load_retrieval_dataset(suite_path)
    manifest = read_json(run_dir / "manifest.json")

    if type(manifest.get("artifact_schema_version")) is not int:
        raise ValueError("Manifest schema version must be an integer.")

    if manifest["artifact_schema_version"] != 1:
        raise ValueError("Unsupported manifest schema.")

    k = manifest.get("k")
    if type(k) is not int or k <= 0:
        raise ValueError("Manifest K must be a positive integer.")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Manifest run ID must be a nonblank string.")

    completed_count = manifest.get("completed_case_count")
    if type(completed_count) is not int or completed_count < 0:
        raise ValueError("Completed case count must be a nonnegative integer.")

    if manifest.get("status") != "completed":
        raise ValueError("Only completed runs can be reviewed.")

    identity = build_keyword_run_identity(
        chunks=dataset.chunks,
        scenarios=dataset.scenarios,
        k=k,
    )

    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("The suite does not match the run identity.")

    replay_retriever = KeywordRetriever(dataset.chunks) if verify_retrieval else None

    rows = read_results(run_dir / "results.jsonl")

    expected_ids = [scenario.query.query_id for scenario in dataset.scenarios]

    if manifest.get("scenario_ids") != expected_ids:
        raise ValueError("Manifest query order does not match the suite.")

    if [row["query_id"] for row in rows] != expected_ids:
        raise ValueError("Results are missing, duplicated or out of order.")

    if manifest.get("completed_case_count") != len(expected_ids):
        raise ValueError("Completed case count does not match the suite.")

    chunks = {chunk.chunk_id: chunk for chunk in dataset.chunks}

    evaluations = []
    rows_by_id = {}

    for scenario, row in zip(dataset.scenarios, rows, strict=True):
        if row.get("run_id") != manifest["run_id"]:
            raise ValueError("A result belongs to another run.")

        if row.get("schema_version") != 1:
            raise ValueError("Unsupported result schema.")

        retrieved_ids = row["retrieved_chunk_ids"]

        if not isinstance(retrieved_ids, list):
            raise ValueError("Retrieved IDs must be a JSON array.")

        ranking = RankedRetrieval(
            query_id=row["query_id"],
            chunk_ids=tuple(retrieved_ids),
        )

        if len(ranking.chunk_ids) > k:
            raise ValueError("A result exceeds K.")

        if any(chunk_id not in chunks for chunk_id in ranking.chunk_ids):
            raise ValueError("A result references an unknown chunk.")

        if replay_retriever is not None:
            replayed = replay_retriever.retrieve(
                scenario.query,
                top_k=k,
            )

            if replayed != ranking:
                raise ValueError("Retrieval replay differs from the recorded ranking.")

        evaluation = evaluate_retrieval(
            case=scenario.evaluation,
            result=ranking,
            k=k,
        )

        if not _metrics_match(row["metrics"], asdict(evaluation)):
            raise ValueError("Stored metrics do not match recomputed metrics.")

        evaluations.append(evaluation)
        rows_by_id[ranking.query_id] = row

    selection = select_review_queries(evaluations)
    partial_recall_ids = sorted(
        item.query_id
        for item in evaluations
        if 0 < item.relevant_retrieved_count < item.relevant_count
    )
    scenarios_by_id = {
        scenario.query.query_id: scenario for scenario in dataset.scenarios
    }

    groups = {}

    for group, query_ids in selection["selected_query_ids"].items():
        cases = []

        for query_id in query_ids:
            scenario = scenarios_by_id[query_id]
            row = rows_by_id[query_id]

            cases.append(
                {
                    "query_id": query_id,
                    "query_text": scenario.query.text,
                    "metrics": row["metrics"],
                    "gold_chunks": [
                        asdict(chunks[chunk_id])
                        for chunk_id in sorted(scenario.evaluation.relevant_chunk_ids)
                    ],
                    "retrieved_chunks": [
                        {
                            "rank": rank,
                            "is_gold": (
                                chunk_id in scenario.evaluation.relevant_chunk_ids
                            ),
                            **asdict(chunks[chunk_id]),
                        }
                        for rank, chunk_id in enumerate(
                            row["retrieved_chunk_ids"],
                            start=1,
                        )
                    ],
                }
            )

        groups[group] = cases

    return {
        **selection,
        "run_id": manifest["run_id"],
        "run_fingerprint": identity["run_fingerprint"],
        "suite_id": dataset.suite_id,
        "k": k,
        "groups": groups,
        "recall_diagnostics": {
            "partial_recall_count": len(partial_recall_ids),
            "partial_recall_query_ids": partial_recall_ids,
        },
        "retrieval_replay": ("matched" if verify_retrieval else "not_requested"),
    }


def write_review(path: Path, report: dict[str, object]) -> None:
    """Publish a complete report without replacing an existing destination.

    A same-directory temporary file is hard-linked into place. Filesystems
    without hard-link support fail rather than falling back to partial writes.
    """
    encoded = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_REVIEW_BYTES:
        raise ValueError("Review exceeds the output size limit.")

    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".review-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verify-retrieval", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="new report path; never overwritten"
    )
    args = parser.parse_args()

    output_path = (
        args.output if args.output is not None else args.run_dir / "review.json"
    )
    try:
        report = build_review(
            args.suite, args.run_dir, verify_retrieval=args.verify_retrieval
        )
        write_review(output_path, report)
    except FileExistsError:
        print(
            "error: Review output already exists; it was not replaced.", file=sys.stderr
        )
        raise SystemExit(2) from None
    except (OSError, ValueError):
        print(
            "error: Review failed. Check inputs, replay consistency, and output storage.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    print(
        json.dumps(
            {
                "output_path": str(output_path.resolve()),
                "population_counts": report["population_counts"],
                "selected_query_ids": report["selected_query_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
