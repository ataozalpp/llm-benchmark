"""Compare query-level outcomes of two validated RAG runs.

Inputs are result rows as written to ``results.jsonl`` plus the suite's
scenarios, which supply corpus-level answerability (rows do not contain
reference answers). Outcomes follow ``summarize_rag`` denominators: a case
without an evaluation (invalid, incomplete or failed generation) counts as
incorrect. No combined score is produced.
"""

from collections.abc import Mapping, Sequence

from .models import RagScenario

_STATUSES = frozenset(
    {"completed", "invalid_response", "incomplete_response", "provider_failed"}
)


def _index_rows(rows: Sequence[Mapping[str, object]], label: str) -> dict[str, Mapping]:
    if not rows:
        raise ValueError(f"{label} run has no result rows.")

    indexed: dict[str, Mapping] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} result rows must be objects.")

        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"{label} query IDs must be nonblank strings.")
        if query_id in indexed:
            raise ValueError(f"{label} run has duplicate query ID {query_id!r}.")

        if row.get("status") not in _STATUSES:
            raise ValueError(f"{label} row {query_id!r} has an unknown status.")

        context_ids = row.get("context_chunk_ids")
        if not isinstance(context_ids, list) or any(
            not isinstance(item, str) for item in context_ids
        ):
            raise ValueError(f"{label} row {query_id!r} has invalid context IDs.")

        for field in ("answer", "evaluation"):
            value = row.get(field)
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"{label} row {query_id!r} has an invalid {field}.")

        if (row.get("evaluation") is None) != (row["status"] != "completed"):
            raise ValueError(
                f"{label} row {query_id!r} status and evaluation are inconsistent."
            )

        indexed[query_id] = row
    return indexed


def _exact_match(row: Mapping) -> bool:
    evaluation = row.get("evaluation")
    return evaluation is not None and evaluation.get("answer_exact_match") is True


def _valid_abstention(row: Mapping) -> bool:
    evaluation = row.get("evaluation")
    return evaluation is not None and evaluation.get("abstention_correct") is True


def _paired_outcomes(
    query_ids: list[str],
    baseline: dict[str, Mapping],
    candidate: dict[str, Mapping],
    outcome,
) -> dict[str, object]:
    groups: dict[str, list[str]] = {
        "both_correct": [],
        "baseline_only": [],
        "candidate_only": [],
        "both_incorrect": [],
    }
    for query_id in query_ids:
        left = outcome(baseline[query_id])
        right = outcome(candidate[query_id])
        if left and right:
            groups["both_correct"].append(query_id)
        elif left:
            groups["baseline_only"].append(query_id)
        elif right:
            groups["candidate_only"].append(query_id)
        else:
            groups["both_incorrect"].append(query_id)

    total = len(query_ids)
    baseline_correct = len(groups["both_correct"]) + len(groups["baseline_only"])
    candidate_correct = len(groups["both_correct"]) + len(groups["candidate_only"])

    def rate(count: int) -> float | None:
        return count / total if total else None

    return {
        "denominator": total,
        "counts": {name: len(ids) for name, ids in groups.items()},
        "query_ids": groups,
        "baseline_rate": rate(baseline_correct),
        "candidate_rate": rate(candidate_correct),
        "delta": rate(candidate_correct - baseline_correct),
    }


def compare_rag_outcomes(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    scenarios: tuple[RagScenario, ...],
) -> dict[str, object]:
    if not scenarios:
        raise ValueError("Comparison requires at least one scenario.")

    baseline = _index_rows(baseline_rows, "Baseline")
    candidate = _index_rows(candidate_rows, "Candidate")
    suite_ids = [scenario.query.query_id for scenario in scenarios]

    if len(set(suite_ids)) != len(suite_ids):
        raise ValueError("Suite has duplicate query IDs.")
    if set(baseline) != set(suite_ids) or set(candidate) != set(suite_ids):
        raise ValueError("Runs must contain exactly the suite query IDs.")

    answerable = sorted(s.query.query_id for s in scenarios if s.answerable)
    unanswerable = sorted(s.query.query_id for s in scenarios if not s.answerable)
    ordered = sorted(suite_ids)

    status_transitions: dict[str, list[str]] = {}
    failure_code_changes = []
    context_changed = []
    answer_changed = []

    for query_id in ordered:
        left, right = baseline[query_id], candidate[query_id]

        if left["status"] != right["status"]:
            key = f"{left['status']}->{right['status']}"
            status_transitions.setdefault(key, []).append(query_id)

        if left.get("failure_code") != right.get("failure_code"):
            failure_code_changes.append(
                {
                    "query_id": query_id,
                    "baseline": left.get("failure_code"),
                    "candidate": right.get("failure_code"),
                }
            )

        if left["context_chunk_ids"] != right["context_chunk_ids"]:
            context_changed.append(query_id)

        if left.get("answer") != right.get("answer"):
            answer_changed.append(
                {
                    "query_id": query_id,
                    "baseline": left.get("answer"),
                    "candidate": right.get("answer"),
                }
            )

    return {
        "schema_version": 1,
        "query_count": len(ordered),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "answer_exact_match": _paired_outcomes(
            answerable, baseline, candidate, _exact_match
        ),
        "unanswerable_abstention": _paired_outcomes(
            unanswerable, baseline, candidate, _valid_abstention
        ),
        "status_transitions": dict(sorted(status_transitions.items())),
        "failure_code_changes": failure_code_changes,
        "context_changed": context_changed,
        "answer_changed": answer_changed,
    }
