"""Compare query-level hit outcomes from validated retrieval runs."""

from collections.abc import Mapping


def compare_hit_outcomes(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
) -> dict[str, object]:
    for outcomes in (baseline, candidate):
        if not outcomes:
            raise ValueError("Comparison requires nonempty query outcomes.")

        if any(
            not isinstance(query_id, str) or not query_id.strip()
            for query_id in outcomes
        ):
            raise ValueError("Query IDs must be nonblank strings.")

        if any(
            type(value) is not bool for value in outcomes.values()
        ):
            raise ValueError("Hit outcomes must be booleans.")

    if set(baseline) != set(candidate):
        raise ValueError("Runs must contain the same query IDs.")

    groups: dict[str, list[str]] = {
        "both_hit": [],
        "baseline_only": [],
        "candidate_only": [],
        "both_miss": [],
    }

    for query_id in sorted(baseline):
        left = baseline[query_id]
        right = candidate[query_id]

        if left and right:
            group = "both_hit"
        elif left:
            group = "baseline_only"
        elif right:
            group = "candidate_only"
        else:
            group = "both_miss"

        groups[group].append(query_id)

    counts = {name: len(ids) for name, ids in groups.items()}
    query_count = len(baseline)

    baseline_hits = counts["both_hit"] + counts["baseline_only"]
    candidate_hits = counts["both_hit"] + counts["candidate_only"]
    union_hits = query_count - counts["both_miss"]

    return {
        "schema_version": 1,
        "query_count": query_count,
        "counts": counts,
        "query_ids": groups,
        "baseline_hit_rate": baseline_hits / query_count,
        "candidate_hit_rate": candidate_hits / query_count,
        "hit_rate_delta": (candidate_hits - baseline_hits) / query_count,
        "candidate_union_hit_rate": union_hits / query_count,
    }