"""Deterministic diagnostic sampling of retrieval evaluations."""

import hashlib
from collections.abc import Sequence

from .evaluation import RetrievalMetrics

REVIEW_VERSION = "retrieval-review-v1"

SAMPLE_LIMITS = {
    "rank_1": 5,
    "rank_2_to_k": 5,
    "miss": 10,
}


def _selection_key(query_id: str) -> tuple[str, str]:
    value = f"{REVIEW_VERSION}:{query_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest(), query_id


def select_review_queries(
    evaluations: Sequence[RetrievalMetrics],
) -> dict[str, object]:
    if not evaluations:
        raise ValueError("Review requires at least one evaluation.")

    query_ids = [item.query_id for item in evaluations]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Review query IDs must be unique.")

    if len({item.k for item in evaluations}) != 1:
        raise ValueError("Review evaluations must use the same K.")

    groups = {name: [] for name in SAMPLE_LIMITS}

    for item in evaluations:
        reciprocal_rank = item.reciprocal_rank_at_k

        if reciprocal_rank == 1.0:
            group = "rank_1"
        elif 0.0 < reciprocal_rank < 1.0:
            group = "rank_2_to_k"
        elif reciprocal_rank == 0.0:
            group = "miss"
        else:
            raise ValueError("Invalid reciprocal rank.")

        groups[group].append(item.query_id)

    selected = {
        group: sorted(query_ids, key=_selection_key)[: SAMPLE_LIMITS[group]]
        for group, query_ids in groups.items()
    }

    return {
        "review_version": REVIEW_VERSION,
        "selection_policy": "sha256-version-and-query-id",
        "requested_counts": dict(SAMPLE_LIMITS),
        "population_counts": {
            group: len(query_ids) for group, query_ids in groups.items()
        },
        "selected_query_ids": selected,
    }
