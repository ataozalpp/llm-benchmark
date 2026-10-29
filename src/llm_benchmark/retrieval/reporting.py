"""Macro aggregation of per-query retrieval metrics."""

from dataclasses import dataclass
from statistics import fmean

from .evaluation import RetrievalMetrics


@dataclass(frozen=True)
class RetrievalSummary:
    query_count: int
    k: int
    hit_rate_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mrr_at_k: float


def summarize_retrieval(
    metrics: tuple[RetrievalMetrics, ...],
) -> RetrievalSummary:
    if not metrics:
        raise ValueError("At least one metric is required.")

    query_ids = tuple(item.query_id for item in metrics)

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Query IDs must be unique.")

    k = metrics[0].k

    if any(item.k != k for item in metrics):
        raise ValueError("All metrics must use the same k.")

    return RetrievalSummary(
        query_count=len(metrics),
        k=k,
        hit_rate_at_k=fmean(item.hit_at_k for item in metrics),
        mean_precision_at_k=fmean(item.precision_at_k for item in metrics),
        mean_recall_at_k=fmean(item.recall_at_k for item in metrics),
        mrr_at_k=fmean(item.reciprocal_rank_at_k for item in metrics),
    )
