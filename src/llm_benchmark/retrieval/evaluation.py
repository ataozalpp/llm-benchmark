"""Pure metrics for ranked chunk retrieval."""

from dataclasses import dataclass

from .models import RankedRetrieval, RetrievalEvaluationCase


@dataclass(frozen=True)
class RetrievalMetrics:
    query_id: str
    k: int
    retrieved_count: int
    relevant_count: int
    relevant_retrieved_count: int
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank_at_k: float


def evaluate_retrieval(
    *,
    case: RetrievalEvaluationCase,
    result: RankedRetrieval,
    k: int,
) -> RetrievalMetrics:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")

    if case.query_id != result.query_id:
        raise ValueError("Query IDs must match.")

    selected = result.chunk_ids[:k]
    relevant = case.relevant_chunk_ids

    hits = sum(chunk_id in relevant for chunk_id in selected)

    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(selected, start=1)
            if chunk_id in relevant
        ),
        None,
    )

    return RetrievalMetrics(
        query_id=case.query_id,
        k=k,
        retrieved_count=len(selected),
        relevant_count=len(relevant),
        relevant_retrieved_count=hits,
        hit_at_k=float(hits > 0),
        precision_at_k= hits / k,
        recall_at_k= hits / len(relevant),
        reciprocal_rank_at_k=(
            0.0
            if first_relevant_rank is None
            else 1.0 / first_relevant_rank
        ),
    )