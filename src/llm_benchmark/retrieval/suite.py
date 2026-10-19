"""Validated orchestration of retrieval evaluation scenarios."""

from collections.abc import Callable
from dataclasses import dataclass

from .evaluation import RetrievalMetrics, evaluate_retrieval
from .models import Chunk, RankedRetrieval, RetrievalScenario
from .reporting import RetrievalSummary, summarize_retrieval
from .retriever import Retriever


@dataclass(frozen=True)
class RetrievalSuiteResult:
    rankings: tuple[RankedRetrieval, ...]
    evaluations: tuple[RetrievalMetrics, ...]
    summary: RetrievalSummary


def run_retrieval_suite(
    *,
    chunks: tuple[Chunk, ...],
    scenarios: tuple[RetrievalScenario, ...],
    retriever_factory: Callable[[tuple[Chunk, ...]], Retriever],
    k: int,
    on_case_completed: Callable[[RankedRetrieval, RetrievalMetrics], None] | None = None,
) -> RetrievalSuiteResult:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")

    if not isinstance(chunks, tuple):
        raise TypeError("Chunks must be a tuple.")

    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("Every corpus item must be a Chunk.")

    if not isinstance(scenarios, tuple):
        raise TypeError("Scenarios must be a tuple.")

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    if any(
        not isinstance(scenario, RetrievalScenario)
        for scenario in scenarios
    ):
        raise TypeError("Expected RetrievalScenario.")

    corpus_ids = frozenset(chunk.chunk_id for chunk in chunks)

    if len(corpus_ids) != len(chunks):
        raise ValueError("Corpus chunk IDs must be unique.")

    query_ids = tuple(
        scenario.query.query_id for scenario in scenarios
    )

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Scenario query IDs must be unique.")

    for scenario in scenarios:
        if not scenario.evaluation.relevant_chunk_ids <= corpus_ids:
            raise ValueError("Gold chunk IDs must exist in the corpus.")

    retriever = retriever_factory(chunks)

    rankings: list[RankedRetrieval] = []
    evaluations: list[RetrievalMetrics] = []

    for scenario in scenarios:
        result = retriever.retrieve(scenario.query, top_k=k)

        if not isinstance(result, RankedRetrieval):
            raise TypeError("Retriever must return RankedRetrieval.")

        if result.query_id != scenario.query.query_id:
            raise ValueError("Retriever returned the wrong query ID.")

        if len(result.chunk_ids) > k:
            raise ValueError("Retriever returned more than top_k results.")

        if not set(result.chunk_ids) <= corpus_ids:
            raise ValueError("Retrieved chunk IDs must exist in the corpus.")

        metrics = evaluate_retrieval(
            case=scenario.evaluation,
            result=result,
            k=k,
        )

        rankings.append(result)
        evaluations.append(metrics)
        if on_case_completed is not None:
            on_case_completed(result, metrics)

    evaluated = tuple(evaluations)

    return RetrievalSuiteResult(
        rankings=tuple(rankings),
        evaluations=evaluated,
        summary=summarize_retrieval(evaluated),
    )
