"""Connect retrieval, context construction and RAG case execution."""

from collections.abc import Callable
from dataclasses import dataclass

from ..retrieval.models import Chunk, RankedRetrieval
from ..retrieval.retriever import Retriever
from .case import RagCaseResult, run_rag_case
from .context import build_context
from .models import RagContext, RagScenario
from .providers import RagProvider


@dataclass(frozen=True)
class RagSuiteCase:
    ranking: RankedRetrieval
    context: RagContext
    result: RagCaseResult


@dataclass(frozen=True)
class RagSuiteResult:
    cases: tuple[RagSuiteCase, ...]


def run_rag_suite(
    *,
    chunks: tuple[Chunk, ...],
    scenarios: tuple[RagScenario, ...],
    retriever: Retriever,
    provider: RagProvider,
    k: int,
    max_context_chars: int,
    on_case_completed: Callable[[RagSuiteCase], None] | None = None,
) -> RagSuiteResult:
    if on_case_completed is not None and not callable(on_case_completed):
        raise TypeError("on_case_completed must be callable or None.")

    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")

    if type(max_context_chars) is not int or max_context_chars < 2:
        raise ValueError("max_context_chars must be at least two.")

    if not isinstance(chunks, tuple):
        raise TypeError("Chunks must be a tuple.")

    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("Expected Chunk instances.")

    if not isinstance(scenarios, tuple):
        raise TypeError("Scenarios must be a tuple.")

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    if any(not isinstance(item, RagScenario) for item in scenarios):
        raise TypeError("Expected RagScenario instances.")

    corpus_ids = frozenset(chunk.chunk_id for chunk in chunks)

    if len(corpus_ids) != len(chunks):
        raise ValueError("Corpus chunk IDs must be unique.")

    query_ids = tuple(item.query.query_id for item in scenarios)

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Scenario query IDs must be unique.")

    for scenario in scenarios:
        if not scenario.relevant_chunk_ids <= corpus_ids:
            raise ValueError("Gold chunk IDs must exist in the corpus.")

    cases: list[RagSuiteCase] = []

    for scenario in scenarios:
        ranking = retriever.retrieve(scenario.query, top_k=k)

        if not isinstance(ranking, RankedRetrieval):
            raise TypeError("Retriever must return RankedRetrieval.")

        if ranking.query_id != scenario.query.query_id:
            raise ValueError("Retriever returned the wrong query ID.")

        if len(ranking.chunk_ids) > k:
            raise ValueError("Retriever returned more than k results.")

        context = build_context(
            chunks=chunks,
            ranking=ranking,
            max_chars=max_context_chars,
        )

        result = run_rag_case(
            scenario=scenario,
            context=context,
            provider=provider,
        )

        case = RagSuiteCase(
            ranking=ranking,
            context=context,
            result=result,
        )
        if on_case_completed is not None:
            on_case_completed(case)
        cases.append(case)

    return RagSuiteResult(cases=tuple(cases))
