"""Deterministic answer and evidence metrics for RAG."""

import unicodedata
from dataclasses import dataclass

from .models import RagAnswer, RagContext, RagScenario


@dataclass(frozen=True)
class RagEvaluation:
    query_id: str
    answer_exact_match: bool | None
    abstention_correct: bool
    context_evidence_recall: float | None
    citation_validity: float | None
    citation_evidence_recall: float | None
    citation_count: int


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def evaluate_rag_answer(
    *, scenario: RagScenario, context: RagContext, answer: RagAnswer
) -> RagEvaluation:
    if not isinstance(scenario, RagScenario):
        raise TypeError("Expected RagScenario.")

    if not isinstance(context, RagContext):
        raise TypeError("Expected RagContext.")

    if not isinstance(answer, RagAnswer):
        raise TypeError("Expected RagAnswer.")

    if scenario.query.query_id != context.query_id:
        raise ValueError("Scenario and context query IDs must match.")

    context_ids = set(context.chunk_ids)
    cited_ids = set(answer.citations)
    valid_cited_ids = cited_ids & context_ids
    gold_ids = scenario.relevant_chunk_ids

    exact_match = None
    context_recall = None
    citation_recall = None

    if scenario.answerable:
        expected_answers = {
            normalize_answer(reference) for reference in scenario.reference_answers
        }

        exact_match = (
            not answer.abstained and normalize_answer(answer.answer) in expected_answers
        )
        context_recall = len(gold_ids & context_ids) / len(gold_ids)
        citation_recall = len(gold_ids & valid_cited_ids) / len(gold_ids)

    citation_validity = len(valid_cited_ids) / len(cited_ids) if cited_ids else None

    return RagEvaluation(
        query_id=scenario.query.query_id,
        answer_exact_match=exact_match,
        abstention_correct=answer.abstained == (not scenario.answerable),
        context_evidence_recall=context_recall,
        citation_validity=citation_validity,
        citation_evidence_recall=citation_recall,
        citation_count=len(cited_ids),
    )
