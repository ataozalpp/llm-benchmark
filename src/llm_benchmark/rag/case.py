"""Execute generation and evaluation for one prepared RAG case."""

from dataclasses import dataclass
from typing import Literal

from .evaluation import RagEvaluation, evaluate_rag_answer
from .models import (
    RagAnswer,
    RagContext,
    RagGenerationRequest,
    RagScenario,
)
from .parsing import RagParseError, parse_rag_answer
from .providers import (
    RagGenerationResponse,
    RagProvider,
    RagProviderError,
)

RagCaseStatus = Literal[
    "completed",
    "invalid_response",
    "incomplete_response",
    "provider_failed",
]


@dataclass(frozen=True)
class RagCaseResult:
    query_id: str
    status: RagCaseStatus
    response: RagGenerationResponse | None = None
    answer: RagAnswer | None = None
    evaluation: RagEvaluation | None = None
    failure_code: str | None = None


def run_rag_case(
    *,
    scenario: RagScenario,
    context: RagContext,
    provider: RagProvider,
) -> RagCaseResult:
    if not isinstance(scenario, RagScenario):
        raise TypeError("Expected RagScenario.")

    request = RagGenerationRequest(
        query=scenario.query,
        context=context,
    )
    query_id = scenario.query.query_id

    try:
        response = provider.generate(request)
    except RagProviderError:
        return RagCaseResult(
            query_id=query_id,
            status="provider_failed",
        )

    if not isinstance(response, RagGenerationResponse):
        raise TypeError("Provider must return RagGenerationResponse.")

    if response.finish_reason != "stop":
        return RagCaseResult(
            query_id=query_id,
            status="incomplete_response",
            response=response,
        )

    try:
        answer = parse_rag_answer(response.text)
    except RagParseError as exc:
        return RagCaseResult(
            query_id=query_id,
            status="invalid_response",
            response=response,
            failure_code=exc.code,
        )

    evaluation = evaluate_rag_answer(
        scenario=scenario,
        context=context,
        answer=answer,
    )

    return RagCaseResult(
        query_id=query_id,
        status="completed",
        response=response,
        answer=answer,
        evaluation=evaluation,
    )
