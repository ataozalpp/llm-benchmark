"""The synthetic RAG provider is a fixed, gold-free software fixture."""

import json

from llm_benchmark.rag.case import run_rag_case
from llm_benchmark.rag.demo_provider import SyntheticRagProvider
from llm_benchmark.rag.models import RagContext, RagGenerationRequest, RagScenario
from llm_benchmark.rag.parsing import parse_rag_answer
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery

QUERY = RetrievalQuery("q1", "Refund period?")
REFUND = Chunk("refund", "policy", "Refunds within 30 days.")
OTHER = Chunk("other", "policy", "Shipping takes 5 days.")


def request(*chunks):
    return RagGenerationRequest(query=QUERY, context=RagContext("q1", chunks))


def test_empty_context_abstains():
    response = SyntheticRagProvider().generate(request())
    answer = parse_rag_answer(response.text)
    assert answer.abstained
    assert answer.answer == ""
    assert answer.citations == ()
    assert response.finish_reason == "stop"


def test_first_context_chunk_is_answered_and_cited():
    response = SyntheticRagProvider().generate(request(REFUND, OTHER))
    assert json.loads(response.text) == {
        "answer": "Refunds within 30 days.",
        "citations": ["refund"],
        "abstained": False,
    }


def test_usage_is_unmeasured_not_zero():
    response = SyntheticRagProvider().generate(request(REFUND))
    assert response.input_tokens is None
    assert response.output_tokens is None


def test_provider_output_is_deterministic():
    provider = SyntheticRagProvider()
    assert provider.generate(request(REFUND)) == provider.generate(request(REFUND))


def test_case_runner_completes_with_synthetic_provider():
    scenario = RagScenario(
        query=QUERY,
        reference_answers=("30 days",),
        relevant_chunk_ids=frozenset({"refund"}),
    )
    result = run_rag_case(
        scenario=scenario,
        context=RagContext("q1", (REFUND,)),
        provider=SyntheticRagProvider(),
    )
    assert result.status == "completed"
    assert result.evaluation.answer_exact_match is False
    assert result.evaluation.citation_validity == 1.0
    assert result.evaluation.citation_evidence_recall == 1.0
