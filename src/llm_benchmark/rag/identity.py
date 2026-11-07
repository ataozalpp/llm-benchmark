"""Canonical identities for RAG evaluation runs."""

from ..retrieval.identity import fingerprint
from ..retrieval.models import Chunk
from .models import RagScenario
from .prompts import PROMPT_VERSION

RAG_EVALUATOR_VERSION = "rag-exact-v1"
CONTEXT_POLICY_VERSION = "whole-chunk-json-chars-v1"


def build_rag_run_identity(
    *,
    chunks: tuple[Chunk, ...],
    scenarios: tuple[RagScenario, ...],
    retriever_profile: dict[str, object],
    k: int,
    max_context_chars: int,
    generation: dict[str, object],
) -> dict[str, str]:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")

    if type(max_context_chars) is not int or max_context_chars < 2:
        raise ValueError("max_context_chars must be at least two.")

    if not isinstance(chunks, tuple) or not isinstance(scenarios, tuple):
        raise TypeError("Chunks and scenarios must be tuples.")

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    if not isinstance(retriever_profile, dict) or not isinstance(generation, dict):
        raise TypeError("Profiles must be dictionaries.")

    corpus_snapshot = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
        }
        for chunk in sorted(chunks, key=lambda item: item.chunk_id)
    ]

    scenario_snapshot = [
        {
            "query_id": scenario.query.query_id,
            "text": scenario.query.text,
            "answerable": scenario.answerable,
            "reference_answers": list(scenario.reference_answers),
            "relevant_chunk_ids": sorted(scenario.relevant_chunk_ids),
        }
        for scenario in scenarios
    ]

    identity = {
        "corpus_hash": fingerprint(corpus_snapshot),
        "suite_hash": fingerprint(scenario_snapshot),
        "retriever_profile_hash": fingerprint(retriever_profile),
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_config_hash": fingerprint(generation),
        "evaluation_config_hash": fingerprint(
            {"k": k, "max_context_chars": max_context_chars}
        ),
        "evaluator_version": RAG_EVALUATOR_VERSION,
    }

    return {**identity, "run_fingerprint": fingerprint(identity)}