"""Canonical identities for keyword retrieval evaluation."""

import hashlib
import json
from dataclasses import asdict

from .models import Chunk, RetrievalScenario
from .profile import (
    KEYWORD_PROFILE,
    DenseRetrievalProfile,
    KeywordRetrieverProfile,
)

EVALUATOR_VERSION = "retrieval-exact-v1"
INPUT_PROFILE_VERSION = "prechunked-text-v1"


def fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def build_retrieval_run_identity(
    *,
    chunks: tuple[Chunk, ...],
    scenarios: tuple[RetrievalScenario, ...],
    k: int,
    profile: KeywordRetrieverProfile | DenseRetrievalProfile,
) -> dict[str, str]:
    if not isinstance(profile, (KeywordRetrieverProfile, DenseRetrievalProfile)):
        raise TypeError("Unsupported retriever profile.")

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

    if any(not isinstance(scenario, RetrievalScenario) for scenario in scenarios):
        raise TypeError("Expected RetrievalScenario items.")

    corpus_ids = frozenset(chunk.chunk_id for chunk in chunks)

    if len(corpus_ids) != len(chunks):
        raise ValueError("Corpus chunk IDs must be unique.")

    query_ids = tuple(scenario.query.query_id for scenario in scenarios)

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Scenario query IDs must be unique.")

    for scenario in scenarios:
        if not scenario.evaluation.relevant_chunk_ids <= corpus_ids:
            raise ValueError("Gold chunk IDs must exist in the corpus.")

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
            "relevant_chunk_ids": sorted(scenario.evaluation.relevant_chunk_ids),
        }
        for scenario in scenarios
    ]

    identity = {
        "input_profile_version": INPUT_PROFILE_VERSION,
        "corpus_hash": fingerprint(corpus_snapshot),
        "suite_hash": fingerprint(scenario_snapshot),
        "retriever_profile_hash": fingerprint(asdict(profile)),
        "evaluation_config_hash": fingerprint({"k": k}),
        "evaluator_version": EVALUATOR_VERSION,
    }

    return {
        **identity,
        "run_fingerprint": fingerprint(identity),
    }


def build_keyword_run_identity(
    *,
    chunks: tuple[Chunk, ...],
    scenarios: tuple[RetrievalScenario, ...],
    k: int,
) -> dict[str, str]:
    """Preserve the existing keyword identity contract."""
    return build_retrieval_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=k,
        profile=KEYWORD_PROFILE,
    )
