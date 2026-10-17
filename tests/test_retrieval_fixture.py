import json
from pathlib import Path

import pytest

from llm_benchmark.retrieval.keyword import KeywordRetriever
from llm_benchmark.retrieval.models import (
    Chunk,
    RetrievalEvaluationCase,
    RetrievalQuery,
    RetrievalScenario,
)
from llm_benchmark.retrieval.suite import run_retrieval_suite


def test_keyword_smoke_fixture():
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "retrieval"
        / "keyword_smoke.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["suite_id"] == "keyword-smoke-v1"

    chunks = tuple(
        Chunk(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            text=item["text"],
        )
        for item in payload["chunks"]
    )

    scenarios = tuple(
        RetrievalScenario(
            query=RetrievalQuery(
                query_id=item["query_id"],
                text=item["text"],
            ),
            evaluation=RetrievalEvaluationCase(
                query_id=item["query_id"],
                relevant_chunk_ids=frozenset(
                    item["relevant_chunk_ids"]
                ),
            ),
        )
        for item in payload["scenarios"]
    )

    result = run_retrieval_suite(
        chunks=chunks,
        scenarios=scenarios,
        retriever_factory=KeywordRetriever,
        k=payload["k"],
    )

    assert tuple(
        ranking.query_id for ranking in result.rankings
    ) == (
        "q-refund",
        "q-account",
        "q-delivery",
        "q-invoice",
    )

    assert tuple(
        ranking.chunk_ids for ranking in result.rankings
    ) == (
        ("refund",),
        ("account",),
        (),
        ("payment",),
    )

    summary = result.summary

    assert summary.query_count == 4
    assert summary.k == 2
    assert summary.hit_rate_at_k == pytest.approx(0.75)
    assert summary.mean_precision_at_k == pytest.approx(0.375)
    assert summary.mean_recall_at_k == pytest.approx(0.75)
    assert summary.mrr_at_k == pytest.approx(0.75)

    repeated = run_retrieval_suite(
        chunks=chunks,
        scenarios=scenarios,
        retriever_factory=KeywordRetriever,
        k=payload["k"],
    )

    assert repeated == result