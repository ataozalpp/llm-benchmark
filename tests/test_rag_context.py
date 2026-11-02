import json

import pytest

from llm_benchmark.rag.context import build_context
from llm_benchmark.rag.models import RagContext
from llm_benchmark.retrieval.models import Chunk, RankedRetrieval


def chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
    )


def test_context_preserves_ranking_order():
    chunks = (
        chunk("a", "First source."),
        chunk("b", "Second source."),
    )

    context = build_context(
        chunks=chunks,
        ranking=RankedRetrieval("q", ("b", "a")),
        max_chars=1000,
    )

    assert context.chunk_ids == ("b", "a")


def test_exact_budget_accepts_chunk():
    source = chunk("a", "Refunds take 30 days.")
    expected = RagContext(query_id="q", chunks=(source,))
    budget = len(expected.render())

    context = build_context(
        chunks=(source,),
        ranking=RankedRetrieval("q", ("a",)),
        max_chars=budget,
    )

    assert context.chunk_ids == ("a",)
    assert len(context.render()) == budget


def test_overflow_chunk_is_skipped_and_later_chunk_can_fit():
    large = chunk("large", "x" * 1000)
    small = chunk("small", "Short.")
    budget = len(RagContext(query_id="q", chunks=(small,)).render())

    context = build_context(
        chunks=(large, small),
        ranking=RankedRetrieval("q", ("large", "small")),
        max_chars=budget,
    )

    assert context.chunk_ids == ("small",)
    assert len(context.render()) <= budget


def test_budget_applies_to_the_combined_rendered_context():
    a = chunk("a", "One.")
    b = chunk("b", "Two.")
    budget = len(RagContext(query_id="q", chunks=(a,)).render())

    context = build_context(
        chunks=(a, b),
        ranking=RankedRetrieval("q", ("a", "b")),
        max_chars=budget,
    )

    assert context.chunk_ids == ("a",)


def test_text_is_preserved_including_unicode_and_quotes():
    text = 'İade süresi: "30 gün".\nKaynak açıklaması.'
    source = chunk("a", text)

    context = build_context(
        chunks=(source,),
        ranking=RankedRetrieval("q", ("a",)),
        max_chars=1000,
    )

    assert json.loads(context.render()) == [{"chunk_id": "a", "text": text}]


def test_empty_ranking_produces_empty_json_array():
    context = build_context(
        chunks=(),
        ranking=RankedRetrieval("q", ()),
        max_chars=2,
    )
    assert context.render() == "[]"


def test_unknown_chunk_is_rejected_even_with_tiny_budget():
    with pytest.raises(ValueError, match="unknown"):
        build_context(
            chunks=(chunk("a", "Known."),),
            ranking=RankedRetrieval("q", ("missing",)),
            max_chars=2,
        )


def test_duplicate_corpus_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        build_context(
            chunks=(chunk("a", "One."), chunk("a", "Two.")),
            ranking=RankedRetrieval("q", ("a",)),
            max_chars=1000,
        )


@pytest.mark.parametrize("budget", [0, 1, -1, True, "100", 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        build_context(
            chunks=(),
            ranking=RankedRetrieval("q", ()),
            max_chars=budget,
        )
