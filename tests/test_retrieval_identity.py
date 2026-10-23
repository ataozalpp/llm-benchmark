from dataclasses import asdict, replace

import pytest

from llm_benchmark.retrieval.identity import (
    build_keyword_run_identity,
    build_retrieval_run_identity,
    fingerprint,
)
from llm_benchmark.retrieval.models import (
    Chunk,
    RetrievalEvaluationCase,
    RetrievalQuery,
    RetrievalScenario,
)
from llm_benchmark.retrieval.profile import (
    KEYWORD_PROFILE,
    DenseRetrievalProfile,
)


def make_inputs():
    chunks = (
        Chunk("a", "doc", "refund policy"),
        Chunk("b", "doc", "shipping policy"),
    )
    scenarios = (
        RetrievalScenario(
            query=RetrievalQuery("q1", "refund"),
            evaluation=RetrievalEvaluationCase(
                query_id="q1",
                relevant_chunk_ids=frozenset({"a"}),
            ),
        ),
    )
    return chunks, scenarios


def test_dictionary_order_does_not_change_fingerprint():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_identity_is_stable():
    chunks, scenarios = make_inputs()

    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    second = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)

    assert first == second


def test_corpus_order_does_not_change_identity():
    chunks, scenarios = make_inputs()

    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    second = build_keyword_run_identity(
        chunks=tuple(reversed(chunks)), scenarios=scenarios, k=2
    )

    assert first == second


def test_chunk_text_changes_run_identity():
    chunks, scenarios = make_inputs()
    changed = (
        replace(chunks[0], text="refund period fourteen days"),
        chunks[1],
    )

    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    second = build_keyword_run_identity(chunks=changed, scenarios=scenarios, k=2)

    assert first["corpus_hash"] != second["corpus_hash"]
    assert first["suite_hash"] == second["suite_hash"]
    assert first["run_fingerprint"] != second["run_fingerprint"]


def test_query_text_changes_suite_identity():
    chunks, scenarios = make_inputs()
    changed = (
        replace(
            scenarios[0],
            query=RetrievalQuery("q1", "refund period"),
        ),
    )

    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    second = build_keyword_run_identity(chunks=chunks, scenarios=changed, k=2)

    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["suite_hash"] != second["suite_hash"]
    assert first["run_fingerprint"] != second["run_fingerprint"]


def test_k_changes_run_identity():
    chunks, scenarios = make_inputs()

    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=1)
    second = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)

    assert first["run_fingerprint"] != second["run_fingerprint"]


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k):
    chunks, scenarios = make_inputs()

    with pytest.raises(ValueError):
        build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=k)


def test_unknown_gold_is_rejected():
    chunks, scenarios = make_inputs()
    changed = (
        replace(
            scenarios[0],
            evaluation=RetrievalEvaluationCase(
                query_id="q1",
                relevant_chunk_ids=frozenset({"missing"}),
            ),
        ),
    )

    with pytest.raises(ValueError):
        build_keyword_run_identity(chunks=chunks, scenarios=changed, k=2)


def test_gold_change_changes_suite_but_not_corpus():
    chunks, scenarios = make_inputs()
    changed = (
        replace(
            scenarios[0],
            evaluation=RetrievalEvaluationCase("q1", frozenset({"b"})),
        ),
    )
    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    second = build_keyword_run_identity(chunks=chunks, scenarios=changed, k=2)
    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["suite_hash"] != second["suite_hash"]
    assert first["run_fingerprint"] != second["run_fingerprint"]


def test_scenario_order_changes_suite_identity():
    chunks, scenarios = make_inputs()
    scenarios += (
        RetrievalScenario(
            RetrievalQuery("q2", "shipping"),
            RetrievalEvaluationCase("q2", frozenset({"b"})),
        ),
    )
    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    second = build_keyword_run_identity(
        chunks=chunks,
        scenarios=tuple(reversed(scenarios)),
        k=2,
    )
    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["suite_hash"] != second["suite_hash"]
    assert first["run_fingerprint"] != second["run_fingerprint"]


def test_gold_construction_order_does_not_change_identity():
    chunks, scenarios = make_inputs()

    def with_gold(ids):
        return (
            replace(
                scenarios[0],
                evaluation=RetrievalEvaluationCase("q1", frozenset(ids)),
            ),
        )

    first = build_keyword_run_identity(
        chunks=chunks, scenarios=with_gold(("a", "b")), k=2
    )
    second = build_keyword_run_identity(
        chunks=chunks, scenarios=with_gold(("b", "a")), k=2
    )
    assert first == second


@pytest.mark.parametrize("field", ["document_id", "chunk_id"])
def test_corpus_metadata_changes_identity(field):
    chunks, scenarios = make_inputs()
    changed = (chunks[0], replace(chunks[1], **{field: "changed"}))
    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    second = build_keyword_run_identity(chunks=changed, scenarios=scenarios, k=2)
    assert first["corpus_hash"] != second["corpus_hash"]
    assert first["run_fingerprint"] != second["run_fingerprint"]


@pytest.mark.parametrize("kind", ["corpus", "queries", "empty"])
def test_invalid_identity_inputs_are_rejected(kind):
    chunks, scenarios = make_inputs()
    if kind == "corpus":
        chunks += (chunks[0],)
        message = "Corpus chunk IDs"
    elif kind == "queries":
        scenarios += scenarios
        message = "Scenario query IDs"
    else:
        scenarios = ()
        message = "At least one scenario"
    with pytest.raises(ValueError, match=message):
        build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)


@pytest.mark.parametrize(
    "kind", ["chunk_list", "chunk_item", "scenario_list", "scenario_item"]
)
def test_invalid_container_and_item_types_are_rejected(kind):
    chunks, scenarios = make_inputs()
    if kind == "chunk_list":
        chunks = list(chunks)
    elif kind == "chunk_item":
        chunks = (object(),)
    elif kind == "scenario_list":
        scenarios = list(scenarios)
    else:
        scenarios = (object(),)
    with pytest.raises(TypeError):
        build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_fingerprint_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        fingerprint({"value": value})


@pytest.mark.parametrize(
    "version_field", ["EVALUATOR_VERSION", "INPUT_PROFILE_VERSION"]
)
def test_version_change_changes_run_identity(monkeypatch, version_field):
    from llm_benchmark.retrieval import identity

    chunks, scenarios = make_inputs()
    first = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    monkeypatch.setattr(identity, version_field, "test-next-version")
    second = build_keyword_run_identity(chunks=chunks, scenarios=scenarios, k=2)
    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["suite_hash"] == second["suite_hash"]
    assert first["run_fingerprint"] != second["run_fingerprint"]


def make_dense_profile():
    return DenseRetrievalProfile(
        model_id="test/embedder",
        model_revision="a" * 40,
        embedding_dimension=3,
        max_sequence_length=128,
        query_prefix="query: ",
        document_prefix="passage: ",
        device="cpu",
        precision="float32",
        batch_size=8,
        adapter_id="stub",
        adapter_version="1",
    )


def test_generic_keyword_identity_matches_existing_entrypoint():
    chunks, scenarios = make_inputs()

    existing = build_keyword_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
    )
    generic = build_retrieval_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
        profile=KEYWORD_PROFILE,
    )

    assert existing == generic
    assert existing["retriever_profile_hash"] == fingerprint(asdict(KEYWORD_PROFILE))


def test_dense_and_keyword_have_different_run_identity():
    chunks, scenarios = make_inputs()

    keyword = build_keyword_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
    )
    dense = build_retrieval_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
        profile=make_dense_profile(),
    )

    assert keyword["corpus_hash"] == dense["corpus_hash"]
    assert keyword["suite_hash"] == dense["suite_hash"]
    assert keyword["evaluation_config_hash"] == dense["evaluation_config_hash"]

    assert keyword["retriever_profile_hash"] != dense["retriever_profile_hash"]
    assert keyword["run_fingerprint"] != dense["run_fingerprint"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model_id", "test/other"),
        ("model_revision", "b" * 40),
        ("embedding_dimension", 6),
        ("max_sequence_length", 256),
        ("query_prefix", ""),
        ("document_prefix", ""),
        ("device", "cuda:0"),
        ("precision", "float16"),
        ("batch_size", 16),
        ("adapter_id", "other-adapter"),
        ("adapter_version", "2"),
        ("truncation_policy", "truncate"),
    ],
)
def test_dense_profile_changes_identity(field_name, value):
    chunks, scenarios = make_inputs()
    profile = make_dense_profile()

    original = build_retrieval_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
        profile=profile,
    )
    changed = build_retrieval_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
        profile=replace(profile, **{field_name: value}),
    )

    assert original["corpus_hash"] == changed["corpus_hash"]
    assert original["suite_hash"] == changed["suite_hash"]
    assert original["retriever_profile_hash"] != changed["retriever_profile_hash"]
    assert original["run_fingerprint"] != changed["run_fingerprint"]


def test_dense_identity_is_stable():
    chunks, scenarios = make_inputs()
    profile = make_dense_profile()

    first = build_retrieval_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
        profile=profile,
    )
    second = build_retrieval_run_identity(
        chunks=chunks,
        scenarios=scenarios,
        k=2,
        profile=profile,
    )

    assert first == second
