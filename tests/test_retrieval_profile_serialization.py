from dataclasses import asdict

import pytest

from llm_benchmark.retrieval.profile import (
    KEYWORD_PROFILE,
    DenseRetrievalProfile,
)
from llm_benchmark.retrieval.profile import BM25_PROFILE
from llm_benchmark.retrieval.profile_serialization import profile_from_dict


def dense_profile():
    return DenseRetrievalProfile(
        model_id="test/encoder",
        model_revision="a" * 40,
        embedding_dimension=2,
        max_sequence_length=4,
        query_prefix="",
        document_prefix="",
        device="cpu",
        precision="float32",
        batch_size=8,
        adapter_id="sentence-transformers-cpu",
        adapter_version="1",
        truncation_policy="reject",
    )


def test_keyword_profile_round_trip():
    assert profile_from_dict(asdict(KEYWORD_PROFILE)) == KEYWORD_PROFILE


def test_dense_profile_round_trip():
    profile = dense_profile()

    assert profile_from_dict(asdict(profile)) == profile


@pytest.mark.parametrize("payload", [None, [], "dense"])
def test_profile_requires_object(payload):
    with pytest.raises(ValueError, match="JSON object"):
        profile_from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", "unknown"),
        ("version", "999"),
        ("similarity", ("dot-product")),
        ("normalization", "none"),
        ("tie_break", "random"),
        ("embedding_dimension", True),
    ],
)
def test_unsupported_recorded_profile_is_rejected(field_name, value):
    payload = asdict(dense_profile())
    payload[field_name] = value

    with pytest.raises(ValueError):
        profile_from_dict(payload)


def test_missing_profile_field_is_rejected():
    payload = asdict(dense_profile())
    del payload["model_revision"]

    with pytest.raises(ValueError):
        profile_from_dict(payload)


def test_unknown_profile_field_is_rejected():
    payload = asdict(dense_profile())
    payload["unexpected"] = "value"

    with pytest.raises(ValueError):
        profile_from_dict(payload)


def test_bm25_profile_round_trip():
    assert profile_from_dict(asdict(BM25_PROFILE))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_version", "unexpected"),
        ("algorithm", "BM25Plus"),
        ("k1", 2.0),
        ("epsilon", 0.5),
        ("tie_break", "input_order"),
        ("candidate_policy", "all-documents"),
        ("corpus_statistics", "nonempty-only"),
    ],
)
def test_modified_bm25_profile_is_rejected(field, value):
    payload = asdict(BM25_PROFILE)
    payload[field] = value

    with pytest.raises(ValueError):
        profile_from_dict(payload)
