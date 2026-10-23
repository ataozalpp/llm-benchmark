from dataclasses import asdict, replace

import pytest

from llm_benchmark.retrieval.profile import DenseRetrievalProfile


def make_profile():
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


def test_profile_preserves_prefix_whitespace():
    profile = make_profile()

    assert profile.query_prefix == "query: "
    assert profile.document_prefix == "passage: "
    assert asdict(profile)["similarity"] == "cosine"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model_id", "")("model_revision", "main"),
        ("model_revision", "latest"),
        ("embedding_dimension", 0),
        ("embedding_dimension", True),
        ("embedding_dimension", -1),
        ("batch_size", "8"),
        ("batch_size", False),
        ("query_prefix", None),
        ("document_prefix", "\x00"),
        ("adapter_version", " "),
        ("truncation_policy", "silent"),
    ],
)
def test_invalid_profile_fields(field_name, value):
    with pytest.raises(ValueError):
        replace(make_profile(), **{field_name: value})


def test_empty_prefixes_are_supported():
    profile = replace(
        make_profile(),
        query_prefix="",
        document_prefix="",
    )

    assert profile.query_prefix == ""
    assert profile.document_prefix == ""
