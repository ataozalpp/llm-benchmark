from dataclasses import asdict, replace

import pytest
from pydantic import ValidationError

from llm_benchmark.retrieval.identity import fingerprint
from llm_benchmark.retrieval.profile import HybridRetrievalProfile
from llm_benchmark.retrieval.profile_serialization import profile_from_dict
from llm_benchmark.retrieval.run_config import RetrievalRunConfig


def payload():
    return {
        "schema_version": 1,
        "suite_path": "suite.json",
        "retriever": "hybrid",
        "k": 2,
        "embedding": {
            "model_id": "test/model",
            "model_revision": "a" * 40,
            "embedding_dimension": 2,
            "max_sequence_length": 256,
        },
        "hybrid": {
            "candidate_k": 5,
            "rank_constant": 60,
        },
    }


def make_profile():
    config = RetrievalRunConfig.model_validate(payload())
    return HybridRetrievalProfile(
        dense=config.embedding.to_profile(),
        candidate_k=config.hybrid.candidate_k,
        rank_constant=config.hybrid.rank_constant,
    )


def test_valid_hybrid_config_and_profile_round_trip():
    profile = make_profile()

    assert profile_from_dict(asdict(profile)) == profile
    assert profile.lexical.backend == "rank-bm25"
    assert profile.candidate_k == 5


@pytest.mark.parametrize("field", ["embedding", "hybrid"])
def test_required_hybrid_settings(field):
    data = payload()
    del data[field]

    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(data)


@pytest.mark.parametrize("value", [0, -1, True, "5"])
@pytest.mark.parametrize("field", ["candidate_k", "rank_constant"])
def test_invalid_fusion_parameters(field, value):
    data = payload()
    data["hybrid"][field] = value

    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(data)


def test_k_cannot_exceed_candidate_depth():
    data = payload()
    data["k"] = 6

    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(data)


def test_dense_config_rejects_unused_fusion_settings():
    data = payload()
    data["retriever"] = "dense"

    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(data)


def test_fusion_and_child_changes_affect_profile_hash():
    profile = make_profile()
    original = fingerprint(asdict(profile))

    variants = (
        replace(profile, candidate_k=10),
        replace(profile, rank_constant=30),
        replace(profile, dense=replace(profile.dense, model_revision="b" * 40)),
    )

    assert all(fingerprint(asdict(variant)) != original for variant in variants)


def test_modified_nested_backend_profile_is_rejected():
    data = asdict(make_profile())
    data["lexical"]["backend_version"] = "unexpected"

    with pytest.raises(ValueError):
        profile_from_dict(data)
