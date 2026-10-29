from dataclasses import asdict

import pytest
import yaml
from pydantic import ValidationError

from llm_benchmark.retrieval.run_config import (
    RetrievalRunConfig,
    load_retrieval_run_config,
)


def embedding_settings():
    return {
        "model_id": "test/model",
        "model_revision": "a" * 40,
        "embedding_dimension": 2,
        "max_sequence_length": 4,
        "batch_size": 8,
    }


def base_config():
    return {
        "schema_version": 1,
        "suite_path": "suite.json",
        "retriever": "keyword",
        "k": 2,
    }


def test_keyword_config_remains_valid():
    config = RetrievalRunConfig.model_validate(base_config())

    assert config.retriever == "keyword"
    assert config.embedding is None


def test_dense_requires_embedding_settings():
    payload = {**base_config(), "retriever": "dense"}

    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(payload)


def test_keyword_rejects_unused_embedding_settings():
    payload = {
        **base_config(),
        "embedding": embedding_settings(),
    }

    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(payload)


def test_dense_config_builds_profile_and_defaults_to_offline():
    payload = {
        **base_config(),
        "retriever": "dense",
        "embedding": embedding_settings(),
    }

    config = RetrievalRunConfig.model_validate(payload)
    profile = config.embedding.to_profile()

    assert config.embedding.local_files_only is True
    assert profile.model_revision == "a" * 40
    assert profile.device == "cpu"
    assert profile.precision == "float32"
    assert profile.truncation_policy == "reject"


def dense_config(**overrides):
    return {
        **base_config(),
        "retriever": "dense",
        "embedding": {**embedding_settings(), **overrides},
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", ""),
        ("model_id", " \t"),
        ("model_id", "bad\x00id"),
        ("model_id", 123),
        ("model_revision", "main"),
        ("model_revision", "a" * 39),
        ("model_revision", "a" * 41),
        ("model_revision", "g" * 40),
        ("model_revision", "a" * 40 + "\n"),
        ("embedding_dimension", 0),
        ("embedding_dimension", -1),
        ("embedding_dimension", True),
        ("embedding_dimension", "2"),
        ("max_sequence_length", 0),
        ("max_sequence_length", -1),
        ("max_sequence_length", True),
        ("max_sequence_length", "4"),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", True),
        ("batch_size", 1.5),
        ("local_files_only", "true"),
        ("local_files_only", 1),
        ("local_files_only", None),
        ("unexpected", "value"),
    ],
)
def test_invalid_embedding_settings_are_rejected(field, value):
    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(dense_config(**{field: value}))


@pytest.mark.parametrize("embedding", [None, {}, [], "model"])
def test_dense_rejects_missing_or_malformed_embedding(embedding):
    payload = {**base_config(), "retriever": "dense", "embedding": embedding}
    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(payload)


def test_misspelled_sequence_length_is_not_accepted_as_alias():
    payload = dense_config()
    settings = payload["embedding"]
    settings["max_seqeunce_length"] = settings.pop("max_sequence_length")

    with pytest.raises(ValidationError) as caught:
        RetrievalRunConfig.model_validate(payload)

    errors = {(tuple(error["loc"]), error["type"]) for error in caught.value.errors()}
    assert (("embedding", "max_sequence_length"), "missing") in errors
    assert (("embedding", "max_seqeunce_length"), "extra_forbidden") in errors


def test_profile_preserves_all_configured_values():
    config = RetrievalRunConfig.model_validate(
        dense_config(
            model_id="example/encoder",
            model_revision="b" * 40,
            embedding_dimension=384,
            max_sequence_length=256,
            batch_size=16,
            local_files_only=False,
        )
    )
    profile = config.embedding.to_profile()

    assert profile.model_id == "example/encoder"
    assert profile.model_revision == "b" * 40
    assert profile.embedding_dimension == 384
    assert profile.max_sequence_length == 256
    assert profile.batch_size == 16
    assert profile.query_prefix == profile.document_prefix == ""
    assert profile.adapter_id == "sentence-transformers-cpu"
    assert profile.adapter_version == "1"
    assert config.embedding.local_files_only is False


def test_load_policy_does_not_change_embedding_profile():
    offline = RetrievalRunConfig.model_validate(dense_config(local_files_only=True))
    online = RetrievalRunConfig.model_validate(dense_config(local_files_only=False))

    assert asdict(offline.embedding.to_profile()) == asdict(
        online.embedding.to_profile()
    )


def test_omitted_batch_size_defaults_to_eight():
    payload = dense_config()
    del payload["embedding"]["batch_size"]
    config = RetrievalRunConfig.model_validate(payload)

    assert config.embedding.batch_size == 8


def test_yaml_dense_config_resolves_paths_relative_to_config(tmp_path, monkeypatch):
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "dense.yaml"
    payload = dense_config()
    payload["suite_path"] = "../data/suite.json"
    payload["output_dir"] = "../outputs/retrieval"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    loaded = load_retrieval_run_config(path)

    assert loaded.config.retriever == "dense"
    assert loaded.config.embedding.max_sequence_length == 4
    assert loaded.suite_path == (tmp_path / "data/suite.json").resolve()
    assert loaded.output_dir == (tmp_path / "outputs/retrieval").resolve()


def test_duplicate_nested_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "dense.yaml"
    content = yaml.safe_dump(dense_config())
    content = content.replace("  batch_size: 8", "  batch_size: 8\n  batch_size: 16")
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid retrieval run configuration"):
        load_retrieval_run_config(path)


@pytest.mark.parametrize("policy", ["reject", "truncate"])
def truncation_policy_is_preserved_in_profile(policy):
    config = RetrievalRunConfig.model_validate(
        dense_config(truncation_policy=policy)
    )

    assert config.embedding.truncation_policy == policy
    assert config.embedding.to_profile().truncation_policy == policy


@pytest.mark.parametrize("policy", ["ignore", "", None, True, 1])
def test_invalid_truncation_policy_is_rejected(policy):
    with pytest.raises(ValidationError):
        RetrievalRunConfig.model_validate(dense_config(truncation_policy=policy))


def test_truncation_policy_changes_profile_fingerprint():
    from llm_benchmark.retrieval.identity import fingerprint

    reject = RetrievalRunConfig.model_validate(dense_config(truncation_policy="reject"))
    truncate = RetrievalRunConfig.model_validate(dense_config(truncation_policy="truncate"))

    assert fingerprint(asdict(reject.embedding.to_profile())) != fingerprint(
        asdict(truncate.embedding.to_profile())
    )