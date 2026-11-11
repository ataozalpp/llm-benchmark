"""Server-controlled embedding profiles for API RAG runs."""

from pathlib import Path

import pytest
import yaml

from llm_benchmark.rag.embedding_profiles import (
    DEFAULT_EMBEDDING_PROFILES_PATH,
    EMBEDDING_PROFILES_ENV,
    EmbeddingProfileError,
    embedding_profiles_path,
    load_embedding_profiles,
)

ROOT = Path(__file__).parents[1]
PROFILE = {
    "model_id": "test/encoder",
    "model_revision": "a" * 40,
    "embedding_dimension": 4,
    "max_sequence_length": 128,
}


def write(tmp_path, payload):
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_shipped_profiles_load_and_match_the_hybrid_example():
    profiles = load_embedding_profiles(ROOT / DEFAULT_EMBEDDING_PROFILES_PATH)
    example = yaml.safe_load(
        (ROOT / "configs/rag_hybrid_openai_compatible_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )["embedding"]

    profile = profiles["minilm-l6-v2"]
    assert profile.local_files_only is True
    assert profile.model_id == example["model_id"]
    assert profile.model_revision == example["model_revision"]


def test_missing_file_offers_no_profiles(tmp_path):
    assert load_embedding_profiles(tmp_path / "absent.yaml") == {}


def test_environment_overrides_the_profile_path(tmp_path, monkeypatch):
    monkeypatch.delenv(EMBEDDING_PROFILES_ENV, raising=False)
    assert embedding_profiles_path() == DEFAULT_EMBEDDING_PROFILES_PATH
    path = write(tmp_path, {"schema_version": 1, "profiles": {"enc": PROFILE}})
    monkeypatch.setenv(EMBEDDING_PROFILES_ENV, str(path))
    assert set(load_embedding_profiles()) == {"enc"}


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "profiles": {}},
        {"schema_version": 1, "profiles": {"enc": {**PROFILE, "unknown": 1}}},
        {"schema_version": 1, "profiles": {"enc": {**PROFILE, "local_files_only": False}}},
        {"schema_version": 1, "profiles": {"Bad Name": PROFILE}},
        {"schema_version": 1, "profiles": {"enc": {**PROFILE, "model_revision": "main"}}},
        {"schema_version": 1, "extra": True},
    ],
)
def test_invalid_profile_files_are_rejected(tmp_path, payload):
    with pytest.raises(EmbeddingProfileError):
        load_embedding_profiles(write(tmp_path, payload))
