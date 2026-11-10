"""Server-controlled embedding profiles for API-submitted RAG runs.

API clients name a profile; the server supplies the pinned model settings.
Every profile loads from the local cache only.
"""

import importlib.util
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from ..retrieval.run_config import (
    MAX_CONFIG_BYTES,
    DenseEmbeddingConfig,
    _UniqueKeySafeLoader,
)

EMBEDDING_PROFILES_ENV = "LLM_BENCHMARK_EMBEDDING_PROFILES"
DEFAULT_EMBEDDING_PROFILES_PATH = Path("configs/api/embedding_profiles.yaml")
PROFILE_NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class _ProfileFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    profiles: dict[str, DenseEmbeddingConfig] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Unsupported embedding profile file version.")
        return value


class EmbeddingProfileError(ValueError):
    """Raised when the profile file is invalid or a profile is unknown."""


def embedding_profiles_path() -> Path:
    configured = os.environ.get(EMBEDDING_PROFILES_ENV)
    return Path(configured) if configured else DEFAULT_EMBEDDING_PROFILES_PATH


def load_embedding_profiles(path: Path | None = None) -> dict[str, DenseEmbeddingConfig]:
    """Load profiles; a missing file means no dense/hybrid profiles are offered."""
    source = path if path is not None else embedding_profiles_path()
    if not source.exists():
        return {}
    try:
        raw = source.read_bytes()
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError("Profile file too large.")
        payload = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeySafeLoader)
        parsed = _ProfileFile.model_validate(payload)
    except (OSError, ValueError, yaml.YAMLError):
        raise EmbeddingProfileError("Invalid embedding profile file.") from None

    for name, profile in parsed.profiles.items():
        if not re.fullmatch(PROFILE_NAME_PATTERN, name):
            raise EmbeddingProfileError(f"Invalid embedding profile name {name!r}.")
        if profile.local_files_only is not True:
            raise EmbeddingProfileError(
                f"Embedding profile {name!r} must load from the local cache only."
            )
    return dict(parsed.profiles)


def embeddings_available() -> bool:
    """Whether the optional embedding dependency is installed."""
    return importlib.util.find_spec("sentence_transformers") is not None
