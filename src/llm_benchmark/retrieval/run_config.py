"""Validated local config for retrieval runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .profile import DenseRetrievalProfile

MAX_CONFIG_BYTES = 64 * 1024


class DenseEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: StrictStr = Field(min_length=1)
    model_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_dimension: StrictInt = Field(gt=0)
    max_sequence_length: StrictInt = Field(gt=0)
    batch_size: StrictInt = Field(default=8, gt=0)
    local_files_only: StrictBool = True
    truncation_policy: Literal["reject", "truncate"] = "reject"

    def to_profile(self) -> DenseRetrievalProfile:
        return DenseRetrievalProfile(
            model_id=self.model_id,
            model_revision=self.model_revision,
            embedding_dimension=self.embedding_dimension,
            max_sequence_length=self.max_sequence_length,
            query_prefix="",
            document_prefix="",
            device="cpu",
            precision="float32",
            batch_size=self.batch_size,
            adapter_id="sentence-transformers-cpu",
            adapter_version="1",
            truncation_policy=self.truncation_policy,
        )

    @model_validator(mode="after")
    def validate_profile(self):
        self.to_profile()
        return self


class RetrievalRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    suite_path: StrictStr
    retriever: Literal["keyword", "dense", "bm25"]
    embedding: DenseEmbeddingConfig | None = None
    k: StrictInt = Field(gt=0)
    output_dir: StrictStr = "outputs/retrieval"

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Unsupported retrieval config version.")
        return value

    @field_validator("suite_path", "output_dir")
    @classmethod
    def validate_path_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Paths must be non blank valid strings.")
        return value

    @model_validator(mode="after")
    def validate_retriever_settings(self):
        if self.retriever == "dense" and self.embedding is None:
            raise ValueError("Dense retrieval requires embedding settings.")

        if self.retriever == "keyword" and self.embedding is not None:
            raise ValueError("Keyword retrieval must not include embedding settings.")

        if self.retriever == "bm25" and self.embedding is not None:
            raise ValueError("BM25 retrieval must not include embedding settings.")

        return self


@dataclass(frozen=True)
class LoadedRetrievalRunConfig:
    config: RetrievalRunConfig
    suite_path: Path
    output_dir: Path


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate keys without changing the global YAML loader."""


def _construct_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("Invalid configuration key.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_retrieval_run_config(path: Path) -> LoadedRetrievalRunConfig:
    try:
        config_path = path.resolve(strict=True)
        with config_path.open("rb") as stream:
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError("Config too large.")
        payload = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeySafeLoader)
        config = RetrievalRunConfig.model_validate(payload)

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            return candidate.resolve()

        return LoadedRetrievalRunConfig(
            config, resolve(config.suite_path), resolve(config.output_dir)
        )
    except (OSError, ValueError, RuntimeError, yaml.YAMLError):
        raise ValueError("Invalid retrieval run configuration.") from None
