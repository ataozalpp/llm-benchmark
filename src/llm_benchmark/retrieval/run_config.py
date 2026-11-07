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

from .profile import (
    BM25_PROFILE,
    KEYWORD_PROFILE,
    BM25RetrievalProfile,
    DenseRetrievalProfile,
    HybridRetrievalProfile,
    KeywordRetrieverProfile,
)

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


class HybridFusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_k: StrictInt = Field(default=50, gt=0)
    rank_constant: StrictInt = Field(default=60, gt=0)


RetrieverName = Literal["keyword", "dense", "bm25", "hybrid"]

RetrievalProfile = (
    KeywordRetrieverProfile
    | BM25RetrievalProfile
    | DenseRetrievalProfile
    | HybridRetrievalProfile
)


def validate_retriever_settings(
    *,
    retriever: str,
    embedding: DenseEmbeddingConfig | None,
    hybrid: HybridFusionConfig | None,
    k: int,
) -> None:
    """Shared retriever setting rules for retrieval and RAG run configs."""
    if retriever == "dense" and embedding is None:
        raise ValueError("Dense retrieval requires embedding settings.")

    if retriever == "keyword" and embedding is not None:
        raise ValueError("Keyword retrieval must not include embedding settings.")

    if retriever == "bm25" and embedding is not None:
        raise ValueError("BM25 retrieval must not include embedding settings.")

    if retriever == "hybrid":
        if embedding is None:
            raise ValueError("Hybrid retrieval requires embedding settings.")

        if hybrid is None:
            raise ValueError("Hybrid retrieval requires fusion settings.")

        if k > hybrid.candidate_k:
            raise ValueError("k must not exceed hybrid candidate_k.")

    elif hybrid is not None:
        raise ValueError("Fusion settings are only valid for hybrid retrieval.")


def build_retrieval_profile(
    *,
    retriever: str,
    embedding: DenseEmbeddingConfig | None,
    hybrid: HybridFusionConfig | None,
) -> RetrievalProfile:
    if retriever == "keyword":
        return KEYWORD_PROFILE

    if retriever == "bm25":
        return BM25_PROFILE

    if retriever not in {"dense", "hybrid"}:
        raise ValueError("Unsupported retriever.")

    if embedding is None:
        raise ValueError("Dense embedding settings are missing.")

    dense_profile = embedding.to_profile()

    if retriever == "dense":
        return dense_profile

    if hybrid is None:
        raise ValueError("Hybrid fusion settings are missing.")

    return HybridRetrievalProfile(
        dense=dense_profile,
        candidate_k=hybrid.candidate_k,
        rank_constant=hybrid.rank_constant,
    )


class RetrievalRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    suite_path: StrictStr
    retriever: RetrieverName
    embedding: DenseEmbeddingConfig | None = None
    hybrid: HybridFusionConfig | None = None
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
        validate_retriever_settings(
            retriever=self.retriever,
            embedding=self.embedding,
            hybrid=self.hybrid,
            k=self.k,
        )
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
