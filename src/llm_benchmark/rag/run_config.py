"""Validated local config for RAG runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..config import ModelConfig
from ..retrieval.run_config import (
    MAX_CONFIG_BYTES,
    DenseEmbeddingConfig,
    HybridFusionConfig,
    RetrieverName,
    _UniqueKeySafeLoader,
    validate_retriever_settings,
)
from .prompts import DEFAULT_PROMPT_VERSION, validate_prompt_version


class RagRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    suite_path: StrictStr
    retriever: RetrieverName
    embedding: DenseEmbeddingConfig | None = None
    hybrid: HybridFusionConfig | None = None
    k: StrictInt = Field(gt=0)
    max_context_chars: StrictInt = Field(ge=2)
    provider: Literal["synthetic", "openai_compatible"] = "synthetic"
    model: ModelConfig | None = None
    prompt_version: StrictStr = DEFAULT_PROMPT_VERSION
    structured_output: Literal["none", "json_schema"] = "none"
    output_dir: StrictStr = "outputs/rag"

    @field_validator("prompt_version")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        return validate_prompt_version(value)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Unsupported RAG config version.")
        return value

    @field_validator("suite_path", "output_dir")
    @classmethod
    def validate_path_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Paths must be nonblank valid strings.")
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

    @model_validator(mode="after")
    def validate_provider(self):
        if self.provider == "synthetic":
            if self.model is not None:
                raise ValueError("Synthetic runs must not include a model.")
            if self.structured_output != "none":
                raise ValueError("Synthetic runs do not use structured output.")
            return self

        model = self.model
        if model is None or model.provider != "openai_compatible":
            raise ValueError("An OpenAI-compatible model configuration is required.")
        if model.credential_env_var is not None:
            raise ValueError("RAG generation does not support credentials yet.")
        if model.max_output_tokens is None:
            raise ValueError("RAG generation requires max_output_tokens.")
        if model.temperature != 0:
            raise ValueError("RAG generation currently uses temperature zero.")
        return self


@dataclass(frozen=True)
class LoadedRagRunConfig:
    config: RagRunConfig
    suite_path: Path
    output_dir: Path


def load_rag_run_config(path: Path) -> LoadedRagRunConfig:
    try:
        config_path = path.resolve(strict=True)
        with config_path.open("rb") as stream:
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError("Config too large.")
        payload = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeySafeLoader)
        config = RagRunConfig.model_validate(payload)

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            return candidate.resolve()

        return LoadedRagRunConfig(
            config, resolve(config.suite_path), resolve(config.output_dir)
        )
    except (OSError, ValueError, RuntimeError, yaml.YAMLError):
        raise ValueError("Invalid RAG run configuration.") from None