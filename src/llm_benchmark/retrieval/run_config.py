"""Validated local config for retrieval runs."""

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
)

MAX_CONFIG_BYTES = 64 * 1024


class RetrievalRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    suite_path: StrictStr
    retriever: Literal["keyword"]
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
