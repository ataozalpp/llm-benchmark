from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictModel):
    source: Literal["local", "huggingface"]
    name: str
    path: Path | None = None
    revision: str | None = None
    split: str = "test"
    profile: Literal["smoke", "poc", "full"] = "smoke"
    sample_size: int | None = Field(default=None, gt=0)
    samples_per_category: int = Field(default=10, gt=0)
    category_filter: list[str] = Field(default_factory=list)
    sample_ids: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def validate_source(self) -> "DatasetConfig":
        if self.source == "local" and self.path is None:
            raise ValueError("local dataset source requires path")
        if self.source == "huggingface" and not self.revision:
            raise ValueError("huggingface dataset source requires a pinned revision")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("dataset sample_ids must be unique")
        if self.sample_ids and self.sample_size is not None and self.sample_size != len(self.sample_ids):
            raise ValueError("dataset sample_size must equal the number of explicit sample_ids")
        return self


class ModelConfig(StrictModel):
    provider: Literal["mock", "lm_studio"] = "mock"
    endpoint_alias: str = "mock-local"
    model_id: str
    base_url: str | None = None
    reasoning: Literal["off", "on", "auto"] | None = None
    temperature: float = Field(default=0, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1, exclude_if=lambda value: value is None)
    top_k: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    min_p: float | None = Field(default=None, ge=0, le=1, exclude_if=lambda value: value is None)
    repeat_penalty: float | None = Field(default=None, gt=0, exclude_if=lambda value: value is None)
    max_output_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=120, gt=0)
    scenario_cycle: list[Literal["correct", "incorrect", "unparseable", "request_failed", "missing_usage"]] = Field(
        default_factory=lambda: ["correct"]
    )
    scenario_overrides: dict[str, Literal["correct", "incorrect", "unparseable", "request_failed", "missing_usage"]] = Field(default_factory=dict)
    mock_latency_ms: float = Field(default=12.5, ge=0)

    @property
    def output_budget_provenance(self) -> Literal["fixed", "provider_default"]:
        return "fixed" if self.max_output_tokens is not None else "provider_default"

    @model_validator(mode="after")
    def validate_provider(self) -> "ModelConfig":
        if self.provider == "lm_studio":
            if not self.base_url:
                raise ValueError("lm_studio provider requires base_url")
            if self.reasoning is None:
                raise ValueError("lm_studio provider requires an explicit reasoning mode")
            if not self.base_url.startswith(("http://", "https://")):
                raise ValueError("lm_studio base_url must use http or https")
        return self


class EvaluationConfig(StrictModel):
    prompt_version: str = "mmlu_pro_cot_v1"
    parser_version: str = "mcq_parser_v1"
    evaluator_version: str = "mcq_evaluator_v1"
    scoring_mode: Literal["generated_answer"] = "generated_answer"


class RunConfig(StrictModel):
    schema_version: Literal[1]
    experiment_name: str
    seed: int = 42
    output_dir: Path = Path("outputs")
    dataset: DatasetConfig
    models: list[ModelConfig] = Field(min_length=1)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            raise ValueError(f"cannot override non-object config path: {dotted_key}")
        cursor = nested
    cursor[parts[-1]] = value


def load_config(path: Path, overrides: list[str] | None = None) -> RunConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("config root must be a YAML mapping")
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"CLI override must be key=value: {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(raw, key, yaml.safe_load(raw_value))
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"config validation failed:\n{exc}") from exc


def canonical_config(config: RunConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
