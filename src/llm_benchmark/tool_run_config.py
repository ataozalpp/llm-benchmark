"""Separate, versioned CLI configuration; MCQ config hashes are unchanged."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ModelConfig


class ToolRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    suite: Literal["example-tools-v1"] = "example-tools-v1"
    provider: Literal["synthetic", "openai_compatible"] = "synthetic"
    model: ModelConfig | None = None
    max_wall_time_seconds: float = Field(default=60, gt=0, le=3600, allow_inf_nan=False)
    output_dir: Path = Path("outputs/tool")

    @model_validator(mode="after")
    def validate_provider(self) -> "ToolRunConfig":
        if self.provider == "synthetic" and self.model is not None:
            raise ValueError("Synthetic runs must not include a model configuration.")
        if self.provider == "openai_compatible":
            if self.model is None or self.model.provider != "openai_compatible":
                raise ValueError(
                    "An OpenAI-compatible model configuration is required."
                )
            if self.model.timeout_seconds > self.max_wall_time_seconds:
                raise ValueError(
                    "Provider timeout must not exceed the scenario budget."
                )
        return self


def load_tool_run_config(path: Path) -> ToolRunConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ToolRunConfig.model_validate(payload)
    except (OSError, yaml.YAMLError, ValueError):
        raise ValueError("Invalid tool-run configuration.") from None
