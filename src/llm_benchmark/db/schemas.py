from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelCapabilities(BaseModel):
    """Known capability flags accepted at the registry boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_output: bool | None = None
    streaming: bool | None = None
    tool_calling: bool | None = None
    structured_output: bool | None = None
    vision: bool | None = None
    embeddings: bool | None = None
