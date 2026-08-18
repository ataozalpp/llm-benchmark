"""Strict request and response contracts for registry routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from llm_benchmark.db.records import DatasetRecord, ModelRecord, ProviderEndpointRecord
from llm_benchmark.db.schemas import ModelCapabilities


_ENV_VAR_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "bearertoken",
    "credential",
    "credentials",
    "password",
    "secret",
    "secret_value",
    "token",
}


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimestampedResponse(StrictSchema):
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class EndpointCreate(StrictSchema):
    name: str = Field(min_length=1, max_length=255)
    provider_type: Literal["mock", "openai_compatible", "lm_studio_native"]
    base_url: str = Field(min_length=1, max_length=2048)
    credential_env_var: str | None = Field(default=None, pattern=_ENV_VAR_PATTERN, max_length=255)

    @model_validator(mode="after")
    def validate_url(self) -> "EndpointCreate":
        if self.provider_type != "mock" and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("non-mock endpoint base_url must use http or https")
        return self


class EndpointPatch(StrictSchema):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    provider_type: Literal["mock", "openai_compatible", "lm_studio_native"] | None = None
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    credential_env_var: str | None = Field(default=None, pattern=_ENV_VAR_PATTERN, max_length=255)

    @model_validator(mode="after")
    def validate_patch(self) -> "EndpointPatch":
        _require_nonempty_patch(self)
        _reject_explicit_nulls(self, {"name", "provider_type", "base_url"})
        return self


class EndpointResponse(TimestampedResponse):
    id: int
    name: str
    provider_type: Literal["mock", "openai_compatible", "lm_studio_native"]
    base_url: str
    credential_env_var: str | None
    is_active: bool

    @classmethod
    def from_record(cls, record: ProviderEndpointRecord) -> "EndpointResponse":
        return cls.model_validate(record.model_dump(mode="python"))


class ModelCreate(StrictSchema):
    name: str = Field(min_length=1, max_length=255)
    model_identifier: str = Field(min_length=1, max_length=512)
    endpoint_id: int = Field(gt=0)
    reasoning_policy: Literal["unsupported", "toggle", "always_on", "provider_managed"]
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    default_generation_config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_secrets(self) -> "ModelCreate":
        _reject_sensitive_mapping(self.default_generation_config)
        _reject_sensitive_mapping(self.metadata)
        return self


class ModelPatch(StrictSchema):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    reasoning_policy: Literal["unsupported", "toggle", "always_on", "provider_managed"] | None = None
    capabilities: ModelCapabilities | None = None
    default_generation_config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> "ModelPatch":
        _require_nonempty_patch(self)
        _reject_explicit_nulls(
            self,
            {"name", "reasoning_policy", "capabilities", "default_generation_config", "metadata"},
        )
        if self.default_generation_config is not None:
            _reject_sensitive_mapping(self.default_generation_config)
        if self.metadata is not None:
            _reject_sensitive_mapping(self.metadata)
        return self


class ModelResponse(TimestampedResponse):
    id: int
    name: str
    model_identifier: str
    endpoint_id: int
    reasoning_policy: Literal["unsupported", "toggle", "always_on", "provider_managed"]
    capabilities: ModelCapabilities
    default_generation_config: dict[str, Any]
    metadata: dict[str, Any]
    is_active: bool

    @field_serializer("capabilities")
    def serialize_capabilities(self, value: ModelCapabilities) -> dict[str, Any]:
        return value.model_dump(exclude_none=True)

    @classmethod
    def from_record(cls, record: ModelRecord) -> "ModelResponse":
        return cls(
            id=record.id,
            name=record.name,
            model_identifier=record.model_identifier,
            endpoint_id=record.endpoint_id,
            reasoning_policy=record.reasoning_policy.value,
            capabilities=ModelCapabilities.model_validate(record.capabilities_json),
            default_generation_config=record.default_generation_config_json,
            metadata=record.metadata_json,
            is_active=record.is_active,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class DatasetCreate(StrictSchema):
    name: str = Field(min_length=1, max_length=255)
    source_type: str = Field(min_length=1, max_length=64)
    source_uri: str = Field(min_length=1, max_length=2048)
    revision: str | None = Field(default=None, max_length=255)
    split: str = Field(min_length=1, max_length=128)
    task_type: str = Field(min_length=1, max_length=128)
    adapter_type: str = Field(min_length=1, max_length=255)
    license: str | None = Field(default=None, max_length=255)
    checksum: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_secrets(self) -> "DatasetCreate":
        _reject_sensitive_mapping(self.metadata)
        return self


class DatasetPatch(StrictSchema):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    license: str | None = Field(default=None, max_length=255)
    checksum: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> "DatasetPatch":
        _require_nonempty_patch(self)
        _reject_explicit_nulls(self, {"name", "metadata"})
        if self.metadata is not None:
            _reject_sensitive_mapping(self.metadata)
        return self


class DatasetResponse(TimestampedResponse):
    id: int
    name: str
    source_type: str
    source_uri: str
    revision: str | None
    split: str
    task_type: str
    adapter_type: str
    license: str | None
    checksum: str | None
    metadata: dict[str, Any]
    is_active: bool

    @classmethod
    def from_record(cls, record: DatasetRecord) -> "DatasetResponse":
        return cls(
            id=record.id,
            name=record.name,
            source_type=record.source_type,
            source_uri=record.source_uri,
            revision=record.revision,
            split=record.split,
            task_type=record.task_type,
            adapter_type=record.adapter_type,
            license=record.license,
            checksum=record.checksum,
            metadata=record.metadata_json,
            is_active=record.is_active,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


def _require_nonempty_patch(value: BaseModel) -> None:
    if not value.model_fields_set:
        raise ValueError("PATCH request must contain at least one field")


def _reject_explicit_nulls(value: BaseModel, fields: set[str]) -> None:
    invalid = sorted(field for field in fields if field in value.model_fields_set and getattr(value, field) is None)
    if invalid:
        raise ValueError("Fields cannot be null: " + ", ".join(invalid))


def _reject_sensitive_mapping(value: dict[str, Any]) -> None:
    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                normalized = str(key).lower().replace("-", "_").replace(" ", "_")
                if normalized in _SENSITIVE_KEYS:
                    raise ValueError("Secret or credential values are not accepted")
                walk(nested)
        elif isinstance(item, list):
            for nested in item:
                walk(nested)

    walk(value)
