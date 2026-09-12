from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .tool_calling import (
    NormalizationToolTurn,
    ToolCallNormalizationErrorCode,
)


class ToolProviderStatus(StrEnum):
    SUCCEEDED = "succeeded"
    REQUEST_FAILED = "request_failed"
    RESPONSE_INVALID = "response_invalid"


class ToolProviderErrorCode(StrEnum):
    MISSING_CREDENTIAL = "missing_credential"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    MALFORMED_JSON = "malformed_json"


@dataclass(frozen=True)
class ToolProviderTelemetry:
    """Transport duration and reported usage; missing measurements stay null."""

    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    ttft_ms: float | None = None
    throughput_tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        for value in (
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.total_tokens,
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    "Token measurements must be non-negative integers or null."
                )
        for value in (self.latency_ms, self.ttft_ms, self.throughput_tokens_per_second):
            if value is None:
                continue
            if type(value) not in (int, float) or value < 0:
                raise ValueError("Timing measurements must be finite and non-negative.")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("Timing measurements must be finite and non-negative.")
        if self.latency_ms is None:
            raise ValueError("Transport latency is required.")


@dataclass(frozen=True)
class ToolProviderResult:
    status: ToolProviderStatus
    turn: NormalizationToolTurn | None = field(
        default=None,
        repr=False,
    )
    error_code: ToolProviderErrorCode | None = None
    normalization_error_code: ToolCallNormalizationErrorCode | None = None
    telemetry: ToolProviderTelemetry | None = None

    def __post_init__(self) -> None:
        if (
            self.telemetry is not None
            and type(self.telemetry) is not ToolProviderTelemetry
        ):
            raise TypeError("Invalid tool-provider telemetry.")
        if type(self.status) is not ToolProviderStatus:
            raise TypeError("Invalid tool-provider status.")

        if self.turn is not None and type(self.turn) is not NormalizationToolTurn:
            raise TypeError("Invalid normalized tool turn.")

        if (
            self.error_code is not None
            and type(self.error_code) is not ToolProviderErrorCode
        ):
            raise TypeError("Invalid tool-provider error code.")

        if (
            self.normalization_error_code is not None
            and type(self.normalization_error_code)
            is not ToolCallNormalizationErrorCode
        ):
            raise TypeError("Invalid normalization error code.")

        if self.status is ToolProviderStatus.SUCCEEDED:
            if self.turn is None:
                raise ValueError("Successful results require a turn.")

            if self.error_code is not None or self.normalization_error_code is not None:
                raise ValueError("Successful results cannot contain errors.")

        elif self.status is ToolProviderStatus.REQUEST_FAILED:
            if self.turn is not None:
                raise ValueError("Failed requests cannot contain a turn.")

            if self.error_code is None:
                raise ValueError("Failed requests require an error code.")

            if self.normalization_error_code is not None:
                raise ValueError(
                    "Failed requests cannot contain a normalization error."
                )

        elif self.status is ToolProviderStatus.RESPONSE_INVALID:
            if self.turn is not None:
                raise ValueError("Invalid responses cannot contain a turn.")

            if self.error_code is not None:
                raise ValueError("Invalid responses use normalization error codes.")

            if self.normalization_error_code is None:
                raise ValueError("Invalid responses require a normalization error.")
