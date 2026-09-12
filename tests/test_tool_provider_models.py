from dataclasses import FrozenInstanceError

import pytest

from llm_benchmark.tool_calling import (
    NormalizationToolTurn,
    ToolCallNormalizationErrorCode,
)
from llm_benchmark.tool_provider_models import (
    ToolProviderErrorCode,
    ToolProviderResult,
    ToolProviderStatus,
    ToolProviderTelemetry,
)


def text_turn() -> NormalizationToolTurn:
    return NormalizationToolTurn(
        content="Synthetic final text.",
        tool_calls=(),
        finish_reason="stop",
    )


def test_successful_result() -> None:
    turn = text_turn()

    result = ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=turn,
    )

    assert result.turn is turn
    assert result.error_code is None
    assert result.normalization_error_code is None


@pytest.mark.parametrize("code", list(ToolProviderErrorCode))
def test_failed_request_result(code: ToolProviderErrorCode) -> None:
    result = ToolProviderResult(
        status=ToolProviderStatus.REQUEST_FAILED,
        error_code=code,
    )

    assert result.turn is None
    assert result.error_code is code


@pytest.mark.parametrize("code", list(ToolCallNormalizationErrorCode))
def test_invalid_response_result(
    code: ToolCallNormalizationErrorCode,
) -> None:
    result = ToolProviderResult(
        status=ToolProviderStatus.RESPONSE_INVALID,
        normalization_error_code=code,
    )

    assert result.turn is None
    assert result.normalization_error_code is code


@pytest.mark.parametrize(
    "values",
    [
        {
            "status": ToolProviderStatus.SUCCEEDED,
        },
        {
            "status": ToolProviderStatus.SUCCEEDED,
            "turn": text_turn(),
            "error_code": ToolProviderErrorCode.TIMEOUT,
        },
        {
            "status": ToolProviderStatus.SUCCEEDED,
            "turn": text_turn(),
            "normalization_error_code": ToolCallNormalizationErrorCode.INVALID_RESPONSE,
        },
        {
            "status": ToolProviderStatus.REQUEST_FAILED,
        },
        {
            "status": ToolProviderStatus.REQUEST_FAILED,
            "turn": text_turn(),
            "error_code": ToolProviderErrorCode.TIMEOUT,
        },
        {
            "status": ToolProviderStatus.REQUEST_FAILED,
            "error_code": ToolProviderErrorCode.TIMEOUT,
            "normalization_error_code": ToolCallNormalizationErrorCode.INVALID_RESPONSE,
        },
        {
            "status": ToolProviderStatus.RESPONSE_INVALID,
        },
        {
            "status": ToolProviderStatus.RESPONSE_INVALID,
            "turn": text_turn(),
            "normalization_error_code": ToolCallNormalizationErrorCode.INVALID_RESPONSE,
        },
        {
            "status": ToolProviderStatus.RESPONSE_INVALID,
            "error_code": ToolProviderErrorCode.TIMEOUT,
            "normalization_error_code": ToolCallNormalizationErrorCode.INVALID_RESPONSE,
        },
    ],
)
def test_invalid_state_combinations(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ToolProviderResult(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"status": "succeeded", "turn": text_turn()},
        {"status": ToolProviderStatus.SUCCEEDED, "turn": {}},
        {
            "status": ToolProviderStatus.REQUEST_FAILED,
            "error_code": "timeout",
        },
        {
            "status": ToolProviderStatus.RESPONSE_INVALID,
            "normalization_error_code": "invalid_response",
        },
    ],
)
def test_wrong_field_types(values: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        ToolProviderResult(**values)


def test_result_is_frozen_and_repr_omits_turn() -> None:
    result = ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED,
        turn=text_turn(),
    )

    assert "Synthetic final text." not in repr(result)

    with pytest.raises(FrozenInstanceError):
        result.turn = None


def test_status_spelling() -> None:
    assert ToolProviderStatus.SUCCEEDED.value == "succeeded"


@pytest.mark.parametrize(
    "field", ["input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"]
)
@pytest.mark.parametrize(
    "value", [-1, True, False, 1.5, "1", float("nan"), float("inf")]
)
def test_invalid_token_measurements(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ToolProviderTelemetry(latency_ms=0, **{field: value})


@pytest.mark.parametrize(
    "field", ["latency_ms", "ttft_ms", "throughput_tokens_per_second"]
)
@pytest.mark.parametrize(
    "value", [-1, True, "1", float("nan"), float("inf"), float("-inf"), 10**1000]
)
def test_invalid_timing_measurements(field: str, value: object) -> None:
    values = {"latency_ms": 0, field: value}
    with pytest.raises(ValueError):
        ToolProviderTelemetry(**values)


def test_telemetry_nullable_and_frozen() -> None:
    telemetry = ToolProviderTelemetry(latency_ms=1.5, input_tokens=0, output_tokens=4)
    assert telemetry.reasoning_tokens is None
    assert telemetry.total_tokens is None
    assert telemetry.ttft_ms is None
    assert telemetry.throughput_tokens_per_second is None
    with pytest.raises(FrozenInstanceError):
        telemetry.input_tokens = 1
    result = ToolProviderResult(
        status=ToolProviderStatus.SUCCEEDED, turn=text_turn(), telemetry=telemetry
    )
    assert result.telemetry is telemetry
    with pytest.raises(TypeError):
        ToolProviderResult(
            status=ToolProviderStatus.SUCCEEDED, turn=text_turn(), telemetry={}
        )
    with pytest.raises(ValueError):
        ToolProviderTelemetry(latency_ms=None)
