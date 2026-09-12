from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import traceback
from collections import UserDict
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import llm_benchmark.tool_calling as tool_calling
from llm_benchmark.tool_calling import (
    NormalizationToolTurn,
    ToolCallNormalizationError,
    ToolCallNormalizationErrorCode,
    normalize_openai_tool_response,
)
from llm_benchmark.tool_runtime import ToolCall


def response(
    *,
    content: object = "Done",
    calls: object = None,
    finish_reason: object = "stop",
) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls,
                },
                "finish_reason": finish_reason,
            }
        ]
    }


def function_call(
    *,
    call_id: str = "call-1",
    name: str = "calculator",
    arguments: object = '{"operation":"add", "left":2,"right":3}',
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def assert_rejected(
    body: object,
    expected: ToolCallNormalizationErrorCode,
    **limits: int,
) -> None:
    with pytest.raises(ToolCallNormalizationError) as captured:
        normalize_openai_tool_response(body, **limits)

    assert captured.value.code is expected


def test_text_only_response_preserves_content() -> None:
    turn = normalize_openai_tool_response(response(content="  Done\n"))

    assert turn.content == "  Done\n"
    assert turn.tool_calls == ()
    assert turn.finish_reason == "stop"


def test_null_content_with_tool_call_is_valid() -> None:
    turn = normalize_openai_tool_response(
        response(
            content=None,
            calls=[function_call()],
            finish_reason="tool_calls",
        )
    )

    assert turn.content is None
    assert turn.finish_reason == "tool_calls"
    assert len(turn.tool_calls) == 1

    call = turn.tool_calls[0]
    assert call.call_id == "call-1"
    assert call.tool_name == "calculator"
    assert call.arguments == {
        "operation": "add",
        "left": 2,
        "right": 3,
    }


def test_text_and_ordered_calls_are_preserved() -> None:
    turn = normalize_openai_tool_response(
        response(
            content="I will use tools.",
            calls=[
                function_call(call_id="call-2", name="unknown_tool"),
                function_call(call_id="call-1"),
            ],
        )
    )

    assert turn.content == "I will use tools."
    assert [call.call_id for call in turn.tool_calls] == [
        "call-2",
        "call-1",
    ]
    assert turn.tool_calls[0].tool_name == "unknown_tool"


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        UserDict(),
        {},
        {"choices": None},
        {"choices": []},
        {"choices": ({},)},
        {"choices": [{}, {}]},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": {"content": "text"}}]},
        {"choices": [{"message": {"role": "user", "content": "text"}}]},
    ],
)
def test_invalid_response_envelopes_are_rejected(body: object) -> None:
    assert_rejected(body, ToolCallNormalizationErrorCode.INVALID_RESPONSE)


@pytest.mark.parametrize("content", [False, 1, [], {}])
def test_invalid_content_is_not_rescued_by_valid_calls(content: object) -> None:
    assert_rejected(
        response(content=content, calls=[function_call()]),
        ToolCallNormalizationErrorCode.INVALID_CONTENT,
    )


@pytest.mark.parametrize("reason", [False, 1, [], {}])
def test_invalid_finish_reason_is_rejected(reason: object) -> None:
    assert_rejected(
        response(finish_reason=reason),
        ToolCallNormalizationErrorCode.INVALID_FINISH_REASON,
    )


@pytest.mark.parametrize("reason", [None, "", "custom_reason", "tool_calls"])
def test_finish_reason_is_preserved_without_inference(reason: str | None) -> None:
    turn = normalize_openai_tool_response(response(finish_reason=reason))
    assert turn.finish_reason == reason
    assert turn.tool_calls == ()


def test_missing_optional_fields_are_supported() -> None:
    turn = normalize_openai_tool_response(
        {"choices": [{"message": {"role": "assistant", "content": "text"}}]}
    )
    assert turn.content == "text"
    assert turn.tool_calls == ()
    assert turn.finish_reason is None

    turn = normalize_openai_tool_response(
        {
            "choices": [
                {"message": {"role": "assistant", "tool_calls": [function_call()]}}
            ]
        }
    )
    assert turn.content is None
    assert len(turn.tool_calls) == 1


@pytest.mark.parametrize("content", [None, "", " \t\n"])
def test_empty_turn_is_rejected_but_valid_calls_are_accepted(
    content: str | None,
) -> None:
    assert_rejected(
        response(content=content), ToolCallNormalizationErrorCode.EMPTY_TURN
    )
    turn = normalize_openai_tool_response(
        response(content=content, calls=[function_call()])
    )
    assert turn.content == content
    assert len(turn.tool_calls) == 1


def test_reasoning_and_unrelated_metadata_are_not_used() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "PRIVATE_REASONING_SENTINEL",
                }
            }
        ]
    }
    assert_rejected(body, ToolCallNormalizationErrorCode.EMPTY_TURN)
    body["choices"][0]["message"]["content"] = "text"
    turn = normalize_openai_tool_response(body)
    assert turn.content == "text"
    assert set(vars(turn)) == {"content", "tool_calls", "finish_reason"}
    assert "PRIVATE_REASONING_SENTINEL" not in repr(vars(turn))


@pytest.mark.parametrize("calls", [{}, "calls", 1, (function_call(),), [None], [{}]])
def test_malformed_call_list_is_rejected_even_with_text(calls: object) -> None:
    assert_rejected(
        response(calls=calls), ToolCallNormalizationErrorCode.INVALID_TOOL_CALL
    )


@pytest.mark.parametrize("field", ["id", "type", "function"])
def test_required_call_fields(field: str) -> None:
    call = function_call()
    del call[field]
    assert_rejected(
        response(calls=[call]), ToolCallNormalizationErrorCode.INVALID_TOOL_CALL
    )


@pytest.mark.parametrize("value", [None, [], "function"])
def test_function_must_be_an_object(value: object) -> None:
    call = function_call()
    call["function"] = value
    assert_rejected(
        response(calls=[call]), ToolCallNormalizationErrorCode.INVALID_TOOL_CALL
    )


@pytest.mark.parametrize("value", [None, 1, "", "custom"])
def test_only_function_tool_type_is_supported(value: object) -> None:
    call = function_call()
    call["type"] = value
    expected = (
        ToolCallNormalizationErrorCode.UNSUPPORTED_TOOL_TYPE
        if type(value) is str
        else ToolCallNormalizationErrorCode.INVALID_TOOL_CALL
    )
    assert_rejected(response(calls=[call]), expected)


@pytest.mark.parametrize("field", ["name", "arguments"])
def test_required_function_fields(field: str) -> None:
    call = function_call()
    del call["function"][field]
    expected = (
        ToolCallNormalizationErrorCode.INVALID_ARGUMENTS
        if field == "arguments"
        else ToolCallNormalizationErrorCode.INVALID_TOOL_CALL
    )
    assert_rejected(response(calls=[call]), expected)


@pytest.mark.parametrize("arguments", [None, {}, [], 1, True, b"{}"])
def test_arguments_must_be_json_text(arguments: object) -> None:
    assert_rejected(
        response(calls=[function_call(arguments=arguments)]),
        ToolCallNormalizationErrorCode.INVALID_ARGUMENTS,
    )


@pytest.mark.parametrize("field", ["id", "name"])
@pytest.mark.parametrize("value", [None, 1, "", "bad value", "tool/name", "a" * 129])
def test_identifiers_use_existing_tool_call_contract(field: str, value: object) -> None:
    call = function_call()
    target = call if field == "id" else call["function"]
    target[field] = value
    assert_rejected(
        response(calls=[call]), ToolCallNormalizationErrorCode.INVALID_TOOL_CALL
    )


def test_identifier_boundaries_and_schema_independence() -> None:
    turn = normalize_openai_tool_response(
        response(
            calls=[
                function_call(
                    call_id="A" * 128,
                    name="a" * 64,
                    arguments='{"unregistered_field":true}',
                )
            ]
        )
    )
    assert turn.tool_calls[0].call_id == "A" * 128
    assert turn.tool_calls[0].tool_name == "a" * 64
    assert turn.tool_calls[0].arguments == {"unregistered_field": True}
    assert_rejected(
        response(calls=[function_call(name="a" * 65)]),
        ToolCallNormalizationErrorCode.INVALID_TOOL_CALL,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "",
        "{",
        "{} trailing",
        "[]",
        "null",
        "42",
        "true",
        '"text"',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e999}',
        '{"value":-1e999}',
        '{"value":"\\ud800"}',
        '{"value":"\ud800"}',
        '{"value":' + "[" * 2000 + "0" + "]" * 2000 + "}",
    ],
)
def test_invalid_json_arguments_are_rejected(arguments: str) -> None:
    assert_rejected(
        response(calls=[function_call(arguments=arguments)]),
        ToolCallNormalizationErrorCode.INVALID_ARGUMENTS,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        '{"a":1,"a":2}',
        '{"nested":{"a":1,"a":2}}',
        '{"items":[{"a":1,"a":2}]}',
        '{"a":1,"\\u0061":2}',
    ],
)
def test_duplicate_argument_keys_are_rejected(arguments: str) -> None:
    assert_rejected(
        response(calls=[function_call(arguments=arguments)]),
        ToolCallNormalizationErrorCode.DUPLICATE_ARGUMENT_KEY,
    )


def test_nested_json_values_are_preserved() -> None:
    arguments = {"a": [None, True, False, 2, 1.25, "ş", {"a": "nested"}], "b": {"a": 3}}
    turn = normalize_openai_tool_response(
        response(
            calls=[
                function_call(
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            ]
        )
    )
    assert turn.tool_calls[0].arguments == arguments


def test_duplicate_call_ids_are_rejected() -> None:
    assert_rejected(
        response(calls=[function_call(), function_call(name="another_tool")]),
        ToolCallNormalizationErrorCode.DUPLICATE_CALL_ID,
    )


def test_bad_second_call_rejects_whole_response() -> None:
    assert_rejected(
        response(
            calls=[
                function_call(),
                function_call(call_id="call-2", arguments="{"),
            ]
        ),
        ToolCallNormalizationErrorCode.INVALID_ARGUMENTS,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "{}",
        "  {} \n",
        '{"value":"ş"}',
        '{"value":"\\u015f"}',
        '{"value":"' + "ş" * 1100 + '"}',
    ],
)
def test_exact_utf8_per_call_limit(arguments: str) -> None:
    size = len(arguments.encode("utf-8"))
    body = response(calls=[function_call(arguments=arguments)])
    turn = normalize_openai_tool_response(body, max_arguments_bytes=size)
    assert turn.tool_calls[0].arguments == json.loads(arguments)
    assert_rejected(
        body,
        ToolCallNormalizationErrorCode.ARGUMENTS_TOO_LARGE,
        max_arguments_bytes=size - 1,
    )


@pytest.mark.parametrize("per_call_limit", [2, 10])
def test_aggregate_limit_is_independent_of_per_call_limit(per_call_limit: int) -> None:
    body = response(
        calls=[
            function_call(call_id="first", arguments="{}"),
            function_call(call_id="second", arguments="{}"),
        ]
    )
    turn = normalize_openai_tool_response(
        body, max_arguments_bytes=per_call_limit, max_total_arguments_bytes=4
    )
    assert len(turn.tool_calls) == 2
    assert_rejected(
        body,
        ToolCallNormalizationErrorCode.TOTAL_ARGUMENTS_TOO_LARGE,
        max_arguments_bytes=per_call_limit,
        max_total_arguments_bytes=3,
    )


def test_size_checks_precede_json_parsing() -> None:
    body = response(calls=[function_call(arguments="{{{")])
    assert_rejected(
        body, ToolCallNormalizationErrorCode.ARGUMENTS_TOO_LARGE, max_arguments_bytes=2
    )
    assert_rejected(
        body,
        ToolCallNormalizationErrorCode.TOTAL_ARGUMENTS_TOO_LARGE,
        max_arguments_bytes=3,
        max_total_arguments_bytes=2,
    )


@pytest.mark.parametrize(
    "limit_name", ["max_arguments_bytes", "max_total_arguments_bytes"]
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "100", None])
def test_invalid_limits_are_configuration_errors(
    limit_name: str, value: object
) -> None:
    with pytest.raises(ValueError, match="Argument byte limits") as captured:
        normalize_openai_tool_response(response(), **{limit_name: value})
    assert type(captured.value) is ValueError


def test_default_per_call_byte_limit() -> None:
    text = '{"value":"' + "a" * (65_536 - len('{"value":""}')) + '"}'
    assert len(text.encode("utf-8")) == 65_536
    assert (
        len(
            normalize_openai_tool_response(
                response(calls=[function_call(arguments=text)])
            ).tool_calls
        )
        == 1
    )
    assert_rejected(
        response(calls=[function_call(arguments=text + " ")]),
        ToolCallNormalizationErrorCode.ARGUMENTS_TOO_LARGE,
    )


def test_default_total_byte_limit() -> None:
    text = '{"value":"' + "a" * (65_536 - len('{"value":""}')) + '"}'
    calls = [
        function_call(call_id=f"call-{index}", arguments=text) for index in range(4)
    ]
    assert len(normalize_openai_tool_response(response(calls=calls)).tool_calls) == 4
    calls.append(function_call(call_id="extra", arguments="{}"))
    assert_rejected(
        response(calls=calls), ToolCallNormalizationErrorCode.TOTAL_ARGUMENTS_TOO_LARGE
    )


def test_source_and_nested_snapshots_are_independent() -> None:
    raw_call = function_call(arguments='{"items":[{"value":1}]}')
    body = response(calls=[raw_call])
    before = copy.deepcopy(body)
    turn = normalize_openai_tool_response(body)
    assert body == before
    raw_call["function"] = {"name": "changed", "arguments": "{}"}
    body.clear()
    returned = turn.tool_calls[0].arguments
    returned["items"][0]["value"] = 999
    assert turn.tool_calls[0].tool_name == "calculator"
    assert turn.tool_calls[0].arguments == {"items": [{"value": 1}]}


@pytest.mark.parametrize(
    "field,value",
    [("content", "changed"), ("tool_calls", ()), ("finish_reason", "changed")],
)
def test_turn_is_frozen(field: str, value: object) -> None:
    turn = normalize_openai_tool_response(response())
    with pytest.raises(FrozenInstanceError):
        setattr(turn, field, value)


@pytest.mark.parametrize(
    "updates",
    [
        {"content": 1},
        {"finish_reason": []},
        {"tool_calls": []},
        {"tool_calls": (object(),)},
    ],
)
def test_direct_constructor_rejects_invalid_types(updates: dict[str, object]) -> None:
    fields = {"content": "text", "tool_calls": (), "finish_reason": None}
    fields.update(updates)
    with pytest.raises(TypeError):
        NormalizationToolTurn(**fields)


@pytest.mark.parametrize("content", [None, "", " \n"])
def test_direct_constructor_rejects_empty_turn(content: str | None) -> None:
    with pytest.raises(ValueError, match="must contain"):
        NormalizationToolTurn(content=content, tool_calls=(), finish_reason=None)


def test_direct_constructor_rejects_duplicate_ids() -> None:
    call = ToolCall(call_id="call-1", tool_name="calculator", arguments={})
    with pytest.raises(ValueError, match="must be unique"):
        NormalizationToolTurn(content=None, tool_calls=(call, call), finish_reason=None)


@pytest.mark.parametrize(
    "arguments", ["broken", "duplicate", "oversized", "invalid_id"]
)
def test_public_errors_do_not_retain_sensitive_details(arguments: str) -> None:
    sentinel = "PRIVATE_PAYLOAD_SENTINEL"
    call = function_call()
    limits = {}
    if arguments == "broken":
        call["function"]["arguments"] = '{"value":"' + sentinel
    elif arguments == "duplicate":
        call["function"]["arguments"] = (
            json.dumps({sentinel: 1})[:-1] + "," + json.dumps(sentinel) + ":2}"
        )
    elif arguments == "oversized":
        call["function"]["arguments"] = json.dumps({"value": sentinel})
        limits = {"max_arguments_bytes": 2}
    else:
        call["id"] = "password=" + sentinel
    with pytest.raises(ToolCallNormalizationError) as captured:
        normalize_openai_tool_response(response(calls=[call]), **limits)
    error = captured.value
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert sentinel not in str(error)
    assert sentinel not in repr(error)
    assert sentinel not in repr(error.args)
    assert sentinel not in repr(vars(error))
    assert sentinel not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("code", list(ToolCallNormalizationErrorCode))
def test_error_contract_uses_only_closed_codes(
    code: ToolCallNormalizationErrorCode,
) -> None:
    error = ToolCallNormalizationError(code)
    assert error.code is code
    assert error.args == (f"Tool-call normalization failed: {code.value}.",)


@pytest.mark.parametrize("code", ["invalid_arguments", "password=private", None, 1])
def test_arbitrary_error_codes_are_rejected(code: object) -> None:
    with pytest.raises(TypeError, match="Invalid normalization error code"):
        ToolCallNormalizationError(code)


def test_turn_repr_omits_payload_fields() -> None:
    sentinel = "PRIVATE_CONTENT_SENTINEL"
    turn = normalize_openai_tool_response(
        response(
            content=sentinel,
            finish_reason=sentinel,
            calls=[function_call(arguments=json.dumps({"value": sentinel}))],
        )
    )
    assert sentinel not in repr(turn)


class FatalSignal(BaseException):
    pass


@pytest.mark.parametrize(
    "error_type", [RuntimeError, TypeError, AttributeError, AssertionError, FatalSignal]
)
def test_unexpected_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    expected = error_type("internal failure")

    def fail(*args: object, **kwargs: object) -> object:
        raise expected

    monkeypatch.setattr(tool_calling, "_normalize_call", fail)
    with pytest.raises(error_type) as captured:
        normalize_openai_tool_response(response(calls=[function_call()]))
    assert captured.value is expected


def test_import_has_no_runtime_initialization_or_io(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import os
import sys
sys.path.insert(0, sys.argv[1])
def reject_io(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.mkdir')):
        raise AssertionError('Unexpected runtime I/O')
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ):
            raise AssertionError('Unexpected file write')
sys.addaudithook(reject_io)
import llm_benchmark.tool_runtime as runtime
def reject_execution(*args, **kwargs):
    raise AssertionError('Unexpected tool initialization or execution')
runtime.ToolRegistry.__init__ = reject_execution
runtime.ToolRegistry.get = reject_execution
runtime.ToolRuntime.__init__ = reject_execution
runtime.ToolRuntime.execute = reject_execution
import llm_benchmark.tool_calling as module
turn = module.normalize_openai_tool_response({'choices': [{'message': {
    'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'call-1', 'type': 'function',
        'function': {'name': 'unregistered_tool', 'arguments': '{}'}
    }]
}}]})
assert len(turn.tool_calls) == 1
for name in ('providers', 'runner', 'trace', 'api', 'worker', 'db', 'tools'):
    assert 'llm_benchmark.' + name not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, source_root],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []
