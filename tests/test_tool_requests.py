from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from llm_benchmark.config import ModelConfig
from llm_benchmark.tool_requests import (
    ToolTurnRequest,
    build_openai_tool_payload,
    serialize_tool_registration,
    serialize_tool_registrations,
)
from llm_benchmark.tool_runtime import ToolDefinition, ToolRegistration
from llm_benchmark.tools import create_example_tool_registry


class ExampleArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    value: int = Field(ge=-10, le=10)


def forbidden_handler(arguments: BaseModel) -> object:
    raise AssertionError("Request preparation must not execute handlers.")


def registration(name: str = "example") -> ToolRegistration:
    return ToolRegistration(
        definition=ToolDefinition(
            name=name,
            description="Synthetic tool for request-mapping tests.",
        ),
        argument_model=ExampleArguments,
        handler=forbidden_handler,
    )


def model_config(**updates: object) -> ModelConfig:
    values: dict[str, object] = {
        "provider": "openai_compatible",
        "endpoint_alias": "synthetic-endpoint",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "synthetic-model",
        "temperature": 0,
        "timeout_seconds": 45,
    }
    values.update(updates)
    return ModelConfig.model_validate(values)


def test_calculator_schema_preserves_argument_contract() -> None:
    registry = create_example_tool_registry()
    calculator = registry.get("calculator")
    assert calculator is not None

    result = serialize_tool_registration(calculator)

    assert set(result) == {"type", "function"}
    assert result["type"] == "function"

    function = result["function"]
    assert set(function) == {"name", "description", "parameters"}
    assert function["name"] == "calculator"
    assert function["description"] == calculator.definition.description

    schema = function["parameters"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"operation", "left", "right"}

    properties = schema["properties"]
    assert properties["operation"]["enum"] == [
        "add",
        "subtract",
        "multiply",
        "divide",
    ]

    for operand in ("left", "right"):
        assert properties[operand]["type"] == "integer"
        assert properties[operand]["minimum"] == -1_000_000
        assert properties[operand]["maximum"] == 1_000_000


def test_serialization_does_not_execute_handler() -> None:
    result = serialize_tool_registration(registration())

    assert result["function"]["name"] == "example"


def test_tools_are_sorted_without_changing_input_order() -> None:
    registrations = (
        registration("zeta"),
        registration("alpha"),
    )

    result = serialize_tool_registrations(registrations)

    assert [tool["function"]["name"] for tool in result] == [
        "alpha",
        "zeta",
    ]
    assert [item.definition.name for item in registrations] == [
        "zeta",
        "alpha",
    ]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [registration()],
        (object(),),
    ],
)
def test_registration_collection_rejects_invalid_types(value: object) -> None:
    with pytest.raises(TypeError):
        serialize_tool_registrations(value)


def test_empty_tool_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        serialize_tool_registrations(())


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        serialize_tool_registrations(
            (
                registration("duplicate"),
                registration("duplicate"),
            )
        )


def test_schema_results_are_independent() -> None:
    selected = registration()

    first = serialize_tool_registration(selected)
    first["function"]["parameters"]["properties"]["value"]["minimum"] = -999

    second = serialize_tool_registration(selected)

    assert second["function"]["parameters"]["properties"]["value"]["minimum"] == -10


@pytest.mark.parametrize("value", [None, {}, object(), ExampleArguments])
def test_single_registration_rejects_wrong_types(value: object) -> None:
    with pytest.raises(TypeError, match="ToolRegistration"):
        serialize_tool_registration(value)


@pytest.mark.parametrize(
    "content", [None, "", " ", "\n\t", 123, True, [], {}, "\ud800"]
)
def test_invalid_user_content(content: object) -> None:
    with pytest.raises(ValueError):
        ToolTurnRequest(user_content=content, registrations=(registration(),))


@pytest.mark.parametrize("content", ["", " ", "\n\t", 123, True, [], {}, "\ud800"])
def test_invalid_system_content(content: object) -> None:
    with pytest.raises(ValueError):
        ToolTurnRequest(
            user_content="Synthetic request.",
            system_content=content,
            registrations=(registration(),),
        )


@pytest.mark.parametrize("selected", [(), (registration(), registration())])
def test_request_rejects_empty_or_duplicate_selection(selected: object) -> None:
    with pytest.raises(ValueError):
        ToolTurnRequest(user_content="Synthetic request.", registrations=selected)


@pytest.mark.parametrize("selected", [None, [], [registration()], (object(),)])
def test_request_rejects_invalid_selection_types(selected: object) -> None:
    with pytest.raises(TypeError):
        ToolTurnRequest(user_content="Synthetic request.", registrations=selected)


@pytest.mark.parametrize("field", ["user_content", "system_content", "registrations"])
def test_request_is_frozen_and_repr_omits_payload(field: str) -> None:
    sentinel = "PRIVATE_TEST_CONTENT"
    request = ToolTurnRequest(
        user_content=sentinel,
        system_content=sentinel,
        registrations=(registration(),),
    )
    assert sentinel not in repr(request)
    assert "forbidden_handler" not in repr(request)
    with pytest.raises(FrozenInstanceError):
        setattr(request, field, None)


def test_exact_default_payload_without_system_message() -> None:
    selected = registration()
    config = model_config()
    request = ToolTurnRequest(
        user_content="  Synthetic request.  ", registrations=(selected,)
    )
    assert request.system_content is None
    assert build_openai_tool_payload(config, request) == {
        "model": "synthetic-model",
        "messages": [{"role": "user", "content": "  Synthetic request.  "}],
        "tools": [serialize_tool_registration(selected)],
        "temperature": 0,
        "stream": False,
    }
    assert config.output_budget_provenance == "provider_default"


def test_system_message_order_and_unicode_preservation() -> None:
    request = ToolTurnRequest(
        system_content="  Türkçe yönerge.  ",
        user_content="İşlem yap 🧪",
        registrations=(registration(),),
    )
    payload = build_openai_tool_payload(model_config(), request)
    assert payload["messages"] == [
        {"role": "system", "content": "  Türkçe yönerge.  "},
        {"role": "user", "content": "İşlem yap 🧪"},
    ]
    json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


@pytest.mark.parametrize("top_p", [0.0, 0.9, 1.0])
def test_explicit_generation_settings(top_p: float) -> None:
    config = model_config(temperature=0.5, top_p=top_p, max_output_tokens=128)
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(registration(),)
    )
    payload = build_openai_tool_payload(config, request)
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == top_p
    assert payload["max_tokens"] == 128
    assert "max_output_tokens" not in payload
    assert config.output_budget_provenance == "fixed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning", "on"),
        ("reasoning", "off"),
        ("reasoning", "auto"),
        ("top_k", 0),
        ("top_k", 20),
        ("min_p", 0.0),
        ("repeat_penalty", 1.0),
    ],
)
def test_unsupported_settings_are_rejected(field: str, value: object) -> None:
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(registration(),)
    )
    with pytest.raises(ValueError, match="Unsupported"):
        build_openai_tool_payload(model_config(**{field: value}), request)


@pytest.mark.parametrize("provider", ["mock", "lm_studio"])
def test_wrong_provider_is_rejected(provider: str) -> None:
    config = model_config(provider=provider, reasoning="off")
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(registration(),)
    )
    with pytest.raises(ValueError, match="requires openai_compatible"):
        build_openai_tool_payload(config, request)


def test_wrong_builder_input_types() -> None:
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(registration(),)
    )
    with pytest.raises(TypeError, match="ModelConfig"):
        build_openai_tool_payload(object(), request)
    with pytest.raises(TypeError, match="ToolTurnRequest"):
        build_openai_tool_payload(model_config(), object())


def test_nonfinite_temperature() -> None:
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(registration(),)
    )
    with pytest.raises(ValueError, match="finite"):
        build_openai_tool_payload(model_config(temperature=float("inf")), request)


def test_payload_and_source_config_independence() -> None:
    config = model_config()
    original = config.model_dump(mode="json")
    canonical = json.dumps(original, sort_keys=True)
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(registration(),)
    )
    first = build_openai_tool_payload(config, request)
    expected = build_openai_tool_payload(config, request)
    first["messages"][0]["content"] = "changed"
    first["tools"][0]["function"]["parameters"]["required"].append("extra")
    assert build_openai_tool_payload(config, request) == expected
    assert json.dumps(config.model_dump(mode="json"), sort_keys=True) == canonical
    assert request.user_content == "Synthetic request."


def test_explicit_tool_selection_and_payload_order() -> None:
    registry = create_example_tool_registry()
    selected = registry.get("calculator")
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(selected,)
    )
    assert [
        t["function"]["name"]
        for t in build_openai_tool_payload(model_config(), request)["tools"]
    ] == ["calculator"]
    request = ToolTurnRequest(
        user_content="Synthetic request.",
        registrations=(registration("zeta"), registration("alpha")),
    )
    assert [
        t["function"]["name"]
        for t in build_openai_tool_payload(model_config(), request)["tools"]
    ] == ["alpha", "zeta"]


@pytest.mark.parametrize("credential_name", [None, "TEST_TOOL_CREDENTIAL"])
def test_no_credential_reads_or_transport_fields(
    monkeypatch: pytest.MonkeyPatch, credential_name: str | None
) -> None:
    sentinel = "SYNTHETIC_CREDENTIAL_SENTINEL"
    monkeypatch.setenv("TEST_TOOL_CREDENTIAL", sentinel)
    config = model_config(credential_env_var=credential_name)
    request = ToolTurnRequest(
        user_content="Synthetic request.", registrations=(registration(),)
    )

    def reject_read(*args: object, **kwargs: object) -> str:
        raise AssertionError("Unexpected environment access")

    with monkeypatch.context() as patch:
        patch.setattr(os, "getenv", reject_read)
        patch.setattr(type(os.environ), "__getitem__", reject_read)
        payload = build_openai_tool_payload(config, request)
    encoded = json.dumps(payload, allow_nan=False)
    assert sentinel not in encoded
    assert "TEST_TOOL_CREDENTIAL" not in encoded
    assert {
        "base_url",
        "endpoint_alias",
        "credential_env_var",
        "Authorization",
        "headers",
        "timeout_seconds",
        "max_output_tokens",
        "top_k",
        "min_p",
        "repeat_penalty",
        "reasoning",
        "tool_choice",
        "parallel_tool_calls",
    }.isdisjoint(payload)


class ChildArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    label: str


class NestedArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    child: ChildArguments


def test_nested_schema_references_are_preserved() -> None:
    selected = ToolRegistration(
        definition=ToolDefinition(name="nested", description="Nested schema."),
        argument_model=NestedArguments,
        handler=forbidden_handler,
    )
    schema = serialize_tool_registration(selected)["function"]["parameters"]
    assert schema == NestedArguments.model_json_schema(mode="validation")
    assert schema["properties"]["child"]["$ref"] == "#/$defs/ChildArguments"
    assert schema["$defs"]["ChildArguments"]["additionalProperties"] is False


def install_schema(monkeypatch: pytest.MonkeyPatch, schema: object) -> None:
    def get_schema(cls: type[BaseModel], *, mode: str) -> object:
        assert mode == "validation"
        return schema

    monkeypatch.setattr(ExampleArguments, "model_json_schema", classmethod(get_schema))


@pytest.mark.parametrize(
    "schema", [None, [], {}, {"type": "array"}, {"type": "string"}]
)
def test_nonobject_schemas_rejected(
    monkeypatch: pytest.MonkeyPatch, schema: object
) -> None:
    install_schema(monkeypatch, schema)
    with pytest.raises(ValueError, match="must describe an object"):
        serialize_tool_registration(registration())


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), object(), b"bytes", "\ud800"]
)
def test_invalid_schema_json_is_safe(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    sentinel = "PRIVATE_SCHEMA_SENTINEL"
    install_schema(monkeypatch, {"type": "object", sentinel: value})
    with pytest.raises(ValueError, match="must be valid JSON") as captured:
        serialize_tool_registration(registration())
    error = captured.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sentinel not in str(error)
    assert sentinel not in "".join(traceback.format_exception(error))


def test_shared_custom_schema_is_not_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    install_schema(monkeypatch, schema)
    first = serialize_tool_registration(registration())
    first["function"]["parameters"]["properties"]["value"]["type"] = "string"
    assert schema["properties"]["value"]["type"] == "integer"
    assert (
        serialize_tool_registration(registration())["function"]["parameters"] == schema
    )


def test_cyclic_schema_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {"type": "object"}
    schema["self"] = schema
    install_schema(monkeypatch, schema)
    with pytest.raises(ValueError, match="must be valid JSON") as captured:
        serialize_tool_registration(registration())
    assert captured.value.__context__ is None


def test_schema_serialization_recursion_error_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Do not assume a fixed nesting depth fails on every Python build.
    install_schema(monkeypatch, {"type": "object"})

    def fail(*args: object, **kwargs: object) -> str:
        raise RecursionError("PRIVATE_SCHEMA_SENTINEL")

    with monkeypatch.context() as patch:
        patch.setattr(json, "dumps", fail)
        with pytest.raises(ValueError, match="must be valid JSON") as captured:
            serialize_tool_registration(registration())
    assert captured.value.__context__ is None
    assert "PRIVATE_SCHEMA_SENTINEL" not in str(captured.value)


class FatalSignal(BaseException):
    pass


@pytest.mark.parametrize(
    "error_type", [RuntimeError, TypeError, ValueError, AssertionError, FatalSignal]
)
def test_schema_hook_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    expected = error_type("trusted hook failure")

    def fail(cls: type[BaseModel], **kwargs: object) -> object:
        raise expected

    monkeypatch.setattr(ExampleArguments, "model_json_schema", classmethod(fail))
    with pytest.raises(error_type) as captured:
        serialize_tool_registration(registration())
    assert captured.value is expected


def test_invalid_message_errors_do_not_retain_content() -> None:
    sentinel = "PRIVATE_MESSAGE_SENTINEL\ud800"
    with pytest.raises(ValueError) as captured:
        ToolTurnRequest(user_content=sentinel, registrations=(registration(),))
    assert "PRIVATE_MESSAGE_SENTINEL" not in str(captured.value)
    assert captured.value.__context__ is None


def test_import_and_build_have_no_runtime_io(tmp_path: Path) -> None:
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
from llm_benchmark.tool_requests import ToolTurnRequest, build_openai_tool_payload
from llm_benchmark.config import ModelConfig
from llm_benchmark.tool_runtime import ToolDefinition, ToolRegistration, ToolRegistry, ToolRuntime
from pydantic import BaseModel, ConfigDict
def fail(*args, **kwargs):
    raise AssertionError('Unexpected tool runtime operation')
ToolRegistry.__init__ = fail
ToolRegistry.get = fail
ToolRuntime.__init__ = fail
ToolRuntime.execute = fail
class Arguments(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    value: int
selected = ToolRegistration(ToolDefinition('example', 'Synthetic tool.'), Arguments, fail)
request = ToolTurnRequest(user_content='Synthetic request.', registrations=(selected,))
config = ModelConfig(provider='openai_compatible', model_id='synthetic', base_url='http://127.0.0.1:1234/v1')
assert build_openai_tool_payload(config, request)['stream'] is False
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
