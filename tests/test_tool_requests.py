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
from llm_benchmark.tool_calling import NormalizationToolTurn
from llm_benchmark.tool_conversation import (
    AssistantMessage,
    SystemMessage,
    ToolConversation,
    ToolResultMessage,
    UserMessage,
    serialize_conversation,
)
from llm_benchmark.tool_descriptors import (
    ToolDescriptor,
    descriptor_from_registration,
)
from llm_benchmark.tool_requests import (
    ToolConversationRequest,
    ToolTurnRequest,
    build_openai_tool_conversation_payload,
    build_openai_tool_payload,
    serialize_tool_descriptors,
    serialize_tool_registration,
    serialize_tool_registrations,
)
from llm_benchmark.tool_runtime import (
    ToolCall,
    ToolDefinition,
    ToolExecutionStatus,
    ToolRegistration,
    ToolResult,
)
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


def descriptor(name: str = "example") -> ToolDescriptor:
    return descriptor_from_registration(registration(name))


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


def completed_conversation() -> ToolConversation:
    call = ToolCall(call_id="call_1", tool_name="example", arguments={"value": 3})
    return ToolConversation(
        messages=(
            UserMessage("Synthetic task."),
            AssistantMessage(
                NormalizationToolTurn(
                    content=None,
                    tool_calls=(call,),
                    finish_reason="tool_calls",
                )
            ),
            ToolResultMessage(
                ToolResult(
                    call_id="call_1",
                    tool_name="example",
                    status=ToolExecutionStatus.SUCCEEDED,
                    output={"result": 3},
                )
            ),
        )
    )


@pytest.mark.parametrize("system", [None, "  Türkçe yönerge.  "])
def test_conversation_initial_payload_matches_existing_builder(
    system: str | None,
) -> None:
    selected = (registration(),)
    user = "  Synthetic task.  "
    messages = (
        (UserMessage(user),)
        if system is None
        else (SystemMessage(system), UserMessage(user))
    )
    cfg = model_config()
    expected = build_openai_tool_payload(
        cfg,
        ToolTurnRequest(
            user_content=user,
            system_content=system,
            registrations=selected,
        ),
    )
    actual = build_openai_tool_conversation_payload(
        cfg,
        ToolConversationRequest(
            conversation=ToolConversation(messages),
            registrations=selected,
        ),
    )
    assert actual == expected


def test_conversation_complete_exchange_and_independent_payloads() -> None:
    conversation = completed_conversation()
    req = ToolConversationRequest(
        conversation=conversation, registrations=(registration(),)
    )
    cfg = model_config()
    before = cfg.model_dump_json()
    payload = build_openai_tool_conversation_payload(cfg, req)
    expected = {
        "model": "synthetic-model",
        "messages": serialize_conversation(conversation),
        "tools": serialize_tool_registrations(req.registrations),
        "temperature": 0,
        "stream": False,
    }
    assert payload == expected
    assert [item["role"] for item in payload["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert cfg.output_budget_provenance == "provider_default"
    payload["messages"][0]["content"] = "changed"
    payload["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{}"
    payload["tools"][0]["function"]["parameters"]["required"].append("extra")
    assert build_openai_tool_conversation_payload(cfg, req) == expected
    assert serialize_conversation(conversation) == expected["messages"]
    assert cfg.model_dump_json() == before
    with pytest.raises(FrozenInstanceError):
        req.conversation = None
    assert "Synthetic task." not in repr(req)


@pytest.mark.parametrize("terminal", [False, True])
def test_conversation_request_requires_provider_readiness(terminal: bool) -> None:
    conversation = ToolConversation(completed_conversation().messages[:2])
    if terminal:
        conversation = ToolConversation(
            (
                UserMessage("Task"),
                AssistantMessage(
                    NormalizationToolTurn(
                        content="Final", tool_calls=(), finish_reason="stop"
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="not ready"):
        ToolConversationRequest(
            conversation=conversation, registrations=(registration(),)
        )


def test_conversation_request_types_and_registration_validation() -> None:
    with pytest.raises(TypeError, match="ToolConversation"):
        ToolConversationRequest(
            conversation=registration(), registrations=(registration(),)
        )
    for selected in ((), (registration(), registration())):
        with pytest.raises(ValueError):
            ToolConversationRequest(
                conversation=completed_conversation(), registrations=selected
            )
    with pytest.raises(TypeError, match="tuple"):
        ToolConversationRequest(
            conversation=completed_conversation(), registrations=[registration()]
        )
    with pytest.raises(TypeError, match="ToolConversationRequest"):
        build_openai_tool_conversation_payload(model_config(), object())


def test_conversation_generation_policy_is_shared() -> None:
    req = ToolConversationRequest(
        conversation=completed_conversation(), registrations=(registration(),)
    )
    cfg = model_config(max_output_tokens=128, top_p=0.9, temperature=0.5)
    payload = build_openai_tool_conversation_payload(cfg, req)
    assert payload["max_tokens"] == 128
    assert payload["top_p"] == 0.9
    assert payload["temperature"] == 0.5
    assert cfg.output_budget_provenance == "fixed"
    with pytest.raises(ValueError, match="Unsupported"):
        build_openai_tool_conversation_payload(model_config(reasoning="off"), req)
    with pytest.raises(ValueError, match="openai_compatible"):
        build_openai_tool_conversation_payload(model_config(provider="mock"), req)


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
from llm_benchmark.tool_requests import (
    ToolTurnRequest, build_openai_tool_payload,
    ToolConversationRequest, build_openai_tool_conversation_payload,
)
from llm_benchmark.tool_conversation import ToolConversation, UserMessage
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
conversation_request = ToolConversationRequest(
    conversation=ToolConversation((UserMessage('Synthetic request.'),)),
    registrations=(selected,),
)
assert build_openai_tool_conversation_payload(config, conversation_request)['stream'] is False
from llm_benchmark.tool_descriptors import ToolDescriptor
descriptor = ToolDescriptor(
    definition=ToolDefinition('external_example', 'Synthetic description.'),
    parameters={'type': 'object', '$ref': 'https://example.invalid/schema'},
)
request = ToolTurnRequest(user_content='Synthetic request.', descriptors=(descriptor,))
assert build_openai_tool_payload(config, request)['tools'][0]['function']['name'] == 'external_example'
conversation_request = ToolConversationRequest(
    conversation=ToolConversation((UserMessage('Synthetic request.'),)),
    descriptors=(descriptor,),
)
assert build_openai_tool_conversation_payload(config, conversation_request)['stream'] is False
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


def test_descriptor_serialization_is_sorted_without_mutating_input():
    selected = (
        descriptor("zeta"),
        descriptor("alpha"),
    )

    result = serialize_tool_descriptors(selected)

    assert [item["function"]["name"] for item in result] == ["alpha", "zeta"]

    assert [item.definition.name for item in selected] == ["zeta", "alpha"]


@pytest.mark.parametrize(
    "value",
    [None, [], [descriptor()], (object(),)],
)
def test_descriptor_collection_rejects_invalid_types(value):
    with pytest.raises(TypeError):
        serialize_tool_descriptors(value)


def test_descriptor_collection_rejects_empty_selection():
    with pytest.raises(ValueError, match="At least one"):
        serialize_tool_descriptors(())


def test_descriptor_collection_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        serialize_tool_descriptors(
            (
                descriptor("duplicate"),
                descriptor("duplicate"),
            )
        )


def descriptor_request(kind, **sources):
    if kind == "initial":
        return ToolTurnRequest(user_content="Synthetic task.", **sources)
    return ToolConversationRequest(
        conversation=ToolConversation((UserMessage("Synthetic task."),)),
        **sources,
    )


def descriptor_payload(kind, request, config=None):
    builder = (
        build_openai_tool_payload
        if kind == "initial"
        else build_openai_tool_conversation_payload
    )
    return builder(model_config() if config is None else config, request)


@pytest.mark.parametrize("kind", ["initial", "conversation"])
def test_handler_free_descriptor_payload(kind, monkeypatch):
    selected = ToolDescriptor(
        definition=ToolDefinition("external_example", "Synthetic description."),
        parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
    )

    def reject(*args, **kwargs):
        raise AssertionError("Local schema hooks must not run.")

    monkeypatch.setattr(ExampleArguments, "model_json_schema", reject)
    request = descriptor_request(kind, descriptors=(selected,))
    payload = descriptor_payload(kind, request)
    assert request.registrations == ()
    assert payload["tools"] == serialize_tool_descriptors((selected,))
    assert payload["messages"] == [{"role": "user", "content": "Synthetic task."}]


@pytest.mark.parametrize("kind", ["initial", "conversation"])
def test_two_nonempty_sources_rejected(kind):
    with pytest.raises(ValueError, match="exactly one"):
        descriptor_request(
            kind, registrations=(registration(),), descriptors=(descriptor(),)
        )


@pytest.mark.parametrize("kind", ["initial", "conversation"])
def test_missing_tool_sources_rejected(kind):
    with pytest.raises(ValueError, match="At least one"):
        descriptor_request(kind)


@pytest.mark.parametrize("kind", ["initial", "conversation"])
@pytest.mark.parametrize("selected", [None, [], [descriptor()], (object(),)])
def test_invalid_descriptor_request_types(kind, selected):
    with pytest.raises(TypeError):
        descriptor_request(kind, descriptors=selected)


@pytest.mark.parametrize("kind", ["initial", "conversation"])
@pytest.mark.parametrize("source", ["registrations", "descriptors"])
@pytest.mark.parametrize("invalid", [None, []])
def test_unused_source_still_requires_tuple(kind, source, invalid):
    sources = (
        {"registrations": invalid, "descriptors": (descriptor(),)}
        if source == "registrations"
        else {"registrations": (registration(),), "descriptors": invalid}
    )
    with pytest.raises(TypeError, match="tuple"):
        descriptor_request(kind, **sources)


@pytest.mark.parametrize("kind", ["initial", "conversation"])
def test_duplicate_request_descriptors_rejected(kind):
    with pytest.raises(ValueError, match="unique"):
        descriptor_request(kind, descriptors=(descriptor(), descriptor()))


@pytest.mark.parametrize("kind", ["initial", "conversation"])
def test_descriptor_request_payload_parity(kind):
    selected = registration()
    local = descriptor_request(kind, registrations=(selected,))
    described = descriptor_request(
        kind, descriptors=(descriptor_from_registration(selected),)
    )
    assert descriptor_payload(kind, local) == descriptor_payload(kind, described)


@pytest.mark.parametrize("kind", ["initial", "conversation"])
def test_descriptor_request_snapshot_and_repr(kind):
    selected = ToolDescriptor(
        definition=ToolDefinition("example", "PRIVATE_DESCRIPTOR_SENTINEL"),
        parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
    )
    request = descriptor_request(kind, descriptors=(selected,))
    payload = descriptor_payload(kind, request)
    payload["tools"][0]["function"]["parameters"]["properties"].clear()
    payload["messages"][0]["content"] = "changed"
    fresh = descriptor_payload(kind, request)
    assert "value" in fresh["tools"][0]["function"]["parameters"]["properties"]
    assert fresh["messages"][0]["content"] == "Synthetic task."
    assert "PRIVATE_DESCRIPTOR_SENTINEL" not in repr(request)
    with pytest.raises(FrozenInstanceError):
        request.descriptors = ()


@pytest.mark.parametrize("terminal", [False, True])
def test_descriptor_conversation_requires_readiness(terminal):
    conversation = ToolConversation(completed_conversation().messages[:2])
    if terminal:
        conversation = ToolConversation(
            (
                UserMessage("Task"),
                AssistantMessage(
                    NormalizationToolTurn(
                        content="Final", tool_calls=(), finish_reason="stop"
                    )
                ),
            )
        )
    with pytest.raises(ValueError, match="not ready"):
        ToolConversationRequest(conversation=conversation, descriptors=(descriptor(),))


def test_legacy_positional_constructors_and_keyword_only_descriptors():
    selected = (registration(),)
    initial = ToolTurnRequest("Task", selected, "System")
    conversation = ToolConversation((UserMessage("Task"),))
    request = ToolConversationRequest(conversation, selected)
    assert initial.system_content == "System"
    assert initial.registrations == request.registrations == selected
    assert initial.descriptors == request.descriptors == ()
    with pytest.raises(TypeError):
        ToolTurnRequest("Task", (), None, (descriptor(),))
    with pytest.raises(TypeError):
        ToolConversationRequest(conversation, (), (descriptor(),))


@pytest.mark.parametrize("kind", ["initial", "conversation"])
def test_descriptor_generation_settings_unchanged(kind):
    request = descriptor_request(kind, descriptors=(descriptor(),))
    payload = descriptor_payload(
        kind, request, model_config(max_output_tokens=128, temperature=0.5, top_p=0.9)
    )
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["stream"] is False
    default = descriptor_payload(kind, request)
    assert "max_tokens" not in default
    assert "top_p" not in default
    with pytest.raises(ValueError, match="Unsupported"):
        descriptor_payload(kind, request, model_config(reasoning="off"))


def test_local_schema_path_keeps_existing_size_policy(monkeypatch):
    selected = registration()
    schema = {"type": "object", "description": "x" * 65_537}
    monkeypatch.setattr(
        selected.argument_model, "model_json_schema", lambda **kwargs: schema
    )
    request = ToolTurnRequest("Task", (selected,))
    assert (
        build_openai_tool_payload(model_config(), request)["tools"][0]["function"][
            "parameters"
        ]
        == schema
    )
    with pytest.raises(ValueError, match="size limit"):
        descriptor_from_registration(selected)
