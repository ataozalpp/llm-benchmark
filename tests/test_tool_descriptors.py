import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from llm_benchmark.tool_descriptors import (
    MAX_TOOL_SCHEMA_BYTES,
    MAX_TOOL_SCHEMA_DEPTH,
    ToolDescriptor,
    descriptor_from_registration,
    serialize_tool_descriptor,
)
from llm_benchmark.tool_requests import serialize_tool_registration
from llm_benchmark.tool_runtime import ToolDefinition, ToolRegistration
from llm_benchmark.tools import create_example_tool_registry


def make_descriptor(parameters=None):
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
            },
            "required": ["value"],
            "additionalProperties": False,
        }

    return ToolDescriptor(
        definition=ToolDefinition(
            name="example",
            description="Synthetic example tool.",
        ),
        parameters=parameters,
    )


def test_schema_is_snapshotted():
    original = {
        "type": "object",
        "properties": {
            "value": {"type": "integer"},
        },
    }

    descriptor = make_descriptor(original)
    original["properties"]["value"]["type"] = "string"

    assert descriptor.parameters["properties"]["value"]["type"] == "integer"


def test_returned_schema_is_an_independent_copy():
    descriptor = make_descriptor()
    returned = descriptor.parameters

    returned["properties"]["value"]["type"] = "string"

    assert descriptor.parameters["properties"]["value"]["type"] == "integer"


def test_descriptor_is_frozen():
    descriptor = make_descriptor()

    with pytest.raises(FrozenInstanceError):
        descriptor.definition = ToolDefinition(
            name="other",
            description="Another tool.",
        )


def test_schema_is_not_in_repr():
    descriptor = make_descriptor(
        {
            "type": "object",
            "description": "SCHEMA_SENTINEL",
        }
    )

    assert "SCHEMA_SENTINEL" not in repr(descriptor)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array"},
        {"properties": {}},
        {"type": "object", "invalid": (1, 2)},
        {"type": "object", "invalid": float("nan")},
        {"type": "object", "invalid": float("inf")},
        {"type": "object", 123: "invalid"},
    ],
)
def test_invalid_schema_is_rejected(schema):
    with pytest.raises(ValueError):
        make_descriptor(schema)


def test_schema_key_order_does_not_change_snapshot_equality():
    first = make_descriptor({"type": "object", "properties": {}})
    second = make_descriptor({"properties": {}, "type": "object"})

    assert first == second


@pytest.mark.parametrize(
    "name",
    ["calculator", "lookup_city_code"],
)
def test_local_descriptor_preserves_existing_tool_payload(name):
    registry = create_example_tool_registry()
    registration = registry.get(name)
    assert registration is not None

    descriptor = descriptor_from_registration(registration)

    assert serialize_tool_descriptor(descriptor) == serialize_tool_registration(
        registration
    )


@pytest.mark.parametrize("value", [None, [], "object", 1, True])
def test_non_object_root(value):
    with pytest.raises(TypeError, match="must be an object"):
        ToolDescriptor(
            definition=ToolDefinition("example", "Example."), parameters=value
        )


@pytest.mark.parametrize("value", [None, {}, "example"])
def test_invalid_definition(value):
    with pytest.raises(TypeError, match="ToolDefinition"):
        ToolDescriptor(definition=value, parameters={"type": "object"})


@pytest.mark.parametrize(
    "value",
    [
        b"bytes",
        {1},
        object(),
        {"nested": (1,)},
        {"nested": {1: "value"}},
        float("-inf"),
        "\ud800",
    ],
)
def test_nested_non_json_values_are_rejected(value):
    with pytest.raises(ValueError):
        make_descriptor({"type": "object", "data": [value]})


def test_invalid_utf8_key_is_rejected():
    with pytest.raises(ValueError, match="UTF-8"):
        make_descriptor({"type": "object", "\ud800": None})


@pytest.mark.parametrize("kind", ["dict", "list"])
def test_cycles_are_bounded(kind):
    cycle = {} if kind == "dict" else []
    if kind == "dict":
        cycle["self"] = cycle
    else:
        cycle.append(cycle)
    with pytest.raises(ValueError, match="nesting limit"):
        make_descriptor({"type": "object", "cycle": cycle})


@pytest.mark.parametrize("unicode_text", [False, True])
def test_exact_utf8_size_boundary(unicode_text):
    schema = {"type": "object", "description": ""}
    overhead = len(
        json.dumps(
            schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    remaining = MAX_TOOL_SCHEMA_BYTES - overhead
    schema["description"] = (
        "é" * (remaining // 2) + "x" * (remaining % 2)
        if unicode_text
        else "x" * remaining
    )
    assert make_descriptor(schema).parameters == schema
    schema["description"] += "x"
    with pytest.raises(ValueError, match="size limit"):
        make_descriptor(schema)


def test_exact_depth_boundary():
    value = None
    for _ in range(MAX_TOOL_SCHEMA_DEPTH - 1):
        value = [value]
    assert make_descriptor({"type": "object", "data": value})
    with pytest.raises(ValueError, match="nesting limit"):
        make_descriptor({"type": "object", "data": [value]})


def test_json_types_and_reference_data_are_preserved():
    schema = {
        "type": "object",
        "data": [None, True, 1, 1.5, "Türkçe", [], {}],
        "$ref": "https://example.invalid/schema",
    }
    assert make_descriptor(schema).parameters == schema


def test_serialized_payload_is_independent():
    descriptor = make_descriptor()
    payload = serialize_tool_descriptor(descriptor)
    payload["function"]["parameters"]["properties"].clear()
    payload["function"]["name"] = "changed"
    assert descriptor.definition.name == "example"
    assert "value" in descriptor.parameters["properties"]


@pytest.mark.parametrize(
    "function", [descriptor_from_registration, serialize_tool_descriptor]
)
@pytest.mark.parametrize("value", [None, {}, object()])
def test_conversion_requires_exact_contract(function, value):
    with pytest.raises(TypeError):
        function(value)


def test_conversion_does_not_execute_handler():
    original = create_example_tool_registry().get("calculator")

    def reject(arguments):
        raise AssertionError("Handler must not execute.")

    registration = ToolRegistration(
        original.definition, original.argument_model, reject
    )
    assert descriptor_from_registration(registration).definition == original.definition


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_trusted_schema_hook_errors_propagate(monkeypatch, error_type):
    registration = create_example_tool_registry().get("calculator")
    error = error_type("Synthetic hook failure.")

    def reject(*args, **kwargs):
        raise error

    monkeypatch.setattr(registration.argument_model, "model_json_schema", reject)
    with pytest.raises(error_type) as captured:
        descriptor_from_registration(registration)
    assert captured.value is error


def test_import_and_reference_snapshot_have_no_runtime_io(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import os
import sys
sys.path.insert(0, sys.argv[1])
def guard(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.mkdir')):
        raise AssertionError('Unexpected I/O')
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ):
            raise AssertionError('Unexpected write')
sys.addaudithook(guard)
from llm_benchmark.tool_runtime import ToolDefinition, ToolRegistry, ToolRuntime
def reject(*args, **kwargs):
    raise AssertionError('Unexpected runtime initialization')
ToolRegistry.__init__ = reject
ToolRuntime.__init__ = reject
from llm_benchmark.tool_descriptors import ToolDescriptor, serialize_tool_descriptor
descriptor = ToolDescriptor(
    definition=ToolDefinition('example', 'Example.'),
    parameters={'type': 'object', '$ref': 'https://example.invalid/schema'},
)
assert serialize_tool_descriptor(descriptor)['function']['name'] == 'example'
for name in ('providers', 'runner', 'api', 'worker', 'db', 'tools'):
    assert 'llm_benchmark.' + name not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, source_root],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
