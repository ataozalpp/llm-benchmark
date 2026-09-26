import traceback

import pytest

from llm_benchmark.external_tool_schema import (
    ExternalToolSchemaCode,
    ExternalToolSchemaError,
    validate_external_tool_arguments,
    validate_external_tool_schema,
)
from llm_benchmark.tool_descriptors import ToolDescriptor
from llm_benchmark.tool_runtime import ToolCall, ToolDefinition


def make_descriptor():
    return ToolDescriptor(
        definition=ToolDefinition(
            name="calculator",
            description="Synthetic calculator.",
        ),
        parameters={
            "type": "object",
            "properties": {
                "left": {"type": "integer", "minimum": -100, "maximum": 100},
            },
            "required": ["left"],
            "additionalProperties": False,
        },
    )


def test_supported_schema_is_accepted():
    validate_external_tool_schema(make_descriptor())


def test_valid_arguments_are_accepted():
    call = ToolCall(
        call_id="call_1",
        tool_name="calculator",
        arguments={"left": 17},
    )

    validate_external_tool_arguments(
        descriptor=make_descriptor(),
        call=call,
    )

    assert call.arguments == {"left": 17}


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"left": "17"},
        {"left": True},
        {"left": 101},
        {"left": 17, "unexpected": 1},
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    call = ToolCall(
        call_id="call_1",
        tool_name="calculator",
        arguments=arguments,
    )

    with pytest.raises(ExternalToolSchemaError) as caught:
        validate_external_tool_arguments(
            descriptor=make_descriptor(),
            call=call,
        )

    assert caught.value.code is ExternalToolSchemaCode.INVALID_ARGUMENTS


def test_reference_schema_is_unsupported():
    descriptor = ToolDescriptor(
        definition=ToolDefinition(
            name="calculator",
            description="Synthetic calculator.",
        ),
        parameters={
            "type": "object",
            "$ref": "https://example.invalid/schema12",
        },
    )

    with pytest.raises(ExternalToolSchemaError) as caught:
        validate_external_tool_schema(descriptor)

    assert caught.value.code is ExternalToolSchemaCode.UNSUPPORTED_SCHEMA


def descriptor_with(schema):
    return ToolDescriptor(
        definition=ToolDefinition("calculator", "Synthetic calculator."),
        parameters=schema,
    )


def validate(schema, arguments):
    validate_external_tool_arguments(
        descriptor=descriptor_with(schema),
        call=ToolCall(call_id="call_1", tool_name="calculator", arguments=arguments),
    )


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("$ref", "https://example.invalid/schema"),
        ("$dynamicRef", "#node"),
        ("$defs", {}),
        ("$id", "urn:example"),
        ("$schema", "https://json-schema.org/draft/2020-12/schema"),
        ("allOf", [{}]),
        ("anyOf", [{}]),
        ("oneOf", [{}]),
        ("pattern", ".*"),
        ("patternProperties", {}),
        ("format", "email"),
        ("unknown_keyword", True),
    ],
)
@pytest.mark.parametrize(
    "position", ["root", "properties", "items", "additionalProperties"]
)
def test_unsupported_keywords_rejected_at_every_schema_position(
    keyword, value, position
):
    nested = {keyword: value}
    if position == "root":
        schema = {"type": "object", **nested}
    elif position == "properties":
        schema = {"type": "object", "properties": {"value": nested}}
    else:
        schema = {"type": "object", position: nested}
    with pytest.raises(ExternalToolSchemaError) as caught:
        validate_external_tool_schema(descriptor_with(schema))
    assert caught.value.code is ExternalToolSchemaCode.UNSUPPORTED_SCHEMA


@pytest.mark.parametrize(
    "fragment",
    [
        {"required": "value"},
        {"required": [1]},
        {"properties": []},
        {"properties": {"value": 1}},
        {"items": []},
        {"additionalProperties": "false"},
        {"minLength": -1},
        {"maxItems": True},
        {"minimum": "zero"},
        {"properties": {"value": {"type": "unknown"}}},
    ],
)
def test_malformed_supported_schema_is_rejected(fragment):
    with pytest.raises(ExternalToolSchemaError) as caught:
        validate_external_tool_schema(descriptor_with({"type": "object", **fragment}))
    assert caught.value.code is ExternalToolSchemaCode.INVALID_SCHEMA


@pytest.mark.parametrize("value", [-100, 100, 17.0])
def test_numeric_boundaries_and_standard_integer_semantics(value):
    call = ToolCall(call_id="call_1", tool_name="calculator", arguments={"left": value})
    validate_external_tool_arguments(descriptor=make_descriptor(), call=call)
    assert type(call.arguments["left"]) is type(value)


@pytest.mark.parametrize("value", [-101, 17.5])
def test_invalid_numeric_values(value):
    with pytest.raises(ExternalToolSchemaError) as caught:
        validate_external_tool_arguments(
            descriptor=make_descriptor(),
            call=ToolCall(
                call_id="call_1", tool_name="calculator", arguments={"left": value}
            ),
        )
    assert caught.value.code is ExternalToolSchemaCode.INVALID_ARGUMENTS


@pytest.mark.parametrize("valid", [True, False])
def test_nested_objects_arrays_and_constraints(valid):
    schema = {
        "type": "object",
        "required": ["values"],
        "additionalProperties": False,
        "properties": {
            "values": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "required": ["name", "score"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 3},
                        "score": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "exclusiveMaximum": 10,
                        },
                    },
                },
            }
        },
    }
    arguments = {"values": [{"name": "ok", "score": 1 if valid else 0}]}
    if valid:
        validate(schema, arguments)
    else:
        with pytest.raises(ExternalToolSchemaError) as caught:
            validate(schema, arguments)
        assert caught.value.code is ExternalToolSchemaCode.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    "rule,good,bad",
    [
        ({"enum": ["a", "b"]}, "a", "c"),
        ({"const": "a"}, "a", "b"),
        ({"type": ["string", "null"]}, None, 1),
        ({"type": "string", "minLength": 1, "maxLength": 2}, "ab", "abc"),
        ({"type": "array", "minItems": 1, "maxItems": 2}, [1], []),
    ],
)
def test_supported_constraints(rule, good, bad):
    schema = {"type": "object", "properties": {"value": rule}}
    validate(schema, {"value": good})
    with pytest.raises(ExternalToolSchemaError):
        validate(schema, {"value": bad})


def test_data_positions_are_not_interpreted_as_schema_keywords():
    schema = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}},
        "default": {"$ref": "synthetic-data"},
        "examples": [{"unknown_keyword": 1}],
        "const": {"$ref": "synthetic-data"},
    }
    validate(schema, {"$ref": "synthetic-data"})


def test_defaults_do_not_mutate_arguments_and_extras_default_to_allowed():
    schema = {"type": "object", "properties": {"value": {"default": 17}}}
    descriptor = descriptor_with(schema)
    arguments = {"other": "kept"}
    call = ToolCall(call_id="call_1", tool_name="calculator", arguments=arguments)
    validate_external_tool_arguments(descriptor=descriptor, call=call)
    assert call.arguments == arguments
    assert descriptor.parameters == schema


@pytest.mark.parametrize("position", ["properties", "items", "additionalProperties"])
def test_boolean_nested_schemas(position):
    for allowed in (True, False):
        if position == "properties":
            schema = {"type": "object", "properties": {"value": allowed}}
            arguments = {"value": 1}
        elif position == "items":
            schema = {
                "type": "object",
                "properties": {"value": {"type": "array", "items": allowed}},
            }
            arguments = {"value": [1]}
        else:
            schema = {"type": "object", "additionalProperties": allowed}
            arguments = {"value": 1}
        if allowed:
            validate(schema, arguments)
        else:
            with pytest.raises(ExternalToolSchemaError):
                validate(schema, arguments)


@pytest.mark.parametrize("value", [None, {}, object()])
def test_contract_types_are_strict(value):
    with pytest.raises(TypeError):
        validate_external_tool_schema(value)
    with pytest.raises(TypeError):
        validate_external_tool_arguments(descriptor=make_descriptor(), call=value)


def test_call_name_must_match():
    with pytest.raises(ValueError, match="does not match"):
        validate_external_tool_arguments(
            descriptor=make_descriptor(),
            call=ToolCall(call_id="call_1", tool_name="other", arguments={"left": 1}),
        )


@pytest.mark.parametrize("value", ["invalid_schema", None, 1])
def test_error_codes_are_closed(value):
    with pytest.raises(TypeError):
        ExternalToolSchemaError(value)


@pytest.mark.parametrize("phase", ["schema", "arguments"])
def test_errors_do_not_expose_private_values(phase):
    sentinel = "private-value-sentinel"
    with pytest.raises(ExternalToolSchemaError) as caught:
        if phase == "schema":
            validate_external_tool_schema(
                descriptor_with({"type": "object", "required": sentinel})
            )
        else:
            validate(
                {"type": "object", "properties": {"value": {"type": "integer"}}},
                {"value": sentinel},
            )
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    formatted = "".join(traceback.format_exception(caught.value))
    assert sentinel not in formatted
    assert caught.value.__suppress_context__
