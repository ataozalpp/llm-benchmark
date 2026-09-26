import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from llm_benchmark.external_tool_policy import (
    ExternalToolAdmissionCode,
    ExternalToolAdmissionError,
    ExternalToolPolicy,
    admit_tool_descriptors,
)
from llm_benchmark.tool_descriptors import ToolDescriptor
from llm_benchmark.tool_runtime import ToolDefinition


def descriptor(name="calculator"):
    return ToolDescriptor(
        definition=ToolDefinition(
            name=name,
            description="Synthetic tool.",
        ),
        parameters={"type": "object"},
    )


def policy(**updates):
    values = {
        "allowed_tool_names": (
            "calculator",
            "lookup_city_code",
        ),
        "max_tools": 2,
        "max_total_schema_bytes": 1024,
    }
    values.update(updates)
    return ExternalToolPolicy(**values)


def test_admission_is_sorted_without_mutating_input():
    calculator = descriptor("calculator")
    city = descriptor("lookup_city_code")
    original = (city, calculator)

    admitted = admit_tool_descriptors(
        descriptors=original,
        policy=policy(),
    )

    assert admitted == (calculator, city)
    assert original == (city, calculator)
    assert admitted[0] is calculator
    assert admitted[1] is city


def test_empty_collection_is_rejected():
    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=(),
            policy=policy(),
        )

    assert caught.value.code is ExternalToolAdmissionCode.EMPTY_COLLECTION


def test_tool_count_limit_is_inclusive():
    selected = (
        descriptor("calculator"),
        descriptor("lookup_city_code"),
    )

    assert (
        len(
            admit_tool_descriptors(
                descriptors=selected,
                policy=policy(max_tools=2),
            )
        )
        == 2
    )

    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=selected,
            policy=policy(max_tools=1),
        )

    assert caught.value.code is ExternalToolAdmissionCode.TOO_MANY_TOOLS


def test_duplicate_names_are_rejected():
    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=(descriptor(), descriptor()),
            policy=policy(),
        )

    assert caught.value.code is ExternalToolAdmissionCode.DUPLICATE_TOOL_NAME


def test_disallowed_tool_is_not_silently_filtered():
    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=(
                descriptor("calculator"),
                descriptor("private_tool"),
            ),
            policy=policy(),
        )

    assert caught.value.code is ExternalToolAdmissionCode.TOOL_NOT_ALLOWED
    assert "private_tool" not in str(caught.value)
    assert "private_tool" not in repr(caught.value)


def test_empty_allowlist_denies_all_tools():
    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=(descriptor(),),
            policy=policy(allowed_tool_names=()),
        )

    assert caught.value.code is ExternalToolAdmissionCode.TOOL_NOT_ALLOWED


def test_total_schema_budget_is_inclusive():
    selected = (
        descriptor("calculator"),
        descriptor("lookup_city_code"),
    )

    admitted = admit_tool_descriptors(
        descriptors=selected,
        policy=policy(max_total_schema_bytes=34),
    )
    assert len(admitted) == 2

    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=selected,
            policy=policy(max_total_schema_bytes=33),
        )

    assert caught.value.code is ExternalToolAdmissionCode.SCHEMA_BUDGET_EXCEEDED


@pytest.mark.parametrize(
    "field",
    ["max_tools", "max_total_schema_bytes"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_limits_are_rejected(field, value):
    with pytest.raises(ValueError):
        policy(**{field: value})


def test_policy_is_frozen():
    selected_policy = policy()

    with pytest.raises(FrozenInstanceError):
        selected_policy.max_tools = 100


@pytest.mark.parametrize("value", [None, [], {}, "calculator"])
def test_allowlist_requires_tuple(value):
    with pytest.raises(TypeError):
        policy(allowed_tool_names=value)


@pytest.mark.parametrize("name", [None, True, 1, "", "1tool", "a b", "a/b", "a" * 65])
def test_allowlist_reuses_tool_name_validation(name):
    with pytest.raises(ValueError):
        policy(allowed_tool_names=(name,))


def test_duplicate_allowlist_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        policy(allowed_tool_names=("calculator", "calculator"))


def test_valid_name_boundaries_and_case_sensitive_admission():
    names = ("A", "a" * 64, "tool_1-X")
    selected = tuple(descriptor(name) for name in names)
    assert (
        len(
            admit_tool_descriptors(
                descriptors=selected,
                policy=policy(allowed_tool_names=names, max_tools=3),
            )
        )
        == 3
    )

    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(descriptors=(descriptor("Calculator"),), policy=policy())
    assert caught.value.code is ExternalToolAdmissionCode.TOOL_NOT_ALLOWED


@pytest.mark.parametrize("value", [None, [], {}, "tools"])
def test_catalog_requires_tuple(value):
    with pytest.raises(TypeError):
        admit_tool_descriptors(descriptors=value, policy=policy())


@pytest.mark.parametrize("value", [None, {}, "calculator", object()])
def test_catalog_rejects_non_descriptor_elements(value):
    with pytest.raises(TypeError):
        admit_tool_descriptors(descriptors=(descriptor(), value), policy=policy())


@pytest.mark.parametrize("value", [None, {}, object()])
def test_admission_requires_policy_contract(value):
    with pytest.raises(TypeError):
        admit_tool_descriptors(descriptors=(descriptor(),), policy=value)


@pytest.mark.parametrize("value", [None, "tool_not_allowed", "unknown", 1])
def test_error_requires_closed_code(value):
    with pytest.raises(TypeError):
        ExternalToolAdmissionError(value)


@pytest.mark.parametrize("code", list(ExternalToolAdmissionCode))
def test_every_error_code_has_a_fixed_message(code):
    error = ExternalToolAdmissionError(code)
    assert error.code is code
    assert len(error.args) == 1
    assert type(error.args[0]) is str
    assert str(error) == str(ExternalToolAdmissionError(code))


def test_schema_budget_counts_utf8_not_characters():
    schema = {"type": "object", "description": "\u00e7\U0001f600"}
    selected = ToolDescriptor(
        definition=ToolDefinition("calculator", "Synthetic tool."), parameters=schema
    )
    # Two UTF-8 bytes for c-cedilla, four for the emoji; no ASCII escaping.
    expected_bytes = len(b'{"description":"') + 2 + 4 + len(b'","type":"object"}')
    assert admit_tool_descriptors(
        descriptors=(selected,), policy=policy(max_total_schema_bytes=expected_bytes)
    ) == (selected,)
    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=(selected,),
            policy=policy(max_total_schema_bytes=expected_bytes - 1),
        )
    assert caught.value.code is ExternalToolAdmissionCode.SCHEMA_BUDGET_EXCEEDED


def test_budget_ignores_description_and_uses_immutable_schema_snapshot():
    source = {"type": "object"}
    selected = ToolDescriptor(
        definition=ToolDefinition("calculator", "x" * 512), parameters=source
    )
    source["description"] = "private-sentinel"
    selected.parameters["description"] = "private-sentinel"
    assert admit_tool_descriptors(
        descriptors=(selected,), policy=policy(max_total_schema_bytes=17)
    ) == (selected,)
    assert selected.parameters == {"type": "object"}


def test_budget_error_does_not_echo_schema_or_description():
    sentinel = "private-schema-sentinel"
    selected = ToolDescriptor(
        definition=ToolDefinition("calculator", sentinel),
        parameters={"type": "object", "description": sentinel},
    )
    with pytest.raises(ExternalToolAdmissionError) as caught:
        admit_tool_descriptors(
            descriptors=(selected,), policy=policy(max_total_schema_bytes=1)
        )
    assert caught.value.code is ExternalToolAdmissionCode.SCHEMA_BUDGET_EXCEEDED
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)


def test_repeated_admission_and_catalog_order_are_deterministic():
    selected = (descriptor("lookup_city_code"), descriptor())
    selected_policy = policy()
    first = admit_tool_descriptors(descriptors=selected, policy=selected_policy)
    assert first == admit_tool_descriptors(
        descriptors=selected[::-1], policy=selected_policy
    )
    assert first == admit_tool_descriptors(descriptors=selected, policy=selected_policy)
    assert selected_policy.allowed_tool_names == ("calculator", "lookup_city_code")
    assert "calculator" not in repr(selected_policy)


def test_admission_is_not_a_semantic_schema_validator():
    # Invalid JSON Schema keyword semantics are deliberately outside this gate.
    selected = ToolDescriptor(
        definition=ToolDefinition("calculator", "Synthetic tool."),
        parameters={"type": "object", "required": 42},
    )
    assert admit_tool_descriptors(descriptors=(selected,), policy=policy()) == (
        selected,
    )


def test_import_and_admission_have_no_runtime_io(tmp_path):
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
from llm_benchmark.external_tool_policy import ExternalToolPolicy, admit_tool_descriptors
from llm_benchmark.tool_descriptors import ToolDescriptor
descriptor = ToolDescriptor(
    definition=ToolDefinition('example', 'Example.'),
    parameters={'type': 'object', '$ref': 'https://example.invalid/schema'},
)
policy = ExternalToolPolicy(('example',), 1, 1024)
assert admit_tool_descriptors(descriptors=(descriptor,), policy=policy) == (descriptor,)
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
