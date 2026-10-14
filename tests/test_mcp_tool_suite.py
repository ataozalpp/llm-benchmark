import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure_stage", [None, "discovery", "provider", "executor"])
def test_suite_exits_connection(monkeypatch, failure_stage):
    pytest.importorskip("mcp")

    from llm_benchmark import mcp_runtime, mcp_tool_suite
    from llm_benchmark.mcp_errors import MCPTransportFailure
    from llm_benchmark.mcp_mapping import descriptor_from_mcp_fields
    from llm_benchmark.tool_demo_provider import DemoToolProvider
    from llm_benchmark.tool_runtime import ToolExecutionStatus, ToolResult
    from llm_benchmark.tool_scenarios import create_example_tool_scenarios

    events = []
    original = (
        MCPTransportFailure("test transport failure")
        if failure_stage == "executor"
        else ValueError("test discovery or provider failure")
    )
    from types import SimpleNamespace

    fake_client = SimpleNamespace(protocol_version="test-protocol")

    descriptor = descriptor_from_mcp_fields(
        name="calculator",
        description="Apply an arithmetic operation.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["operation", "left", "right"],
            "additionalProperties": False,
        },
    )

    @asynccontextmanager
    async def fake_connect(*, server_python, project_root):
        events.append("opened")
        try:
            yield fake_client
        finally:
            events.append("closed")

    async def fake_discover(client):
        assert client is fake_client
        if failure_stage == "discovery":
            raise original
        return (descriptor,)

    class FakeExecutor:
        def __init__(self, *, client):
            assert client is fake_client

        def execute(self, call):
            if failure_stage == "executor":
                raise original
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.SUCCEEDED,
                output={"result": 391},
            )

    class FailingProvider:
        def generate_tool_conversation(self, request):
            raise original

    monkeypatch.setattr(
        mcp_runtime,
        "connect_calculator",
        fake_connect,
    )
    monkeypatch.setattr(
        mcp_runtime,
        "discover_calculator",
        fake_discover,
    )
    monkeypatch.setattr(
        mcp_runtime,
        "MCPToolExecutor",
        FakeExecutor,
    )

    async def run():
        return await mcp_tool_suite.run_calculator_mcp_suite(
            scenarios=create_example_tool_scenarios(),
            provider_factory=(
                FailingProvider if failure_stage == "provider" else DemoToolProvider
            ),
            server_python=Path("unused-python"),
            project_root=Path("."),
        )

    if failure_stage is None:
        result = asyncio.run(run())
        assert len(result.evaluations) == 2
    else:
        with pytest.raises(type(original)) as caught:
            asyncio.run(run())
        assert caught.value is original
    assert events == ["opened", "closed"]
