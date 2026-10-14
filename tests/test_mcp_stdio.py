import asyncio
import os
from pathlib import Path

import pytest


@pytest.fixture
def server_python():
    value = os.environ.get("LLM_BENCHMARK_MCP_PYTHON")

    if value is None:
        pytest.skip("LLM_BENCHMARK_MCP_PYTHON is not configured.")

    path = Path(value).resolve()

    if not path.is_file():
        pytest.fail("Configured MCP server Python does not exist.")

    return path


def test_real_stdio_round_trip(server_python):
    pytest.importorskip("mcp")

    from llm_benchmark.demo_mcp_stdio_smoke import run_smoke

    project_root = Path(__file__).resolve().parents[1]

    asyncio.run(
        run_smoke(
            server_python=server_python,
            project_root=project_root,
        )
    )


def test_server_rejects_invalid_arguments(server_python):
    pytest.importorskip("mcp")

    import anyio
    from mcp.shared.exceptions import MCPError

    from llm_benchmark.mcp_client import connect_calculator

    project_root = Path(__file__).resolve().parents[1]

    invalid_arguments = (
        {
            "operation": "multiply",
            "left": "17",
            "right": 23,
        },
        {
            "operation": "multiply",
            "left": 1_000_001,
            "right": 23,
        },
        {
            "operation": "unknown",
            "left": 17,
            "right": 23,
        },
        {
            "operation": "multiply",
            "left": 17,
            "right": 23,
            "unexpected": 1,
        },
        {
            "operation": "multiply",
            "left": True,
            "right": 23,
        }
    )

    async def check():
        with anyio.fail_after(30):
            async with connect_calculator(
                server_python=server_python,
                project_root=project_root,
            ) as client:
                for arguments in invalid_arguments:
                    try:
                        response = await client.call_tool(
                            "calculator",
                            arguments,
                            read_timeout_seconds=10.0,
                        )
                    except MCPError as error:
                        # Invalid params is an acceptable protocol rejection.
                        # Other protocol failures must still fail this test.
                        assert error.code == -32602
                    else:
                        assert response.is_error is True

    asyncio.run(check())


def test_mcp_suite_matches_local_suite(server_python):
    pytest.importorskip("mcp")

    from llm_benchmark.mcp_tool_suite import run_calculator_mcp_suite
    from llm_benchmark.tool_demo_provider import DemoToolProvider
    from llm_benchmark.tool_scenarios import create_example_tool_scenarios
    from llm_benchmark.tool_suite import run_tool_suite
    from llm_benchmark.tools import create_example_tool_registry

    scenarios = create_example_tool_scenarios()

    local_result = run_tool_suite(
        scenarios=scenarios,
        registry=create_example_tool_registry(),
        provider_factory=DemoToolProvider,
    )

    mcp_result = asyncio.run(
        run_calculator_mcp_suite(
            scenarios=scenarios,
            provider_factory=DemoToolProvider,
            server_python=server_python,
            project_root=Path(__file__).resolve().parents[1],
        )
    )

    assert len(mcp_result.evaluations) == len(scenarios)
    assert mcp_result.evaluations == local_result.evaluations
    assert mcp_result.summary == local_result.summary


def test_executor_maps_real_tool_error(server_python):
    pytest.importorskip("mcp")

    import anyio

    from llm_benchmark.mcp_client import connect_calculator
    from llm_benchmark.mcp_executor import MCPToolExecutor
    from llm_benchmark.tool_runtime import (
        ToolCall,
        ToolErrorCode,
        ToolExecutionStatus,
    )

    async def check():
        async with connect_calculator(
            server_python=server_python,
            project_root=Path(__file__).resolve().parents[1],
        ) as client:
            executor = MCPToolExecutor(client=client)
            call = ToolCall(
                call_id="division-error-1",
                tool_name="calculator",
                arguments={
                    "operation": "divide",
                    "left": 17,
                    "right": 0,
                },
            )

            result = await anyio.to_thread.run_sync(
                executor.execute,
                call,
            )

            assert result.status is ToolExecutionStatus.EXECUTION_FAILED
            assert result.error_code is ToolErrorCode.TOOL_EXECUTION_FAILED
            assert result.call_id == call.call_id
            assert result.output is None
    asyncio.run(check())


def test_server_exit_during_call_is_transport_failure(server_python):
    pytest.importorskip("mcp")

    import anyio
    from mcp import Client, StdioServerParameters

    from llm_benchmark.mcp_errors import MCPTransportFailure
    from llm_benchmark.mcp_executor import MCPToolExecutor
    from llm_benchmark.tool_runtime import ToolCall

    project_root = Path(__file__).resolve().parents[1]
    server_script = (
        project_root / "tests" / "fixtures" / "mcp_disconnect_server.py"
    )

    parameters = StdioServerParameters(
        command=str(server_python),
        args=[str(server_script)],
        cwd=str(project_root),
        env={"PYTHONUNBUFFERED": "1"},
    )

    async def check():
        with anyio.fail_after(30):
            async with Client(
                parameters,
                mode="legacy",
                read_timeout_seconds=10.0,
            ) as client:
                executor = MCPToolExecutor(client=client, timeout_seconds=10.0)
                invocation = ToolCall(
                    call_id="disconnect-test-1",
                    tool_name="disconnect",
                    arguments={},
                )

                with pytest.raises(MCPTransportFailure):
                    await anyio.to_thread.run_sync(executor.execute, invocation)

    asyncio.run(check())


def test_prepared_runtime_exposes_discovered_catalog(server_python):
    pytest.importorskip("mcp")

    from llm_benchmark.mcp_runtime import prepare_calculator_runtime

    async def check():
        async with prepare_calculator_runtime(
            server_python=server_python,
            project_root=Path(__file__).resolve().parents[1],
        ) as runtime:
            snapshot = runtime.catalog_snapshot()

            assert len(snapshot) == 1
            assert snapshot[0]["name"] == "calculator"
            assert snapshot[0]["parameters"] == (runtime.descriptors[0].parameters)
            assert isinstance(runtime.protocol_version, str)
            assert runtime.protocol_version

    asyncio.run(check())


def test_mcp_tool_run_writes_artifacts(
    server_python,
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("mcp")

    import json

    from llm_benchmark.tool_run_config import ToolRunConfig
    from llm_benchmark.tool_run_service import run_tool_benchmark

    monkeypatch.setenv("LLM_BENCHMARK_MCP_PYTHON", str(server_python))

    config = ToolRunConfig(
        runtime="mcp_stdio",
        mcp_profile="calculator",
        output_dir=tmp_path / "runs",
    )

    directory, summary = run_tool_benchmark(
        config,
        project_root=Path(__file__).resolve().parents[1],
    )

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in (
            directory / "results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    trace = [
        json.loads(line)
        for line in (
            directory / "trace.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["status"] == "completed"
    assert manifest["runtime_type"] == "mcp_stdio"
    assert manifest["runtime_version"] == "mcp-calculator-v1"
    assert manifest["runtime_metadata"]["profile"] == "calculator"
    assert manifest["runtime_metadata"]["protocol_version"]
    assert summary["completion"]["rate"] == 1.0
    assert len(results) == 2
    assert [event["sequence"] for event in trace] == list(range(1, len(trace) + 1))