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