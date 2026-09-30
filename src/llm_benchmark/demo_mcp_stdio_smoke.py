"""Real stdio smoke check using a seperate FastMCP server process."""

import argparse
import asyncio
from pathlib import Path

import anyio

from .external_tool_schema import validate_external_tool_arguments
from .mcp_client import connect_calculator, discover_calculator
from .mcp_mapping import result_from_mcp_fields
from .tool_runtime import ToolCall, ToolExecutionStatus


async def run_smoke(
    *,
    server_python: Path,
    project_root: Path,
) -> None:
    # This is a smoke-level deadline, not the production executor policy.
    with anyio.fail_after(30):
        async with connect_calculator(
            server_python=server_python,
            project_root=project_root,
        ) as client:
            descriptors = await discover_calculator(client)

            call = ToolCall(
                call_id="stdio-smoke-1",
                tool_name="calculator",
                arguments={
                    "operation": "multiply",
                    "left": 17,
                    "right": 23,
                },
            )

            validate_external_tool_arguments(
                descriptor=descriptors[0],
                call=call,
            )

            response = await client.call_tool(
                call.tool_name,
                call.arguments,
                read_timeout_seconds=10.0,
            )

            result = result_from_mcp_fields(
                call=call,
                is_error=response.is_error,
                structured_content=response.structured_content,
            )

            if result.status is not ToolExecutionStatus.SUCCEEDED:
                raise RuntimeError("Calculator call did not succeed.")

            if result.output != {"result": 391}:
                raise RuntimeError("Calculator returned an unexpected result.")

            if result.call_id != call.call_id:
                raise RuntimeError("Tool call identity was not preserved.")

            protocol_version = client.protocol_version
    print(
        "MCP stdio smoke passed. "
        f"Protocol: {protocol_version}; result: 391."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-python",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    args = parser.parse_args()

    asyncio.run(
        run_smoke(
            server_python=args.server_python,
            project_root=args.project_root,
        )
    )


if __name__ == "__main__":
    main()