"""Async client helpers for the project-owned calculator MCP server."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters

from .external_tool_policy import (
    ExternalToolPolicy,
    admit_tool_descriptors,
)
from .mcp_mapping import descriptor_from_mcp_fields
from .tool_descriptors import ToolDescriptor


def calculator_server_parameters(
    *,
    server_python: Path,
    project_root: Path,
) -> StdioServerParameters:
    """Build a trusted local launch profile for the demo server."""

    python_path = server_python.resolve(strict=True)
    root = project_root.resolve(strict=True)
    source_root = root / "src"
    server_module = source_root / "llm_benchmark" / "demo_mcp_server.py"

    if not python_path.is_file():
        raise ValueError("Server Python must be a file.")

    if not server_module.is_file():
        raise ValueError("Calculator server module was not found.")

    return StdioServerParameters(
        command=str(python_path),
        args=["-m", "llm_benchmark.demo_mcp_server"],
        cwd=str(root),
        env={
            "PYTHONPATH": str(source_root),
            "PYTHONUNBUFFERED": "1",
        },
    )


@asynccontextmanager
async def connect_calculator(
    *,
    server_python: Path,
    project_root: Path,
) -> AsyncIterator[Client]:
    parameters = calculator_server_parameters(
        server_python=server_python,
        project_root=project_root,
    )

    async with Client(
        parameters,
        mode="legacy",
        read_timeout_seconds=10.0,
    ) as client:
        if client.server_capabilities.tools is None:
            raise ValueError("MCP server does not advertise tools.")

        yield client


async def discover_calculator(
    client: Client,
) -> tuple[ToolDescriptor, ...]:
    """Accept the bounded, single-page calculator demo catalog."""

    listing = await client.list_tools()

    if listing.next_cursor is not None:
        raise ValueError("The demo profile requires a single page catalog.")

    if len(listing.tools) != 1:
        raise ValueError("The demo profile requires exactly one tool.")

    descriptors = tuple(
        descriptor_from_mcp_fields(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )
        for tool in listing.tools
    )

    return admit_tool_descriptors(
        descriptors=descriptors,
        policy=ExternalToolPolicy(
            allowed_tool_names=("calculator",),
            max_tools=1,
            max_total_schema_bytes=65_536,
        ),
    )
