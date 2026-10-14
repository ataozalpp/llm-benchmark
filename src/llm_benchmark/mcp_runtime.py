"""Prepare the calculator MCP runtime for suite execution."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from .mcp_client import connect_calculator, discover_calculator
from .mcp_executor import MCPToolExecutor
from .tool_descriptors import ToolDescriptor


@dataclass(frozen=True)
class PreparedCalculatorRuntime:
    descriptors: tuple[ToolDescriptor, ...]
    executor: MCPToolExecutor
    protocol_version: str

    def catalog_snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "name": descriptor.definition.name,
                "description": descriptor.definition.description,
                "parameters": descriptor.parameters,
            }
            for descriptor in sorted(
                self.descriptors,
                key=lambda item: item.definition.name,
            )
        ]


@asynccontextmanager
async def prepare_calculator_runtime(
    *,
    server_python: Path,
    project_root: Path,
) -> AsyncIterator[PreparedCalculatorRuntime]:
    async with connect_calculator(
        server_python=server_python,
        project_root=project_root,
    ) as client:
        descriptors = await discover_calculator(client)

        yield PreparedCalculatorRuntime(
            descriptors=descriptors,
            executor=MCPToolExecutor(client=client),
            protocol_version=client.protocol_version,
        )