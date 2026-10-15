"""Trusted local launch settings for the calculator MCP profile."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalculatorLaunch:
    server_python: Path
    project_root: Path


def resolve_calculator_launch(
    *,
    project_root: Path,
) -> CalculatorLaunch:
    value = os.environ.get("LLM_BENCHMARK_MCP_PYTHON")

    if value is None or not value.strip():
        raise ValueError(
            "Set LLM_BENCHMARK_MCP_PYTHON to the server Python executable."
        )

    try:
        server_python = Path(value).resolve(strict=True)
        root = project_root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise ValueError(
            "Invalid calculator MCP launch config."
        ) from None

    server_module = (
        root / "src" / "llm_benchmark" / "demo_mcp_server.py"
    )

    if not server_python.is_file() or not server_module.is_file():
        raise ValueError("Invalid calculator MCP launch config.")

    return CalculatorLaunch(server_python=server_python, project_root=root)