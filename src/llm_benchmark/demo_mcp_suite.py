"""Run synthetic calculator benchmark suite through MCP."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .mcp_integration.suite import run_calculator_mcp_suite
from .tool_use.demo_provider import DemoToolProvider
from .tool_use.scenarios import create_example_tool_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic calculator suite through MCP.",
    )
    parser.add_argument(
        "--server-python",
        type=Path,
        required=True,
        help="Python executable containing the FastMCP server dependencies.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the src directory.",
    )
    args = parser.parse_args()


    result = asyncio.run(
        run_calculator_mcp_suite(
            scenarios=create_example_tool_scenarios(),
            provider_factory=DemoToolProvider,
            server_python=args.server_python,
            project_root=args.project_root,
        )
    )

    payload = {
        "synthetic": True,
        "runtime": "mcp_stdio",
        "summary": asdict(result.summary),
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()