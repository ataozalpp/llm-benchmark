"""Test-only MCP server that exits during a tool request."""

import os

from fastmcp import FastMCP

server = FastMCP("disconnect-test-server")


@server.tool
def disconnect() -> dict[str, bool]:
    """Terminate this test server process during the request."""
    os._exit(17)


if __name__ == "__main__":
    server.run(transport="stdio")
