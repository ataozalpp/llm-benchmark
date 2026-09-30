"""In process smoke check for the FastMCP calculator."""

import asyncio

from fastmcp import Client

from .demo_mcp_server import server


async def main() -> None:
    async with Client(server) as client:
        tools = await client.list_tools()

        names = [tool.name for tool in tools]
        assert names == ["calculator"], names

        result = await client.call_tool(
            "calculator",
            {
                "operation": "multiply",
                "left": 17,
                "right": 23,
            },
        )

        assert result.structured_content == {"result": 391}

        print("FastMCP calculator smoke passed.")


if __name__ == "__main__":
    asyncio.run(main())