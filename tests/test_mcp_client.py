import asyncio
from types import SimpleNamespace

import pytest

from llm_benchmark.tools import CalculatorArguments


@pytest.mark.parametrize("cursor", [None, "next-page"])
def test_single_page_catalog_admission(cursor):
    pytest.importorskip("mcp")
    from llm_benchmark.mcp_client import discover_calculator

    class FakeClient:
        async def list_tools(self):
            return SimpleNamespace(
                next_cursor=cursor,
                tools=[SimpleNamespace(
                    name="calculator",
                    description="Synthetic calculator.",
                    input_schema=CalculatorArguments.model_json_schema(),
                )],
            )

    if cursor is None:
        descriptors = asyncio.run(discover_calculator(FakeClient()))
        assert len(descriptors) == 1
        assert descriptors[0].definition.name == "calculator"
    else:
        with pytest.raises(ValueError, match="single page catalog"):
            asyncio.run(discover_calculator(FakeClient()))
