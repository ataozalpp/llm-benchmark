"""FastMCP calculator server for local benchamark integration."""

from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from .tools import CalculatorArguments, calculator_handler

server = FastMCP(
    "llm-benchmark-calculator",
    strict_input_validation=True,
    mask_error_details=True,
)


@server.tool
def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    left: Annotated[
        int,
        Field(strict=True, ge=-1_000_000, le=1_000_000),
    ],
    right: Annotated[
        int,
        Field(strict=True, ge=-1_000_000, le=1_000_000),
    ],
) -> dict[str, int | float]:
    """Apply a bounded arithmetic operation to two integers."""

    arguments = CalculatorArguments.model_validate(
        {
            "operation": operation,
            "left": left,
            "right": right,
        }
    )

    return calculator_handler(arguments)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()