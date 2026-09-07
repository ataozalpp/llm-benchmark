"""Deterministic example tools for offline runtime validation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .tool_runtime import (
    ToolDefinition,
    ToolRegistration,
    ToolRegistry,
)


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    operation: Literal["add", "subtract", "multiply", "divide"]
    left: int = Field(ge=-1_000_000, le=1_000_000)
    right: int = Field(ge=-1_000_000, le=1_000_000)


class CityCodeArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    city: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z]+(?:_[a-z]+)*$",
    )


_CITY_CODES = MappingProxyType(
    {
        "northport": "SYN-001",
        "rivergate": "SYN-002",
        "sunvale": "SYN-003",
    }
)


def calculator_handler(arguments: BaseModel) -> dict[str, int | float]:
    if not isinstance(arguments, CalculatorArguments):
        raise TypeError("Unexpected calculator argument model.")

    left = arguments.left
    right = arguments.right
    result: int | float

    if arguments.operation == "add":
        result = left + right
    elif arguments.operation == "subtract":
        result = left - right
    elif arguments.operation == "multiply":
        result = left * right
    else:
        if right == 0:
            raise ValueError("Division by zero is not supported.")
        result = left / right

    return {"result": result}


def lookup_city_code_handler(
    arguments: BaseModel,
) -> dict[str, str | bool | None]:
    if not isinstance(arguments, CityCodeArguments):
        raise TypeError("Unexpected city code argument model.")

    code = _CITY_CODES.get(arguments.city)

    return {
        "found": code is not None,
        "code": code,
    }


def create_example_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="calculator",
                description=(
                    "Apply add, subtract, multiply or divide to two integers "
                    "between -1000000 and 1000000."
                ),
            ),
            argument_model=CalculatorArguments,
            handler=calculator_handler,
        )
    )
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="lookup_city_code",
                description=(
                    "Look up a lowercase synthetic city key in a fixed "
                    "offline mapping; report whether a code was found."
                ),
            ),
            argument_model=CityCodeArguments,
            handler=lookup_city_code_handler,
        )
    )
    return registry
