import asyncio
from types import SimpleNamespace

import pytest


@pytest.fixture
def mcp_dependencies():
    pytest.importorskip("mcp")

    import anyio
    from mcp.shared.exceptions import MCPError
    from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

    from llm_benchmark.mcp_integration.errors import (
        McpExecutionTimeout,
        MCPProtocolFailure,
        MCPTransportFailure,
    )
    from llm_benchmark.mcp_integration.executor import MCPToolExecutor

    return SimpleNamespace(
        anyio=anyio,
        executor=MCPToolExecutor,
        sdk_error=MCPError,
        timeout_code=REQUEST_TIMEOUT,
        closed_code=CONNECTION_CLOSED,
        timeout_error=McpExecutionTimeout,
        transport_error=MCPTransportFailure,
        protocol_error=MCPProtocolFailure,
    )


@pytest.fixture
def call():
    from llm_benchmark.tool_use.runtime import ToolCall

    return ToolCall(
        call_id="executor-test-1",
        tool_name="calculator",
        arguments={
            "operation": "multiply",
            "left": 17,
            "right": 23,
        },
    )


class FakeClient:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def call_tool(
        self,
        name,
        arguments,
        *,
        read_timeout_seconds,
    ):
        self.calls.append((name, arguments, read_timeout_seconds))

        if self.error is not None:
            raise self.error

        return self.response


def execute_in_worker(deps, executor, call):
    async def run():
        return await deps.anyio.to_thread.run_sync(
            executor.execute,
            call,
        )

    return asyncio.run(run())


def test_success_preserves_identity_and_arguments(
    mcp_dependencies,
    call,
):
    from llm_benchmark.tool_use.runtime import ToolExecutionStatus

    deps = mcp_dependencies
    client = FakeClient(
        response=SimpleNamespace(
            is_error=False,
            structured_content={"result": 391},
        )
    )
    executor = deps.executor(client=client)

    result = execute_in_worker(deps, executor, call)

    assert result.status is ToolExecutionStatus.SUCCEEDED
    assert result.call_id == call.call_id
    assert result.tool_name == call.tool_name
    assert result.output == {"result": 391}
    assert client.calls == [
        ("calculator", call.arguments, 10.0),
    ]


@pytest.mark.parametrize(
    ("is_error", "content", "budget", "expected_status"),
    [
        (True, {"detail": "internal"}, 65_536, "execution_failed"),
        (False, ["not", "an", "object"], 65_536, "invalid_output"),
        (False, {"result": 391}, 1, "output_too_large"),
    ],
)
def test_response_mapping(
    mcp_dependencies,
    call,
    is_error,
    content,
    budget,
    expected_status,
):
    deps = mcp_dependencies
    client = FakeClient(
        response=SimpleNamespace(
            is_error=is_error,
            structured_content=content,
        )
    )
    executor = deps.executor(
        client=client,
        max_output_bytes=budget,
    )

    result = execute_in_worker(deps, executor, call)

    assert result.status.value == expected_status
    assert result.call_id == call.call_id
    assert result.output is None


@pytest.mark.parametrize(
    "kind",
    ["sdk_timeout", "closed", "protocol", "broken_stream"],
)
def test_infrastructure_failures(
    mcp_dependencies,
    call,
    kind,
):
    deps = mcp_dependencies

    if kind == "sdk_timeout":
        error = deps.sdk_error(deps.timeout_code, "raw detail")
        expected = deps.timeout_error
    elif kind == "closed":
        error = deps.sdk_error(deps.closed_code, "raw detail")
        expected = deps.transport_error
    elif kind == "protocol":
        error = deps.sdk_error(-32602, "raw detail")
        expected = deps.protocol_error
    else:
        error = deps.anyio.BrokenResourceError()
        expected = deps.transport_error

    executor = deps.executor(client=FakeClient(error=error))

    with pytest.raises(expected) as caught:
        execute_in_worker(deps, executor, call)

    assert "raw detail" not in str(caught.value)

    if kind == "protocol":
        assert caught.value.code == -32602


def test_executor_deadline(mcp_dependencies, call):
    deps = mcp_dependencies

    class WaitingClient:
        async def call_tool(
            self,
            name,
            arguments,
            *,
            read_timeout_seconds,
        ):
            await deps.anyio.sleep_forever()

    executor = deps.executor(
        client=WaitingClient(),
        timeout_seconds=0.05,
    )

    with pytest.raises(deps.timeout_error):
        execute_in_worker(deps, executor, call)


def test_unexpected_error_propagates(mcp_dependencies, call):
    deps = mcp_dependencies
    original = ValueError("unexpected implementation failure.")
    executor = deps.executor(client=FakeClient(error=original))

    with pytest.raises(ValueError) as caught:
        execute_in_worker(deps, executor, call)

    assert caught.value is original


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, float("inf"), float("nan")],
)
def test_invalid_timeout_is_rejected(mcp_dependencies, timeout):
    with pytest.raises(ValueError):
        mcp_dependencies.executor(
            client=FakeClient(),
            timeout_seconds=timeout,
        )


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "1024"])
def test_invalid_output_budget_is_rejected(mcp_dependencies, budget):
    with pytest.raises(ValueError):
        mcp_dependencies.executor(client=FakeClient(), max_output_bytes=budget)


def test_invalid_call_is_rejected_before_client_use(mcp_dependencies):
    client = FakeClient()
    executor = mcp_dependencies.executor(client=client)
    with pytest.raises(TypeError, match="Expected a ToolCall"):
        executor.execute({"tool_name": "calculator"})
    assert client.calls == []


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(2.0, 2.0), (30.0, 10.0)],
)
def test_per_call_timeout_is_clamped(
    mcp_dependencies,
    call,
    budget,
    expected,
):
    from functools import partial

    deps = mcp_dependencies
    client = FakeClient(
        response=SimpleNamespace(
            is_error=False,
            structured_content={"result": 391},
        )
    )
    executor = deps.executor(
        client=client,
        timeout_seconds=10.0,
    )

    async def run():
        return await deps.anyio.to_thread.run_sync(
            partial(
                executor.execute_with_timeout,
                call,
                timeout_seconds=budget,
            )
        )

    result = asyncio.run(run())

    assert result.output == {"result": 391}
    assert client.calls == [
        ("calculator", call.arguments, expected),
    ]
