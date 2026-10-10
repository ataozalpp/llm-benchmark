import pytest

from llm_benchmark.tool_conversation import (
    ToolConversation,
    UserMessage,
)
from llm_benchmark.tool_demo_provider import DemoToolProvider
from llm_benchmark.tool_descriptors import ToolDescriptor
from llm_benchmark.tool_execution import ToolExecutionTimeout
from llm_benchmark.tool_loop import (
    ToolLoopPolicy,
    ToolLoopStopReason,
    run_tool_loop,
)
from llm_benchmark.tool_requests import ToolConversationRequest
from llm_benchmark.tool_runtime import (
    ToolDefinition,
    ToolExecutionStatus,
    ToolResult,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class AdvancingProvider:
    def __init__(self, clock, first_turn_seconds):
        self.clock = clock
        self.first_turn_seconds = first_turn_seconds
        self.calls = 0
        self.delegate = DemoToolProvider()

    def generate_tool_conversation(self, request):
        if self.calls == 0:
            self.clock.now += self.first_turn_seconds

        self.calls += 1
        return self.delegate.generate_tool_conversation(request)


class BudgetExecutor:
    def __init__(self, clock, *, timeout_at=None):
        self.clock = clock
        self.timeout_at = timeout_at
        self.budgets = []
        self.error = ToolExecutionTimeout("test timeout")

    def execute(self, call):
        raise AssertionError("Expected the timed execution path.")

    def execute_with_timeout(self, call, *, timeout_seconds):
        self.budgets.append(timeout_seconds)

        if self.timeout_at is not None:
            self.clock.now = self.timeout_at
            raise self.error

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.SUCCEEDED,
            output={"result": 391},
        )


def make_request():
    descriptor = ToolDescriptor(
        definition=ToolDefinition(
            name="calculator",
            description="Apply an arithmetic operation.",
        ),
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["operation", "left", "right"],
            "additionalProperties": False,
        },
    )

    return ToolConversationRequest(
        conversation=ToolConversation(
            messages=(
                UserMessage(
                    "Use the calculator to multiply 17 by 23. "
                    "Return only the resulting integer."
                ),
            )
        ),
        descriptors=(descriptor,),
    )


def make_policy():
    return ToolLoopPolicy(
        max_provider_turns=2,
        max_tool_calls=1,
        max_wall_time_seconds=10.0,
    )


def test_remaining_budget_reaches_executor():
    clock = FakeClock()
    executor = BudgetExecutor(clock)

    result = run_tool_loop(
        provider=AdvancingProvider(clock, first_turn_seconds=3.0),
        request=make_request(),
        policy=make_policy(),
        executor=executor,
        clock=clock,
    )

    assert executor.budgets == [pytest.approx(7.0)]
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE
    assert result.final_text == "391"


def test_expired_scenario_does_not_call_executor():
    clock = FakeClock()
    executor = BudgetExecutor(clock)

    result = run_tool_loop(
        provider=AdvancingProvider(clock, first_turn_seconds=10.0),
        request=make_request(),
        policy=make_policy(),
        executor=executor,
        clock=clock,
    )

    assert result.stop_reason is ToolLoopStopReason.WALL_TIME_LIMIT
    assert executor.budgets == []
    assert result.tool_results == ()


def test_timeout_before_scenario_deadline_propagates():
    clock = FakeClock()
    executor = BudgetExecutor(clock, timeout_at=4.0)

    with pytest.raises(ToolExecutionTimeout) as caught:
        run_tool_loop(
            provider=AdvancingProvider(clock, first_turn_seconds=3.0),
            request=make_request(),
            policy=make_policy(),
            executor=executor,
            clock=clock,
        )

    assert caught.value is executor.error
    assert executor.budgets == [pytest.approx(7.0)]


def test_legacy_executor_still_works_with_scenario_budget():
    clock = FakeClock()

    class LegacyExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, call):
            self.calls.append(call)
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolExecutionStatus.SUCCEEDED,
                output={"result": 391},
            )

    executor = LegacyExecutor()

    result = run_tool_loop(
        provider=AdvancingProvider(clock, first_turn_seconds=3.0),
        request=make_request(),
        policy=make_policy(),
        executor=executor,
        clock=clock,
    )

    assert len(executor.calls) == 1
    assert result.stop_reason is ToolLoopStopReason.FINAL_RESPONSE


def test_timeout_after_scenario_deadline_becomes_wall_time_limit():
    clock = FakeClock()
    executor = BudgetExecutor(clock, timeout_at=10.0)

    result = run_tool_loop(
        provider=AdvancingProvider(clock, first_turn_seconds=3.0),
        request=make_request(),
        policy=make_policy(),
        executor=executor,
        clock=clock,
    )

    assert executor.budgets == [pytest.approx(7.0)]
    assert result.stop_reason is ToolLoopStopReason.WALL_TIME_LIMIT
    assert result.tool_results == ()
