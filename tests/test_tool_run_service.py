import json
from dataclasses import replace

import pytest

from llm_benchmark.cli import main
from llm_benchmark.tool_demo_provider import DemoToolProvider
from llm_benchmark.tool_loop import ToolLoopPolicy, ToolLoopStopReason, run_tool_loop
from llm_benchmark.tool_provider_models import (
    ToolProviderErrorCode,
    ToolProviderResult,
    ToolProviderStatus,
)
from llm_benchmark.tool_run_config import ToolRunConfig, load_tool_run_config
from llm_benchmark.tool_run_service import run_tool_benchmark
from llm_benchmark.tool_scenario_requests import build_tool_scenario_request
from llm_benchmark.tool_scenarios import create_example_tool_scenarios
from llm_benchmark.tool_suite import run_tool_suite
from llm_benchmark.tools import create_example_tool_registry


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_demo_artifacts_and_fingerprints(tmp_path):
    config = ToolRunConfig(output_dir=tmp_path)
    first, summary = run_tool_benchmark(config)
    second, _ = run_tool_benchmark(config)
    a, b = read_json(first / "manifest.json"), read_json(second / "manifest.json")
    assert first != second
    assert a["run_fingerprint"] == b["run_fingerprint"]
    assert a["status"] == "completed"
    assert a["synthetic"] is True
    assert summary["completion"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert summary["all_case_final_answer_accuracy"]["rate"] == 1.0
    results = read_lines(first / "results.jsonl")
    assert len(results) == 2
    assert results[0]["provider_telemetry"] == [None, None]
    assert summary["token_usage"]["total_tokens"] == {
        "observed_sum": None,
        "measured_turn_count": 0,
        "total_turn_count": 3,
    }
    events = read_lines(first / "trace.jsonl")
    assert [row["sequence"] for row in events] == list(range(1, len(events) + 1))
    assert [
        row["event_type"] for row in events if row["case_id"] == "calculator-multiply"
    ] == [
        "scenario_started",
        "model_request",
        "model_response",
        "tool_call",
        "tool_result",
        "model_request",
        "model_response",
        "scenario_completed",
    ]
    assert all(
        "arguments" not in row["data"] and "output" not in row["data"] for row in events
    )
    changed, _ = run_tool_benchmark(
        config.model_copy(update={"max_wall_time_seconds": 120})
    )
    assert (
        read_json(changed / "manifest.json")["run_fingerprint"] != a["run_fingerprint"]
    )


def test_provider_failure_preserves_partial_results_without_secret(
    tmp_path, monkeypatch
):
    original = DemoToolProvider.generate_tool_conversation

    def fail_second(self, request):
        if request.conversation.messages[0].content.startswith("Return exactly"):
            raise RuntimeError("Authorization: Bearer private-value C:\\private\\file")
        return original(self, request)

    monkeypatch.setattr(DemoToolProvider, "generate_tool_conversation", fail_second)
    with pytest.raises(RuntimeError, match="Tool run failed"):
        run_tool_benchmark(ToolRunConfig(output_dir=tmp_path))
    directory = next(tmp_path.iterdir())
    assert read_json(directory / "manifest.json")["status"] == "failed"
    assert read_json(directory / "manifest.json")["completed_case_count"] == 1
    assert len(read_lines(directory / "results.jsonl")) == 1
    assert read_lines(directory / "trace.jsonl")[-1]["event_type"] == "model_request"
    assert not (directory / "summary.json").exists()
    assert "private-value" not in "".join(p.read_text() for p in directory.iterdir())


def test_deadline_stops_after_slow_provider_before_tool_execution():
    now = [0.0]
    registry = create_example_tool_registry()
    case = create_example_tool_scenarios()[0]

    class SlowProvider(DemoToolProvider):
        def generate_tool_conversation(self, request):
            now[0] = 2.0
            return super().generate_tool_conversation(request)

    events = []
    result = run_tool_loop(
        provider=SlowProvider(),
        request=build_tool_scenario_request(scenario=case, registry=registry),
        policy=replace(case.policy, max_wall_time_seconds=1.0),
        clock=lambda: now[0],
        observer=lambda kind, data: events.append(kind),
    )
    assert result.stop_reason is ToolLoopStopReason.WALL_TIME_LIMIT
    assert not result.tool_results
    assert "tool_call" not in events
    assert events[-1] == "scenario_completed"


@pytest.mark.parametrize("value", [0, -1, True, "1", float("inf"), float("nan")])
def test_invalid_wall_budget(value):
    with pytest.raises(ValueError):
        ToolLoopPolicy(2, 1, max_wall_time_seconds=value)


def test_suite_callbacks_preserve_case_identity():
    cases = create_example_tool_scenarios()
    observed, completed = [], []
    result = run_tool_suite(
        scenarios=cases,
        registry=create_example_tool_registry(),
        provider_factory=DemoToolProvider,
        observer=lambda case_id, kind, data: observed.append((case_id, kind)),
        on_case_completed=lambda case, loop, evaluation: completed.append(
            (case.case_id, evaluation)
        ),
    )
    assert tuple(e for _, e in completed) == result.evaluations
    assert {case_id for case_id, _ in observed} == {case.case_id for case in cases}


def test_cli_demo(tmp_path, capsys):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        json.dumps({"output_dir": str(tmp_path / "outputs")}), encoding="utf-8"
    )
    main(["tool-run", "--config", str(config_path)])
    output = json.loads(capsys.readouterr().out)
    assert output["overall"]["synthetic"] is True


def test_invalid_config_does_not_echo_secrets(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("unknown: private-value", encoding="utf-8")
    with pytest.raises(ValueError, match="^Invalid tool-run configuration.$"):
        load_tool_run_config(path)


def test_real_provider_requires_compatible_model():
    with pytest.raises(ValueError):
        ToolRunConfig(provider="openai_compatible")


def test_normalized_provider_failure_is_completed_evaluation(tmp_path, monkeypatch):
    original = DemoToolProvider.generate_tool_conversation

    def fail_second(self, request):
        if request.conversation.messages[0].content.startswith("Return exactly"):
            return ToolProviderResult(
                status=ToolProviderStatus.REQUEST_FAILED,
                error_code=ToolProviderErrorCode.TIMEOUT,
            )
        return original(self, request)

    monkeypatch.setattr(DemoToolProvider, "generate_tool_conversation", fail_second)
    directory, summary = run_tool_benchmark(ToolRunConfig(output_dir=tmp_path))
    assert read_json(directory / "manifest.json")["status"] == "completed"
    assert summary["completion"]["rate"] == 0.5
    assert summary["final_answer_match"]["rate"] == 1.0
    assert summary["all_case_final_answer_accuracy"]["rate"] == 0.5
    results = read_lines(directory / "results.jsonl")
    assert results[1]["provider_outcomes"][0]["error_code"] == "timeout"


def test_deadline_after_tool_retains_result_and_prevents_next_turn():
    from llm_benchmark.tool_runtime import ToolRuntime

    now = [0.0]
    registry = create_example_tool_registry()
    case = create_example_tool_scenarios()[0]
    runtime = ToolRuntime(registry)

    class SlowExecutor:
        def execute(self, call):
            now[0] = 2.0
            return runtime.execute(call)

    result = run_tool_loop(
        provider=DemoToolProvider(),
        request=build_tool_scenario_request(scenario=case, registry=registry),
        policy=replace(case.policy, max_wall_time_seconds=1.0),
        executor=SlowExecutor(),
        clock=lambda: now[0],
    )
    assert result.stop_reason is ToolLoopStopReason.WALL_TIME_LIMIT
    assert result.provider_turn_count == 1
    assert len(result.tool_results) == 1
