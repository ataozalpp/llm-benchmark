"""Local tool-run orchestration; transport integration stays outside the loop."""

import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .providers import OpenAICompatibleProvider
from .tool_artifacts import ToolArtifactWriter, fingerprint
from .tool_demo_provider import DemoToolProvider
from .tool_evaluation import ToolEvaluationResult
from .tool_loop import ToolLoopResult
from .tool_run_config import ToolRunConfig
from .tool_scenarios import ToolScenario, create_example_tool_scenarios
from .tool_suite import run_tool_suite
from .tools import create_example_tool_registry


def run_tool_benchmark(config: ToolRunConfig) -> tuple[Path, dict[str, object]]:
    scenarios = tuple(
        replace(
            case,
            policy=replace(
                case.policy, max_wall_time_seconds=config.max_wall_time_seconds
            ),
        )
        for case in create_example_tool_scenarios()
    )
    registry = create_example_tool_registry()
    # Include exactly the exposed tools, not unrelated registry entries.
    selected = sorted({name for case in scenarios for name in case.available_tools})
    catalog = []
    for name in selected:
        registration = registry.get(name)
        assert registration is not None
        catalog.append(
            {
                "name": name,
                "description": registration.definition.description,
                "parameters": registration.argument_model.model_json_schema(),
            }
        )
    suite_snapshot = [
        {
            "case_id": case.case_id,
            "user_text": case.user_text,
            "available_tools": case.available_tools,
            "policy": asdict(case.policy),
            "expected_calls": [
                {"name": call.tool_name, "arguments": call.arguments}
                for call in case.evaluation.expected_calls
            ],
            "expected_final_text": case.evaluation.expected_final_text,
        }
        for case in scenarios
    ]
    config_snapshot = config.model_dump(mode="json", exclude={"output_dir"})
    identity = {
        "suite_hash": fingerprint(suite_snapshot),
        "tool_catalog_hash": fingerprint(catalog),
        "config_hash": fingerprint(config_snapshot),
        "evaluator_version": "tool-exact-v1",
        "runtime_version": "local-tool-v1",
        "artifact_schema_version": 1,
    }
    run_id = uuid4().hex
    writer = ToolArtifactWriter(config.output_dir / run_id)
    manifest = {
        **identity,
        "run_id": run_id,
        "run_fingerprint": fingerprint(identity),
        "suite": config.suite,
        "provider": config.provider,
        "runtime_type": "local",
        "synthetic": config.provider == "synthetic",
        "python_version": platform.python_version(),
        "scenario_ids": [case.case_id for case in scenarios],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "loop_policies": [asdict(case.policy) for case in scenarios],
    }
    # Config is hashed, not persisted: URLs may carry credentials or local paths.
    writer.write_json("manifest.json", manifest)
    started = time.monotonic()
    sequence = 0
    completed_cases = 0
    provider_turns = 0
    token_measurements: dict[str, list[int]] = {
        "input_tokens": [],
        "output_tokens": [],
        "total_tokens": [],
    }

    def observe(case_id: str, kind: str, data: dict[str, object]) -> None:
        nonlocal sequence
        sequence += 1
        writer.append(
            "trace.jsonl",
            {
                "schema_version": 1,
                "sequence": sequence,
                "run_id": run_id,
                "case_id": case_id,
                "event_type": kind,
                "elapsed_ms": (time.monotonic() - started) * 1000,
                "data": data,
            },
        )

    def save_case(
        case: ToolScenario, loop: ToolLoopResult, evaluation: ToolEvaluationResult
    ) -> None:
        nonlocal completed_cases, provider_turns
        telemetry = [
            asdict(result.telemetry) if result.telemetry else None
            for result in loop.provider_results
        ]
        writer.append(
            "results.jsonl",
            {
                "schema_version": 1,
                **asdict(evaluation),
                "provider_turn_count": loop.provider_turn_count,
                "provider_telemetry": telemetry,
                "provider_outcomes": [
                    {
                        "status": item.status.value,
                        "error_code": item.error_code.value
                        if item.error_code
                        else None,
                        "normalization_error_code": (
                            item.normalization_error_code.value
                            if item.normalization_error_code
                            else None
                        ),
                    }
                    for item in loop.provider_results
                ],
                "tool_outcomes": [
                    {
                        "call_id": item.call_id,
                        "tool_name": item.tool_name,
                        "status": item.status.value,
                        "error_code": item.error_code.value
                        if item.error_code
                        else None,
                    }
                    for item in loop.tool_results
                ],
            },
        )
        completed_cases += 1
        provider_turns += loop.provider_turn_count
        for provider_result in loop.provider_results:
            for field_name, values in token_measurements.items():
                value = (
                    getattr(provider_result.telemetry, field_name)
                    if provider_result.telemetry is not None
                    else None
                )
                if value is not None:
                    values.append(value)

    def provider_factory():
        if config.provider == "synthetic":
            return DemoToolProvider()
        assert config.model is not None
        return OpenAICompatibleProvider(config.model)

    try:
        result = run_tool_suite(
            scenarios=scenarios,
            registry=registry,
            provider_factory=provider_factory,
            observer=observe,
            on_case_completed=save_case,
        )
        metrics = result.summary
        summary = asdict(metrics)
        for name in (
            "completion",
            "tool_sequence_match",
            "argument_match",
            "final_answer_match",
        ):
            summary[name]["rate"] = getattr(metrics, name).rate
        summary["all_case_final_answer_accuracy"] = {
            "numerator": metrics.final_answer_match.numerator,
            "denominator": metrics.case_count,
            "rate": metrics.final_answer_match.numerator / metrics.case_count,
        }
        summary.update(
            run_id=run_id,
            synthetic=config.provider == "synthetic",
            wall_time_ms=(time.monotonic() - started) * 1000,
        )
        summary["token_usage"] = {
            name: {
                "observed_sum": sum(values) if values else None,
                "measured_turn_count": len(values),
                "total_turn_count": provider_turns,
            }
            for name, values in token_measurements.items()
        }
        writer.write_json("summary.json", summary)
        manifest.update(
            status="completed",
            completed_case_count=completed_cases,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        writer.write_json("manifest.json", manifest)
    except Exception:
        # Never persist exception strings: a provider may include credentials.
        manifest.update(
            status="failed",
            completed_case_count=completed_cases,
            error_code="tool_run_failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        writer.write_json("manifest.json", manifest)
        raise RuntimeError(
            f"Tool run failed; inspect artifacts for run {run_id}."
        ) from None
    return writer.directory, summary
