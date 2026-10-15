"""Local and MCP runtime preparation with shared artifact orchestration."""

import asyncio
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from llm_benchmark.providers import OpenAICompatibleProvider
from llm_benchmark.tool_use.artifacts import ToolArtifactWriter, fingerprint
from llm_benchmark.tool_use.builtin_tools import create_example_tool_registry
from llm_benchmark.tool_use.demo_provider import DemoToolProvider
from llm_benchmark.tool_use.evaluation import ToolEvaluationResult
from llm_benchmark.tool_use.loop import ConversationProvider, ToolLoopResult
from llm_benchmark.tool_use.run_config import ToolRunConfig
from llm_benchmark.tool_use.scenarios import ToolScenario, create_example_tool_scenarios
from llm_benchmark.tool_use.suite import ToolSuiteResult, run_tool_suite

type ToolRunObserver = Callable[
    [str, str, dict[str, object]],
    None,
]

type ToolRunCaseCallback = Callable[
    [ToolScenario, ToolLoopResult, ToolEvaluationResult],
    None,
]

type ToolRunProviderFactory = Callable[
    [],
    ConversationProvider,
]

type ToolSuiteRunner = Callable[
    [
        ToolRunProviderFactory,
        ToolRunObserver,
        ToolRunCaseCallback,
    ],
    ToolSuiteResult,
]


@dataclass
class _ToolRunCompletion:
    writer: ToolArtifactWriter
    manifest: dict[str, object]

    def complete(self) -> None:
        self.manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.writer.write_json("manifest.json", self.manifest)

    def fail(self) -> None:
        self.manifest.update(
            status="failed",
            error_code="tool_run_failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.writer.write_json("manifest.json", self.manifest)


def run_tool_benchmark(
    config: ToolRunConfig,
    *,
    project_root: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    scenarios = tuple(
        replace(
            case,
            policy=replace(
                case.policy, max_wall_time_seconds=config.max_wall_time_seconds
            ),
        )
        for case in create_example_tool_scenarios()
    )

    if config.runtime == "mcp_stdio":
        return asyncio.run(
            _run_mcp_tool_benchmark(
                config=config,
                scenarios=scenarios,
                project_root=(
                    Path.cwd()
                    if project_root is None
                    else project_root
                ),
            )
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

    def execute_suite(
        provider_factory: ToolRunProviderFactory,
        observer: ToolRunObserver,
        on_case_completed: ToolRunCaseCallback,
    ) -> ToolSuiteResult:
        return run_tool_suite(
            scenarios=scenarios,
            registry=registry,
            provider_factory=provider_factory,
            observer=observer,
            on_case_completed=on_case_completed,
        )

    return _run_tool_benchmark_with_artifacts(
        config=config,
        scenarios=scenarios,
        catalog=catalog,
        runtime_type="local",
        runtime_version="local-tool-v1",
        execute_suite=execute_suite,
    )


def _run_tool_benchmark_with_artifacts(
    *,
    config: ToolRunConfig,
    scenarios: tuple[ToolScenario, ...],
    catalog: list[dict[str, object]],
    runtime_type: str,
    runtime_version: str,
    execute_suite: ToolSuiteRunner,
    runtime_metadata: dict[str, object] | None = None,
    defer_completion: bool = False,
    on_run_created: Callable[[_ToolRunCompletion], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Persist suite artifacts, optionally deferring final completion.

    Local execution completes immediately. Async runtimes may finalize only
    after their session has closed successfully.
    """
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
    excluded_fields = {"output_dir"}

    if config.runtime == "local":
        excluded_fields.update({"runtime", "mcp_profile"})

    config_snapshot = config.model_dump(mode="json", exclude=excluded_fields)
    identity = {
        "suite_hash": fingerprint(suite_snapshot),
        "tool_catalog_hash": fingerprint(catalog),
        "config_hash": fingerprint(config_snapshot),
        "evaluator_version": "tool-exact-v1",
        "runtime_version": runtime_version,
        "artifact_schema_version": 1,
    }
    if runtime_metadata is not None:
        identity["runtime_metadata_hash"] = fingerprint(runtime_metadata)
    run_id = uuid4().hex
    writer = ToolArtifactWriter(config.output_dir / run_id)
    manifest = {
        **identity,
        "run_id": run_id,
        "run_fingerprint": fingerprint(identity),
        "suite": config.suite,
        "provider": config.provider,
        "runtime_type": runtime_type,
        "synthetic": config.provider == "synthetic",
        "python_version": platform.python_version(),
        "scenario_ids": [case.case_id for case in scenarios],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "loop_policies": [asdict(case.policy) for case in scenarios],
    }
    # Config is hashed, not persisted: URLs may carry credentials or local paths.
    if runtime_metadata is not None:
        manifest["runtime_metadata"] = runtime_metadata

    completion = _ToolRunCompletion(writer=writer, manifest=manifest)
    writer.write_json("manifest.json", manifest)
    if on_run_created is not None:
        on_run_created(completion)
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
        result = execute_suite(provider_factory, observe, save_case)
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
        manifest["completed_case_count"] = completed_cases

        if defer_completion:
            writer.write_json("manifest.json", manifest)
        else:
            completion.complete()
    except Exception:
        # Never persist exception strings: a provider may include credentials.
        manifest["completed_case_count"] = completed_cases
        completion.fail()

        raise RuntimeError(
            f"Tool run failed; inspect artifacts for run {run_id}."
        ) from None
    return writer.directory, summary


async def _run_mcp_tool_benchmark(
    *,
    config: ToolRunConfig,
    scenarios: tuple[ToolScenario, ...],
    project_root: Path,
) -> tuple[Path, dict[str, object]]:
    from functools import partial

    from llm_benchmark.mcp_integration.launch import resolve_calculator_launch

    launch = resolve_calculator_launch(project_root=project_root)

    try:
        import anyio

        from llm_benchmark.mcp_integration.runtime import prepare_calculator_runtime
    except ImportError:
        raise RuntimeError(
            "MCP runtime dependencies are unavailable; install the mcp extra."
        ) from None

    from llm_benchmark.tool_use.external.policy import ExternalToolPolicy
    from llm_benchmark.tool_use.external.suite import run_external_tool_suite

    completions: list[_ToolRunCompletion] = []

    try:
        async with prepare_calculator_runtime(
            server_python=launch.server_python,
            project_root=launch.project_root,
        ) as runtime:

            def execute_suite(
                provider_factory: ToolRunProviderFactory,
                observer: ToolRunObserver,
                on_case_completed: ToolRunCaseCallback,
            ) -> ToolSuiteResult:
                return run_external_tool_suite(
                    scenarios=scenarios,
                    descriptors=runtime.descriptors,
                    policy=ExternalToolPolicy(
                        allowed_tool_names=("calculator",),
                        max_tools=1,
                        max_total_schema_bytes=65_536,
                    ),
                    provider_factory=provider_factory,
                    executor_factory=lambda request: runtime.executor,
                    observer=observer,
                    on_case_completed=on_case_completed,
                )

            run_with_artifacts = partial(
                _run_tool_benchmark_with_artifacts,
                config=config,
                scenarios=scenarios,
                catalog=runtime.catalog_snapshot(),
                runtime_type="mcp_stdio",
                runtime_version="mcp-calculator-v1",
                execute_suite=execute_suite,
                runtime_metadata={
                    "profile": "calculator",
                    "protocol_version": runtime.protocol_version,
                },
                defer_completion=True,
                on_run_created=completions.append,
            )

            result = await anyio.to_thread.run_sync(run_with_artifacts)

        # The session has closed successfully at this point.
        completion = completions[0]
        completion.complete()
        return result

    except Exception:
        if completions:
            completion = completions[0]

            if completion.manifest["status"] != "failed":
                completion.fail()

            run_id = completion.manifest["run_id"]
            raise RuntimeError(
                f"MCP tool run failed; inspect artifacts for run {run_id}."
            ) from None

        raise RuntimeError(
            "MCP runtime preparation failed before artifact creation."
        ) from None
