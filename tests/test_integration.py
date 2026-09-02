import json
from collections import Counter
from pathlib import Path

import pytest

from llm_benchmark.config import load_config
from llm_benchmark.models import DatasetExample, ProviderResponse
from llm_benchmark.runner import _evaluate, execute_benchmark, run_benchmark
from llm_benchmark.task_adapters import (
    MultipleChoiceTaskAdapter,
    TaskEvaluationOutcome,
)
from llm_benchmark.trace import (
    InMemoryTraceRecorder,
    TraceEvent,
    TraceEventData,
    TraceEventType,
)


class RecordingTaskAdapter:
    """Record  runner interactions while preserving MCQ behaviour."""

    task_type = "multiple_choice"

    def __init__(self) -> None:
        self.prompt_sample_ids: list[str] = []
        self.evaluated_sample_ids: list[str] = []
        self._delegate = MultipleChoiceTaskAdapter()

    def build_prompt(self, example: DatasetExample) -> str:
        self.prompt_sample_ids.append(example.sample_id)
        return self._delegate.build_prompt(example)

    def prompt_template_hash(self) -> str:
        return self._delegate.prompt_template_hash()

    def evaluate_response(
        self,
        example: DatasetExample,
        response: ProviderResponse,
    ) -> TaskEvaluationOutcome:
        self.evaluated_sample_ids.append(example.sample_id)
        return self._delegate.evaluate_response(example, response)


class AuthoritativeTaskAdapter:
    task_type = "test_task"

    def build_prompt(self, example: DatasetExample) -> str:
        return f"Test prompt for {example.sample_id}"

    def prompt_template_hash(self) -> str:
        return "a" * 64

    def evaluate_response(
        self,
        example: DatasetExample,
        response: ProviderResponse,
    ) -> TaskEvaluationOutcome:
        del example, response
        return TaskEvaluationOutcome(
            parsed_answer=None,
            parse_status="adapter_controlled_status",
            candidate_answers=(),
            evaluation_status="unparseable",
            is_correct=False,
        )


class FailingTraceRecorder:
    def record(self, *args: object, **kwargs: object) -> TraceEvent:
        del args, kwargs
        raise OSError("trace recorder failed")

    def events(self) -> tuple[TraceEvent, ...]:
        return ()


class ExplodingProvider:
    def generate(self, prompt: str, example: DatasetExample) -> ProviderResponse:
        del prompt, example
        raise RuntimeError("original provider failure")


class ExplodingTaskAdapter(AuthoritativeTaskAdapter):
    def evaluate_response(
        self,
        example: DatasetExample,
        response: ProviderResponse,
    ) -> TaskEvaluationOutcome:
        del example, response
        raise ValueError("original adapter failure")


class StaticProvider:
    def __init__(self, response: ProviderResponse) -> None:
        self._response = response

    def generate(self, prompt: str, example: DatasetExample) -> ProviderResponse:
        del prompt, example
        return self._response


class UnsafeTraceRecorder:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(self, *args: object, **kwargs: object) -> TraceEvent:
        del args, kwargs
        event = TraceEvent(
            sequence=len(self._events) + 1,
            event_type=TraceEventType.ERROR,
            timestamp="2026-09-01T10:00:00+00:00",
            run_id="Authorization: Bearer injected-credential",
            sample_id=r"C:\Users\person\private\sample.jsonl",
            model="/home/person/private/model.gguf",
            data=TraceEventData(
                provider="api_key=injected-credential",
                error_type="password=injected-credential",
            ),
        )
        self._events.append(event)
        return event

    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)



def test_fixture_pipeline_end_to_end(tmp_path: Path) -> None:
    config = load_config(Path("configs/mock_smoke.yaml"), [f"output_dir={json.dumps(str(tmp_path))}"])
    run_dir, summary = run_benchmark(config)
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "dataset_manifest.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "trace.jsonl").exists()
    result_rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert summary["overall"]["total_samples"] == 16
    assert len(result_rows) == 16

    status_counts = Counter(
        row["evaluation_status"]
        for row in result_rows
    )

    assert status_counts == Counter(
        {
            "correct": 10,
            "incorrect": 4,
            "unparseable": 1,
            "request_failed": 1,
        }
    )

    first_result = result_rows[0]

    assert first_result["sample_id"] == "q01"
    assert first_result["parsed_answer"] == "B"
    assert first_result["correct_answer"] == "B"
    assert first_result["evaluation_status"] == "correct"
    assert first_result["parse_status"] == "explicit_final_marker"

    assert summary["dataset"]["selected_categories"] == ["geography", "language", "math", "science"]
    assert summary["dataset"]["selected_sample_categories"]["q01"] == "math"
    persisted = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert persisted["run_fingerprint"] == summary["run_fingerprint"]


def test_runner_writes_execution_trace_artifact(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )

    execution = execute_benchmark(config)

    trace_path = execution.run_dir / "trace.jsonl"

    assert trace_path.exists()

    rows = [
        json.loads(line)
        for line in trace_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert rows
    assert [row["sequence"] for row in rows] == list(
        range(1, len(rows) + 1)
    )


def test_runner_records_expected_trace_event_order(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )
    recorder = InMemoryTraceRecorder(
        clock=lambda: "2026-09-01T10:00:00+00:00"
    )

    execute_benchmark(
        config,
        trace_recorder_factory=lambda: recorder,
    )

    expected_per_model = [
        TraceEventType.SCENARIO_STARTED,
        TraceEventType.MODEL_REQUEST,
        TraceEventType.MODEL_RESPONSE,
        TraceEventType.TASK_EVALUATION,
        TraceEventType.SCENARIO_COMPLETED,
    ]
    expected = expected_per_model * len(config.models)

    assert [
        event.event_type
        for event in recorder.events()
    ] == expected
    request_events = [
        event
        for event in recorder.events()
        if event.event_type is TraceEventType.MODEL_REQUEST
    ]
    assert [event.data.output_budget_provenance for event in request_events] == [
        model.output_budget_provenance for model in config.models
    ]


def test_request_failed_outcome_completes_trace_scenario(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q05"]',
            "dataset.sample_size=1",
        ],
    )
    recorder = InMemoryTraceRecorder(
        clock=lambda: "2026-09-01T10:00:00+00:00"
    )

    execution = execute_benchmark(
        config,
        trace_recorder_factory=lambda: recorder,
    )

    failed_result = next(
        result
        for result in execution.results
        if result.request_status == "failed"
    )

    failed_scenario_events = [
        event
        for event in recorder.events()
        if event.sample_id == failed_result.sample_id
        and event.model == failed_result.model
    ]

    assert [
        event.event_type
        for event in failed_scenario_events
    ] == [
        TraceEventType.SCENARIO_STARTED,
        TraceEventType.MODEL_REQUEST,
        TraceEventType.MODEL_RESPONSE,
        TraceEventType.TASK_EVALUATION,
        TraceEventType.SCENARIO_COMPLETED,
    ]


def test_trace_artifact_excludes_sensitive_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )

    sentinel = "sensitive-bearer-token-value"
    provider = StaticProvider(
        ProviderResponse(
            request_status="succeeded",
            raw_response=f"FINAL ANSWER: B\n{sentinel}\nraw-reasoning-sentinel",
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            latency_ms=5.0,
            error_type=f"error-{sentinel}",
            provider_error_type=f"provider-{sentinel}",
            provider_error_code=f"code-{sentinel}",
            provider_error_message=(
                f"Authorization: Bearer {sentinel}; password={sentinel}"
            ),
        )
    )
    monkeypatch.setattr(
        "llm_benchmark.runner.create_provider",
        lambda model_config: provider,
    )

    execution = execute_benchmark(config)

    trace_text = (
        execution.run_dir / "trace.jsonl"
    ).read_text(encoding="utf-8")
    normalized_trace = trace_text.lower()

    assert "authorization" not in normalized_trace
    assert "api_key" not in normalized_trace
    assert "password" not in normalized_trace
    assert "provider_error_message" not in normalized_trace
    assert "raw_response" not in normalized_trace
    assert "raw_reasoning" not in normalized_trace
    assert "answer the following multiple-choice question" not in normalized_trace
    assert "what is 2 + 2?" not in normalized_trace
    assert sentinel not in normalized_trace
    assert "raw-reasoning-sentinel" not in normalized_trace


def test_injected_recorder_cannot_bypass_artifact_sanitization(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )

    execution = execute_benchmark(
        config,
        trace_recorder_factory=UnsafeTraceRecorder,
    )
    trace_text = (execution.run_dir / "trace.jsonl").read_text(encoding="utf-8")

    assert "injected-credential" not in trace_text
    assert r"C:\Users\person" not in trace_text
    assert "/home/person" not in trace_text
    assert "[REDACTED]" in trace_text
    assert "[REDACTED_PATH]" in trace_text


def test_each_execution_uses_a_fresh_trace_recorder(tmp_path: Path) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )
    recorders: list[InMemoryTraceRecorder] = []

    def factory() -> InMemoryTraceRecorder:
        recorder = InMemoryTraceRecorder(
            clock=lambda: "2026-09-01T10:00:00+00:00"
        )
        recorders.append(recorder)
        return recorder

    execute_benchmark(config, trace_recorder_factory=factory)
    execute_benchmark(config, trace_recorder_factory=factory)

    assert len(recorders) == 2
    assert recorders[0] is not recorders[1]
    assert recorders[0].events()[0].sequence == 1
    assert recorders[1].events()[0].sequence == 1


def test_trace_recorder_failure_does_not_fail_benchmark(tmp_path: Path) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )

    execution = execute_benchmark(
        config,
        trace_recorder_factory=FailingTraceRecorder,
    )

    assert len(execution.results) == len(config.models)
    assert execution.summary["overall"]["total_samples"] == len(config.models)


def test_trace_artifact_failure_does_not_fail_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )

    def fail_trace_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("trace artifact failed")

    monkeypatch.setattr("llm_benchmark.runner.write_jsonl", fail_trace_write)

    execution = execute_benchmark(config)

    assert len(execution.results) == len(config.models)
    assert not (execution.run_dir / "trace.jsonl").exists()


def test_trace_failure_does_not_mask_provider_error(tmp_path: Path) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [f"output_dir={json.dumps(str(tmp_path))}"],
    )
    example = DatasetExample("q01", "Question?", ["A", "B"], "B", "test")

    with pytest.raises(RuntimeError, match="original provider failure"):
        _evaluate(
            "run-1",
            config,
            example,
            config.models[0],
            ExplodingProvider(),
            trace_recorder=FailingTraceRecorder(),
        )


def test_trace_failure_does_not_mask_adapter_error(tmp_path: Path) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [f"output_dir={json.dumps(str(tmp_path))}"],
    )
    example = DatasetExample("q01", "Question?", ["A", "B"], "B", "test")
    response = ProviderResponse(
        request_status="succeeded",
        raw_response="B",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_ms=1.0,
    )

    with pytest.raises(ValueError, match="original adapter failure"):
        _evaluate(
            "run-1",
            config,
            example,
            config.models[0],
            StaticProvider(response),
            task_adapter=ExplodingTaskAdapter(),
            trace_recorder=FailingTraceRecorder(),
        )


def test_trace_failures_preserve_results_summary_and_hashes(tmp_path: Path) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )

    baseline = execute_benchmark(config)
    failing_trace = execute_benchmark(
        config,
        trace_recorder_factory=FailingTraceRecorder,
    )

    assert _stable_results(baseline.results) == _stable_results(
        failing_trace.results
    )
    assert _stable_summary(baseline.summary) == _stable_summary(
        failing_trace.summary
    )
    for key in (
        "resolved_config_hash",
        "dataset_manifest_hash",
        "run_fingerprint",
        "prompt_template_hash",
    ):
        assert baseline.summary[key] == failing_trace.summary[key]


def _stable_results(results: tuple) -> list[dict[str, object]]:
    normalized = []
    for result in results:
        row = result.to_dict()
        for key in ("run_id", "started_at", "completed_at"):
            row.pop(key)
        normalized.append(row)
    return normalized


def _stable_summary(summary: dict[str, object]) -> dict[str, object]:
    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in value.items()
                if key
                not in {
                    "run_id",
                    "started_at",
                    "completed_at",
                    "run_wall_time_ms",
                }
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(summary)  # type: ignore[return-value]


def test_runner_executes_through_injected_task_adapter(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [f"output_dir={json.dumps(str(tmp_path))}"],
    )
    adapter = RecordingTaskAdapter()

    execution = execute_benchmark(
        config,
        task_adapter=adapter,
    )

    assert len(execution.results) == 16
    assert len(adapter.prompt_sample_ids) == 16
    assert len(adapter.evaluated_sample_ids) == 16
    assert adapter.prompt_sample_ids == adapter.evaluated_sample_ids


def test_runner_uses_injected_task_evaluation_outcome(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )
    adapter = AuthoritativeTaskAdapter()

    execution = execute_benchmark(
        config,
        task_adapter=adapter,
    )

    assert len(execution.results) == len(config.models)
    assert {result.model for result in execution.results} == {
        model.model_id for model in config.models
    }
    for result in execution.results:
        assert result.sample_id == "q01"
        assert result.parsed_answer is None
        assert result.parse_status == "adapter_controlled_status"
        assert result.evaluation_status == "unparseable"
        assert result.is_correct is False
    assert execution.summary["prompt_template_hash"] == "a" * 64


def test_task_prompt_hash_affects_run_fingerprint(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/mock_smoke.yaml"),
        [
            f"output_dir={json.dumps(str(tmp_path))}",
            'dataset.sample_ids=["q01"]',
            "dataset.sample_size=1",
        ],
    )

    default_execution = execute_benchmark(config)
    custom_execution = execute_benchmark(
        config,
        task_adapter=AuthoritativeTaskAdapter(),
    )

    assert (
        default_execution.summary["prompt_template_hash"]
        != custom_execution.summary["prompt_template_hash"]
    )
    assert(
        default_execution.summary["run_fingerprint"]
        != custom_execution.summary["run_fingerprint"]
    )
