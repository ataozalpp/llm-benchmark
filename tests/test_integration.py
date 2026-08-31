import json
from collections import Counter
from pathlib import Path

from llm_benchmark.config import load_config
from llm_benchmark.models import DatasetExample, ProviderResponse
from llm_benchmark.runner import execute_benchmark, run_benchmark
from llm_benchmark.task_adapters import (
    MultipleChoiceTaskAdapter,
    TaskEvaluationOutcome,
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
        example:DatasetExample,
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



def test_fixture_pipeline_end_to_end(tmp_path: Path) -> None:
    config = load_config(Path("configs/mock_smoke.yaml"), [f"output_dir={json.dumps(str(tmp_path))}"])
    run_dir, summary = run_benchmark(config)
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "dataset_manifest.json").exists()
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
