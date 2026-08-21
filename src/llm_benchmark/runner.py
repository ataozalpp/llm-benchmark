from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RunConfig
from .datasets import load_and_sample
from .metrics import summarize
from .models import BenchmarkResult, DatasetExample
from .parser import parse_multiple_choice
from .prompting import build_prompt, prompt_hash
from .providers import Provider, create_provider
from .reproducibility import canonical_hash, environment_snapshot
from .storage import append_result, write_json


@dataclass(frozen=True)
class PipelineExecution:
    run_dir: Path
    summary: dict[str, Any]
    results: tuple[BenchmarkResult, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluate(
    run_id: str, config: RunConfig, example: DatasetExample, model_config: Any, provider: Provider | None = None
) -> BenchmarkResult:
    prompt = build_prompt(example)
    provider = provider or create_provider(model_config)
    started_at = utc_now()
    response = provider.generate(prompt, example)
    completed_at = utc_now()
    parsed = parse_multiple_choice(response.raw_response, example.allowed_labels)
    if response.request_status != "succeeded":
        evaluation_status = "request_failed"
    elif parsed.parsed_answer is None:
        evaluation_status = "unparseable"
    elif parsed.parsed_answer == example.correct_answer:
        evaluation_status = "correct"
    else:
        evaluation_status = "incorrect"
    return BenchmarkResult(
        run_id=run_id,
        sample_id=example.sample_id,
        dataset_name=config.dataset.name,
        dataset_split=config.dataset.split,
        category=example.category,
        provider=model_config.provider,
        endpoint_alias=model_config.endpoint_alias,
        model=model_config.model_id,
        returned_model=response.returned_model,
        raw_response=response.raw_response,
        parsed_answer=parsed.parsed_answer,
        correct_answer=example.correct_answer,
        is_correct=evaluation_status == "correct",
        evaluation_status=evaluation_status,
        parse_status=parsed.parse_status,
        request_status=response.request_status,
        error_type=response.error_type,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        estimated_cost=None,
        attempt_count=1,
        attempt_latency_ms=response.latency_ms,
        logical_request_latency_ms=response.latency_ms,
        started_at=started_at,
        completed_at=completed_at,
        parser_version=config.evaluation.parser_version,
        evaluator_version=config.evaluation.evaluator_version,
        reasoning_mode=model_config.reasoning,
        input_tokens=response.input_tokens,
        total_output_tokens=response.total_output_tokens,
        reasoning_output_tokens=response.reasoning_output_tokens,
        final_output_tokens=response.final_output_tokens,
        reasoning_observed=response.reasoning_observed,
        tokens_per_second=response.tokens_per_second,
        time_to_first_token_ms=response.time_to_first_token_ms,
        stop_reason=response.stop_reason,
        timeout_seconds=model_config.timeout_seconds,
        output_budget_provenance=(
            model_config.output_budget_provenance
            if model_config.provider in {"lm_studio", "openai_compatible"}
            else None
        ),
        http_status_code=response.http_status_code,
        provider_error_type=response.provider_error_type,
        provider_error_code=response.provider_error_code,
        provider_error_message=response.provider_error_message,
    )


def execute_benchmark(config: RunConfig) -> PipelineExecution:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = config.model_dump(mode="json")
    config_hash = canonical_hash(resolved)
    examples, manifest = load_and_sample(config.dataset, config.seed)
    manifest_hash = canonical_hash(manifest)
    environment = environment_snapshot()
    fingerprint_parts = {
        "resolved_config_hash": config_hash,
        "dataset_manifest_hash": manifest_hash,
        "prompt_template_hash": prompt_hash(),
        "parser_version": config.evaluation.parser_version,
        "evaluator_version": config.evaluation.evaluator_version,
        "git_revision": environment["git_commit"],
    }
    run_fingerprint = canonical_hash(fingerprint_parts)
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "dataset_manifest.json", {**manifest, "dataset_manifest_hash": manifest_hash})
    write_json(run_dir / "environment.json", environment)
    results: list[BenchmarkResult] = []
    started_at = utc_now()
    started_perf = time.perf_counter()
    for model_config in config.models:
        provider = create_provider(model_config)
        for example in examples:
            result = _evaluate(run_id, config, example, model_config, provider)
            append_result(run_dir / "results.jsonl", result)
            results.append(result)
    wall_time_ms = (time.perf_counter() - started_perf) * 1000
    completed_at = utc_now()
    by_model: dict[str, list[BenchmarkResult]] = defaultdict(list)
    by_category: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        by_model[result.model].append(result)
        by_category[f"{result.model}:{result.category}"].append(result)
    summary = {
        "run_id": run_id,
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "resolved_config_hash": config_hash,
        "dataset_manifest_hash": manifest_hash,
        "run_fingerprint": run_fingerprint,
        "prompt_template_hash": prompt_hash(),
        "parser_version": config.evaluation.parser_version,
        "evaluator_version": config.evaluation.evaluator_version,
        "dataset": manifest,
        "providers": sorted({m.provider for m in config.models}),
        "models": [m.model_id for m in config.models],
        "started_at": started_at,
        "completed_at": completed_at,
        "overall": summarize(results, wall_time_ms),
        "by_model": {name: summarize(items, wall_time_ms) for name, items in sorted(by_model.items())},
        "by_model_category": {name: summarize(items, wall_time_ms) for name, items in sorted(by_category.items())},
    }
    write_json(run_dir / "summary.json", summary)
    return PipelineExecution(run_dir=run_dir, summary=summary, results=tuple(results))


def run_benchmark(config: RunConfig) -> tuple[Path, dict[str, Any]]:
    """Preserve the original CLI-facing runner contract."""

    execution = execute_benchmark(config)
    return execution.run_dir, execution.summary
