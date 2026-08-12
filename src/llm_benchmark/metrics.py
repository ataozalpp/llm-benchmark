from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any

from .models import BenchmarkResult


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(results: list[BenchmarkResult], run_wall_time_ms: float) -> dict[str, Any]:
    total = len(results)
    correct = sum(r.evaluation_status == "correct" for r in results)
    incorrect = sum(r.evaluation_status == "incorrect" for r in results)
    unparseable = sum(r.evaluation_status == "unparseable" for r in results)
    failed = sum(r.evaluation_status == "request_failed" for r in results)
    request_successes = total - failed
    parseable = correct + incorrect
    known = [r for r in results if r.total_tokens is not None]
    total_prompt = sum(r.prompt_tokens or 0 for r in known)
    total_completion = sum(r.completion_tokens or 0 for r in known)
    total_tokens = sum(r.total_tokens or 0 for r in known)
    known_input = [r.input_tokens for r in results if r.input_tokens is not None]
    known_output = [r.total_output_tokens for r in results if r.total_output_tokens is not None]
    known_reasoning = [r.reasoning_output_tokens for r in results if r.reasoning_output_tokens is not None]
    throughput = [r.tokens_per_second for r in results if r.tokens_per_second is not None]
    ttft = [r.time_to_first_token_ms for r in results if r.time_to_first_token_ms is not None]
    success_latencies = [r.logical_request_latency_ms for r in results if r.request_status == "succeeded"]
    failure_latencies = [r.logical_request_latency_ms for r in results if r.request_status != "succeeded"]
    return {
        "total_samples": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "unparseable_count": unparseable,
        "failed_request_count": failed,
        "accuracy_numerator": correct,
        "accuracy_denominator": total,
        "accuracy": correct / total if total else None,
        "answered_accuracy_numerator": correct,
        "answered_accuracy_denominator": parseable,
        "answered_accuracy": correct / parseable if parseable else None,
        "request_success_rate": request_successes / total if total else None,
        "parse_success_rate": parseable / request_successes if request_successes else None,
        "format_failure_rate": unparseable / request_successes if request_successes else None,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "average_tokens_per_sample": total_tokens / total if total else None,
        "average_tokens_per_successful_request": total_tokens / request_successes if request_successes else None,
        "tokens_per_correct_answer": total_tokens / correct if correct else None,
        "token_usage_missing_count": total - len(known),
        "total_input_tokens": sum(known_input),
        "total_output_tokens": sum(known_output),
        "total_reasoning_output_tokens": sum(known_reasoning),
        "reasoning_token_usage_missing_count": total - len(known_reasoning),
        "tokens_per_second_mean": mean(throughput) if throughput else None,
        "time_to_first_token_mean_ms": mean(ttft) if ttft else None,
        "time_to_first_token_p50_ms": percentile(ttft, 0.50),
        "latency_population": "successful_logical_requests",
        "latency_count": len(success_latencies),
        "latency_mean_ms": mean(success_latencies) if success_latencies else None,
        "latency_p50_ms": percentile(success_latencies, 0.50),
        "latency_p95_ms": percentile(success_latencies, 0.95),
        "latency_min_ms": min(success_latencies) if success_latencies else None,
        "latency_max_ms": max(success_latencies) if success_latencies else None,
        "failure_latency_count": len(failure_latencies),
        "failure_latency_mean_ms": mean(failure_latencies) if failure_latencies else None,
        "run_wall_time_ms": run_wall_time_ms,
        "error_type_distribution": dict(Counter(r.error_type for r in results if r.error_type)),
    }
