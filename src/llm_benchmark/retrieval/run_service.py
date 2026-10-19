"""Run keyword retrieval with content identity, timings, and local artifacts."""

import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import RetrievalArtifactWriter
from .dataset import load_retrieval_dataset
from .identity import build_keyword_run_identity
from .keyword import KeywordRetriever
from .profile import KEYWORD_PROFILE
from .run_config import LoadedRetrievalRunConfig
from .suite import run_retrieval_suite


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_retrieval_benchmark(
    loaded: LoadedRetrievalRunConfig,
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    dataset = load_retrieval_dataset(loaded.suite_path)
    identity = build_keyword_run_identity(
        chunks=dataset.chunks, scenarios=dataset.scenarios, k=loaded.config.k
    )
    run_id = uuid4().hex
    manifest = {
        **identity,
        "artifact_schema_version": 1,
        "run_id": run_id,
        "suite_id": dataset.suite_id,
        "retriever_profile": asdict(KEYWORD_PROFILE),
        "python_version": platform.python_version(),
        "k": loaded.config.k,
        "scenario_ids": [item.query.query_id for item in dataset.scenarios],
        "started_at": _now(),
        "status": "running",
        "completed_case_count": 0,
    }
    writer = None
    latencies: dict[str, float] = {}
    index_time_ms = 0.0

    class TimedRetriever:
        def __init__(self, chunks):
            nonlocal index_time_ms
            before = time.perf_counter()
            self.delegate = KeywordRetriever(chunks)
            index_time_ms = (time.perf_counter() - before) * 1000

        def retrieve(self, query, *, top_k):
            before = time.perf_counter()
            result = self.delegate.retrieve(query, top_k=top_k)
            latencies[query.query_id] = (time.perf_counter() - before) * 1000
            return result

    def save_case(ranking, evaluation):
        writer.append_result(
            {
                "schema_version": 1,
                "run_id": run_id,
                "query_id": ranking.query_id,
                "retrieved_chunk_ids": ranking.chunk_ids,
                "metrics": asdict(evaluation),
                "retrieval_latency_ms": latencies[ranking.query_id],
            }
        )
        manifest["completed_case_count"] += 1

    try:
        writer = RetrievalArtifactWriter(loaded.output_dir / run_id)
        writer.write_json("manifest.json", manifest)
        result = run_retrieval_suite(
            chunks=dataset.chunks,
            scenarios=dataset.scenarios,
            retriever_factory=TimedRetriever,
            k=loaded.config.k,
            on_case_completed=save_case,
        )
        summary = {
            **asdict(result.summary),
            "run_id": run_id,
            "index_build_ms": index_time_ms,
            "retrieval_total_ms": sum(latencies.values()),
            "retrieval_mean_ms": sum(latencies.values()) / len(latencies),
            "wall_time_ms": (time.perf_counter() - started) * 1000,
        }
        writer.write_json("summary.json", summary)
        manifest.update(status="completed", completed_at=_now())
        writer.write_json("manifest.json", manifest)
        return writer.directory, summary
    except Exception:
        if writer is not None:
            manifest.update(
                status="failed", completed_at=_now(), error_code="retrieval_run_failed"
            )
            try:
                writer.write_json("manifest.json", manifest)
            except OSError:
                pass  # Best effort only when storage itself has failed.
        raise RuntimeError(f"Retrieval run failed; run ID: {run_id}.") from None
