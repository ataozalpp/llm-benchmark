"""Run retrieval benchmarks with content identity, timings, and local artifacts."""

import sys
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import RetrievalArtifactWriter
from .bm25 import BM25Retriever
from .dataset import load_retrieval_dataset
from .dense import DenseRetriever
from .embedding_loader import load_cpu_embedder
from .embedding_runtime import collect_embedding_runtime
from .identity import build_retrieval_run_identity
from .keyword import KeywordRetriever
from .profile import BM25_PROFILE, KEYWORD_PROFILE, DenseRetrievalProfile
from .run_config import LoadedRetrievalRunConfig
from .suite import run_retrieval_suite


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_retrieval_benchmark(
    loaded: LoadedRetrievalRunConfig,
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    dataset = load_retrieval_dataset(loaded.suite_path)

    if loaded.config.retriever == "keyword":
        profile = KEYWORD_PROFILE
    elif loaded.config.retriever == "bm25":
        profile = BM25_PROFILE
    else:
        if loaded.config.embedding is None:
            raise ValueError("Dense embedding settings are missing.")
        profile = loaded.config.embedding.to_profile()

    identity = build_retrieval_run_identity(
        chunks=dataset.chunks,
        scenarios=dataset.scenarios,
        k=loaded.config.k,
        profile=profile,
    )

    run_id = uuid4().hex
    manifest = {
        **identity,
        "artifact_schema_version": 1,
        "run_id": run_id,
        "suite_id": dataset.suite_id,
        "retriever_profile": asdict(profile),
        "python_version": platform.python_version(),
        "k": loaded.config.k,
        "scenario_ids": [item.query.query_id for item in dataset.scenarios],
        "started_at": _now(),
        "status": "running",
        "completed_case_count": 0,
    }
    writer = None
    latencies: dict[str, float] = {}
    model_load_time_ms = 0.0
    index_time_ms = 0.0
    dense_embedder = None

    def update_embedding_statistics():
        if dense_embedder is not None:
            manifest["embedding_input_statistics"] = dense_embedder.input_statistics()

    class TimedRetriever:
        def __init__(self, chunks):
            nonlocal model_load_time_ms, index_time_ms, dense_embedder

            if loaded.config.retriever == "keyword":
                before = time.perf_counter()
                self.delegate = KeywordRetriever(chunks)
                index_time_ms = (time.perf_counter() - before) * 1000
                return

            if loaded.config.retriever == "bm25":
                before = time.perf_counter()
                self.delegate = BM25Retriever(chunks)
                index_time_ms = (time.perf_counter() - before) * 1000
                return

            embedding_config = loaded.config.embedding
            if embedding_config is None:
                raise ValueError("Dense embedding settings are missing.")

            if not isinstance(profile, DenseRetrievalProfile):
                raise TypeError("Dense retrieval requires a dense profile.")

            before = time.perf_counter()
            embedder = load_cpu_embedder(
                profile,
                local_files_only=embedding_config.local_files_only,
            )
            model_load_time_ms = (time.perf_counter() - before) * 1000

            dense_embedder = embedder
            update_embedding_statistics()

            manifest["embedding_runtime"] = collect_embedding_runtime(embedder.model)
            manifest["embedding_load_policy"] = {
                "local_files_only": embedding_config.local_files_only,
            }

            if writer is None:
                raise RuntimeError("Artifact writer is not initialized.")

            writer.write_json("manifest.json", manifest)

            before = time.perf_counter()
            self.delegate = DenseRetriever(chunks, embedder=embedder)
            index_time_ms = (time.perf_counter() - before) * 1000

            update_embedding_statistics()
            writer.write_json("manifest.json", manifest)

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
        completed = manifest["completed_case_count"]
        total = len(dataset.scenarios)

        if completed % 25 == 0 or completed == total:
            print(
                f"[retrieval] Completed {completed}/{total} queries",
                file=sys.stderr,
                flush=True,
            )

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
            "model_load_ms": model_load_time_ms,
            "index_build_ms": index_time_ms,
            "retrieval_total_ms": sum(latencies.values()),
            "retrieval_mean_ms": sum(latencies.values()) / len(latencies),
            "wall_time_ms": (time.perf_counter() - started) * 1000,
        }
        writer.write_json("summary.json", summary)
        update_embedding_statistics()
        manifest.update(status="completed", completed_at=_now())
        writer.write_json("manifest.json", manifest)
        return writer.directory, summary
    except KeyboardInterrupt:
        if writer is not None:
            manifest.update(
                status="failed",
                completed_at=_now(),
                error_code="retrieval_run_interrupted",
            )
            try:
                update_embedding_statistics()
                writer.write_json("manifest.json", manifest)
            except OSError:
                pass
        raise
    except Exception:
        if writer is not None:
            manifest.update(
                status="failed", completed_at=_now(), error_code="retrieval_run_failed"
            )
            try:
                update_embedding_statistics()
                writer.write_json("manifest.json", manifest)
            except OSError:
                pass  # Best effort only when storage itself has failed.
        raise RuntimeError(f"Retrieval run failed; run ID: {run_id}.") from None
