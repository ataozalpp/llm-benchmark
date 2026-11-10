"""Run RAG benchmarks with content identity, timings and local artifacts."""

import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..providers import JsonTransport
from ..retrieval.embedding_runtime import collect_embedding_runtime
from ..retrieval.factory import build_retriever
from ..retrieval.run_config import build_retrieval_profile
from .answer_schema import ANSWER_SCHEMA_VERSION
from .artifacts import RagArtifactWriter, serialize_rag_case
from .dataset import load_rag_dataset
from .demo_provider import SyntheticRagProvider
from .identity import build_rag_run_identity
from .openai_compatible import OpenAICompatibleRagProvider
from .reporting import summarize_rag
from .run_config import LoadedRagRunConfig
from .suite import run_rag_suite


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_provider(loaded: LoadedRagRunConfig, transport: JsonTransport | None):
    config = loaded.config
    if config.provider == "synthetic":
        return SyntheticRagProvider(), {"provider": "synthetic"}

    model = config.model
    if model is None:
        raise ValueError("Model settings are missing.")

    provider = OpenAICompatibleRagProvider(
        base_url=model.base_url,
        model_id=model.model_id,
        timeout_seconds=model.timeout_seconds,
        max_tokens=model.max_output_tokens,
        transport=transport,
        prompt_version=config.prompt_version,
        structured_output=config.structured_output,
    )
    generation = {
        "provider": "openai_compatible",
        "endpoint_alias": model.endpoint_alias,
        "model_id": model.model_id,
        "temperature": 0.0,
        "max_tokens": model.max_output_tokens,
    }
    # Added only when enabled so unconstrained runs keep their earlier
    # generation_config_hash.
    if config.structured_output != "none":
        generation["structured_output"] = config.structured_output
        generation["answer_schema_version"] = ANSWER_SCHEMA_VERSION
    return provider, generation


def run_rag_benchmark(
    loaded: LoadedRagRunConfig,
    *,
    transport: JsonTransport | None = None,
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    config = loaded.config
    dataset = load_rag_dataset(loaded.suite_path)

    profile = build_retrieval_profile(
        retriever=config.retriever,
        embedding=config.embedding,
        hybrid=config.hybrid,
    )
    local_files_only = (
        config.embedding.local_files_only if config.embedding is not None else True
    )
    provider, generation = _build_provider(loaded, transport)

    identity = build_rag_run_identity(
        chunks=dataset.chunks,
        scenarios=dataset.scenarios,
        retriever_profile=asdict(profile),
        k=config.k,
        max_context_chars=config.max_context_chars,
        generation=generation,
        prompt_version=config.prompt_version,
    )

    run_id = uuid4().hex
    manifest = {
        **identity,
        "artifact_schema_version": 1,
        "run_id": run_id,
        "suite_id": dataset.suite_id,
        "retriever_profile": asdict(profile),
        "generation": generation,
        "python_version": platform.python_version(),
        "k": config.k,
        "max_context_chars": config.max_context_chars,
        "scenario_ids": [item.query.query_id for item in dataset.scenarios],
        "started_at": _now(),
        "status": "running",
        "completed_case_count": 0,
    }
    writer = None
    embedder = None
    retrieval_latencies: dict[str, float] = {}
    generation_latencies: dict[str, float] = {}

    def update_embedding_statistics():
        if embedder is not None:
            manifest["embedding_input_statistics"] = embedder.input_statistics()

    def record_embedder(loaded_embedder):
        nonlocal embedder
        embedder = loaded_embedder
        manifest["embedding_runtime"] = collect_embedding_runtime(embedder.model)
        manifest["embedding_load_policy"] = {"local_files_only": local_files_only}
        update_embedding_statistics()
        writer.write_json("manifest.json", manifest)

    def save_case(case):
        query_id = case.result.query_id
        writer.append_result(
            serialize_rag_case(
                case,
                run_id=run_id,
                retrieval_latency_ms=retrieval_latencies.get(query_id),
                generation_latency_ms=generation_latencies.get(query_id),
            )
        )
        manifest["completed_case_count"] += 1
        completed = manifest["completed_case_count"]
        total = len(dataset.scenarios)

        if completed % 10 == 0 or completed == total:
            print(
                f"[rag] Completed {completed}/{total} queries",
                file=sys.stderr,
                flush=True,
            )

    def mark_failed(error_code: str) -> None:
        if writer is None:
            return
        manifest.update(status="failed", completed_at=_now(), error_code=error_code)
        try:
            update_embedding_statistics()
            writer.write_json("manifest.json", manifest)
        except OSError:
            pass  # Best effort only when storage itself has failed.

    try:
        writer = RagArtifactWriter(loaded.output_dir / run_id)
        writer.write_json("manifest.json", manifest)

        built = build_retriever(
            profile,
            dataset.chunks,
            local_files_only=local_files_only,
            on_embedder_loaded=record_embedder,
        )
        update_embedding_statistics()
        writer.write_json("manifest.json", manifest)

        class TimedRetriever:
            def retrieve(self, query, *, top_k):
                before = time.perf_counter()
                result = built.retriever.retrieve(query, top_k=top_k)
                retrieval_latencies[query.query_id] = (
                    time.perf_counter() - before
                ) * 1000
                return result

        class TimedProvider:
            def generate(self, request):
                before = time.perf_counter()
                try:
                    return provider.generate(request)
                finally:
                    generation_latencies[request.query.query_id] = (
                        time.perf_counter() - before
                    ) * 1000

        result = run_rag_suite(
            chunks=dataset.chunks,
            scenarios=dataset.scenarios,
            retriever=TimedRetriever(),
            provider=TimedProvider(),
            k=config.k,
            max_context_chars=config.max_context_chars,
            on_case_completed=save_case,
        )
        summary = {
            **summarize_rag(scenarios=dataset.scenarios, cases=result.cases),
            "run_id": run_id,
            "model_load_ms": built.model_load_ms,
            "index_build_ms": built.index_build_ms,
            "retrieval_total_ms": sum(retrieval_latencies.values()),
            "generation_total_ms": sum(generation_latencies.values()),
            "wall_time_ms": (time.perf_counter() - started) * 1000,
        }
        writer.write_json("summary.json", summary)
        update_embedding_statistics()
        manifest.update(status="completed", completed_at=_now())
        writer.write_json("manifest.json", manifest)
        return writer.directory, summary
    except KeyboardInterrupt:
        mark_failed("rag_run_interrupted")
        raise
    except Exception:
        mark_failed("rag_run_failed")
        raise RuntimeError(f"RAG run failed; run ID: {run_id}.") from None
