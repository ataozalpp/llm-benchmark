"""RAG metadata replacement and incremental case persistence."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from .suite import RagSuiteCase


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


class RagArtifactWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "results.jsonl").touch(exist_ok=False)

    def write_json(self, name: str, value: object) -> None:
        if name not in {"manifest.json", "summary.json"}:
            raise ValueError("Unknown RAG artifact.")
        target = self.directory / name
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)

    def append_result(self, value: object) -> None:
        line = _json(value)
        with (self.directory / "results.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def serialize_rag_case(
    case: RagSuiteCase,
    *,
    run_id: str,
    retrieval_latency_ms: float | None,
    generation_latency_ms: float | None,
) -> dict[str, object]:
    result = case.result
    response = result.response
    answer = result.answer

    return {
        "schema_version": 1,
        "run_id": run_id,
        "query_id": result.query_id,
        "status": result.status,
        "failure_code": result.failure_code,
        "retrieved_chunk_ids": list(case.ranking.chunk_ids),
        "context_chunk_ids": list(case.context.chunk_ids),
        "response": None
        if response is None
        else {
            "text": response.text,
            "finish_reason": response.finish_reason,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
        "answer": None
        if answer is None
        else {
            "answer": answer.answer,
            "citations": list(answer.citations),
            "abstained": answer.abstained,
        },
        "evaluation": None if result.evaluation is None else asdict(result.evaluation),
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
    }