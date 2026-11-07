"""RAG artifact writing and per-case row serialization."""

import json

import pytest

from llm_benchmark.rag.artifacts import RagArtifactWriter, serialize_rag_case
from llm_benchmark.rag.case import RagCaseResult
from llm_benchmark.rag.evaluation import evaluate_rag_answer
from llm_benchmark.rag.models import RagAnswer, RagContext, RagScenario
from llm_benchmark.rag.providers import RagGenerationResponse
from llm_benchmark.rag.suite import RagSuiteCase
from llm_benchmark.retrieval.models import Chunk, RankedRetrieval, RetrievalQuery

REFUND = Chunk("refund", "policy", "Refunds within 30 days.")
SCENARIO = RagScenario(
    query=RetrievalQuery("q1", "Refund period?"),
    reference_answers=("30 days",),
    relevant_chunk_ids=frozenset({"refund"}),
)
CONTEXT = RagContext("q1", (REFUND,))
RANKING = RankedRetrieval("q1", ("refund", "other"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def case(status):
    response = None
    answer = None
    evaluation = None

    if status != "provider_failed":
        response = RagGenerationResponse(
            text="not json",
            finish_reason="length" if status == "incomplete_response" else "stop",
            input_tokens=12,
            output_tokens=None,
        )
    if status == "completed":
        answer = RagAnswer(answer="30 days", citations=("refund",))
        evaluation = evaluate_rag_answer(
            scenario=SCENARIO, context=CONTEXT, answer=answer
        )

    return RagSuiteCase(
        ranking=RANKING,
        context=CONTEXT,
        result=RagCaseResult(
            query_id="q1",
            status=status,
            response=response,
            answer=answer,
            evaluation=evaluation,
        ),
    )


def test_writer_creates_empty_results_and_rejects_reuse(tmp_path):
    writer = RagArtifactWriter(tmp_path / "run")
    assert (writer.directory / "results.jsonl").read_text(encoding="utf-8") == ""
    with pytest.raises(FileExistsError):
        RagArtifactWriter(writer.directory)


def test_writer_replaces_json_without_temporary_files(tmp_path):
    writer = RagArtifactWriter(tmp_path / "run")
    writer.write_json("manifest.json", {"status": "running"})
    writer.write_json("manifest.json", {"status": "completed"})
    writer.write_json("summary.json", {"case_count": 1})
    assert read_json(writer.directory / "manifest.json") == {"status": "completed"}
    assert read_json(writer.directory / "summary.json") == {"case_count": 1}
    assert not list(writer.directory.glob("*.tmp"))


@pytest.mark.parametrize("name", ["results.jsonl", "../escape.json", "other.json"])
def test_writer_rejects_unknown_artifacts(tmp_path, name):
    writer = RagArtifactWriter(tmp_path / "run")
    with pytest.raises(ValueError, match="^Unknown RAG artifact.$"):
        writer.write_json(name, {})


def test_invalid_values_preserve_existing_content(tmp_path):
    writer = RagArtifactWriter(tmp_path / "run")
    writer.write_json("manifest.json", {"status": "running"})
    writer.append_result({"query_id": "one"})
    with pytest.raises(ValueError):
        writer.write_json("manifest.json", {"bad": float("nan")})
    with pytest.raises(ValueError):
        writer.append_result({"bad": float("inf")})
    assert read_json(writer.directory / "manifest.json") == {"status": "running"}
    lines = (writer.directory / "results.jsonl").read_text(encoding="utf-8")
    assert lines.splitlines() == ['{"query_id": "one"}']


def test_completed_case_row():
    row = serialize_rag_case(
        case("completed"),
        run_id="run",
        retrieval_latency_ms=1.5,
        generation_latency_ms=20.0,
    )
    assert row["schema_version"] == 1
    assert row["status"] == "completed"
    assert row["retrieved_chunk_ids"] == ["refund", "other"]
    assert row["context_chunk_ids"] == ["refund"]
    assert row["answer"] == {
        "answer": "30 days",
        "citations": ["refund"],
        "abstained": False,
    }
    assert row["evaluation"]["answer_exact_match"] is True
    assert row["response"]["input_tokens"] == 12
    assert row["response"]["output_tokens"] is None
    assert row["retrieval_latency_ms"] == 1.5
    assert row["generation_latency_ms"] == 20.0


@pytest.mark.parametrize("status", ["invalid_response", "incomplete_response"])
def test_unsuccessful_rows_keep_raw_response_without_evaluation(status):
    row = serialize_rag_case(
        case(status), run_id="run", retrieval_latency_ms=None, generation_latency_ms=None
    )
    assert row["status"] == status
    assert row["response"]["text"] == "not json"
    assert row["answer"] is None
    assert row["evaluation"] is None


def test_provider_failure_row_has_no_response():
    row = serialize_rag_case(
        case("provider_failed"),
        run_id="run",
        retrieval_latency_ms=None,
        generation_latency_ms=3.0,
    )
    assert row["response"] is None
    assert row["answer"] is None
    assert row["evaluation"] is None


@pytest.mark.parametrize(
    "status",
    ["completed", "invalid_response", "incomplete_response", "provider_failed"],
)
def test_rows_are_strict_json_without_context_text(status):
    row = serialize_rag_case(
        case(status), run_id="run", retrieval_latency_ms=1.0, generation_latency_ms=1.0
    )
    encoded = json.dumps(row, allow_nan=False)
    assert REFUND.text not in encoded
    assert "reference_answers" not in encoded
