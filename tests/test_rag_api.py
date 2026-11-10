"""Registered RAG runs through the API and the worker."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from llm_benchmark.api import create_app
from llm_benchmark.db.registry import create_registry_repositories
from llm_benchmark.rag import run_service
from llm_benchmark.rag.run_preflight import RagRunGuardrailPolicy
from llm_benchmark.rag.run_service import execute_rag_benchmark
from llm_benchmark.retrieval import factory
from llm_benchmark.retrieval.run_config import DenseEmbeddingConfig
from llm_benchmark.worker import create_worker

SUITE = Path(__file__).parents[1] / "examples/rag/policy_qa_smoke.json"
VOCABULARY = ("refund", "shipping", "warranty", "password", "gift")
PROFILES = {
    "test-encoder": DenseEmbeddingConfig(
        model_id="test/encoder",
        model_revision="a" * 40,
        embedding_dimension=len(VOCABULARY) + 1,
        max_sequence_length=256,
    )
}


class VocabularyEmbedder:
    def __init__(self):
        self.model = object()

    def _vector(self, text):
        lowered = text.casefold()
        return tuple(float(word in lowered) for word in VOCABULARY) + (0.01,)

    def encode_documents(self, texts):
        return tuple(self._vector(text) for text in texts)

    def encode_query(self, text):
        return self._vector(text)

    def input_statistics(self):
        return {"schema_version": 1}


class FakeTransport:
    def __init__(self):
        self.calls = []

    def post_json(self, url, payload, timeout_seconds, headers=None):
        self.calls.append((url, payload, timeout_seconds))
        answer = {"answer": "", "citations": [], "abstained": True}
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(answer)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    loads = []

    def load(profile, *, local_files_only):
        loads.append((profile, local_files_only))
        return VocabularyEmbedder()

    monkeypatch.setattr(factory, "load_cpu_embedder", load)
    monkeypatch.setattr(
        run_service,
        "collect_embedding_runtime",
        lambda model: {"schema_version": 1, "runtime_versions": {"test": "1"}},
    )
    return loads


@pytest.fixture
def rag_api(tmp_path: Path) -> Iterator[dict[str, Any]]:
    database_url = f"sqlite:///{(tmp_path / 'rag-api.sqlite').as_posix()}"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    registry = create_registry_repositories(database_url)
    storage_root = tmp_path / "datasets"
    output_root = tmp_path / "artifacts"
    state = {"embeddings": True}
    app = create_app(
        registry,
        run_output_root=output_root,
        dataset_storage_root=storage_root,
        rag_guardrail_policy=RagRunGuardrailPolicy(max_scenarios=10),
        embedding_profiles=PROFILES,
        embeddings_check=lambda: state["embeddings"],
    )
    with TestClient(app) as client:
        yield {
            "client": client,
            "registry": registry,
            "storage_root": storage_root,
            "output_root": output_root,
            "state": state,
        }


def register(context, *, provider_type="mock", credential=None, defaults=None):
    client = context["client"]
    endpoint_payload = {
        "name": f"{provider_type}-endpoint",
        "provider_type": provider_type,
        "base_url": "http://127.0.0.1:1234/v1",
    }
    if credential is not None:
        endpoint_payload["credential_env_var"] = credential
    endpoint = client.post("/api/v1/endpoints", json=endpoint_payload).json()
    model = client.post(
        "/api/v1/models",
        json={
            "name": f"{provider_type}-model",
            "model_identifier": "test-model",
            "endpoint_id": endpoint["id"],
            "reasoning_policy": "unsupported",
            "default_generation_config": defaults or {},
        },
    ).json()
    dataset = client.post(
        "/api/v1/datasets/upload",
        data={"name": f"policy-qa-{provider_type}", "file_format": "rag_json"},
        files={"file": ("suite.json", SUITE.read_bytes(), "application/json")},
    ).json()
    return model["id"], dataset["id"]


def submit(context, model_id, dataset_id, **overrides):
    payload = {
        "experiment_name": "rag-api",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "retriever": "bm25",
        "k": 2,
        "max_context_chars": 2000,
    }
    payload.update(overrides)
    return context["client"].post("/api/v1/rag-runs", json=payload)


def work(context, **kwargs):
    worker = create_worker(
        registry_factory=lambda: context["registry"],
        dataset_storage_root=context["storage_root"],
        **kwargs,
    )
    return worker.run_once()


def run_count(context):
    return len(context["client"].get("/api/v1/runs").json())


def test_synthetic_bm25_run_end_to_end(rag_api, fake_embeddings):
    model_id, dataset_id = register(rag_api)
    response = submit(rag_api, model_id, dataset_id)

    assert response.status_code == 201
    queued = response.json()
    assert queued["status"] == "queued"
    assert queued["task_type"] == "rag"
    assert queued["seed"] is None
    assert queued["sample_count"] == 5
    assert queued["summary"] is None

    assert work(rag_api) is True
    run = rag_api["client"].get(f"/api/v1/runs/{queued['id']}").json()
    assert run["status"] == "completed"
    assert run["summary"]["case_count"] == 5
    assert run["summary"]["status_counts"]["completed"] == 5
    assert fake_embeddings == []

    results = rag_api["client"].get(f"/api/v1/rag-runs/{queued['id']}/results").json()
    assert [row["query_id"] for row in results] == [
        "q-refund",
        "q-shipping",
        "q-warranty",
        "q-password",
        "q-gift-card",
    ]
    assert sum(not row["answerable"] for row in results) == 1
    assert all(row["context_chunk_ids"] for row in results[:4])

    persisted = rag_api["registry"].runs.get_by_id(queued["id"])
    artifacts = Path(persisted.artifact_directory)
    assert artifacts.parent == rag_api["output_root"] / "rag"
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"] == {"provider": "synthetic"}
    assert manifest["prompt_version"] == "rag-grounded-json-v1"


def test_mcq_results_route_rejects_rag_runs_and_vice_versa(rag_api):
    model_id, dataset_id = register(rag_api)
    run_id = submit(rag_api, model_id, dataset_id).json()["id"]
    client = rag_api["client"]

    assert client.get(f"/api/v1/runs/{run_id}/results").status_code == 409
    assert client.get("/api/v1/rag-runs/999999/results").status_code == 404


def test_openai_compatible_run_uses_registered_model_and_options(rag_api):
    model_id, dataset_id = register(
        rag_api,
        provider_type="openai_compatible",
        defaults={"max_output_tokens": 128, "timeout_seconds": 5},
    )
    response = submit(
        rag_api,
        model_id,
        dataset_id,
        prompt_version="rag-grounded-json-v3",
        structured_output="json_schema",
    )
    assert response.status_code == 201

    transport = FakeTransport()
    assert work(
        rag_api,
        rag_executor=lambda loaded: execute_rag_benchmark(loaded, transport=transport),
    )

    run = rag_api["client"].get(f"/api/v1/runs/{response.json()['id']}").json()
    assert run["status"] == "completed"
    assert run["summary"]["unanswerable_abstention"]["numerator"] == 1
    url, payload, timeout = transport.calls[0]
    assert url == "http://127.0.0.1:1234/v1/chat/completions"
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 128
    assert payload["response_format"]["type"] == "json_schema"
    assert timeout == 5.0
    results = rag_api["client"].get(
        f"/api/v1/rag-runs/{response.json()['id']}/results"
    ).json()
    assert results[0]["input_tokens"] == 10
    assert results[0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "retriever, extra",
    [
        ("dense", {}),
        ("hybrid", {"hybrid": {"candidate_k": 5, "rank_constant": 60}}),
    ],
)
def test_dense_and_hybrid_runs_use_cache_only_profiles(
    rag_api, fake_embeddings, retriever, extra
):
    model_id, dataset_id = register(rag_api)
    response = submit(
        rag_api,
        model_id,
        dataset_id,
        retriever=retriever,
        embedding_profile="test-encoder",
        **extra,
    )
    assert response.status_code == 201
    assert work(rag_api)

    run = rag_api["client"].get(f"/api/v1/runs/{response.json()['id']}").json()
    assert run["status"] == "completed"
    ((profile, local_files_only),) = fake_embeddings
    assert profile.model_id == "test/encoder"
    assert local_files_only is True


@pytest.mark.parametrize(
    "overrides, status, code",
    [
        ({"retriever": "dense"}, 422, None),
        ({"embedding_profile": "test-encoder"}, 422, None),
        ({"retriever": "hybrid", "embedding_profile": "test-encoder"}, 422, None),
        ({"retriever": "dense", "embedding_profile": "unknown"}, 422, "invalid_rag_run_request"),
        ({"prompt_version": "rag-grounded-json-v9"}, 422, None),
        ({"structured_output": "json_schema"}, 422, "invalid_rag_run_request"),
        ({"k": 21}, 422, "run_guardrail_violation"),
        ({"max_context_chars": 20_001}, 422, "run_guardrail_violation"),
        ({"suite_path": "/etc/passwd"}, 422, None),
        ({"output_dir": "elsewhere"}, 422, None),
    ],
)
def test_invalid_requests_create_no_run(rag_api, overrides, status, code):
    model_id, dataset_id = register(rag_api)
    response = submit(rag_api, model_id, dataset_id, **overrides)

    assert response.status_code == status
    if code is not None:
        assert response.json()["code"] == code
    assert run_count(rag_api) == 0
    assert not rag_api["output_root"].exists()


def test_missing_embedding_dependency_is_reported(rag_api):
    rag_api["state"]["embeddings"] = False
    model_id, dataset_id = register(rag_api)
    response = submit(
        rag_api, model_id, dataset_id, retriever="dense", embedding_profile="test-encoder"
    )
    assert response.status_code == 409
    assert response.json()["code"] == "embeddings_unavailable"
    assert run_count(rag_api) == 0


def test_suite_over_scenario_limit_is_rejected(rag_api, tmp_path):
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    base = suite["scenarios"][0]
    suite["scenarios"] = [
        {**base, "query_id": f"q{index}"} for index in range(11)
    ]
    client = rag_api["client"]
    model_id, _ = register(rag_api)
    dataset = client.post(
        "/api/v1/datasets/upload",
        data={"name": "large", "file_format": "rag_json"},
        files={"file": ("suite.json", json.dumps(suite).encode(), "application/json")},
    ).json()

    response = submit(rag_api, model_id, dataset["id"])
    assert response.status_code == 422
    assert run_count(rag_api) == 0


@pytest.mark.parametrize(
    "registration",
    [
        {"provider_type": "openai_compatible", "credential": "RAG_API_KEY"},
        {"provider_type": "lm_studio_native"},
        {"provider_type": "openai_compatible", "defaults": {"temperature": 0.7}},
    ],
)
def test_unsupported_registrations_are_rejected(rag_api, registration):
    model_id, dataset_id = register(rag_api, **registration)
    response = submit(rag_api, model_id, dataset_id)
    assert response.status_code == 409
    assert run_count(rag_api) == 0


def test_mcq_dataset_cannot_start_a_rag_run(rag_api):
    client = rag_api["client"]
    model_id, _ = register(rag_api)
    csv_dataset = client.post(
        "/api/v1/datasets/upload",
        data={"name": "mcq", "file_format": "csv"},
        files={
            "file": (
                "mcq.csv",
                b"sample_id,question,option_A,option_B,correct_answer,category\n"
                b"1,Q?,A,B,B,x\n",
                "text/csv",
            )
        },
    ).json()
    response = submit(rag_api, model_id, csv_dataset["id"])
    assert response.status_code == 409
    assert run_count(rag_api) == 0


def test_tampered_suite_fails_the_run_without_results(rag_api):
    model_id, dataset_id = register(rag_api)
    run_id = submit(rag_api, model_id, dataset_id).json()["id"]
    (stored,) = rag_api["storage_root"].rglob("*.json")
    stored.write_bytes(stored.read_bytes() + b" ")

    assert work(rag_api)
    run = rag_api["client"].get(f"/api/v1/runs/{run_id}").json()
    assert run["status"] == "failed"
    assert rag_api["client"].get(f"/api/v1/rag-runs/{run_id}/results").json() == []


def test_worker_drains_a_queued_rag_run(rag_api):
    model_id, dataset_id = register(rag_api)
    rag_id = submit(rag_api, model_id, dataset_id).json()["id"]

    assert work(rag_api) is True
    assert work(rag_api) is False
    assert rag_api["client"].get(f"/api/v1/runs/{rag_id}").json()["status"] == (
        "completed"
    )
