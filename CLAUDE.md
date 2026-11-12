# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python 3.12 (`>=3.12,<3.13`) backend proof of concept for reproducible LLM evaluation: multiple-choice benchmarks (MMLU-Pro / fixtures), tool-use and MCP tool suites, retrieval evaluation, and RAG evaluation. Package lives in `src/llm_benchmark/` (installed as `llm-benchmark-poc`, console script `llm-benchmark`). Windows/PowerShell is the primary dev environment.

## Commands

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"            # add ,huggingface or ,mcp extras when needed

ruff check .                                  # lint gate (ruff format is NOT enforced)
pytest -q                                     # full offline suite
pytest tests/test_rag_prompts.py              # single file
pytest tests/test_rag_run.py -k some_name     # single test
pytest --run-network                          # include tests marked `network` (skipped by default)
```

Force offline mode before running tests or cached-model configs: set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Tests marked `postgres` need an explicitly configured PostgreSQL test DB; symlink tests may skip on Windows. `tests/conftest.py` puts `src/` on `sys.path` and unsets `LLM_BENCHMARK_DATABASE_URL` for every test.

CLI subcommands (all take `--config <yaml>`; `run` also accepts `--set key=value` overrides):

```powershell
python -m llm_benchmark run           --config configs/mock_smoke.yaml               # MCQ benchmark
python -m llm_benchmark tool-run      --config configs/tool_mock_smoke.yaml          # tool-use suite
python -m llm_benchmark retrieval-run --config configs/retrieval_keyword_smoke.yaml  # retrieval
python -m llm_benchmark rag-run       --config configs/rag_synthetic_smoke.yaml      # RAG
```

`*_synthetic*`, `mock_*`, `*_smoke` keyword/BM25 configs are fully offline. `*lm_studio*` / `*openai_compatible*` configs need an already-running local endpoint (the project never downloads/starts models; see `docs/local-model-setup.md`). The MCP tool config needs a separate server venv and `LLM_BENCHMARK_MCP_PYTHON` (see `docs/mcp.md`, `requirements-mcp-server.txt`). Dense/hybrid configs need the pinned embedding model in the local HF cache.

Compare two recorded runs (same suite/evaluation settings; reports which of prompt, generation, retriever varied):

```powershell
python -m scripts.compare_rag_runs --suite <suite.json> --baseline outputs/rag/<a> --candidate outputs/rag/<b> --output <new-file.json>
python -m scripts.compare_retrieval_runs --suite <suite.json> --baseline ... --candidate ... --output ...
```

API / persistence:

```powershell
alembic upgrade head                          # DB from LLM_BENCHMARK_DATABASE_URL (default sqlite:///runtime/llm_benchmark.db)
python -m llm_benchmark.api                   # FastAPI on 127.0.0.1:8000 (/docs)
python -m llm_benchmark.worker                # separate process that executes queued runs
docker compose up                             # postgres + migrate + api + worker
```

## Architecture

Four independent evaluation families share conventions but not execution paths:

- **MCQ benchmark** — two entry paths into the same runner:
  - CLI: `cli.py` → `config.py` (strict immutable Pydantic `schema_version: 1` YAML) → `runner.py` → `datasets.py` sampling → `task_adapters.py` (prompt via `prompting.py`, strict `parser.py`) → `providers.py` → `metrics.py` → artifacts in `outputs/<run_id>/` via `storage.py`. CLI bypasses the DB, repositories, and API guardrails (`full` profile allowed).
  - API: `api/` routes → `run_resolution.py` (`RegisteredRunConfigResolver` builds a `RunConfig` only from registered records; clients cannot override provider/URL/paths) → `run_preflight.py` (only `smoke`/`poc`, ≤100 samples, validates exact selection before any row is created) → queued `benchmark_runs` row. `worker.py` atomically claims runs and calls `application.py` (`BenchmarkApplicationService`), which keeps DB transactions short and never holds one during provider calls. Artifacts in `outputs/api/<run_id>/`.
  - `benchmark_runs.task_type` (`multiple_choice` | `rag`) selects the executor: `worker.py:TaskTypeDispatcher` routes to `BenchmarkApplicationService` or `rag/application.py:RagApplicationService`. RAG API runs: `api/rag_run_routes.py` -> `rag/run_resolution.py` -> `rag/run_preflight.py` -> queued run; results in `rag_case_results` (`/api/v1/rag-runs/{id}/results`). Embedding models for API dense/hybrid runs come only from `configs/api/embedding_profiles.yaml` (cache-only). The `api/` package must not import SQLAlchemy or `db.models` (enforced by a test).
  - `db/` is synchronous SQLAlchemy 2.x with repositories returning immutable records; schema via Alembic in `migrations/`. Uploaded datasets are content-addressed (`upload://sha256/...`) under runtime storage (`dataset_storage.py`, `dataset_ingestion.py`, `dataset_adapters.py`) and re-verified by checksum + path containment before use.
- **Tool use** (`tool_use/`, `tool_use/external/`, `mcp_integration/`) — CLI `tool-run` only, not wired into API/worker. Core tool contracts must not import the MCP SDK; MCP is loaded lazily only when the MCP runtime is selected, and package `__init__`s must not eagerly import submodules/optional deps. No compatibility aliases for old flat module paths.
- **Retrieval** (`retrieval/`) — `retrieval-run`: `run_config.py` → `run_service.py` → `dataset.py` → `identity.py`/`profile.py` → `suite.py` → retriever (`keyword`, `bm25`, `dense`, `hybrid` via `factory.py`) → `evaluation.py` → `reporting.py` → `artifacts.py` in `outputs/retrieval/<run_id>/`. Must not import tool-use or MCP.
- **RAG** (`rag/`) — `rag-run`: reuses retrieval retrievers and `validate_retriever_settings`, then `context.py` (whole-chunk JSON character budget, never partial chunks) → `prompts.py` → provider (`synthetic` demo or `openai_compatible`) → `parsing.py` → `evaluation.py` → `reporting.py`; artifacts in `outputs/rag/<run_id>/`. Gold answers/evidence and answerability must never reach the generation request. System prompts live in the `PROMPTS` registry in `rag/prompts.py` and are selected with `prompt_version`; registered texts are SHA-pinned in tests and never edited (add a new version instead). Optional `structured_output: json_schema` uses `rag/answer_schema.py` and enters `generation_config_hash` only when enabled.

Cross-cutting conventions to preserve:

- **Versioned identity.** Prompts, parsers, evaluators, context policies, etc. carry version constants (e.g. `PROMPT_VERSION` in `prompting.py` / `rag/prompts.py`, `RAG_EVALUATOR_VERSION` and `CONTEXT_POLICY_VERSION` in `rag/identity.py`, `EVALUATOR_VERSION` in `retrieval/identity.py`) that feed manifest hashes and run fingerprints. Changing behavior means bumping the version, and the matching tests and docs (`docs/rag.md` identity table, etc.).
- **Metrics stay separate.** Quality, format compliance, reliability, tokens, and latency are reported independently with explicit denominators; never add a composite score. Missing telemetry is `null`, never invented.
- **Case-level failures are results, not crashes.** Provider errors, unparseable/invalid responses, and incomplete generations become rows; only unexpected/programming errors fail a run. Suites take an optional `on_case_completed` callback so persistence stays out of evaluation logic; results JSONL is appended incrementally.
- **Strict config.** YAML rejects unknown fields and duplicate keys; paths in retrieval/RAG configs resolve relative to the config file. Invalid config is rejected before any output directory is created (CLI exit code 2).
- **Secrets.** Only credential *environment-variable names* are stored or registered; never commit real `model_id`/credential replacements into the opt-in configs.
- `MockProvider` / synthetic provider outputs are pipeline validation only, never model-quality claims; the same caveat applies when writing docs or summaries.

## Workflow

Full plan and phase status: `docs/roadmap.md`.

- One roadmap step = one branch from `main` (`feature/…`, `chore/…`, `release/…`) = one PR, squash-merged.
- Claude codes, tests, commits, and pushes (announce before pushing; `gh` is not installed). The maintainer opens the PR on GitHub web and merges. After a push, give the compare link `https://github.com/ataozalpp/llm-benchmark/compare/main...<branch>?expand=1`, a PR title, and a description with the checklist from `docs/roadmap.md`.
- CI (`.github/workflows/ci.yml`: ruff + offline pytest) is the merge gate; migrations additionally need local PostgreSQL tests, API/worker changes a Docker Compose smoke.
- After merge: `git switch main && git pull`, then `git branch -D <branch>` (squash makes `-d` refuse).

## Docs

`docs/architecture.md` (component table, lifecycles, package boundaries), `docs/rag.md`, `docs/retrieval.md`, `docs/mcp.md`, `docs/validation.md` (recorded test counts and calibration runs), `docs/assumptions.md`, `docs/roadmap.md` (plan and Git workflow). Behavior changes are expected to be reflected in these docs.
