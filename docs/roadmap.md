# Roadmap and Git workflow

This is the repository copy of the working plan. It tracks which parts of the
long-term "LLM evaluation platform" goal are implemented, the order of the
remaining steps, and the branch/PR rules every step follows.

Levels:

- **S1 — MCQ backend.** Done.
- **S2 — Agentic Evaluation POC.** Current target, released as `v0.3.0`.
- **S3 — Production-oriented** (worker reliability, security, vLLM). Released
  as `v1.0.0`.

Releases so far: `v0.2.0` (RAG evaluation through the API and worker).

## Phase status

| Phase | Status | Evidence / gap |
|---|---|---|
| 0 Baseline | Done | Offline suite, PostgreSQL and Docker smoke recorded in [Validation](validation.md) |
| 1 Task adapter | Done | `task_adapters.py` (`MultipleChoiceTaskAdapter`) |
| 2 Execution trace | Partial | `trace.py` covers MCQ events only; RAG has no trace; tool runs use a separate format |
| 3 Tool runtime | Done | `tool_use/runtime.py`, `builtin_tools.py` |
| 4 Tool-call normalization | Done | `tool_use/calling.py`, `generate_tool_turn()` |
| 5 Bounded loop | Done | `tool_use/loop.py` |
| 6 Tool/trajectory evaluation | Partial | `tool_use/evaluation.py` checks one expected sequence; no no-tool scenarios, alternative trajectories, or error taxonomy |
| 7 MCP client | Done | `mcp_integration/` (stdio, separate venv) |
| 8 Retrieval | Partial | keyword/BM25/dense/hybrid + SciFact; project corpus is small |
| 9 RAG generation | Partial | Deterministic metrics, prompt registry, constrained decoding; no judge |
| 10 Agentic RAG | Missing | No `search_documents` tool |
| 11 DB extension | Partial | `task_type`, `rag_case_results` |
| 12 Agentic run API | Partial | RAG API only; no tool/agent API or trace endpoint |
| 13–15, 17, 18 | Missing | Worker reliability, security, vLLM, UI, release |
| 16 Comparison API | Missing | MCQ service exists without a route; RAG comparison is script-only |

## Workflow

- **Roles.** Claude writes code, runs tests, commits, and pushes (announcing
  the push). The maintainer opens the PR on GitHub web and squash-merges it.
  After each push Claude provides the compare link
  (`https://github.com/ataozalpp/llm-benchmark/compare/main...<branch>?expand=1`),
  a PR title, and a description containing the checklist below.
- **One session = one step = one branch.** The roadmap is not executed in one
  go; each step ends with a push and PR text.
- **Needs the maintainer:** PostgreSQL URL / Docker Desktop running, real-model
  runs (LM Studio), opening and merging PRs, approving remote branch deletion.
- **Interruption safety:** small, frequent commits and a push at the end of each
  session; a new session resumes from Git state and this file.

### Branch and PR rules

1. Each step is one branch from `main` (`feature/<topic>`, `chore/<topic>`,
   `release/<version>`); one branch = one PR = one squash commit on `main`.
2. Push early (`git push -u origin <branch>`) and at the end of every session.
3. Open a draft PR from the provided link; mark it ready when the gate passes,
   then **Squash and merge** and delete the branch on GitHub.
4. **Merge gate:** CI green (ruff + offline pytest); for migrations, SQLite
   upgrade/downgrade/re-upgrade and `pytest tests/test_postgresql.py` locally;
   for API/worker changes, a Docker Compose smoke; README/docs/validation
   updated; no `outputs/`, `runtime/`, model or dataset caches committed.
5. **After merge:** `git switch main && git pull`, delete the local branch with
   `git branch -D` (squash makes `-d` refuse), start the next branch from the
   updated `main`.
6. **Tags:** annotated tags plus a GitHub Release at milestones: `v0.2.0`,
   `v0.3.0`, `v1.0.0`.
7. **Size:** a PR is roughly one to two weeks of work; split larger phases.

### PR checklist

```text
[ ] CI: ruff check . and offline pytest green
[ ] Tests for new/changed behavior; existing contracts (config hashes, prompt/evaluator versions, artifact fields) preserved or versions bumped
[ ] Migration: SQLite upgrade/downgrade/re-upgrade + pytest tests/test_postgresql.py (local)
[ ] API/worker changed: docker compose smoke (submit -> worker -> completed)
[ ] README / docs / docs/validation.md updated
[ ] outputs/, runtime/, model and dataset caches not committed
[ ] Branch deleted after merge
```

## Steps

### Step 0 — Close `feature/rag-prompt-v2` · done

PostgreSQL tests and Docker smoke recorded, PR squash-merged, tag `v0.2.0`.

### Step 1 — Repository hygiene and CI · `chore/ci-and-branch-cleanup`

- This file and a short Workflow section in `CLAUDE.md`.
- `.github/workflows/ci.yml`: Python 3.12, `pip install -e ".[dev]"`, offline
  Hugging Face environment, `ruff check .`, `pytest -q`. PostgreSQL, MCP-SDK
  and network tests skip themselves in CI.
- Delete stale remote branches whose content is already on `main` under
  squash commits: `feature/external-tool-policy`, `feature/mcp-tool-executor`,
  `feature/retrieval-evaluation`, `test/mcp-real-model-smoke`, plus the local
  `feature/rag-failure-cases`.
- After merge: enable branch protection on `main` with the CI status check
  required.

### S2 — Agentic Evaluation POC

| # | Branch | Phase | Scope |
|---|---|---|---|
| 2.1 | `feature/unified-trace` | 2 | Event types `retrieval_query`, `retrieval_result`, `tool_call`, `tool_result`, `final_answer`; RAG runs write `trace.jsonl`; tool-run traces move to the shared schema (existing fields kept); secret/path tests |
| 2.2 | `feature/trajectory-evaluation` | 6 | `tool_necessity` (no-tool scenarios), multiple accepted trajectories, error taxonomy (`wrong_tool`, `missing_tool_call`, `unnecessary_tool_call`, `invalid_arguments`, …), explicit denominators; evaluator version bump |
| 2.3 | `feature/rag-corpus-v2` | 8–9 | 20–50 document project corpus, 20–30 questions across the seven scenario types, versioned chunking policy; real-model baseline in `validation.md`; no judge |
| 2.4 | `feature/agentic-rag` | 10 | Read-only `search_documents` tool wrapping a retriever; search + calculator scenarios; separate retrieval, tool-sequence, citation, and final-answer metrics through the bounded loop |
| 2.5 | `feature/evaluation-runs-api` | 11–12 | `task_type` `tool_use` and `agentic_rag` reusing the resolver/preflight/dispatcher pattern; server-controlled builtin toolsets (MCP disabled over the API); `GET /runs/{id}/trace` reads the artifact (traces stay out of the DB); migration + PostgreSQL test |
| 2.6 | `feature/comparison-api` | 16 | Routes for MCQ `comparison.py` and `rag/comparison.py`; tool/agentic comparison |
| 2.7 | `feature/results-ui` | 17 | Read-only UI: run list, run detail, case detail, trace timeline, comparison |
| 2.8 | `release/v0.3.0` | 18 (partial) | Methodology, metrics glossary, demo script, known limitations; tag `v0.3.0` |

Optional after 2.3: `feature/rag-judge` — versioned judge (model/prompt/version
recorded), groundedness as a separately flagged metric.

### S3 — Production-oriented

Each item is its own branch/PR:

- Worker reliability (phase 13): `feature/worker-lease` →
  `feature/worker-heartbeat` → `feature/stale-recovery` →
  `feature/run-cancellation` → `feature/run-resume` → `feature/retry-policy` →
  `feature/concurrency-limits`.
- Security (phase 14): `feature/authn-authz`, `feature/endpoint-allowlist`
  (SSRF), `feature/tool-risk-policy`, `feature/rate-limits-audit`. Must land
  before any UI is exposed.
- vLLM (phase 15, if a GPU is available): `feature/vllm-runtime-validation`.
- Release (phase 18): `release/v1.0.0` with the full gate; tag `v1.0.0`.
