# Architecture

## Design goals

The project separates benchmark semantics from transport, persistence, and HTTP
delivery. Dataset selection, prompt construction, parsing, scoring, request
configuration, and measurement metadata remain explicit and reproducible.

The implementation currently provides two execution paths:

- The CLI directly executes a validated YAML configuration and writes
  filesystem artifacts.
- The Run API resolves registered resources, preflights the exact workload,
  and persists a queued run. A separate worker atomically claims and executes
  queued work through the application service.

## Component responsibilities

| Component | Primary file | Responsibility |
| --- | --- | --- |
| CLI | `src/llm_benchmark/cli.py` | Load YAML, apply explicit overrides, call the runner, print a concise result |
| Configuration | `src/llm_benchmark/config.py` | Strict immutable Pydantic configuration and canonical serialization inputs |
| Dataset layer | `src/llm_benchmark/datasets.py` | Local, pinned Hugging Face, or verified uploaded loading; filtering, deterministic sampling, manifest construction |
| Dataset adapters | `src/llm_benchmark/dataset_adapters.py` | Validate pandas-backed CSV/JSONL input and normalize it to `DatasetExample` records |
| Dataset storage | `src/llm_benchmark/dataset_storage.py` | Bounded content-addressed storage, opaque-key parsing, checksum and physical-containment verification |
| Dataset ingestion | `src/llm_benchmark/dataset_ingestion.py` | Coordinate storage, adapter validation, compact registration metadata, and failure compensation |
| Prompt layer | `src/llm_benchmark/prompting.py` | Versioned multiple-choice prompt and template hash |
| Task adapter | `src/llm_benchmark/task_adapters.py` | Own task-specific prompt construction and normalized-response evaluation; currently implements multiple choice only |
| Provider layer | `src/llm_benchmark/providers.py` | Provider protocol, factory, Mock, LM Studio native, and OpenAI-compatible adapters |
| Parser | `src/llm_benchmark/parser.py` | Strict deterministic parsing against actual allowed labels |
| Runner | `src/llm_benchmark/runner.py` | Orchestrate provider execution, timing and telemetry, write artifacts, and aggregate metrics |
| Metrics | `src/llm_benchmark/metrics.py` | Quality, reliability, token, latency, and category aggregation |
| Artifact storage | `src/llm_benchmark/storage.py` | Append JSONL and atomically replace individual JSON files |
| Reproducibility | `src/llm_benchmark/reproducibility.py` | Canonical hashes and runtime/Git environment metadata |
| Database foundation | `src/llm_benchmark/db/engine.py`, `models.py` | SQLAlchemy engine/session, portable ORM schema, SQLite foreign keys |
| Repositories | `src/llm_benchmark/db/repositories.py` | Transactional persistence rules and immutable records |
| Registered config resolver | `src/llm_benchmark/run_resolution.py` | Build a safe RunConfig from active registry records |
| Run preflight | `src/llm_benchmark/run_preflight.py` | Apply bounded submission limits and validate the exact dataset selection |
| Application service | `src/llm_benchmark/application.py` | Run lifecycle, short transaction boundaries, runner invocation, result persistence |
| Worker | `src/llm_benchmark/worker.py` | Atomically claim one queued run, execute it outside the API request, and poll without busy waiting |
| Comparison service | `src/llm_benchmark/comparison.py` | Read two completed runs through reader protocols, validate strict comparability, and return immutable comparisons |
| FastAPI boundary | `src/llm_benchmark/api/` | Strict schemas, dependency wiring, thin CRUD and Run routes, safe HTTP errors |

## Overall architecture

```mermaid
flowchart LR
    subgraph EntryPoints[Entry points]
        CLI[CLI]
        RegistryAPI[Registry CRUD API]
        UploadAPI[Dataset upload API]
        RunAPI[Run submission API]
        Worker[Benchmark worker]
    end

    CLI --> Config[RunConfig]
    Config --> Runner[Benchmark runner]

    RegistryAPI --> Repositories[Repositories]
    UploadAPI --> Ingestion[DatasetIngestionService]
    Ingestion --> UploadedFiles[(Uploaded dataset files)]
    Ingestion --> Repositories
    RunAPI --> Resolver[RegisteredRunConfigResolver]
    Resolver --> Preflight[RunPreflightService]
    UploadedFiles --> Preflight
    Preflight --> Queued[(Queued benchmark runs)]
    Worker --> Queued
    Worker --> AppService[BenchmarkApplicationService]
    AppService --> Runner
    AppService --> Repositories

    Runner --> Dataset[Dataset loading and sampling]
    UploadedFiles --> Dataset
    Runner --> Providers[Provider adapters]
    Runner --> Artifacts[JSONL and JSON artifacts]
    Repositories --> DB[(SQLite default or PostgreSQL integration DB)]
```

## CLI execution

```text
YAML + --set overrides
-> load_config
-> immutable RunConfig
-> run_benchmark / execute_benchmark
-> dataset loading and sampling
-> prompt / provider / parser / evaluation
-> results.jsonl and JSON artifacts
```

The CLI does not use repositories, the application service, or Run API
guardrails. Its configured `full` profile remains available.

## Evaluation task boundary

The framework-independent `TaskAdapter` protocol separates task-specific
behavior from provider-neutral runner orchestration. The runner selects data,
calls providers, records timing and normalized telemetry, writes artifacts,
and aggregates metrics. The adapter builds the task prompt, supplies its
stable prompt-template hash, and evaluates the normalized provider response.

```mermaid
flowchart LR
    Example[DatasetExample] --> Adapter[TaskAdapter]
    Adapter --> Prompt[Task prompt]
    Prompt --> Provider[Provider adapter]
    Provider --> Response[ProviderResponse]
    Response --> Adapter
    Adapter --> Outcome[TaskEvaluationOutcome]
    Outcome --> Runner[Benchmark runner]
    Runner --> Results[Results, artifacts, and metrics]
```

`MultipleChoiceTaskAdapter` is the only implemented task adapter. It delegates
to the existing versioned multiple-choice prompt builder and strict parser,
then preserves the established `correct`, `incorrect`, `unparseable`, and
`request_failed` classification policy. Its prompt-template hash remains the
existing prompt hash and continues to participate in the run fingerprint.

This boundary does not add RAG, tool calling, MCP, agent loops, execution-trace
persistence, or additional task types. Existing CLI, Run API, worker,
providers, configuration hashes, artifacts, comparison, and database
persistence continue to use the same contracts.

## Registry API

The registry API exposes CRUD operations for endpoint, model, and dataset
registrations. Lists return active records; individual GET requests may return
soft-deleted records for historical administration. Routes use strict Pydantic
schemas and repositories rather than ORM objects.

Only a credential environment-variable name may be stored. Secret values are
rejected and resolved only by a provider at request time when configured.

The upload route is separate from generic dataset registry CRUD. It accepts a
bounded CSV or JSONL stream and server-controlled storage behavior, then
returns the resulting dataset registration. Complete uploaded rows are not
copied into the registry database.

## Uploaded dataset ingestion and execution

```mermaid
flowchart LR
    Upload[POST /api/v1/datasets/upload]
    Stream[Bounded binary stream]
    Store[LocalDatasetStorage]
    File[(Content-addressed file)]
    Adapter[CSV or JSONL adapter]
    Registry[(Dataset registration)]
    Run[POST /api/v1/runs]
    Verify[Key, checksum, content and containment verification]
    Preflight[RunPreflightService]
    Runner[Benchmark runner]
    Results[Artifacts and database results]

    Upload --> Stream --> Store
    Store --> File
    Store --> Adapter --> Registry
    Registry --> Run
    Run --> Verify
    File --> Verify
    Verify --> Adapter
    Adapter --> Preflight --> Runner --> Results
```

`LocalDatasetStorage.store()` reads positive bounded chunks, enforces the size
limit, computes SHA-256 while streaming, and finalizes a content-addressed file
on the same filesystem. Its application-facing key is
`upload://sha256/<digest>.csv` or `.jsonl`; it contains no physical path.

The ingestion service validates content through the selected adapter before
repository persistence. It registers `source_type=uploaded`, the opaque key as
`source_uri`, the adapter, `sha256:<digest>`, license when supplied, and compact
format/size/sample/category metadata. Newly created content is removed after a
validation or registration failure; reused content-addressed files remain.

For execution, `RegisteredRunConfigResolver` copies only portable provenance
into `DatasetConfig`. The application factory constructs one `DatasetLoader`
from its server-controlled `dataset_storage_root`; the same loader semantics
are injected into Run API preflight and the default runner executor.

Verification requires the key digest, registered checksum digest, and actual
file digest to agree. The physical candidate is resolved strictly and must
remain beneath the resolved root; escaping symlink or junction targets are
rejected. Verification creates no directory or file. CSV/JSONL adapters then
return the ordinary `list[DatasetExample]` boundary used by sampling.

CSV requires `sample_id`, `question`, `correct_answer`, `category`, and
contiguous ordered `option_A` through at most `option_J` columns. JSONL uses
the same scalar fields and a real `options` list. Both reject empty required
values, fewer than two options, unavailable answer labels, and duplicate IDs.

## Registered Run API sequence

```mermaid
sequenceDiagram
    participant Client
    participant API as Run API
    participant Resolver
    participant Preflight
    participant Worker
    participant Service as BenchmarkApplicationService
    participant DB
    participant Runner
    participant Provider

    Client->>API: POST /api/v1/runs
    API->>Resolver: registered IDs and safe request fields
    Resolver-->>API: validated RunConfig
    API->>Preflight: exact dataset selection
    Preflight-->>API: immutable selection result
    API->>Service: enqueue with exact selected count
    Service->>DB: create queued
    Service-->>API: immutable queued run
    API-->>Client: HTTP 201, status queued
    Client->>API: GET /api/v1/runs/{id}
    API-->>Client: current state
    Worker->>DB: atomically claim oldest queued run
    DB-->>Worker: immutable running run
    Worker->>Service: execute_claimed(running run)
    Service->>Runner: execute with no open DB transaction
    Runner->>Provider: one request per selected sample
    Provider-->>Runner: normalized ProviderResponse
    Runner-->>Service: results, summary, artifact directory
    Service->>DB: atomic sample add_many
    Service->>DB: transition completed
    Client->>API: GET /api/v1/runs/{id}
    API-->>Client: completed or failed state
```

If runner execution or required persistence fails after run creation, the
application service attempts to transition the run to `failed`. Unparseable and
request-failed samples remain benchmark outcomes when the pipeline itself
completes.

## Run API preflight

The default immutable `RunApiGuardrailPolicy` allows `smoke` and `poc`, rejects
`full`, limits exact selected samples to 100, and limits explicit sample IDs to
100. It performs cheap limits before dataset loading and exact count validation
after reusing the existing `sample_examples()` implementation.

Unknown IDs, unknown categories, mixed valid/invalid categories, and empty
selections are rejected before `BenchmarkApplicationService` is called. A
rejected preflight creates no provider request, database row, or artifact
directory.

Preflight and execution currently load the dataset independently. This avoids
changing runner and CLI contracts but leaves duplicate I/O and a theoretical
time-of-check/time-of-use boundary.

For uploaded datasets, each load independently resolves the same opaque key
through the same injected storage root and repeats checksum, containment, and
adapter validation. A preflight failure therefore occurs before run creation;
an execution-time failure after successful preflight follows the ordinary
failed-run lifecycle.

## Persistence relationships

```mermaid
erDiagram
    PROVIDER_ENDPOINTS ||--o{ MODELS : serves
    MODELS ||--o{ BENCHMARK_RUNS : executes
    DATASETS ||--o{ BENCHMARK_RUNS : evaluates
    BENCHMARK_RUNS ||--o{ SAMPLE_RESULTS : contains
```

SQLAlchemy 2.x uses SQLite by default for local/development persistence.
PostgreSQL 16 is the configured Docker integration runtime through Psycopg 3.
The schema uses generic JSON and portable enum/check-constraint storage, and
Alembic provides the initial schema migration. PostgreSQL migration,
repository, and API integration tests are opt-in and target only an explicitly
configured disposable database whose name ends in `_test`. The migration,
repository, and uploaded-dataset Run API paths have been executed successfully
against PostgreSQL.

Endpoints, models, and datasets use soft deletion. Foreign keys protect
historical runs; there is no destructive cascade from registry records to run
history. Sample bulk insertion is atomic.

## Run lifecycle

```mermaid
stateDiagram-v2
    [*] --> queued
    queued --> running
    queued --> failed
    queued --> cancelled
    running --> completed
    running --> failed
    running --> cancelled
    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

Completed, failed, and cancelled are terminal states. Submission persists the
exact selected sample count but no samples or artifacts. The worker repository
claim selects the oldest queued run by creation time and ID and conditionally
updates only a still-queued row to `running`. Each repository write uses a
short explicit transaction; the claim transaction closes before provider
execution. A completed run cannot be claimed again, including after a worker
restart.

## Provider architecture

`create_provider()` selects one adapter per model profile:

```mermaid
flowchart LR
    ModelConfig --> Factory[Provider factory]
    Factory --> Mock[MockProvider]
    Factory --> Native[LM Studio native]
    Factory --> Compatible[OpenAI-compatible]
    Mock --> Response[ProviderResponse]
    Native --> Response
    Compatible --> Response
    Response --> Parser[Parser and evaluator]
```

The LM Studio adapter calls native `POST /api/v1/chat`, sends `store=false`,
and parses only `type=message` output as the final answer. The generic adapter
calls `POST {base_url}/chat/completions` and scores only standard message
content. Native reasoning output is telemetry and is never scored directly.

## Artifacts and database boundaries

Runner artifacts are:

- `results.jsonl`
- `summary.json`
- `resolved_config.json`
- `dataset_manifest.json`
- `environment.json`

CLI runs write under `outputs/<run_id>/`. Registered Run API executions use
`outputs/api/<run_id>/` by default and additionally persist run/sample records.
Filesystem artifacts and database records are deliberately both retained, but
they do not form a single cross-storage transaction.

## Current security and operational boundaries

- The API has no authentication or authorization.
- Run submission is synchronous, but provider execution occurs in a separate
  database-backed worker process.
- The bounded design supports one sequential worker and one attempt per run.
  There is no retry, lease, heartbeat, stale-run recovery, cancellation,
  resume, concurrency control, or pagination.
- There is no endpoint SSRF allowlist or dataset-root allowlist.
- Uploaded content uses local runtime disk; there is no object storage or
  distributed locking.
- Generic registry CRUD remains a trusted administrative boundary and is not
  protected by authentication or authorization.
- There is no explicit Hugging Face network/cache policy at the API boundary.
- Raw model responses are exposed by the sample-result API.
- Public run responses do not expose persisted internal error messages.
- Actual credentials are never persisted; only environment-variable names may
  be registered.
- The API is not production-ready.

## Phase 5A strict run comparison

`BenchmarkComparisonService` is a framework-independent, read-only application
service. It depends on reader protocols for benchmark runs, sample results,
models, datasets, and endpoints. It does not import ORM models, write database
rows, read or write artifacts, or expose an HTTP route.

```mermaid
flowchart LR
    Runs[(benchmark_runs)]
    Samples[(sample_results)]
    Readers[Reader protocols]
    Comparison[BenchmarkComparisonService]
    Compatibility[Comparability assessment]
    Result[Immutable comparison result]

    Runs --> Readers
    Samples --> Readers
    Readers --> Comparison
    Comparison --> Compatibility
    Comparison --> Result
```

Both run IDs must be distinct, both runs must be `completed`, and each must
contain sample rows. Strict mode requires identical sample-ID populations.
Rows are indexed and sorted by `sample_id`, so alignment does not depend on
database row order. Duplicate IDs, empty runs, asymmetric populations, and
correct-answer or category mismatches raise typed comparison errors.

Comparability differences have three roles:

- Blocking: dataset identity/source, revision, split, task type, adapter,
  prompt-template hash, parser version, evaluator version, or scoring mode.
- Contextual: model identity is expected; endpoint and provider identity are
  reported as serving context.
- Conditional: reasoning policy/mode, temperature, sampling parameters,
  timeout, seed, fixed/provider-default output budgeting, and other supported
  generation settings limit causal attribution but do not discard the result.

Supported metrics are recomputed from persisted sample rows rather than copied
from summary aggregates. `summary_json` is used only for required provenance,
run wall time, and consistency warnings. Quality, format compliance,
reliability, token usage, and latency remain separate; there is no universal
composite score. Results include aggregate, per-category, and per-sample
comparisons and request/parse/evaluation transitions.

Null optional metrics remain null. Each optional metric carries
`available_count`, `missing_count`, and `complete`. Successful-request P50/P95
values use available non-null latency observations, while coverage uses the
total successful-request population. Incomplete coverage remains
`complete=false`, and no delta is emitted when either side is incomplete.

Comparison records use an explicit safe-field allowlist. They exclude raw
responses, provider and internal run error messages, endpoint URLs, credential
environment-variable names, dataset URIs and local paths, artifact directories,
and unrestricted registry metadata.

Phase 5A does not support partial-overlap comparison, attempt/retry/backoff
analysis, stop-reason or HTTP/provider-code comparison, final-output-token
comparison, cost comparison, or a FastAPI comparison endpoint. Some of those
features require additional persisted data or a separate product contract.

## Next architectural increments

- Expanded API usage and operational documentation
- Worker lease, heartbeat, stale-run recovery, and controlled retry design
- Authentication, authorization, and registry trust-boundary hardening
- Safe Comparison API, retry/resume, pricing, and additional task families
- Frontend only after stable backend contracts and operational boundaries
