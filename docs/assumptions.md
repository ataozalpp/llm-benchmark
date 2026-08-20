# Current V1.x assumptions

- Python support is `>=3.12,<3.13`.
- Configuration uses strict, immutable Pydantic models with
  `schema_version: 1`. Unknown fields and unsupported schema versions are
  errors.
- CLI configuration precedence is built-in defaults, YAML, then explicit
  `--set` overrides. Layered project/profile configuration and automatic schema
  migration are not implemented.
- The benchmark task is currently deterministic multiple-choice evaluation.
  Additional task families remain future work.
- The project-owned fixture data exists only for software validation.
  MockProvider scores, tokens, and latency are synthetic and must not be
  interpreted as LLM performance.
- MMLU-Pro requires a pinned revision. The recorded source is
  `TIGER-Lab/MMLU-Pro` revision
  `475d58ba0cc18a15fd5d4221f41919199e692331`, split `test`, citation
  MMLU-Pro arXiv:2406.01574, with dataset license metadata recorded as `MIT`.
- Dataset licensing is separate from project licensing. No project license has
  been selected yet.
- The MMLU-Pro loader normalizes upstream options and answers and uses
  `question_id` when available. Dataset contents are not copied into SQLite or
  redistributed in this repository.
- The CLI executes `RunConfig` directly and writes artifacts under
  `outputs/<run_id>/`. CLI behavior is independent from registry and Run API
  guardrails, including availability of a configured `full` profile.
- Registered Run API executions use active endpoint, model, and dataset
  records, write artifacts under `outputs/api/<run_id>/`, and persist run/sample
  records through repositories.
- The synchronous Run API accepts `smoke` and `poc`, rejects `full`, limits
  selected samples and explicit IDs to 100, and validates the exact selection
  before creating a run.
- Preflight and execution independently load the dataset. This is accepted
  V1.x technical debt and may be replaced by a prepared execution plan later.
- SQLite is the default local/development database. SQLAlchemy types and
  migrations are designed for future PostgreSQL portability, but PostgreSQL
  runtime support has not been validated.
- Endpoint, model, and dataset deletion is soft. Historical benchmark runs are
  preserved.
- Repository writes use explicit short transactions. Provider execution does
  not hold a database transaction. Sample-result bulk persistence is atomic,
  but filesystem artifacts and database records are not one cross-storage
  transaction.
- Actual API keys, bearer tokens, passwords, and secret values are not stored.
  Only credential environment-variable names may be registered. Providers
  resolve credential values at request time when configured.
- Provider calls have one attempt and no automatic retry. Logical request
  latency therefore equals the single attempt latency in V1.x.
- Runs are sequential. There is no worker, queue, concurrency control, rate
  limiting, resume, pagination, or authentication.
- Streaming is not implemented. Provider-reported native TTFT and throughput
  may be stored when available; missing telemetry remains null.
- The primary accuracy denominator includes all scheduled samples.
  `incorrect`, `unparseable`, and `request_failed` remain distinct outcomes.
- Missing token usage remains null. Cost calculation and pricing tables are not
  implemented.
- The parser uses each question's actual allowed labels. It does not use fuzzy
  matching, answer-text inference, log-probability scoring, self-consistency,
  or numeric tolerance.
- Reasoning content is telemetry only and is never scored unless the provider
  also places the answer in its standard final-message field.
- The API has no endpoint SSRF allowlist, dataset-root allowlist, or explicit
  Hugging Face network policy. It is not production-ready.
