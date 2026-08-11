# V1/MVP assumptions

- In V1, the result store consists of append-friendly JSONL and a derived JSON
  summary for each run. SQLite/PostgreSQL is deferred to a later increment.
- Configuration uses a single YAML file. Precedence is built-in Pydantic
  defaults, YAML, and explicit `--set` overrides. Layered configuration and
  migration are V2 concerns.
- `schema_version` accepts only `1`; unknown fields are errors.
- The fixture data is synthetic and is intended only for software validation.
  It must not be interpreted as a benchmark score.
- The MMLU-Pro `revision` field is required. The `smoke` profile selects 14
  samples, `poc` selects 10 samples per category when possible, and `full` uses
  the entire filtered test split.
- The MMLU-Pro loader normalizes upstream `options` and `answer` fields. It uses
  `question_id` as the identifier when available and otherwise uses the source
  row index.
- The mock provider uses deterministic per-sample scenarios and does not sleep.
  Its configured synthetic latency is returned in the provider response.
- V1 does not execute real retries: `attempt_count=1`, and logical latency is
  equal to attempt latency. Retry execution belongs to the second increment.
- The primary latency summary uses logical latency only for successful requests.
  Failed-request latency is reported separately in `failure_latency_*` fields.
- `incorrect_count` represents parseable wrong answers. `unparseable` and
  `request_failed` are counted separately. The primary accuracy denominator
  includes every scheduled sample.
- Cost is not calculated in V1, and `estimated_cost` remains null.
- Missing token usage remains null. Token totals include only known usage, and
  the number of records with missing usage is reported separately.
- The parser uses each question's actual `allowed_labels` set instead of an A-J
  assumption.
- Log-probability scoring, self-consistency, numeric tolerance, streaming/TTFT,
  real providers, databases, a web UI, and orchestration are later increments.
- The default output store is `outputs`, and generated runs are ignored by Git.
