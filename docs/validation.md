# Validation record

This document summarizes completed software and local operational validation.
The real-model results are small diagnostic smoke runs, not statistically
sufficient model benchmarks.

> [!WARNING]
> `MockProvider` values are synthetic pipeline metrics. LM Studio values are
> real local-run telemetry for the stated model/runtime/configuration only and
> must not be generalized into a full MMLU-Pro capability claim.

## Offline fixture validation

The offline integration path uses eight project-owned synthetic
multiple-choice questions from `data/fixtures/mcq_fixture.jsonl` and two mock
model profiles from `configs/mock_smoke.yaml`.

The validated path is:

```text
YAML config
→ strict Pydantic validation
→ local JSONL loading
→ normalized DatasetExample objects
→ versioned prompt construction
→ deterministic MockProvider responses
→ deterministic answer parsing
→ evaluation classification
→ append-friendly results.jsonl
→ summary.json and reproducibility artifacts
```

The fixture data is owned by the project and exists only to exercise software
behavior. Its scores are not benchmark results for any model.

## Pinned MMLU-Pro MockProvider smoke validation

A controlled smoke validation was completed with the official
`TIGER-Lab/MMLU-Pro` Hugging Face dataset and the deterministic MockProvider.

| Field | Validated value |
| --- | --- |
| Dataset | `TIGER-Lab/MMLU-Pro` |
| Revision | `475d58ba0cc18a15fd5d4221f41919199e692331` |
| Split | `test` |
| Seed | `42` |
| Profile | `smoke` |
| Selected samples | `14` |
| Selected categories | `14` |
| Provider | `MockProvider` |
| Real endpoint calls | None |
| API keys | None |

The selected set covered these categories:

```text
biology, business, chemistry, computer science, economics, engineering,
health, history, law, math, other, philosophy, physics, psychology
```

The generated dataset manifest recorded the source, pinned revision, split,
seed, selected sample IDs, sample-to-category mapping, category set, sample
count, license, homepage, citation, and manifest hash.

The POC and full profiles were not run as part of this validation.

## Current forced-offline automated validation

The complete forced-offline suite most recently completed with:

```text
234 passed
0 failed
```

Coverage includes:

- Valid and invalid configuration behavior
- Unsupported schema versions and unknown fields
- Local fixture loading
- Network-free MMLU-Pro row normalization and deterministic sampling
- Prompt construction
- Deterministic MockProvider scenarios
- Parser normalization, ambiguity, and unparseable responses
- Request-failure and missing-token behavior
- Accuracy, answered accuracy, reliability, token, and percentile metrics
- JSONL persistence and summary generation
- Complete fixture-to-artifacts integration flow
- SQLAlchemy engine/session behavior and SQLite foreign-key enforcement
- Alembic upgrade, downgrade, and re-upgrade behavior
- Repository CRUD, uniqueness, soft deletion, transactions, and run lifecycle
- Framework-independent application-service execution and failure handling
- FastAPI registry CRUD routes and strict public schemas
- OpenAI-compatible request/response normalization with mocked transports
- Synchronous registered Run API execution
- Exact Run API preflight selection, limits, and safe public errors
- Framework-independent strict comparison of completed runs
- Deterministic sample alignment and typed incompatibility handling
- Comparison metric null/coverage behavior, including partial P50/P95 coverage

## Phase 5A comparison-service validation

The framework-independent `BenchmarkComparisonService` was validated with
in-memory immutable repository records and the complete forced-offline suite.
It accepts two distinct completed runs, aligns identical sample populations by
`sample_id` independently of row order, and rejects duplicate, empty,
asymmetric, or reference-inconsistent samples with typed errors.

Tests cover model differences as expected comparison context, provider and
endpoint differences as serving context, blocking evaluation-protocol
differences, and conditional reasoning, sampling, timeout, seed, and
output-budget differences. Quality, format-compliance, reliability, token, and
latency metrics are recomputed from persisted sample rows at aggregate,
category, and sample levels.

Null optional telemetry is preserved. Coverage records expose
`available_count`, `missing_count`, and `complete`. P50/P95 values are computed
from known successful-request latency values while coverage uses all successful
requests; incomplete coverage suppresses the absolute delta. The validated
snapshot was:

```text
Comparison tests: 21 passed
Comparison, repository, and application tests: 50 passed
Complete forced-offline suite: 234 passed, 0 failed, 0 warnings
```

The service performs no database or artifact writes and emits no raw response,
provider/internal error message, endpoint URL, credential reference, dataset
URI/path, artifact path, or unrestricted metadata. It does not produce a
composite score and is not yet exposed by a FastAPI comparison route.

Attempt/retry/backoff details, stop reasons, final-output tokens, HTTP/provider
codes, costs, and partial-overlap comparisons are outside Phase 5A because the
required persisted data or product contract is not currently available.

An earlier `57 passed, 0 failed` result was a historical development checkpoint
before the database, repository, application-service, API, and preflight
increments. It is not the current total.

## Manual LM Studio endpoint validation

The local runtime was manually validated before implementing the provider:

| Field | Validated value |
| --- | --- |
| Runtime | LM Studio |
| Model repository | `unsloth/Qwen3.5-0.8B-GGUF` |
| Quantization | `Q8_0` |
| Model ID | `qwen3.5-0.8b` |
| Server | `http://127.0.0.1:1234` |
| Authentication | Disabled |
| Local-network serving | Disabled |

`GET /v1/models` returned the expected model. OpenAI-compatible chat reached
the model, but it consumed the output budget as reasoning tokens and returned
empty message content. LM Studio native `POST /api/v1/chat` with
`reasoning="off"` returned `FINAL ANSWER: B`; therefore the first local POC uses
the native API and records reasoning mode explicitly.

The automated LM Studio tests use mocked HTTP responses and require neither LM
Studio nor a downloaded model. A real localhost fixture smoke remains opt-in.

### One-request fixture smoke result

After the offline suite passed, one explicitly approved localhost fixture
request was made with the tracked config. The native request succeeded and
reported:

- Input tokens: `75`
- Total output tokens: `64`
- Reasoning output tokens: `0`
- Tokens per second: approximately `16.09`
- Time to first token: approximately `944.09 ms`
- Logical request latency: approximately `5226.17 ms`

The message output contained an explanation but exhausted the configured
64-token output budget before emitting the required `FINAL ANSWER: <letter>`
line. The deterministic parser therefore marked the response `unparseable`.
This confirms native message extraction and telemetry capture, but it is not a
successful answer-format validation and is not evidence of model accuracy.

The corrective prompt contract now requires the marker on the first output
line and permits explanations to be omitted. The localhost fixture safety
margin is 128 output tokens. The parser remains strict: semantic option text
without `FINAL ANSWER: <letter>` is still unparseable. Stop reason is optional
because the native response may not provide it.

At a historical development checkpoint with 30 passing offline tests, one
approved localhost request was made with the corrected prompt and 128-token
limit. The request succeeded and returned five
output tokens, but its exact text was `FINAL ANSW: B`. Because the required
marker is exactly `FINAL ANSWER: B`, the parser correctly retained
`parse_status=no_answer_found`. LM Studio did not provide a stop reason, so the
recorded `stop_reason` is null. No heuristic alias for the truncated marker was
added.

Following the truncated-marker result, the canonical contract was narrowed
again: the entire response must now be exactly one uppercase option label from
the question's actual allowed labels. Exact `FINAL ANSWER: <label>` remains
accepted only for backward compatibility. Approximate markers, fuzzy matching,
semantic answer-text mapping, and arbitrary letters in prose remain rejected.

At a later historical development checkpoint with 34 passing offline tests,
one final approved localhost request was made with the standalone-label
contract. The exact native message was `B`; it was
parsed with `parse_status=normalized_label` and matched `correct_answer=B`.
The request used 89 input tokens and 2 output tokens, reported zero reasoning
tokens, approximately `30.59` tokens per second, `691.47 ms` time to first
token, and `995.72 ms` logical latency. LM Studio did not provide a stop reason,
so `stop_reason` remained null. No further request was made.

## Pinned MMLU-Pro LM Studio smoke result

A real-model smoke run was completed through the LM Studio native provider with
the cached official dataset and no external API calls.

| Field | Value |
| --- | --- |
| Dataset | `TIGER-Lab/MMLU-Pro` |
| Revision | `475d58ba0cc18a15fd5d4221f41919199e692331` |
| Split | `test` |
| Seed | `42` |
| Profile | `smoke` |
| Requests | `14`, sequential |
| Model | `qwen3.5-0.8b` |
| Reasoning | `off` |
| Temperature | `0` |
| Maximum output tokens | `128` |

Aggregate results:

| Metric | Value |
| --- | ---: |
| Total | 14 |
| Correct | 2 |
| Incorrect | 12 |
| Unparseable | 0 |
| Failed | 0 |
| Accuracy | 14.29% |
| Answered accuracy | 14.29% |
| Request success rate | 100% |
| Parse success rate | 100% |
| Format failure rate | 0% |
| Input tokens | 4,298 |
| Output tokens | 28 |
| Reasoning output tokens | 0 |
| Total tokens | 4,326 |
| Latency P50 | 1,496.08 ms |
| Latency P95 | 2,363.42 ms |
| Sum of logical-request durations | 22,121.40 ms |
| Run wall time | 22,136.70 ms |

Per-category outcomes:

| Category | Sample ID | Correct | Parsed | Result |
| --- | --- | --- | --- | --- |
| Biology | 3463 | C | A | Incorrect |
| Business | 215 | A | A | Correct |
| Chemistry | 3577 | F | C | Incorrect |
| Computer science | 10743 | H | A | Incorrect |
| Economics | 7111 | F | B | Incorrect |
| Engineering | 11537 | D | A | Incorrect |
| Health | 6232 | C | C | Correct |
| History | 4740 | C | A | Incorrect |
| Law | 1213 | E | A | Incorrect |
| Math | 8808 | C | A | Incorrect |
| Other | 5149 | B | C | Incorrect |
| Philosophy | 11083 | E | A | Incorrect |
| Physics | 9560 | I | A | Incorrect |
| Psychology | 2019 | I | B | Incorrect |

Selected sample IDs:

```text
10743, 11083, 11537, 1213, 2019, 215, 3463,
3577, 4740, 5149, 6232, 7111, 8808, 9560
```

The historical run identifier was `20260812T130136Z-ff5ca994`. Generated
artifacts were ignored by Git and are not asserted to remain present in a
current checkout. This 14-question result validates the recorded local
execution path and strict output-format compliance; it is not a statistically
sufficient MMLU-Pro score.

## Reasoning-on Gate 2 result

Run ID: `20260813T081214Z-0410f308`

The same 14 pinned sample IDs, categories, dataset revision, split, seed,
model, strict prompt/parser contract, and sequential policy were used with
`reasoning="on"` and a bounded 1,024-token output budget.

| Metric | Value |
| --- | ---: |
| Requests | 14 |
| HTTP request success | 14/14 |
| Correct / incorrect | 0 / 0 |
| Unparseable / failed | 14 / 0 |
| Parse success / format failure | 0% / 100% |
| Input tokens | 4,270 |
| Total output tokens | 14,336 |
| Reasoning output tokens | 14,336 |
| Derived final-output tokens | 0 |
| Latency P50 / P95 | 64,536.63 / 67,980.39 ms |
| Mean TTFT | 493.04 ms |
| Mean throughput | 16.09 tokens/s |
| Logical-duration sum | 906,158.60 ms |
| Run wall time | 906,178.97 ms |

Every response used exactly 1,024 output tokens as reasoning tokens and
produced an empty native final message. Reasoning was never scored as an
answer, so all results were correctly classified as unparseable.

## Single-sample 2,048-token calibration

Successful run ID: `20260813T125426Z-951e53ca`
Sample: `2019` (`psychology`, correct label `I`)

The request succeeded without retry or timeout. It reported 154 input tokens,
2,048 total-output tokens, 2,048 reasoning tokens, zero derived final-output
tokens, 2,202 total tokens, 122,546.88 ms latency, 795.47 ms TTFT, and 16.85
tokens/s. The final message was empty and the strict parser returned
`unparseable`.

An earlier attempt (`20260813T105947Z-2005f467`) ended at the runtime/provider
boundary with `server_error`. Local LM Studio Developer Logs later reported a
sanitized `internal_error` with code `unknown`; it did not establish that
2,048 tokens had been generated. The successful calibration supersedes that
attempt for output-budget analysis, while both artifacts remain separate.

## Partial native-supported sampling calibration

Run ID: `20260813T132112Z-9f0ac78f`
Sample: `2019`

The profile used `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`,
and `repeat_penalty=1.0`. Native `presence_penalty` support was unverified and
the field was omitted. The result again reported 154 input tokens, 2,048
total-output tokens, 2,048 reasoning tokens, zero final-output tokens, and an
empty final message. Latency was 119,960.24 ms, TTFT was 176.64 ms, and
throughput was 17.15 tokens/s.

Sanitized inspection through local LM Studio Developer Logs showed strong
repetitive, non-convergent deliberation until output-budget exhaustion. This
supports the conclusion of probable non-terminating or excessively long
reasoning under the tested model/runtime/prompt/sampling configuration. It
does not prove a literal infinite loop. Raw reasoning text is not stored in
the repository, documentation, or tracked artifacts.

## Provider-default context-bounded calibration

Run ID: `20260814T082432Z-39883af2`

This single-sample calibration reused sample `2019`, the pinned dataset
revision, reasoning-on mode, and the partial native-supported sampling profile.
`max_output_tokens` and `presence_penalty` were absent from the native payload.
The result recorded `output_budget_provenance=provider_default` and used a
660-second transport safety timeout without retry.

LM Studio reported an 8,192-token context for the loaded instance and a
262,144-token maximum in model metadata. The loaded-instance value is the
relevant runtime configuration, but neither value establishes the exact output
allowance because input, template, special tokens, and runtime overhead share
the context.

| Metric | Bounded 1,024 | Bounded 2,048 | Provider default |
| --- | ---: | ---: | ---: |
| Input tokens | 154 | 154 | 154 |
| Reasoning tokens | 1,024 | 2,048 | 3,591 |
| Derived final-output tokens | 0 | 0 | 4 |
| Total-output tokens | 1,024 | 2,048 | 3,595 |
| Total tokens | 1,178 | 2,202 | 3,749 |
| Final message | Empty | Empty | `B` |
| Parsed/evaluation status | Unparseable | Unparseable | `B` / incorrect |
| Latency | 62,444.60 ms | 122,546.88 ms | 209,883.45 ms |
| TTFT | 808.346 ms | 795.465 ms | 215.925 ms |
| Throughput | 16.7643 tokens/s | 16.8524 tokens/s | 17.1926 tokens/s |

The provider-default request succeeded, did not time out, and reached a final
message after exceeding both earlier bounded budgets. The final label was
format-compliant but differed from correct label `I`. No explicit configured
output limit ended the request, but the null stop reason prevents a stronger
termination claim. The request did not consume the full loaded context. This
single sample neither proves that provider-default generation always
terminates nor that reasoning improves accuracy; it is an operational
calibration rather than a benchmark score.

## Three-sample provider-default operational calibration

Run ID: `20260814T104711Z-3e7b92e5`

Fixed samples `215`, `6232`, and `2019` were executed sequentially with one
attempt each, a 660-second timeout, and provider-default output budgeting.
`max_output_tokens` and `presence_penalty` were absent. All three requests
succeeded, parsed successfully, and reached a native final message. One answer
was correct and two were incorrect.

| Sample | Category | Expected / parsed | Result | Input | Reasoning | Final output | Total output | Total tokens | Latency | TTFT | Throughput |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `215` | business | `A` / `A` | correct | 285 | 6,804 | 4 | 6,808 | 7,093 | 439,171.65 ms | 2,284.082 ms | 15.6165 tok/s |
| `6232` | health | `C` / `J` | incorrect | 328 | 5,116 | 4 | 5,120 | 5,448 | 329,634.88 ms | 2,100.948 ms | 15.6703 tok/s |
| `2019` | psychology | `I` / `B` | incorrect | 154 | 2,203 | 4 | 2,207 | 2,361 | 135,853.82 ms | 1,003.439 ms | 16.4733 tok/s |

Aggregate operational measurements:

- Final messages reached: 3/3 (100%)
- Request success: 3/3 (100%)
- Parse success: 3/3 (100%)
- Calibration-only correct: 1/3 (33.33%)
- Total tokens: 14,902
- Total reasoning tokens: 14,123
- Logical-duration sum: 904,660.36 ms
- Run wall time: 904,667.38 ms
- Latency P50/P95: 329,634.88 / 428,217.97 ms

No request timed out or showed explicit output-limit exhaustion, and all final
messages were format-compliant. Native stop reasons were not provided. These
three selected samples characterize operational final-message reliability and
cost only; they are not an MMLU-Pro accuracy estimate.

### Paired reasoning-off comparison

The same three samples were extracted from reasoning-off run
`20260812T130136Z-ff5ca994` for a paired descriptive comparison.

| Aggregate for the same samples | Reasoning off | Context bounded |
| --- | ---: | ---: |
| Correct / incorrect | 2 / 1 | 1 / 2 |
| Unparseable / failed | 0 / 0 | 0 / 0 |
| Final-message rate | 100% | 100% |
| Request / parse success | 100% / 100% | 100% / 100% |
| Total tokens | 779 | 14,902 |
| Reasoning tokens | 0 | 14,123 |
| Logical-duration sum | 4,053.07 ms | 904,660.36 ms |
| Latency P50/P95 | 1,403.38 / 1,516.33 ms | 329,634.88 / 428,217.97 ms |

This comparison does not isolate reasoning. The reasoning-off run used
`reasoning=off`, `temperature=0`, and a fixed 128-token output budget. The
context-bounded run used `reasoning=on`, `temperature=1.0`, `top_p=0.95`,
`top_k=20`, `min_p=0.0`, `repeat_penalty=1.0`, and provider-default output
budgeting. The runs also used different workload sizes, Git revisions, dates,
and runtime states. Therefore token, latency, and quality differences cannot be
causally attributed solely to reasoning, and no statistical significance can
be inferred from three samples.

## OpenAI-compatible localhost interoperability validation

Exactly one explicitly approved synthetic fixture request validated the generic
`OpenAICompatibleProvider` against LM Studio. The base URL was
`http://127.0.0.1:1234/v1`, the API model identifier was `qwen3.5-0.8b`, and
the provider called `POST /v1/chat/completions`. No credential or
`Authorization` header was used.

The fixture selected only `q01` (`math`, correct label `B`). The standard final
message was `B`, the strict parser returned `parsed_answer=B`, and the request
completed successfully with `stop_reason=stop`.

| Metric | Value |
| --- | ---: |
| Requests | 1 |
| Input tokens | 90 |
| Total-output tokens | 326 |
| Reasoning tokens | 322 |
| Derived final-output tokens | 4 |
| Total tokens | 416 |
| Latency | 21,464.63 ms |
| TTFT | null, not estimated |
| Throughput | null, not estimated |

The request omitted `max_tokens`, so the result recorded
`output_budget_provenance=provider_default`; this does not mean unlimited
generation. No reasoning mode was transmitted. `reasoning_mode` remained null,
while the endpoint independently reported reasoning-token usage. The behavior
is therefore provider-managed or unverified and must not be labelled
reasoning-on or reasoning-off.

This successful result validates interoperability, standard response mapping,
strict parsing, and telemetry preservation for one fixture request. It is not
a model-quality result, an MMLU-Pro score, or evidence about general endpoint
reliability. No raw reasoning content is stored in this documentation.

## Interpretation boundary

The validation proves that the current pipeline can load and sample data,
produce deterministic provider responses, parse and classify those responses,
calculate metrics, and persist reproducibility artifacts.

It does not establish:

- Full MMLU-Pro performance for any real model
- General provider latency or reliability beyond the recorded local runs
- Monetary cost
- Compatibility beyond the individually recorded endpoint validations
- Statistical validity of a 14-sample score

Real provider results must be recorded in separate runs with explicit provider,
endpoint alias, requested and returned model IDs, request parameters, and
measurement protocol metadata.

## Licensing note

No project license has been selected yet. MMLU-Pro attribution and its dataset
license metadata remain recorded separately from the project license.
