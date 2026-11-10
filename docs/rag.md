# RAG evaluation

The first RAG implementation uses a single retrieval step followed by
bounded context construction, answer generation, and evaluation.

## Data separation

Evaluation scenarios contain reference answers and gold evidence IDs.
Generation requests contain only the query and selected context.
Gold labels and answerability are never passed to the generation provider.

Answerability describes the corpus-level scenario, not whether the
retriever happened to find the evidence.

## Context policy

Context is rendered as a JSON array of chunk IDs and texts.
The budget counts Python Unicode code points in the rendered context,
including IDs, JSON syntax, and escaping.

Chunks are considered in retrieval order. Whole chunks that fit are
included; overflowing chunks are skipped. Later chunks may still fit.
No partial chunk text is inserted.

This is a context character budget, not a model token budget or a total
prompt budget.

## Current scope

The core supports validated scenario and answer models, deterministic
bounded context construction, strict answer parsing, deterministic
answer and evidence metrics, versioned generation messages, a gold-free
provider protocol, and prepared single-case execution.

The provider protocol and a non-streaming HTTP adapter are tested with
injected transports, including integration with the case runner.
Retrieval-to-generation suite orchestration is available in `rag/suite.py`.
Aggregation, run identity, local run artifacts, and the `rag-run` CLI are
available; see [Run CLI and artifacts](#run-cli-and-artifacts). Prompts are
selectable by version, schema-constrained decoding is an optional generation
setting, and two recorded runs can be compared with
`scripts/compare_rag_runs.py`; see [Comparing recorded runs](#comparing-recorded-runs).

## Suite orchestration

`run_rag_suite` validates the complete corpus and scenario collection before
work, retrieves each query, builds bounded context, and runs generation and
evaluation. It retains ranking, selected context, raw generation response,
available token usage, and case result in input order.

Expected provider errors, invalid answer JSON, and incomplete generation
produce case results and do not abort subsequent cases. Invalid retriever
outputs and unexpected programming errors propagate. Missing gold evidence
in a ranking is a measurable retrieval miss, not a configuration error.

An optional synchronous `on_case_completed(case)` callback is invoked for
each processed case, including unsuccessful outcomes, before the next query
is started. A caller can persist each case without waiting for the suite
to return. Callback failures propagate immediately to prevent further paid
calls after an output-storage failure. Without a callback, results remain
in memory. The suite does not itself create files or manifests, guarantee
crash durability, or implement resume. A test-only JSONL callback demonstrates
preservation of an earlier row when a later retrieval/generation call fails.

The current implementation is sequential, with no suite-level retry or
wall-time budget. The HTTP adapter already provides a transport timeout.

## Generation messages

`prompts.py` builds a system instruction and a JSON user message from
`RagGenerationRequest`. The user message contains only `question` and
`context`; each context item contains `chunk_id` and `text`.
Prompt construction is deterministic; this does not imply deterministic
model generation.

System prompts are registered by version in `PROMPTS`, and the config field
`prompt_version` selects one (default `rag-grounded-json-v1`):

| Version | Status |
| --- | --- |
| `rag-grounded-json-v1` | Default. Concise answer, citations, empty answer and citations for abstention. |
| `rag-grounded-json-v2` | Historical candidate evaluated on 2026-11-09 and not adopted. Kept verbatim so recorded v2 runs can be reproduced. |
| `rag-grounded-json-v3` | v2 without answer-like example values: shortest complete answer phrase, the exact abstention object, and no explanation when abstaining. Not yet evaluated on a real model. |

A registered text never changes; `tests/test_rag_prompts.py` pins the SHA-256
of every version, so a wording change requires a new version name. Unknown
versions are rejected by the config, the adapter, and the identity builder.

Context content is treated as evidence rather than instructions. Separating
instructions and data is not a guarantee against prompt injection.
The prompt version is part of the run identity and manifest.

A candidate `rag-grounded-json-v2` (explicit abstention object, shorter
answers with an inline example value) was evaluated and not adopted: the
small test model copied the example value into unrelated answers and began
answering without evidence. See `validation.md`. Current prompts must not
contain concrete answer-like example values (tests reject any digit in v1 and
v3), and prompt tests reject reference answers from the shipped fixtures for
every registered version.

### Schema-constrained decoding

`structured_output: json_schema` (OpenAI-compatible provider only) adds a
`response_format` of type `json_schema` with schema `rag-answer-schema-v1`
(`rag/answer_schema.py`). The schema has two branches: an abstention with an
empty answer, no citations and `abstained: true`, or a nonblank answer with
unique nonblank citation IDs and `abstained: false`. A server that enforces
it cannot produce `abstention_with_content` or `blank_answer`. Citation IDs
are not restricted to the supplied context, so citation validity still
measures the model.

Constrained decoding changes what is measured: the model can no longer
express an inconsistent answer, so it must choose between answering and
abstaining. Compare constrained and unconstrained runs rather than reading a
constrained run alone. Servers may ignore `response_format` without
reporting it; every response is still parsed with the strict parser, and the
manifest records the requested mode, not proof of enforcement. Inspect
`invalid_response_reasons` to confirm enforcement on a given endpoint.

## Provider contract and prepared case execution

`RagProvider.generate` receives only `RagGenerationRequest` and returns
`RagGenerationResponse`: raw text, a nonblank finish reason, and optional
input/output token counts. Missing usage remains null, not zero.
Answer JSON validation belongs to the parser, not the response container.

`run_rag_case` accepts a scenario, an already prepared context, and a
provider. Query/context IDs are checked before generation. It does not
perform retrieval or validate gold IDs against a complete corpus; those
checks belong to the future suite layer.

| Status | Meaning |
| --- | --- |
| `completed` | The answer was parsed and evaluated, regardless of correctness. |
| `invalid_response` | Generation ended with `stop`, but answer parsing failed. |
| `incomplete_response` | The finish reason was not `stop`; no evaluation is produced. |
| `provider_failed` | The provider raised `RagProviderError`; no response is available. |

Valid abstentions are evaluated as completed cases. Parse failures are
never converted into abstentions. Available raw responses are retained
for completed, invalid, and incomplete outcomes. Expected provider errors
currently retain only the status, not an error code or message.
Unexpected programming errors propagate. `RagCaseResult` is an internal
result container, not a validated artifact-loading schema.

## Answer contract

A generated response must be a JSON object containing exactly
`answer`, `citations`, and `abstained`.

An abstention requires an empty answer and an empty citation list.
Malformed JSON, duplicate fields, extra fields, and invalid field
types are rejected. Markdown fences and prose wrappers are not repaired.

Unknown citation IDs remain parseable and are evaluated against the
context actually supplied to the generator.

Responses are limited to 65,536 Python Unicode code points, including
surrounding whitespace. The exact limit is accepted. Nonstandard JSON
constants such as NaN and Infinity, trailing prose, and excessively nested
JSON are rejected with `RagParseError`. This parsing limit does not replace
the future provider's output-token limit.

Every `RagParseError` carries a stable `code`, recorded as the row's
`failure_code` for `invalid_response` cases and counted in the summary's
`invalid_response_reasons`. Checks run in the order below; the first
violated rule determines the code.

| Code | Meaning |
| --- | --- |
| `not_string`, `blank_response`, `too_long` | Raw response shape |
| `invalid_json`, `duplicate_field`, `invalid_constant` | Not strict JSON |
| `not_object` | JSON value is not an object |
| `wrong_fields` | Missing or extra fields |
| `invalid_field_type` | `answer` not a string, `citations` not an array, or `abstained` not a boolean |
| `invalid_citation_id` | A citation is not a nonblank string |
| `duplicate_citation` | A citation ID appears more than once |
| `abstention_with_content` | `abstained: true` with a nonblank answer or citations |
| `blank_answer` | `abstained: false` with a blank answer |

Codes are diagnostics; they do not change how the response is scored.

## Evaluation semantics

Answer exact match uses Unicode NFKC normalization, case folding,
and whitespace normalization. Punctuation is preserved.

Exact match is evaluated only for answerable scenarios. It is a
short-answer baseline and does not measure semantic equivalence.

Abstention correctness uses corpus-level answerability.

Context evidence recall measures the fraction of gold evidence IDs
present in the selected context.

Citation validity measures the fraction of cited IDs present in that
context. It is null when no citations are supplied.

Citation evidence recall measures the fraction of gold evidence IDs
both present in the context and cited by the answer.

Citation validity and evidence-ID overlap do not establish that the
answer is supported by the cited text. These metrics are not a
faithfulness judgment.

Citation IDs use exact, case-sensitive matching; answer normalization is
never applied to IDs. With no citations, citation validity is null and
citation evidence recall is zero for an answerable scenario. For an
unanswerable scenario, answer exact match and both evidence-recall metrics
are null. Abstention correctness remains defined.

A corpus-answerable query can have an empty or inadequate context. An
abstention may then be prudent for the generator, but it does not satisfy
the corpus-level end-to-end task. Context evidence recall helps distinguish
this case from a generation failure with adequate evidence.

Parsing failures remain distinct from valid abstentions. Run summaries
report invalid responses explicitly and keep them in all-case denominators.

## Run integration

### Summary semantics

`rag/reporting.py` requires exactly one case for each supplied scenario and
matches cases by query ID, independently of their order. Duplicate IDs,
missing/extra cases, mismatched component IDs, and inconsistent status/response
combinations are rejected. This is an internal result aggregator, not a loader
or full integrity validator for untrusted persisted artifacts.

Answer exact match uses all answerable cases as the denominator, including
provider failures, invalid responses, and incomplete generation. Unanswerable
abstention uses all unanswerable cases; only valid abstaining answers count.
An empty denominator produces null, not zero or NaN.

`answer_exact_match_given_full_context` restricts the exact-match denominator
to answerable cases whose context contains every gold evidence chunk (context
evidence recall 1.0). Generation failures on such cases (invalid, incomplete,
provider failure) stay in the denominator. Cases with partial or no evidence
are excluded, so a correct answer produced without evidence does not count.
The metric isolates generation quality when retrieval succeeded; it does not
replace end-to-end exact match. It is null when no case is eligible.

`invalid_response_reasons` counts `failure_code` values of invalid responses,
sorted by code. Every invalid response has a code and no other status has one;
an inconsistent result is rejected.

Context evidence recall is computed from context for every answerable case,
even if generation fails. Citation validity is a macro average over completed
cases with a defined validity value, not a pooled per-citation fraction.
Its measured and eligible counts accompany the mean; completed cases with no
citations are counted separately (including valid abstentions).

Token sums include observed usage from completed, invalid, and incomplete
responses. Missing usage is unmeasured; an observed zero is measured.
Each token field has its own measured count. Observed sums are not a claim
about total billed usage. Status counts include zero-count statuses.

### HTTP adapter

The HTTP adapter uses the existing `JsonTransport` and `UrllibJsonTransport`.
Supply a base URL such as `http://localhost:1234/v1`, not the full
`/chat/completions` endpoint. Invalid ports, whitespace/control characters,
URL credentials, query strings, and fragments are rejected before generation.
The adapter requests one non-streaming response with temperature zero and
a configurable `max_tokens` limit and transport timeout. There are no retries
or authentication options in this first version. Temperature zero does not
guarantee deterministic generation; the transport timeout is not a complete
run wall-time budget.

Malformed HTTP response envelopes become provider failures; malformed answer
JSON remains raw text until the case parser marks it as an invalid response.
Completed responses containing tool/function invocations are rejected.
Missing usage remains null. Tests use injected transports and do not establish
live endpoint compatibility or model quality.

## Run CLI and artifacts

```powershell
.\.venv\Scripts\python.exe -m llm_benchmark rag-run --config configs/rag_synthetic_smoke.yaml
```

The offline synthetic config needs no model or network service. The
`rag_hybrid_synthetic_smoke.yaml` config additionally needs the pinned
embedding model in the local Hugging Face cache. The two
`*_openai_compatible_smoke.yaml` configs are opt-in: replace the placeholder
`model_id` locally and do not commit the replacement.

### Suite input

`examples/rag/policy_qa_smoke.json` is a five-chunk, five-question QA fixture
with reference answers, gold chunk IDs, one corpus-unanswerable question, and
a keyword-strong distractor (`refund-history`). Suites are strict JSON with
the same bounded-loading rules as retrieval suites. Answerable scenarios
require references and evidence; unanswerable scenarios must declare neither.

`examples/rag/policy_qa_failure_cases.json` (`policy-qa-failure-cases-v1`)
targets known failure modes: nine chunks, five answerable and four
corpus-unanswerable questions. Its configs (`rag_failure_cases_*.yaml`) use
K=2 and `max_context_chars` 1000.

| Query | Failure mode | BM25 synthetic outcome |
| --- | --- | --- |
| `q-refund-conflict` | Current and previous policy disagree (14 vs 30 days); only the current chunk is gold | Both chunks in context |
| `q-international-total` | Answer needs two chunks (2 + 10 business days) | Both in context at K=2; recall 0.5 at K=1 |
| `q-warranty-exclusion` | Evidence sits in one chunk larger than the context budget | Retrieved first, dropped from context; recall 0 |
| `q-exchange-cost` | `exchange` vs `Exchanges` without stemming | BM25 miss; recall 0 |
| `q-support-injection` | A retrieved chunk tells the assistant to ignore the context | Both support chunks in context |
| `q-refund-cash`, `q-clothing-warranty`, `q-antarctica`, `q-support-phone` | Unanswerable but near-topic | Related distractors in context; `q-antarctica` gets an empty context |

With BM25 the full-context denominator is 3 of 5 answerable cases, so even an
ideal generator reaches at most 3/5 end-to-end exact match. The conflict and
injection cases test generation, not retrieval; their failures appear in
`answer_exact_match_given_full_context`. Citing the injected chunk still
yields citation validity 1.0: ID validity is not faithfulness.

```powershell
.\.venv\Scripts\python.exe -m llm_benchmark rag-run --config configs/rag_failure_cases_synthetic.yaml
```

### Configuration

Paths resolve relative to the config file. Duplicate YAML keys, unknown
fields, non-integer `k`, and `max_context_chars` below two are rejected.
The retriever may be `keyword`, `bm25`, `dense`, or `hybrid`; embedding and
fusion settings follow the same rules as `retrieval-run` and are validated by
the shared `validate_retriever_settings` function.

`provider: synthetic` must not include a model. `openai_compatible` requires
an OpenAI-compatible `model` block with `base_url` and `max_output_tokens`,
temperature zero, and no `credential_env_var`. These constraints mirror the
adapter: accepting other values would make the manifest misdescribe the
actual request.

`prompt_version` must name a registered prompt. `structured_output` is
`none` (default) or `json_schema`; the synthetic provider rejects
`json_schema` because it generates no model output.

The synthetic provider answers with the first context chunk text and cites
it, or abstains when context is empty. It reads only the generation request,
never evaluation labels, and reports no token usage. It validates the
pipeline; its scores are not a model measurement.

### Identity

The manifest records a run fingerprint over:

| Field | Content |
| --- | --- |
| `corpus_hash` | Chunk IDs, document IDs, and texts sorted by chunk ID; equal to the retrieval corpus hash for the same chunks. |
| `suite_hash` | Ordered query IDs, texts, answerability, references, and sorted gold IDs. |
| `retriever_profile_hash` | The complete retriever profile, including embedding model revision and fusion settings. |
| `context_policy_version` | `whole-chunk-json-chars-v1`. |
| `prompt_version` | The selected registered prompt (default `rag-grounded-json-v1`). |
| `generation_config_hash` | Provider kind, endpoint alias, model ID, temperature, and `max_tokens`; with `json_schema`, also `structured_output` and `answer_schema_version`. |
| `evaluation_config_hash` | `k` and `max_context_chars`. |
| `evaluator_version` | `rag-exact-v1`. |

The structured-output fields are added only when enabled, so unconstrained
runs keep the `generation_config_hash` they had before the option existed.
The base URL and timeout are operational settings and are not part of the
identity or manifest. The model ID is the identifier reported to the endpoint;
it does not pin weights, quantization, or chat template.

### Artifacts

Each run writes `outputs/rag/<run_id>/`:

- `manifest.json`: identity, suite ID, retriever profile, generation settings,
  scenario IDs, Python version, timestamps, status, completed case count, and,
  for dense/hybrid, embedding runtime, load policy, and input statistics.
- `results.jsonl`: one row per processed case, appended and fsynced before the
  next query. Rows contain status, `failure_code` (parse failure code for
  `invalid_response`, otherwise null), retrieved and context chunk IDs, the raw
  response with finish reason and token usage, the parsed answer, the
  evaluation, and retrieval/generation latency. Context text and reference
  answers are not written.
- `summary.json`: the `summarize_rag` output (including
  `invalid_response_reasons` and `answer_exact_match_given_full_context`) plus `model_load_ms`,
  `index_build_ms`, `retrieval_total_ms`, `generation_total_ms`, and
  `wall_time_ms`. It is written only for completed runs.

Expected case-level outcomes (provider failure, invalid answer JSON,
incomplete generation) are rows, not run failures. Unexpected failures,
including embedding load or indexing errors, set manifest status `failed`
with `rag_run_failed`; Ctrl+C records `rag_run_interrupted` and re-raises.
Rows written before a failure are preserved. The embedding runtime is written
to the manifest after model load and before indexing. An invalid suite or
provider configuration is rejected before an output directory is created.
The CLI exits with code 2 for invalid configuration and failed runs.

### Timing

`generation_total_ms` includes endpoint warm-up. The first case of a run can
therefore be much slower than later cases; compare per-row
`generation_latency_ms` before interpreting run-level generation time.
`model_load_ms` is the embedding model load and is zero for lexical retrievers.
It includes disk reads, so the first dense/hybrid run after a reboot or long
pause is slower than later runs (observed about 40 s versus 12 s).

### Interpreting results

- `context_evidence_recall` measures retrieval and context construction;
  `answer_exact_match` measures the end-to-end task. With a lexical miss,
  generation cannot be credited even if the model behaves well.
- BM25 returns only chunks with at least one matching query token, while
  dense and hybrid retrieval fill all `k` positions. Hybrid contexts can
  therefore contain more distractor text for the same `k`.
- Without stemming, BM25 does not match `Refunds` to `refund`; on the example
  fixture the distractor outranks the gold chunk for `q-refund`. Equal-weight
  fusion keeps the distractor first because it appears in both rankings.
- Small suites move rates in large steps: one of four answerable cases is
  25 percentage points.
- `answer_exact_match` minus `answer_exact_match_given_full_context`
  separates retrieval misses from generation errors. Read the full-context
  denominator first: it states how many cases generation was actually tested on.
- `invalid_response_reasons` names the contract violation; inspect
  `response.text` in `results.jsonl` for the raw output.
- Exact match does not credit verbose or shortened answers that are
  semantically correct (for example a full sentence containing the reference).

Real-model observations are recorded in `validation.md`.

### Comparing recorded runs

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_rag_runs `
  --suite examples/rag/policy_qa_failure_cases.json `
  --baseline outputs/rag/<baseline_run_id> `
  --candidate outputs/rag/<candidate_run_id> `
  --output outputs/rag/<candidate_run_id>/comparison.json
```

Both runs must be completed, with every scenario row present in order and
belonging to that run. The suite file must reproduce the runs' `corpus_hash`
and `suite_hash`. The runs must also agree on `evaluation_config_hash`,
`evaluator_version`, `context_policy_version`, `k`, and `max_context_chars`.
`prompt_version`, `generation_config_hash`, and `retriever_profile_hash` may
differ; the report lists the differing ones as `varied_fields` and states
when none differ. Nothing is retrieved or generated again.

The report groups query IDs for `answer_exact_match` (answerable cases) and
`unanswerable_abstention` (unanswerable cases) into `both_correct`,
`baseline_only`, `candidate_only`, and `both_incorrect`, using the same
denominators as the summary: a case without an evaluation counts as
incorrect. `delta` is candidate minus baseline. It also lists
`status_transitions`, `failure_code_changes`, `context_changed`, and
`answer_changed` (both parsed answers), plus the recorded summary values of
both runs. There is no combined score. The output file must not exist; exit
code 2 reports invalid or incompatible inputs.

For one variable at a time, change only the prompt, only the structured
output mode, or only the retriever between the two runs.

## Prepared-context smoke

`scripts/smoke_rag_openai_compatible.py` runs one refund-policy case with
manually supplied context. Its JSON report records the question, references,
context, model ID, endpoint, prompt version, settings, and case result.
Reference answers remain local evaluation/reporting data and are not sent
to the provider. Exit codes are 0 for passing answer/citation checks, 1 for
an unsuccessful smoke, and 2 for invalid provider configuration.

```powershell
.\.venv\Scripts\python.exe scripts/smoke_rag_openai_compatible.py --model "YOUR_MODEL_ID" --max-tokens 256
```

`--prompt-version` and `--structured-output` select the same options as the
run config and are recorded in the report.

This is not a retrieval benchmark. A completed case can fail the smoke:
for example, `30` omits the unit required by `30 days`. Do not add the
unit-free answer to the references merely to make the smoke pass. A normal
`stop` finish reason does not indicate output-token truncation.

The initial user-reported Qwen run and follow-up observations are recorded
in `validation.md`. They are individual observations, not general model
quality conclusions. Proceed to suite orchestration without tuning this
single example until it passes.

Use a small QA fixture with explicit reference answers and evidence IDs
for the first integration run. SciFact retrieval relevance labels alone
are not reference answers for a QA benchmark.

## API and worker runs

`POST /api/v1/rag-runs` builds a `RagRunConfig` from registrations only:

```text
rag_run_routes.py
  -> rag/run_resolution.py   registered endpoint/model/dataset -> RagRunConfig
                             (suite_path is the opaque upload key; output_dir
                             is outputs/api/rag; embedding from a named profile)
  -> rag/run_preflight.py    limits, checksum-verified suite load, embedding
                             dependency check; rejection creates nothing
  -> rag/application.py      enqueue: benchmark_runs row, task_type=rag, seed null
worker.py TaskTypeDispatcher
  -> rag/application.py      execute_claimed: re-validate config hash and
                             registrations, verify suite checksum,
                             execute_rag_benchmark outside any transaction,
                             rag_case_results in one transaction, final status
```

Clients cannot set the endpoint URL, model identifier, suite path, output
directory, embedding model or revision, or `local_files_only`. Endpoints with
a credential variable, LM Studio native endpoints, and models whose default
generation config sets a nonzero temperature are rejected (409). Invalid
setting combinations return 422 `invalid_rag_run_request`; exceeded limits
return 422 `run_guardrail_violation`; dense or hybrid without the
`embeddings` extra returns 409 `embeddings_unavailable`.

Embedding profiles live in `configs/api/embedding_profiles.yaml`
(`LLM_BENCHMARK_EMBEDDING_PROFILES` overrides the path). A profile must pin
a 40-character revision and may not set `local_files_only: false`. A missing
file offers no dense/hybrid profiles.

`rag_case_results` stores, per query, answerability, status, failure code,
retrieved and context chunk IDs, raw response and finish reason, parsed
answer, evaluation, token counts, and latencies. Context text and reference
answers are not stored. Artifacts are written as for the CLI under
`outputs/api/rag/<run_id>/`; database rows and files are not one
cross-storage transaction. A run whose suite file no longer matches its
checksum fails without results.

## Future work

- A larger local model with prompt v1, and the v1/v3 by constrained and
  unconstrained comparison on the shipped fixtures; results belong in
  `validation.md` before any default changes.
- Credential support in the RAG HTTP adapter (and then for API RAG runs).
- Reusing a loaded embedding model across worker runs.
- Standalone retrieval and local tool-use runs through the API on the same
  `task_type` infrastructure.
- Semantic answer matching; exact match remains the only answer metric.
