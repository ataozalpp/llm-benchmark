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
available; see [Run CLI and artifacts](#run-cli-and-artifacts).

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
The prompt version is `rag-grounded-json-v1`. Prompt construction is
deterministic; this does not imply deterministic model generation.

Context content is treated as evidence rather than instructions. Separating
instructions and data is not a guarantee against prompt injection.
The prompt version should be recorded when run manifests are implemented.

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

Parsing failures must remain distinct from valid abstentions.
Future run summaries must report invalid responses explicitly rather
than silently excluding them from all-case denominators.

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
| `prompt_version` | `rag-grounded-json-v1`. |
| `generation_config_hash` | Provider kind, endpoint alias, model ID, temperature, and `max_tokens`. |
| `evaluation_config_hash` | `k` and `max_context_chars`. |
| `evaluator_version` | `rag-exact-v1`. |

The base URL and timeout are operational settings and are not part of the
identity or manifest. The model ID is the identifier reported to the endpoint;
it does not pin weights, quantization, or chat template.

### Artifacts

Each run writes `outputs/rag/<run_id>/`:

- `manifest.json`: identity, suite ID, retriever profile, generation settings,
  scenario IDs, Python version, timestamps, status, completed case count, and,
  for dense/hybrid, embedding runtime, load policy, and input statistics.
- `results.jsonl`: one row per processed case, appended and fsynced before the
  next query. Rows contain status, retrieved and context chunk IDs, the raw
  response with finish reason and token usage, the parsed answer, the
  evaluation, and retrieval/generation latency. Context text and reference
  answers are not written.
- `summary.json`: the `summarize_rag` output plus `model_load_ms`,
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
- `invalid_response` currently records the raw response but not the parser
  failure reason; inspect `response.text` in `results.jsonl`.

Real-model observations are recorded in `validation.md`.

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
