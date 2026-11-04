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
Retrieval-to-generation suite orchestration, run artifacts, aggregation,
and CLI execution are subsequent integration work.

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

## Next integration milestone

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

Next, perform a small prepared-context real-model smoke and record the model,
endpoint, settings, raw response, finish reason, and evaluation. This smoke
does not measure retrieval quality or create full benchmark run artifacts.

Then connect retrieval and bounded context construction to the case runner,
validate corpus/scenario consistency, and add aggregation and run artifacts.
Only after those contracts are tested should real-model CLI runs be added.
No `rag-run` CLI is implemented yet.

### Prepared-context smoke

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
