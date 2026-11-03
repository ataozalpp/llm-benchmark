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

The provider protocol is tested with stubs. A real HTTP generation adapter,
retrieval-to-generation suite orchestration, run artifacts, aggregation,
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

Add a non-streaming OpenAI-compatible HTTP adapter with injected transport
and tests for request construction, response envelopes, missing usage,
timeouts, and incomplete generation. Keep malformed HTTP response envelopes
distinct from malformed answer JSON. The former is a provider failure;
the latter is an invalid answer response.

Then connect retrieval and bounded context construction to the case runner,
validate corpus/scenario consistency, and add aggregation and run artifacts.
Only after those contracts are tested should real-model CLI runs be added.
No `rag-run` CLI is implemented yet.

Use a small QA fixture with explicit reference answers and evidence IDs
for the first integration run. SciFact retrieval relevance labels alone
are not reference answers for a QA benchmark.
