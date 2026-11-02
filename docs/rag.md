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
bounded context construction, strict answer parsing, and deterministic
answer and evidence metrics.

Generation providers, run artifacts, aggregation, and CLI execution
are subsequent integration work.

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

Build versioned generation messages from `RagGenerationRequest` only.
Keep instructions separate from retrieved data and explicitly treat context
text as evidence, not executable instructions. This is a prompt policy,
not a guarantee against prompt injection.

Then add a gold-free generation provider, a single-case runner, and tests
using a fake provider before introducing real-model CLI runs. Preserve
provider failures and parse failures as distinct outcomes rather than
turning them into valid abstentions. No `rag-run` CLI is implemented yet.

Use a small QA fixture with explicit reference answers and evidence IDs
for the first integration run. SciFact retrieval relevance labels alone
are not reference answers for a QA benchmark.
