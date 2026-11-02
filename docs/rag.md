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

The initial milestone covers validated models and deterministic context
construction. Generation providers, answer evaluation, run artifacts,
and CLI execution are subsequent integration work.