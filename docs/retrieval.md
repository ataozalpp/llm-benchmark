# Retrieval evaluation

The first retrieval slice evaluates ranked chunk IDs independently of
language models, MCP, and generation.

## Contracts

- `RetrievalEvaluationCase` contains a query ID and non-empty gold chunk IDs.
- `RankedRetrieval` contains a query ID and ordered, unique retrieved IDs.
- Query IDs must match.
- Empty retrieval results are valid.
- Queries without relevant chunks are outside this initial profile.

Gold labels are evaluator inputs and must not be passed to a retriever.

## Metrics

Evaluation uses only the first K retrieved IDs.

- Hit@K: 1 if any relevant chunk is retrieved, otherwise 0.
- Precision@K: relevant retrieved count divided by K.
- Recall@K: relevant retrieved count divided by the gold relevant count.
- Reciprocal rank@K: inverse rank of the first relevant result, otherwise 0.

Precision uses K even when fewer results are returned. Duplicate retrieved
IDs are rejected. `retrieved_count` records the number evaluated within K.

MRR requires aggregation across queries and is not computed by the
single-query evaluator.

## Keyword baseline

`KeywordRetriever` ranks pre-chunked text using the number of unique tokens shared
with the query.

- Tokenization uses Unicode word matching after `casefold`.
- Repeated tokens do not increase the score.
- Zero-score chunks are omitted.
- Equal scores are ordered by chunk ID.
- Results contain at most `top_k` unique chunk IDs.
- Empty corpora and queries without token overlap return empty results.
- Gold labels are never passed to the retriever.

This is a lexical baseline, not BM25 or semantic retrieval. It does not
perform stemming, language-specific normalization, chunking or embedding.

## Suite execution

The suite validates corpus IDs, scenario IDs, and gold membership before
creating the retriever. Retrievers receive queries and corpus chunks,
never gold labels.

Returned IDs must belong to the corpus, match the query, and respect top_k.
Contract violations raise errors rather than becoming zero-quality scores.

Invalid K, empty scenarios, duplicate IDs, and unknown gold IDs are rejected
before retriever creation. Invalid result types are rejected after retrieval.
Retriever execution exceptions propagate rather than becoming zero-score results.

## Aggregation

Summary metrics use macro averaging: each query has equal weight.
Queries with no retrieved results remain in the denominator.

MRR@K is the mean reciprocal rank@K across all evaluated queries.
All queries in a summary must use the same K.

## Offline fixture

`tests/fixtures/retrieval/keyword_smoke.json` contains four project-authored
chunks and four queries.

Three queries produce exact lexical matches. The delivery query is relevant
to the shipping chunk but has no token overlap, exposing the keyword
baseline's semantic limitation.

At K=2, expected summary metrics are:

- Hit rate: 0.75
- Mean precision: 0.375
- Mean recall: 0.75
- MRR: 0.75

The fixture validates implementation behaviour. It is not a representative
retrieval-quality benchmark or a public corpus-loading interface.

## Run identity

`profile.py` describes the versioned keyword baseline. `identity.py` hashes
corpus content and metadata, ordered scenarios, gold relevance sets, K,
the retriever profile, and evaluator/input-profile versions.

Corpus entries are sorted by chunk ID before hashing; scenario order is
preserved. Gold IDs are sorted because relevance-set order has no meaning.
Run IDs, timestamps, and output paths are not part of this content identity.
The identity does not capture the complete execution environment; runtime
provenance will need to be recorded separately when artifacts are implemented.

The current input profile is pre-chunked text, not an automatic chunking
algorithm. Identity generation does not execute a retriever or write artifacts.
