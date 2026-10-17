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
creating the retriever. Retrievers recieve queries and corpus chunks,
never gold labels.

Returned IDs must belong to the corpus, match the query, and respect top_k.
Contract violations raise errors rather than becoming zero-quality scores.

## Aggregation

Summary metrics use macro averaging: each query has equal weight.
Queries with no retrieved results remain in the denominator.

MRR@K is the mean reciprocal rank@K across all evaluated queries.
All queries in a summary must use the same K.