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

## Current scope

This slice does not implement retrieval, chunking, corpus loading, embeddings,
artifact persistence, CLI execution, or RAG answer generation.