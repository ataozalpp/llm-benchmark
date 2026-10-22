# Retrieval evaluation

The first retrieval slice evaluates ranked chunk IDs independently of
language models, MCP, and generation.

## Diagnostic review of a completed keyword run

The local review command checks a saved run against its prepared suite and
selects a reproducible diagnostic sample. From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/review_retrieval_run.py --suite data/cache/retrieval/scifact-test-v1/suite.json --run-dir outputs/retrieval/4188b7e1e9f246b8b0bb2a908060d4a1 --verify-retrieval
```

Use the run directory for your own experiment when it differs from this example.
The default output is `review.json` inside that directory. Use `--output` with
a new filename to preserve an earlier review; its parent directory must exist.
Existing files are never replaced.

The script validates completed status, artifact schema, content identity, query
coverage/order, result IDs, and recomputed metrics. Duplicate JSON keys,
non-finite constants, invalid objects and boolean metric values are rejected.
Each manifest/results input file has a 64 MiB read limit.

`--verify-retrieval` builds the current keyword index once and checks every saved
ranking against current retrieval output. The report records `matched` or
`not_requested`; a mismatch fails the command. Replay never modifies original
rankings, metrics or timings. Matching outputs do not prove source-code identity,
and a mismatch alone does not establish its cause.

The selection uses `retrieval-review-v1`, sorting query IDs by a versioned
SHA-256 key within three groups:

- Up to 5 queries whose first relevant result is at rank 1.
- Up to 5 queries whose first relevant result is at rank 2 through K.
- Up to 10 queries with no relevant result in the top K.

Small groups are not padded. The sample is diagnostic, not representative;
do not derive overall quality scores from these selected cases. Do not tune
query-specific rules on the reviewed test split.

`recall_diagnostics` separately lists all queries with some, but not all, gold
items retrieved. It is computed over the entire run, including rank-1 hits.
These queries are not guaranteed to occur in the selected sample. Each selected
case includes full query, gold and retrieved text for local inspection. Imported
corpus licensing and attribution still apply to exported text.

Reports are serialized before writing and limited to 16 MiB of UTF-8 output.
This is an output bound, not a memory bound: the report is built in memory.
A same-directory temporary file is flushed and fsynced, then published using
an atomic hard-link creation that fails if the target exists. The filesystem
must support hard links; unsupported storage fails without a partial target.
Temporary files are cleaned on normal exceptions. Forced process termination
may leave a temporary file; power-loss recovery is not guaranteed.

Invalid input, replay mismatch and storage errors produce exit code 2 with a
sanitized message. Core functions retain detailed exceptions for debugging.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_review.py tests/test_retrieval_run.py
```

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
The identity does not capture the complete execution environment. The run
manifest records the Python version separately; timings are observations, not
part of content identity.

The current input profile is pre-chunked text, not an automatic chunking
algorithm. Identity generation does not execute a retriever or write artifacts.

## Run the local keyword benchmark

From the repository root, with the project installed in the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m llm_benchmark retrieval-run --config configs/retrieval_keyword_smoke.yaml
```

No model server, model download, API key, or MCP process is needed. The command
prints JSON with `run_id`, `output_dir`, and `overall`. The example produces the
four-query metrics above. These are correctness checks, not model-quality claims.

The config is:

```yaml
schema_version: 1
suite_path: ../examples/retrieval/keyword_smoke.json
retriever: keyword
k: 2
output_dir: ../outputs/retrieval
```

Both paths resolve relative to the YAML file, not the terminal working directory.
Absolute paths are also supported. If omitted, `output_dir` defaults to
`outputs/retrieval` relative to that YAML file. Configs are limited to 64 KiB;
unknown fields, duplicate keys, unsupported versions/retrievers, blank paths,
and non-positive or non-integer K (including booleans) are rejected.

`examples/retrieval/keyword_smoke.json` is the runnable input. Unlike the older
test-only fixture, it has no `k` field: K belongs exclusively to the run config.
The input contract is strict UTF-8 JSON, at most 32 MiB:

```json
{
  "schema_version": 1,
  "suite_id": "my-suite-v1",
  "chunks": [
    {"chunk_id": "refund", "document_id": "policy", "text": "refund period thirty days"}
  ],
  "scenarios": [
    {"query_id": "q-refund", "text": "refund period", "relevant_chunk_ids": ["refund"]}
  ]
}
```

Both lists must be nonempty. Chunk IDs and query IDs must be unique. Gold IDs
must be nonempty, unique within each query, and present in the corpus. Unknown
fields, duplicate JSON keys, invalid text, and unsupported versions fail before
run-directory creation. Gold labels are used for evaluation, never retrieval.
The loader is for controlled local pre-chunked inputs, not a public upload API.

## Artifacts and failure semantics

Every invocation creates a fresh UUID directory under the configured output:

```text
outputs/retrieval/<run_id>/
  manifest.json
  results.jsonl
  summary.json
```

- `manifest.json`: artifact schema version 1, content hashes/fingerprint,
  retriever profile, Python version, K, ordered query IDs, timestamps, status,
  and completed-case count. The human-readable suite ID is metadata; changing
  only that label does not change the fingerprint.
- `results.jsonl`: one line per successfully evaluated query, in input order;
  query/run IDs, retrieved chunk IDs, metrics, and retrieval latency. Empty
  rankings are valid results and remain in the aggregate denominator.
- `summary.json`: macro metrics plus run ID and measured timings; written after
  every query succeeds. Timings vary between identical runs.

`index_build_ms` measures keyword index construction. Per-query latency and
`retrieval_total_ms`/`retrieval_mean_ms` measure retrieval calls only, excluding
evaluation and artifact writes. `wall_time_ms` starts before input loading and
includes validation, identity, indexing, evaluation, and incremental result
writes; it ends before writing the summary and final manifest. It is not total
CLI process startup/shutdown time.

Metadata files use temporary-file replacement. Results are appended, flushed,
and fsynced after each completed query. An execution/storage failure raises a
sanitized error and attempts to mark the manifest `failed`; earlier result
lines remain available. Invalid input is a preflight error with no run artifacts.
The CLI exits with code 2 for these errors. Failed runs must not be interpreted
as successful zero-quality runs or compared as complete benchmarks.

Consumers must require `manifest.status == "completed"`; a summary might exist
if writing the final manifest failed. Disk failures can also prevent recording
failure status. A killed process may leave `running`, a temporary file, or a
partial final JSONL line. This increment does not implement recovery, resume,
locking across processes, or power-loss durability guarantees. Existing run
directories are never reused. Artifacts omit raw corpus/query text and raw
exception details, but include user-supplied IDs; use suitable IDs for your data.

## Implementation boundaries and validation

- `run_config.py`: YAML validation and path resolution.
- `dataset.py`: bounded JSON loading and domain-object construction.
- `run_service.py`: identity, lifecycle, timings, and orchestration.
- `artifacts.py`: local persistence, without a dependency on tool-use artifacts.
- `suite.py`: evaluation contracts and an optional completed-case callback;
  existing callers remain compatible. Callback errors abort execution.
- `cli.py`: the `retrieval-run` command; existing commands remain available.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_run.py tests/test_retrieval_identity.py tests/test_retrieval_suite.py
.\.venv\Scripts\python.exe -m ruff check .
```

The small smoke corpus remains an implementation correctness check.
The BEIR SciFact import provides an external retrieval baseline using
the full source corpus and the selected relevance-labelled split.

See `validation.md` for the recorded keyword baseline. Dense retrieval,
automatic chunking, and RAG generation evaluation remain separate
future increments.

## BEIR SciFact import

The project can prepare a local BEIR-formatted SciFact distribution
for the existing `retrieval-run` command.

### Source and attribution

Dataset distribution and published archive checksum:

https://github.com/beir-cellar/beir/wiki/Datasets-available

Original dataset and licensing information:

- https://github.com/allenai/scifact
- https://github.com/allenai/scifact/blob/master/LICENSE.md
- https://huggingface.co/datasets/BeIR/scifact

### Local source layout

Extract the BEIR distribution under:

```text
data/downloads/beir/scifact/
  corpus.jsonl
  queries.jsonl
  qrels/
    test.tsv
```

Downloaded files and prepared datasets remain in Git-ignored directories.

### Prepare the dataset

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_beir_retrieval.py --source-dir data/downloads/beir/scifact --split test --output-dir data/cache/retrieval/scifact-test-v1
```

The command creates:

```text
suite.json
provenance.json
```

Existing output directories are rejected to prevent accidental
replacement.

Source files are limited to 64 MiB each. The prepared suite must
fit the retrieval loader's 32 MiB limit.

### Conversion contract

- Each source document becomes one chunk.
- Chunk IDs use `beir-doc:<source-document-id>`.
- Titles and body text are stripped and joined with a blank line.
- Every corpus document remains in the search space.
- Evaluation queries come from the selected qrels file.
- Positive integer relevance scores become binary gold membership.
- Zero scores are not gold.
- Duplicate IDs/judgements, unknown references, and evaluated queries
  without positive gold are rejected.

This representation preserves document-level retrieval through the
existing chunk contract. It does not perform automatic chunking or
introduce passage-level relevance labels.

### Provenance and file formatting

The provenance sidecar records source SHA-256 hashes, the importer
version, split, conversion policies, selected query IDs, counts,
and the prepared suite SHA-256.

The sidecar is not automatically embedded in run manifests. Preserve
it alongside the prepared suite.

`suite.json` uses compact JSON serialization. It is generated input,
not a hand-edited document. Formatting it in place changes its file
hash and breaks the recorded byte-level provenance match.

Use a separate pretty-printed copy when inspecting the data.

Preparation validates the suite before creating its output directory.
Storage failures can still leave incomplete output. Only use output
from a successful preparation command.

### Run the keyword baseline

```powershell
.\.venv\Scripts\python.exe -m llm_benchmark retrieval-run --config configs/retrieval_keyword_scifact.yaml
```

The example uses K=10 and the full imported corpus.

The recorded baseline is documented in
[Validation](validation.md#scifact-keyword-baseline-2026-10-20).

This experiment evaluates English scientific-claim retrieval.
It does not measure RAG answer correctness, citation support, or
general conversational quality.
