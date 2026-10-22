"""Import the local BEIR SciFact distribution into retrieval schema v1."""

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from . import dataset

IMPORTER_VERSION = "beir-scifact-v1"
MAX_SOURCE_BYTES = 64 * 1024 * 1024

SOURCE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)

LICENSE_REFERENCES = [
    "https://github.com/allenai/scifact/blob/master/LICENSE.md",
    "https://huggingface.co/datasets/BeIR/scifact",
]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_source(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)

    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("BEIR source file exceeds the size limit.")

    return raw


def _unique_object(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in BEIR input.")
        result[key] = value

    return result


def _reject_constant(value):
    raise ValueError("Non finite JSON value in BEIR input.")


def _validate_text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("BEIR text fields must be strings.")

    if "\x00" in value:
        raise ValueError("BEIR text contains a null character.")

    value.encode("utf-8")

    if not allow_empty and not value.strip():
        raise ValueError("BEIR text fields must not be blank.")

    return value


def _read_records(raw: bytes) -> dict[str, dict]:
    records = {}

    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            raise ValueError("Blank JSONL records are not supported.")

        record = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )

        if not isinstance(record, dict):
            raise ValueError("BEIR JSONL records must be objects.")

        record_id = _validate_text(record.get("_id"))

        if record_id in records:
            raise ValueError("Duplicate BEIR record ID.")

        records[record_id] = record

    if not records:
        raise ValueError("BEIR input must not be empty.")

    return records


def _read_qrels(
    raw: bytes,
    *,
    corpus_ids: set[str],
    query_ids: set[str],
) -> dict[str, set[str]]:
    reader = csv.reader(
        io.StringIO(raw.decode("utf-8")),
        delimiter="\t",
    )

    if next(reader, None) != ["query-id", "corpus-id", "score"]:
        raise ValueError("Unexpected BEIR qrels header.")

    seen_pairs = set()
    gold = {}

    for row in reader:
        if len(row) != 3:
            raise ValueError("BEIR qrels rows must have three columns.")

        query_id, document_id, score_text = row

        if query_id not in query_ids:
            raise ValueError("Qrels reference an unknown query.")

        if document_id not in corpus_ids:
            raise ValueError("Qrels reference an unknown document.")

        pair = (query_id, document_id)

        if pair in seen_pairs:
            raise ValueError("Duplicate query-document judgement.")

        seen_pairs.add(pair)

        try:
            score = int(score_text)
        except ValueError:
            raise ValueError("Qrels scores must be integers.") from None

        if score < 0:
            raise ValueError("Negative relevance scores are unsupported.")

        relevant_ids = gold.setdefault(query_id, set())

        if score > 0:
            relevant_ids.add(document_id)

    if not gold:
        raise ValueError("No evaluation queries were found.")

    if any(not relevant_ids for relevant_ids in gold.values()):
        raise ValueError(
            "Each evaluated query must have a positive relevance judgement."
        )

    return gold


def _chunk_id(document_id: str) -> str:
    return f"beir-doc:{document_id}"


def prepare_scifact(
    source_dir: Path,
    output_dir: Path,
    *,
    split: str = "test",
) -> dict[str, object]:
    """Validate local inputs and create a new suite/provenance directory."""

    if split not in {"train", "test"}:
        raise ValueError("Supported SciFact splits are train and test.")

    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new one.")

    sources = {
        "corpus.jsonl": _read_source(source_dir / "corpus.jsonl"),
        "queries.jsonl": _read_source(source_dir / "queries.jsonl"),
        f"qrels/{split}.tsv": _read_source(source_dir / "qrels" / f"{split}.tsv"),
    }

    corpus = _read_records(sources["corpus.jsonl"])
    queries = _read_records(sources["queries.jsonl"])

    query_texts = {
        query_id: _validate_text(record.get("text"))
        for query_id, record in queries.items()
    }

    gold = _read_qrels(
        sources[f"qrels/{split}.tsv"],
        corpus_ids=set(corpus),
        query_ids=set(queries),
    )

    chunks = []

    for document_id in sorted(corpus):
        record = corpus[document_id]

        title = _validate_text(
            record.get("title", ""),
            allow_empty=True,
        ).strip()

        text = _validate_text(
            record.get("text"),
            allow_empty=True,
        ).strip()

        combined_text = "\n\n".join(part for part in (title, text) if part)

        if not combined_text:
            raise ValueError("A corpus document has no usable text.")

        chunks.append(
            {
                "chunk_id": _chunk_id(document_id),
                "document_id": document_id,
                "text": combined_text,
            }
        )

    scenarios = [
        {
            "query_id": query_id,
            "text": query_texts[query_id],
            "relevant_chunk_ids": [
                _chunk_id(document_id) for document_id in sorted(gold[query_id])
            ],
        }
        for query_id in sorted(gold)
    ]

    payload = {
        "schema_version": 1,
        "suite_id": f"beir-scifact-{split}-v1",
        "chunks": chunks,
        "scenarios": scenarios,
    }

    suite_bytes = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    if len(suite_bytes) > dataset.MAX_DATASET_BYTES:
        raise ValueError("Converted suite exceeds the retrieval size limit.")

    with TemporaryDirectory(prefix="retrieval-import-") as temporary_dir:
        temporary_path = Path(temporary_dir) / "suite.json"
        temporary_path.write_bytes(suite_bytes)
        dataset.load_retrieval_dataset(temporary_path)

    provenance = {
        "schema_version": 1,
        "dataset_name": "beir/scifact",
        "distribution_url": SOURCE_URL,
        "split": split,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "importer_version": IMPORTER_VERSION,
        "source_file_sha256": {
            name: _sha256(raw) for name, raw in sorted(sources.items())
        },
        "representation": "one-document-one-chunk",
        "text_join_policy": "strip-title-text-join-double-newline-v1",
        "relevance_policy": "integer-score-greater-than-zero",
        "query_selection": "all-query-ids-in-selected-qrels",
        "selected_query_ids": sorted(gold),
        "corpus_count": len(chunks),
        "query_count": len(scenarios),
        "suite_size_bytes": len(suite_bytes),
        "output_suite_sha256": _sha256(suite_bytes),
        "license_references": LICENSE_REFERENCES,
    }

    provenance_bytes = (
        json.dumps(
            provenance,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "suite.json").write_bytes(suite_bytes)
    (output_dir / "provenance.json").write_bytes(provenance_bytes)

    return provenance
