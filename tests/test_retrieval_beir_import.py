import hashlib
import json

import pytest

from llm_benchmark.retrieval import beir_import, dataset
from llm_benchmark.retrieval.beir_import import prepare_scifact
from llm_benchmark.retrieval.dataset import load_retrieval_dataset


def write_jsonl(path, records):
    text = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def source_dir(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "qrels").mkdir()

    write_jsonl(
        source / "corpus.jsonl",
        [
            {"_id": "d2", "title": "", "text": "Second evidence."},
            {"_id": "d1", "title": "First title", "text": " First evidence. "},
            {"_id": "d3", "title": "Distractor", "text": "Unrelated content."},
        ],
    )

    write_jsonl(
        source / "queries.jsonl",
        [
            {"_id": "q2", "text": "Second query"},
            {"_id": "q1", "text": "First query"},
            {"_id": "unused", "text": "Another split"},
        ],
    )

    (source / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t2\nq1\td3\t0\nq2\td2\t1\n",
        encoding="utf-8",
    )

    return source


def test_import_mapping_and_provenance(source_dir, tmp_path):
    output = tmp_path / "prepared"

    provenance = prepare_scifact(source_dir, output)
    loaded = load_retrieval_dataset(output / "suite.json")

    assert loaded.suite_id == "beir-scifact-test-v1"
    assert len(loaded.chunks) == 3
    assert len(loaded.scenarios) == 2

    assert loaded.chunks[0].chunk_id == "beir-doc:d1"
    assert loaded.chunks[0].document_id == "d1"
    assert loaded.chunks[0].text == "First title\n\nFirst evidence."

    assert loaded.scenarios[0].query.query_id == "q1"
    assert loaded.scenarios[0].evaluation.relevant_chunk_ids == frozenset(
        {"beir-doc:d1", "beir-doc:d2"}
    )

    assert provenance["selected_query_ids"] == ["q1", "q2"]
    assert provenance["corpus_count"] == 3

    actual_hash = hashlib.sha256((output / "suite.json").read_bytes()).hexdigest()

    assert provenance["output_suite_sha256"] == actual_hash

    recorded = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert recorded == provenance


def test_repeated_import_produces_identical_suite(source_dir, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    prepare_scifact(source_dir, first)
    prepare_scifact(source_dir, second)

    assert (first / "suite.json").read_bytes() == (second / "suite.json").read_bytes()


@pytest.mark.parametrize(
    "rows",
    [
        "q1\tmissing\t1\n",
        "missing\td1\t1\n",
        "q1\td1\t-1\n",
        "q1\td1\tnot-a-number\n",
        "q1\td1\t0\n",
        "q1\td1\t1\nq1\td1\t2\n",
        "q1\td1\n",
    ],
)
def test_invalid_qrels_create_no_output(source_dir, tmp_path, rows):
    (source_dir / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + rows,
        encoding="utf-8",
    )

    output = tmp_path / "prepared"

    with pytest.raises(ValueError):
        prepare_scifact(source_dir, output)

    assert not output.exists()


def test_duplicate_corpus_id_is_rejected(source_dir, tmp_path):
    path = source_dir / "corpus.jsonl"
    existing = path.read_text(encoding="utf-8")
    first_line = existing.splitlines()[0]

    path.write_text(existing + first_line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate BEIR record ID"):
        prepare_scifact(source_dir, tmp_path / "prepared")


def test_duplicate_json_key_is_rejected(source_dir, tmp_path):
    (source_dir / "queries.jsonl").write_text(
        '{"_id":"q1","_id":"q2","text":"query"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        prepare_scifact(source_dir, tmp_path / "prepared")


def test_existing_output_is_not_overwritten(source_dir, tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        prepare_scifact(source_dir, output)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_source_size_limit(source_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(beir_import, "MAX_SOURCE_BYTES", 8)

    with pytest.raises(ValueError, match="source file exceeds"):
        prepare_scifact(source_dir, tmp_path / "prepared")


def test_output_size_limit(source_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "MAX_DATASET_BYTES", 8)

    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="Converted suite exceeds"):
        prepare_scifact(source_dir, output)

    assert not output.exists()
