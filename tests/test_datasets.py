from io import BytesIO
from pathlib import Path

import pytest

from llm_benchmark.config import DatasetConfig
from llm_benchmark.dataset_storage import LocalDatasetStorage
from llm_benchmark.datasets import (
    DatasetLoader,
    LocalJsonlDatasetSource,
    UploadedDatasetLoadError,
    UploadedDatasetSource,
    load_and_sample,
    normalize_mmlu_rows,
    sample_examples,
)


def test_local_fixture_loads() -> None:
    config = DatasetConfig(source="local", name="fixture", path=Path("data/fixtures/mcq_fixture.jsonl"), profile="full")
    examples = LocalJsonlDatasetSource().load(config)
    assert len(examples) == 8
    assert examples[0].allowed_labels == ["A", "B", "C", "D"]


def test_mmlu_normalization_and_seeded_balanced_sampling() -> None:
    rows = [
        {"question_id": f"{category}-{index}", "question": "Q", "options": ["x", "y"], "answer": index % 2, "category": category}
        for category in ("a", "b") for index in range(10)
    ]
    examples = normalize_mmlu_rows(rows)
    config = DatasetConfig(source="huggingface", name="TIGER-Lab/MMLU-Pro", revision="abc", profile="poc", samples_per_category=3)
    first = sample_examples(examples, config, 42)
    second = sample_examples(examples, config, 42)
    assert [x.sample_id for x in first] == [x.sample_id for x in second]
    assert len(first) == 6
    assert first[0].correct_answer in {"A", "B"}


def test_explicit_sample_ids_preserve_configured_order() -> None:
    examples = normalize_mmlu_rows([
        {"question_id": "2019", "question": "Q1", "options": ["x", "y"], "answer": 1, "category": "a"},
        {"question_id": "215", "question": "Q2", "options": ["x", "y"], "answer": 0, "category": "b"},
    ])
    config = DatasetConfig(
        source="huggingface",
        name="TIGER-Lab/MMLU-Pro",
        revision="abc",
        profile="smoke",
        sample_size=2,
        sample_ids=["215", "2019"],
    )
    assert [example.sample_id for example in sample_examples(examples, config, 42)] == ["215", "2019"]


def test_explicit_sample_id_must_exist() -> None:
    examples = normalize_mmlu_rows([
        {"question_id": "2019", "question": "Q", "options": ["x", "y"], "answer": 1, "category": "a"},
    ])
    config = DatasetConfig(
        source="huggingface", name="TIGER-Lab/MMLU-Pro", revision="abc", sample_size=1, sample_ids=["missing"]
    )
    with pytest.raises(ValueError, match="missing"):
        sample_examples(examples, config, 42)


@pytest.mark.parametrize(
    ("file_format", "adapter_type", "content"),
    [
        (
            "csv",
            "tabular_mcq_csv_v1",
            b"sample_id,question,option_A,option_B,correct_answer,category\n"
            b"002,Q2,A2,B2,B,second\n001,Q1,A1,B1,A,first\n",
        ),
        (
            "jsonl",
            "tabular_mcq_jsonl_v1",
            b'{"sample_id":"002","question":"Q2","options":["A2","B2"],'
            b'"correct_answer":"B","category":"second"}\n'
            b'{"sample_id":"001","question":"Q1","options":["A1","B1"],'
            b'"correct_answer":"A","category":"first"}\n',
        ),
    ],
)
def test_uploaded_source_verifies_and_loads_with_registered_adapter(
    tmp_path: Path,
    file_format: str,
    adapter_type: str,
    content: bytes,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    stored = storage.store(BytesIO(content), file_format)  # type: ignore[arg-type]
    config = DatasetConfig.model_validate({
        "source": "uploaded",
        "name": "uploaded-fixture",
        "storage_key": stored.storage_key,
        "adapter_type": adapter_type,
        "checksum": f"sha256:{stored.checksum_sha256}",
        "profile": "smoke",
        "sample_size": 2,
        "sample_ids": ["001", "002"],
    })

    examples, manifest = load_and_sample(
        config,
        42,
        loader=DatasetLoader(storage).load,
    )

    assert [example.sample_id for example in examples] == ["001", "002"]
    assert manifest["storage_key"] == stored.storage_key
    assert manifest["adapter_type"] == adapter_type
    assert manifest["checksum"] == f"sha256:{stored.checksum_sha256}"
    assert str(tmp_path) not in repr(manifest)


def test_uploaded_source_rejects_unsupported_adapter_defensively(tmp_path: Path) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    stored = storage.store(BytesIO(b"content"), "csv")
    config = DatasetConfig.model_construct(
        source="uploaded",
        name="uploaded",
        storage_key=stored.storage_key,
        adapter_type="unsupported",
        checksum=f"sha256:{stored.checksum_sha256}",
    )

    with pytest.raises(UploadedDatasetLoadError, match="adapter is unsupported"):
        UploadedDatasetSource(storage).load(config)
