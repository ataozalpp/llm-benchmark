from pathlib import Path

import pytest

from llm_benchmark.config import DatasetConfig
from llm_benchmark.datasets import (
    LocalJsonlDatasetSource,
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
