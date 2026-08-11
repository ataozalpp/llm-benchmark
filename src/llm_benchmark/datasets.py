from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

from .config import DatasetConfig
from .models import DatasetExample


class DatasetSource(Protocol):
    def load(self, config: DatasetConfig) -> list[DatasetExample]: ...


class LocalJsonlDatasetSource:
    def load(self, config: DatasetConfig) -> list[DatasetExample]:
        assert config.path is not None
        try:
            lines = config.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"cannot load local dataset {config.path}: {exc}") from exc
        examples = [DatasetExample(**json.loads(line)) for line in lines if line.strip()]
        return examples


class HuggingFaceMMLUProSource:
    def load(self, config: DatasetConfig) -> list[DatasetExample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("MMLU-Pro requires the optional 'huggingface' dependencies") from exc
        try:
            rows = load_dataset(config.name, split=config.split, revision=config.revision)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load {config.name}@{config.revision} split={config.split}; check network/cache and revision: {exc}"
            ) from exc
        return normalize_mmlu_rows(rows)


def normalize_mmlu_rows(rows: Iterable[dict[str, Any]]) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for index, row in enumerate(rows):
        options = [str(value) for value in row["options"]]
        answer = row["answer"]
        if isinstance(answer, int):
            answer = chr(65 + answer)
        examples.append(DatasetExample(
            sample_id=str(row.get("question_id", row.get("sample_id", index))),
            question=str(row["question"]),
            options=options,
            correct_answer=str(answer).strip().upper(),
            category=str(row.get("category", "unknown")),
        ))
    return examples


def sample_examples(examples: list[DatasetExample], config: DatasetConfig, seed: int) -> list[DatasetExample]:
    filtered = [e for e in examples if not config.category_filter or e.category in config.category_filter]
    if config.profile == "full":
        return filtered
    grouped: dict[str, list[DatasetExample]] = defaultdict(list)
    for example in filtered:
        grouped[example.category].append(example)
    rng = random.Random(seed)
    selected: list[DatasetExample] = []
    if config.profile == "poc":
        per_category = config.samples_per_category
    else:
        categories = max(1, len(grouped))
        target = config.sample_size or 14
        per_category = max(1, (target + categories - 1) // categories)
    for category in sorted(grouped):
        values = sorted(grouped[category], key=lambda e: e.sample_id)
        selected.extend(rng.sample(values, min(per_category, len(values))))
    selected.sort(key=lambda e: e.sample_id)
    if config.profile == "smoke":
        selected = selected[: config.sample_size or 14]
    return selected


def load_and_sample(config: DatasetConfig, seed: int) -> tuple[list[DatasetExample], dict[str, Any]]:
    source: DatasetSource = LocalJsonlDatasetSource() if config.source == "local" else HuggingFaceMMLUProSource()
    loaded = source.load(config)
    selected = sample_examples(loaded, config, seed)
    source_hash = None
    if config.path and config.path.exists():
        source_hash = hashlib.sha256(config.path.read_bytes()).hexdigest()
    is_mmlu_pro = config.name == "TIGER-Lab/MMLU-Pro"
    manifest = {
        "dataset_name": config.name,
        "dataset_source": config.source,
        "dataset_split": config.split,
        "dataset_revision": config.revision,
        "profile": config.profile,
        "category_filter": config.category_filter,
        "seed": seed,
        "loaded_row_count": len(loaded),
        "final_sample_count": len(selected),
        "selected_sample_ids": [e.sample_id for e in selected],
        "source_file_sha256": source_hash,
        "dataset_license": "MIT" if is_mmlu_pro else "project-owned synthetic fixture",
        "dataset_homepage": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro" if is_mmlu_pro else None,
        "dataset_citation": "MMLU-Pro, arXiv:2406.01574" if is_mmlu_pro else None,
    }
    return selected, manifest
