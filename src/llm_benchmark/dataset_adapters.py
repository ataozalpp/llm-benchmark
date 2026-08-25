from pathlib import Path
from typing import Literal

import pandas as pd

from .models import DatasetExample

DatasetFileFormat = Literal["csv", "jsonl"]

_REQUIRED_CSV_COLUMNS = frozenset({
    "sample_id",
    "question",
    "correct_answer",
    "category",
})

_OPTION_LABELS = tuple(
    chr(ord("A") + index)
    for index in range(10)
)

_OPTION_COLUMNS = tuple(
    f"option_{label}"
    for label in _OPTION_LABELS
)

_REQUIRED_JSONL_COLUMNS = frozenset({
    "sample_id",
    "question",
    "options",
    "correct_answer",
    "category",
})

class DatasetAdapterError(ValueError):
    """Raised when a tabular dataset cannot be normalized safely."""


def _validate_required_columns(
    frame:pd.DataFrame,
    required_columns: frozenset[str],
    source_name: str,
) -> None:
    missing_columns = sorted(
        required_columns - set(frame.columns)
    )
    if missing_columns:
        raise DatasetAdapterError(
            f"{source_name} dataset is missing required columns: "
            + ", ".join(missing_columns)
        )


def _get_csv_option_columns(
        frame: pd.DataFrame,
) -> list[str]:
    present_columns = [
        column
        for column in frame.columns
        if column.startswith("option_")
    ]

    unsupported_columns = [
        column
        for column in present_columns
        if column not in _OPTION_COLUMNS
    ]
    if unsupported_columns:
        raise DatasetAdapterError(
            "CSV option columns must use labels from option_A to option_J"
        )

    if len(present_columns) < 2:
        raise DatasetAdapterError(
            "CSV dataset must contain at least two option columns"
        )

    expected_columns = list(
        _OPTION_COLUMNS[:len(present_columns)]
    )
    if present_columns != expected_columns:
        raise DatasetAdapterError(
            "CSV option columns must be contiguous and ordered from option_A"
        )

    return present_columns


def _require_non_empty(
        value: object,
        field_name: str,
        source_name: str,
) -> str:
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        raise DatasetAdapterError(
            f"{source_name} dataset contains an empty {field_name}"
        )
    normalized_value = str(value).strip()
    if not normalized_value:
        raise DatasetAdapterError(
            f"{source_name} dataset contains an empty {field_name}"
        )
    return normalized_value


def _normalize_example(
        *,
        sample_id: object,
        question: object,
        options: list[object],
        correct_answer: object,
        category: object,
        source_name: str,
) -> DatasetExample:
    normalized_sample_id = _require_non_empty(
        sample_id,
        "sample_id",
        source_name,
    )
    normalized_question = _require_non_empty(
        question,
        "question",
        source_name,
    )
    normalized_options = [
        _require_non_empty(
            option,
            "option",
            source_name
        )
        for option in options
    ]
    normalized_correct_answer = _require_non_empty(
        correct_answer,
        "correct_answer",
        source_name,
    ).upper()
    normalized_category = _require_non_empty(
        category,
        "category",
        source_name,
    )

    if len(normalized_options) < 2:
        raise DatasetAdapterError(
            f"{source_name} dataset must contain at least two options"
        )

    if len(normalized_options) > len(_OPTION_LABELS):
        raise DatasetAdapterError(
            f"{source_name} dataset cannot contain more than ten options"
        )

    allowed_labels = _OPTION_LABELS[:len(normalized_options)]
    if normalized_correct_answer not in allowed_labels:
        raise DatasetAdapterError(
            f"{source_name} correct_answer must match "
            "an available option label"
        )

    return DatasetExample(
        sample_id=normalized_sample_id,
        question=normalized_question,
        options=normalized_options,
        correct_answer=normalized_correct_answer,
        category=normalized_category,
    )


def _load_csv(path:Path) -> list[DatasetExample]:
    try:
        frame = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8",
            )
    except(
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        raise DatasetAdapterError(
            "CSV dataset could not be parsed"
        ) from exc

    _validate_required_columns(
        frame,
        _REQUIRED_CSV_COLUMNS,
        "CSV",
    )

    option_columns = _get_csv_option_columns(frame)

    examples: list[DatasetExample] = []
    seen_sample_ids: set[str] = set()

    for row in frame.to_dict(orient="records"):
        example = _normalize_example(
            sample_id=row["sample_id"],
            question=row["question"],
            options=[
                row[column]
                for column in option_columns
            ],
            correct_answer=row["correct_answer"],
            category=row["category"],
            source_name="CSV",
        )

        if example.sample_id in seen_sample_ids:
            raise DatasetAdapterError(
                "CSV dataset contains duplicate sample IDs"
            )
        seen_sample_ids.add(example.sample_id)

        examples.append(example)

    return examples


def _load_jsonl(path: Path) ->list[DatasetExample]:
    try:
        frame = pd.read_json(
            path,
            lines=True,
            encoding="utf-8",
            dtype=False,
        )
    except(
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise DatasetAdapterError(
            "JSONL dataset could not be parsed"
        ) from exc
    _validate_required_columns(
        frame,
        _REQUIRED_JSONL_COLUMNS,
        "JSONL",
    )

    examples: list[DatasetExample] = []
    seen_sample_ids: set[str] = set()

    for row in frame.to_dict(orient="records"):
        raw_options = row["options"]
        if not isinstance(raw_options, list):
            raise DatasetAdapterError(
                "JSONL options must be a list"
            )

        example = _normalize_example(
            sample_id=row["sample_id"],
            question=row["question"],
            options=raw_options,
            correct_answer=row["correct_answer"],
            category=row["category"],
            source_name="JSONL",
        )

        if example.sample_id in seen_sample_ids:
            raise DatasetAdapterError(
                "JSONL dataset contains duplicate sample IDs"
            )
        seen_sample_ids.add(example.sample_id)

        examples.append(example)

    return examples


def load_tabular_examples(
        path: Path,
        file_format: DatasetFileFormat,
) -> list[DatasetExample]:
    if file_format == "csv":
        examples = _load_csv(path)
    elif file_format == "jsonl":
        examples = _load_jsonl(path)
    else:
        raise DatasetAdapterError(
            "Unsupported dataset format"
        )

    if not examples:
        raise DatasetAdapterError(
            "Dataset must contain at least one sample"
        )

    return examples