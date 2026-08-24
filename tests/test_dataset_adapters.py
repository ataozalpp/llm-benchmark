from pathlib import Path

import pandas as pd
import pytest

from llm_benchmark.dataset_adapters import (
    DatasetAdapterError,
    _require_non_empty,
    load_tabular_examples,
)


def test_csv_adapter_preserves_ids_and_option_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.csv"
    dataset_path.write_text(
        "sample_id,question,option_A,option_B,option_C,correct_answer,category\n"
        "001,Choose a letter,Alpha,Beta,Gamma,B,test\n",
        encoding="utf-8",
    )

    examples = load_tabular_examples(dataset_path, "csv")

    assert len(examples) == 1
    assert examples[0].sample_id == "001"
    assert examples[0].options == ["Alpha", "Beta", "Gamma"]
    assert examples[0].correct_answer == "B"


def test_csv_adapter_preserves_row_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.csv"
    dataset_path.write_text(
        "sample_id,question,option_A,option_B,correct_answer,category\n"
        "002,Second question,Yes,No,A,test\n"
        "001,First question,True,False,B,test\n",
        encoding="utf-8",
    )

    examples = load_tabular_examples(dataset_path, "csv")

    assert [example.sample_id for example in examples] == ["002", "001"]


def test_csv_adapter_rejects_missing_required_column(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing-category.csv"
    dataset_path.write_text(
        "sample_id,question,option_A,option_B,correct_answer\n"
        "001,Choose a letter,Alpha,Beta,B\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="missing required columns",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_rejects_non_contiguous_option_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "non-contiguous-options.csv"
    dataset_path.write_text(
        "sample_id,question,option_A,option_B,option_D,"
        "correct_answer,category\n"
        "001,Choose a letter,Alpha,Beta,Delta,B,test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="contiguous",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_requires_at_least_two_options(tmp_path: Path) -> None:
    dataset_path = tmp_path / "one-option.csv"
    dataset_path.write_text(
         "sample_id,question,option_A,correct_answer,category\n"
        "001,Choose a letter,Alpha,A,test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="at least two",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_rejects_empty_question(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty-question.csv"
    dataset_path.write_text(
        "sample_id,question,option_A,option_B,correct_answer,category\n"
        "001,   ,Alpha,Beta,A,test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="empty question",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_rejects_empty_option(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty-option.csv"
    dataset_path.write_text(
         "sample_id,question,option_A,option_B,correct_answer,category\n"
        "001,Choose a letter,Alpha,   ,A,test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="empty option",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_rejects_genuinely_missing_field(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing-category-value.csv"
    dataset_path.write_text(
        "sample_id,question,option_A,option_B,correct_answer,category\n"
        "001,Choose a letter,Alpha,Beta,A,\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="empty category",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_rejects_sample_id(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty-sample-id.csv"
    dataset_path.write_text(
         "sample_id,question,option_A,option_B,correct_answer,category\n"
        "   ,Choose a letter,Alpha,Beta,A,test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="empty sample_id",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapters_rejects_empty_category(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty-category.csv"
    dataset_path.write_text(
         "sample_id,question,option_A,option_B,correct_answer,category\n"
        "001,Choose a letter,Alpha,Beta,A,   \n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="empty category",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapeter_rejects_out_of_range_answer(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid-answer.csv"
    dataset_path.write_text(
         "sample_id,question,option_A,option_B,correct_answer,category\n"
        "001,Choose a letter,Alpha,Beta,C,test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="available option",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_normalizes_answer_to_uppercase(tmp_path: Path) -> None:
    dataset_path = tmp_path / "lowercase-answer.csv"
    dataset_path.write_text(
         "sample_id,question,option_A,option_B,correct_answer,category\n"
        "001,Choose a letter,Alpha,Beta,b,test\n",
        encoding="utf-8",
    )

    examples = load_tabular_examples(dataset_path, "csv")

    assert examples[0].correct_answer == "B"


def test_csv_adapter_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "duplicate-ids.csv"
    dataset_path.write_text(
        "sample_id,question,option_A,option_B,correct_answer,category\n"
        "001,First question,Alpha,Beta,A,test\n"
        "001,Second question,Gamma,Delta,B,test\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="duplicate sample IDs",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_csv_adapter_accepts_options_through_j(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ten-options.csv"
    dataset_path.write_text(
        "sample_id,question,"
        "option_A,option_B,option_C,option_D,option_E,"
        "option_F,option_G,option_H,option_I,option_J,"
        "correct_answer,category\n"
        "001,Choose a letter,"
        "A text,B text,C text,D text,E text,"
        "F text,G text,H text,I text,J text,"
        "J,test\n",
        encoding="utf-8",
    )

    examples = load_tabular_examples(dataset_path, "csv")

    assert examples[0].allowed_labels == [
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J",
    ]
    assert examples[0].correct_answer == "J"


def test_csv_adapter_rejects_option_beyond_j(tmp_path: Path) -> None:
    dataset_path = tmp_path / "eleven-options.csv"
    dataset_path.write_text(
        "sample_id,question,"
        "option_A,option_B,option_C,option_D,option_E,"
        "option_F,option_G,option_H,option_I,option_J,option_K,"
        "correct_answer,category\n"
        "001,Choose a letter,"
        "A,B,C,D,E,F,G,H,I,J,K,K,test\n",
        encoding="utf-8"
    )

    with pytest.raises(
        DatasetAdapterError,
        match="option_A to option_J",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_adapter_rejects_unknown_format(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.xlsx"
    dataset_path.write_text("not-a-supported-format", encoding="utf-8")

    with pytest.raises(
        DatasetAdapterError,
        match="Unsupported dataset format",
    ):
        load_tabular_examples(dataset_path, "unknown")


def test_csv_adapter_wraps_parsing_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty.csv"
    dataset_path.write_text(
        "", encoding="utf-8"
    )

    with pytest.raises(
        DatasetAdapterError,
        match="could not be parsed",
    ):
        load_tabular_examples(dataset_path, "csv")


def test_jsonl_adapter_preserves_ids_and_option_order(
    tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        '{"sample_id":"001","question":"Choose a letter",'
        '"options":["Alpha","Beta","Gamma"],'
        '"correct_answer":"B","category":"test"}\n',
        encoding="utf-8",
    )

    examples = load_tabular_examples(dataset_path, "jsonl")

    assert len(examples) == 1
    assert examples[0].sample_id == "001"
    assert examples[0].options == ["Alpha", "Beta", "Gamma"]
    assert examples[0].correct_answer == "B"
    assert examples[0].category == "test"


def test_jsonl_adapter_preserves_row_order(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        '{"sample_id":"002","question":"Second",'
        '"options":["Yes","No"],'
        '"correct_answer":"A","category":"test"}\n'
        '{"sample_id":"001","question":"First",'
        '"options":["True","False"],'
        '"correct_answer":"B","category":"test"}\n',
        encoding="utf-8",
    )

    examples = load_tabular_examples(dataset_path, "jsonl")

    assert [example.sample_id for example in examples] == ["002", "001"]


def test_jsonl_adapter_rejects_null_question(tmp_path: Path) -> None:
    dataset_path = tmp_path / "null-question.jsonl"
    dataset_path.write_text(
        '{"sample_id":"001","question":null,'
        '"options":["Alpha","Beta"],'
        '"correct_answer":"A","category":"test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="empty question",
    ):
        load_tabular_examples(dataset_path, "jsonl")


def test_jsonl_adapter_rejects_null_option(tmp_path: Path) -> None:
    dataset_path = tmp_path / "null-option.jsonl"
    dataset_path.write_text(
        '{"sample_id":"001","question":"Question",'
        '"options":["Alpha",null],'
        '"correct_answer":"A","category":"test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="empty option",
    ):
        load_tabular_examples(dataset_path, "jsonl")


def test_jsonl_adapter_preserves_scalar_string_normalization(tmp_path: Path) -> None:
    dataset_path = tmp_path / "normalized-scalars.jsonl"
    dataset_path.write_text(
        '{"sample_id":123,"question":"  Question  ",'
        '"options":["  Alpha  ","Beta"],'
        '"correct_answer":" a ","category":"  test  "}\n',
        encoding="utf-8",
    )

    example = load_tabular_examples(dataset_path, "jsonl")[0]

    assert example.sample_id == "123"
    assert example.question == "Question"
    assert example.options == ["Alpha", "Beta"]
    assert example.correct_answer == "A"
    assert example.category == "test"


@pytest.mark.parametrize("missing_value", [pd.NA, float("nan")])
def test_missing_scalar_never_becomes_literal_text(missing_value: object) -> None:
    with pytest.raises(DatasetAdapterError, match="empty question"):
        _require_non_empty(missing_value, "question", "test")


@pytest.mark.parametrize("non_scalar", [["text"], {"value": "text"}])
def test_non_scalar_text_field_is_rejected(non_scalar: object) -> None:
    with pytest.raises(DatasetAdapterError, match="empty question"):
        _require_non_empty(non_scalar, "question", "test")


def test_jsonl_adapter_rejects_missing_required_field(
        tmp_path: Path
) -> None:
    dataset_path = tmp_path / "missing-category.jsonl"
    dataset_path.write_text(
        '{"sample_id":"001","question":"Question",'
        '"options":["A text","B text"],'
        '"correct_answer":"A"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="missing required columns"
    ):

        load_tabular_examples(dataset_path, "jsonl")


def test_jsonl_adapter_requires_options_list(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "invalid-options.jsonl"
    dataset_path.write_text(
        '{"sample_id":"001","question":"Question",'
        '"options":"Alpha|Beta",'
        '"correct_answer":"A","category":"test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="options must be a list",
    ):
        load_tabular_examples(dataset_path, "jsonl")


def test_jsonl_adapter_wraps_parsing_error(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "malformed.jsonl"
    dataset_path.write_text(
        '{"sample_id":',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="could not be parsed",
    ):
        load_tabular_examples(dataset_path, "jsonl")


def test_jsonl_adapter_rejects_duplicate_sample_ids(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "duplicate-ids.jsonl"
    dataset_path.write_text(
        '{"sample_id":"001","question":"First",'
        '"options":["A","B"],'
        '"correct_answer":"A","category":"test"}\n'
        '{"sample_id":"001","question":"Second",'
        '"options":["C","D"],'
        '"correct_answer":"B","category":"test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="duplicate sample IDs",
    ):
        load_tabular_examples(dataset_path, "jsonl")


def test_jsonl_adapter_applies_common_answer_validation(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "invalid-answer.jsonl"
    dataset_path.write_text(
        '{"sample_id":"001","question":"Question",'
        '"options":["Alpha","Beta"],'
        '"correct_answer":"C","category":"test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetAdapterError,
        match="available option",
    ):
        load_tabular_examples(dataset_path, "jsonl")