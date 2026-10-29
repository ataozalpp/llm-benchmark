from dataclasses import replace

import pytest

from llm_benchmark.retrieval.local_embeddings import (
    EmbeddingInputTooLongError,
    SentenceTransformerCpuEmbedder,
)
from llm_benchmark.retrieval.profile import DenseRetrievalProfile


def make_profile():
    return DenseRetrievalProfile(
        model_id="test/model",
        model_revision="a" * 40,
        embedding_dimension=2,
        max_sequence_length=4,
        query_prefix="",
        document_prefix="",
        device="cpu",
        precision="float32",
        batch_size=8,
        adapter_id="test",
        adapter_version="1",
        truncation_policy="reject",
    )


class FakeMatrix:
    def __init__(self, count, dtype="float32"):
        self.shape = (count, 2)
        self.dtype = dtype
        self.rows = [[1.0, 0.0] for _ in range(count)]

    def __iter__(self):
        return iter(self.rows)


class FakeModel:
    max_seq_length = 4

    def __init__(self):
        self.encode_calls = 0
        self.dtype = "float32"

    def get_embedding_dimension(self):
        return 2

    def tokenizer(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is True
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False

        return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        assert kwargs["device"] == "cpu"
        assert kwargs["precision"] == "float32"
        assert kwargs["prompt"] == ""
        assert kwargs["normalize_embeddings"] is False
        return FakeMatrix(len(texts), self.dtype)


class OutputModel(FakeModel):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.received_texts = None
        self.received_options = None

    def encode(self, texts, **kwargs):
        super().encode(texts, **kwargs)
        self.received_texts = list(texts)
        self.received_options = dict(kwargs)
        return self.output


def test_empty_input_does_not_encode():
    model = FakeModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    assert embedder.encode_documents(()) == ()
    assert model.encode_calls == 0


def test_over_limit_input_is_rejected_before_encoding():
    model = FakeModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(EmbeddingInputTooLongError, match="token limit"):
        embedder.encode_query("one two three")

    assert model.encode_calls == 0


def test_exact_token_limit_is_accepted():
    model = FakeModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    assert embedder.encode_query("one two") == (1.0, 0.0)
    assert model.encode_calls == 1


def test_wrong_output_dtype_is_rejected():
    model = FakeModel()
    model.dtype = "float64"
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(ValueError, match="dtype"):
        embedder.encode_query("one")


def test_model_dimension_must_match_profile():
    model = FakeModel()
    model.get_embedding_dimension = lambda: 3

    with pytest.raises(ValueError, match="dimension"):
        SentenceTransformerCpuEmbedder(model, make_profile())


def test_document_order_and_encoding_options_are_preserved():
    matrix = FakeMatrix(2)
    matrix.rows = [
        [3.0, 4.0],
        [5.0, 12.0],
    ]
    model = OutputModel(matrix)
    profile = replace(make_profile(), batch_size=2)
    embedder = SentenceTransformerCpuEmbedder(model, profile)

    result = embedder.encode_documents(("alpha", "beta"))

    assert model.received_texts == ["alpha", "beta"]
    assert result == (
        (3.0, 4.0),
        (5.0, 12.0),
    )
    assert all(type(value) is float for row in result for value in row)

    assert model.received_options["batch_size"] == 2
    assert model.received_options["convert_to_numpy"] is True
    assert model.received_options["normalize_embeddings"] is False


@pytest.mark.parametrize(
    "shape",
    [
        (0, 2),
        (2, 2),
        (1, 3),
    ],
)
def test_invalid_embedding_shape_is_rejected(shape):
    matrix = FakeMatrix(1)
    matrix.shape = shape
    model = OutputModel(matrix)
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(ValueError, match="embedding shape"):
        embedder.encode_query("one")


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_nonfinite_embedding_values_are_rejected(invalid_value):
    matrix = FakeMatrix(1)
    matrix.rows = [[invalid_value, 1.0]]
    model = OutputModel(matrix)
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(ValueError, match="non-finite"):
        embedder.encode_query("one")


def test_tokenizer_output_count_must_match_inputs():
    class MissingTokenRowsModel(FakeModel):
        def tokenizer(self, texts, **kwargs):
            return {"input_ids": []}

    model = MissingTokenRowsModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(ValueError, match="Tokenizer output"):
        embedder.encode_query("one")

    assert model.encode_calls == 0


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("device", "cuda", "CPU float32"),
        ("precision", "float16", "CPU float32"),
        ("query_prefix", "query: ", "empty prefixes"),
        ("document_prefix", "passage: ", "empty prefixes"),
    ],
)
def test_unsupported_adapter_settings_are_rejected(field_name, value, message):
    profile = replace(make_profile(), **{field_name: value})
    model = FakeModel()

    with pytest.raises(ValueError, match=message):
        SentenceTransformerCpuEmbedder(model, profile)

    assert model.encode_calls == 0


def test_model_sequence_limit_must_match_profile():
    model = FakeModel()
    model.max_seq_length = 8

    with pytest.raises(ValueError, match="sequence length"):
        SentenceTransformerCpuEmbedder(model, make_profile())


@pytest.mark.parametrize("texts", ["one", ["one"], None])
def test_document_input_requires_tuple_before_tokenization(texts):
    class NoTokenizationModel(FakeModel):
        def tokenizer(self, *args, **kwargs):
            pytest.fail("Invalid input must not reach tokenization.")

    model = NoTokenizationModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(TypeError, match="tuple"):
        embedder.encode_documents(texts)

    assert model.encode_calls == 0


@pytest.mark.parametrize("text", ["", " \t\n", None, 42])
def test_invalid_text_is_rejected_before_tokenization(text):
    class NoTokenizationModel(FakeModel):
        def tokenizer(self, *args, **kwargs):
            pytest.fail("Invalid input must not reach tokenization.")

    model = NoTokenizationModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(ValueError, match="nonblank strings"):
        embedder.encode_query(text)

    assert model.encode_calls == 0


def test_later_over_limit_document_rejects_entire_input_before_encoding():
    model = FakeModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(EmbeddingInputTooLongError):
        embedder.encode_documents(("one", "one two three"))

    assert model.encode_calls == 0


def test_truncate_accepts_over_limit_documents_and_counts_inputs():
    model = FakeModel()
    profile = replace(make_profile(), truncation_policy="truncate")
    embedder = SentenceTransformerCpuEmbedder(model, profile)

    vectors = embedder.encode_documents(("one", "one two three"))

    assert len(vectors) == 2
    assert model.encode_calls == 1
    assert embedder.input_statistics() == {
        "schema_version": 1,
        "encoded_document_count": 2,
        "over_limit_document_count": 1,
        "encoded_query_count": 0,
        "over_limit_query_count": 0,
    }


def test_query_statistics_are_separate_from_document_statistics():
    model = FakeModel()
    profile = replace(make_profile(), truncation_policy="truncate")
    embedder = SentenceTransformerCpuEmbedder(model, profile)

    embedder.encode_documents(("one",))
    embedder.encode_query("one two three")

    assert embedder.input_statistics() == {
        "schema_version": 1,
        "encoded_document_count": 1,
        "over_limit_document_count": 0,
        "encoded_query_count": 1,
        "over_limit_query_count": 1,
    }


def test_exact_limit_is_not_counted_as_over_limit():
    model = FakeModel()
    profile = replace(make_profile(), truncation_policy="truncate")
    embedder = SentenceTransformerCpuEmbedder(model, profile)

    embedder.encode_query("one two")

    stats = embedder.input_statistics()
    assert stats["encoded_query_count"] == 1
    assert stats["over_limit_query_count"] == 0


def test_rejected_input_does_not_increment_success_counters():
    embedder = SentenceTransformerCpuEmbedder(FakeModel(), make_profile())
    before = embedder.input_statistics()

    with pytest.raises(EmbeddingInputTooLongError):
        embedder.encode_documents(("one", "one two three"))

    assert embedder.input_statistics() == before


def test_invalid_embedding_output_does_not_increment_counters():
    model = FakeModel()
    model.dtype = "float64"
    profile = replace(make_profile(), truncation_policy="truncate")
    embedder = SentenceTransformerCpuEmbedder(model, profile)
    before = embedder.input_statistics()

    with pytest.raises(ValueError, match="dtype"):
        embedder.encode_query("one two three")

    assert embedder.input_statistics() == before


def test_statistics_count_repeated_input_occurrences():
    embedder = SentenceTransformerCpuEmbedder(FakeModel(), make_profile())

    embedder.encode_query("one")
    embedder.encode_query("one")

    assert embedder.input_statistics()["encoded_query_count"] == 2


def test_statistics_snapshot_cannot_mutate_internal_counters():
    embedder = SentenceTransformerCpuEmbedder(FakeModel(), make_profile())

    snapshot = embedder.input_statistics()
    snapshot["encoded_query_count"] = 999

    assert embedder.input_statistics()["encoded_query_count"] == 0


def test_empty_documents_do_not_change_statistics():
    model = FakeModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())
    before = embedder.input_statistics()

    assert embedder.encode_documents(()) == ()
    assert embedder.input_statistics() == before
    assert model.encode_calls == 0