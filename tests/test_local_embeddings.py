import pytest

from llm_benchmark.retrieval.local_embeddings import SentenceTransformerCpuEmbedder
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


def test_empty_input_does_not_encode():
    model = FakeModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    assert embedder.encode_documents(()) == ()
    assert model.encode_calls == 0


def test_over_limit_input_is_rejected_before_encoding():
    model = FakeModel()
    embedder = SentenceTransformerCpuEmbedder(model, make_profile())

    with pytest.raises(ValueError, match="token limit"):
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
