import json
import runpy
import sys
from pathlib import Path

import pytest

from llm_benchmark.retrieval.profile import DenseRetrievalProfile

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_dense_cpu.py"


@pytest.fixture
def smoke():
    return runpy.run_path(str(SCRIPT_PATH))


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


def test_empty_input_does_not_encode(smoke):
    model = FakeModel()
    embedder = smoke["SmokeEmbedder"](model, make_profile())

    assert embedder.encode_documents(()) == ()
    assert model.encode_calls == 0


def test_over_limit_input_is_rejected_before_encoding(smoke):
    model = FakeModel()
    embedder = smoke["SmokeEmbedder"](model, make_profile())

    with pytest.raises(ValueError, match="token limit"):
        embedder.encode_query("one two three")

    assert model.encode_calls == 0


def test_exact_token_limit_is_accepted(smoke):
    model = FakeModel()
    embedder = smoke["SmokeEmbedder"](model, make_profile())

    assert embedder.encode_query("one two") == (1.0, 0.0)
    assert model.encode_calls == 1


def test_wrong_output_dtype_is_rejected(smoke):
    model = FakeModel()
    model.dtype = "float64"
    embedder = smoke["SmokeEmbedder"](model, make_profile())

    with pytest.raises(ValueError, match="dtype"):
        embedder.encode_query("one")


def test_model_dimension_must_match_profile(smoke):
    model = FakeModel()
    model.get_embedding_dimension = lambda: 3

    with pytest.raises(ValueError, match="dimension"):
        smoke["SmokeEmbedder"](model, make_profile())


def test_max_abs_difference(smoke):
    result = smoke["max_abs_difference"](
        (1.0, 2.0),
        (1.0, 2.25),
    )

    assert result == pytest.approx(0.25)


def test_tokenizer_class_name_is_json_serializable(smoke):
    model = FakeModel()
    name = smoke["qualified_class_name"](model)

    assert name == (f"{FakeModel.__module__}.{FakeModel.__qualname__}")

    serialized = json.dumps(
        {"tokenizer_class": name},
        allow_nan=False,
    )
    assert json.loads(serialized)["tokenizer_class"] == name


def test_long_input_fixture_must_actually_exceed_limit(smoke):
    class CollapsingTokenizerModel(FakeModel):
        def tokenizer(self, texts, **kwargs):
            return {"input_ids": [[1, 2, 3] for _ in texts]}

    model = CollapsingTokenizerModel()
    embedder = smoke["SmokeEmbedder"](model, make_profile())

    with pytest.raises(RuntimeError, match="fixture did not exceed"):
        smoke["check_long_input_rejection"](embedder)

    assert model.encode_calls == 0


def test_long_input_check_detects_missing_rejection(smoke):
    class AcceptingEmbedder:
        def __init__(self):
            self.model = FakeModel()
            self.profile = make_profile()

        def encode_query(self, text):
            return (1.0, 0.0)

    with pytest.raises(RuntimeError, match="Long input was not rejected"):
        smoke["check_long_input_rejection"](AcceptingEmbedder())


def test_cli_reports_serialization_failure(smoke, monkeypatch, capsys):
    main = smoke["main"]

    monkeypatch.setitem(
        main.__globals__,
        "run_smoke",
        lambda args: {"tokenizer_class": FakeModel},
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])

    assert main() == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "TypeError" in captured.err
    assert "Traceback" not in captured.err


def test_cli_prints_valid_json(smoke, monkeypatch, capsys):
    main = smoke["main"]
    report = {
        "status": "passed",
        "tokenizer_class": smoke["qualified_class_name"](FakeModel()),
    }

    monkeypatch.setitem(
        main.__globals__,
        "run_smoke",
        lambda args: report,
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])

    assert main() == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == report
    assert captured.err == ""


def test_long_input_fixture_exceeds_limit_and_is_rejected(smoke):
    model = FakeModel()
    embedder = smoke["SmokeEmbedder"](model, make_profile())

    count = smoke["check_long_input_rejection"](embedder)

    assert count > embedder.profile.max_sequence_length
    assert model.encode_calls == 0


def test_long_input_fixture_at_limit_is_invalid(smoke):
    class ExactLimitTokenizerModel(FakeModel):
        def tokenizer(self, texts, **kwargs):
            return {"input_ids": [list(range(self.max_seq_length)) for _ in texts]}

    model = ExactLimitTokenizerModel()
    embedder = smoke["SmokeEmbedder"](model, make_profile())

    with pytest.raises(RuntimeError, match="fixture did not exceed"):
        smoke["check_long_input_rejection"](embedder)

    assert model.encode_calls == 0
