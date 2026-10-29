import json
from types import SimpleNamespace

from llm_benchmark.retrieval import embedding_runtime


class FakeTokenizer:
    is_fast = True


class FakeParameter:
    def __init__(self, dtype, floating):
        self.dtype = dtype
        self.floating = floating

    def is_floating_point(self):
        return self.floating


def test_runtime_metadata_is_json_serializable(monkeypatch):
    monkeypatch.setattr(
        embedding_runtime,
        "version",
        lambda package: f"test-{package}",
    )
    monkeypatch.setattr(
        embedding_runtime,
        "_thread_counts",
        lambda: {
            "torch_num_threads": 4,
            "torch_num_interop_threads": 2,
        },
    )

    model = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        device="cpu",
        max_seq_length=256,
        get_embedding_dimension=lambda: 384,
        parameters=lambda: iter(
            [
                FakeParameter("torch.float32", True),
                FakeParameter("torch.float32", True),
                FakeParameter("torch.in64", False),
            ]
        ),
    )

    result = embedding_runtime.collect_embedding_runtime(model)

    assert result["schema_version"] == 1
    assert result["device"] == "cpu"
    assert result["embedding_dimension"] == 384
    assert result["max_sequence_length"] == 256
    assert result["parameter_dtypes"] == ["torch.float32"]
    assert result["tokenizer_is_fast"] is True
    assert result["torch_num_threads"] == 4
    assert result["torch_num_interop_threads"] == 2

    assert result["tokenizer_class"] == (
        f"{FakeTokenizer.__module__}."
        f"{FakeTokenizer.__qualname__}"
    )

    assert result["runtime_versions"] == {
        package: f"test-{package}"
        for package in embedding_runtime.RUNTIME_PACKAGES
    }

    assert json.loads(json.dumps(result, allow_nan=False)) == result