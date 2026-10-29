import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from llm_benchmark.retrieval import embedding_loader
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


class FakeParameter:
    dtype = "torch.float32"

    def is_floating_point(self):
        return True


class FakeModel:
    device = "cpu"
    max_seq_length = 4

    def __init__(self):
        self.float_called = False
        self.eval_called = False

    def float(self):
        self.float_called = True
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return iter([FakeParameter()])

    def get_embedding_dimension(self):
        return 2


def test_loader_defaults_to_offline_and_prepares_model(monkeypatch):
    model = FakeModel()
    captured = SimpleNamespace()

    def fake_load(profile, *, local_files_only):
        captured.profile = profile
        captured.local_files_only = local_files_only
        return model

    monkeypatch.setattr(embedding_loader, "_load_model", fake_load)

    profile = make_profile()
    embedder = embedding_loader.load_cpu_embedder(profile)

    assert captured.profile is profile
    assert captured.local_files_only is True
    assert model.float_called is True
    assert model.eval_called is True
    assert embedder.model is model
    assert embedder.profile is profile


@pytest.mark.parametrize("offline", [True, False])
def test_model_constructor_receives_pinned_options(monkeypatch, offline):
    captured = {}
    model = FakeModel()
    module = ModuleType("sentence_transformers")

    def constructor(model_id, **kwargs):
        captured.update(model_id=model_id, **kwargs)
        return model

    module.SentenceTransformer = constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    profile = make_profile()

    result = embedding_loader.load_cpu_embedder(profile, local_files_only=offline)

    assert result.model is model
    assert captured == {
        "model_id": profile.model_id,
        "revision": profile.model_revision,
        "device": "cpu",
        "local_files_only": offline,
        "trust_remote_code": False,
    }


@pytest.fixture
def forbid_loading(monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid settings must fail before model loading.")

    monkeypatch.setattr(embedding_loader, "_load_model", unexpected_load)


@pytest.mark.parametrize("profile", [None, {}, "model"])
def test_invalid_profile_type_fails_before_loading(forbid_loading, profile):
    with pytest.raises(TypeError, match="profile"):
        embedding_loader.load_cpu_embedder(profile)


@pytest.mark.parametrize("offline", [None, 0, 1, "true"])
def test_offline_flag_is_strict_boolean(forbid_loading, offline):
    with pytest.raises(TypeError, match="boolean"):
        embedding_loader.load_cpu_embedder(make_profile(), local_files_only=offline)


@pytest.mark.parametrize("revision", ["v1.0", "a" * 39, "a" * 41, "g" * 40])
def test_revision_requires_full_hex_sha(forbid_loading, revision):
    with pytest.raises(ValueError, match="commit SHA"):
        embedding_loader.load_cpu_embedder(
            replace(make_profile(), model_revision=revision)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device", "cuda", "CPU float32"),
        ("precision", "float16", "CPU float32"),
        ("query_prefix", "query: ", "empty prefixes"),
        ("document_prefix", "passage: ", "empty prefixes"),
    ],
)
def test_unsupported_settings_fail_before_loading(
    forbid_loading, field, value, message
):
    with pytest.raises(ValueError, match=message):
        embedding_loader.load_cpu_embedder(replace(make_profile(), **{field: value}))


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("device", "cuda:0", "not on CPU"),
        ("max_seq_length", 8, "sequence length"),
        ("get_embedding_dimension", lambda: 3, "dimension"),
    ],
)
def test_effective_model_settings_must_match_profile(
    monkeypatch, attribute, value, message
):
    model = FakeModel()
    setattr(model, attribute, value)
    monkeypatch.setattr(embedding_loader, "_load_model", lambda *a, **k: model)

    with pytest.raises(ValueError, match=message):
        embedding_loader.load_cpu_embedder(make_profile())


@pytest.mark.parametrize(
    "dtypes",
    [[], ["torch.float16"], ["torch.float32", "torch.float64"]],
)
def test_parameter_dtypes_checked_after_float_conversion(monkeypatch, dtypes):
    model = FakeModel()

    def parameters():
        assert model.float_called
        assert model.eval_called
        return iter(
            SimpleNamespace(dtype=dtype, is_floating_point=lambda: True)
            for dtype in dtypes
        )

    model.parameters = parameters
    monkeypatch.setattr(embedding_loader, "_load_model", lambda *a, **k: model)

    with pytest.raises(ValueError, match="float32 floating-point"):
        embedding_loader.load_cpu_embedder(make_profile())


def test_nonfloating_parameters_are_not_dtype_failures(monkeypatch):
    model = FakeModel()
    model.parameters = lambda: iter(
        [
            FakeParameter(),
            SimpleNamespace(dtype="torch.int64", is_floating_point=lambda: False),
        ]
    )
    monkeypatch.setattr(embedding_loader, "_load_model", lambda *a, **k: model)

    assert embedding_loader.load_cpu_embedder(make_profile()).model is model


def test_load_failure_propagates_without_retry(monkeypatch):
    error = OSError("Model is not cached.")
    calls = []

    def fail_load(*args, **kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(embedding_loader, "_load_model", fail_load)

    with pytest.raises(OSError) as caught:
        embedding_loader.load_cpu_embedder(make_profile())

    assert caught.value is error
    assert calls == [{"local_files_only": True}]


def test_loader_accepts_truncate_policy(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(
        embedding_loader, "_load_model", lambda *args, **kwargs: model,
    )
    profile = replace(make_profile(), truncation_policy="truncate")

    embedder = embedding_loader.load_cpu_embedder(profile)

    assert embedder.profile is profile
    assert embedder.profile.truncation_policy == "truncate"
    assert model.float_called
    assert model.eval_called