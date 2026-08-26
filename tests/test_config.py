import json
from pathlib import Path

import pytest

from llm_benchmark.config import (
    DatasetConfig,
    ModelConfig,
    RunConfig,
    canonical_config,
    load_config,
)
from llm_benchmark.reproducibility import canonical_hash

_CSV_DIGEST = "a" * 64
_JSONL_DIGEST = "b" * 64


def _uploaded_dataset(**overrides: object) -> DatasetConfig:
    values: dict[str, object] = {
        "source": "uploaded",
        "name": "uploaded-dataset",
        "storage_key": f"upload://sha256/{_CSV_DIGEST}.csv",
        "adapter_type": "tabular_mcq_csv_v1",
        "checksum": f"sha256:{_CSV_DIGEST}",
    }
    values.update(overrides)
    return DatasetConfig.model_validate(values)


def _run_config(dataset: DatasetConfig) -> RunConfig:
    return RunConfig(
        schema_version=1,
        experiment_name="uploaded-config-test",
        dataset=dataset,
        models=[ModelConfig(model_id="mock-model")],
    )


def test_uploaded_dataset_config_accepts_valid_provenance() -> None:
    dataset = _uploaded_dataset(license="CC0-1.0")

    assert dataset.source == "uploaded"
    assert dataset.storage_key == (
        f"upload://sha256/{_CSV_DIGEST}.csv"
    )
    assert dataset.adapter_type == "tabular_mcq_csv_v1"
    assert dataset.checksum == f"sha256:{_CSV_DIGEST}"
    assert dataset.license == "CC0-1.0"
    assert dataset.path is None


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("storage_key", "requires storage_key"),
        ("adapter_type", "requires adapter_type"),
        ("checksum", "requires checksum"),
    ],
)
def test_uploaded_dataset_config_requires_provenance_fields(
    field_name: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _uploaded_dataset(**{field_name: None})


@pytest.mark.parametrize(
    ("storage_key", "adapter_type", "checksum"),
    [
        (
            f"upload://sha256/{_CSV_DIGEST}.csv",
            "tabular_mcq_jsonl_v1",
            f"sha256:{_CSV_DIGEST}",
        ),
        (
            f"upload://sha256/{_JSONL_DIGEST}.jsonl",
            "tabular_mcq_csv_v1",
            f"sha256:{_JSONL_DIGEST}",
        ),
    ],
)
def test_uploaded_dataset_adapter_must_match_storage_format(
    storage_key: str,
    adapter_type: str,
    checksum: str,
) -> None:
    with pytest.raises(ValueError, match="does not match storage format"):
        _uploaded_dataset(
            storage_key=storage_key,
            adapter_type=adapter_type,
            checksum=checksum,
        )


def test_uploaded_dataset_checksum_must_match_storage_key() -> None:
    with pytest.raises(ValueError, match="checksum does not match storage_key"):
        _uploaded_dataset(checksum=f"sha256:{_JSONL_DIGEST}")


def test_uploaded_dataset_rejects_path() -> None:
    with pytest.raises(ValueError, match="must not define path"):
        _uploaded_dataset(path=Path("runtime/datasets/file.csv"))


def test_local_dataset_rejects_uploaded_provenance() -> None:
    with pytest.raises(ValueError, match="does not accept uploaded provenance"):
        DatasetConfig(
            source="local",
            name="local",
            path=Path("data/fixtures/questions.jsonl"),
            storage_key=f"upload://sha256/{_CSV_DIGEST}.csv",
        )


def test_huggingface_dataset_rejects_uploaded_adapter() -> None:
    with pytest.raises(ValueError, match="does not accept uploaded provenance"):
        DatasetConfig(
            source="huggingface",
            name="dataset/repository",
            revision="pinned-revision",
            adapter_type="tabular_mcq_jsonl_v1",
        )


@pytest.mark.parametrize(
    "storage_key",
    [
        "",
        "upload://sha256/abc.csv",
        f"upload://sha256/{'A' * 64}.csv",
        f"upload://sha256/../{_CSV_DIGEST}.csv",
        f"upload://sha256/{_CSV_DIGEST}.csv/../other",
        f"file://sha256/{_CSV_DIGEST}.csv",
        f"upload://sha256/{_CSV_DIGEST}.csv?path=../other",
    ],
)
def test_uploaded_dataset_rejects_invalid_storage_keys(
    storage_key: str,
) -> None:
    with pytest.raises(ValueError, match="storage_key is invalid"):
        _uploaded_dataset(storage_key=storage_key)


def test_uploaded_canonical_config_contains_only_portable_provenance() -> None:
    parsed = json.loads(
        canonical_config(
            _run_config(_uploaded_dataset(license="CC0-1.0"))
        )
    )
    dataset = parsed["dataset"]

    assert dataset["storage_key"] == (
        f"upload://sha256/{_CSV_DIGEST}.csv"
    )
    assert dataset["adapter_type"] == "tabular_mcq_csv_v1"
    assert dataset["checksum"] == f"sha256:{_CSV_DIGEST}"
    assert dataset["license"] == "CC0-1.0"
    assert dataset["path"] is None
    assert not {
        "resolved_path",
        "storage_root",
        "runtime_dataset_path",
    } & dataset.keys()
    for value in dataset.values():
        if isinstance(value, str):
            lowered = value.lower()
            assert "c:\\" not in lowered
            assert "/app/" not in lowered
            assert "/runtime/" not in lowered
            assert "runtime/datasets" not in lowered


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/mock_smoke.yaml",
        "configs/mmlu_pro_lm_studio_smoke.yaml",
    ],
)
def test_existing_dataset_canonical_mapping_omits_uploaded_fields(
    config_path: str,
) -> None:
    parsed = json.loads(canonical_config(load_config(Path(config_path))))

    assert not {
        "storage_key",
        "adapter_type",
        "checksum",
        "license",
    } & parsed["dataset"].keys()


def test_valid_config_and_cli_override() -> None:
    config = load_config(Path("configs/mock_smoke.yaml"), ["seed=7"])
    assert config.schema_version == 1
    assert config.seed == 7
    assert config.models[0].model_id == "mock-model-a"


def test_lm_studio_config_requires_native_settings() -> None:
    config = load_config(Path("configs/lm_studio_fixture_smoke.yaml"))
    model = config.models[0]
    assert model.provider == "lm_studio"
    assert model.base_url == "http://127.0.0.1:1234"
    assert model.reasoning == "off"
    assert model.max_output_tokens == 128


def test_reasoning_on_fixture_config_is_separate_and_bounded() -> None:
    config = load_config(Path("configs/lm_studio_fixture_reasoning_on.yaml"))
    model = config.models[0]
    assert config.dataset.sample_size == 1
    assert model.provider == "lm_studio"
    assert model.reasoning == "on"
    assert model.temperature == 0
    assert model.max_output_tokens == 1024
    assert model.timeout_seconds == 300


def test_mmlu_reasoning_on_config_preserves_baseline_except_expected_fields() -> None:
    baseline = load_config(Path("configs/mmlu_pro_lm_studio_smoke.yaml"))
    reasoning_on = load_config(Path("configs/mmlu_pro_lm_studio_reasoning_on_smoke.yaml"))
    assert reasoning_on.dataset == baseline.dataset
    assert reasoning_on.seed == baseline.seed == 42
    assert reasoning_on.output_dir == baseline.output_dir
    assert reasoning_on.evaluation == baseline.evaluation

    baseline_model = baseline.models[0]
    reasoning_model = reasoning_on.models[0]
    expected_model_differences = {
        "reasoning": ("off", "on"),
        "max_output_tokens": (128, 1024),
    }
    actual_model_differences = {
        field: (getattr(baseline_model, field), getattr(reasoning_model, field))
        for field in type(baseline_model).model_fields
        if getattr(baseline_model, field) != getattr(reasoning_model, field)
    }
    assert actual_model_differences == expected_model_differences
    assert baseline.experiment_name == "mmlu_pro_lm_studio_smoke"
    assert reasoning_on.experiment_name == "mmlu_pro_lm_studio_reasoning_on_smoke"


def test_reasoning_on_2048_calibration_config_selects_one_fixed_sample() -> None:
    config = load_config(Path("configs/mmlu_pro_lm_studio_reasoning_on_2048_calibration.yaml"))
    assert config.dataset.sample_ids == ["2019"]
    assert config.dataset.sample_size == 1
    model = config.models[0]
    assert model.reasoning == "on"
    assert model.max_output_tokens == 2048
    assert model.timeout_seconds == 300


def test_sampling_calibration_config_validates_supported_fields() -> None:
    config = load_config(Path("configs/mmlu_pro_lm_studio_reasoning_on_sampling_calibration.yaml"))
    model = config.models[0]
    assert model.temperature == 1.0
    assert model.top_p == 0.95
    assert model.top_k == 20
    assert model.min_p == 0.0
    assert model.repeat_penalty == 1.0
    assert "presence_penalty" not in type(model).model_fields


def test_context_bounded_calibration_uses_provider_default_output_budget() -> None:
    config = load_config(Path("configs/mmlu_pro_lm_studio_reasoning_on_context_bounded_calibration.yaml"))
    assert config.dataset.sample_ids == ["2019"]
    assert config.dataset.sample_size == 1
    model = config.models[0]
    assert model.provider == "lm_studio"
    assert model.max_output_tokens is None
    assert model.output_budget_provenance == "provider_default"
    assert model.timeout_seconds == 660
    assert config.model_dump(mode="json")["models"][0]["max_output_tokens"] is None


def test_context_bounded_mini_is_an_ordered_three_sample_operational_calibration() -> None:
    config = load_config(Path("configs/mmlu_pro_lm_studio_reasoning_on_context_bounded_mini.yaml"))
    assert config.experiment_name.endswith("mini_operational_calibration")
    assert config.dataset.sample_ids == ["215", "6232", "2019"]
    assert config.dataset.sample_size == 3
    model = config.models[0]
    assert model.max_output_tokens is None
    assert model.output_budget_provenance == "provider_default"
    assert model.timeout_seconds == 660
    assert "presence_penalty" not in type(model).model_fields
    assert "retry" not in type(config).model_fields


@pytest.mark.parametrize("value", [0, -1])
def test_max_output_tokens_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError):
        ModelConfig(model_id="mock", max_output_tokens=value)


@pytest.mark.parametrize(("field", "value"), [
    ("top_p", -0.01),
    ("top_p", 1.01),
    ("top_k", -1),
    ("min_p", -0.01),
    ("min_p", 1.01),
    ("repeat_penalty", 0),
])
def test_sampling_fields_reject_out_of_range_values(tmp_path: Path, field: str, value: float) -> None:
    text = Path("configs/lm_studio_fixture_smoke.yaml").read_text(encoding="utf-8")
    text = text.replace("    temperature: 0", f"    temperature: 0\n    {field}: {value}")
    path = tmp_path / "invalid-sampling.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="config validation failed"):
        load_config(path)


def test_unset_sampling_fields_do_not_change_existing_resolved_config() -> None:
    config = load_config(Path("configs/mmlu_pro_lm_studio_reasoning_on_2048_calibration.yaml"))
    resolved_model = config.model_dump(mode="json")["models"][0]
    assert not {"top_p", "top_k", "min_p", "repeat_penalty"} & resolved_model.keys()


def test_openai_compatible_config_requires_url_and_omits_unset_credential() -> None:
    with pytest.raises(ValueError, match="requires base_url"):
        ModelConfig(provider="openai_compatible", model_id="model")

    model = ModelConfig(
        provider="openai_compatible",
        model_id="model",
        base_url="http://127.0.0.1:1234/v1",
    )
    assert "credential_env_var" not in model.model_dump(mode="json")


def test_openai_compatible_config_stores_only_credential_reference() -> None:
    model = ModelConfig(
        provider="openai_compatible",
        model_id="model",
        base_url="https://provider.example/v1",
        credential_env_var="BENCHMARK_API_KEY",
    )
    assert model.model_dump(mode="json")["credential_env_var"] == "BENCHMARK_API_KEY"
    with pytest.raises(ValueError, match="environment-variable name"):
        ModelConfig(
            provider="openai_compatible",
            model_id="model",
            base_url="https://provider.example/v1",
            credential_env_var="not a variable",
        )


def test_openai_compatible_fixture_smoke_is_one_provider_managed_request() -> None:
    config = load_config(Path("configs/openai_compatible_fixture_smoke.yaml"))
    assert config.dataset.sample_ids == ["q01"]
    assert config.dataset.sample_size == 1
    model = config.models[0]
    assert model.provider == "openai_compatible"
    assert model.base_url == "http://127.0.0.1:1234/v1"
    assert model.model_id == "qwen3.5-0.8b"
    assert model.reasoning is None
    assert model.max_output_tokens is None
    assert model.output_budget_provenance == "provider_default"
    assert model.top_p is None
    assert model.top_k is None
    assert model.min_p is None
    assert model.repeat_penalty is None
    assert model.credential_env_var is None
    assert model.timeout_seconds == 300
    assert "retry" not in type(config).model_fields


@pytest.mark.parametrize(("config_path", "expected_hash"), [
    ("configs/mmlu_pro_lm_studio_smoke.yaml", "d007c0d070d70648d4affa994f981116ba4955dc5a69392ffb71f493c0b52ac0"),
    ("configs/mmlu_pro_lm_studio_reasoning_on_smoke.yaml", "9cb104a7c63df2df50b17f7c7c82da3f4f7582c1e086f17c1a37c28ce4419b7e"),
])
def test_existing_config_hashes_remain_stable(config_path: str, expected_hash: str) -> None:
    config = load_config(Path(config_path))
    assert canonical_hash(config.model_dump(mode="json")) == expected_hash


@pytest.mark.parametrize("text", [
    "schema_version: 2\nexperiment_name: x\ndataset: {}\nmodels: []\n",
    "schema_version: 1\nexperiment_name: x\nunknown: true\ndataset: {}\nmodels: []\n",
])
def test_invalid_config_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="config validation failed"):
        load_config(path)
