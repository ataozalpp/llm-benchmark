from pathlib import Path

import pytest

from llm_benchmark.config import ModelConfig, load_config
from llm_benchmark.reproducibility import canonical_hash


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
