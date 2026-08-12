from pathlib import Path

import pytest

from llm_benchmark.config import load_config


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


@pytest.mark.parametrize("text", [
    "schema_version: 2\nexperiment_name: x\ndataset: {}\nmodels: []\n",
    "schema_version: 1\nexperiment_name: x\nunknown: true\ndataset: {}\nmodels: []\n",
])
def test_invalid_config_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="config validation failed"):
        load_config(path)
