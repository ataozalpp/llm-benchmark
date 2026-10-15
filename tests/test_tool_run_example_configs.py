from pathlib import Path

from llm_benchmark.tool_use.run_config import load_tool_run_config


def test_mcp_openai_example_config():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "tool_mcp_openai_compatible_smoke.yaml"

    config = load_tool_run_config(path)

    assert config.provider == "openai_compatible"
    assert config.runtime == "mcp_stdio"
    assert config.mcp_profile == "calculator"

    assert config.model is not None
    assert config.model.provider == "openai_compatible"
    assert config.model.timeout_seconds <= config.max_wall_time_seconds