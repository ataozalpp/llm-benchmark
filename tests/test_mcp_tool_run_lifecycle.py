import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from llm_benchmark.tool_use.builtin_tools import create_example_tool_registry
from llm_benchmark.tool_use.descriptors import ToolDescriptor
from llm_benchmark.tool_use.run_config import ToolRunConfig
from llm_benchmark.tool_use.run_service import run_tool_benchmark
from llm_benchmark.tool_use.runtime import ToolDefinition, ToolRuntime


@pytest.mark.parametrize("fail_on_close", [False, True])
def test_manifest_completion_waits_for_session_close(
    monkeypatch,
    tmp_path,
    fail_on_close,
):
    pytest.importorskip("mcp")

    from llm_benchmark.mcp_integration import launch as mcp_launch
    from llm_benchmark.mcp_integration import runtime as mcp_runtime
    from llm_benchmark.tool_use.artifacts import fingerprint

    registry = create_example_tool_registry()
    registration = registry.get("calculator")
    assert registration is not None

    descriptor = ToolDescriptor(
        definition=ToolDefinition(
            name="calculator",
            description="Discovered test calculator.",
        ),
        parameters=registration.argument_model.model_json_schema(),
    )
    catalog = [
        {
            "name": "calculator",
            "description": descriptor.definition.description,
            "parameters": descriptor.parameters,
        }
    ]

    config = ToolRunConfig(
        runtime="mcp_stdio",
        mcp_profile="calculator",
        output_dir=tmp_path / "runs",
    )
    states_at_close = []

    monkeypatch.setattr(
        mcp_launch,
        "resolve_calculator_launch",
        lambda **kwargs: SimpleNamespace(
            server_python=tmp_path / "unused-python",
            project_root=tmp_path,
        ),
    )

    @asynccontextmanager
    async def fake_prepare(**kwargs):
        yield SimpleNamespace(
            descriptors=(descriptor,),
            executor=ToolRuntime(registry),
            protocol_version="test-protocol",
            catalog_snapshot=lambda: catalog,
        )

        manifest_path = next(config.output_dir.glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        states_at_close.append(manifest["status"])

        if fail_on_close:
            raise RuntimeError("private cleanup error")

    monkeypatch.setattr(
        mcp_runtime,
        "prepare_calculator_runtime",
        fake_prepare,
    )

    if fail_on_close:
        with pytest.raises(RuntimeError) as caught:
            run_tool_benchmark(config, project_root=tmp_path)
        
        assert "private cleanup error" not in str(caught.value)
    else:
        run_tool_benchmark(config, project_root=tmp_path)

    manifest_path = next(config.output_dir.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert states_at_close == ["running"]
    assert manifest["status"] == (
        "failed" if fail_on_close else "completed"
    )
    assert manifest["tool_catalog_hash"] == fingerprint(catalog)
    assert manifest["runtime_metadata"]["protocol_version"] == ("test-protocol")
    assert "private cleanup error" not in manifest_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("stage", ["preparation", "execution"])
def test_mcp_run_failure_artifacts_and_sanitization(monkeypatch, tmp_path, stage):
    pytest.importorskip("mcp")
    from llm_benchmark.mcp_integration import launch as mcp_launch
    from llm_benchmark.mcp_integration import runtime as mcp_runtime
    from llm_benchmark.tool_use.demo_provider import DemoToolProvider

    registry = create_example_tool_registry()
    registration = registry.get("calculator")
    descriptor = ToolDescriptor(
        definition=registration.definition,
        parameters=registration.argument_model.model_json_schema(),
    )
    config = ToolRunConfig(
        runtime="mcp_stdio", mcp_profile="calculator", output_dir=tmp_path / "runs"
    )
    closed = []
    monkeypatch.setattr(
        mcp_launch, "resolve_calculator_launch",
        lambda **kwargs: SimpleNamespace(server_python=tmp_path, project_root=tmp_path),
    )

    @asynccontextmanager
    async def prepare(**kwargs):
        if stage == "preparation":
            raise RuntimeError("private failure detail")
        try:
            yield SimpleNamespace(
                descriptors=(descriptor,), executor=ToolRuntime(registry),
                protocol_version="test-protocol",
                catalog_snapshot=lambda: [{
                    "name": "calculator", "description": descriptor.definition.description,
                    "parameters": descriptor.parameters,
                }],
            )
        finally:
            closed.append(True)

    def fail_provider(self, request):
        raise RuntimeError("private failure detail")

    monkeypatch.setattr(mcp_runtime, "prepare_calculator_runtime", prepare)
    monkeypatch.setattr(DemoToolProvider, "generate_tool_conversation", fail_provider)
    with pytest.raises(RuntimeError) as caught:
        run_tool_benchmark(config, project_root=tmp_path)
    assert "private failure detail" not in str(caught.value)
    assert caught.value.__suppress_context__ is True

    if stage == "preparation":
        assert not config.output_dir.exists()
    else:
        assert closed == [True]
        directory = next(config.output_dir.iterdir())
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "failed"
        assert manifest["completed_case_count"] == 0
        assert manifest["error_code"] == "tool_run_failed"
        assert not (directory / "summary.json").exists()
        for path in directory.iterdir():
            assert "private failure detail" not in path.read_text(encoding="utf-8")
