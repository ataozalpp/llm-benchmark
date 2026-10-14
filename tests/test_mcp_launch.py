import pytest

from llm_benchmark.mcp_launch import resolve_calculator_launch


def make_project(tmp_path):
    root = tmp_path / "project"
    module = root / "src" / "llm_benchmark" / "demo_mcp_server.py"
    module.parent.mkdir(parents=True)
    module.write_text("# test placeholder\n", encoding="utf-8")
    return root


def test_missing_server_python_is_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_BENCHMARK_MCP_PYTHON", raising=False)

    with pytest.raises(ValueError, match="LLM_BENCHMARK_MCP_PYTHON"):
        resolve_calculator_launch(project_root=tmp_path)


def test_existing_launch_paths_are_resolved(monkeypatch, tmp_path):
    root = make_project(tmp_path)
    python = tmp_path / "server-python.exe"
    python.write_bytes(b"test placeholder")

    monkeypatch.setenv("LLM_BENCHMARK_MCP_PYTHON", str(python))

    launch = resolve_calculator_launch(project_root=root)

    assert launch.server_python == python.resolve()
    assert launch.project_root == root.resolve()


def test_missing_server_module_is_rejected(monkeypatch, tmp_path):
    python = tmp_path / "server-python.exe"
    python.write_bytes(b"test placeholder")

    monkeypatch.setenv("LLM_BENCHMARK_MCP_PYTHON", str(python))

    with pytest.raises(ValueError, match="Invalid calculator"):
        resolve_calculator_launch(project_root=tmp_path)


def test_directory_is_not_accepted_as_python(monkeypatch, tmp_path):
    root = make_project(tmp_path)
    monkeypatch.setenv("LLM_BENCHMARK_MCP_PYTHON", str(tmp_path))

    with pytest.raises(ValueError, match="Invalid calculator"):
        resolve_calculator_launch(project_root=root)