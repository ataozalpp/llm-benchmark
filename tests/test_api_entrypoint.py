from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_benchmark.api.__main__ import main


def test_main_starts_uvicorn_with_application_factory(monkeypatch: pytest.MonkeyPatch,) -> None:
    captured: dict[str, object] = {}

    def fake_run(target: str, **kwargs: object) -> None:
        captured["target"] = target
        captured.update(kwargs)

    monkeypatch.setattr("llm_benchmark.api.__main__.uvicorn.run", fake_run)

    main()

    assert captured == {
        "target": "llm_benchmark.api:create_app",
        "factory": True,
        "host": "127.0.0.1",
        "port": 8000,
    }


def test_importing_api_entrypoint_does_not_create_runtime_files(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("LLM_BENCHMARK_DATABASE_URL", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import llm_benchmark.api.__main__",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "runtime").exists()
    assert not list(tmp_path.glob("*.db"))
    assert not list(tmp_path.glob("*.sqlite*"))
