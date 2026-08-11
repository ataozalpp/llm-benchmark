import json
from pathlib import Path

from llm_benchmark.config import load_config
from llm_benchmark.runner import run_benchmark


def test_fixture_pipeline_end_to_end(tmp_path: Path) -> None:
    config = load_config(Path("configs/mock_smoke.yaml"), [f"output_dir={json.dumps(str(tmp_path))}"])
    run_dir, summary = run_benchmark(config)
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "dataset_manifest.json").exists()
    assert summary["overall"]["total_samples"] == 16
    assert len((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 16
    persisted = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert persisted["run_fingerprint"] == summary["run_fingerprint"]
