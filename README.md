# LLM Benchmark

A small, reproducible benchmark core for multiple-choice LLM comparisons,
compatible with Python 3.12. V1 uses only the deterministic `MockProvider` and
does not make real or paid API calls.

## Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,huggingface]"
```

If Hugging Face support is not required, `python -m pip install -e ".[dev]"` is
sufficient.

## Running benchmarks

Run the fully offline local fixture benchmark:

```powershell
llm-benchmark run --config configs/mock_smoke.yaml
# or: python -m llm_benchmark run --config configs/mock_smoke.yaml
```

Run the MMLU-Pro smoke benchmark. The first run requires network access;
subsequent runs can use the Hugging Face cache:

```powershell
llm-benchmark run --config configs/mmlu_pro_smoke.yaml
```

Example CLI override:

```powershell
llm-benchmark run --config configs/mock_smoke.yaml --set output_dir=outputs/custom --set seed=7
```

Each run creates `results.jsonl`, `summary.json`, `resolved_config.json`,
`dataset_manifest.json`, and `environment.json` under `outputs/<run_id>/`.

## Tests

```powershell
pytest
```

The standard test suite does not use the network. The MMLU-Pro network test is
optional:

```powershell
pytest -m network --run-network
```

## MMLU-Pro

The real dataset source is
[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), and the
example configuration is pinned to a full commit revision for reproducibility.
The dataset is not copied into the repository; the `datasets` library cache is
used. Verify license and citation information against the upstream dataset card
before running an experiment. Smoke and POC results are not official full
benchmark scores.

## Purpose of MockProvider and fixture results

`MockProvider` exercises successful, incorrect, unparseable, failed-request,
and missing-token-usage paths deterministically without network access, API
credentials, cost, or real waiting. The bundled fixture dataset is synthetic
and exists only to verify the software pipeline. Fixture scores must not be
interpreted as real model performance.

## Limitations

V1 does not include retry execution, real provider adapters, streaming, a
pricing table, a database, or a web interface. The data model is ready to record
retry information and unknown token usage. See the documented
[assumptions](docs/assumptions.md) for details.
