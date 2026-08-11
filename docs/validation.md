# Validation record

This document summarizes completed software validation. It does not report real
language-model capability.

> [!WARNING]
> All accuracy, token, and latency values described here are deterministic
> `MockProvider` validation metrics. They are synthetic and must not be cited as
> real LLM or MMLU-Pro model performance.

## Offline fixture validation

The offline integration path uses eight project-owned synthetic
multiple-choice questions from `data/fixtures/mcq_fixture.jsonl` and two mock
model profiles from `configs/mock_smoke.yaml`.

The validated path is:

```text
YAML config
→ strict Pydantic validation
→ local JSONL loading
→ normalized DatasetExample objects
→ versioned prompt construction
→ deterministic MockProvider responses
→ deterministic answer parsing
→ evaluation classification
→ append-friendly results.jsonl
→ summary.json and reproducibility artifacts
```

The fixture data is owned by the project and exists only to exercise software
behavior. Its scores are not benchmark results for any model.

## Pinned MMLU-Pro MockProvider smoke validation

A controlled smoke validation was completed with the official
`TIGER-Lab/MMLU-Pro` Hugging Face dataset and the deterministic MockProvider.

| Field | Validated value |
| --- | --- |
| Dataset | `TIGER-Lab/MMLU-Pro` |
| Revision | `475d58ba0cc18a15fd5d4221f41919199e692331` |
| Split | `test` |
| Seed | `42` |
| Profile | `smoke` |
| Selected samples | `14` |
| Selected categories | `14` |
| Provider | `MockProvider` |
| Real endpoint calls | None |
| API keys | None |

The selected set covered these categories:

```text
biology, business, chemistry, computer science, economics, engineering,
health, history, law, math, other, philosophy, physics, psychology
```

The generated dataset manifest recorded the source, pinned revision, split,
seed, selected sample IDs, sample-to-category mapping, category set, sample
count, license, homepage, citation, and manifest hash.

The POC and full profiles were not run as part of this validation.

## Offline automated tests

The offline test suite completed with:

```text
23 passed
0 failed
```

Coverage includes:

- Valid and invalid configuration behavior
- Unsupported schema versions and unknown fields
- Local fixture loading
- Network-free MMLU-Pro row normalization and deterministic sampling
- Prompt construction
- Deterministic MockProvider scenarios
- Parser normalization, ambiguity, and unparseable responses
- Request-failure and missing-token behavior
- Accuracy, answered accuracy, reliability, token, and percentile metrics
- JSONL persistence and summary generation
- Complete fixture-to-artifacts integration flow

## Interpretation boundary

The validation proves that the current pipeline can load and sample data,
produce deterministic provider responses, parse and classify those responses,
calculate metrics, and persist reproducibility artifacts.

It does not establish:

- MMLU-Pro performance for any real model
- Provider latency or reliability
- Real tokenizer usage
- Monetary cost
- Real endpoint compatibility
- Statistical validity of a 14-sample score

Real provider results must be recorded in separate runs with explicit provider,
endpoint alias, requested and returned model IDs, request parameters, and
measurement protocol metadata.

## Licensing note

The project does not yet include a project license. Repository licensing and
all third-party dataset obligations must be confirmed before making the
repository public.
