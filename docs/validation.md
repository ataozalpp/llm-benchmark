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
34 passed
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

## Manual LM Studio endpoint validation

The local runtime was manually validated before implementing the provider:

| Field | Validated value |
| --- | --- |
| Runtime | LM Studio |
| Model repository | `unsloth/Qwen3.5-0.8B-GGUF` |
| Model file | `Qwen3.5-0.8B-Q8_0.gguf` |
| Quantization | `Q8_0` |
| Model ID | `qwen3.5-0.8b` |
| Server | `http://127.0.0.1:1234` |
| Authentication | Disabled |
| Local-network serving | Disabled |

`GET /v1/models` returned the expected model. OpenAI-compatible chat reached
the model, but it consumed the output budget as reasoning tokens and returned
empty message content. LM Studio native `POST /api/v1/chat` with
`reasoning="off"` returned `FINAL ANSWER: B`; therefore the first local POC uses
the native API and records reasoning mode explicitly.

The automated LM Studio tests use mocked HTTP responses and require neither LM
Studio nor a downloaded model. A real localhost fixture smoke remains opt-in.

### One-request fixture smoke result

After the offline suite passed, one explicitly approved localhost fixture
request was made with the tracked config. The native request succeeded and
reported:

- Input tokens: `75`
- Total output tokens: `64`
- Reasoning output tokens: `0`
- Tokens per second: approximately `16.09`
- Time to first token: approximately `944.09 ms`
- Logical request latency: approximately `5226.17 ms`

The message output contained an explanation but exhausted the configured
64-token output budget before emitting the required `FINAL ANSWER: <letter>`
line. The deterministic parser therefore marked the response `unparseable`.
This confirms native message extraction and telemetry capture, but it is not a
successful answer-format validation and is not evidence of model accuracy.

The corrective prompt contract now requires the marker on the first output
line and permits explanations to be omitted. The localhost fixture safety
margin is 128 output tokens. The parser remains strict: semantic option text
without `FINAL ANSWER: <letter>` is still unparseable. Stop reason is optional
because the native response may not provide it.

After 30 offline tests passed, one approved localhost request was made with the
corrected prompt and 128-token limit. The request succeeded and returned five
output tokens, but its exact text was `FINAL ANSW: B`. Because the required
marker is exactly `FINAL ANSWER: B`, the parser correctly retained
`parse_status=no_answer_found`. LM Studio did not provide a stop reason, so the
recorded `stop_reason` is null. No heuristic alias for the truncated marker was
added.

Following the truncated-marker result, the canonical contract was narrowed
again: the entire response must now be exactly one uppercase option label from
the question's actual allowed labels. Exact `FINAL ANSWER: <label>` remains
accepted only for backward compatibility. Approximate markers, fuzzy matching,
semantic answer-text mapping, and arbitrary letters in prose remain rejected.

After 34 offline tests passed, one final approved localhost request was made
with the standalone-label contract. The exact native message was `B`; it was
parsed with `parse_status=normalized_label` and matched `correct_answer=B`.
The request used 89 input tokens and 2 output tokens, reported zero reasoning
tokens, approximately `30.59` tokens per second, `691.47 ms` time to first
token, and `995.72 ms` logical latency. LM Studio did not provide a stop reason,
so `stop_reason` remained null. No further request was made.

## Pinned MMLU-Pro LM Studio smoke result

A real-model smoke run was completed through the LM Studio native provider with
the cached official dataset and no external API calls.

| Field | Value |
| --- | --- |
| Dataset | `TIGER-Lab/MMLU-Pro` |
| Revision | `475d58ba0cc18a15fd5d4221f41919199e692331` |
| Split | `test` |
| Seed | `42` |
| Profile | `smoke` |
| Requests | `14`, sequential |
| Model | `qwen3.5-0.8b` |
| Reasoning | `off` |
| Temperature | `0` |
| Maximum output tokens | `128` |

Aggregate results:

| Metric | Value |
| --- | ---: |
| Total | 14 |
| Correct | 2 |
| Incorrect | 12 |
| Unparseable | 0 |
| Failed | 0 |
| Accuracy | 14.29% |
| Answered accuracy | 14.29% |
| Request success rate | 100% |
| Parse success rate | 100% |
| Format failure rate | 0% |
| Input tokens | 4,298 |
| Output tokens | 28 |
| Reasoning output tokens | 0 |
| Total tokens | 4,326 |
| Latency P50 | 1,496.08 ms |
| Latency P95 | 2,363.42 ms |
| Sum of logical-request durations | 22,121.40 ms |
| Run wall time | 22,136.70 ms |

Per-category outcomes:

| Category | Sample ID | Correct | Parsed | Result |
| --- | --- | --- | --- | --- |
| Biology | 3463 | C | A | Incorrect |
| Business | 215 | A | A | Correct |
| Chemistry | 3577 | F | C | Incorrect |
| Computer science | 10743 | H | A | Incorrect |
| Economics | 7111 | F | B | Incorrect |
| Engineering | 11537 | D | A | Incorrect |
| Health | 6232 | C | C | Correct |
| History | 4740 | C | A | Incorrect |
| Law | 1213 | E | A | Incorrect |
| Math | 8808 | C | A | Incorrect |
| Other | 5149 | B | C | Incorrect |
| Philosophy | 11083 | E | A | Incorrect |
| Physics | 9560 | I | A | Incorrect |
| Psychology | 2019 | I | B | Incorrect |

Selected sample IDs:

```text
10743, 11083, 11537, 1213, 2019, 215, 3463,
3577, 4740, 5149, 6232, 7111, 8808, 9560
```

Artifacts were generated under the ignored run directory
`outputs/20260812T130136Z-ff5ca994/`. This 14-question result validates the real
local execution path and strict output-format compliance; it is not a
statistically sufficient MMLU-Pro score.

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
