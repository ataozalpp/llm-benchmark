# Local model setup

This document records the manually validated LM Studio setup for the first
opt-in local-provider proof of concept. Model weights and LM Studio state must
remain outside the repository.

## Validated model

- Runtime: LM Studio
- Repository: `unsloth/Qwen3.5-0.8B-GGUF`
- Model file: `Qwen3.5-0.8B-Q8_0.gguf`
- Quantization: `Q8_0`
- Download size: `1.21 GB`
- LM Studio model identifier: `qwen3.5-0.8b`

Quantization is recorded as model provenance only. This development slice does
not compare quantization variants or measure GPU/CPU resources.

## Server settings

- Base URL: `http://127.0.0.1:1234`
- Authentication: disabled
- Serve on Local Network: disabled
- Required native endpoint: `POST /api/v1/chat`

Keep the server bound to localhost. Do not add private addresses, API keys,
model weights, or LM Studio cache/state files to the repository.

## Why the native API is used

Manual checks established that:

- `GET /v1/models` succeeded and returned `qwen3.5-0.8b`.
- OpenAI-compatible `POST /v1/chat/completions` reached the model.
- Qwen3.5 used the full output budget as reasoning tokens, returned empty
  `message.content`, and ended with `finish_reason=length`.
- The older Qwen3 `/no_think` soft switch is not officially supported by
  Qwen3.5.
- `enable_thinking=false` on the OpenAI-compatible endpoint was ineffective.
- Native `POST /api/v1/chat` with `reasoning="off"` returned
  `FINAL ANSWER: B`.

For this reason, `LMStudioProvider` is isolated from any future generic
OpenAI-compatible provider. The generic adapter is now implemented separately;
the native adapter remains necessary for explicitly controlled native reasoning.

## Request configuration

The tracked config is `configs/lm_studio_fixture_smoke.yaml`:

```yaml
provider: lm_studio
base_url: http://127.0.0.1:1234
model_id: qwen3.5-0.8b
reasoning: "off"
temperature: 0
max_output_tokens: 128
timeout_seconds: 120
```

The provider also sends `store=false`, ensuring each request is independent and
does not create persistent native chat state.

## Opt-in smoke command

After LM Studio is running with the validated model loaded:

```powershell
python -m llm_benchmark run --config configs/lm_studio_fixture_smoke.yaml
```

This config selects one project-owned fixture question and therefore makes one
real localhost request. It does not run MMLU-Pro, the POC profile, or the full
dataset.

## OpenAI-compatible fixture command

LM Studio may alternatively expose the same loaded model through its
OpenAI-compatible surface. The tracked one-question config uses:

```yaml
provider: openai_compatible
base_url: http://127.0.0.1:1234/v1
model_id: qwen3.5-0.8b
temperature: 0
timeout_seconds: 300
```

Run it only as an explicitly approved localhost validation:

```powershell
python -m llm_benchmark run --config configs/openai_compatible_fixture_smoke.yaml
```

The config sends one `POST /v1/chat/completions` request for synthetic fixture
sample `q01`. It omits `max_tokens`, provider-specific reasoning and sampling
fields, and `Authorization`. Reasoning behavior is provider-managed or
unverified (`reasoning_mode=null`), while any explicit reasoning-token usage
reported by the endpoint remains telemetry only. Native and OpenAI-compatible
providers are separate adapters and must not be treated as identical transport
contracts.

## Pinned MMLU-Pro smoke command

The separate `configs/mmlu_pro_lm_studio_smoke.yaml` profile selects the same 14
pinned, category-stratified samples previously validated with MockProvider. It
makes exactly 14 sequential localhost requests and must use the cached dataset:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
python -m llm_benchmark run --config configs/mmlu_pro_lm_studio_smoke.yaml
```

It does not run the POC or full profile and does not retry failed, incorrect, or
unparseable samples.

## Native response handling

The provider reads the native response `output` array and concatenates text
only from items with `type="message"`. Items such as `type="reasoning"` are not
sent to the multiple-choice parser and cannot become the scored answer.

When reported, the result stores input tokens, total output tokens, reasoning
output tokens, safely derived final-output tokens, reasoning-observed status,
tokens per second, time to first token, timeout, and stop reason. HTTP failures
can also record optional sanitized status/code/type/message fields. Missing
telemetry is stored as null rather than zero.

The canonical multiple-choice response is one uppercase option label as the
entire output, with the valid range derived from the question's actual options.
The strict parser keeps exact `FINAL ANSWER: <label>` support for backward
compatibility, but it does not accept approximate markers or infer an option
letter from answer text such as a city, planet, or person name. Native stop
reason is recorded only when LM Studio includes one; otherwise it remains null.

## Reasoning and sampling validation

Separate tracked configs preserve the reasoning-off baseline and define
bounded, opt-in reasoning-on calibrations. The pinned 14-sample reasoning-on
run consumed all 1,024 output tokens as reasoning for every sample and produced
no final messages. Sample `2019` repeated the same outcome with a 2,048-token
budget.

A partial native-supported sampling calibration used:

```yaml
temperature: 1.0
top_p: 0.95
top_k: 20
min_p: 0.0
repeat_penalty: 1.0
```

Native `presence_penalty` support was not verified and the field is omitted.
Unset optional sampling fields are not sent in the native request. Local LM
Studio logs showed strong repetitive, non-convergent deliberation until the
output budget was exhausted, but did not prove a literal infinite loop.

Raw reasoning text is not stored in tracked files. Reasoning remains separate
from the final `message` channel and is never passed to the parser or scorer.

## Optional output budget

`max_output_tokens` may be a positive integer, omitted, or null. Positive values
are sent to LM Studio and recorded as `output_budget_provenance=fixed`. Omitted
or null values are excluded from the native payload and recorded as
`output_budget_provenance=provider_default`. Provider-default behavior is still
context-bounded and must not be described as unlimited generation.

The validated model metadata reported a 262,144-token maximum context, while
the loaded instance used an 8,192-token context. The latter is the active
runtime configuration, but the exact output allowance remains unknown because
input and runtime overhead consume part of that context.

In the single-sample provider-default calibration (`2019`), the request used
154 input tokens, 3,591 reasoning tokens, 4 derived final-output tokens, and
3,595 total-output tokens. It reached the format-compliant label `B` after
209,883.45 ms, with 215.925 ms TTFT and 17.1926 tokens/s. The reference label
was `I`, so the result was incorrect. This one sample demonstrates operational
termination under the tested configuration; it does not prove general
termination or improved accuracy.

The follow-up three-sample operational calibration used fixed IDs `215`,
`6232`, and `2019` in that order. All three requests and parses succeeded and
all three reached final messages; one answer was correct. Total usage was
14,902 tokens, including 14,123 reasoning tokens, and total logical duration
was 904,660.36 ms. No explicit output-limit exhaustion was observed, but LM
Studio did not provide stop reasons.

This mini-calibration is intended for final-message reliability and cost
analysis. It is not an MMLU-Pro score. Its reasoning-off comparison also
changes output-budget and sampling policies, so observed differences cannot be
attributed solely to reasoning.
