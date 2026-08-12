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
OpenAI-compatible provider.

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
output tokens, tokens per second, and time to first token. Missing telemetry is
stored as null rather than zero.

The canonical multiple-choice response is one uppercase option label as the
entire output, with the valid range derived from the question's actual options.
The strict parser keeps exact `FINAL ANSWER: <label>` support for backward
compatibility, but it does not accept approximate markers or infer an option
letter from answer text such as a city, planet, or person name. Native stop
reason is recorded only when LM Studio includes one; otherwise it remains null.
