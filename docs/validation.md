# Validation record

This document summarizes completed software and local operational validation.
The real-model results are small diagnostic smoke runs, not statistically
sufficient model benchmarks.

## Package-layout refactor checkpoint (2026-10-15)

The working tree on `refactor/packages-layout` was validated after moving tool
modules into `tool_use` (including `tool_use.external`) and MCP modules into
`mcp_integration`. Import references and test patch targets were updated; CLI
commands, demo entry points, policies, and artifact contracts were preserved.

- Full regression with the real MCP server configured: **1672 passed, 7 skipped
  in 66.80 seconds**. Five skips require PostgreSQL configuration; two require
  filesystem symlink support.
- Ruff passed; `git diff --check` passed.
- Synthetic MCP CLI run `6fe609c3b6d24266a6325a283a0ed003` completed both cases
  with one successful tool call and all-case final-answer accuracy 2/2.
- No real-model requests were made for this refactor.

The first full-test attempt had one MCP `initialize` timeout while a separate
CLI smoke was also running; that smoke failed during preparation too. The full
suite and smoke both passed when rerun sequentially without changing timeouts.
Concurrent startup contention is a possible explanation, not a proven cause.
The existing startup-timeout sensitivity remains a limitation.

## Real-model MCP diagnostic repeats (2026-10-15)

Three user-executed LM Studio runs were reviewed from their manifests, summaries,
and per-case results. Dates below use Europe/Istanbul; artifact timestamps use
UTC and fall on 2026-10-14. The current checkout is based on `91156d7`; the
artifacts do not record a Git revision, so this is not independently verified
run provenance.

The supplied configuration selects `qwen3.5-0.8b`, OpenAI-compatible inference,
temperature 0, a 2,048-token output budget, a 120-second provider timeout, and a
300-second scenario budget. All three artifacts share the same config, suite,
catalog, runtime-metadata, and run fingerprints. These fingerprints do not
capture LM Studio version, model-file revision/quantization, context length, or
server-side thinking settings; those settings were not independently verified.

| Run ID | Manifest | Calculator | Direct answer | Execution interval |
| --- | --- | --- | --- | ---: |
| `8191967720614742bef9287115ca159b` | completed | passed | provider_turn_limit | 16.359 s |
| `839345e5eb564afd86dc4fb5d6a50b00` | completed | passed | provider_turn_limit | 9.218 s |
| `4dc76b95fd8e450185cb895d8798a683` | completed | passed | provider_turn_limit | 8.297 s |

Every run used `mcp_stdio`, `mcp-calculator-v1`, and negotiated protocol
`2025-11-25`, with `synthetic: false`. The calculator case made one successful
MCP call and produced the expected final answer on the second model turn.
The direct-answer case requested a tool despite the instruction to return
`READY` without tools. Its single provider turn was exhausted, so the loop
stopped before executing that request. This is an evaluated behavior failure,
not a failed MCP transport or incomplete run lifecycle.

Each run recorded completion 1/2, tool-sequence match 1/2, conditional argument
match 1/1, conditional final-answer match 1/1, and all-case final-answer accuracy
1/2. Requested/executed/successful tool-call counts were 2/1/1. Token coverage
was 3/3 model turns: 1,225 input, 118 output, and 1,343 total tokens per run.
The endpoint reported zero reasoning tokens for every turn; this does not
independently establish the server's reasoning configuration.

The execution interval excludes MCP preparation and final session cleanup.
Timing differences do not establish a cold-start cause. Raw arguments for the
unexecuted call are absent from the artifacts. Three repeats of two scenarios
are diagnostic evidence, not a statistically sufficient model-quality study.
The repeated failure is specific to this model/server/configuration setup;
it does not isolate model weights from chat-template or serving behavior.
Run artifacts remain ignored by Git. No scenario limits or expected outcomes
were relaxed to make the model pass.

## MCP tool-run artifact checkpoint (2026-10-13)

The current working tree was validated with the real server Python configured:
**1671 passed, 7 skipped in 72.32 seconds**. Five skips require the PostgreSQL
test URL; two require filesystem symlink support. Ruff passed. The final
`git diff --check` passed with only Git line-ending conversion warnings.

The initial two lifecycle failures were reproduced. The fake runtime exposed
`catalog_snaphsot` instead of `catalog_snapshot`, so preparation failed before
manifest creation. Correcting it restored both lifecycle paths. The runtime
version typo was also aligned to `mcp-calculator-v1`, and sanitized local-run
exceptions again suppress the original traceback context.

Additional regressions cover preparation failures without run artifacts,
execution failures with a failed manifest, omission of raw failure details,
and local execution without MCP imports. Existing lifecycle assertions verify
that the manifest remains running during session closure and becomes completed
only after successful closure; cleanup failure instead produces failed status.

The actual CLI command was also executed:

```powershell
$env:LLM_BENCHMARK_MCP_PYTHON = (Resolve-Path .\.venv-mcp\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m llm_benchmark tool-run --config configs/tool_mcp_smoke.yaml
```

Local diagnostic run `5b2d5bacf7b14944b76eccaae6b8f382` completed two synthetic
scenarios, one successful tool call, two result records, and twelve trace
records. Its manifest recorded `mcp_stdio`, `mcp-calculator-v1`, and negotiated
protocol `2025-11-25`. Artifacts were generated under `outputs/tool/` and are
not source files to commit. This is software validation, not model quality.

Preparation/discovery failures before run creation have no manifest. A summary
can exist for a run whose session cleanup failed, so consumers must check
manifest status. `wall_time_ms` excludes launch/discovery and final session
cleanup. Forced termination and cleanup of a live, unresponsive child remain
outside this validation.

## MCP scenario-budget checkpoint (2026-10-10)

### Follow-up: deadline and real server-exit coverage

After completing the deadline-boundary test and adding the test-only disconnect
server, full regression passed: **1649 passed, 7 skipped in 59.95 seconds**.
The five PostgreSQL and two filesystem-symlink skips have the same environment
causes described below. `LLM_BENCHMARK_MCP_PYTHON` was configured; the real
server-exit test ran and passed. Ruff and the whole-tree `git diff --check`
also passed (Git emitted only line-ending conversion warnings).

The review corrected fixture import spacing, a disconnect call-ID typo, and
the legacy executor test name. The explicit post-deadline timeout test verifies
`WALL_TIME_LIMIT` without a fabricated tool result. The server-exit test requires
`MCPTransportFailure`; a generic timeout cannot satisfy it. Cleanup of a live,
unresponsive child process remains outside this evidence, and MCP `tool-run`
runtime selection remains unimplemented. These results supersede the earlier
open test/whitespace items below; the earlier record is retained as history.

### Earlier working-tree validation

Validation was run against the working tree containing the scenario-budget
changes, not only the last committed revision. The server Python environment
variable was configured so real MCP stdio tests ran.

- Full regression: **1647 passed, 7 skipped** in 102.61 seconds.
- Skips: five PostgreSQL tests without `LLM_BENCHMARK_TEST_POSTGRES_URL` and
  two filesystem-symlink tests unavailable in the Windows environment.
- `python -m ruff check .`: passed.
- Documentation diff check: passed. The whole-tree `git diff --check` still
  reports trailing whitespace in `tool_use/external/executor.py` and
  `tool_use/execution.py`, plus an extra blank line at the end of the latter.

Coverage includes remaining-budget forwarding, validation-time subtraction,
expired-budget rejection before delegation, early-timeout propagation, and
legacy executor compatibility. An explicit loop test for a tool timeout after
scenario expiry is still missing. Real server-disconnection/process-cleanup
coverage and MCP `tool-run` runtime selection remain pending. See
[MCP integration](mcp.md) for the current behavior and limits.

These results validate synthetic software behavior, not model quality or a
hard process-termination deadline.

> [!WARNING]
> `MockProvider` values are synthetic pipeline metrics. LM Studio values are
> real local-run telemetry for the stated model/runtime/configuration only and
> must not be generalized into a full MMLU-Pro capability claim.

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

### Generic evaluation task-adapter Phase 1 checkpoint

Focused adapter and runner tests verify that the default
`MultipleChoiceTaskAdapter` reproduces the existing prompt and prompt hash,
strict parsing across the actual option-label range, and the established
correct, incorrect, ambiguous, unparseable, and request-failed outcomes.
Runner-injection tests also verify that an injected adapter controls prompt
construction, evaluation, and prompt-hash provenance while provider calls,
telemetry, artifacts, and metric aggregation remain runner responsibilities.

This checkpoint preserves existing CLI, Run API, worker, provider,
configuration-hash, artifact, comparison, and persistence contracts. It
validates a task boundary, not support for additional task families; multiple
choice remains the only implemented adapter. RAG, tools, MCP, and agent loops
were not introduced.

### Execution-trace Phase 2 checkpoint

Focused tests verify immutable typed events, deterministic ordering from one,
a fresh recorder per execution, Windows/POSIX path redaction, UTF-8 JSONL
temporary-file replacement, and exclusion of prompt, response, reasoning,
credential, Authorization, provider-error-message, and sensitive sentinel
content. Directly constructed events and injected recorders are passed through
the same centralized exact-allowlist serializer. Tests cover bounded strings,
path-like values, and recognizable Authorization, Bearer, API-key, token,
password, and secret assignment patterns; this does not claim detection of
arbitrary unlabelled secrets. Tests also verify that trace recording or
persistence failure neither changes successful benchmark results nor masks
provider/task-adapter exceptions. Normalized request failures remain benchmark
outcomes.

Compatibility tests preserve results, summaries, config and manifest hashes,
prompt hashes, and run fingerprints. No result/summary schema, CLI, Run API,
worker, repository, or database-schema change was introduced. Successful
pipeline traces are diagnostic artifact-only `trace.jsonl` files; events are
buffered in memory, unexpected failures do not finalize error events, and a
missing trace is not separately reported in summary/DB. No trace API, database
table, streaming recorder, tools, RAG, MCP, or agent loop was added.

## Tool-runtime validation

`tests/test_tool_runtime.py` exercises the standalone runtime with in-process
test handlers. Its coverage includes:

- Successful execution and normalized unknown-tool outcomes.
- Missing, extra, and incorrectly typed arguments rejected before handler execution.
- Tool-name, description, and call-ID validation, including length boundaries.
- Strict JSON input/output rejection and frozen nested snapshots that remain
  unchanged when source objects or returned copies are mutated.
- Strict Pydantic registration, duplicate rejection, and sorted registry listing.
- Fixed handler-error outcomes without sensitive exception details and
  propagation of a custom `BaseException` subclass.
- Closed `ToolErrorCode` validation: arbitrary strings and a mismatched
  status/code pair are rejected; execution tests cover the emitted outcome pairs.
- Oversized-output rejection, UTF-8 byte boundaries, and invalid output limits.
- Import in a subprocess with a temporary working directory, verifying no
  runtime/output directories or SQLite database files are created there.

From the activated project environment, run focused tests and the complete
suite with Hugging Face and Transformers offline modes forced:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest tests/test_tool_runtime.py -q
python -m pytest -q
ruff check .
```

The previously reported tool-runtime checkpoint was `67 passed` for focused
tests and `516 passed, 7 skipped` for the complete forced-offline suite; Ruff
passed. Two skips were platform-dependent symlink tests, and five were
PostgreSQL tests whose test URL was not configured. These are prior reported
results, not a new test run performed for this documentation update, and do
not establish PostgreSQL validation for this checkpoint.

Coverage does not exhaust every direct-constructor status/code combination or
explicitly test cyclic containers. The import test checks local filesystem
side effects, not a general network sandbox. Registration checks the root
Pydantic config; it does not audit nested models or field-level overrides.
Call-ID syntax validation cannot identify every secret. Trusted handlers are
not sandboxed, and the serialized-output limit does not bound their execution
time or peak memory. These runtime tests do not validate provider integration,
bounded agent loops, or tool-call evaluation. The separate initial provider
operation, bounded loop, and tool-loop evaluation are covered separately below.

### Deterministic example-tool coverage

`tests/test_tools.py` calls the registered examples through `ToolRuntime`.
It covers all four calculator operations, negative and zero operands, selected
range boundaries, rejected string/bool/fractional inputs, unsupported
operations, missing/extra fields, and normalized division-by-zero failure.
Lookup tests cover all three synthetic entries, successful not-found results,
and rejected missing, incorrectly typed, empty, uppercase, spaced, oversized,
or extra-field arguments. Further tests verify repeated calculator results,
sorted tool names, and independent registry membership after adding a tool to
only one factory-created registry.

With the offline environment variables above set, run both focused files:

```powershell
python -m pytest tests/test_tools.py tests/test_tool_runtime.py -q
```

The 2026-09-08 example-tool checkpoint completed with `96 passed` across
these two focused files and `545 passed, 7 skipped` in the complete forced-offline
suite. Ruff passed. Two skips were platform-dependent symlink tests and five
were PostgreSQL tests with no configured test URL; PostgreSQL was not exercised
at this checkpoint.

These are software-behavior tests, not measurements of LLM tool selection or
answer quality. They do not exhaust both operands' boundary combinations or
the city-key grammar, and do not include a separate subprocess import test for
`tool_use/builtin_tools.py`; the existing import test covers `tool_use/runtime.py` only. No general
handler timeout or sandbox guarantee is established.

## Tool-call normalization validation

`tests/test_tool_calling.py` supplies in-memory response dictionaries to
`normalize_openai_tool_response()`; it makes no HTTP or model request and does
not execute tools. Coverage includes:

- Text-only, calls-only, and mixed turns; call order and exact accepted IDs;
  null/blank content with valid calls; finish reasons preserved without inference.
- Invalid envelopes, multiple choices, wrong roles and field types, missing
  required call fields, unsupported tool types, and duplicate call IDs.
- All-or-nothing rejection when one call is malformed, even with valid text.
- JSON-object-only arguments, nested duplicate keys, nonstandard constants,
  non-finite numbers, invalid Unicode, and deeply nested invalid input.
- Exact UTF-8 boundaries, whitespace and escape-sequence accounting, default
  budgets, independent per-call/aggregate limits, and invalid limit settings.
- Frozen turns and independent nested snapshots after source or returned-copy
  mutation; direct-constructor contract enforcement.
- Closed safe errors, sensitive-sentinel exclusion from standard error
  rendering, absence of chained parser exceptions, and payload-free turn repr.
- Propagation of injected unexpected programming errors and a `BaseException`
  subclass, without broad error normalization.
- Subprocess import and normalization in a temporary working directory with
  bytecode writes disabled, audit checks rejecting file writes, directory
  creation, socket and process operations, and guards rejecting registry/runtime
  initialization or execution. The test also checks that unrelated provider,
  runner, trace, API, worker, database, and example-tool modules are not imported.

Run focused checks from the project environment:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest tests/test_tool_calling.py tests/test_tool_runtime.py tests/test_openai_compatible_provider.py -q
python -m pytest -q
ruff check .
git diff --check
```

The previously reported normalization checkpoint was `242 passed` across those
three focused files, including 155 normalization cases, and `700 passed, 7
skipped` in the complete forced-offline suite. Ruff and whitespace checks
passed. Two skips were platform-dependent symlink tests; five were PostgreSQL
tests with no configured test URL. These are prior execution results, not a
fresh test run for this documentation update or a PostgreSQL validation claim.

These tests validate a pure supplied-response contract, not endpoint
interoperability, tool-selection quality, or a benchmark score. The subprocess
guards are regression checks, not a general sandbox. Argument budgets do not
bound the whole response or execution time; successful payloads are not
secret-redacted. Registry/schema resolution, actual provider requests,
streaming, tool execution, agent loops, and tool-call evaluation remain outside
this slice. See [Architecture](architecture.md#tool-call-response-normalization)
for the exact contract and closed error codes.

## Initial tool-request and provider validation

`tests/test_tool_requests.py` covers exact payload fields, explicit tool
selection, deterministic order, Pydantic argument constraints and nested schema
references, independent schema/payload copies, frozen requests, UTF-8 messages,
optional system text, output-budget omission, and rejection of unsupported
generation settings. It also checks malformed/cyclic schema handling, preserved
trusted-hook failures, no credential reads, and import/build side effects in a
temporary-directory subprocess. These checks are not a general schema sandbox.

`tests/test_tool_provider_models.py` covers closed enum/status combinations,
frozen results and telemetry, invalid token/timing types and values, and nullable
measurements. `tests/test_openai_compatible_tool_provider.py` uses only an
injected fake transport. It verifies:

- Exact URL, payload, timeout, Authorization omission/injection, and one
  transport invocation per attempted operation, with no automatic retry.
- Request-time credential reads; missing credentials cause no transport call.
- Text-only, calls-only, and mixed responses; call IDs, arguments, and finish
  reasons preserved by the existing normalizer.
- HTTP 400/401/403/429/5xx, timeout, wrapped timeout, network and JSON/Unicode
  decoding failure codes; provider error bodies are not read or returned.
- Invalid responses and all-or-nothing call rejection, separate from transport
  failures; reported usage can survive normalization rejection.
- Explicit usage aliases, nullable invalid/missing usage, no synthesized total
  tokens, and null non-streaming TTFT/throughput.
- Safe result serialization/repr and credential exclusion from configs/payloads;
  request validation before transport and propagation of unexpected exceptions.
- No runtime execution and preserved classic `generate()` empty-text rejection.

With the forced-offline environment variables described above set:

```powershell
python -m pytest tests/test_tool_requests.py tests/test_tool_provider_models.py tests/test_openai_compatible_tool_provider.py tests/test_openai_compatible_provider.py -q
python -m pytest -q
ruff check .
git diff --check
```

The previously executed provider checkpoint passed 150 tests across the result
models, new tool-provider tests, and existing OpenAI-compatible provider tests.
The complete forced-offline suite passed `912 passed, 7 skipped`; Ruff and
whitespace checks passed. Two skips were platform-dependent symlink tests and
five were PostgreSQL tests with no configured test URL. These are previous
execution results, not fresh test executions for this documentation update.
The focused command above additionally includes request-mapping tests and is
therefore broader than the 150-test checkpoint command.

No real model, localhost endpoint, external service, or PostgreSQL database was
validated by that checkpoint. These are software-contract checks, not an
interoperability result or model-quality score. The tests do not establish
bounded agent execution, schema dialect support at an endpoint, or total
request/response memory limits. See
[Architecture](architecture.md#initial-tool-turn-provider-boundary) for the
execution boundary, telemetry semantics, and remaining limitations.

## Bounded tool-loop validation

`tests/test_tool_conversation.py` covers immutable message histories, pending
calls, conversation-wide unique IDs, result ordering/matching, and independent
serialization. Request and fake-transport provider tests cover ready-history
payloads and `generate_tool_conversation()` without contacting an endpoint.

`tests/test_tool_loop.py` uses a scripted in-process provider and synthetic
handlers. It covers:

- Positive integer policies, frozen result contracts, and final-text consistency.
- Direct final responses and tool-call/result/final round trips; non-final finish
  reasons yield `completion_unconfirmed`, not confirmed task completion.
- Provider failures before or after tool execution, preserved prior outcomes,
  and no automatic retry.
- Whole-batch rejection of unselected tools or insufficient tool budget; exact
  budget success and no new tool execution on the last provider turn.
- Ordered multi-call results, duplicate IDs across turns, unchanged input/history
  snapshots, and independent execution state.
- Invalid arguments without handler execution, normalized handler errors sent
  back to the provider, and propagation of unexpected provider errors and
  `BaseException` from providers or handlers.
- Invalid setup without execution and invalid provider return-type rejection.
- Subprocess import checks rejecting writes, directory creation, socket/process
  operations, runtime initialization, and unrelated execution-layer imports.
  Ordinary Python import reads are permitted; this is not a sandbox guarantee.

From the existing project environment:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
Remove-Item Env:LLM_BENCHMARK_TEST_POSTGRES_URL -ErrorAction SilentlyContinue
python -m pytest tests/test_tool_loop.py -q
python -m pytest tests/test_tool_loop.py tests/test_tool_conversation.py tests/test_tool_requests.py tests/test_tool_provider_models.py tests/test_openai_compatible_tool_provider.py -q
python -m pytest -q
ruff check .
git diff --check
```

The previously executed loop checkpoint included all 55 loop cases in the
300-test related package and completed the full forced-offline suite with
`1000 passed, 7 skipped`; Ruff and whitespace checks passed. Two skips were
platform-dependent symlink tests and five were unconfigured PostgreSQL tests.
These are prior results, not a fresh execution for this documentation update.

The loop and provider transport are tested at separate boundaries; this does
not establish real-model interoperability, tool-selection quality, or task
accuracy. No hard handler deadline, side-effect rollback, general secret
redaction, or total memory budget is established. A returned result may retain
pending calls on a budget/authorization stop. Tool-call evaluation and
runner/API/worker/trace/database integration remain outside this slice.
See [Bounded tool loop](architecture.md#bounded-tool-loop) for stop semantics.

## Tool-loop evaluation validation

`tests/test_tool_evaluation.py` validates the pure evaluator with synthetic
expectations and loop outcomes. Coverage includes:

- Wrong, missing, extra, or reordered tools; different call IDs do not alter
  matching. Regression tests ensure expected names come from the case, not
  from the observed calls themselves.
- Strict JSON argument equality, nested values, object-key order independence,
  ordered lists, missing versus null fields, Unicode, and boolean/integer/float/
  string distinctions. Expectations reuse runtime JSON validation.
- Exact final text without trimming; nullable comparisons for inapplicable
  argument or final-answer evaluation.
- Strict case/result contracts, frozen snapshots, boolean types, invalid
  counts, and inconsistent completion/match combinations.
- Real `run_tool_loop()` integration with scripted providers and successful,
  invalid-argument, and failing synthetic handlers. Evaluation does not execute
  providers or tools again and produces deterministic results.
- Every non-final loop stop reason, requested-but-unexecuted calls, and exclusion
  of supplied prior history from current-invocation counts.
- Subprocess import/evaluation guards against writes, directory creation,
  socket/process operations, runtime initialization, and unrelated execution
  layer imports. These are regression checks, not a general sandbox.

With the forced-offline variables set and the PostgreSQL test URL unset as in
the loop commands above:

```powershell
python -m pytest tests/test_tool_evaluation.py tests/test_tool_loop.py -q
python -m pytest -q
ruff check .
git diff --check
```

The previously executed evaluation checkpoint passed `131` evaluation/loop
tests (including 76 evaluation cases) and `1076 passed, 7 skipped` in the full
forced-offline suite. Ruff and whitespace checks passed. Two skips were
platform-dependent symlink tests and five were unconfigured PostgreSQL tests.
These are prior results, not fresh test runs for this documentation update.

No real model or PostgreSQL connection was validated. This checks measurement
logic, not actual model quality. Exact ordered plans and exact text are the
only matching policies; semantic judging and benchmark pipeline/persistence
integration are not implemented. Standalone aggregation is covered below. See
[Tool-loop evaluation](architecture.md#tool-loop-evaluation) for field semantics.

## Tool descriptor validation

`tests/test_tool_descriptors.py` covers frozen fields, input/output snapshot
isolation, deterministic key ordering, strict JSON values, invalid root and
contract types, UTF-8 keys/values, finite numbers, cyclic inputs, inclusive
byte/depth boundaries, and multi-byte size accounting. Conversion tests compare
both example tools with the unchanged request serializer, verify that handlers
are not executed, and preserve trusted schema-hook exceptions. A subprocess
guard checks imports and reference serialization without writes, sockets,
process execution, or runtime initialization. This is regression coverage, not
a sandbox guarantee.

```powershell
python -m pytest tests/test_tool_descriptors.py tests/test_tool_requests.py -q
```

No real endpoint, full JSON Schema validator, or MCP transport is validated.
Reference strings are only retained as data. The
byte bound applies after serialization, not to transport or peak memory. See
[Tool descriptors](architecture.md#tool-descriptors).

The descriptor checkpoint passed 136 focused descriptor/request tests and
`1264 passed, 7 skipped` in the full forced-offline suite. Two skips were
platform-dependent symlink tests and five were PostgreSQL tests without a
configured test URL. Ruff and whitespace checks passed. This does not establish
real-model or PostgreSQL interoperability for this checkpoint.

## External tool admission validation

Focused offline command:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest tests/test_external_tool_policy.py tests/test_external_tool_schema.py tests/test_external_tool_executor.py tests/test_external_tool_suite.py tests/test_tool_suite.py -q
```

Policy tests cover strict contract types and positive limits, frozen policy,
tool-name boundaries, empty and duplicate allowlists, case-sensitive matching,
whole-catalog rejection, inclusive count/aggregate UTF-8 byte limits (including
multibyte text), immutable schema snapshots, deterministic sorting, and closed
error codes with fixed messages. Invalid JSON Schema keyword semantics are
deliberately not rejected by this gate; references remain data, not retrieval
instructions.

Suite-boundary tests cover rejected catalogs without provider/executor factory
calls, including unselected disallowed tools; missing tools in later scenarios
before any factory call; a scripted calculator execution and evaluation;
direct-suite result/summary parity; selected-only requests; executor-before-
provider creation; fresh test-factory instances; repeated-run input preservation;
and unchanged propagation of runtime errors, `KeyboardInterrupt`, and
`SystemExit` from factories and execution. Fresh subprocess checks guard import
against filesystem writes, socket/process activity, and runtime initialization.

Tests use local synthetic tools and scripted providers, not real model or MCP
requests. They do not establish full JSON Schema support, arbitrary executor
safety, resource cleanup, sandboxing, or production interoperability.
The full-suite offline command and optional PostgreSQL test configuration are
documented separately; no live PostgreSQL validation is implied by this slice.

Schema tests cover supported and malformed schemas, rejected keywords at every
supported schema position, boolean nested schemas, object/array constraints,
numeric boundaries, standard integer semantics, no coercion/default insertion,
data-versus-schema positions, immutable snapshots, identity checks, and fixed
safe errors. Suite tests reject invalid selected and unselected schemas before
factories run. Reference-rejection checks run under subprocess I/O guards.
The local city schema's regex is explicitly rejected by the external profile;
local tool behavior is not changed to force external compatibility.

Executor tests cover setup contracts, invalid arguments and unknown tools with
zero delegate calls, one exact delegate call for valid arguments, unchanged
normalized delegate failures, invalid result types/identity, and unexpected
exception/BaseException propagation. Integration tests prove that an argument
rejection reaches the next provider turn as a tool result without delegate
execution. Existing execution-attempt counts still include that rejection;
successful-call counts do not. Import guards also exercise wrapper construction
and synthetic execution without network, subprocess, or filesystem-write I/O.

## Descriptor request and loop integration validation

Request tests cover mutually exclusive sources, strict collection types,
duplicate names, readiness, frozen snapshots, hidden repr fields, legacy
positional constructors, payload parity, and unchanged local schema-size policy.
Descriptor-only serialization is included in subprocess no-I/O checks.

Executor tests cover early rejection without an executor, preservation of
descriptor selection across turns, no local runtime initialization, and both
request sources for authorization/budget checks, malformed executor results,
normalized failures, and exception propagation. Provider tests use an injected
fake transport to check exact initial/conversation payloads, URL, timeout, and
omitted authorization without executing tools or making HTTP requests.

```powershell
python -m pytest tests/test_tool_execution.py tests/test_tool_requests.py tests/test_tool_descriptors.py tests/test_tool_loop.py tests/test_tool_suite.py tests/test_openai_compatible_tool_provider.py -q
```

At this integration checkpoint, the suite used local registry-based preparation;
descriptor-suite coverage is described below. MCP transport, external-schema
semantic validation, executor isolation, and real-model interoperability remain
outside this slice.

The integration checkpoint passed 334 focused tests and `1319 passed, 7 skipped`
in the full forced-offline suite. The skips were two platform-dependent symlink
tests and five PostgreSQL tests without a configured test URL. Ruff and
whitespace checks passed. All provider transport checks were mocked.

## Tool executor validation

`tests/test_tool_execution.py` checks the existing runtime's executor contract,
default/injected local result parity, and avoidance of unused runtime
initialization. It verifies invalid executor rejection before provider calls,
returned result types and call identity, whole-batch authorization and budget
checks before execution, normalized failures continuing through the conversation,
and propagation of unexpected exceptions including `KeyboardInterrupt` and
`SystemExit`. Existing loop import-side-effect tests also run with this boundary.

```powershell
python -m pytest tests/test_tool_execution.py tests/test_tool_loop.py tests/test_tool_suite.py -q
```

These tests use scripted providers and local synthetic tools. They do not prove
that arbitrary injected executors validate arguments, enforce output limits,
support cancellation, or isolate handler state. MCP interoperability is not
tested or implemented.

The executor checkpoint on 2026-09-20 passed 272 focused tests across execution,
loop, suite, reporting, scenarios, scenario requests, and evaluation. The full
forced-offline suite passed `1217 passed, 7 skipped`: two platform-dependent
symlink skips and five PostgreSQL skips without a configured test URL. Ruff
and the working-tree-versus-HEAD whitespace check passed. No real model or
PostgreSQL interoperability was established by this run.

## Tool scenario, suite, and reporting validation

The standalone suite uses scripted providers and local synthetic handlers.
Coverage is split across:

- `tests/test_tool_scenarios.py`: immutable definitions, independent factories,
  strict text/tool/policy contracts, expected-tool selection, and example
  compatibility with local schemas and loop budgets.
- `tests/test_tool_scenario_requests.py`: selected registrations or descriptors, preserved
  order, fresh histories, missing tools, payload mutation isolation, and no
  copying of gold evaluation fields into serialized requests. Request building
  does not initialize the runtime or execute handlers. Descriptor collections
  also reject duplicate unselected names and use safe missing-tool messages.
- `tests/test_tool_suite.py`: end-to-end synthetic execution, all-request
  preparation before either factory, duplicate IDs, invalid factory results,
  normalized failures continuing to the next case, and unexpected exceptions
  (including `KeyboardInterrupt` and `SystemExit`) propagating without further
  case execution. Frozen results, derived summaries, repeated execution, and
  selected-tool stability are also covered. Descriptor/local result parity,
  executor-before-provider ordering, mutually exclusive sources, and executor
  factory/execution exceptions are verified with synthetic implementations.
- `tests/test_tool_reporting.py`: explicit match denominators, unavailable
  coverage, null rates, strict counts, immutable summaries, duplicate rejection,
  stable stop-reason distributions, consistency checks, and real evaluator
  output flowing into aggregation.

With the forced-offline variables set and the PostgreSQL test URL unset:

```powershell
python -m pytest tests/test_tool_suite.py tests/test_tool_scenario_requests.py tests/test_tool_scenarios.py tests/test_tool_reporting.py tests/test_tool_evaluation.py tests/test_tool_loop.py -q
python -m pytest -q
ruff check .
git diff --check
```

The previously reported suite checkpoint was `256 passed` for that focused
selection and `1201 passed, 7 skipped` for the complete offline suite. Ruff and
whitespace checks passed. Two skips were platform-dependent symlink tests and
five were PostgreSQL tests without a configured test URL. These are prior
results, not tests rerun for this documentation update.

The later descriptor-suite checkpoint passed 284 focused tests with:

```powershell
python -m pytest tests/test_tool_suite.py tests/test_tool_scenario_requests.py tests/test_tool_requests.py tests/test_tool_execution.py tests/test_tool_loop.py -q
```

The complete forced-offline result was `1357 passed, 7 skipped`; Ruff and
whitespace checks passed. The seven skips had the same platform/PostgreSQL
configuration causes described above. These are previously executed results,
not a fresh test run for this documentation-only update.

This coverage does not verify a real tool-calling endpoint, MCP transport,
suite persistence, wall-clock cancellation, or shared model/profile provenance.
Provider/executor factories are trusted to supply independent instances and
manage acquired resources; the suite does not close them. Local handlers are
not isolated from process state. Preparation is not remote preflight, and
an exception aborting a suite does not undo earlier calls. See
[Tool-evaluation suites](architecture.md#tool-evaluation-suites).

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

## Historical execution-trace forced-offline validation

At the execution-trace checkpoint, the complete forced-offline suite completed with:

```text
449 passed
7 skipped
0 failed
```

Five PostgreSQL integration tests were skipped because the disposable test URL
was not configured for this run; their separate verified execution is recorded
below. The other two skipped cases are platform-dependent real
symlink/junction tests; the deterministic physical-containment regression ran
and passed.

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
- Immutable typed execution traces, safe redaction, deterministic ordering,
  and best-effort artifact persistence
- SQLAlchemy engine/session behavior and SQLite foreign-key enforcement
- Alembic upgrade, downgrade, and re-upgrade behavior
- Repository CRUD, uniqueness, soft deletion, transactions, and run lifecycle
- Framework-independent application-service execution and failure handling
- FastAPI registry CRUD routes and strict public schemas
- OpenAI-compatible request/response normalization with mocked transports
- Queued Run API submission and separate worker execution
- Exact Run API preflight selection, limits, and safe public errors
- Framework-independent strict comparison of completed runs
- Deterministic sample alignment and typed incompatibility handling
- Comparison metric null/coverage behavior, including partial P50/P95 coverage
- Bounded CSV/JSONL storage, ingestion compensation, and upload API behavior
- Uploaded storage-key/checksum/content and physical-containment verification
- Uploaded dataset preflight and complete upload-to-Run-API execution

## PostgreSQL integration validation

PostgreSQL 16 is configured as the Docker integration runtime with Psycopg 3,
SQLAlchemy 2.x, and Alembic revision `20260817_0001`. Compose orders services
as PostgreSQL health, migration completion, then independent API and worker
startup. SQLite remains the default local/offline database.

The integration test gate is deliberately separate from ordinary tests. It
reads only `LLM_BENCHMARK_TEST_POSTGRES_URL`, requires a PostgreSQL database
whose name ends in `_test`, and never derives a destructive target from the
runtime database setting. When configured, the tests cover the Alembic
upgrade/downgrade/re-upgrade cycle, schema constraints, repository persistence
and rollback, and an uploaded synthetic dataset through the FastAPI Run API
with `MockProvider`.

The Docker runtime and the disposable `llm_benchmark_test` database were
executed successfully. Observed Compose states were a healthy `postgres`
service, a `migrate` service that exited successfully after applying Alembic
head, and a running `api` service. The verified migration revision was
`20260817_0001`.

The PostgreSQL test command and result were:

```text
.\.venv\Scripts\python.exe -m pytest -m postgres -q
5 passed, 408 deselected
```

The complete forced-offline suite, with the PostgreSQL test URL configured,
reported:

```text
411 passed, 2 skipped, 0 failed
```

The PostgreSQL-backed API integration registered a Mock endpoint and model and
uploaded one synthetic CSV multiple-choice dataset. POST returned HTTP 201
with `status=queued`, exact `sample_count=1`, null summary/start/completion
fields, and no sample rows. The real worker factory used the same registry and
uploaded-dataset root, claimed the run once, and completed it with one correct
synthetic `MockProvider` result. Summary, timestamps, artifact directory, and
the expected JSONL/config/manifest/summary artifacts were then present.
Dataset content remained in filesystem storage; PostgreSQL stored registration
and provenance metadata rather than uploaded rows. Nullable TTFT and
throughput remained null. No real model, Hugging Face, or external API request
was made.

Manual Docker validation used the same PostgreSQL database,
`benchmark_runtime` uploaded-dataset volume, and `benchmark_outputs` artifact
volume across API and worker services. Restarting the worker after completion
did not duplicate the run or its single sample result because terminal runs
are not claimable. This bounded observation does not establish lease,
heartbeat, stale-run recovery, retry, or general crash recovery. The
local-development Compose credential is only a reproducible placeholder and
is not suitable for production.

Run the PostgreSQL gate only against an explicitly disposable test database:

```powershell
$env:LLM_BENCHMARK_TEST_POSTGRES_URL = "postgresql+psycopg://<user>:<password>@127.0.0.1:5432/llm_benchmark_test"
.\.venv\Scripts\python.exe -m pytest -m postgres -q
```

Production secrets, connection-pool calibration, backup/restore, high
availability, zero-downtime migration, and deployment hardening remain out of
scope.

## Uploaded-dataset run integration validation

The completed integration was validated entirely with temporary SQLite,
temporary dataset storage, temporary artifact output, synthetic CSV/JSONL, and
`MockProvider`. No model, localhost, Hugging Face, or external request was made
during this validation.

The verified flow was:

```text
multipart CSV or JSONL upload
-> bounded streaming and SHA-256 content-addressed storage
-> adapter validation and dataset registration
-> POST /api/v1/runs with model_id and dataset_id
-> portable RunConfig resolution
-> uploaded file checksum and containment verification
-> exact dataset preflight
-> existing runner and MockProvider
-> persisted run/sample rows and reproducibility artifacts
```

Tests established that CSV and JSONL normalize to `DatasetExample`, explicit
sample-ID order is preserved, and an uploaded CSV run completes through the
actual Run API application path. Resolved dataset provenance and the manifest
contain the opaque storage key, adapter, and checksum without the temporary
physical storage root. Sample rows, summary data, JSONL results,
`dataset_manifest.json`, and `resolved_config.json` were produced through the
existing persistence and artifact contracts.

Missing files, tampered content, and invalid tabular data were rejected during
preflight before executor invocation. Those failures created no benchmark-run
row, sample-result row, or artifact directory, and their public responses were
sanitized. Storage tests separately cover malformed keys/checksums,
key/checksum disagreement, unreadable files, complete multi-chunk hashing, and
content-digest mismatch.

Physical storage containment is checked after strict path resolution. A
deterministic root-escape regression passed. Two additional tests that create
real file and directory symlinks were skipped because the Windows environment
did not allow symlink creation; they remain available on capable platforms.

The accepted repository snapshot was:

```text
Focused uploaded storage/dataset/preflight/Run API: 120 passed, 2 skipped
Complete forced-offline suite: 378 passed, 2 skipped, 0 failed
Ruff: passed
git diff --check: passed
```

All quality values produced by `MockProvider` in this path are synthetic
software-validation results, not measurements of model performance.

## Phase 5A comparison-service validation

The framework-independent `BenchmarkComparisonService` was validated with
in-memory immutable repository records and the complete forced-offline suite.
It accepts two distinct completed runs, aligns identical sample populations by
`sample_id` independently of row order, and rejects duplicate, empty,
asymmetric, or reference-inconsistent samples with typed errors.

Tests cover model differences as expected comparison context, provider and
endpoint differences as serving context, blocking evaluation-protocol
differences, and conditional reasoning, sampling, timeout, seed, and
output-budget differences. Quality, format-compliance, reliability, token, and
latency metrics are recomputed from persisted sample rows at aggregate,
category, and sample levels.

Null optional telemetry is preserved. Coverage records expose
`available_count`, `missing_count`, and `complete`. P50/P95 values are computed
from known successful-request latency values while coverage uses all successful
requests; incomplete coverage suppresses the absolute delta. The historical
Phase 5A snapshot was:

```text
Comparison tests: 21 passed
Comparison, repository, and application tests: 50 passed
Complete forced-offline suite at that checkpoint: 234 passed, 0 failed, 0 warnings
```

The service performs no database or artifact writes and emits no raw response,
provider/internal error message, endpoint URL, credential reference, dataset
URI/path, artifact path, or unrestricted metadata. It does not produce a
composite score and is not yet exposed by a FastAPI comparison route.

Attempt/retry/backoff details, stop reasons, final-output tokens, HTTP/provider
codes, costs, and partial-overlap comparisons are outside Phase 5A because the
required persisted data or product contract is not currently available.

An earlier `57 passed, 0 failed` result was a historical development checkpoint
before the database, repository, application-service, API, and preflight
increments. It is not the current total.

## Manual LM Studio endpoint validation

The local runtime was manually validated before implementing the provider:

| Field | Validated value |
| --- | --- |
| Runtime | LM Studio |
| Model repository | `unsloth/Qwen3.5-0.8B-GGUF` |
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

At a historical development checkpoint with 30 passing offline tests, one
approved localhost request was made with the corrected prompt and 128-token
limit. The request succeeded and returned five
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

At a later historical development checkpoint with 34 passing offline tests,
one final approved localhost request was made with the standalone-label
contract. The exact native message was `B`; it was
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

The historical run identifier was `20260812T130136Z-ff5ca994`. Generated
artifacts were ignored by Git and are not asserted to remain present in a
current checkout. This 14-question result validates the recorded local
execution path and strict output-format compliance; it is not a statistically
sufficient MMLU-Pro score.

## Reasoning-on Gate 2 result

Run ID: `20260813T081214Z-0410f308`

The same 14 pinned sample IDs, categories, dataset revision, split, seed,
model, strict prompt/parser contract, and sequential policy were used with
`reasoning="on"` and a bounded 1,024-token output budget.

| Metric | Value |
| --- | ---: |
| Requests | 14 |
| HTTP request success | 14/14 |
| Correct / incorrect | 0 / 0 |
| Unparseable / failed | 14 / 0 |
| Parse success / format failure | 0% / 100% |
| Input tokens | 4,270 |
| Total output tokens | 14,336 |
| Reasoning output tokens | 14,336 |
| Derived final-output tokens | 0 |
| Latency P50 / P95 | 64,536.63 / 67,980.39 ms |
| Mean TTFT | 493.04 ms |
| Mean throughput | 16.09 tokens/s |
| Logical-duration sum | 906,158.60 ms |
| Run wall time | 906,178.97 ms |

Every response used exactly 1,024 output tokens as reasoning tokens and
produced an empty native final message. Reasoning was never scored as an
answer, so all results were correctly classified as unparseable.

## Single-sample 2,048-token calibration

Successful run ID: `20260813T125426Z-951e53ca`
Sample: `2019` (`psychology`, correct label `I`)

The request succeeded without retry or timeout. It reported 154 input tokens,
2,048 total-output tokens, 2,048 reasoning tokens, zero derived final-output
tokens, 2,202 total tokens, 122,546.88 ms latency, 795.47 ms TTFT, and 16.85
tokens/s. The final message was empty and the strict parser returned
`unparseable`.

An earlier attempt (`20260813T105947Z-2005f467`) ended at the runtime/provider
boundary with `server_error`. Local LM Studio Developer Logs later reported a
sanitized `internal_error` with code `unknown`; it did not establish that
2,048 tokens had been generated. The successful calibration supersedes that
attempt for output-budget analysis, while both artifacts remain separate.

## Partial native-supported sampling calibration

Run ID: `20260813T132112Z-9f0ac78f`
Sample: `2019`

The profile used `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`,
and `repeat_penalty=1.0`. Native `presence_penalty` support was unverified and
the field was omitted. The result again reported 154 input tokens, 2,048
total-output tokens, 2,048 reasoning tokens, zero final-output tokens, and an
empty final message. Latency was 119,960.24 ms, TTFT was 176.64 ms, and
throughput was 17.15 tokens/s.

Sanitized inspection through local LM Studio Developer Logs showed strong
repetitive, non-convergent deliberation until output-budget exhaustion. This
supports the conclusion of probable non-terminating or excessively long
reasoning under the tested model/runtime/prompt/sampling configuration. It
does not prove a literal infinite loop. Raw reasoning text is not stored in
the repository, documentation, or tracked artifacts.

## Provider-default context-bounded calibration

Run ID: `20260814T082432Z-39883af2`

This single-sample calibration reused sample `2019`, the pinned dataset
revision, reasoning-on mode, and the partial native-supported sampling profile.
`max_output_tokens` and `presence_penalty` were absent from the native payload.
The result recorded `output_budget_provenance=provider_default` and used a
660-second transport safety timeout without retry.

LM Studio reported an 8,192-token context for the loaded instance and a
262,144-token maximum in model metadata. The loaded-instance value is the
relevant runtime configuration, but neither value establishes the exact output
allowance because input, template, special tokens, and runtime overhead share
the context.

| Metric | Bounded 1,024 | Bounded 2,048 | Provider default |
| --- | ---: | ---: | ---: |
| Input tokens | 154 | 154 | 154 |
| Reasoning tokens | 1,024 | 2,048 | 3,591 |
| Derived final-output tokens | 0 | 0 | 4 |
| Total-output tokens | 1,024 | 2,048 | 3,595 |
| Total tokens | 1,178 | 2,202 | 3,749 |
| Final message | Empty | Empty | `B` |
| Parsed/evaluation status | Unparseable | Unparseable | `B` / incorrect |
| Latency | 62,444.60 ms | 122,546.88 ms | 209,883.45 ms |
| TTFT | 808.346 ms | 795.465 ms | 215.925 ms |
| Throughput | 16.7643 tokens/s | 16.8524 tokens/s | 17.1926 tokens/s |

The provider-default request succeeded, did not time out, and reached a final
message after exceeding both earlier bounded budgets. The final label was
format-compliant but differed from correct label `I`. No explicit configured
output limit ended the request, but the null stop reason prevents a stronger
termination claim. The request did not consume the full loaded context. This
single sample neither proves that provider-default generation always
terminates nor that reasoning improves accuracy; it is an operational
calibration rather than a benchmark score.

## Three-sample provider-default operational calibration

Run ID: `20260814T104711Z-3e7b92e5`

Fixed samples `215`, `6232`, and `2019` were executed sequentially with one
attempt each, a 660-second timeout, and provider-default output budgeting.
`max_output_tokens` and `presence_penalty` were absent. All three requests
succeeded, parsed successfully, and reached a native final message. One answer
was correct and two were incorrect.

| Sample | Category | Expected / parsed | Result | Input | Reasoning | Final output | Total output | Total tokens | Latency | TTFT | Throughput |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `215` | business | `A` / `A` | correct | 285 | 6,804 | 4 | 6,808 | 7,093 | 439,171.65 ms | 2,284.082 ms | 15.6165 tok/s |
| `6232` | health | `C` / `J` | incorrect | 328 | 5,116 | 4 | 5,120 | 5,448 | 329,634.88 ms | 2,100.948 ms | 15.6703 tok/s |
| `2019` | psychology | `I` / `B` | incorrect | 154 | 2,203 | 4 | 2,207 | 2,361 | 135,853.82 ms | 1,003.439 ms | 16.4733 tok/s |

Aggregate operational measurements:

- Final messages reached: 3/3 (100%)
- Request success: 3/3 (100%)
- Parse success: 3/3 (100%)
- Calibration-only correct: 1/3 (33.33%)
- Total tokens: 14,902
- Total reasoning tokens: 14,123
- Logical-duration sum: 904,660.36 ms
- Run wall time: 904,667.38 ms
- Latency P50/P95: 329,634.88 / 428,217.97 ms

No request timed out or showed explicit output-limit exhaustion, and all final
messages were format-compliant. Native stop reasons were not provided. These
three selected samples characterize operational final-message reliability and
cost only; they are not an MMLU-Pro accuracy estimate.

### Paired reasoning-off comparison

The same three samples were extracted from reasoning-off run
`20260812T130136Z-ff5ca994` for a paired descriptive comparison.

| Aggregate for the same samples | Reasoning off | Context bounded |
| --- | ---: | ---: |
| Correct / incorrect | 2 / 1 | 1 / 2 |
| Unparseable / failed | 0 / 0 | 0 / 0 |
| Final-message rate | 100% | 100% |
| Request / parse success | 100% / 100% | 100% / 100% |
| Total tokens | 779 | 14,902 |
| Reasoning tokens | 0 | 14,123 |
| Logical-duration sum | 4,053.07 ms | 904,660.36 ms |
| Latency P50/P95 | 1,403.38 / 1,516.33 ms | 329,634.88 / 428,217.97 ms |

This comparison does not isolate reasoning. The reasoning-off run used
`reasoning=off`, `temperature=0`, and a fixed 128-token output budget. The
context-bounded run used `reasoning=on`, `temperature=1.0`, `top_p=0.95`,
`top_k=20`, `min_p=0.0`, `repeat_penalty=1.0`, and provider-default output
budgeting. The runs also used different workload sizes, Git revisions, dates,
and runtime states. Therefore token, latency, and quality differences cannot be
causally attributed solely to reasoning, and no statistical significance can
be inferred from three samples.

## OpenAI-compatible localhost interoperability validation

Exactly one explicitly approved synthetic fixture request validated the generic
`OpenAICompatibleProvider` against LM Studio. The base URL was
`http://127.0.0.1:1234/v1`, the API model identifier was `qwen3.5-0.8b`, and
the provider called `POST /v1/chat/completions`. No credential or
`Authorization` header was used.

The fixture selected only `q01` (`math`, correct label `B`). The standard final
message was `B`, the strict parser returned `parsed_answer=B`, and the request
completed successfully with `stop_reason=stop`.

| Metric | Value |
| --- | ---: |
| Requests | 1 |
| Input tokens | 90 |
| Total-output tokens | 326 |
| Reasoning tokens | 322 |
| Derived final-output tokens | 4 |
| Total tokens | 416 |
| Latency | 21,464.63 ms |
| TTFT | null, not estimated |
| Throughput | null, not estimated |

The request omitted `max_tokens`, so the result recorded
`output_budget_provenance=provider_default`; this does not mean unlimited
generation. No reasoning mode was transmitted. `reasoning_mode` remained null,
while the endpoint independently reported reasoning-token usage. The behavior
is therefore provider-managed or unverified and must not be labelled
reasoning-on or reasoning-off.

This successful result validates interoperability, standard response mapping,
strict parsing, and telemetry preservation for one fixture request. It is not
a model-quality result, an MMLU-Pro score, or evidence about general endpoint
reliability. No raw reasoning content is stored in this documentation.

## Interpretation boundary

The validation proves that the current pipeline can load and sample data,
produce deterministic provider responses, parse and classify those responses,
calculate metrics, and persist reproducibility artifacts.

It does not establish:

- Full MMLU-Pro performance for any real model
- General provider latency or reliability beyond the recorded local runs
- Monetary cost
- Compatibility beyond the individually recorded endpoint validations
- Statistical validity of a 14-sample score

Real provider results must be recorded in separate runs with explicit provider,
endpoint alias, requested and returned model IDs, request parameters, and
measurement protocol metadata.

## Pre-MCP tool-run checkpoint (2026-09-28)

Validated on Windows with Python 3.12.10 using the existing virtual environment:

- `ruff check .`: passed.
- Forced-offline `pytest -q`: **1596 passed, 7 skipped**, 44.63 seconds.
- Five PostgreSQL tests skipped because `LLM_BENCHMARK_TEST_POSTGRES_URL` was
  not configured; two filesystem symlink tests skipped due to OS support.
- `python -m llm_benchmark tool-run --config configs/tool_mock_smoke.yaml`:
  completed both synthetic cases, one successful calculator call, three
  provider turns and twelve ordered trace events.
- The generated manifest, summary, results and trace were read back and checked.
  Runtime artifacts remain ignored by Git.

New coverage checks artifact fingerprint stability and policy sensitivity,
partial results on unexpected provider failure, omission of exception secrets,
nullable telemetry, conditional versus all-case final accuracy, normalized
provider failure as an evaluation outcome, deadline checks after provider/tool
calls, provider timeout clamping, descriptor-suite callback forwarding and CLI
execution. Existing regression coverage remains passing.

This checkpoint does not validate a real model endpoint, MCP, Docker execution,
PostgreSQL integration, hard cancellation, or production recovery. Tool traces
contain metadata rather than raw conversations. The deadline is cooperative;
transport-specific cancellation belongs to the future MCP adapter.

See [the Turkish implementation walkthrough](projeyi-anlama-ve-mcp-oncesi.md).

## Retrieval CLI checkpoint (2026-10-19)

Validated locally on Windows with Python 3.12.10:

- Focused run/identity/suite tests: **89 passed**.
- Full offline regression suite: **1813 passed, 7 skipped**, 60.06 seconds.
- `ruff check .`: passed after correcting the new config module's import layout.
- `git diff --check`: passed.
- Actual `retrieval-run --config configs/retrieval_keyword_smoke.yaml`: completed
  four queries at K=2, hit rate 0.75, mean precision 0.375, mean recall 0.75,
  MRR 0.75. Manifest and incremental results were read back after execution.

Full-suite command (repository root, existing environments):

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:LLM_BENCHMARK_MCP_PYTHON = (Resolve-Path .venv-mcp/Scripts/python.exe).Path
.\.venv\Scripts\python.exe -m pytest -q
```

The MCP interpreter variable enables existing real stdio integration tests; it
is not required by retrieval. Five PostgreSQL tests were skipped because the
test database URL was absent, and two symlink tests were skipped due to Windows
filesystem support. No external model endpoint or PostgreSQL integration was
validated in this checkpoint.

New tests cover strict config and dataset ingestion (including duplicate keys,
invalid UTF-8, size limits, versions, duplicate IDs, and gold membership), path
resolution independent of the working directory, deterministic content identity
with fresh run IDs, known smoke metrics, artifact consistency, preflight without
output creation, partial execution failures, storage failure propagation, safe
error messages, and CLI output/exit behavior. Existing identity and evaluator
tests remain unchanged and passing.

This establishes local correctness, not representative retrieval quality or
performance. The four-query corpus is intentionally tiny; timings are individual
observations. API/worker integration, semantic retrieval, RAG, recovery/resume,
and complete execution-environment reproducibility remain outside this increment.

## SciFact keyword baseline (2026-10-20)

The local BEIR SciFact test run completed successfully.

- Run ID: `4188b7e1e9f246b8b0bb2a908060d4a1`
- Dataset: BEIR SciFact, test split
- Corpus size: 5,183 documents
- Evaluated queries: 300
- Representation: one source document per chunk
- Retriever: keyword-overlap, version 1
- K: 10
- Evaluator: retrieval-exact-v1
- Python: 3.12.10
- Manifest status: completed

| Metric | Observed value |
|---|---:|
| Hit rate at 10 | 0.62 |
| Mean precision at 10 | 0.065 |
| Mean recall at 10 | 0.5969444444444445 |
| MRR at 10 | 0.45690740740740743 |
| Index build time | 460.06 ms |
| Total retrieval time | 1997.43 ms |
| Mean retrieval time | 6.66 ms |
| Measured wall time | 3456.80 ms |

Artifact inspection confirmed:

- 300 per-query result records.
- 186 queries with at least one relevant result in the top 10.
- 114 queries with no relevant result in the top 10.
- 115 queries with a relevant result at rank 1.
- 195 retrieved relevant query-document pairs out of 339 gold pairs.
- The prepared suite SHA-256 matches its provenance sidecar.

Recall is macro-averaged across queries. It is not calculated as
195 divided by 339.

Timing values are observations from one local run, not a performance
guarantee. The wall-time measurement excludes final summary and
manifest persistence.

This checkpoint validates a real external-dataset execution. It does
not establish RAG answer quality, semantic retrieval quality, or
equivalence to published BEIR leaderboard results.

For the full test suite, 1830 offline test passed, 7 skipped. Skipped
tests are symlink and postgresql tests.

## Retrieval review verification (2026-10-23)

Focused verification of the diagnostic review changes:

- `python -m pytest -q tests/test_retrieval_review.py tests/test_retrieval_run.py`:
  **90 passed**, 7.80 seconds.
- Coverage includes deterministic selection, malformed/mismatched artifacts,
  duplicate result IDs, metric types, replay agreement and disagreement,
  partial recall despite rank-1 success, input/output limits, non-overwriting
  publication, storage failure cleanup, and CLI errors.
- The hard-link report publication path was exercised on this Windows workspace.
- `ruff check .`: passed after fixing the existing preparation-test import layout.

The saved SciFact diagnostic review was subsequently inspected.
Its report records `retrieval_replay: matched`, confirming that current
keyword retrieval reproduced the saved rankings.

Population counts:

- Rank 1: 115 queries.
- First relevant result at ranks 2–10: 71 queries.
- No relevant result in the top 10: 114 queries.
- Partial recall: 13 queries.

The selected diagnostic sample contains 5 rank-1 queries, 5 later-hit
queries, and 10 misses.

These observations do not constitute a new full regression run.
Qualitative causes of retrieval errors require inspection of the query,
gold evidence, and retrieved documents.300-query SciFact run.
Run those separately before merging, and record the
actual results. No qualitative findings about the sampled SciFact cases are
claimed by this checkpoint.

## Dense CPU smoke evidence (recorded 2026-10-25)

This checkpoint records a successful smoke report supplied by the user.
The JSON report does not contain an execution timestamp or invocation flags;
the date above is the recording date. The model was not rerun while adding
this record. The user subsequently confirmed a successful pinned offline run.
The automated test results below were independently rerun against the working
tree on 2026-10-25; the real model was not rerun during that verification.

### Model and retrieval profile

| Setting | Recorded value |
| --- | --- |
| Status | `passed` |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Retriever | `dense-exact-cosine`, version `1` |
| Adapter | `smoke-sentence-transformers-cpu`, version `1` |
| Device / precision | `cpu` / `float32` |
| Embedding dimension | `384` |
| Maximum sequence length | `256` |
| Batch size | `8` |
| Query / document prefixes | Both empty |
| Normalization | `l2-scaled-v1` |
| Similarity | `cosine` |
| Tie-break | `chunk-id-ascending` |
| Score filter | `none` |
| Truncation policy | `reject` |

Reported retriever profile hash (copied from the supplied output):

```text
559c5fd48fbbaca8435308a46bbe2e77a6b9b333382ec3694143384b63127f4d
```

### Observed model behavior

| Observation | Recorded value |
| --- | --- |
| Model device | `cpu` |
| Model embedding dimension | `384` |
| Model sequence limit | `256` |
| Floating-point parameter dtypes | `["torch.float32"]` |
| Embedding output dtype | `float32` |
| Tokenizer class | `transformers.models.bert.tokenization_bert.BertTokenizer` |
| Tokenizer `is_fast` | `true` |
| CPU threads / interop threads | `4` / `4` |
| Exact-query ranking | `cat`, `bread`, `space` |
| Paraphrase-query ranking | `cat`, `bread`, `space` |
| Long input rejected | `true` |
| Long-input token count | `514` |
| Repeated-encoding maximum absolute difference | `0.0` |
| Batched-encoding maximum absolute difference | `1.564621925354004e-07` |

The exact query was `A cat is sleeping on the sofa.`; the paraphrase
was `A kitten naps on the couch.`. Both ranked the expected `cat` document
first in the three-document corpus. The long input contained 514 tokens,
exceeded the 256-token limit, and was rejected.

For the probe text, repeated encoding within the process produced no observed
numeric difference. Encoding it alongside the other documents produced a
nonzero maximum absolute difference of approximately `1.56e-7`. This value
is diagnostic, not a configured acceptance threshold or evidence of general
batch independence.

Reported probe vector hash (copied from the supplied output):

```text
5f2fb87878bd29ba47c6cbfe89286d6bd2c8e460cfdb106f6668c12204a4a616
```

These hashes are recorded as reported, not independently recomputed in this
documentation update. The probe vector hash describes one vector, not the
complete corpus embedding set or model files.

### Runtime package versions

```json
{
  "torch": "2.14.1+cpu",
  "sentence-transformers": "6.2.0",
  "transformers": "5.20.2",
  "huggingface-hub": "1.33.0",
  "tokenizers": "0.23.3",
  "numpy": "2.5.2"
}
```

### Reproduction and follow-up commands

Commands use the repository root. Evidence sources are identified below;
pytest results are from actual reruns, not inferred from the supplied JSON.

Pinned offline smoke (requires the revision to be cached):

```powershell
.\.venv\Scripts\python.exe scripts/smoke_dense_cpu.py `
  --offline `
  --revision "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
```

The user confirmed that the pinned offline command completed successfully.
This is user-reported offline validation, separate from the automated pytest
reruns below. It does not establish bit-identical vectors between processes.

Additional dependency and scoped lint commands (not rerun for this update):

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check scripts/smoke_dense_cpu.py tests/test_smoke_dense_cpu.py src/llm_benchmark/retrieval/profile.py src/llm_benchmark/retrieval/identity.py tests/test_dense_profile.py
```

Smoke unit tests executed on 2026-10-25:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_smoke_dense_cpu.py
```

Result: **13 passed in 0.08 seconds**, exit code 0; no failures or skips.

Focused retrieval regression executed on 2026-10-25:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_dense_profile.py `
  tests/test_retrieval_identity.py `
  tests/test_dense_retriever.py `
  tests/test_retrieval_run.py `
  tests/test_retrieval_review.py `
  tests/test_smoke_dense_cpu.py
```

Result: **190 passed in 6.31 seconds**, exit code 0; no failures or skips.

The focused regression includes the 13 smoke unit tests; their counts must
not be added together. Fake-model unit tests validate control flow without
loading the real embedding model. These are scoped checks, not a new
full-project regression.

### Interpretation and next boundary

The supplied result supports CPU model integration, the checked profile and
output properties, expected ranking for two small examples, and the long-input
rejection path. It does not establish SciFact retrieval quality, bit-identical
embeddings across environments, model artifact integrity, large-corpus
performance, hybrid retrieval quality, or RAG answer quality.

Dense retrieval is not yet connected to the benchmark CLI. The next increment
is to extract the validated embedding bridge into a package adapter, preserve
its tests, and then integrate dense configuration, profile-aware run identity,
runtime metadata, and artifact reporting.

## Licensing note

No project license has been selected yet. MMLU-Pro attribution and its dataset
license metadata remain recorded separately from the project license.
