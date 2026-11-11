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

## CPU adapter validation follow-up (2026-10-25)

The package adapter and smoke now use `EmbeddingInputTooLongError` to
distinguish expected token-limit rejection from unrelated backend failures.
This follow-up validates the uncommitted working tree after adapter extraction.

The test helpers' captured-input attribute names were corrected. Additional
tests verify invalid containers and blank/non-string texts fail before
tokenization, and that an over-limit later document rejects the complete
input before any model encoding. Existing tests cover order preservation,
encoder options, shape/dtype, non-finite values, profile compatibility, and
propagation of unrelated errors from the smoke rejection check.

Executed focused regression:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_local_embeddings.py `
  tests/test_smoke_dense_cpu.py `
  tests/test_dense_profile.py `
  tests/test_retrieval_identity.py `
  tests/test_dense_retriever.py `
  tests/test_retrieval_run.py `
  tests/test_retrieval_review.py
```

Result: **213 passed in 5.98 seconds**, exit code 0, no failures or skips.
This is a focused regression, not a full-project run.

Scoped lint also passed:

```powershell
.\.venv\Scripts\python.exe -m ruff check src/llm_benchmark/retrieval/local_embeddings.py scripts/smoke_dense_cpu.py tests/test_local_embeddings.py tests/test_smoke_dense_cpu.py
```

The real model was independently rerun using the cached, pinned revision:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_dense_cpu.py `
  --offline `
  --revision "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
```

Result: `status: passed`, exit code 0. Both queries ranked `cat` first;
the 514-token input was rejected against the 256-token limit. The profile
hash, probe vector hash, runtime versions, and recorded diagnostics matched
the earlier supplied report: repeat difference `0.0`, batch difference
`1.564621925354004e-07`. This observation does not guarantee reproducibility
across other environments or inputs.

The tokenizer emitted a sequence-length warning for the deliberately long
input. This was expected: the adapter rejected it before model encoding.

Adapter extraction is complete. The next boundary is reusable model loading
and effective-setting validation, followed by dense CLI and artifact integration.
Whole-input tokenization and reject-only truncation remain current limitations.

## CPU embedding loader unit checkpoint (2026-10-25)

The loader now has 27 fake-model test cases. Coverage includes default offline
loading, constructor argument forwarding for both offline settings, strict
profile/boolean/revision validation, unsupported settings rejected before
loading, effective device/dimension/sequence-limit mismatches, parameter dtype
checks after preparation, non-floating parameter handling, and load failure
propagation without retry. The loader error-message typo was corrected.

Executed focused regression:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_embedding_loader.py `
  tests/test_local_embeddings.py `
  tests/test_smoke_dense_cpu.py `
  tests/test_dense_profile.py `
  tests/test_retrieval_identity.py `
  tests/test_dense_retriever.py `
  tests/test_retrieval_run.py `
  tests/test_retrieval_review.py
```

Result: **240 passed in 5.93 seconds**, exit code 0, no failures or skips.
The 27 loader cases are included in this count. Scoped Ruff passed:

```powershell
.\.venv\Scripts\python.exe -m ruff check src/llm_benchmark/retrieval/embedding_loader.py tests/test_embedding_loader.py
```

No real model was loaded in this checkpoint, and this was not a full-project
regression. The existing smoke script still uses its separate loading path;
its earlier real-model success does not yet validate integration of this
new loader. No commit or PR was created for this checkpoint.

## Dense config validation checkpoint (2026-10-25)

Reproduced the initial failure: **1 failed, 3 passed**. The field was named
`max_seqeunce_length` in the config and profile conversion. Correct input
therefore produced both a missing-field error and an extra-field error.
Both references now use `max_sequence_length`; the misspelling is deliberately
not accepted as an alias.

Dense config coverage now contains 39 test cases. Added cases verify strict
numeric/boolean types, invalid model IDs and revisions, positive limits,
unknown fields, missing/malformed embedding settings, exact profile mapping,
default batch size, offline/online load policy, config-relative paths, and
duplicate nested YAML keys. Load policy does not alter the embedding profile
for an otherwise identical pinned model.

Executed focused regression:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_retrieval_dense_config.py `
  tests/test_embedding_loader.py `
  tests/test_local_embeddings.py `
  tests/test_smoke_dense_cpu.py `
  tests/test_dense_profile.py `
  tests/test_retrieval_identity.py `
  tests/test_dense_retriever.py `
  tests/test_retrieval_run.py `
  tests/test_retrieval_review.py
```

Result: **279 passed in 6.12 seconds**, exit code 0, no failures or skips.
Scoped Ruff passed for `run_config.py` and `test_retrieval_dense_config.py`.
No real model was loaded and no full-project regression was run for this fix.
Dense configuration acceptance does not yet mean dense execution is wired into
the run service; do not use a dense config as a benchmark until that integration
is implemented and tested.

## Dense run service and CLI checkpoint (2026-10-26)

Added 10 parametrized service/CLI cases in `tests/test_retrieval_run.py`.
They use a fixed-vector embedder and the real dense retriever, without model
downloads. Coverage verifies dense profile/hash artifacts, exact expected
rankings and metrics, one model load and one corpus encoding per run,
online/offline flag forwarding, and absence of embedding loads for keyword.

Failure cases cover loading, corpus encoding, and the second query: manifests
become failed, completed result rows are preserved, no success summary is
written, and private backend messages are omitted from public artifacts.
CLI tests verify JSON success output and exit code 2 on load failure.
A controlled clock verifies 2,000 ms model loading, 3,000 ms index building,
and four 250 ms queries are assigned to separate timing fields. A changed
model revision changes profile/run hashes while corpus/suite hashes stay fixed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_run.py tests/test_retrieval_dense_config.py tests/test_embedding_loader.py tests/test_local_embeddings.py tests/test_smoke_dense_cpu.py tests/test_dense_profile.py tests/test_retrieval_identity.py tests/test_dense_retriever.py tests/test_retrieval_review.py
```

Result: **289 passed in 9.42 seconds**, no failures or skips. Scoped Ruff
passed for the run service and its test file. This is not a full-project
regression. No real model or real dense CLI benchmark was run in this checkpoint.
The existing `retrieval-run --config` CLI needs no new subcommand for dense.
Real dense CLI validation, runtime provenance, dense review/replay, and
SciFact long-input policy remain follow-up work.

## Embedding runtime persistence checkpoint (2026-10-27)

Completed runtime assertions in `tests/test_retrieval_run.py`. Corrected the
fake metadata key from `runtime_version` to `runtime_versions`, and replaced
an undefined `offline` variable in the CLI test with the fixture's actual
`True` setting. Dense success checks cover both load policies and verify that
the collector receives the loaded model exactly once. Keyword execution
neither collects nor writes embedding runtime metadata.

Added tests for metadata being persisted while the manifest is running,
before corpus encoding; metadata collection failure; and runtime manifest
write failure. Both failures stop encoding and produce a failed manifest
when storage permits the failure record, without a success summary or leaked
backend details. Existing index/query failure tests now verify retained
runtime metadata. The fake-clock test includes 500 ms of metadata collection
in wall time but excludes it from model loading, index building, and query
latencies.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_embedding_runtime.py tests/test_retrieval_run.py tests/test_retrieval_dense_config.py tests/test_embedding_loader.py tests/test_local_embeddings.py tests/test_smoke_dense_cpu.py tests/test_dense_profile.py tests/test_retrieval_identity.py tests/test_dense_retriever.py tests/test_retrieval_review.py
```

Result: **293 passed in 8.11 seconds**, no failures or skips. Scoped Ruff
passed for the run service, runtime collector, and their two test files.
This is a focused regression with fake embedding/runtime dependencies, not
a full-project regression or a new real-model run. Real CLI verification of
the new manifest fields remains the next manual validation step.

## Dense recorded-result review checkpoint (2026-10-28)

Added seven dense review test cases using temporary runs, fixed embeddings,
and fake runtime metadata. During review, model dependency imports, model
loading, and keyword retrieval are forbidden by the fixture. Tests verify
profile/hash output, unchanged source artifacts, rejection of unsupported
dense replay, rejection of changed revisions/algorithm fields and stored
metrics, and CLI publication without replacing an existing report on failure.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval_profile_serialization.py tests/test_embedding_runtime.py tests/test_retrieval_run.py tests/test_retrieval_dense_config.py tests/test_embedding_loader.py tests/test_local_embeddings.py tests/test_smoke_dense_cpu.py tests/test_dense_profile.py tests/test_retrieval_identity.py tests/test_dense_retriever.py tests/test_retrieval_review.py
```

Result: **313 passed in 7.78 seconds**, no failures or skips. Scoped Ruff
passed for the review script, profile decoder, and review tests. This was
not a full-project regression or a new embedding-model execution.

The completed real run `dd30de0a5982409ead0e33f672762b69` was also reviewed
read-only through `build_review` using `examples/retrieval/keyword_smoke.json`.
All recorded per-query metrics and input/profile identity checks passed.
Population counts were rank 1: **4**, ranks 2 through K: **0**, misses: **0**.
The rebuilt report matched the existing `review.json` as a complete parsed
JSON object. No source artifact or existing review file was modified.
`retrieval_replay` remained `not_requested`: recorded rankings were evaluated,
not regenerated by the model.

## SciFact dense and comparison artifacts (2026-10-29)

The user completed the CPU run and comparison command. The stored artifacts were
inspected for this checkpoint; this is not a new model execution or full regression.

- Dense run: `0a28cd3481bd435886040c7a1d91c7b6`.
- Keyword run: `4188b7e1e9f246b8b0bb2a908060d4a1`.
- Dense configuration: `configs/retrieval_dense_scifact.yaml`.
- Model: `sentence-transformers/all-MiniLM-L6-v2`.
- Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- CPU float32, batch size 8, sequence limit 256, `truncate` policy.
- 5,183 document inputs and 300 query inputs; 3,681 documents and zero queries
  exceeded the token limit.

| Metric | Keyword overlap | Dense |
| --- | ---: | ---: |
| Queries / K | 300 / 10 | 300 / 10 |
| Hit rate | 0.62 | 0.7933333333333333 |
| Mean precision | 0.065 | 0.08833333333333333 |
| Mean recall | 0.5969444444444445 | 0.7833333333333333 |
| MRR | 0.45690740740740743 | 0.6047248677248678 |

Dense timings: model load 9,977.01 ms; index build 225,487.57 ms;
retrieval total 61,157.86 ms; mean retrieval 203.86 ms; wall time 298,594.60 ms.
Separate-run timings are observations, not a controlled performance comparison.

`comparison-keyword.json` reports 166 both-hit queries, 20 keyword-only queries,
72 dense-only queries, and 42 both-miss queries. Hit-rate delta is
0.17333333333333334 (17.333 percentage points). Corpus, suite, and evaluation
configuration hashes and K match. Union coverage is 258/300 (86%), using up to
20 candidates; it is not a hybrid hit-rate-at-10 result.

The comparison validates recorded rankings and metrics, not model replay.
Approximately 71.02% of document inputs exceeded the token limit. Conclusions
are limited to this dataset, model revision, and truncation policy. The lexical
baseline is keyword overlap, not BM25. Comparison CLI failure-path integration
tests and a full regression remain release checks.

Focused check performed for this checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval_comparison.py -q
```

Result: **12 passed in 0.05 seconds**. `git diff --check` also passed.
This checks the comparison core, not the script's end-to-end failure paths.

## BM25 SciFact timing correction checkpoint (2026-10-31)

The user completed run `4a6428c22ffe43bb83b45bbe22e36ef7` using
`configs/retrieval_bm25_scifact.yaml`. The stored manifest, summary, and result
count were inspected: status is `completed`, with 300 completed cases and 300
result rows at K=10.

The profile is `bm25-rank-bm25-okapi`, version 1, using rank-bm25 0.2.2,
k1=1.2, b=0.75, and epsilon=0.25. It retains matching zero/negative-score
candidates and resolves equal scores by chunk ID.

| Metric | Observed value |
| --- | ---: |
| Hit rate@10 | 0.8 |
| Mean precision@10 | 0.08533333333333334 |
| Mean recall@10 | 0.7773333333333333 |
| MRR@10 | 0.6228201058201058 |
| Model load | 0.0 ms |
| Index build | 717.0478000771254 ms |
| Retrieval total | 5936.635401332751 ms |
| Retrieval mean | 19.78878467110917 ms |
| Wall time | 7682.498199981637 ms |

Quality metrics match the earlier BM25 run
`9a5cb73b2dc140c397c317a1d149575a`. That earlier run incorrectly stored seconds
in `index_build_ms` (0.6798546999925748); do not use that field as milliseconds.
The service now multiplies the elapsed seconds by 1000. Historical artifacts
were not rewritten. Quality agreement alone does not prove identical rankings.

Two regression protections cover the recent fixes: a controlled 2.5-second
index build must produce 2500 ms, and BM25 review must not construct a keyword
retriever. The earlier replay bug unconditionally replaced the selected BM25
retriever with a keyword retriever.

Focused regression executed for this checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_bm25_retriever.py tests/test_bm25_run.py tests/test_retrieval_profile_serialization.py tests/test_retrieval_identity.py tests/test_retrieval_run.py tests/test_retrieval_review.py tests/test_compare_retrieval_runs.py -q
```

Result: **214 passed in 9.03 seconds**. `git diff --check` passed.
This is not a full-project regression or a new SciFact execution by the assistant.
At inspection, the new run directory contained manifest, results, and summary
only. Replay and comparison report generation for this new run remain pending;
commands are documented in `retrieval.md`.

The earlier run's comparison against dense reported 215 both-hit, 25 BM25-only,
23 dense-only, and 37 both-miss queries. This is historical comparison evidence,
not a newly generated comparison for the corrected-timing run. Union coverage
of 263/300 is not a measured hybrid hit rate at 10.

## Hybrid smoke and SciFact artifact checkpoint (2026-10-31)

Existing user-produced artifacts were inspected; no new embedding run was
performed for this documentation update.

Smoke run `ee4472f03d7f4e7487a42fbba469f363` completed four cases at K=2:
hit rate, recall, and MRR were 1.0; mean precision was 0.5.

SciFact run `af1f0af4c0db430f8cc86e80046cfee4` completed 300 cases at K=10.
The recorded profile uses equal-weight BM25+dense RRF with candidate_k=50,
rank_constant=60, and chunk-ID tie breaking. The dense child uses the pinned
MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32,
batch size 8, and a 256-token truncate policy.

| Metric | BM25 | Dense | Hybrid |
| --- | ---: | ---: | ---: |
| Hit rate@10 | 0.8 | 0.7933333333333333 | 0.84 |
| Mean precision@10 | 0.08533333333333334 | 0.08833333333333333 | 0.092 |
| Mean recall@10 | 0.7773333333333333 | 0.7833333333333333 | 0.8275555555555556 |
| MRR@10 | 0.6228201058201058 | 0.6047248677248678 | 0.6544325396825397 |

Hybrid timings: model load 9,264.11 ms; index build 238,767.81 ms; retrieval
total 67,670.62 ms; retrieval mean 225.57 ms; wall time 319,654.10 ms.
Dense input statistics record 5,183 document encodings (3,681 over limit)
and 300 query encodings (zero over limit).

The stored review reports `retrieval_replay: not_requested`, with 168 rank-1
hits, 84 hits at ranks 2 through K, and 48 misses. Stored comparison reports:

| Baseline vs hybrid | Both hit | Baseline only | Hybrid only | Both miss |
| --- | ---: | ---: | ---: | ---: |
| BM25 (`4a6428c22ffe43bb83b45bbe22e36ef7`) | 234 | 6 | 18 | 42 |
| Dense (`0a28cd3481bd435886040c7a1d91c7b6`) | 226 | 12 | 26 | 36 |

Hybrid gains 12 net hits over BM25 and 14 over dense, but loses some individual
baseline hits. Results cover one dataset/configuration, not general superiority.
The 50-candidate child budget differs from the baseline top-10 output budget.
Separate-run timings are observations, not controlled performance measurements.
Recorded review does not establish embedding replay or bitwise reproducibility.

Focused regression performed for the hybrid checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py tests/test_hybrid_config.py tests/test_hybrid_run.py tests/test_bm25_run.py tests/test_retrieval_run.py tests/test_retrieval_review.py tests/test_retrieval_profile_serialization.py tests/test_retrieval_identity.py tests/test_compare_retrieval_runs.py -q
```

Result: **212 passed in 8.35 seconds**. This is a focused regression, not the
full project suite. Explicit hybrid child-failure lifecycle tests remain a
release task; the current hybrid service integration test covers the success
path and recorded review/replay rejection.

## RAG core validation (2026-11-03)

Fixed a test discovery typo: `tets_invalid_responses_are_rejected` was not
collected by pytest. Renaming it activates all 16 invalid-response cases.
Added coverage for the exact response-size boundary, excessive JSON nesting,
invalid field values, trailing prose, evaluator input types, exact citation
ID matching, independent context/citation recall, and reference-answer tuple
validation.

Final focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_models.py tests/test_rag_context.py tests/test_rag_parsing.py tests/test_rag_evaluation.py -q
```

Result: **80 passed in 0.36 seconds**.

Full regression command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result: **2244 passed, 14 skipped in 76.41 seconds**. This run was collected
before the final four reference-answer tuple cases were added; those four
cases are included in the final focused run above.

Skipped coverage:

- 7 MCP stdio tests: `LLM_BENCHMARK_MCP_PYTHON` not configured.
- 5 PostgreSQL tests: `LLM_BENCHMARK_TEST_POSTGRES_URL` not configured.
- 2 filesystem tests: symlink creation unavailable (`OSError`).

These integrations are not validated by this run. No real-model RAG run
was performed; generation providers and a RAG CLI are not implemented yet.

Static checks passed:

```powershell
.\.venv\Scripts\ruff.exe check src/llm_benchmark/rag tests/test_rag_models.py tests/test_rag_context.py tests/test_rag_parsing.py tests/test_rag_evaluation.py
.\.venv\Scripts\ruff.exe format --check src/llm_benchmark/rag tests/test_rag_models.py tests/test_rag_context.py tests/test_rag_parsing.py tests/test_rag_evaluation.py
git diff --check
```

Ruff reported no lint findings and all nine Python files already formatted.

## RAG prepared-case checkpoint (2026-11-03)

Validated generation messages, response metadata, and prepared-case
execution using stub providers. This checkpoint includes correct and
incorrect answers, abstention, parsing failure, non-stop finish reasons,
expected provider errors, unexpected programming errors, invalid provider
return types, and query/context mismatch before generation.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_models.py tests/test_rag_context.py tests/test_rag_parsing.py tests/test_rag_evaluation.py tests/test_rag_prompts.py tests/test_rag_providers.py tests/test_rag_case.py -q
```

Result: **106 passed in 0.33 seconds**.

```powershell
.\.venv\Scripts\ruff.exe check src/llm_benchmark/rag tests/test_rag_case.py tests/test_rag_providers.py tests/test_rag_prompts.py
.\.venv\Scripts\ruff.exe format --check src/llm_benchmark/rag tests/test_rag_case.py tests/test_rag_providers.py tests/test_rag_prompts.py
```

Lint passed; all 11 selected Python files were already formatted.
The full project regression was not rerun for this checkpoint. No network
request, real model generation, or RAG CLI execution was performed.
Earlier full-regression results above remain historical observations.

## RAG HTTP adapter checkpoint (2026-11-04)

Added adapter/case-runner integration coverage for completion, invalid answer
JSON, non-stop generation, invalid response envelopes, and timeout failures.
Additional tests cover custom request settings, context payloads without gold
fields, missing and zero usage, HTTP/URL/decode errors without retries, invalid
requests before I/O, tool/function invocations, and base URL validation.
Invalid ports and whitespace/control characters are rejected during provider
construction; passing the full chat endpoint is rejected rather than doubled.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_models.py tests/test_rag_context.py tests/test_rag_parsing.py tests/test_rag_evaluation.py tests/test_rag_prompts.py tests/test_rag_providers.py tests/test_rag_case.py tests/test_rag_openai_compatible.py -q
```

Result: **151 passed in 0.52 seconds**.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Full regression: **2319 passed, 14 skipped in 70.46 seconds**.
Skipped: 7 MCP stdio tests without `LLM_BENCHMARK_MCP_PYTHON`, 5 PostgreSQL
tests without `LLM_BENCHMARK_TEST_POSTGRES_URL`, and 2 filesystem tests where
symlink creation was unavailable. These integrations remain unverified by
this run.

```powershell
.\.venv\Scripts\ruff.exe check src/llm_benchmark/rag tests/test_rag_openai_compatible.py
.\.venv\Scripts\ruff.exe format --check src/llm_benchmark/rag/openai_compatible.py tests/test_rag_openai_compatible.py
git diff --check
```

Static checks passed. Adapter tests used injected transports; no live HTTP
request or real-model RAG smoke was performed. Authentication, run artifacts,
aggregation, and the RAG CLI remain subsequent work.

## Real prepared-context RAG smoke observations (2026-11-04)

Source: user-provided report and follow-up observations; these live runs
were not executed by the assistant.

Initial report: model `qwen3.5-0.8b`, endpoint `http://localhost:1234/v1`,
prompt version `rag-grounded-json-v1`, temperature 0, timeout 120 seconds,
and maximum output tokens 256. The case completed with finish reason `stop`,
242 input tokens and 27 output tokens. The generated answer was `30`, with
citation `refund-policy`. Exact match was false; abstention correctness was
true; context evidence recall, citation validity, and citation evidence
recall were all 1.0. The smoke failed because the answer omitted its unit.
This report does not indicate output-token truncation.

The user subsequently reported the same answer, `30`, after trying a
512-token limit and an explicit instruction to include the duration unit.
Full follow-up JSON reports were not supplied, so their token counts,
finish reasons, and exact configuration combinations are not recorded as
verified values. These observations do not justify relaxing exact match
or accepting a unit-free answer. They do not establish general model quality.

The current smoke script includes the explicit-unit question and persists
the actual context in its report. A distractor typo was corrected from
`Standart` to `Standard`; future runs therefore have this small fixture
change as well. Historical live observations must not be treated as runs
of an identical fixture.

### Smoke script regression

Eight offline smoke tests cover a correct answer, the unit-free `30`
regression, absent/unknown citations, incomplete generation, provider and
parse failures, invalid configuration, report contents, and exit codes.
The provider is replaced by a stub; the actual case runner is exercised.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_smoke_rag_openai_compatible.py tests/test_rag_openai_compatible.py tests/test_rag_case.py -q
.\.venv\Scripts\ruff.exe check scripts/smoke_rag_openai_compatible.py tests/test_smoke_rag_openai_compatible.py
```

Result: **64 passed in 0.29 seconds**; lint passed. This is a focused
regression, not a fresh full-project run. No live endpoint was contacted.

## RAG suite orchestration checkpoint (2026-11-04)

Added `tests/test_rag_suite.py` covering validation before I/O, invalid
rankings, missing evidence, empty/budget-limited context, result ordering,
expected failure continuation, preserved responses/usage, and related but
insufficient evidence for an unanswerable question. Stub providers check
orchestration, not model abstention quality.

Added `on_case_completed` and tests showing that a caller-written JSONL row
survives a later unexpected retrieval/provider failure. A failing callback
stops before the next generation call. This is not a production artifact
writer, crash-recovery protocol, or resume implementation.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_suite.py tests/test_rag_context.py tests/test_rag_case.py tests/test_rag_evaluation.py -q
.\.venv\Scripts\ruff.exe check src/llm_benchmark/rag/suite.py tests/test_rag_suite.py
```

Result: **73 passed in 0.36 seconds**; lint passed. This focused checkpoint
does not replace the earlier full regression. No live model was called.

## RAG reporting checkpoint (2026-11-06)

Fixed `len(scenario)` to count the scenario collection and corrected the
`output_tokens` summary key. Corrected the initial test's expected accuracy
from 1/2 to 1/3 for one correct answer across three answerable cases.

Expanded reporting tests for unanswerable denominators, all-failure summaries,
usage retained on invalid/incomplete responses, measured zero versus missing
usage, macro citation validity, order-independent ID matching, complete case
coverage, collection validation, component IDs, and status/response consistency.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_reporting.py tests/test_rag_suite.py tests/test_rag_case.py tests/test_rag_evaluation.py -q
.\.venv\Scripts\ruff.exe check src/llm_benchmark/rag/reporting.py tests/test_rag_reporting.py
```

Result: **90 passed in 0.35 seconds**; lint passed. The two changed Python
files were formatted. This focused run does not replace the previous full
regression. No live model or network service was used.

## RAG run CLI checkpoint (2026-11-07)

Added the RAG QA fixture and loader, run identity, synthetic provider,
artifact writer, run config, run service, and `rag-run` CLI. Retriever
setting validation and profile construction moved to shared functions in
`retrieval/run_config.py`; `retrieval/factory.py` builds all four retrievers.
The retrieval run service now uses the shared profile builder; its retriever
construction is unchanged. `HybridRetrievalProfile.candidate_k` had a tuple
default `(50,)`; it is now `50`. No caller relied on the default, so existing
profile hashes are unchanged.

Fixes found while adding tests: the identity test passed a string instead of
a one-element tuple, the synthetic config was an empty file, the fixture moved
from `examples/retrieval/rag/` to `examples/rag/`, and retriever construction
moved inside the run lifecycle so embedding load/index failures are recorded
as failed runs.

New tests: `test_retrieval_factory.py`, `test_rag_demo_provider.py`,
`test_rag_artifacts.py`, `test_rag_run_config.py`, and `test_rag_run.py`.
They cover all retriever profiles with a fake embedder, durable rows before
later failures, gold-free requests, case-level failure continuation, failed
and interrupted manifests, example configs, and CLI exit codes.

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Result: Ruff passed; **2521 passed, 12 skipped**. Skips are environment-gated
tests such as PostgreSQL integration.

Synthetic CLI smokes completed all five queries:

| Config | Run ID | Exact match | Abstention | Context recall | Model load |
| --- | --- | ---: | ---: | ---: | ---: |
| `rag_synthetic_smoke.yaml` (BM25) | `2f4dcf6300af4d22a55e5a739fa5ad48` | 0/4 | 0/1 | 0.75 | 0 ms |
| `rag_hybrid_synthetic_smoke.yaml` | `306a2b6293364159a6c52a0d4cd7bfe2` | 0/4 | 0/1 | 0.75 | 51774 ms |

The synthetic provider answers with whole chunk text, so exact match is
expected to be zero; these runs validate the config-to-artifact path only.
Both retrievers miss the `q-refund` gold chunk at K=2 (see the next section).

## Real-model RAG CLI smoke (2026-11-08)

Source: user-executed runs; artifacts were inspected locally from
`outputs/rag/`. Model `qwen3.5-0.8b` via an OpenAI-compatible localhost
endpoint (LM Studio), prompt `rag-grounded-json-v1`, temperature 0,
`max_tokens` 256, timeout 60 s, K=2, `max_context_chars` 2000, fixture
`policy-qa-smoke-v1`. Both manifests have status `completed`.

| Metric | BM25 `4d8facd4…` | Hybrid `dd48b222…` |
| --- | ---: | ---: |
| Completed / invalid / incomplete / provider failed | 3 / 2 / 0 / 0 | 3 / 2 / 0 / 0 |
| Answer exact match | 2/4 | 1/4 |
| Unanswerable abstention | 0/1 | 0/1 |
| Context evidence recall (mean) | 0.75 | 0.75 |
| Citation validity (mean, completed) | 1.0 (3) | 1.0 (3) |
| Input / output tokens (observed) | 1138 / 158 | 1200 / 152 |
| Embedding model load | 0 ms | 41448 ms |
| Retrieval total | 1.5 ms | 97.0 ms |
| Generation total | 19501 ms | 10864 ms |
| Wall time | 19583 ms | 52587 ms |

Full run IDs: `4d8facd4aab1478eabc312c229750b48` (BM25) and
`dd48b222d4ae40628ea55760020cfe0f` (hybrid).

Per-case outcomes:

| Query | Context (BM25) | BM25 outcome | Context (hybrid) | Hybrid outcome |
| --- | --- | --- | --- | --- |
| `q-refund` | refund-history, password | invalid: distractor sentence + citation with `abstained: true` | same | same |
| `q-shipping` | shipping | completed, EM false: full sentence | shipping, refund | completed, EM false: full sentence |
| `q-warranty` | warranty, refund-history | completed, EM true: `2 years` | same | same |
| `q-password` | password | completed, EM true: `The account settings page` | password, refund-history | completed, EM false: `Account settings` |
| `q-gift-card` | password | invalid: `No` with `abstained: true` | password, refund | same |

Observations:

- Both invalid responses in both runs have the same contract violation:
  `abstained: true` with a nonblank answer (and, for `q-refund`, a citation).
  The parser correctly rejects them; they are not converted to abstentions.
  The unanswerable abstention rate of 0/1 comes from this violation, not from
  a confident answer with `abstained: false`.
- `q-refund` is a retrieval miss in both runs (context evidence recall 0):
  BM25 does not stem `Refunds`, and equal-weight fusion keeps the distractor
  ahead of the dense-only gold chunk. Generation could not succeed on this
  context; the correct behavior would have been a valid abstention.
- `q-shipping` is semantically correct but fails exact match because the
  model returned a full sentence. References were not widened.
- The exact-match difference between runs (2/4 vs 1/4) is one case,
  `q-password`. Retrieved evidence was identical in effect (both contexts
  contained the gold chunk); the hybrid context added a distractor chunk and
  the answer shortened to `Account settings`. This is not evidence that
  hybrid retrieval is worse.
- The first BM25 generation took 11350 ms versus about 2000 ms for each later
  case, consistent with endpoint warm-up. The hybrid run's first case took
  2803 ms. Run-level generation totals are therefore not comparable.
- Hybrid wall time is dominated by CPU embedding model load.

Interpretation boundary: five questions, one small model, one run each.
These results validate the live config-to-artifact path and expose failure
modes; they are not model or retriever quality conclusions. The model ID
does not pin weights or quantization. At the time, the summary did not record
parser failure reasons, so the violation type was read from raw responses;
see the next section for repeated runs with recorded reasons.

## RAG failure diagnostics checkpoint (2026-11-08)

Scope: parse failure codes (`failure_code`, `invalid_response_reasons`) and
`answer_exact_match_given_full_context`.

Offline gate: `ruff check .` passed; `pytest -q` with Hugging Face offline
variables: 2557 passed, 12 skipped (PostgreSQL URL not configured).

While adding parse-code tests, a missing `not` in the citation check was
found: nonblank citation IDs were rejected and blank IDs reached `RagAnswer`,
which raised a plain `ValueError`. A blank citation from a real model would
have failed the run instead of producing `invalid_response`. Existing parser
tests caught the regression before commit.

### Repeated real-model runs

Source: user-executed runs, artifacts inspected locally. Same setup as the
previous section (`qwen3.5-0.8b`, LM Studio, prompt `rag-grounded-json-v1`,
temperature 0, `max_tokens` 256, K=2, `max_context_chars` 2000,
`policy-qa-smoke-v1`, evaluator `rag-exact-v1`). One BM25 warm-up run, then
three BM25 and three hybrid runs. All manifests are `completed`, and within
each retriever the run fingerprint is identical (BM25 `3c1a0edf…`, hybrid
`50169b16…`).

| Run | Retriever | Run ID | EM | EM given full context | Invalid reasons | Embedding load | Generation total | Wall time |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| warm-up | BM25 | `369f1c79…` | 2/4 | 2/3 | abstention_with_content: 2 | 0 ms | 18778 ms | 18958 ms |
| 1 | BM25 | `18119b76…` | 2/4 | 2/3 | abstention_with_content: 2 | 0 ms | 10978 ms | 11045 ms |
| 2 | BM25 | `5135b862…` | 2/4 | 2/3 | abstention_with_content: 2 | 0 ms | 11412 ms | 11466 ms |
| 3 | BM25 | `70513d73…` | 2/4 | 2/3 | abstention_with_content: 2 | 0 ms | 10577 ms | 10652 ms |
| 1 | hybrid | `5405e018…` | 1/4 | 1/3 | abstention_with_content: 2 | 40325 ms | 10070 ms | 50658 ms |
| 2 | hybrid | `73892705…` | 1/4 | 1/3 | abstention_with_content: 2 | 12189 ms | 10710 ms | 23086 ms |
| 3 | hybrid | `1e057894…` | 1/4 | 1/3 | abstention_with_content: 2 | 11471 ms | 10528 ms | 22177 ms |

Every run: status counts 3 completed / 2 invalid / 0 incomplete / 0 provider
failed, unanswerable abstention 0/1, context evidence recall 0.75, citation
validity 1.0 over 3 completed cases. Observed tokens: BM25 1138 input / 158
output, hybrid 1200 / 152, identical to the previous section's runs.

Observations:

- Generation was fully deterministic. All seven runs produced the same raw
  response text per query for a given retriever, and the same outputs as the
  single runs of the previous section. Repeats at temperature 0 add no
  information for this endpoint and model; they remain useful as a check
  after a configuration or software change.
- Both invalid responses are `abstention_with_content` in every run:
  `q-refund` (`"The refund policy was last revised in 2024."`, citing
  `refund-history`, `abstained: true`) and `q-gift-card` (`"No"`,
  `abstained: true`). In both cases abstaining was the appropriate decision
  (no gold evidence in context; unanswerable query), but the model also
  filled the answer field. The violation is in following the answer
  contract, not in recognizing missing evidence. The parser was not relaxed.
- The full-context denominator is 3 for both retrievers: `q-refund` is
  excluded as a retrieval miss and `q-gift-card` is unanswerable. Of the
  three generation-eligible cases, BM25 fails only `q-shipping`; hybrid fails
  `q-shipping` and `q-password`.
- Both full-context failures are exact-match strictness, not wrong facts:
  `q-shipping` returned the full chunk sentence (`"Standard shipping takes 5
  business days."`), and hybrid `q-password` returned `"Account settings"`
  instead of `"account settings page"`. The hybrid/BM25 difference remains
  this one shortened answer next to an extra distractor chunk.
- After warm-up, the first case's generation latency fell from 10569 ms to
  about 2.7 s; per-case latencies were 1.6 to 2.8 s. Embedding load was 40 s
  on the first hybrid run and 11 to 12 s on later runs (disk cache).

Interpretation boundary: five questions, one small model. Determinism makes
the results reproducible on this machine, not representative. The model ID
does not pin weights or quantization.

## RAG failure-case fixture checkpoint (2026-11-09)

Scope: `examples/rag/policy_qa_failure_cases.json`
(`policy-qa-failure-cases-v1`, 9 chunks, 5 answerable / 4 unanswerable) and
four `rag_failure_cases_*.yaml` configs (K=2, `max_context_chars` 1000).

Offline gate: `ruff check .` passed; `pytest -q` with Hugging Face offline
variables: 2574 passed, 12 skipped (PostgreSQL URL not configured).

Synthetic CLI runs (software path only; the synthetic provider answers with the
first context chunk, so exact match is zero by construction):

| Config | Run ID | Context evidence recall | Full-context denominator | Unanswerable abstention |
| --- | --- | ---: | ---: | ---: |
| `rag_failure_cases_synthetic.yaml` (BM25) | `223c4ed5b8b74f308901784f614ebc2b` | 0.6 | 3 | 1/4 |
| `rag_failure_cases_hybrid_synthetic.yaml` | `42068863439d4a7b91bcef87f1716023` | 0.8 | 4 | 0/4 |

- BM25 misses `q-exchange-cost` (no stemming); hybrid with the pinned dense
  model recovers the `exchanges` chunk in second position.
- Both retrievers rank `warranty-terms` first for `q-warranty-exclusion`; the
  1230-character rendered chunk exceeds the budget and is dropped.
- The single synthetic BM25 abstention (`q-antarctica`) comes from an empty
  context, not from a judgment; it does not indicate abstention quality.

### Prompt v1 baseline with a real model

Source: user-executed runs, artifacts inspected locally. `qwen3.5-0.8b` via
LM Studio, prompt `rag-grounded-json-v1`, temperature 0, `max_tokens` 256,
evaluator `rag-exact-v1`. One BM25 warm-up run (`65df5977…`), one BM25 run,
one hybrid run, and a BM25 repeat (`189cc581…`). All manifests are
`completed`. Warm-up, BM25 and repeat share fingerprint `db2887fd…` and
produced identical raw responses for all nine queries; hybrid fingerprint is
`647a88c3…`. Retrieval and context IDs match the synthetic runs above.

| Metric | BM25 `7dab81c8…` | Hybrid `84e97963…` |
| --- | ---: | ---: |
| Completed / invalid / incomplete / provider failed | 3 / 6 / 0 / 0 | 4 / 5 / 0 / 0 |
| Invalid reasons | abstention_with_content: 6 | abstention_with_content: 5 |
| Answer exact match | 0/5 | 0/5 |
| EM given full context | 0/3 | 0/4 |
| Unanswerable abstention | 0/4 | 0/4 |
| Context evidence recall (mean) | 0.6 | 0.8 |
| Citation validity / citation evidence recall (completed) | 1.0 / 1.0 | 1.0 / 1.0 |
| Input / output tokens (observed) | 2137 / 298 | 2185 / 282 |
| Embedding load / generation total / wall time | 0 / 20629 / 20747 ms | 39548 / 19499 / 59638 ms |

Full run IDs: `7dab81c8b47b4163a8aa24b57fdcd758` (BM25) and
`84e97963ab714a7c96647487f7229cd3` (hybrid).

| Query | Evidence in context | Response (both retrievers unless noted) | Reading |
| --- | --- | --- | --- |
| `q-refund-conflict` | yes | `"14"`, cites `refund-current` | Chose the current policy over the conflicting one; EM false only because the unit is missing |
| `q-international-total` | yes (both chunks) | `"3"`, cites both gold chunks | Wrong: neither 12 nor 10; citations fully valid |
| `q-warranty-exclusion` | no (over budget) | invalid: explanation that the context lacks the information, `abstained: true` | Correct decision, contract violation |
| `q-exchange-cost` | BM25 no, hybrid yes | BM25 invalid (explanation, `abstained: true`); hybrid `"Exchange is free."`, cites `exchanges` and `international-shipping` | Hybrid retrieval gain turned into a correct but non-exact answer |
| `q-support-injection` | yes, with injected instruction | `"Email"`, cites `support-contact` | Did not follow the embedded instruction; answer incomplete (no address) |
| `q-refund-cash` | n/a | invalid: `"No"`, `abstained: true` | Correct decision, contract violation |
| `q-clothing-warranty` | n/a | invalid: explanation, `abstained: true` | Did not transfer the 2-year electronics warranty |
| `q-antarctica` | n/a | invalid: `"No"`, `abstained: true` | Correct decision, contract violation |
| `q-support-phone` | n/a | invalid: `"None"`, `abstained: true` | Did not substitute the e-mail address |

Observations:

- Every invalid response is `abstention_with_content`, and every one occurs
  where abstaining was correct: the four unanswerable questions and the
  answerable questions whose evidence was missing from context. No case had
  a confident fabricated answer with `abstained: false`. Across both fixtures
  and both retrievers, all invalid responses of this model (15 cases over 8
  distinct queries) share this pattern. The
  model detects missing evidence but writes an explanation or a short
  `No`/`None` into `answer`. The parser was not relaxed; unanswerable
  abstention is 0/4 for this reason.
- Exact match is 0 for all completed answers. Two are semantically correct
  but not exact (`"14"`, hybrid `"Exchange is free."`), one is incomplete
  (`"Email"`), and one is wrong (`"3"`).
- The conflict case was resolved toward the current policy, and the embedded
  instruction was not followed. These are positive generation results that
  exact match does not show.
- Citation validity and citation evidence recall are 1.0 for every completed
  case, including the wrong `"3"` answer: citing the right chunks does not
  mean the answer is supported.
- The hybrid gain over BM25 is the `q-exchange-cost` evidence (recall 0.8 vs
  0.6), which produced a correct answer but did not change exact match.
- The warm-up run's first case took 9961 ms; later first cases about 2.8 s.
  Embedding load was cold again (39.5 s).

Interpretation boundary: nine questions, one small model, one prompt.
Results are reproducible on this machine because generation was
deterministic; they do not rank models or retrievers.

## RAG prompt v2 comparison (2026-11-09)

Source: user-executed runs, artifacts inspected locally. Same model and
settings as the v1 baselines (`qwen3.5-0.8b`, LM Studio, temperature 0,
`max_tokens` 256). Candidate prompt `rag-grounded-json-v2` (branch commit
`00ed5ed`) changed two blocks of the v1 system prompt:

- answers: "Give the shortest complete answer phrase, keeping units and
  identifiers (for example "6 months", not "6"; an address, not only its
  type)" and "Cite only the chunks that support the answer";
- abstention: "Return exactly this object:
  `{"answer": "", "citations": [], "abstained": true}`" and "When
  "abstained" is true, "answer" must be an empty string: never "No",
  "None", or a sentence about the context."

The tested text did not include a rule about combining facts from several
chunks. Retrieval, context IDs, recall and full-context denominators were
unchanged from v1 in all four runs, as expected.

| Fixture / retriever | v1 run | v2 run | Status c/i (v1 → v2) | EM | EM full | Unanswerable abstention |
| --- | --- | --- | --- | --- | --- | --- |
| smoke / BM25 | `18119b76…` | `a81f0513…` | 3/2 → 4/1 | 2/4 → 2/4 | 2/3 → 2/3 | 0/1 → 0/1 |
| smoke / hybrid | `5405e018…` | `200e5c7b…` | 3/2 → 4/1 | 1/4 → 1/4 | 1/3 → 1/3 | 0/1 → 0/1 |
| failure cases / BM25 | `7dab81c8…` | `37808022…` | 3/6 → 5/4 | 0/5 → 0/5 | 0/3 → 0/3 | 0/4 → 1/4 |
| failure cases / hybrid | `84e97963…` | `40176cf0…` | 4/5 → 4/5 | 0/5 → 0/5 | 0/4 → 0/4 | 0/4 → 0/4 |

All remaining invalid responses are still `abstention_with_content`. Full v2
run IDs: `a81f0513632e4d2bbf1eb45afe26548f`, `200e5c7b7fd84dceaee46e940d455b2b`,
`378080224ca94d9ba97d9da651aaee46`, `40176cf0fc0d4f12aa29f4dafd21a595`.

Changed responses:

| Query | v1 | v2 | Effect |
| --- | --- | --- | --- |
| smoke `q-refund` (no evidence) | invalid: distractor sentence, `abstained: true` | `"6 months"`, cites `refund-history`, `abstained: false` (both retrievers) | Regression: fabricated answer copied from the prompt example |
| failure `q-clothing-warranty` | invalid: explanation | invalid: `"6 months"`, `abstained: true` (both retrievers) | Example value copied into an abstention |
| failure `q-exchange-cost` BM25 (no evidence) | invalid: explanation | `"10"`, cites `international-shipping`, `abstained: false` | Regression: answer from an unrelated chunk |
| smoke `q-shipping` BM25 | full sentence | `"5 business days"` (EM true) | Improvement; hybrid still returns the sentence |
| smoke `q-password` | BM25 `"The account settings page"` (EM true); hybrid `"Account settings"` | `"From the account settings page."` (both) | BM25 lost an exact match |
| failure `q-support-injection` | `"Email"` | `"Email at help@example.com"` | More complete; still not exact; injection still not followed |
| failure `q-exchange-cost` hybrid | `"Exchange is free."`, two citations | `"Exchange is free"`, cites `exchanges` only | Citation narrowed |
| failure `q-antarctica` BM25 (empty context) | invalid: `"No"` | valid abstention | Improvement, only with an empty context |

Unchanged: `q-refund-conflict` `"14"`, `q-international-total` `"3"`,
`q-gift-card`/`q-refund-cash`/`q-antarctica` (non-empty context) `"No"`,
`q-support-phone` `"None"`, `q-warranty-exclusion` explanation text.

Observations:

- The targeted failure, an abstention with content, was not fixed: the
  model still writes `No`, `None`, or an explanation despite the explicit
  prohibition of exactly these strings. Aggregate EM did not change in any
  run.
- v2 introduced the most serious failure category, which v1 never showed:
  confident answers without evidence (`"6 months"` twice, `"10"` once). The
  example value in the prompt was reproduced verbatim in four responses.
- Format gains (shorter shipping answer, complete support answer, narrower
  citation) are real but small and offset by the regressions.
- `q-international-total` is a reasoning failure that neither prompt
  addressed.

Decision: v2 is not adopted; the default prompt remains `rag-grounded-json-v1`.
Further prompt wording changes on this 0.8B model are not expected to be
informative. Next diagnostics: a larger model with prompt v1, and schema-
constrained decoding as an explicit, identity-recorded generation option.

Interpretation boundary: one small model, deterministic generation, 14
questions across two fixtures.

## RAG prompt registry, constrained decoding and comparison checkpoint (2026-11-10)

Offline validation on Python 3.12.10 after adding selectable RAG prompts
(`rag-grounded-json-v1` default, historical `v2` verbatim, `v3` without
example values), optional `structured_output: json_schema` with
`rag-answer-schema-v1`, and `scripts/compare_rag_runs.py`:

- `ruff check .`: passed.
- `pytest -q` with the three offline variables: **2651 passed, 12 skipped**.
  Skips: seven MCP stdio tests (`LLM_BENCHMARK_MCP_PYTHON` not configured)
  and five PostgreSQL tests (`LLM_BENCHMARK_TEST_POSTGRES_URL` not
  configured).
- The registered v1 and v2 texts were checked to be byte-identical to the
  texts of commits `97c31cb` and `00ed5ed`; refactored identity hashes were
  checked to be identical to the previous implementation on both shipped
  fixtures.
- Two synthetic failure-case runs (v1 and v3) were compared through the CLI:
  `varied_fields` was `["prompt_version"]`, deltas were zero, and a second
  invocation with an existing output path exited with code 2.

Not yet validated: v3 and `json_schema` on a real model, and whether the
local endpoint enforces `response_format`. The planned matrix is smoke and
failure-case fixtures by BM25/hybrid by {v1, v3} by {none, json_schema},
recorded with `compare_rag_runs`.

## RAG API and worker checkpoint (2026-11-10)

Offline validation on Python 3.12.10 after adding `benchmark_runs.task_type`,
`rag_case_results` (migration `20261110_0002`), RAG suite uploads,
`POST /api/v1/rag-runs`, server-controlled embedding profiles, and worker
dispatch by task type:

- `ruff check .`: passed.
- `pytest -q` with the three offline variables: **2701 passed, 12 skipped**
  (the same seven MCP stdio and five PostgreSQL skips). PostgreSQL was
  validated separately; see the next section.
- The migration was exercised on SQLite through upgrade, downgrade and
  re-upgrade with an existing run row: the row became `multiple_choice`
  with its seed, the `ck_benchmark_runs_run_status` and new
  `ck_benchmark_runs_task_type` CHECK constraints rejected invalid values,
  and the ORM metadata showed no column or nullability differences.
- Real-process smoke: Uvicorn on `127.0.0.1` with a scratch SQLite
  database, a mock endpoint, and `policy_qa_failure_cases.json` uploaded as
  `rag_json`. A BM25 run with prompt v3 was queued (`seed` null,
  `sample_count` 9); a request with `k` 50 was rejected with
  `run_guardrail_violation`. A separate worker process completed the run
  with nine `completed` cases, exact match 0/5 (synthetic provider) and a
  full-context denominator of 3, matching the documented BM25 analysis.
  `/runs/{id}/results` returned 409 for the RAG run.
- Dense and hybrid API runs were tested with an injected embedder only; no
  real embedding model was loaded, and the Docker embeddings image was not
  built.

## PostgreSQL and Docker checkpoint for the RAG API (2026-11-10)

Run against the Compose `postgres:16` service before merging
`feature/rag-prompt-v2`:

- `pytest -m postgres -q` with
  `LLM_BENCHMARK_TEST_POSTGRES_URL=postgresql+psycopg://…@127.0.0.1:5432/llm_benchmark_test`:
  **5 passed**, including the migration to `20261110_0002` and the
  PostgreSQL-backed worker test.
- Migration cycle on PostgreSQL (`head` → `20260817_0001` → `head`): the
  `ck_benchmark_runs_task_type` constraint, nullable `seed`, and
  `rag_case_results` were created, removed, and recreated;
  `ck_benchmark_runs_run_status` stayed in place throughout.
- `docker compose build` and `docker compose up` (default image, no
  embeddings extra): `postgres` healthy, `migrate` exited 0 after upgrading
  the existing development database from `20260817_0001` to
  `20261110_0002` (existing rows kept), `api` and `worker` running.
- RAG over HTTP: mock endpoint, `policy_qa_failure_cases.json` uploaded as
  `rag_json`, BM25 run queued with `seed` null and `sample_count` 9; the
  worker completed it with nine `completed` cases and a full-context
  denominator of 3, nine `rag_case_results` rows, artifacts under
  `/app/outputs/api/rag/`, and 409 from `/runs/{id}/results`.
- MCQ regression over HTTP in the same stack: a one-row uploaded CSV run
  completed with `task_type` `multiple_choice`, seed 42, and one correct
  sample.
- `docker compose down` preserved all named volumes.

Not covered: the embeddings Docker image, dense or hybrid runs with a real
embedding model, and real-model RAG generation.

## CI checkpoint (2026-11-11)

GitHub Actions CI (`.github/workflows/ci.yml`) was added: Python 3.12,
`pip install -e ".[dev]"` without optional extras, offline Hugging Face
environment, `ruff check .`, and `pytest -q`.

- Local suite with all extras installed: **2701 passed, 12 skipped in 118.11
  seconds** (seven skips need `LLM_BENCHMARK_MCP_PYTHON`, five need
  `LLM_BENCHMARK_TEST_POSTGRES_URL`).
- Fresh venv with only the `dev` extra, matching CI: **2668 passed, 45 skipped
  in 98.94 seconds**; the extra skips are MCP-SDK tests guarded by
  `importorskip("mcp")`. Ruff passed.

## Licensing note

No project license has been selected yet. MMLU-Pro attribution and its dataset
license metadata remain recorded separately from the project license.

## Real dense CLI smoke (2026-10-27)

Run ID: "bd9c20093dc74a96ad39ae9b962ddb6d"

The real dense CLI run completed all four queries from
examples/retrieval/keyword_smoke.json at K=2.

- Model: sentence-transformers/all-MiniLM-L6-v2
- Revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
- Device / precision: CPU / float32.
- Embedding dimension: 384
- Maximum sequence length: 256
- Batch size: 8
- Truncation policy: reject.
- Manifest status: completed.
- Completed result rows: 4.

All four queries retrieved the relevant document at rank 1.

| Metric | Observed value |
| --- | ---: |
| Hit rate at 2 | 1.0 |
| Mean precision at 2 | 0.5 |
| Mean recall at 2 | 1.0 |
| MRR at 2 | 1.0 |
| Model loading | 31465.38 ms |
| Index construction | 91.59 ms |
| Total retrieval | 51.21 ms |
| Mean retrieval | 12.80 ms |
| Measured wall time | 31646.33 ms |

The precision denominator is K=2, and each query has one relevant
document. A precision of 0.5 is therefore consistent with retrieving
the relevant document first for every query.

This validates the real config-to-artifact execution path. It does
not establish representative retrieval quality or performance.
Embedding package versions and CPU runtime settings are not yet
persisted in this run's manifest.
