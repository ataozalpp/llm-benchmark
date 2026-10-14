# MCP calculator integration

This guide covers setup, execution, and validation of the local calculator MCP
integration. Existing calculator logic runs in a separate FastMCP process; the
benchmark discovers the tool and maps its responses into its own tool contracts.

## Current scope

- One local FastMCP server exposing `calculator` over stdio.
- Separate Python environments for the benchmark client and server.
- Legacy initialization and tools-capability checks.
- A single-page, single-tool catalog with an explicit allowlist.
- Input validation using the restricted external JSON Schema profile.
- Structured JSON object responses mapped to `ToolResult`.
- A synchronous executor backed by an open asynchronous MCP client.
- Existing tool loop and evaluation integration via `run_calculator_mcp_suite`.
- Smoke checks, executor tests, lifecycle tests, and local/MCP suite comparison.

The supplied demos use synthetic behavior and do not call a real LLM. Obtaining
`17 * 23 = 391` validates software integration, not model quality. MCP runtime
selection is supported by the `tool-run` CLI. API and worker integration remain
outside this path.

## Architecture and files

```text
Benchmark environment (.venv)
    demo_mcp_suite
    -> mcp_tool_suite: session ownership and discovery
    -> AnyIO worker thread: synchronous external tool suite
       -> tool loop and argument validation
       -> mcp_executor: dispatch to the async client event loop
          -> stdio
Server environment (.venv-mcp)
    demo_mcp_server: FastMCP
    -> CalculatorArguments -> calculator_handler
    -> structured JSON response
Benchmark environment
    -> mcp_mapping -> ToolResult -> evaluation -> suite summary
```

| File | Responsibility |
|---|---|
| `src/llm_benchmark/demo_mcp_server.py` | Exposes the calculator as a FastMCP tool |
| `src/llm_benchmark/demo_mcp_smoke.py` | Checks server behavior in the same process |
| `src/llm_benchmark/mcp_client.py` | Trusted local launch profile, session, and catalog admission |
| `src/llm_benchmark/mcp_launch.py` | Resolves trusted local server Python and project-root settings |
| `src/llm_benchmark/mcp_runtime.py` | Prepares an open session, executor, catalog snapshot, and protocol metadata |
| `src/llm_benchmark/mcp_mapping.py` | SDK-independent descriptor and result mapping |
| `src/llm_benchmark/mcp_errors.py` | Tool-request infrastructure exception types |
| `src/llm_benchmark/mcp_executor.py` | Bridges synchronous execution to an open async client |
| `src/llm_benchmark/mcp_tool_suite.py` | Owns the session and runs the external tool suite |
| `src/llm_benchmark/demo_mcp_stdio_smoke.py` | End-to-end smoke check with a real subprocess |
| `src/llm_benchmark/demo_mcp_suite.py` | Runs synthetic scenarios and prints a JSON summary |
| `tests/test_mcp_mapping.py` | Pure mapping and output-budget checks |
| `tests/test_mcp_client.py` | Single-page catalog acceptance and rejection |
| `tests/test_mcp_executor.py` | Execution, timeout, error mapping, and input-contract checks |
| `tests/test_mcp_tool_suite.py` | Context exit on success and discovery/provider/executor failures |
| `tests/test_mcp_tool_run_lifecycle.py` | Manifest completion after session closure and sanitized preparation/execution failures |
| `tests/test_tool_loop_time_budget.py` | Remaining-budget forwarding, expired scenarios, early timeouts, and legacy executor compatibility |
| `tests/test_mcp_stdio.py` | Real connection, invalid inputs, tool errors, and suite comparison |
| `tests/fixtures/mcp_disconnect_server.py` | Test-only server that exits during a tool request |
| `requirements-mcp-server.txt` | Pins the server framework version |

## Why separate environments?

The backend FastAPI dependencies and the selected FastMCP 4.0.4 server stack
require incompatible Starlette version ranges. The MCP SDK client runs in
`.venv`; the FastMCP server runs in `.venv-mcp`. They communicate through stdio
rather than sharing Python objects.

Keep FastMCP out of the backend environment. The server requirements file pins
the framework version; it is not a complete transitive dependency lock.

## Setup: Windows PowerShell

Run commands from the repository root. The project requires Python 3.12.
If the backend environment does not exist, create it:

```powershell
py -3.12 -m venv .venv
```

Install the backend, test tools, and MCP client dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,mcp]"
```

Create the server environment if it does not exist, then install its dependencies:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-mcp
.\.venv-mcp\Scripts\python.exe -m pip install -r requirements-mcp-server.txt
```

The full backend does not need to be installed in the server environment: the
commands below make project sources available through `PYTHONPATH`. Dependency
installation may require network access; the demos do not require a model service.

Check the environments independently:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv-mcp\Scripts\python.exe -m pip check
```

## 1. In-process smoke check

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv-mcp\Scripts\python.exe -m llm_benchmark.demo_mcp_smoke
```

Expected message: `FastMCP calculator smoke passed.`

This checks tool registration and invocation, not a separate process or stdio
transport. `PYTHONPATH` is set for the current terminal session.

## 2. Real stdio smoke check

```powershell
.\.venv\Scripts\python.exe -m llm_benchmark.demo_mcp_stdio_smoke --server-python .\.venv-mcp\Scripts\python.exe
```

The client starts the server; do not launch it manually in another terminal.
The first executable is the backend Python; `--server-python` selects the
server Python. The client sets the subprocess working directory, `PYTHONPATH`,
and `PYTHONUNBUFFERED`.

Expected message format:

```text
MCP stdio smoke passed. Protocol: <negotiated-version>; result: 391.
```

When running from another directory, pass the repository root with
`--project-root` and use appropriate executable paths. Starting the server
directly makes it wait for protocol messages on stdin; it does not create an
HTTP endpoint or browser URL.

## 3. Synthetic MCP suite

```powershell
.\.venv\Scripts\python.exe -m llm_benchmark.demo_mcp_suite --server-python .\.venv-mcp\Scripts\python.exe
```

This runs the existing scenarios through `DemoToolProvider`: calculator
multiplication and a direct `READY` response without tool use. It exercises
the tool loop, external argument validation, MCP executor, and evaluation.
The command also accepts `--project-root`.

The JSON output contains `synthetic: true`, `runtime: "mcp_stdio"`, and the
complete suite summary. On success, key summary fields are:

| Field | Expected value |
|---|---|
| `case_count` | `2` |
| `requested_call_count` | `1` |
| `executed_call_count` | `1` |
| `successful_call_count` | `1` |
| `final_answer_match` | `numerator: 2`, `denominator: 2` |

This demo prints a summary; it does not write `manifest.json`, `results.jsonl`,
or `trace.jsonl`. Use the `tool-run` command below to produce those artifacts.

## 4. MCP tool-run with artifacts

From the repository root, after installing both environments:

```powershell
$env:LLM_BENCHMARK_MCP_PYTHON = (Resolve-Path .\.venv-mcp\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m llm_benchmark tool-run --config configs/tool_mcp_smoke.yaml
```

The YAML selects `runtime: mcp_stdio` and `mcp_profile: calculator`. The server
executable is trusted local environment configuration, not a model-supplied
command. This source-checkout launch profile resolves the server module relative
to the current project root. The Python service also accepts `project_root`.

The command prints a run ID and output directory. Each run contains
`manifest.json`, `summary.json`, `results.jsonl`, and `trace.jsonl`. The manifest
records `runtime_type: mcp_stdio`, `runtime_version: mcp-calculator-v1`, and
`runtime_metadata` containing the calculator profile and negotiated protocol.
The discovered catalog and runtime metadata contribute to the run fingerprint.
Existing local config hashes exclude the new default runtime fields to preserve
compatibility. Local runs do not import the MCP SDK.

Artifact lifecycle:

1. Resolve launch settings and prepare the MCP session/catalog.
2. Create the run artifacts and a `running` manifest.
3. Run the suite through the existing result and trace callbacks.
4. Write the summary while leaving the manifest `running`.
5. Close the MCP session, then mark the manifest `completed`.

A session-close failure marks the manifest `failed`, even if a summary was
already written. Consumers must use manifest status, not summary existence,
to determine success. Execution failures retain available partial artifacts
without persisting raw exception messages. Preparation/discovery failures before
artifact creation produce no run directory. Filesystem write failures or forced
process termination cannot guarantee a final manifest update.

`summary.wall_time_ms` measures the shared artifact execution interval; it does
not include MCP launch/discovery and final session cleanup. Scenario budgets are
applied to tool-loop execution, not the complete startup/shutdown lifecycle.
The supplied config remains synthetic and does not measure a real model.

## Tests and validation

Configure the server executable so the subprocess tests actually run:

```powershell
$env:LLM_BENCHMARK_MCP_PYTHON = (Resolve-Path .\.venv-mcp\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m pytest -q tests/test_mcp_mapping.py tests/test_mcp_client.py tests/test_mcp_executor.py tests/test_mcp_tool_suite.py tests/test_mcp_stdio.py
```

Service and launch coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_mcp_launch.py tests/test_mcp_tool_run_lifecycle.py tests/test_tool_run_service.py
```

Without `LLM_BENCHMARK_MCP_PYTHON`, stdio tests are skipped. A configured path
that does not point to a file fails the tests. SDK-dependent tests may skip if
the MCP package is missing. Confirm that subprocess tests ran when recording
PR validation.

Invalid-input tests send numeric strings, booleans, out-of-range values, unknown
operations, and extra fields directly to the server. They check server-side
rejection in addition to client validation. A tool error or invalid-params
protocol error is accepted; a timeout or connection failure is not.

Suite lifecycle tests check context exit on success and discovery, provider,
and executor failures. They do not prove that a real operating-system child
process has terminated. Stdio tests separately cover real calls, division-by-zero
tool errors, and local/MCP evaluation equivalence.

The server-exit test launches a separate test-only FastMCP process. Its
`disconnect` tool exits that process during a request; the executor must report
`MCPTransportFailure`. A generic request timeout does not satisfy the assertion.
This is distinct from proving cleanup of an unresponsive process that stays
alive. The production calculator server does not expose this test tool.

Before merging, run general checks:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

The Hugging Face variables enable offline behavior in those libraries, not
OS-level network isolation. Report PostgreSQL and platform-dependent skips
separately. Record results for the actual commit in the PR rather than treating
a fixed test count in this guide as proof of current validation.

## Execution and error behavior

`mcp_tool_suite.py` owns the session. It runs the synchronous external suite in
an AnyIO worker thread; `mcp_executor.py` dispatches calls back to the open async
client. The validation wrapper checks selected-tool arguments before delegating
execution. The current calculator suite runs sequentially and shares one session.

| Condition during a tool request | Behavior |
|---|---|
| Server returns a tool-error response | Failed `ToolResult` |
| Missing or unsupported structured output | `invalid_output` result |
| Output exceeds the configured budget | `output_too_large` result |
| Request timeout | `McpExecutionTimeout` |
| Known connection or stream failure | `MCPTransportFailure` |
| Other MCP protocol error | `MCPProtocolFailure`, retaining the error code |
| Unexpected implementation exception | Propagates |

Infrastructure exceptions normally stop the suite instead of producing a completed
`ToolSuiteResult`; they are not scored as a model's incorrect answer. Session
startup and discovery errors are not all covered by the executor's request-error
classification. Async context managers may wrap errors in an `ExceptionGroup`.

### Scenario time budgets

When `ToolLoopPolicy.max_wall_time_seconds` is configured, the loop passes the
remaining duration to executors exposing `execute_with_timeout`. The external
validation wrapper subtracts argument-validation time before delegation. The
MCP executor uses the smaller of its configured request timeout and the supplied
budget. The original `execute(call)` path remains available.

`McpExecutionTimeout` also inherits the framework-independent
`ToolExecutionTimeout`. If the loop catches this exception after the scenario
deadline has expired, it returns `WALL_TIME_LIMIT`; otherwise the exception
propagates as an infrastructure failure. No fabricated successful tool result
is appended for a timed-out request.

This support is conditional on a configured scenario budget. The example
scenarios used by `demo_mcp_suite` do not themselves configure one. Executors
without timed execution retain cooperative checks between calls. Relative
duration handoffs, event-loop scheduling, cancellation, and cleanup can add
overhead; this is not a hard process-termination deadline.

Run the budget-specific checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_loop_time_budget.py tests/test_external_tool_executor.py tests/test_mcp_executor.py
```

## Limits and assumptions

- Only the `calculator` catalog is accepted; additional tools or continuation
  pages are rejected. `next_cursor=None` means there is no next page and is accepted.
- The catalog schema budget is 65,536 bytes; supported schema keywords are restricted.
- Result mapping supports structured JSON objects, not arbitrary text or media.
- The default output budget is 65,536 UTF-8 bytes, enforced after decoding. It
  does not cap transport message size or peak memory usage.
- The original `call_id` is preserved in `ToolResult`.
- Raw server tool-error payloads are not copied into failed result output.
- Launch paths are trusted local developer configuration, not model-provided commands.
- Server stdout is reserved for the stdio protocol; do not add debug `print()` calls.
- The stdio smoke has a 30-second outer budget and 10-second request timeouts.
  The executor defaults to a 10-second call timeout. These are not guarantees of
  forced termination of arbitrary Python code or production process recovery.
- With a configured scenario budget, timed execution receives the remaining
  duration; scheduling and cleanup overhead are not a hard wall-time guarantee.

## Troubleshooting

| Symptom | Check |
|---|---|
| `No module named fastmcp` | Use the `.venv-mcp` Python for the server and install its requirements. |
| Project module not found | Check `PYTHONPATH` for the in-process smoke and `--project-root` for stdio/suite demos. |
| `single page catalog` error | Confirm the check is `next_cursor is not None` and whether the server returned a continuation cursor. |
| Schema rejected | Compare the discovered input schema with the supported external schema profile. |
| `ExceptionGroup` | Inspect nested exceptions for the underlying failure. |
| Stdio tests skipped | Check the server Python environment variable and the backend MCP extra. |
| FastAPI/Starlette errors in backend tests | Run backend `pip check`; keep FastMCP server dependencies in the separate environment. |
| Suite prints JSON but no artifact directory appears | Expected for `demo_mcp_suite`; use `tool-run` for artifact output. |
| Runtime preparation failed before artifact creation | Check the server Python setting, dependencies, project root, and server startup; no run directory is expected. |

## Remaining work

1. Verify cleanup of an unresponsive child process separately from the
   server-exit detection test. Do not claim hard process-termination guarantees.
2. Validate real-model tool behavior separately from the synthetic calculator
   smoke before interpreting results as model-quality measurements.
3. Treat additional server profiles and API/worker integration as separate work.

Deadline-boundary tests now cover both early request timeouts that propagate and
timeouts after scenario expiry that produce `WALL_TIME_LIMIT`, without appending
a fabricated tool result.

The current integration supports MCP-backed suite execution and CLI artifacts;
API and worker runtime integration are not included.
