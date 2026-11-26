"""Version constants for tool-use run identity; dependency-free by design."""

from __future__ import annotations

# Bump when evaluation or taxonomy semantics change. Previous: "tool-exact-v1".
TOOL_EVALUATOR_VERSION = "tool-trajectory-v1"

# Bump when manifest/summary/results shapes change. Previous: 1.
TOOL_ARTIFACT_SCHEMA_VERSION = 2

# `schema_version` written on each results.jsonl row. Previous: 1.
TOOL_RESULT_ROW_SCHEMA_VERSION = 2
