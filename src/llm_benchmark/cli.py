from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-benchmark", description="Run reproducible LLM benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a benchmark")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    tool_run = subparsers.add_parser("tool-run", help="run the versioned local tool suite")
    tool_run.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "tool-run":
            from .tool_run_config import load_tool_run_config
            from .tool_run_service import run_tool_benchmark

            run_dir, summary = run_tool_benchmark(load_tool_run_config(args.config))
        else:
            config = load_config(args.config, args.set)
            run_dir, summary = run_benchmark(config)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps({"run_id": summary["run_id"], "output_dir": str(run_dir), "overall": summary.get("overall", summary)}, indent=2))
