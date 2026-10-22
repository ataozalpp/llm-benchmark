"""Prepare a local BEIR SciFact dataset for retrieval-run."""

import argparse
import csv
import json
import sys
from pathlib import Path

from llm_benchmark.retrieval.beir_import import prepare_scifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    args = parser.parse_args()

    try:
        provenance = prepare_scifact(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            split=args.split,
        )
    except (OSError, ValueError, csv.Error):
        print(
            "error: Dataset preparation failed. Check local input files, "
            "the output directory and importer validation rules.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "corpus_count": provenance["corpus_count"],
                "query_count": provenance["query_count"],
                "suite_size_bytes": provenance["suite_size_bytes"],
                "output_suite_sha256": provenance["output_suite_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
