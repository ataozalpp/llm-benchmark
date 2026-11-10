"""Run one prepared-context RAG smoke against a compatible endpoint."""

import argparse
import json
import sys
from dataclasses import asdict

from llm_benchmark.rag.case import run_rag_case
from llm_benchmark.rag.models import RagContext, RagScenario
from llm_benchmark.rag.openai_compatible import OpenAICompatibleRagProvider
from llm_benchmark.rag.prompts import DEFAULT_PROMPT_VERSION, PROMPTS
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-version", choices=sorted(PROMPTS), default=DEFAULT_PROMPT_VERSION
    )
    parser.add_argument(
        "--structured-output", choices=["none", "json_schema"], default="none"
    )
    args = parser.parse_args()

    try:
        provider = OpenAICompatibleRagProvider(
            base_url=args.base_url,
            model_id=args.model,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
            prompt_version=args.prompt_version,
            structured_output=args.structured_output,
        )
    except ValueError as exc:
        print(f"Invalid smoke configuration: {exc}", file=sys.stderr)
        return 2

    scenario = RagScenario(
        query=RetrievalQuery(
            query_id="refund-period",
            text="What is the refund period? Give the duration including its unit, with no additional explanation.",
        ),
        reference_answers=("30 days", "thirty days"),
        relevant_chunk_ids=frozenset({"refund-policy"}),
    )

    context = RagContext(
        query_id=scenario.query.query_id,
        chunks=(
            Chunk(
                chunk_id="refund-policy",
                document_id="policy",
                text="Customers may request a refund within 30 days.",
            ),
            Chunk(
                chunk_id="shipping-policy",
                document_id="shipping",
                text="Standard shipping takes 5 business days.",
            ),
        ),
    )

    result = run_rag_case(
        scenario=scenario,
        context=context,
        provider=provider,
    )

    evaluation = result.evaluation
    passed = (
        result.status == "completed"
        and evaluation is not None
        and evaluation.answer_exact_match is True
        and evaluation.citation_validity == 1.0
        and evaluation.citation_evidence_recall == 1.0
    )

    report = {
        "smoke_passed": passed,
        "model_id": args.model,
        "base_url": args.base_url,
        "prompt_version": args.prompt_version,
        "context": json.loads(context.render()),
        "scenario": {
            "query_id": scenario.query.query_id,
            "question": scenario.query.text,
            "reference_answers": list(scenario.reference_answers),
            "relevant_chunk_ids": sorted(scenario.relevant_chunk_ids),
        },
        "settings": {
            "temperature": 0.0,
            "timeout_seconds": args.timeout,
            "max_tokens": args.max_tokens,
            "structured_output": args.structured_output,
        },
        "case": asdict(result),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
