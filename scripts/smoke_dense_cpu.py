"""Check real CPU embeddings against the existing dense retriever."""

import argparse
import json
import re
import sys
from dataclasses import asdict
from importlib.metadata import version
from typing import Any

from llm_benchmark.retrieval.dense import DenseRetriever
from llm_benchmark.retrieval.embedding_loader import load_cpu_embedder
from llm_benchmark.retrieval.identity import fingerprint
from llm_benchmark.retrieval.local_embeddings import (
    EmbeddingInputTooLongError,
    SentenceTransformerCpuEmbedder,
)
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery
from llm_benchmark.retrieval.profile import DenseRetrievalProfile

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384
EXPECTED_MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 8


def max_abs_difference(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Vectors must have the same nonzero dimension.")

    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def check_long_input_rejection(
    embedder: SentenceTransformerCpuEmbedder,
) -> int:
    limit = embedder.profile.max_sequence_length
    long_text = " ".join(["cat"] * (limit * 2))

    tokenized = embedder.model.tokenizer(
        [long_text],
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    rows = tokenized["input_ids"]

    if len(rows) != 1:
        raise ValueError("Expected one tokenized long input.")

    token_count = len(rows[0])
    if token_count <= limit:
        raise RuntimeError("Long-input fixture did not exceed the token limit.")

    try:
        embedder.encode_query(long_text)
    except EmbeddingInputTooLongError:
        pass
    else:
        raise RuntimeError("Long input was not rejected.")

    return token_count


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from huggingface_hub import HfApi

    revision = args.revision
    if revision is None:
        revision = HfApi().model_info(MODEL_ID).sha

    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full model commit SHA as the revision.")

    profile = DenseRetrievalProfile(
        model_id=MODEL_ID,
        model_revision=revision,
        embedding_dimension=EXPECTED_DIMENSION,
        max_sequence_length=EXPECTED_MAX_SEQUENCE_LENGTH,
        query_prefix="",
        document_prefix="",
        device="cpu",
        precision="float32",
        batch_size=BATCH_SIZE,
        adapter_id="smoke-sentence-transformers-cpu",
        adapter_version="1",
        truncation_policy="reject",
    )

    embedder = load_cpu_embedder(
        profile,
        local_files_only=args.offline,
    )
    model = embedder.model

    observed_dimension = model.get_embedding_dimension()
    observed_limit = model.max_seq_length
    observed_device = str(model.device)

    parameter_dtypes = sorted(
        {
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )

    chunks = (
        Chunk(
            chunk_id="cat",
            document_id="doc-cat",
            text="A cat is sleeping on the sofa.",
        ),
        Chunk(
            chunk_id="space",
            document_id="doc-space",
            text="A spacecraft travels through outer space.",
        ),
        Chunk(
            chunk_id="bread",
            document_id="doc-bread",
            text="Fresh bread is baked in an oven.",
        ),
    )

    retriever = DenseRetriever(chunks, embedder=embedder)

    rankings = {}
    checks = (
        ("exact", "A cat is sleeping on the sofa."),
        ("paraphrase", "A kitten naps on the couch."),
    )
    for query_id, text in checks:
        result = retriever.retrieve(
            RetrievalQuery(query_id=query_id, text=text),
            top_k=3,
        )
        if len(result.chunk_ids) != 3 or result.chunk_ids[0] != "cat":
            raise RuntimeError(f"{query_id} check failed: {result.chunk_ids}")
        rankings[query_id] = list(result.chunk_ids)

    probe_text = chunks[0].text
    single = embedder.encode_query(probe_text)
    repeated = embedder.encode_query(probe_text)
    batched = embedder.encode_documents(tuple(chunk.text for chunk in chunks))[0]

    long_input_token_count = check_long_input_rejection(embedder)

    profile_data = asdict(profile)

    return {
        "status": "passed",
        "profile": profile_data,
        "retriever_profile_hash": fingerprint(profile_data),
        "observed": {
            "device": observed_device,
            "embedding_dimension": observed_dimension,
            "max_sequence_length": observed_limit,
            "parameter_dtypes": parameter_dtypes,
            "embedding_dtype": "float32",
            "tokenizer_class": qualified_class_name(model.tokenizer),
            "tokenizer_is_fast": bool(getattr(model.tokenizer, "is_fast", False)),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "rankings": rankings,
        "long_input_rejected": True,
        "long_input_token_count": long_input_token_count,
        "diagnostics": {
            "repeat_max_abs_difference": max_abs_difference(single, repeated),
            "batch_max_abs_difference": max_abs_difference(single, batched),
            "probe_vector_hash": fingerprint(single),
        },
        "runtime_versions": {
            name: version(name)
            for name in (
                "torch",
                "sentence-transformers",
                "transformers",
                "huggingface-hub",
                "tokenizers",
                "numpy",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.offline and args.revision is None:
        parser.error("--offline requires --revision.")

    try:
        report = run_smoke(args)
        serialized = json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )
    except Exception as exc:
        if args.debug:
            raise
        print(
            f"Dense CPU smoke failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        print(
            "Use --debug to display the traceback.",
            file=sys.stderr,
        )
        return 2

    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
