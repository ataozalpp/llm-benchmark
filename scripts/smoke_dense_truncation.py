"""Check over-limit input handling with the cached CPU embedding model."""

import json
import sys
from dataclasses import replace
from pathlib import Path

from llm_benchmark.retrieval.embedding_loader import load_cpu_embedder
from llm_benchmark.retrieval.local_embeddings import (
    EmbeddingInputTooLongError,
    SentenceTransformerCpuEmbedder,
)
from llm_benchmark.retrieval.run_config import load_retrieval_run_config


def run_smoke() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    loaded = load_retrieval_run_config(
        root / "configs/retrieval_dense_smoke.yaml"
    )
    embedding = loaded.config.embedding
    if embedding is None:
        raise ValueError("The smoke config requires embedding settings.")

    profile = replace(
        embedding.to_profile(),
        truncation_policy="truncate",
    )
    embedder = load_cpu_embedder(profile, local_files_only=True)
    model = embedder.model

    long_text = " ".join(
        ["cat"] * (profile.max_sequence_length * 2)
    )
    raw_ids = model.tokenizer(
        long_text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        verbose=False,
    )["input_ids"]

    if len(raw_ids) <= profile.max_sequence_length:
        raise RuntimeError("The probe does not exceed the token limit.")

    features = model.tokenize([long_text])
    encoded_length = int(features["input_ids"].shape[1])

    if encoded_length != profile.max_sequence_length:
        raise RuntimeError(f"Unexpected effective token length: {encoded_length}")

    document_vectors = embedder.encode_documents(
        ("A cat is sleeping.", long_text)
    )
    query_vector = embedder.encode_query(long_text)

    if len(document_vectors) != 2:
        raise RuntimeError("Unexpected document vector count.")
    if len(query_vector) != profile.embedding_dimension:
        raise RuntimeError("Unexpected query vector dimension.")

    expected_statistics = {
        "schema_version": 1,
        "encoded_document_count": 2,
        "over_limit_document_count": 1,
        "encoded_query_count": 1,
        "over_limit_query_count": 1,
    }
    if embedder.input_statistics() != expected_statistics:
        raise RuntimeError("Unexpected input statistics.")

    rejecting = SentenceTransformerCpuEmbedder(
        model,
        replace(profile, truncation_policy="reject"),
    )
    try:
        rejecting.encode_query(long_text)
    except EmbeddingInputTooLongError:
        pass
    else:
        raise RuntimeError("Reject policy accepted an over-limit input.")

    return {
        "status": "passed",
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "raw_token_count": len(raw_ids),
        "effective_token_count": encoded_length,
        "max_sequence_length": profile.max_sequence_length,
        "reject_policy_verified": True,
        "embedding_input_statistics": embedder.input_statistics(),
    }


def main() -> int:
    try:
        report = run_smoke()
        serialized = json.dumps(report, indent=2, allow_nan=False)
    except Exception as exc:
        print(
            f"Dense truncation smoke failed"
            f"({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(serialized)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())