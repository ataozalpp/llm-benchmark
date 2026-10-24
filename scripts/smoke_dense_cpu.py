"""Check real CPU embeddings against the existing dense retriever."""

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from importlib.metadata import version
from typing import Any

from llm_benchmark.retrieval.dense import DenseRetriever
from llm_benchmark.retrieval.identity import fingerprint
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery
from llm_benchmark.retrieval.profile import DenseRetrievalProfile

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384
EXPECTED_MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 8


class SmokeEmbedder:
    """Minimal bridge for this smoke check, not the production adapter."""

    def __init__(
        self,
        model: Any,
        profile: DenseRetrievalProfile,
    ) -> None:
        if profile.device != "cpu" or profile.precision != "float32":
            raise ValueError("The smoke supports CPU float32 only.")

        if profile.truncation_policy != "reject":
            raise ValueError("The smoke requires reject truncation policy.")

        if profile.query_prefix or profile.document_prefix:
            raise ValueError("The smoke requires empty prefixes.")

        if model.get_embedding_dimension() != profile.embedding_dimension:
            raise ValueError("Model dimension does not match the profile.")

        if model.max_seq_length != profile.max_sequence_length:
            raise ValueError("Model sequence limit does not match the profile.")

        self.model = model
        self.profile = profile

    def encode_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        if not isinstance(texts, tuple):
            raise TypeError("Texts must be a tuple.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Texts must be nonblank strings.")

        if not texts:
            return ()

        tokenized = self.model.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )

        if len(tokenized["input_ids"]) != len(texts):
            raise ValueError("Tokenizer output does not match inputs.")

        if any(
            len(ids) > self.profile.max_sequence_length
            for ids in tokenized["input_ids"]
        ):
            raise ValueError("Smoke inputs must fit the model token limit.")

        matrix = self.model.encode(
            list(texts),
            batch_size=self.profile.batch_size,
            device=self.profile.device,
            precision=self.profile.precision,
            prompt="",
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        expected_shape = (
            len(texts),
            self.profile.embedding_dimension,
        )

        if matrix.shape != expected_shape:
            raise ValueError(f"Unexpected embedding shape: {matrix.shape}")

        if str(matrix.dtype) != "float32":
            raise ValueError(f"Unexpected embedding dtype: {matrix.dtype}")

        vectors = tuple(tuple(float(value) for value in row) for row in matrix)

        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Embeddings contain non-finite values.")

        return vectors

    def encode_query(self, text: str) -> tuple[float, ...]:
        return self.encode_documents((text,))[0]


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
    embedder: SmokeEmbedder,
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
    except ValueError as exc:
        if str(exc) != "Smoke inputs must fit the model token limit.":
            raise
    else:
        raise RuntimeError("Long input was not rejected.")

    return token_count


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    revision = args.revision
    if revision is None:
        revision = HfApi().model_info(MODEL_ID).sha

    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full model commit SHA as the revision.")

    model = SentenceTransformer(
        MODEL_ID,
        revision=revision,
        device="cpu",
        local_files_only=args.offline,
        trust_remote_code=False,
    )
    model.float()
    model.eval()

    observed_dimension = model.get_embedding_dimension()
    observed_limit = model.max_seq_length
    observed_device = str(model.device)

    if observed_dimension != EXPECTED_DIMENSION:
        raise ValueError("Unexpected model embedding dimension.")

    if observed_limit != EXPECTED_MAX_SEQUENCE_LENGTH:
        raise ValueError("Unexpected model sequence limit.")

    if observed_device != "cpu":
        raise ValueError("The model is not running on CPU.")

    parameter_dtypes = sorted(
        {
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if parameter_dtypes != ["torch.float32"]:
        raise ValueError(f"Unexpected model parameter dtypes: {parameter_dtypes}")

    profile = DenseRetrievalProfile(
        model_id=MODEL_ID,
        model_revision=revision,
        embedding_dimension=observed_dimension,
        max_sequence_length=observed_limit,
        query_prefix="",
        document_prefix="",
        device=observed_device,
        precision="float32",
        batch_size=BATCH_SIZE,
        adapter_id="smoke-sentence-transformers-cpu",
        adapter_version="1",
        truncation_policy="reject",
    )

    embedder = SmokeEmbedder(model, profile)

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
