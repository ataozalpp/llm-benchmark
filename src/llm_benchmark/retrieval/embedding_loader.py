"""Load and validate a CPU embedding model for retrieval."""

import re
from typing import Any

from .local_embeddings import SentenceTransformerCpuEmbedder
from .profile import DenseRetrievalProfile


def _load_model(
    profile: DenseRetrievalProfile,
    *,
    local_files_only: bool,
) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        profile.model_id,
        revision=profile.model_revision,
        device="cpu",
        local_files_only=local_files_only,
        trust_remote_code=False,
    )


def load_cpu_embedder(
    profile: DenseRetrievalProfile,
    *,
    local_files_only: bool = True,
) -> SentenceTransformerCpuEmbedder:
    if not isinstance(profile, DenseRetrievalProfile):
        raise TypeError("Expected a dense retrieval profile.")

    if type(local_files_only) is not bool:
        raise TypeError("local_files_only must be a boolean.")

    if not re.fullmatch(r"[0-9a-f]{40}", profile.model_revision):
        raise ValueError("A full model commit SHA is required.")

    if profile.device != "cpu" or profile.precision != "float32":
        raise ValueError("The loader supports CPU float32 only.")

    if profile.query_prefix or profile.document_prefix:
        raise ValueError("The loader currently requires empty prefixes.")

    model = _load_model(
        profile,
        local_files_only=local_files_only,
    )
    model.float()
    model.eval()

    if str(model.device) != "cpu":
        raise ValueError("The loaded model is not on CPU.")

    parameter_dtypes = {
        str(parameter.dtype)
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes != {"torch.float32"}:
        raise ValueError(
            "The loaded model must have float32 floating-point parameters."
        )

    return SentenceTransformerCpuEmbedder(model=model, profile=profile)
