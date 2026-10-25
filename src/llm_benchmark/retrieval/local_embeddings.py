"""CPU embedding adapter for an already loaded Sentence Transformer."""

import math
from typing import Any

from .profile import DenseRetrievalProfile


class SentenceTransformerCpuEmbedder:
    """Encode text using an injected CPU float32 model.

    This initial adapter supports empty prefixes and rejects inputs
    exceeding the configured token limit. Model loading and parameter
    dtype verification remain the caller's responsibility.
    """

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
            raise ValueError("Model sequence length does not match the profile.")

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
