"""CPU embedding adapter for an already loaded Sentence Transformer."""

import math
from typing import Any

from .profile import DenseRetrievalProfile


class EmbeddingInputTooLongError(ValueError):
    """An input exceeds the configured embedding token limit."""


class SentenceTransformerCpuEmbedder:
    """Encode text using an injected CPU float32 model.

    Only empty prefixes are supported. Over-limit inputs are rejected
    or delegated to the model's configured sequence-length truncation.
    Model loading and parameter dtype verification remin the caller's
    responsibility.

    Input statistics count successfully validated encoding outputs.
    They count input occurences, not unique texts.
    """

    def __init__(
        self,
        model: Any,
        profile: DenseRetrievalProfile,
    ) -> None:
        if profile.device != "cpu" or profile.precision != "float32":
            raise ValueError("The CPU adapter supports CPU float32 only.")

        if profile.query_prefix or profile.document_prefix:
            raise ValueError("The CPU adapter requires empty prefixes.")

        if model.get_embedding_dimension() != profile.embedding_dimension:
            raise ValueError("Model dimension does not match the profile.")

        if model.max_seq_length != profile.max_sequence_length:
            raise ValueError("Model sequence length does not match the profile.")

        self.model = model
        self.profile = profile
        self._encoded_document_count = 0
        self._over_limit_document_count = 0
        self._encoded_query_count = 0
        self._over_limit_query_count = 0

    def _encode(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[tuple[float, ...], ...], int]:
        if not isinstance(texts, tuple):
            raise TypeError("Texts must be a tuple.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Texts must be nonblank strings.")

        if not texts:
            return (), 0

        tokenized = self.model.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )

        if len(tokenized["input_ids"]) != len(texts):
            raise ValueError("Tokenizer output does not match inputs.")

        over_limit_count = sum(
            len(ids) > self.profile.max_sequence_length
            for ids in tokenized["input_ids"]
        )

        if self.profile.truncation_policy == "reject" and over_limit_count:
            raise EmbeddingInputTooLongError("Embedding inputs must fit the model token limit.")

        matrix = self.model.encode(
            list(texts),
            batch_size=self.profile.batch_size,
            device=self.profile.device,
            precision=self.profile.precision,
            prompt="",
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=len(texts) > 1,
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

        return vectors, over_limit_count

    def encode_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        vectors, over_limit_count = self._encode(texts)

        self._encoded_document_count += len(vectors)
        self._over_limit_document_count += over_limit_count

        return vectors

    def encode_query(self, text: str) -> tuple[float, ...]:
        vectors, over_limit_count = self._encode((text,))

        self._encoded_query_count += 1
        self._over_limit_query_count += over_limit_count

        return vectors[0]

    def input_statistics(self) -> dict[str, int]:
        return {
            "schema_version": 1,
            "encoded_document_count": self._encoded_document_count,
            "over_limit_document_count": self._over_limit_document_count,
            "encoded_query_count": self._encoded_query_count,
            "over_limit_query_count": self._over_limit_query_count,
        }
