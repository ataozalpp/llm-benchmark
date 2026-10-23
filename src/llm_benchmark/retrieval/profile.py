"""Versioned description of the keyword retrieval baseline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KeywordRetrieverProfile:
    name: str = field(default="keyword-overlap", init=False)
    version: str = field(default="1", init=False)
    tokenizer: str = field(
        default="python-unicode-word-casefold-v1",
        init=False,
    )
    scoring: str = field(
        default="unique-token-intersection-count",
        init=False,
    )
    tie_break: str = field(default="chunk-id-ascending", init=False)
    zero_score_policy: str = field(default="omit", init=False)


KEYWORD_PROFILE = KeywordRetrieverProfile()


@dataclass(frozen=True)
class DenseRetrievalProfile:
    model_id: str
    model_revision: str
    embedding_dimension: int
    max_sequence_length: int
    query_prefix: str
    document_prefix: str
    device: str
    precision: str
    batch_size: int
    adapter_id: str
    adapter_version: str

    name: str = field(default="dense-exact-cosine", init=False)
    version: str = field(default="1", init=False)
    normalization: str = field(default="l2-scaled-v1", init=False)
    similarity: str = field(default="cosine", init=False)
    tie_break: str = field(default="chunk-id-ascending", init=False)
    score_filter: str = field(default="none", init=False)
    truncation_policy: str = "reject"

    def __post_init__(self) -> None:
        required_text_fields = (
            "model_id",
            "model_revision",
            "device",
            "precision",
            "adapter_id",
            "adapter_version",
        )

        for field_name in required_text_fields:
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a nonblank string.")

            if "\x00" in value:
                raise ValueError(f"{field_name} must not contain null characters.")

            value.encode("utf-8")

        for field_name in ("query_prefix", "document_prefix"):
            value = getattr(self, field_name)

            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string.")

            if "\x00" in value:
                raise ValueError(f"{field_name} must not contain null characters.")

            value.encode("utf-8")

        for field_name in (
            "embedding_dimension",
            "max_sequence_length",
            "batch_size",
        ):
            value = getattr(self, field_name)

            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

        if self.truncation_policy not in {"reject", "truncate"}:
            raise ValueError("Unsupported truncation policy.")

        if self.model_revision.casefold() in {"main", "master", "latest"}:
            raise ValueError(
                "Use a resolved model revision rather than a moving alias."
            )
