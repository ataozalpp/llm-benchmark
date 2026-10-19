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
