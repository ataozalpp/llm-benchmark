"""Strict decoding of recorded retrieval profiles."""

from dataclasses import asdict, fields

from .profile import (
    BM25_PROFILE,
    KEYWORD_PROFILE,
    BM25RetrievalProfile,
    DenseRetrievalProfile,
    KeywordRetrieverProfile,
)


def profile_from_dict(
    payload: object,
) -> KeywordRetrieverProfile | DenseRetrievalProfile | BM25RetrievalProfile:
    if not isinstance(payload, dict):
        raise ValueError("Retriever profile must be a JSON object.")

    name = payload.get("name")

    if name == KEYWORD_PROFILE.name:
        profile = KEYWORD_PROFILE

    elif name == BM25_PROFILE.name:
        profile = BM25_PROFILE

    elif name == "dense-exact-cosine":
        init_names = {
            field.name for field in fields(DenseRetrievalProfile) if field.init
        }

        if not init_names <= payload.keys():
            raise ValueError("Dense profile fields are missing.")

        try:
            profile = DenseRetrievalProfile(
                **{name: payload[name] for name in init_names}
            )
        except (TypeError, ValueError):
            raise ValueError("Invalid recorded dense profile.") from None

    else:
        raise ValueError("Unsupported retriever profile.")

    expected = asdict(profile)

    if payload.keys() != expected.keys():
        raise ValueError("Recorded profile fields do not match the schema.")

    for name, expected_value in expected.items():
        actual_value = payload[name]

        if (
            type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            raise ValueError(f"Unsupported recorded profile field: {name}.")

    return profile
