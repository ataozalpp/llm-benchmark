"""Strict decoding of recorded retrieval profiles."""

from dataclasses import asdict, fields

from .profile import (
    BM25_PROFILE,
    KEYWORD_PROFILE,
    BM25RetrievalProfile,
    DenseRetrievalProfile,
    HybridRetrievalProfile,
    KeywordRetrieverProfile,
)


def profile_from_dict(
    payload: object,
) -> (
    KeywordRetrieverProfile
    | DenseRetrievalProfile
    | BM25RetrievalProfile
    | HybridRetrievalProfile
):
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

    elif name == "hybrid-bm25-dense-rrf":
        dense_payload = payload.get("dense")
        lexical_payload = payload.get("lexical")

        if (
            not isinstance(dense_payload, dict)
            or dense_payload.get("name") != "dense-exact-cosine"
        ):
            raise ValueError("Hybrid dense profile is missing or invalid.")

        if (
            not isinstance(lexical_payload, dict)
            or lexical_payload.get("name") != BM25_PROFILE.name
        ):
            raise ValueError("Hybrid lexical profile is missing or invalid.")

        dense_profile = profile_from_dict(dense_payload)
        lexical_profile = profile_from_dict(lexical_payload)

        if not isinstance(dense_profile, DenseRetrievalProfile):
            raise ValueError("Hybrid requires a dense child profile.")

        if lexical_profile != BM25_PROFILE:
            raise ValueError("Unsupported hybrid lexical profile.")

        try:
            profile = HybridRetrievalProfile(
                dense=dense_profile,
                candidate_k=payload["candidate_k"],
                rank_constant=payload["rank_constant"],
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("Invalid recorded hybrid profile.") from None

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
