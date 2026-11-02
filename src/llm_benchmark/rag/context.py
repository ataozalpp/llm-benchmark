"""Build bounded context from ranked retrieval results."""

from ..retrieval.models import Chunk, RankedRetrieval
from .models import RagContext


def build_context(
    *,
    chunks: tuple[Chunk, ...],
    ranking: RankedRetrieval,
    max_chars: int,
) -> RagContext:
    if not isinstance(chunks, tuple):
        raise TypeError("Chunks must be a tuple.")

    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("Every corpus item must be a Chunk.")

    if not isinstance(ranking, RankedRetrieval):
        raise TypeError("Expected RankedRetrieval.")

    # Even an empty JSON array requires two characters.
    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("max_cahrs must be an integer of at least two.")

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("Corpus chunk IDs must be unique.")

    if any(chunk_id not in by_id for chunk_id in ranking.chunk_ids):
        raise ValueError("Ranking references an unknown chunk.")

    selected: list[Chunk] = []

    for chunk_id in ranking.chunk_ids:
        candidate = RagContext(
            query_id=ranking.query_id,
            chunks=tuple(selected) + (by_id[chunk_id],),
        )

        if len(candidate.render()) <= max_chars:
            selected.append(by_id[chunk_id])

    return RagContext(
        query_id=ranking.query_id,
        chunks=tuple(selected),
    )
