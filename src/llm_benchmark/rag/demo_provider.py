"""Synthetic software fixture, never a measurement of model quality."""

import json

from .models import RagGenerationRequest
from .providers import RagGenerationResponse


class SyntheticRagProvider:
    """Answers with the first context chunk; reads requests, never evaluation gold."""

    def generate(self, request: RagGenerationRequest) -> RagGenerationResponse:
        if not request.context.chunks:
            payload = {"answer": "", "citations": [], "abstained": True}
        else:
            first = request.context.chunks[0]
            payload = {
                "answer": first.text,
                "citations": [first.chunk_id],
                "abstained": False,
            }

        return RagGenerationResponse(
            text=json.dumps(payload, ensure_ascii=False),
            finish_reason="stop",
        )