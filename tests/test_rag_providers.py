import pytest

from llm_benchmark.rag.providers import RagGenerationResponse


def test_generation_response_preserves_raw_text():
    response = RagGenerationResponse(
        text="not valid answer JSON",
        finish_reason="stop",
    )
    assert response.text == ("not valid answer JSON")
    assert response.input_tokens is None
    assert response.output_tokens is None


def test_generation_response_accepts_measured_usage():
    response = RagGenerationResponse(
        text="",
        finish_reason="length",
        input_tokens=120,
        output_tokens=0,
    )

    assert response.input_tokens == 120
    assert response.output_tokens == 0


@pytest.mark.parametrize(
    "values",
    [
        {"text": None, "finish_reason": "stop"},
        {"text": "", "finish_reason": ""},
        {"text": "", "finish_reason": None},
        {"text": "", "finish_reason": "stop", "input_tokens": -1},
        {"text": "", "finish_reason": "stop", "input_tokens": True},
        {"text": "", "finish_reason": "stop", "output_tokens": 1.5},
    ],
)
def test_invalid_response_metadata_is_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        RagGenerationResponse(**values)
