import json
from http.client import HTTPException
from urllib.error import HTTPError, URLError

import pytest

from llm_benchmark.rag.case import run_rag_case
from llm_benchmark.rag.models import RagContext, RagGenerationRequest, RagScenario
from llm_benchmark.rag.openai_compatible import OpenAICompatibleRagProvider
from llm_benchmark.rag.providers import RagProviderError
from llm_benchmark.retrieval.models import Chunk, RetrievalQuery


class StubTransport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post_json(self, *, url, payload, timeout_seconds, headers=None):
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def make_request():
    return RagGenerationRequest(
        query=RetrievalQuery(query_id="q1", text="What is the policy?"),
        context=RagContext(query_id="q1", chunks=()),
    )


def make_response(text="not json", reason="stop"):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": reason,
            }
        ]
    }


def make_provider(transport):
    return OpenAICompatibleRagProvider(
        base_url="http://localhost:1234/v1",
        model_id="test-model",
        transport=transport,
    )


def test_request_and_raw_response():
    transport = StubTransport(response=make_response())
    result = make_provider(transport).generate(make_request())

    assert result.text == "not json"
    assert result.finish_reason == "stop"
    assert result.input_tokens is None

    call = transport.calls[0]
    assert call["url"] == "http://localhost:1234/v1/chat/completions"
    assert call["payload"]["model"] == "test-model"
    assert call["payload"]["stream"] is False
    assert call["payload"]["n"] == 1
    assert call["payload"]["max_tokens"] == 256
    assert [m["role"] for m in call["payload"]["messages"]] == ["system", "user"]


def test_usage_is_preserved():
    payload = make_response()
    payload["usage"] = {"prompt_tokens": 120, "completion_tokens": 25}
    transport = StubTransport(response=payload)

    result = make_provider(transport).generate(make_request())

    assert result.input_tokens == 120
    assert result.output_tokens == 25


def test_incomplete_response_can_have_no_content():
    transport = StubTransport(response=make_response(None, "length"))

    result = make_provider(transport).generate(make_request())

    assert result.text == ""
    assert result.finish_reason == "length"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": {}, "finish_reason": "stop"}]},
        make_response(None, "stop"),
        make_response("text", None),
    ],
)
def test_invalid_envelope_is_a_provider_error(payload):
    transport = StubTransport(response=payload)

    with pytest.raises(RagProviderError):
        make_provider(transport).generate(make_request())


def test_invalid_usage_is_rejected():
    payload = make_response()
    payload["usage"] = {"prompt_tokens": True}
    transport = StubTransport(response=payload)

    with pytest.raises(RagProviderError):
        make_provider(transport).generate(make_request())


def test_timeout_becomes_provider_error():
    transport = StubTransport(error=TimeoutError("timed out"))

    with pytest.raises(RagProviderError, match="request failed"):
        make_provider(transport).generate(make_request())

    assert len(transport.calls) == 1


def test_programming_error_is_not_hidden():
    transport = StubTransport(error=RuntimeError("Bug"))

    with pytest.raises(RuntimeError, match="Bug"):
        make_provider(transport).generate(make_request())


@pytest.mark.parametrize(
    "override",
    [
        {"base_url": "file:///tmp/model"},
        {"model_id": ""},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"max_tokens": True},
        {"max_tokens": 0},
    ],
)
def test_invalid_configuration_is_rejected(override):
    values = {
        "base_url": "http://localhost:1234/v1",
        "model_id": "test-model",
    }
    values.update(override)

    with pytest.raises(ValueError):
        OpenAICompatibleRagProvider(**values)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:bad/v1",
        "http://localhost:65536/v1",
        "http://localhost:0/v1",
        " http://localhost:1234/v1",
        "http://local\nhost:1234/v1",
        "http://localhost:1234/v1?x=1",
        "http://localhost:1234/v1#fragment",
        "http://user:secret@localhost/v1",
        "http://localhost:1234/v1/chat/completions/",
    ],
)
def test_invalid_base_url_is_rejected_before_io(base_url):
    transport = StubTransport()
    with pytest.raises(ValueError):
        OpenAICompatibleRagProvider(
            base_url=base_url, model_id="test-model", transport=transport
        )
    assert transport.calls == []


def test_custom_settings_and_context_are_sent_without_gold():
    transport = StubTransport(response=make_response())
    provider = OpenAICompatibleRagProvider(
        base_url="http://localhost:1234/v1/",
        model_id="custom",
        timeout_seconds=12,
        max_tokens=512,
        transport=transport,
    )
    request = RagGenerationRequest(
        query=RetrievalQuery(query_id="q1", text="Refund period?"),
        context=RagContext(
            query_id="q1",
            chunks=(Chunk(chunk_id="refund", document_id="policy", text="30 days."),),
        ),
    )
    provider.generate(request)
    call = transport.calls[0]
    assert call["url"] == "http://localhost:1234/v1/chat/completions"
    assert call["timeout_seconds"] == 12.0
    assert call["payload"]["max_tokens"] == 512
    assert call["payload"]["temperature"] == 0.0
    assert json.loads(call["payload"]["messages"][1]["content"]) == {
        "question": "Refund period?",
        "context": [{"chunk_id": "refund", "text": "30 days."}],
    }


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 0}])
def test_missing_usage_is_not_invented(usage):
    payload = make_response()
    payload["usage"] = usage
    result = make_provider(StubTransport(response=payload)).generate(make_request())
    assert result.input_tokens == (0 if usage else None)
    assert result.output_tokens is None


@pytest.mark.parametrize("field", ["tool_calls", "function_call"])
def test_completed_tool_invocation_is_rejected(field):
    payload = make_response()
    payload["choices"][0]["message"][field] = {"name": "unexpected"}
    with pytest.raises(RagProviderError, match="tool calls"):
        make_provider(StubTransport(response=payload)).generate(make_request())


@pytest.mark.parametrize(
    "error",
    [
        URLError("Unavailable"),
        HTTPError("http://localhost/v1", 503, "Unavailable", {}, None),
        HTTPException("Bad HTTP"),
        json.JSONDecodeError("Bad JSON", "x", 0),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
    ],
)
def test_transport_errors_are_wrapped_without_retry(error):
    transport = StubTransport(error=error)
    with pytest.raises(RagProviderError) as caught:
        make_provider(transport).generate(make_request())
    assert caught.value.__cause__ is error
    assert len(transport.calls) == 1


def test_invalid_request_fails_before_transport():
    transport = StubTransport()
    with pytest.raises(TypeError):
        make_provider(transport).generate(None)
    assert transport.calls == []


@pytest.mark.parametrize(
    ("payload", "error", "status"),
    [
        (
            make_response(
                '{"answer":"30 days","citations":["refund"],"abstained":false}'
            ),
            None,
            "completed",
        ),
        (make_response("not json"), None, "invalid_response"),
        (
            make_response(
                '{"answer":"30 days","citations":["refund"],"abstained":false}',
                "length",
            ),
            None,
            "incomplete_response",
        ),
        ({}, None, "provider_failed"),
        (None, TimeoutError("Timed out"), "provider_failed"),
    ],
)
def test_http_adapter_and_case_runner_integration(payload, error, status):
    scenario = RagScenario(
        query=RetrievalQuery(query_id="q1", text="Refund period?"),
        reference_answers=("30 days",),
        relevant_chunk_ids=frozenset({"refund"}),
    )
    context = RagContext(
        query_id="q1",
        chunks=(Chunk(chunk_id="refund", document_id="policy", text="30 days."),),
    )
    transport = StubTransport(response=payload, error=error)
    result = run_rag_case(
        scenario=scenario, context=context, provider=make_provider(transport)
    )
    assert result.status == status
    assert len(transport.calls) == 1
    if status == "completed":
        assert result.evaluation.answer_exact_match is True
        assert result.evaluation.citation_validity == 1.0
    else:
        assert result.answer is None
        assert result.evaluation is None
    assert (result.response is None) == (status == "provider_failed")
