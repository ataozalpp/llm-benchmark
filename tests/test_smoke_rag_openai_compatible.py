import json
import runpy
import sys
from pathlib import Path

import pytest

from llm_benchmark.rag.providers import RagGenerationResponse, RagProviderError

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "smoke_rag_openai_compatible.py"
)


@pytest.fixture
def main():
    return runpy.run_path(str(SCRIPT_PATH))["main"]


@pytest.mark.parametrize(
    ("answer", "citations", "reason", "status", "passed"),
    [
        ("30 days", ["refund-policy"], "stop", "completed", True),
        ("30", ["refund-policy"], "stop", "completed", False),
        ("30 days", [], "stop", "completed", False),
        ("30 days", ["invented"], "stop", "completed", False),
        ("30 days", ["refund-policy"], "length", "incomplete_response", False),
    ],
)
def test_smoke_exit_and_report(
    main, monkeypatch, capsys, answer, citations, reason, status, passed
):
    calls = []

    class StubProvider:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def generate(self, request):
            assert not hasattr(request, "reference_answers")
            return RagGenerationResponse(
                text=json.dumps(
                    {"answer": answer, "citations": citations, "abstained": False}
                ),
                finish_reason=reason,
            )

    monkeypatch.setitem(main.__globals__, "OpenAICompatibleRagProvider", StubProvider)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT_PATH), "--model", "test-model", "--max-tokens", "512"]
    )
    assert main() == (0 if passed else 1)
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["smoke_passed"] is passed
    assert report["case"]["status"] == status
    assert report["scenario"]["reference_answers"] == ["30 days", "thirty days"]
    assert report["context"][0]["chunk_id"] == "refund-policy"
    assert report["settings"]["max_tokens"] == calls[0]["max_tokens"] == 512


@pytest.mark.parametrize("failure", ["provider", "parse"])
def test_failure_is_reported_as_json(main, monkeypatch, capsys, failure):
    class StubProvider:
        def __init__(self, **kwargs):
            pass

        def generate(self, request):
            if failure == "provider":
                raise RagProviderError("Unavailable")
            return RagGenerationResponse(text="not json", finish_reason="stop")

    monkeypatch.setitem(main.__globals__, "OpenAICompatibleRagProvider", StubProvider)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--model", "test-model"])
    assert main() == 1
    report = json.loads(capsys.readouterr().out)
    expected = "provider_failed" if failure == "provider" else "invalid_response"
    assert report["case"]["status"] == expected
    assert report["smoke_passed"] is False


def test_invalid_config_exits_before_generation(main, monkeypatch, capsys):
    def unexpected_call(**kwargs):
        pytest.fail("Generation must not run for invalid configuration")

    monkeypatch.setitem(main.__globals__, "run_rag_case", unexpected_call)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT_PATH), "--model", "test-model", "--max-tokens", "0"]
    )
    assert main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid smoke configuration" in captured.err
