"""
Tests for rubra.patch() — OpenAI SDK interceptor.
Uses a mock client — no actual API calls.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch as mock_patch

import pytest

import rubra
from rubra.core.tracer.models import SpanType, TraceStatus
import rubra.core.tracer.decorators as dec


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50
    total_tokens = 150


class FakeMessage:
    content = "Paris is the capital of France."


class FakeChoice:
    message = FakeMessage()
    finish_reason = "stop"


class FakeResponse:
    usage = FakeUsage()
    choices = [FakeChoice()]


class FakeCompletions:
    def create(self, **kwargs):
        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_patch_returns_same_client():
    client = FakeClient()
    result = rubra.patch(client)
    assert result is client


def test_patch_invalid_client():
    with pytest.raises(TypeError, match="rubra.patch\\(\\) expects"):
        rubra.patch("not a client")


def test_patch_captures_llm_span(monkeypatch):
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    client = FakeClient()
    rubra.patch(client)

    @rubra.tool
    def search(q: str) -> str:
        return "Paris info"

    @rubra.agent(task="Capital city question")
    def my_agent(question: str) -> str:
        search(question)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content

    result = my_agent("What is the capital of France?")
    assert result == "Paris is the capital of France."

    trace = captured[0]
    assert trace.status == TraceStatus.COMPLETED

    llm_spans = trace.llm_call_spans
    assert len(llm_spans) == 1
    assert llm_spans[0].span_type == SpanType.LLM_CALL
    assert llm_spans[0].llm_data.model == "gpt-4o-mini"
    assert llm_spans[0].llm_data.prompt_tokens == 100
    assert llm_spans[0].llm_data.completion_tokens == 50
    assert llm_spans[0].llm_data.total_tokens == 150
    # finish() runs before _store_trace, so tokens are already aggregated
    assert trace.token_usage.total_tokens == 150


def test_patch_token_aggregation_on_finish(monkeypatch):
    """Token usage from LLM spans is aggregated when trace.finish() is called."""
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    client = FakeClient()
    rubra.patch(client)

    @rubra.agent(task="test")
    def ag() -> str:
        client.chat.completions.create(model="gpt-4o", messages=[])
        client.chat.completions.create(model="gpt-4o", messages=[])
        return "done"

    ag()
    trace = captured[0]
    # Two LLM calls, 150 tokens each = 300
    assert trace.token_usage.total_tokens == 300
