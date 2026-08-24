"""
Tests for rubra.patch_anthropic() — Anthropic SDK interceptor.
Uses mock client — no actual API calls.
"""
from __future__ import annotations

from types import SimpleNamespace
import pytest

import rubra
from rubra.core.tracer.models import SpanType, TraceStatus
import rubra.core.tracer.decorators as dec


class FakeUsage:
    input_tokens = 120
    output_tokens = 40


class FakeTextBlock:
    type = "text"
    text = "Paris is the capital of France."


class FakeAnthropicResponse:
    usage = FakeUsage()
    content = [FakeTextBlock()]
    stop_reason = "end_turn"
    model = "claude-3-5-sonnet-20241022"


class FakeMessages:
    def create(self, **kwargs):
        return FakeAnthropicResponse()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_patch_returns_same_client():
    client = FakeAnthropicClient()
    result = rubra.patch_anthropic(client)
    assert result is client


def test_patch_invalid_client():
    with pytest.raises(TypeError, match="patch_anthropic"):
        rubra.patch_anthropic("not a client")


def test_patch_captures_llm_span(monkeypatch):
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    client = FakeAnthropicClient()
    rubra.patch_anthropic(client)

    @rubra.tool
    def lookup(q: str) -> str:
        return "some context"

    @rubra.agent(task="Capital city question")
    def my_agent(question: str) -> str:
        lookup(question)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text

    result = my_agent("What is the capital of France?")
    assert result == "Paris is the capital of France."

    trace = captured[0]
    assert trace.status == TraceStatus.COMPLETED

    llm_spans = trace.llm_call_spans
    assert len(llm_spans) == 1
    assert llm_spans[0].span_type == SpanType.LLM_CALL
    assert llm_spans[0].llm_data.model == "claude-3-5-sonnet-20241022"
    assert llm_spans[0].llm_data.prompt_tokens == 120
    assert llm_spans[0].llm_data.completion_tokens == 40
    assert llm_spans[0].llm_data.total_tokens == 160


def test_patch_token_aggregation_on_finish(monkeypatch):
    """Multiple Anthropic calls → tokens aggregated at trace.finish()."""
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    client = FakeAnthropicClient()
    rubra.patch_anthropic(client)

    @rubra.agent(task="multi-call test")
    def ag() -> str:
        client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=50, messages=[])
        client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=50, messages=[])
        return "done"

    ag()
    trace = captured[0]
    # 2 calls × (120 input + 40 output) = 320 total
    assert trace.token_usage.total_tokens == 320


def test_patch_no_span_outside_trace():
    """When no @rubra.agent is active, patched client works without error."""
    client = FakeAnthropicClient()
    rubra.patch_anthropic(client)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response.content[0].text == "Paris is the capital of France."


def test_cost_estimation():
    from rubra.integrations.anthropic.patch import _estimate_cost
    cost = _estimate_cost("claude-3-5-sonnet-20241022", 1000, 500)
    assert cost > 0.0
    assert isinstance(cost, float)
