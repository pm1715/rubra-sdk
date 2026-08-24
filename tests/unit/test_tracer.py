"""
Unit tests for the core tracer — models, context, decorators.
No LLM calls, no SQLAlchemy, no network. Pure Python.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from rubra.core.tracer.models import (
    Span, SpanStatus, SpanType, Trace, TraceStatus,
    ToolCallData, ToolResponseData, TokenUsage,
)
from rubra.core.tracer.context import (
    get_active_trace, get_active_span, TraceContext, SpanContext,
)
from rubra.core.tracer.decorators import agent, tool


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_span_computes_duration_on_finish():
    trace = Trace(agent_name="test")
    span = Span(trace_id=trace.trace_id, span_type=SpanType.TOOL_CALL, name="tool:search")
    assert span.duration_ms is None
    span.finish()
    assert span.duration_ms is not None
    assert span.duration_ms >= 0


def test_span_with_preset_times_computes_duration():
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(milliseconds=250)
    span = Span(
        trace_id="abc",
        span_type=SpanType.LLM_CALL,
        name="llm",
        started_at=t0,
        ended_at=t1,
    )
    assert span.duration_ms == pytest.approx(250.0, abs=1.0)


def test_trace_finish_aggregates_token_usage():
    from rubra.core.tracer.models import LLMCallData
    trace = Trace(agent_name="test")
    t0 = datetime.now(timezone.utc)
    span = Span(
        trace_id=trace.trace_id,
        span_type=SpanType.LLM_CALL,
        name="llm",
        started_at=t0,
        ended_at=t0 + timedelta(milliseconds=100),
        llm_data=LLMCallData(
            model="gpt-4o",
            prompt_tokens=512,
            completion_tokens=128,
            total_tokens=640,
            cost_usd=0.005,
        ),
    )
    trace.spans = [span]
    trace.finish(output="result")
    assert trace.token_usage.total_tokens == 640
    assert trace.token_usage.estimated_cost_usd == pytest.approx(0.005)
    assert trace.status == TraceStatus.COMPLETED


def test_trace_properties():
    trace = Trace(agent_name="test")
    t0 = datetime.now(timezone.utc)

    call = Span(
        trace_id=trace.trace_id, span_type=SpanType.TOOL_CALL, name="tool:search",
        started_at=t0, ended_at=t0 + timedelta(milliseconds=10),
        tool_call_data=ToolCallData(tool_name="search", arguments={"q": "test"}),
    )
    call.finish()
    resp = Span(
        trace_id=trace.trace_id, span_type=SpanType.TOOL_RESPONSE, name="tool_response:search",
        tool_response_data=ToolResponseData(tool_name="search", output="result"),
    )
    resp.finish()
    trace.spans = [call, resp]

    assert trace.total_tool_calls == 1
    assert trace.unique_tools_used == ["search"]
    assert trace.total_llm_calls == 0


# ---------------------------------------------------------------------------
# Context tests
# ---------------------------------------------------------------------------


def test_trace_context_sets_and_restores():
    assert get_active_trace() is None
    trace = Trace(agent_name="ctx_test")
    with TraceContext(trace):
        assert get_active_trace() is trace
    assert get_active_trace() is None


def test_nested_span_context():
    trace = Trace(agent_name="nested")
    s1 = Span(trace_id=trace.trace_id, span_type=SpanType.AGENT_STEP, name="step1")
    s2 = Span(trace_id=trace.trace_id, span_type=SpanType.TOOL_CALL, name="tool:x")

    with SpanContext(s1):
        assert get_active_span() is s1
        with SpanContext(s2):
            assert get_active_span() is s2
        assert get_active_span() is s1
    assert get_active_span() is None


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


def test_agent_decorator_captures_trace(monkeypatch):
    captured = []

    import rubra.core.tracer.decorators as dec
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @agent(task="Unit test task")
    def my_agent(x: str) -> str:
        return f"done:{x}"

    result = my_agent("hello")
    assert result == "done:hello"
    assert len(captured) == 1
    t = captured[0]
    assert t.agent_name == "my_agent"
    assert t.task == "Unit test task"
    assert t.status == TraceStatus.COMPLETED
    assert t.final_output == "done:hello"


def test_tool_decorator_captures_spans(monkeypatch):
    captured = []

    import rubra.core.tracer.decorators as dec
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @tool
    def search(query: str) -> str:
        return f"result:{query}"

    @agent
    def ag(q: str) -> str:
        return search(q)

    ag("test")
    t = captured[0]
    assert t.total_tool_calls == 1
    assert t.unique_tools_used == ["search"]
    # TOOL_CALL + TOOL_RESPONSE = 2 spans
    assert len(t.spans) == 2
    assert t.spans[0].tool_call_data.arguments == {"query": "test"}
    assert t.spans[1].tool_response_data.output == "result:test"


def test_agent_captures_error(monkeypatch):
    captured = []

    import rubra.core.tracer.decorators as dec
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @agent
    def failing_agent() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_agent()

    t = captured[0]
    assert t.status == TraceStatus.FAILED
    assert "boom" in t.error_message


@pytest.mark.asyncio
async def test_async_agent_and_tool(monkeypatch):
    captured = []

    import rubra.core.tracer.decorators as dec
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @tool
    async def async_search(q: str) -> str:
        await asyncio.sleep(0)
        return f"async_result:{q}"

    @agent(task="async test")
    async def async_agent(q: str) -> str:
        return await async_search(q)

    result = await async_agent("foo")
    assert result == "async_result:foo"
    t = captured[0]
    assert t.status == TraceStatus.COMPLETED
    assert t.total_tool_calls == 1
