"""
Unit tests for rubra.integrations.langgraph — no actual LangGraph required.
We test rubra_node directly (it's just a decorator pattern).
"""
from __future__ import annotations

import pytest
import rubra
from rubra.integrations.langgraph.patch import rubra_node, _require_langgraph
from rubra.core.tracer.models import SpanType, TraceStatus
import rubra.core.tracer.decorators as dec


# ---------------------------------------------------------------------------
# rubra_node: no active trace → pure pass-through
# ---------------------------------------------------------------------------


def test_rubra_node_passthrough_no_trace():
    @rubra_node
    def my_node(state: dict) -> dict:
        return {"answer": state["q"] + "_processed"}

    result = my_node({"q": "hello"})
    assert result == {"answer": "hello_processed"}


def test_rubra_node_with_name_kwarg():
    @rubra_node(name="custom_node")
    def process(state: dict) -> dict:
        return {"done": True}

    result = process({"x": 1})
    assert result == {"done": True}


# ---------------------------------------------------------------------------
# rubra_node: inside active trace → spans captured
# ---------------------------------------------------------------------------


def test_rubra_node_captures_spans(monkeypatch):
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @rubra_node
    def search_node(state: dict) -> dict:
        return {"results": "Paris is the capital"}

    @rubra.agent(task="Find capital")
    def my_agent(q: str) -> str:
        out = search_node({"query": q})
        return out["results"]

    result = my_agent("capital of France")
    assert result == "Paris is the capital"
    assert len(captured) == 1

    trace = captured[0]
    assert trace.status == TraceStatus.COMPLETED

    tool_spans = trace.tool_call_spans
    assert len(tool_spans) == 1
    assert tool_spans[0].span_type == SpanType.TOOL_CALL
    assert tool_spans[0].tool_call_data.tool_name == "search_node"


def test_rubra_node_response_span_captured(monkeypatch):
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @rubra_node
    def fetch(state: dict) -> dict:
        return {"data": "42"}

    @rubra.agent(task="Fetch data")
    def ag(x: str) -> str:
        return fetch({"x": x})["data"]

    ag("test")
    trace = captured[0]
    response_spans = [s for s in trace.spans if s.span_type == SpanType.TOOL_RESPONSE]
    assert len(response_spans) == 1
    assert response_spans[0].tool_response_data.tool_name == "fetch"
    assert response_spans[0].tool_response_data.output is not None


# ---------------------------------------------------------------------------
# rubra_node: error handling
# ---------------------------------------------------------------------------


def test_rubra_node_error_captured(monkeypatch):
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @rubra_node
    def bad_node(state: dict) -> dict:
        raise ValueError("node exploded")

    @rubra.agent(task="Will fail")
    def ag() -> str:
        bad_node({})
        return "unreachable"

    with pytest.raises(ValueError, match="node exploded"):
        ag()

    trace = captured[0]
    assert trace.status == TraceStatus.FAILED
    error_spans = [s for s in trace.spans if s.status.value == "error"]
    assert len(error_spans) >= 1


# ---------------------------------------------------------------------------
# rubra_node: async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rubra_node_async(monkeypatch):
    captured = []
    monkeypatch.setattr(dec, "_store_trace", lambda t: captured.append(t))

    @rubra_node
    async def async_node(state: dict) -> dict:
        return {"async_result": "done"}

    @rubra.agent(task="Async node test")
    async def ag() -> str:
        out = await async_node({"x": 1})
        return out["async_result"]

    result = await ag()
    assert result == "done"
    trace = captured[0]
    assert len(trace.tool_call_spans) == 1


# ---------------------------------------------------------------------------
# _require_langgraph: ImportError when not installed
# ---------------------------------------------------------------------------


def test_require_langgraph_raises_when_missing(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "langgraph":
            raise ImportError("No module named 'langgraph'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="rubra\\[langgraph\\]"):
        _require_langgraph()
