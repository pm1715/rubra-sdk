"""
LangGraph integration — records each graph node as a Rubra tool span.

Usage (per-node decorator):
    from rubra.integrations.langgraph import rubra_node

    @rubra_node
    def search_node(state: dict) -> dict:
        return {"results": search(state["query"])}

Usage (bulk-patch a StateGraph before compile):
    from rubra.integrations.langgraph import patch

    graph = StateGraph(MyState)
    graph.add_node("search", search_node)
    app = patch(graph).compile()

Both approaches require the node to run inside a @rubra.agent trace.
When no trace is active, functions pass through unchanged.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from rubra.core.tracer.context import get_active_trace
from rubra.core.tracer.models import (
    Span,
    SpanStatus,
    SpanType,
    ToolCallData,
    ToolResponseData,
)


def rubra_node(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """
    Decorator that captures a LangGraph node execution as TOOL_CALL + TOOL_RESPONSE spans.

    Works inside an active @rubra.agent trace.
    Silently passes through when no trace is active (safe to leave in production).
    """
    if func is None:
        return lambda f: rubra_node(f, name=name)

    node_name = name or func.__name__

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _run_async(func, node_name, args, kwargs)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_sync(func, node_name, args, kwargs)
        return sync_wrapper


def patch(graph: Any) -> Any:
    """
    Instrument all nodes in a LangGraph StateGraph.

    Must be called before .compile():
        app = patch(graph).compile()

    Wraps every node function so its execution is recorded as a tool span
    inside the active Rubra trace.
    """
    _require_langgraph()

    nodes_dict = getattr(graph, "_nodes", None)
    if nodes_dict is None:
        raise TypeError(
            "rubra.integrations.langgraph.patch() expects a StateGraph (before compile). "
            f"Got: {type(graph).__name__}. Call patch(graph) before graph.compile()."
        )

    for node_name in list(nodes_dict.keys()):
        entry = nodes_dict[node_name]
        raw_fn = getattr(entry, "func", None) or entry
        if callable(raw_fn) and not isinstance(raw_fn, type):
            wrapped = rubra_node(raw_fn, name=node_name)
            if hasattr(entry, "func"):
                entry.func = wrapped
            else:
                nodes_dict[node_name] = wrapped

    return graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_sync(func: Callable, node_name: str, args: tuple, kwargs: dict) -> Any:
    trace = get_active_trace()
    if trace is None:
        return func(*args, **kwargs)

    call_span = Span(
        trace_id=trace.trace_id,
        span_type=SpanType.TOOL_CALL,
        name=f"langgraph:{node_name}",
        tool_call_data=ToolCallData(
            tool_name=node_name,
            arguments=_extract_state(args),
        ),
    )
    trace.add_span(call_span)

    error: Exception | None = None
    result: Any = None
    try:
        result = func(*args, **kwargs)
        call_span.finish()
        return result
    except Exception as exc:
        error = exc
        call_span.finish(
            status=SpanStatus.ERROR,
            error_message=str(exc),
            error_type=type(exc).__name__,
        )
        raise
    finally:
        resp_span = Span(
            trace_id=trace.trace_id,
            parent_span_id=call_span.span_id,
            span_type=SpanType.TOOL_RESPONSE,
            name=f"langgraph:{node_name}:response",
            tool_response_data=ToolResponseData(
                tool_name=node_name,
                output=str(result)[:2000] if result is not None else None,
                error=str(error) if error else None,
                was_used_in_next_step=True,
            ),
        )
        resp_span.finish(status=SpanStatus.ERROR if error else SpanStatus.OK)
        trace.add_span(resp_span)


async def _run_async(func: Callable, node_name: str, args: tuple, kwargs: dict) -> Any:
    trace = get_active_trace()
    if trace is None:
        return await func(*args, **kwargs)

    call_span = Span(
        trace_id=trace.trace_id,
        span_type=SpanType.TOOL_CALL,
        name=f"langgraph:{node_name}",
        tool_call_data=ToolCallData(
            tool_name=node_name,
            arguments=_extract_state(args),
        ),
    )
    trace.add_span(call_span)

    error: Exception | None = None
    result: Any = None
    try:
        result = await func(*args, **kwargs)
        call_span.finish()
        return result
    except Exception as exc:
        error = exc
        call_span.finish(
            status=SpanStatus.ERROR,
            error_message=str(exc),
            error_type=type(exc).__name__,
        )
        raise
    finally:
        resp_span = Span(
            trace_id=trace.trace_id,
            parent_span_id=call_span.span_id,
            span_type=SpanType.TOOL_RESPONSE,
            name=f"langgraph:{node_name}:response",
            tool_response_data=ToolResponseData(
                tool_name=node_name,
                output=str(result)[:2000] if result is not None else None,
                error=str(error) if error else None,
                was_used_in_next_step=True,
            ),
        )
        resp_span.finish(status=SpanStatus.ERROR if error else SpanStatus.OK)
        trace.add_span(resp_span)


def _extract_state(args: tuple) -> dict[str, Any]:
    """Pull the state dict from node args (first positional arg is always state)."""
    if not args:
        return {}
    raw = args[0]
    if isinstance(raw, dict):
        return {k: str(v)[:200] for k, v in list(raw.items())[:10]}
    return {"state": str(raw)[:200]}


def _require_langgraph() -> None:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        raise ImportError(
            "rubra[langgraph] requires langgraph. "
            "Install with: pip install 'rubra[langgraph]'"
        )
