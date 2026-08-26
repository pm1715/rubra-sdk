"""
@rubra.agent and @rubra.tool decorators.
Supports sync and async functions transparently.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from rubra.core.tracer.context import (
    SpanContext,
    TraceContext,
    get_active_span,
    get_active_trace,
)
from rubra.core.tracer.models import (
    Span,
    SpanStatus,
    SpanType,
    Trace,
    TraceStatus,
    ToolCallData,
)

F = TypeVar("F", bound=Callable[..., Any])


def agent(
    func: F | None = None,
    *,
    name: str | None = None,
    task: str | None = None,
    task_description: str | None = None,
    tags: list[str] | None = None,
    expected_output: str | None = None,
    expected_tool_calls: list[str] | None = None,
    expected_tool_args: dict[str, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> F | Callable[[F], F]:
    """
    Decorator that wraps an agent function and captures a full Trace.

    Minimal usage:
        @rubra.agent
        def my_agent(q: str) -> str: ...

    With options:
        @rubra.agent(task="Summarise the doc", tags=["prod"])
        async def my_agent(doc: str) -> str: ...
    """

    def decorator(fn: F) -> F:
        agent_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace = Trace(
                    agent_name=agent_name,
                    task=task,
                    task_description=task_description,
                    tags=tags or [],
                    expected_output=expected_output,
                    expected_tool_calls=expected_tool_calls,
                    expected_tool_args=expected_tool_args,
                    metadata=metadata or {},
                )
                async with TraceContext(trace):
                    try:
                        result = await fn(*args, **kwargs)
                        trace.finish(output=str(result) if result is not None else None)
                    except Exception as exc:
                        trace.finish(status=TraceStatus.FAILED, error_message=str(exc))
                        raise
                    finally:
                        _store_trace(trace)
                return result

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace = Trace(
                    agent_name=agent_name,
                    task=task,
                    task_description=task_description,
                    tags=tags or [],
                    expected_output=expected_output,
                    expected_tool_calls=expected_tool_calls,
                    expected_tool_args=expected_tool_args,
                    metadata=metadata or {},
                )
                with TraceContext(trace):
                    try:
                        result = fn(*args, **kwargs)
                        trace.finish(output=str(result) if result is not None else None)
                    except Exception as exc:
                        trace.finish(status=TraceStatus.FAILED, error_message=str(exc))
                        raise
                    finally:
                        _store_trace(trace)
                return result

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> F | Callable[[F], F]:
    """
    Decorator that wraps a tool function and captures TOOL_CALL + TOOL_RESPONSE spans.

    Usage:
        @rubra.tool
        def search_web(query: str) -> str: ...
    """

    def decorator(fn: F) -> F:
        tool_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                call_span = _make_tool_call_span(tool_name, fn, args, kwargs, metadata)
                async with SpanContext(call_span):
                    _attach_span_to_active_trace(call_span)
                    try:
                        result = await fn(*args, **kwargs)
                        call_span.finish()
                        _record_tool_response(tool_name, result, error=None)
                    except Exception as exc:
                        call_span.finish(
                            status=SpanStatus.ERROR,
                            error_message=str(exc),
                            error_type=type(exc).__name__,
                        )
                        _record_tool_response(tool_name, result=None, error=str(exc))
                        raise
                return result

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                call_span = _make_tool_call_span(tool_name, fn, args, kwargs, metadata)
                with SpanContext(call_span):
                    _attach_span_to_active_trace(call_span)
                    try:
                        result = fn(*args, **kwargs)
                        call_span.finish()
                        _record_tool_response(tool_name, result, error=None)
                    except Exception as exc:
                        call_span.finish(
                            status=SpanStatus.ERROR,
                            error_message=str(exc),
                            error_type=type(exc).__name__,
                        )
                        _record_tool_response(tool_name, result=None, error=str(exc))
                        raise
                return result

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_tool_call_span(
    tool_name: str,
    fn: Callable,
    args: tuple,
    kwargs: dict,
    metadata: dict | None,
) -> Span:
    sig = inspect.signature(fn)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        bound_args: dict[str, Any] = dict(bound.arguments)
    except TypeError:
        bound_args = {"args": list(args), **kwargs}

    parent_span = get_active_span()
    trace = get_active_trace()

    return Span(
        trace_id=trace.trace_id if trace else "unknown",
        parent_span_id=parent_span.span_id if parent_span else None,
        span_type=SpanType.TOOL_CALL,
        name=f"tool:{tool_name}",
        tool_call_data=ToolCallData(
            tool_name=tool_name,
            arguments=_sanitize(bound_args),
        ),
        metadata=metadata or {},
    )


def _attach_span_to_active_trace(span: Span) -> None:
    trace = get_active_trace()
    if trace is not None:
        trace.add_span(span)


def _record_tool_response(tool_name: str, result: Any, error: str | None) -> None:
    from rubra.core.tracer.models import ToolResponseData

    trace = get_active_trace()
    if trace is None:
        return

    response_span = Span(
        trace_id=trace.trace_id,
        span_type=SpanType.TOOL_RESPONSE,
        name=f"tool_response:{tool_name}",
        tool_response_data=ToolResponseData(
            tool_name=tool_name,
            output=_sanitize_output(result),
            error=error,
        ),
    )
    response_span.finish()
    trace.add_span(response_span)


def _sanitize(obj: Any, max_depth: int = 4) -> Any:
    if max_depth == 0:
        return "<truncated>"
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, max_depth - 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, max_depth - 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _sanitize_output(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str) and len(obj) > 4096:
        return obj[:4096] + "... [truncated]"
    return _sanitize(obj)


_last_trace: "Trace | None" = None


def _store_trace(trace: "Trace") -> None:
    global _last_trace
    _last_trace = trace
    try:
        from rubra.core.storage.db import auto_init_storage
        storage = auto_init_storage()
        storage.save_trace(trace)
    except Exception:
        pass
