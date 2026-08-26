"""
ContextVar-based trace context propagation.

Async-safe: each asyncio Task gets its own copy automatically.
Thread-safe: each thread gets its own copy automatically.
No globals, no thread-locals, no locks needed.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rubra.core.tracer.models import Span, Trace

_active_trace: ContextVar[Trace | None] = ContextVar("rubra_active_trace", default=None)
_active_span: ContextVar[Span | None] = ContextVar("rubra_active_span", default=None)


def get_active_trace() -> Trace | None:
    return _active_trace.get()


def get_active_span() -> Span | None:
    return _active_span.get()


def set_active_trace(trace: Trace | None) -> Token:
    return _active_trace.set(trace)


def set_active_span(span: Span | None) -> Token:
    return _active_span.set(span)


def reset_active_trace(token: Token) -> None:
    _active_trace.reset(token)


def reset_active_span(token: Token) -> None:
    _active_span.reset(token)


class TraceContext:
    def __init__(self, trace: Trace) -> None:
        self._trace = trace
        self._token: Token | None = None

    def __enter__(self) -> Trace:
        self._token = set_active_trace(self._trace)
        return self._trace

    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            reset_active_trace(self._token)

    async def __aenter__(self) -> Trace:
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)


class SpanContext:
    def __init__(self, span: Span) -> None:
        self._span = span
        self._token: Token | None = None

    def __enter__(self) -> Span:
        self._token = set_active_span(self._span)
        return self._span

    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            reset_active_span(self._token)

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)
