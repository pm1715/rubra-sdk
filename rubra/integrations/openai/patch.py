"""
rubra.patch(client) — OpenAI SDK interceptor.

Wraps openai.OpenAI / openai.AsyncOpenAI so every chat.completions.create()
call automatically captures a LLM_CALL span — zero changes to user code.

Usage:
    import openai
    import rubra

    client = rubra.patch(openai.OpenAI())   # one line

    # From here, all calls are traced:
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
"""

from __future__ import annotations

import time
from datetime import UTC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rubra.core.tracer.models import Span


def patch(client: Any) -> Any:
    """
    Instrument an OpenAI client instance to auto-capture LLM spans.

    Supports:
        - openai.OpenAI (sync)
        - openai.AsyncOpenAI (async)
        - Any object with a .chat.completions.create attribute

    Returns the same client instance (mutated in-place for compatibility).
    """
    try:
        completions = client.chat.completions
    except AttributeError:
        raise TypeError(
            "rubra.patch() expects an OpenAI client with a "
            f".chat.completions attribute. Got: {type(client).__name__}"
        ) from None

    import inspect

    if inspect.iscoroutinefunction(getattr(completions, "create", None)):
        _patch_async(completions)
    else:
        _patch_sync(completions)

    return client


def _patch_sync(completions: Any) -> None:
    original_create = completions.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        from rubra.core.tracer.context import get_active_trace

        trace = get_active_trace()
        t0 = time.perf_counter()

        try:
            response = original_create(*args, **kwargs)
        except Exception:
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if trace is not None:
            span = _build_llm_span(trace.trace_id, kwargs, response, elapsed_ms)
            trace.add_span(span)

        return response

    completions.create = patched_create


def _patch_async(completions: Any) -> None:
    original_create = completions.create

    async def patched_create(*args: Any, **kwargs: Any) -> Any:
        from rubra.core.tracer.context import get_active_trace

        trace = get_active_trace()
        t0 = time.perf_counter()

        try:
            response = await original_create(*args, **kwargs)
        except Exception:
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if trace is not None:
            span = _build_llm_span(trace.trace_id, kwargs, response, elapsed_ms)
            trace.add_span(span)

        return response

    completions.create = patched_create


def _build_llm_span(
    trace_id: str,
    kwargs: dict[str, Any],
    response: Any,
    elapsed_ms: float,
) -> Span:
    from datetime import datetime, timedelta

    from rubra.core.tracer.models import LLMCallData, Span, SpanStatus, SpanType

    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])

    # Extract token usage and cost from response
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (
        prompt_tokens + completion_tokens
    )

    # Rough cost estimate (GPT-4o-mini pricing as fallback)
    cost_usd = _estimate_cost(model, prompt_tokens, completion_tokens)

    # Extract response text
    choices = getattr(response, "choices", [])
    response_text = None
    finish_reason = None
    if choices:
        msg = getattr(choices[0], "message", None)
        response_text = getattr(msg, "content", None) if msg else None
        finish_reason = getattr(choices[0], "finish_reason", None)

    now = datetime.now(UTC)
    span = Span(
        trace_id=trace_id,
        span_type=SpanType.LLM_CALL,
        name=f"llm:{model}",
        started_at=now - timedelta(milliseconds=elapsed_ms),
        ended_at=now,
        duration_ms=elapsed_ms,
        status=SpanStatus.OK,
        llm_data=LLMCallData(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            messages=_sanitize_messages(messages),
            response=response_text,
            finish_reason=finish_reason,
        ),
    )
    return span


# ---------------------------------------------------------------------------
# Cost table (rough estimates, updated periodically)
# ---------------------------------------------------------------------------

_COST_PER_1K = {
    # model_prefix: (input $/1K, output $/1K)
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.010),
    "gpt-4-turbo": (0.010, 0.030),
    "gpt-4": (0.030, 0.060),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-haiku": (0.00025, 0.00125),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    for prefix, (in_rate, out_rate) in _COST_PER_1K.items():
        if model.startswith(prefix):
            return (prompt_tokens / 1000 * in_rate) + (
                completion_tokens / 1000 * out_rate
            )
    # Unknown model: use gpt-4o-mini rates as conservative fallback
    return (prompt_tokens / 1000 * 0.00015) + (completion_tokens / 1000 * 0.0006)


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Keep messages but truncate long content to avoid bloating the DB."""
    sanitized = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 500:
            content = content[:500] + "...[truncated]"
        sanitized.append({"role": msg.get("role", "user"), "content": content})
    return sanitized
