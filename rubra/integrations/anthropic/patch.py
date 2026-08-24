"""
Anthropic SDK interceptor — mirrors rubra.patch() for OpenAI.

Usage:
    import anthropic
    import rubra

    client = rubra.patch_anthropic(anthropic.Anthropic())

    @rubra.agent(task="...")
    def my_agent(question: str) -> str:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text
"""
from __future__ import annotations

import inspect
import time
from typing import Any

from rubra.core.tracer.context import get_active_trace
from rubra.core.tracer.models import LLMCallData, Span, SpanType

# Cost per 1K tokens: (input $/1K, output $/1K)
_ANTHROPIC_COST: dict[str, tuple[float, float]] = {
    "claude-opus-4":                  (0.015,  0.075),
    "claude-sonnet-4-5":              (0.003,  0.015),
    "claude-sonnet-4-5-20251001":     (0.003,  0.015),
    "claude-3-5-sonnet-20241022":     (0.003,  0.015),
    "claude-3-5-sonnet-20240620":     (0.003,  0.015),
    "claude-3-5-haiku-20241022":      (0.0008, 0.004),
    "claude-3-haiku-20240307":        (0.00025,0.00125),
    "claude-3-opus-20240229":         (0.015,  0.075),
    "claude-3-sonnet-20240229":       (0.003,  0.015),
}


def patch(client: Any) -> Any:
    """
    Patch an Anthropic client so every messages.create() call inside a
    @rubra.agent trace is automatically captured as an LLM_CALL span.

    Returns the same client object — no wrapper, no teardown needed.

    Usage:
        client = rubra.patch_anthropic(anthropic.Anthropic())
    """
    if not hasattr(client, "messages") or not hasattr(client.messages, "create"):
        raise TypeError(
            f"rubra.patch_anthropic() expects an Anthropic client, got {type(client).__name__}"
        )

    completions = client.messages
    if inspect.iscoroutinefunction(completions.create):
        _patch_async(completions)
    else:
        _patch_sync(completions)

    return client


def _patch_sync(messages: Any) -> None:
    original = messages.create

    def patched_create(**kwargs: Any) -> Any:
        trace = get_active_trace()
        if trace is None:
            return original(**kwargs)

        t0 = time.perf_counter()
        response = original(**kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        span = _build_llm_span(trace.trace_id, kwargs, response, elapsed_ms)
        trace.add_span(span)
        return response

    messages.create = patched_create


def _patch_async(messages: Any) -> None:
    original = messages.create

    async def patched_create(**kwargs: Any) -> Any:
        trace = get_active_trace()
        if trace is None:
            return await original(**kwargs)

        t0 = time.perf_counter()
        response = await original(**kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        span = _build_llm_span(trace.trace_id, kwargs, response, elapsed_ms)
        trace.add_span(span)
        return response

    messages.create = patched_create


def _build_llm_span(
    trace_id: str, kwargs: dict, response: Any, elapsed_ms: float
) -> Span:
    model = getattr(response, "model", kwargs.get("model", "unknown"))
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    total_tokens = prompt_tokens + completion_tokens
    stop_reason = getattr(response, "stop_reason", None)

    # Extract first text response
    text_response: str | None = None
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            text_response = block.text[:500]
            break

    cost = _estimate_cost(model, prompt_tokens, completion_tokens)
    messages = _sanitize_messages(kwargs.get("messages", []))

    span = Span(
        trace_id=trace_id,
        span_type=SpanType.LLM_CALL,
        name=f"llm:{model}",
        llm_data=LLMCallData(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            messages=messages,
            response=text_response,
            finish_reason=stop_reason,
        ),
    )
    span.finish()
    return span


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _ANTHROPIC_COST.get(model)
    if not rates:
        # fuzzy match: claude-3-5-sonnet → claude-3-5-sonnet-20241022
        for key, val in _ANTHROPIC_COST.items():
            if key in model or model in key:
                rates = val
                break
    if not rates:
        return 0.0
    input_rate, output_rate = rates
    return round(
        (prompt_tokens / 1000) * input_rate + (completion_tokens / 1000) * output_rate,
        8,
    )


def _sanitize_messages(messages: list) -> list[dict]:
    result = []
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "…"
            result.append({"role": m.get("role", ""), "content": content})
    return result
