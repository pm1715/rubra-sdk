"""
LangChain callback handler — records LLM calls and tool invocations as Rubra spans.

Usage:
    from rubra.integrations.langchain import RubraCallbackHandler
    import rubra

    handler = RubraCallbackHandler()

    @rubra.agent(task="...")
    def my_agent(question: str) -> str:
        chain = build_chain(callbacks=[handler])
        return chain.invoke({"question": question})

Works with any LangChain chain or agent — add the handler to the callbacks list.
When no Rubra trace is active the handler is a silent no-op.

Requires: pip install 'rubra[langchain]'
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from rubra.core.tracer.context import get_active_trace
from rubra.core.tracer.models import (
    LLMCallData,
    Span,
    SpanStatus,
    SpanType,
    ToolCallData,
    ToolResponseData,
)


def _require_langchain() -> None:
    try:
        from langchain_core.callbacks import BaseCallbackHandler  # noqa: F401
    except ImportError:
        raise ImportError(
            "rubra[langchain] requires langchain-core. "
            "Install with: pip install 'rubra[langchain]'"
        )


class RubraCallbackHandler:
    """
    LangChain callback handler that records LLM and tool activity as Rubra spans.

    Add to any chain or agent via the `callbacks` parameter.
    Silently passes through when no @rubra.agent trace is active.
    """

    def __init__(self) -> None:
        _require_langchain()
        self._pending_llm: dict[str, tuple[float, dict]] = {}  # run_id → (t0, kwargs)
        self._pending_tool: dict[str, tuple[float, str, Span]] = {}  # run_id → (t0, tool_name, call_span)

    # ------------------------------------------------------------------
    # LangChain BaseCallbackHandler interface (duck-typed — no inheritance
    # needed since LangChain accepts any object with these methods)
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        trace = get_active_trace()
        if trace is None:
            return
        model = (serialized.get("kwargs") or {}).get("model_name", "unknown")
        self._pending_llm[str(run_id)] = (time.perf_counter(), {"model": model, "prompts": prompts})

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        trace = get_active_trace()
        if trace is None:
            return

        run_key = str(run_id)
        if run_key not in self._pending_llm:
            return

        t0, meta = self._pending_llm.pop(run_key)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract token usage (provider-dependent, best-effort)
        llm_output = getattr(response, "llm_output", {}) or {}
        usage = llm_output.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # First generation text
        gen_text: str | None = None
        gens = getattr(response, "generations", [[]])
        if gens and gens[0]:
            gen_text = getattr(gens[0][0], "text", None)
            if gen_text:
                gen_text = gen_text[:500]

        model = meta["model"]
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.LLM_CALL,
            name=f"llm:{model}",
            llm_data=LLMCallData(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                response=gen_text,
            ),
        )
        span.finish()
        trace.add_span(span)

    def on_llm_error(self, error: Exception, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._pending_llm.pop(str(run_id), None)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        trace = get_active_trace()
        if trace is None:
            return

        tool_name = serialized.get("name", "unknown_tool")
        call_span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.TOOL_CALL,
            name=f"tool:{tool_name}",
            tool_call_data=ToolCallData(
                tool_name=tool_name,
                arguments={"input": input_str[:500]},
                raw_arguments=input_str[:500],
            ),
        )
        trace.add_span(call_span)
        self._pending_tool[str(run_id)] = (time.perf_counter(), tool_name, call_span)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        trace = get_active_trace()
        if trace is None:
            return

        run_key = str(run_id)
        if run_key not in self._pending_tool:
            return

        _t0, tool_name, call_span = self._pending_tool.pop(run_key)
        call_span.finish()

        resp_span = Span(
            trace_id=trace.trace_id,
            parent_span_id=call_span.span_id,
            span_type=SpanType.TOOL_RESPONSE,
            name=f"tool_response:{tool_name}",
            tool_response_data=ToolResponseData(
                tool_name=tool_name,
                output=str(output)[:2000] if output else None,
            ),
        )
        resp_span.finish()
        trace.add_span(resp_span)

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        trace = get_active_trace()
        run_key = str(run_id)
        if run_key not in self._pending_tool:
            return

        _t0, tool_name, call_span = self._pending_tool.pop(run_key)
        call_span.finish(
            status=SpanStatus.ERROR,
            error_message=str(error),
            error_type=type(error).__name__,
        )

        resp_span = Span(
            trace_id=trace.trace_id if trace else call_span.trace_id,
            parent_span_id=call_span.span_id,
            span_type=SpanType.TOOL_RESPONSE,
            name=f"tool_response:{tool_name}",
            tool_response_data=ToolResponseData(
                tool_name=tool_name,
                error=str(error),
            ),
        )
        resp_span.finish(status=SpanStatus.ERROR)
        if trace:
            trace.add_span(resp_span)

    # Chain-level hooks (no-op — we don't create spans for chain steps)
    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        pass
