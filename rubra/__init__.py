"""
Rubra — Agentic evaluation framework.
Every aspect, nothing missed.
"""

from rubra.__version__ import __version__
from rubra.core.evaluator.evaluator import EvalReport, evaluate, evaluate_async
from rubra.core.tracer.context import get_active_span, get_active_trace
from rubra.core.tracer.decorators import agent, tool
from rubra.core.tracer.models import Span, SpanType, Trace, TraceStatus
from rubra.integrations.anthropic.patch import patch as patch_anthropic
from rubra.integrations.openai.patch import patch as patch


def get_last_trace() -> "Trace | None":
    """Return the most recently completed trace (convenience for quickstart/scripts)."""
    from rubra.core.tracer.decorators import _last_trace

    return _last_trace


__all__ = [
    "__version__",
    # Primary API — what 99% of users touch
    "agent",
    "tool",
    "patch",  # rubra.patch(openai_client)
    "patch_anthropic",  # rubra.patch_anthropic(anthropic.Anthropic())
    "evaluate",
    "evaluate_async",
    "get_last_trace",
    # Types
    "EvalReport",
    "Trace",
    "Span",
    "SpanType",
    "TraceStatus",
    # Context access (advanced usage)
    "get_active_trace",
    "get_active_span",
]
