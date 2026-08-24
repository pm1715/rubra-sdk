"""
Unit tests for rubra.integrations.otel — no actual OTEL backend required.
Tests graceful error handling and structural correctness of the exporter.
"""
from __future__ import annotations

import builtins
import pytest

from rubra.integrations.otel.exporter import _require_otel, _to_ns
from rubra.core.tracer.models import Trace, TraceStatus, Span, SpanType, SpanStatus


# ---------------------------------------------------------------------------
# _require_otel: fails gracefully when SDK not installed
# ---------------------------------------------------------------------------


def test_require_otel_raises_when_missing(monkeypatch):
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="rubra\\[otel\\]"):
        _require_otel()


# ---------------------------------------------------------------------------
# _to_ns: timestamp conversion
# ---------------------------------------------------------------------------


def test_to_ns_is_integer():
    from datetime import datetime, timezone
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ns = _to_ns(dt)
    assert isinstance(ns, int)
    assert ns > 0


def test_to_ns_is_nanoseconds():
    from datetime import datetime, timezone
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ns = _to_ns(dt)
    # 2024-01-01 is approx 1.7e18 nanoseconds since epoch
    assert ns > 1_700_000_000_000_000_000


# ---------------------------------------------------------------------------
# export_trace: smoke test with in-process OTEL SDK if available
# ---------------------------------------------------------------------------


def _otel_available() -> bool:
    try:
        import opentelemetry  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _otel_available(), reason="opentelemetry not installed")
def test_export_trace_smoke():
    """When otel SDK is present, export_trace should not raise."""
    from rubra.integrations.otel.exporter import export_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry import trace as otel_trace

    provider = TracerProvider()
    otel_trace.set_tracer_provider(provider)

    trace = Trace(agent_name="smoke_agent", task="test export")
    trace.finish(output="done")

    # Should not raise
    export_trace(trace)


@pytest.mark.skipif(not _otel_available(), reason="opentelemetry not installed")
def test_export_trace_with_spans():
    """Spans are iterated without error during export."""
    from rubra.integrations.otel.exporter import export_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry import trace as otel_trace
    from rubra.core.tracer.models import ToolCallData, ToolResponseData

    provider = TracerProvider()
    otel_trace.set_tracer_provider(provider)

    trace = Trace(agent_name="span_agent", task="test spans")
    call_span = Span(
        trace_id=trace.trace_id,
        span_type=SpanType.TOOL_CALL,
        name="tool:search",
        tool_call_data=ToolCallData(tool_name="search", arguments={"q": "test"}),
    )
    call_span.finish()
    trace.add_span(call_span)
    trace.finish(output="found")

    export_trace(trace)  # should not raise
