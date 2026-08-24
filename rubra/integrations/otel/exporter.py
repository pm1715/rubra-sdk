"""
OpenTelemetry export integration.

Converts Rubra traces to OTEL spans and ships them to any compatible backend:
Langfuse, Arize Phoenix, Jaeger, Zipkin, or any OTLP endpoint.

Usage:
    from rubra.integrations.otel import enable_otel

    enable_otel(endpoint="http://localhost:4318", service_name="my-agent")

    # From here on, every @rubra.agent trace is automatically exported.
    # evaluate() still works identically — OTEL runs alongside, not instead.

Requires: pip install 'rubra[otel]'
"""
from __future__ import annotations

from typing import Any

from rubra.core.tracer.models import Span, SpanStatus, SpanType, Trace


def _require_otel() -> None:
    try:
        from opentelemetry import trace  # noqa: F401
    except ImportError:
        raise ImportError(
            "rubra[otel] requires opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http.\n"
            "Install with: pip install 'rubra[otel]'"
        )


def enable_otel(
    endpoint: str = "http://localhost:4318",
    service_name: str = "rubra",
    *,
    headers: dict[str, str] | None = None,
) -> None:
    """
    Configure global OTEL export and patch Rubra's trace store hook so every
    captured trace is also sent to the configured backend.

    Args:
        endpoint:     OTLP HTTP endpoint root (e.g. "http://localhost:4318").
                      The path /v1/traces is appended automatically.
        service_name: The OTEL resource service.name attribute.
        headers:      Optional auth headers, e.g. {"Authorization": "Bearer <key>"}.
                      Langfuse, Phoenix, and Grafana Tempo use this pattern.
    """
    _require_otel()

    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        headers=headers or {},
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    # Patch _store_trace so every Rubra trace also fans out to OTEL
    import rubra.core.tracer.decorators as _dec

    _original = _dec._store_trace

    def _store_and_export(t: Trace) -> None:
        _original(t)
        try:
            export_trace(t)
        except Exception:
            pass  # OTEL export must never crash the agent

    _dec._store_trace = _store_and_export


def export_trace(trace: Trace, tracer_name: str = "rubra") -> None:
    """
    Export a single Rubra Trace as an OTEL trace.

    The global TracerProvider must be configured first (call enable_otel()
    or set up opentelemetry manually).
    """
    _require_otel()

    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import SpanKind, StatusCode

    tracer = otel_trace.get_tracer(tracer_name)

    start_ns = _to_ns(trace.started_at)
    end_ns = _to_ns(trace.ended_at) if trace.ended_at else start_ns + 1_000_000

    root_attrs: dict[str, Any] = {
        "rubra.trace_id": trace.trace_id,
        "rubra.agent_name": trace.agent_name,
        "rubra.task": trace.task or "",
        "rubra.status": trace.status.value,
        "rubra.total_tool_calls": trace.total_tool_calls,
        "rubra.total_llm_calls": trace.total_llm_calls,
        "rubra.total_tokens": trace.token_usage.total_tokens,
        "rubra.cost_usd": trace.token_usage.estimated_cost_usd,
    }

    root = tracer.start_span(
        name=f"agent:{trace.agent_name}",
        kind=SpanKind.INTERNAL,
        start_time=start_ns,
        attributes=root_attrs,
    )

    if trace.status.value == "failed":
        root.set_status(StatusCode.ERROR, trace.error_message or "agent failed")
    else:
        root.set_status(StatusCode.OK)

    for span in trace.spans:
        _export_span(tracer, span)

    root.end(end_time=end_ns)


def _export_span(tracer: Any, span: Span) -> None:
    from opentelemetry.trace import SpanKind, StatusCode

    start_ns = _to_ns(span.started_at)
    end_ns = _to_ns(span.ended_at) if span.ended_at else start_ns + 1_000_000

    attrs: dict[str, Any] = {
        "rubra.span_id": span.span_id,
        "rubra.span_type": span.span_type.value,
    }

    if span.span_type == SpanType.TOOL_CALL and span.tool_call_data:
        attrs["rubra.tool_name"] = span.tool_call_data.tool_name
    elif span.span_type == SpanType.TOOL_RESPONSE and span.tool_response_data:
        attrs["rubra.tool_name"] = span.tool_response_data.tool_name
        if span.tool_response_data.error:
            attrs["rubra.tool_error"] = span.tool_response_data.error
    elif span.span_type == SpanType.LLM_CALL and span.llm_data:
        attrs["rubra.llm.model"] = span.llm_data.model
        attrs["rubra.llm.prompt_tokens"] = span.llm_data.prompt_tokens
        attrs["rubra.llm.completion_tokens"] = span.llm_data.completion_tokens
        attrs["rubra.llm.total_tokens"] = span.llm_data.total_tokens
        attrs["rubra.llm.cost_usd"] = span.llm_data.cost_usd

    child = tracer.start_span(
        name=span.name,
        kind=SpanKind.INTERNAL,
        start_time=start_ns,
        attributes=attrs,
    )
    if span.status == SpanStatus.ERROR:
        child.set_status(StatusCode.ERROR, span.error_message or "")
    else:
        child.set_status(StatusCode.OK)
    child.end(end_time=end_ns)


def _to_ns(dt: Any) -> int:
    """Convert a datetime to OTEL-compatible nanoseconds since epoch."""
    return int(dt.timestamp() * 1_000_000_000)
