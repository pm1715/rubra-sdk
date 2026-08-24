"""
Unit tests for all 13 deterministic execution metrics.
Builds synthetic traces — no decorators, no storage, no LLMs.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from rubra.core.tracer.models import (
    Span, SpanStatus, SpanType, Trace, TraceStatus,
    ToolCallData, ToolResponseData, TokenUsage,
)
from rubra.core.metrics.execution.metrics import (
    run_execution_metrics,
    task_completion_rate,
    tool_call_success_rate,
    error_rate,
    step_efficiency,
    latency_score,
    token_efficiency,
    cost_efficiency,
    tool_diversity,
    retry_rate,
    hallucination_free_calls,
    response_completeness,
    tool_output_utilization,
    execution_time_distribution,
)


def make_trace(
    status: TraceStatus = TraceStatus.COMPLETED,
    output: str = "The answer is Paris.",
    tools: list[str] | None = None,
    error_tools: list[str] | None = None,
    duration_ms: float = 300.0,
    total_tokens: int = 640,
    cost_usd: float = 0.002,
) -> Trace:
    t0 = datetime.now(timezone.utc)
    trace = Trace(
        agent_name="test_agent",
        task="Find capital",
        token_usage=TokenUsage(
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
        ),
    )

    for i, tool_name in enumerate(tools or ["search"]):
        is_error = error_tools and tool_name in error_tools
        call = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.TOOL_CALL,
            name=f"tool:{tool_name}",
            started_at=t0 + timedelta(milliseconds=i * 100),
            ended_at=t0 + timedelta(milliseconds=(i * 100) + 50),
            status=SpanStatus.ERROR if is_error else SpanStatus.OK,
            tool_call_data=ToolCallData(
                tool_name=tool_name,
                arguments={"query": f"test {i}"},
            ),
        )
        call.finish(status=SpanStatus.ERROR if is_error else SpanStatus.OK)
        resp = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.TOOL_RESPONSE,
            name=f"tool_response:{tool_name}",
            started_at=t0 + timedelta(milliseconds=(i * 100) + 50),
            ended_at=t0 + timedelta(milliseconds=(i * 100) + 60),
            tool_response_data=ToolResponseData(
                tool_name=tool_name,
                output=f"Result for {tool_name}",
                error="boom" if is_error else None,
            ),
        )
        resp.finish()
        trace.spans.extend([call, resp])

    trace.status = status
    trace.final_output = output
    trace.started_at = t0
    trace.ended_at = t0 + timedelta(milliseconds=duration_ms)
    trace.duration_ms = duration_ms
    return trace


# ---------------------------------------------------------------------------
# task_completion_rate
# ---------------------------------------------------------------------------


def test_task_completion_rate_completed():
    t = make_trace(status=TraceStatus.COMPLETED)
    r = task_completion_rate(t)
    assert r.score == 1.0
    assert r.passed is True


def test_task_completion_rate_failed():
    t = make_trace(status=TraceStatus.FAILED)
    r = task_completion_rate(t)
    assert r.score == 0.0
    assert r.passed is False


# ---------------------------------------------------------------------------
# tool_call_success_rate
# ---------------------------------------------------------------------------


def test_tool_success_rate_all_pass():
    t = make_trace(tools=["search", "summarize"])
    r = tool_call_success_rate(t)
    assert r.score == 1.0


def test_tool_success_rate_partial_fail():
    t = make_trace(tools=["search", "summarize"], error_tools=["summarize"])
    r = tool_call_success_rate(t)
    assert r.score == pytest.approx(0.5)


def test_tool_success_rate_no_tools():
    t = Trace(agent_name="no_tools")
    t.finish(output="done")
    r = tool_call_success_rate(t)
    assert r.score is None


# ---------------------------------------------------------------------------
# error_rate
# ---------------------------------------------------------------------------


def test_error_rate_no_errors():
    t = make_trace()
    r = error_rate(t)
    assert r.score == 1.0
    assert r.passed is True


def test_error_rate_with_errors():
    t = make_trace(tools=["search"], error_tools=["search"])
    r = error_rate(t)
    assert r.score < 1.0


# ---------------------------------------------------------------------------
# step_efficiency
# ---------------------------------------------------------------------------


def test_step_efficiency_within_budget():
    t = make_trace(tools=["a", "b"])
    r = step_efficiency(t, max_steps=10)
    assert r.score is not None
    assert r.score > 0


def test_step_efficiency_over_budget():
    t = make_trace(tools=[f"tool_{i}" for i in range(12)])
    r = step_efficiency(t, max_steps=10)
    assert r.score == 0.0
    assert r.passed is False


# ---------------------------------------------------------------------------
# latency_score
# ---------------------------------------------------------------------------


def test_latency_score_fast():
    t = make_trace(duration_ms=500.0)
    r = latency_score(t, target_ms=5000.0)
    assert r.score == pytest.approx(0.9, abs=0.01)
    assert r.passed is True


def test_latency_score_slow():
    t = make_trace(duration_ms=10000.0)
    r = latency_score(t, target_ms=5000.0)
    assert r.score == 0.0
    assert r.passed is False


# ---------------------------------------------------------------------------
# token_efficiency
# ---------------------------------------------------------------------------


def test_token_efficiency_within_budget():
    t = make_trace(total_tokens=1000)
    r = token_efficiency(t, target_tokens=4096)
    assert r.score > 0.7


def test_token_efficiency_over_budget():
    t = make_trace(total_tokens=8192)
    r = token_efficiency(t, target_tokens=4096)
    assert r.score == 0.0


# ---------------------------------------------------------------------------
# cost_efficiency
# ---------------------------------------------------------------------------


def test_cost_efficiency_within_budget():
    t = make_trace(cost_usd=0.05)
    r = cost_efficiency(t, budget_usd=0.10)
    assert r.score == 1.0
    assert r.passed is True


def test_cost_efficiency_over_budget():
    t = make_trace(cost_usd=0.25)
    r = cost_efficiency(t, budget_usd=0.10)
    assert r.score == 0.0
    assert r.passed is False


# ---------------------------------------------------------------------------
# tool_diversity
# ---------------------------------------------------------------------------


def test_tool_diversity_all_unique():
    t = make_trace(tools=["search", "summarize", "translate"])
    r = tool_diversity(t)
    assert r.score == 1.0


def test_tool_diversity_all_same():
    t = make_trace(tools=["search", "search", "search"])
    r = tool_diversity(t)
    assert r.score == pytest.approx(1 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# response_completeness
# ---------------------------------------------------------------------------


def test_response_completeness_present():
    t = make_trace(output="The capital of France is Paris, a beautiful city.")
    r = response_completeness(t, min_length=10)
    assert r.passed is True


def test_response_completeness_empty():
    t = make_trace(output="")
    r = response_completeness(t)
    assert r.score == 0.0
    assert r.passed is False


# ---------------------------------------------------------------------------
# run_execution_metrics (all 13 together)
# ---------------------------------------------------------------------------


def test_run_all_13_metrics():
    t = make_trace(tools=["search", "summarize"])
    results = run_execution_metrics(t)
    assert len(results) == 13
    names = {r.metric_name for r in results}
    assert "task_completion_rate" in names
    assert "tool_call_success_rate" in names
    assert "error_rate" in names
    assert "tool_diversity" in names
    assert "tool_output_utilization" in names
    assert "execution_time_distribution" in names
