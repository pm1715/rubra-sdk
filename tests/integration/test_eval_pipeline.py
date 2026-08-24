"""
Integration tests — full evaluate() pipeline with a real trace.
"""
from __future__ import annotations

import pytest

from rubra.core.tracer.models import (
    Span, SpanStatus, SpanType,
    ToolCallData, ToolResponseData,
    Trace, TraceStatus,
)
from rubra.core.evaluator.evaluator import evaluate, EvalReport


def _build_trace(
    agent_name: str = "test_agent",
    task: str = "Answer a question",
    output: str = "The answer is 42.",
    *,
    add_tool: bool = True,
    expected_tool_calls: list[str] | None = None,
) -> Trace:
    trace = Trace(
        agent_name=agent_name,
        task=task,
        expected_tool_calls=expected_tool_calls,
    )

    if add_tool:
        call = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.TOOL_CALL,
            name="tool:search",
            tool_call_data=ToolCallData(
                tool_name="search",
                arguments={"query": "the answer"},
            ),
        )
        call.finish()
        trace.add_span(call)

        resp = Span(
            trace_id=trace.trace_id,
            parent_span_id=call.span_id,
            span_type=SpanType.TOOL_RESPONSE,
            name="tool_response:search",
            tool_response_data=ToolResponseData(
                tool_name="search",
                output="42 is the answer to life, universe, and everything",
            ),
        )
        resp.finish()
        trace.add_span(resp)

    trace.finish(output=output)
    return trace


# ---------------------------------------------------------------------------
# evaluate() returns EvalReport
# ---------------------------------------------------------------------------


def test_evaluate_returns_report():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    assert isinstance(report, EvalReport)


def test_evaluate_all_execution_metrics_present():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    names = {r.metric_name for r in report.results}
    assert "task_completion_rate" in names
    assert "tool_call_success_rate" in names
    assert "error_rate" in names
    assert "step_efficiency" in names
    assert "latency_score" in names
    assert "token_efficiency" in names
    assert "cost_efficiency" in names


def test_evaluate_tool_metrics_present():
    trace = _build_trace(expected_tool_calls=["search"])
    report = evaluate(trace, metrics="tool", persist=False)
    names = {r.metric_name for r in report.results}
    assert "tool_selection_precision" in names
    assert "tool_chain_validity" in names


def test_evaluate_safety_metrics_present():
    trace = _build_trace()
    report = evaluate(trace, metrics="safety", persist=False)
    names = {r.metric_name for r in report.results}
    assert "prompt_injection_resistance" in names
    assert "scope_creep_score" in names
    assert "pii_propagation_count" in names


def test_evaluate_quality_metrics_present():
    trace = _build_trace()
    report = evaluate(trace, metrics="quality", persist=False)
    names = {r.metric_name for r in report.results}
    assert "answer_relevance_proxy" in names
    assert "output_coherence_score" in names


# ---------------------------------------------------------------------------
# Composite scores computed correctly
# ---------------------------------------------------------------------------


def test_rubra_score_is_float():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    assert report.rubra_score is not None
    assert 0.0 <= report.rubra_score <= 1.0


def test_tool_intelligence_score_computed():
    trace = _build_trace(expected_tool_calls=["search"])
    report = evaluate(trace, metrics="all", persist=False)
    if report.tool_intelligence_score is not None:
        assert 0.0 <= report.tool_intelligence_score <= 1.0


def test_agentic_efficiency_score_computed():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    if report.agentic_efficiency_score is not None:
        assert 0.0 <= report.agentic_efficiency_score <= 1.0


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def test_passed_failed_counts_consistent():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    assert report.passed + report.failed + report.not_applicable == report.total_metrics


def test_average_score_is_mean_of_scored():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    scored = [r.score for r in report.results if r.score is not None]
    if scored:
        expected = sum(scored) / len(scored)
        assert abs(report.average_score - expected) < 1e-9


# ---------------------------------------------------------------------------
# evaluate() with "all" runs every category
# ---------------------------------------------------------------------------


def test_evaluate_all_runs_every_category():
    trace = _build_trace()
    report = evaluate(trace, metrics="all", persist=False)
    categories = {r.category for r in report.results}
    assert "execution" in categories
    assert "tool" in categories
    assert "safety" in categories
    assert "quality" in categories
    # goal may be skipped (requires litellm), but should not raise


# ---------------------------------------------------------------------------
# evaluate() with specific metric list
# ---------------------------------------------------------------------------


def test_evaluate_specific_metric_list():
    trace = _build_trace()
    report = evaluate(
        trace,
        metrics=["task_completion_rate", "error_rate"],
        persist=False,
    )
    names = {r.metric_name for r in report.results}
    assert names == {"task_completion_rate", "error_rate"}


# ---------------------------------------------------------------------------
# EvalReport.summary()
# ---------------------------------------------------------------------------


def test_eval_report_summary_contains_agent_name():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    summary = report.summary()
    assert "test_agent" in summary


def test_eval_report_get():
    trace = _build_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    r = report.get("task_completion_rate")
    assert r is not None
    assert r.metric_name == "task_completion_rate"


def test_eval_report_by_category():
    trace = _build_trace()
    report = evaluate(trace, metrics="all", persist=False)
    cats = report.by_category()
    assert "execution" in cats
