"""
Integration tests — real SQLite file on disk, not in-memory.
Verifies the full storage round-trip works end-to-end.
"""
from __future__ import annotations

import os
import tempfile
import pytest

from rubra.core.storage.db import RubraStorage, init_storage, get_storage
from rubra.core.tracer.models import (
    Span, SpanStatus, SpanType, ToolCallData, ToolResponseData,
    Trace, TraceStatus, TokenUsage,
)


@pytest.fixture()
def db_path(tmp_path):
    """A temporary SQLite database file."""
    return str(tmp_path / "test_rubra.db")


@pytest.fixture()
def storage(db_path):
    return RubraStorage(f"sqlite:///{db_path}")


# ---------------------------------------------------------------------------
# save_trace + get_trace
# ---------------------------------------------------------------------------


def test_save_and_get_trace(storage):
    trace = Trace(agent_name="integration_agent", task="answer a question")
    trace.finish(output="42 is the answer")

    storage.save_trace(trace)

    retrieved = storage.get_trace(trace.trace_id)
    assert retrieved is not None
    assert retrieved.trace_id == trace.trace_id
    assert retrieved.agent_name == "integration_agent"
    assert retrieved.task == "answer a question"
    assert retrieved.final_output == "42 is the answer"
    assert retrieved.status == TraceStatus.COMPLETED


def test_get_nonexistent_trace_returns_none(storage):
    result = storage.get_trace("00000000-0000-0000-0000-000000000000")
    assert result is None


def test_trace_with_spans_round_trips(storage):
    trace = Trace(agent_name="span_agent", task="use a tool")
    call_span = Span(
        trace_id=trace.trace_id,
        span_type=SpanType.TOOL_CALL,
        name="tool:search",
        tool_call_data=ToolCallData(
            tool_name="search",
            arguments={"query": "capitals of Europe"},
        ),
    )
    call_span.finish()
    trace.add_span(call_span)

    resp_span = Span(
        trace_id=trace.trace_id,
        parent_span_id=call_span.span_id,
        span_type=SpanType.TOOL_RESPONSE,
        name="tool_response:search",
        tool_response_data=ToolResponseData(
            tool_name="search",
            output="Paris, Berlin, Rome...",
        ),
    )
    resp_span.finish()
    trace.add_span(resp_span)
    trace.finish(output="Paris")

    storage.save_trace(trace)
    retrieved = storage.get_trace(trace.trace_id)

    assert len(retrieved.spans) == 2
    assert retrieved.spans[0].span_type == SpanType.TOOL_CALL
    assert retrieved.spans[0].tool_call_data.tool_name == "search"
    assert retrieved.spans[1].span_type == SpanType.TOOL_RESPONSE


def test_upsert_updates_existing_trace(storage):
    trace = Trace(agent_name="upsert_agent", task="test upsert")
    trace.finish(output="first")
    storage.save_trace(trace)

    # Update and save again — should not create a duplicate
    trace.final_output = "updated"
    storage.save_trace(trace)

    all_traces = storage.list_traces()
    matching = [t for t in all_traces if t.trace_id == trace.trace_id]
    assert len(matching) == 1
    assert matching[0].final_output == "updated"


# ---------------------------------------------------------------------------
# list_traces
# ---------------------------------------------------------------------------


def test_list_traces_returns_all(storage):
    for i in range(3):
        t = Trace(agent_name=f"agent_{i}", task=f"task {i}")
        t.finish()
        storage.save_trace(t)

    traces = storage.list_traces()
    assert len(traces) == 3


def test_list_traces_filter_by_agent(storage):
    for name in ["alpha", "alpha", "beta"]:
        t = Trace(agent_name=name, task="task")
        t.finish()
        storage.save_trace(t)

    alpha_traces = storage.list_traces(agent_name="alpha")
    assert len(alpha_traces) == 2
    assert all(t.agent_name == "alpha" for t in alpha_traces)


def test_list_traces_limit(storage):
    for i in range(5):
        t = Trace(agent_name="agent", task=f"task {i}")
        t.finish()
        storage.save_trace(t)

    traces = storage.list_traces(limit=3)
    assert len(traces) == 3


def test_list_traces_ordered_newest_first(storage):
    import time
    traces_created = []
    for i in range(3):
        t = Trace(agent_name="agent", task=f"task {i}")
        time.sleep(0.01)
        t.finish()
        storage.save_trace(t)
        traces_created.append(t.trace_id)

    listed = storage.list_traces()
    assert listed[0].trace_id == traces_created[-1]  # newest first


# ---------------------------------------------------------------------------
# save_metric_result + get_metric_results
# ---------------------------------------------------------------------------


def test_save_and_get_metric_results(storage):
    trace_id = "test-metrics-trace-001"
    storage.save_metric_result(
        trace_id=trace_id,
        metric_name="task_completion_rate",
        score=1.0,
        passed=True,
        reason="Trace completed successfully",
        category="execution",
        is_deterministic=True,
    )
    storage.save_metric_result(
        trace_id=trace_id,
        metric_name="error_rate",
        score=0.95,
        passed=True,
        reason="Low error rate",
        category="execution",
        is_deterministic=True,
    )

    results = storage.get_metric_results(trace_id)
    assert len(results) == 2
    names = {r["metric_name"] for r in results}
    assert "task_completion_rate" in names
    assert "error_rate" in names


def test_metric_result_score_none(storage):
    trace_id = "test-na-metric-001"
    storage.save_metric_result(
        trace_id=trace_id,
        metric_name="hallucination_score",
        score=None,
        passed=None,
        category="goal",
        is_deterministic=False,
    )
    results = storage.get_metric_results(trace_id)
    assert len(results) == 1
    assert results[0]["score"] is None


# ---------------------------------------------------------------------------
# init_storage creates file in .rubra/
# ---------------------------------------------------------------------------


def test_init_storage_creates_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = init_storage()
    db_file = tmp_path / ".rubra" / "rubra.db"
    assert db_file.exists()
