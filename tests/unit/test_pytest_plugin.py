"""
Tests for rubra.pytest_plugin — the rubra_trace fixture.
Meta-test: we test the fixture itself using monkeypatch.
"""
from __future__ import annotations

import pytest
import rubra
import rubra.core.tracer.decorators as dec
from rubra.pytest_plugin import RubraTestRecorder
from rubra.core.tracer.models import Trace, TraceStatus


# ---------------------------------------------------------------------------
# Basic capture
# ---------------------------------------------------------------------------


def test_recorder_starts_empty():
    recorder = RubraTestRecorder()
    assert recorder.traces == []
    assert recorder.last is None


def test_rubra_trace_fixture_captures_trace(rubra_trace):
    @rubra.agent(task="test capture")
    def ag() -> str:
        return "done"

    ag()
    assert rubra_trace.last is not None
    assert rubra_trace.last.status == TraceStatus.COMPLETED


def test_rubra_trace_fixture_captures_multiple(rubra_trace):
    @rubra.agent(task="t1")
    def ag1() -> str:
        return "a"

    @rubra.agent(task="t2")
    def ag2() -> str:
        return "b"

    ag1()
    ag2()
    assert len(rubra_trace.traces) == 2


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_rubra_trace_evaluate_returns_report(rubra_trace):
    @rubra.agent(task="evaluate test")
    def ag() -> str:
        return "The answer is 42."

    ag()
    report = rubra_trace.evaluate(metrics="execution")
    assert report.total_metrics > 0
    assert report.rubra_score is not None


def test_rubra_trace_evaluate_raises_when_no_trace():
    recorder = RubraTestRecorder()
    with pytest.raises(AssertionError, match="No Rubra traces"):
        recorder.evaluate()


def test_rubra_trace_evaluate_specific_metrics(rubra_trace):
    @rubra.agent(task="specific metrics")
    def ag() -> str:
        return "done"

    ag()
    report = rubra_trace.evaluate(metrics="execution")
    names = {r.metric_name for r in report.results}
    assert "task_completion_rate" in names


# ---------------------------------------------------------------------------
# assert_score()
# ---------------------------------------------------------------------------


def test_assert_score_passes_when_above_threshold(rubra_trace):
    @rubra.agent(task="passing agent")
    def ag() -> str:
        return "The answer to the question is here."

    ag()
    # rubra_score for a completed trace with output should be > 0
    report = rubra_trace.assert_score(metrics="execution", min_rubra_score=0.0)
    assert report is not None


def test_assert_score_fails_when_below_threshold(rubra_trace):
    @rubra.agent(task="failing threshold")
    def ag() -> str:
        return "done"

    ag()
    with pytest.raises(AssertionError, match="rubra_score"):
        rubra_trace.assert_score(metrics="execution", min_rubra_score=1.1)


def test_assert_score_pass_rate(rubra_trace):
    @rubra.agent(task="pass rate test")
    def ag() -> str:
        return "The answer is complete and detailed."

    ag()
    # min_pass_rate=0.0 should always pass
    rubra_trace.assert_score(metrics="execution", min_pass_rate=0.0)


# ---------------------------------------------------------------------------
# get_last_trace()
# ---------------------------------------------------------------------------


def test_get_last_trace_returns_most_recent(rubra_trace):
    @rubra.agent(task="last trace test")
    def ag() -> str:
        return "result"

    ag()
    last = rubra.get_last_trace()
    assert last is not None
    assert last.task == "last trace test"
