"""
Unit tests for EvalReport.to_html() and the rubra report CLI command.
"""
from __future__ import annotations

import os
import tempfile
import pytest

from rubra.core.tracer.models import Trace, TraceStatus
from rubra.core.evaluator.evaluator import evaluate, EvalReport


def _make_completed_trace(task: str = "test task") -> Trace:
    trace = Trace(agent_name="test_agent", task=task)
    trace.finish(output="Done")
    return trace


# ---------------------------------------------------------------------------
# EvalReport.to_html() — string output
# ---------------------------------------------------------------------------


def test_to_html_returns_string():
    trace = _make_completed_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    html = report.to_html()
    assert isinstance(html, str)
    assert html.startswith("<!DOCTYPE html>")


def test_to_html_contains_agent_name():
    trace = _make_completed_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    html = report.to_html()
    assert "test_agent" in html


def test_to_html_contains_rubra_branding():
    trace = _make_completed_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    html = report.to_html()
    assert "Rubra" in html
    assert "#dc2626" in html  # Rubra red


def test_to_html_contains_pass_fail():
    trace = _make_completed_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    html = report.to_html()
    assert "PASS" in html or "FAIL" in html or "N/A" in html


def test_to_html_composites_shown_when_present():
    trace = _make_completed_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    html = report.to_html()
    if report.rubra_score is not None:
        assert "Rubra Score" in html


# ---------------------------------------------------------------------------
# EvalReport.to_html() — file output
# ---------------------------------------------------------------------------


def test_to_html_writes_file():
    trace = _make_completed_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        result = report.to_html(path=path)
        assert result == path
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "test_agent" in content
    finally:
        os.unlink(path)


def test_to_html_file_is_valid_html():
    trace = _make_completed_trace()
    report = evaluate(trace, metrics="execution", persist=False)
    html = report.to_html()
    assert "<html" in html
    assert "</html>" in html
    assert "<title>" in html


# ---------------------------------------------------------------------------
# Escaping: agent names with special HTML chars
# ---------------------------------------------------------------------------


def test_to_html_escapes_agent_name():
    trace = Trace(agent_name='<script>alert("xss")</script>', task="xss test")
    trace.finish(output="safe")
    report = evaluate(trace, metrics="execution", persist=False)
    html = report.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
