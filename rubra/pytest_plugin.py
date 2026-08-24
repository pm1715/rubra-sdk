"""
Rubra pytest plugin — auto-registered when rubra is installed.

Provides the `rubra_trace` fixture for capturing and asserting on
agent traces in tests, with no boilerplate required.

Usage:
    # conftest.py — nothing needed, plugin registers automatically

    # test_my_agent.py
    def test_capital_agent(rubra_trace):
        result = my_agent("What is the capital of France?")
        assert result == "Paris"

        report = rubra_trace.evaluate(metrics="execution")
        assert report.rubra_score >= 0.7
        assert report.get("task_completion_rate").passed

    # Quick one-liner assertion:
    def test_passes_eval(rubra_trace):
        my_agent("Who wrote Hamlet?")
        rubra_trace.assert_score(min_rubra_score=0.7, min_pass_rate=0.8)
"""
from __future__ import annotations

from typing import Any

import pytest

from rubra.core.evaluator.evaluator import EvalReport, evaluate
from rubra.core.tracer.models import Trace


class RubraTestRecorder:
    """
    Collects traces produced during a single test.
    Returned by the `rubra_trace` fixture.
    """

    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def _record(self, trace: Trace) -> None:
        self.traces.append(trace)

    @property
    def last(self) -> Trace | None:
        """The most recently captured trace."""
        return self.traces[-1] if self.traces else None

    def evaluate(
        self,
        metrics: str = "all",
        *,
        trace: Trace | None = None,
        **kwargs: Any,
    ) -> EvalReport:
        """
        Run evaluation on the last captured trace (or pass an explicit trace).

        Args:
            metrics: "all", "execution", "tool", "safety", "quality", or "goal".
            trace:   Override which trace to evaluate.
            **kwargs: Forwarded to rubra.evaluate() (max_steps, target_ms, etc.).
        """
        t = trace or self.last
        if t is None:
            raise AssertionError(
                "No Rubra traces captured in this test.\n"
                "Make sure your agent function is decorated with @rubra.agent."
            )
        return evaluate(t, metrics=metrics, persist=False, **kwargs)

    def assert_score(
        self,
        metrics: str = "all",
        *,
        min_rubra_score: float | None = None,
        min_pass_rate: float | None = None,
        **kwargs: Any,
    ) -> EvalReport:
        """
        Evaluate and assert composite scores in one call.

        Args:
            metrics:         Metric set to evaluate.
            min_rubra_score: Assert rubra_score >= this value (0–1).
            min_pass_rate:   Assert (passed / total_metrics) >= this value (0–1).

        Returns the EvalReport so you can inspect individual metrics too.
        """
        report = self.evaluate(metrics=metrics, **kwargs)
        failures: list[str] = []

        if min_rubra_score is not None and report.rubra_score is not None:
            if report.rubra_score < min_rubra_score:
                failures.append(
                    f"rubra_score {report.rubra_score:.3f} < min {min_rubra_score}"
                )

        if min_pass_rate is not None and report.total_metrics > 0:
            rate = report.passed / report.total_metrics
            if rate < min_pass_rate:
                failures.append(
                    f"pass_rate {rate:.3f} ({report.passed}/{report.total_metrics}) < min {min_pass_rate}"
                )

        if failures:
            raise AssertionError(
                "Rubra eval assertion failed:\n"
                + "\n".join(f"  • {f}" for f in failures)
                + f"\n\nFull report:\n{report.summary()}"
            )

        return report


@pytest.fixture
def rubra_trace(monkeypatch: pytest.MonkeyPatch) -> RubraTestRecorder:
    """
    Pytest fixture that captures Rubra traces produced during a test.

    Example:
        def test_my_agent(rubra_trace):
            my_agent("question")
            report = rubra_trace.evaluate(metrics="execution")
            assert report.rubra_score >= 0.7
    """
    import rubra.core.tracer.decorators as _dec

    recorder = RubraTestRecorder()
    original_store = _dec._store_trace

    def _capture_and_store(trace: Trace) -> None:
        recorder._record(trace)
        original_store(trace)

    monkeypatch.setattr(_dec, "_store_trace", _capture_and_store)
    return recorder
