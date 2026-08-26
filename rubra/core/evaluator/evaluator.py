"""
Main evaluate() entry point.
Accepts a Trace, runs requested metrics, persists results, returns EvalReport.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any, Literal

from rubra.core.metrics.execution.metrics import MetricResult, run_execution_metrics
from rubra.core.tracer.models import Trace

# ---------------------------------------------------------------------------
# EvalReport — structured output of an evaluation run
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    trace_id: str
    agent_name: str
    task: str | None

    # Flat list of all metric results
    results: list[MetricResult] = field(default_factory=list)

    # Summary stats
    total_metrics: int = 0
    passed: int = 0
    failed: int = 0
    not_applicable: int = 0
    average_score: float | None = None

    # Rubra composite scores (set by _compute_composites)
    rubra_score: float | None = None
    tool_intelligence_score: float | None = None
    agentic_efficiency_score: float | None = None

    evaluated_at: float = field(default_factory=time.time)
    evaluation_ms: float | None = None

    def __post_init__(self) -> None:
        self._recompute_summary()

    def _recompute_summary(self) -> None:
        self.total_metrics = len(self.results)
        scored = [r for r in self.results if r.score is not None]
        self.passed = sum(1 for r in self.results if r.passed is True)
        self.failed = sum(1 for r in self.results if r.passed is False)
        self.not_applicable = sum(1 for r in self.results if r.score is None)
        self.average_score = (
            sum(r.score for r in scored) / len(scored) if scored else None
        )

    def by_category(self) -> dict[str, list[MetricResult]]:
        cats: dict[str, list[MetricResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        return cats

    def get(self, metric_name: str) -> MetricResult | None:
        for r in self.results:
            if r.metric_name == metric_name:
                return r
        return None

    def summary(self) -> str:
        lines = [
            f"EvalReport for '{self.agent_name}' (trace {self.trace_id[:8]}…)",
            f"  Metrics: {self.total_metrics}  |  Pass: {self.passed}  "
            f"Fail: {self.failed}  N/A: {self.not_applicable}",
            (
                f"  Avg score: {self.average_score:.3f}"
                if self.average_score is not None
                else "  Avg score: N/A"
            ),
        ]
        if self.rubra_score is not None:
            lines.append(f"  Rubra Score: {self.rubra_score:.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_html(self, path: str | None = None) -> str:
        """
        Generate a self-contained HTML eval report.

        Args:
            path: If provided, write the HTML to this file path and return the path.
                  If None, return the HTML string directly.
        Returns:
            HTML string (or path string if path was given).
        """
        import html as _html
        from datetime import datetime

        from rubra.__version__ import __version__

        def _score_color(score: float | None, passed: bool | None) -> str:
            if score is None:
                return "#888"
            if passed is True:
                return "#22c55e"
            if passed is False:
                return "#ef4444"
            return "#f59e0b"

        def _badge(passed: bool | None) -> str:
            style = "color:#fff;padding:2px 8px;border-radius:4px;font-size:11px"
            if passed is True:
                return f'<span style="background:#22c55e;{style}">PASS</span>'
            if passed is False:
                return f'<span style="background:#ef4444;{style}">FAIL</span>'
            return f'<span style="background:#6b7280;{style}">N/A</span>'

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "—"

        _STAT_BOX_STYLE = (
            "text-align:center;padding:16px;background:#fff;"
            "border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08)"
        )
        _STAT_LABEL_STYLE = (
            "font-size:11px;color:#6b7280;margin-top:4px;"
            "text-transform:uppercase;letter-spacing:.5px"
        )

        cats = self.by_category()
        cat_sections = ""
        for cat, results in cats.items():
            rows = ""
            for r in results:
                sc = f"{r.score:.4f}" if r.score is not None else "—"
                reason = _html.escape(r.reason or "")[:120]
                det = "✓" if r.is_deterministic else "LLM"
                rows += f"""
                <tr>
                  <td style="padding:8px 12px;font-family:monospace;font-size:12px">
                    {_html.escape(r.metric_name)}</td>
                  <td style="padding:8px 12px;text-align:right;font-weight:600;
                    color:{_score_color(r.score, r.passed)}">{sc}</td>
                  <td style="padding:8px 12px;text-align:center">{_badge(r.passed)}</td>
                  <td style="padding:8px 12px;font-size:11px;color:#666">{det}</td>
                  <td style="padding:8px 12px;font-size:11px;color:#555">{reason}</td>
                </tr>"""
            th_style = "padding:8px 12px;font-size:11px;color:#6b7280;font-weight:600"
            cat_sections += f"""
            <div style="margin-bottom:28px">
              <h3 style="text-transform:uppercase;letter-spacing:1px;font-size:12px;
                color:#dc2626;margin:0 0 8px">{_html.escape(cat)}</h3>
              <table style="width:100%;border-collapse:collapse;background:#fff;
                border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)">
                <thead>
                  <tr style="background:#f9fafb;border-bottom:1px solid #e5e7eb">
                    <th style="text-align:left;{th_style}">METRIC</th>
                    <th style="text-align:right;{th_style}">SCORE</th>
                    <th style="text-align:center;{th_style}">RESULT</th>
                    <th style="text-align:center;{th_style}">TYPE</th>
                    <th style="text-align:left;{th_style}">REASON</th>
                  </tr>
                </thead>
                <tbody>{rows}
                </tbody>
              </table>
            </div>"""

        composites = ""
        for label, val in [
            ("Rubra Score", self.rubra_score),
            ("Tool Intelligence", self.tool_intelligence_score),
            ("Agentic Efficiency", self.agentic_efficiency_score),
        ]:
            if val is not None:
                composites += f"""
                <div style="text-align:center;padding:16px 24px;background:#fff;
                  border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08)">
                  <div style="font-size:28px;font-weight:700;color:#dc2626">
                    {val:.3f}</div>
                  <div style="{_STAT_LABEL_STYLE}">{label}</div>
                </div>"""

        evaluated_at = datetime.fromtimestamp(self.evaluated_at, tz=UTC).strftime(
            "%Y-%m-%d %H:%M UTC"
        )

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rubra Eval — {_html.escape(self.agent_name)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f3f4f6; color: #111; min-height: 100vh; }}
  table tr:hover {{ background: #f9fafb; }}
</style>
</head>
<body>
<div style="background:#dc2626;padding:20px 32px;display:flex;
  align-items:center;gap:16px">
  <div style="font-size:22px;font-weight:800;color:#fff;letter-spacing:-0.5px">
    Rubra</div>
  <div style="color:#fca5a5;font-size:13px">Agentic Evaluation Framework</div>
  <div style="margin-left:auto;color:#fca5a5;font-size:12px">{evaluated_at}</div>
</div>

<div style="max-width:960px;margin:32px auto;padding:0 16px">

  <div style="margin-bottom:24px">
    <h1 style="font-size:20px;font-weight:700">{_html.escape(self.agent_name)}</h1>
    <div style="color:#6b7280;font-size:13px;margin-top:4px">
      Trace <code style="background:#e5e7eb;padding:1px 5px;border-radius:3px">
        {self.trace_id[:16]}…</code>
      &nbsp;·&nbsp; {_html.escape(self.task or "No task description")}
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
    margin-bottom:28px">
    <div style="{_STAT_BOX_STYLE}">
      <div style="font-size:28px;font-weight:700">{self.total_metrics}</div>
      <div style="{_STAT_LABEL_STYLE}">Total Metrics</div>
    </div>
    <div style="{_STAT_BOX_STYLE}">
      <div style="font-size:28px;font-weight:700;color:#22c55e">{self.passed}</div>
      <div style="{_STAT_LABEL_STYLE}">Passed</div>
    </div>
    <div style="{_STAT_BOX_STYLE}">
      <div style="font-size:28px;font-weight:700;color:#ef4444">{self.failed}</div>
      <div style="{_STAT_LABEL_STYLE}">Failed</div>
    </div>
    <div style="{_STAT_BOX_STYLE}">
      <div style="font-size:28px;font-weight:700;color:#3b82f6">
        {_fmt(self.average_score)}</div>
      <div style="{_STAT_LABEL_STYLE}">Avg Score</div>
    </div>
  </div>

  {
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);'
            'gap:12px;margin-bottom:28px">' + composites + "</div>"
            if composites
            else ""
        }

  {cat_sections}

  <div style="text-align:center;color:#9ca3af;font-size:11px;padding:24px 0">
    Generated by <strong style="color:#dc2626">Rubra</strong> v{__version__} ·
    Every aspect, nothing missed.
  </div>
</div>
</body>
</html>"""

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_doc)
            return path
        return html_doc


# ---------------------------------------------------------------------------
# Metric set aliases
# ---------------------------------------------------------------------------

MetricSet = Literal["all", "execution", "tool", "safety", "quality", "goal"]

_VALID_SETS: set[str] = {"all", "execution", "tool", "safety", "quality", "goal"}


# ---------------------------------------------------------------------------
# evaluate() — sync entry point
# ---------------------------------------------------------------------------


def evaluate(
    trace: Trace,
    metrics: MetricSet | list[str] = "all",
    *,
    # Execution metric thresholds (overridable per-call)
    max_steps: int = 10,
    target_ms: float = 5000.0,
    target_tokens: int = 4096,
    budget_usd: float = 0.10,
    min_output_length: int = 10,
    # Goal metrics (LLM-judge) — any litellm-supported model string,
    # e.g. "gpt-4o-mini", "ollama/llama3.2", "anthropic/claude-3-5-haiku-20241022"
    judge_model: str = "gpt-4o-mini",
    # Storage
    persist: bool = True,
) -> EvalReport:
    """
    Run evaluation metrics against a completed trace.

    Args:
        trace:       A Rubra Trace object (from @rubra.agent or built manually).
        metrics:     "all", "execution", "tool", "safety", "quality", "goal",
                     or a list of specific metric names.
        judge_model: Model passed to litellm for goal metrics (LLM-as-judge).
                     Supports any litellm-compatible model, including local
                     models via Ollama (e.g. "ollama/llama3.2") — no API key
                     or cost required.
        persist:     If True, save MetricResult rows to storage.

    Returns:
        EvalReport with all results and composite scores.
    """
    t_start = time.perf_counter()
    results: list[MetricResult] = []

    requested = _resolve_metric_set(metrics)

    if "execution" in requested:
        results.extend(
            run_execution_metrics(
                trace,
                max_steps=max_steps,
                target_ms=target_ms,
                target_tokens=target_tokens,
                budget_usd=budget_usd,
                min_output_length=min_output_length,
            )
        )

    if "tool" in requested:
        # Tool orchestration metrics — implemented in Phase 1 next step
        try:
            from rubra.core.metrics.tool.metrics import run_tool_metrics

            results.extend(run_tool_metrics(trace))
        except ImportError:
            pass

    if "safety" in requested:
        try:
            from rubra.core.metrics.safety.metrics import run_safety_metrics

            results.extend(run_safety_metrics(trace))
        except ImportError:
            pass

    if "quality" in requested:
        try:
            from rubra.core.metrics.quality.metrics import run_quality_metrics

            results.extend(run_quality_metrics(trace))
        except ImportError:
            pass

    if "goal" in requested:
        try:
            from rubra.core.metrics.goal.metrics import run_goal_metrics

            results.extend(run_goal_metrics(trace, model=judge_model))
        except Exception:
            # Goal metrics require litellm + a reachable model (API key or
            # local server). Swallow all failures so deterministic metrics
            # still return.
            pass

    # If specific metric names were listed, filter
    if isinstance(metrics, list):
        results = [r for r in results if r.metric_name in metrics]

    t_end = time.perf_counter()
    evaluation_ms = (t_end - t_start) * 1000

    report = EvalReport(
        trace_id=trace.trace_id,
        agent_name=trace.agent_name,
        task=trace.task,
        results=results,
        evaluation_ms=evaluation_ms,
    )

    _compute_composites(report)

    if persist:
        _persist_results(trace.trace_id, results)

    return report


async def evaluate_async(
    trace: Trace,
    metrics: MetricSet | list[str] = "all",
    **kwargs: Any,
) -> EvalReport:
    """
    Async variant — identical signature to evaluate().
    Deterministic metrics run synchronously; LLM-judge metrics run async.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    # Run the sync version in the executor to avoid blocking the event loop
    return await loop.run_in_executor(None, lambda: evaluate(trace, metrics, **kwargs))


# ---------------------------------------------------------------------------
# Composite scores
# ---------------------------------------------------------------------------


def _compute_composites(report: EvalReport) -> None:
    """
    rubra_score: weighted average across all scored metrics.
    tool_intelligence_score: average of tool-category metrics only.
    agentic_efficiency_score: (goal_completion * task_completion) /
        (steps * cost_normalised).
    """
    scored = [r for r in report.results if r.score is not None]
    if scored:
        report.rubra_score = sum(r.score for r in scored) / len(scored)

    tool_scored = [r for r in scored if r.category == "tool"]
    if tool_scored:
        report.tool_intelligence_score = sum(r.score for r in tool_scored) / len(
            tool_scored
        )

    # AES: completion_rate / max(latency_fraction, cost_fraction, step_fraction)
    completion = report.get("task_completion_rate")
    latency = report.get("latency_score")
    efficiency = report.get("step_efficiency")
    cost = report.get("cost_efficiency")

    if completion and completion.score is not None:
        penalty_scores = [
            r.score
            for r in [latency, efficiency, cost]
            if r is not None and r.score is not None
        ]
        if penalty_scores:
            avg_penalty = sum(penalty_scores) / len(penalty_scores)
            report.agentic_efficiency_score = completion.score * avg_penalty


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_results(trace_id: str, results: list[MetricResult]) -> None:
    try:
        from rubra.core.storage.db import get_storage

        storage = get_storage()
        if storage is None:
            return
        for r in results:
            storage.save_metric_result(
                trace_id=trace_id,
                metric_name=r.metric_name,
                score=r.score,
                passed=r.passed,
                reason=r.reason,
                category=r.category,
                is_deterministic=r.is_deterministic,
                metadata=r.metadata,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Metric set resolution
# ---------------------------------------------------------------------------


def _resolve_metric_set(metrics: MetricSet | list[str]) -> set[str]:
    if metrics == "all":
        return {"execution", "tool", "safety", "quality", "goal"}
    if isinstance(metrics, str):
        if metrics not in _VALID_SETS:
            raise ValueError(
                f"Unknown metric set '{metrics}'. "
                f"Choose from: {', '.join(sorted(_VALID_SETS))}"
            )
        return {metrics}
    # list of specific metric names — run all categories and filter later
    return {"execution", "tool", "safety", "quality", "goal"}
