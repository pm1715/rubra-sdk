"""
13 deterministic execution metrics.
Pure math on span data — zero LLM calls, zero API cost, sub-millisecond.
All functions accept a Trace and return a MetricResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rubra.core.tracer.models import SpanStatus, SpanType, Trace


# ---------------------------------------------------------------------------
# MetricResult — shared output shape for all metrics
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    metric_name: str
    score: float | None
    passed: bool | None = None
    reason: str = ""
    category: str = "execution"
    is_deterministic: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score_pct(self) -> str:
        if self.score is None:
            return "N/A"
        return f"{self.score:.1%}"

    def __repr__(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.passed is False else "---")
        return f"MetricResult({self.metric_name}={self.score_pct} [{status}])"


# ---------------------------------------------------------------------------
# 1. task_completion_rate
# ---------------------------------------------------------------------------


def task_completion_rate(trace: Trace) -> MetricResult:
    """
    Binary: did the agent reach COMPLETED status without error?
    Score: 1.0 = completed, 0.0 = failed / timed out / still running.
    """
    from rubra.core.tracer.models import TraceStatus

    completed = trace.status == TraceStatus.COMPLETED
    score = 1.0 if completed else 0.0
    return MetricResult(
        metric_name="task_completion_rate",
        score=score,
        passed=completed,
        reason=(
            "Agent reached COMPLETED status."
            if completed
            else f"Agent ended with status={trace.status.value}."
        ),
    )


# ---------------------------------------------------------------------------
# 2. tool_call_success_rate
# ---------------------------------------------------------------------------


def tool_call_success_rate(trace: Trace) -> MetricResult:
    """
    Fraction of tool calls that completed without raising an error.
    Score: successful_calls / total_calls. N/A when no tools called.
    """
    total = trace.total_tool_calls
    if total == 0:
        return MetricResult(
            metric_name="tool_call_success_rate",
            score=None,
            reason="No tool calls in trace.",
        )

    errors = sum(
        1 for s in trace.tool_call_spans if s.status == SpanStatus.ERROR
    )
    successes = total - errors
    score = successes / total
    return MetricResult(
        metric_name="tool_call_success_rate",
        score=score,
        passed=score == 1.0,
        reason=f"{successes}/{total} tool calls succeeded.",
    )


# ---------------------------------------------------------------------------
# 3. error_rate
# ---------------------------------------------------------------------------


def error_rate(trace: Trace) -> MetricResult:
    """
    Fraction of all spans that ended in ERROR status.
    Score: 1.0 = no errors (best), 0.0 = all spans errored.
    """
    total = len(trace.spans)
    if total == 0:
        return MetricResult(
            metric_name="error_rate",
            score=None,
            reason="No spans captured.",
        )

    error_count = len(trace.error_spans)
    score = 1.0 - (error_count / total)
    return MetricResult(
        metric_name="error_rate",
        score=score,
        passed=error_count == 0,
        reason=(
            "Zero errors across all spans."
            if error_count == 0
            else f"{error_count} error span(s) out of {total} total."
        ),
    )


# ---------------------------------------------------------------------------
# 4. step_efficiency
# ---------------------------------------------------------------------------


def step_efficiency(trace: Trace, max_steps: int = 10) -> MetricResult:
    """
    Penalizes unnecessary steps. Score = max(0, 1 - steps / max_steps).
    A perfectly efficient run uses the minimum steps; excess steps reduce score.
    """
    steps = trace.total_steps or trace.total_tool_calls  # fallback to tool calls
    if steps == 0:
        return MetricResult(
            metric_name="step_efficiency",
            score=None,
            reason="No steps recorded.",
        )

    score = max(0.0, 1.0 - (steps / max_steps))
    return MetricResult(
        metric_name="step_efficiency",
        score=score,
        passed=steps <= max_steps,
        reason=f"{steps} steps taken (threshold: {max_steps}).",
        metadata={"steps": steps, "max_steps": max_steps},
    )


# ---------------------------------------------------------------------------
# 5. latency_score
# ---------------------------------------------------------------------------


def latency_score(trace: Trace, target_ms: float = 5000.0) -> MetricResult:
    """
    Penalizes slow traces. Score = max(0, 1 - duration / target_ms).
    A trace completing in ≤ target_ms scores 1.0.
    """
    if trace.duration_ms is None:
        return MetricResult(
            metric_name="latency_score",
            score=None,
            reason="Trace duration not available (still running or not finished).",
        )

    score = max(0.0, 1.0 - (trace.duration_ms / target_ms))
    return MetricResult(
        metric_name="latency_score",
        score=score,
        passed=trace.duration_ms <= target_ms,
        reason=f"Trace completed in {trace.duration_ms:.0f}ms (target: {target_ms:.0f}ms).",
        metadata={"duration_ms": trace.duration_ms, "target_ms": target_ms},
    )


# ---------------------------------------------------------------------------
# 6. token_efficiency
# ---------------------------------------------------------------------------


def token_efficiency(trace: Trace, target_tokens: int = 4096) -> MetricResult:
    """
    Penalizes token-heavy traces. Score = max(0, 1 - total_tokens / target).
    Incentivizes concise prompting and short chains.
    """
    total = trace.token_usage.total_tokens
    if total == 0:
        return MetricResult(
            metric_name="token_efficiency",
            score=None,
            reason="No token usage recorded (LLM calls not instrumented or no LLM calls).",
        )

    score = max(0.0, 1.0 - (total / target_tokens))
    return MetricResult(
        metric_name="token_efficiency",
        score=score,
        passed=total <= target_tokens,
        reason=f"{total} tokens used (target: ≤{target_tokens}).",
        metadata={"total_tokens": total, "target_tokens": target_tokens},
    )


# ---------------------------------------------------------------------------
# 7. cost_efficiency
# ---------------------------------------------------------------------------


def cost_efficiency(trace: Trace, budget_usd: float = 0.10) -> MetricResult:
    """
    Binary grade against a per-trace USD budget.
    Score: 1.0 if cost ≤ budget, linear decay to 0.0 at 2× budget.
    """
    cost = trace.token_usage.estimated_cost_usd
    if cost == 0.0:
        return MetricResult(
            metric_name="cost_efficiency",
            score=None,
            reason="No cost data (LLM calls not instrumented or free model).",
        )

    score = max(0.0, 1.0 - max(0.0, (cost - budget_usd) / budget_usd))
    return MetricResult(
        metric_name="cost_efficiency",
        score=score,
        passed=cost <= budget_usd,
        reason=f"Trace cost ${cost:.4f} (budget: ${budget_usd:.4f}).",
        metadata={"cost_usd": cost, "budget_usd": budget_usd},
    )


# ---------------------------------------------------------------------------
# 8. tool_diversity
# ---------------------------------------------------------------------------


def tool_diversity(trace: Trace) -> MetricResult:
    """
    Ratio of unique tools used to total tool calls.
    Score 1.0 = every call uses a different tool (maximum diversity).
    Score < 1.0 = redundant / repeated tool usage.
    N/A when no tool calls.
    """
    total = trace.total_tool_calls
    if total == 0:
        return MetricResult(
            metric_name="tool_diversity",
            score=None,
            reason="No tool calls in trace.",
        )

    unique = len(trace.unique_tools_used)
    score = unique / total
    return MetricResult(
        metric_name="tool_diversity",
        score=score,
        passed=score >= 0.5,
        reason=f"{unique} unique tools across {total} calls.",
        metadata={"unique_tools": unique, "total_calls": total},
    )


# ---------------------------------------------------------------------------
# 9. retry_rate
# ---------------------------------------------------------------------------


def retry_rate(trace: Trace) -> MetricResult:
    """
    Fraction of tool calls that were retried (same tool called immediately after an error).
    Score: 1.0 = no retries (best). 0.0 = every call was a retry.
    """
    tool_spans = trace.tool_call_spans
    if not tool_spans:
        return MetricResult(
            metric_name="retry_rate",
            score=None,
            reason="No tool calls in trace.",
        )

    retry_count = 0
    for i in range(1, len(tool_spans)):
        prev = tool_spans[i - 1]
        curr = tool_spans[i]
        if (
            prev.status == SpanStatus.ERROR
            and curr.tool_call_data
            and prev.tool_call_data
            and curr.tool_call_data.tool_name == prev.tool_call_data.tool_name
        ):
            retry_count += 1

    score = 1.0 - (retry_count / len(tool_spans))
    return MetricResult(
        metric_name="retry_rate",
        score=score,
        passed=retry_count == 0,
        reason=(
            "No tool retries detected."
            if retry_count == 0
            else f"{retry_count} retry(ies) detected across {len(tool_spans)} tool calls."
        ),
        metadata={"retry_count": retry_count},
    )


# ---------------------------------------------------------------------------
# 10. hallucination_free_calls (tool parameter hallucination proxy)
# ---------------------------------------------------------------------------


def hallucination_free_calls(trace: Trace) -> MetricResult:
    """
    Detects tool calls with empty or null required parameters — a proxy for
    parameter hallucination without LLM verification.
    Score: 1.0 = all calls have non-empty arguments. 0.0 = all calls empty.
    """
    tool_spans = trace.tool_call_spans
    if not tool_spans:
        return MetricResult(
            metric_name="hallucination_free_calls",
            score=None,
            reason="No tool calls in trace.",
        )

    empty_arg_count = 0
    for span in tool_spans:
        if span.tool_call_data and not span.tool_call_data.arguments:
            empty_arg_count += 1

    score = 1.0 - (empty_arg_count / len(tool_spans))
    return MetricResult(
        metric_name="hallucination_free_calls",
        score=score,
        passed=empty_arg_count == 0,
        reason=(
            "All tool calls had non-empty arguments."
            if empty_arg_count == 0
            else f"{empty_arg_count} tool call(s) had empty arguments."
        ),
        metadata={"empty_arg_calls": empty_arg_count},
    )


# ---------------------------------------------------------------------------
# 11. response_completeness
# ---------------------------------------------------------------------------


def response_completeness(trace: Trace, min_length: int = 10) -> MetricResult:
    """
    Checks whether the agent produced a non-trivially short final output.
    Score: 1.0 = output meets minimum length. 0.0 = no output or too short.
    """
    output = trace.final_output or ""
    output_len = len(output.strip())

    if output_len == 0:
        return MetricResult(
            metric_name="response_completeness",
            score=0.0,
            passed=False,
            reason="Agent produced no final output.",
        )

    score = min(1.0, output_len / (min_length * 10))  # full score at 10× min
    passed = output_len >= min_length
    return MetricResult(
        metric_name="response_completeness",
        score=score,
        passed=passed,
        reason=f"Output length: {output_len} chars (minimum: {min_length}).",
        metadata={"output_length": output_len},
    )


# ---------------------------------------------------------------------------
# 12. tool_output_utilization
# ---------------------------------------------------------------------------


def tool_output_utilization(trace: Trace) -> MetricResult:
    """
    Fraction of tool response outputs that appear (substring match) in the
    final agent output or the next LLM call — verifying the agent actually
    used what it retrieved.

    Requires tool_response_data.was_used_in_next_step to be set by the
    instrumentation layer (or falls back to substring matching in final output).
    """
    response_spans = [
        s for s in trace.spans if s.span_type == SpanType.TOOL_RESPONSE
    ]
    if not response_spans:
        return MetricResult(
            metric_name="tool_output_utilization",
            score=None,
            reason="No tool responses to check.",
        )

    final = trace.final_output or ""
    utilized = 0

    for span in response_spans:
        if span.tool_response_data is None:
            continue
        # Check explicit flag first
        if span.tool_response_data.was_used_in_next_step is True:
            utilized += 1
            continue
        # Fallback: check if output text appears in final response
        output_str = str(span.tool_response_data.output or "").strip()
        if output_str and len(output_str) >= 5 and output_str[:30] in final:
            utilized += 1

    score = utilized / len(response_spans)
    return MetricResult(
        metric_name="tool_output_utilization",
        score=score,
        passed=score >= 0.7,
        reason=f"{utilized}/{len(response_spans)} tool outputs used in response.",
        metadata={"utilized": utilized, "total_responses": len(response_spans)},
        category="tool",
    )


# ---------------------------------------------------------------------------
# 13. execution_time_distribution
# ---------------------------------------------------------------------------


def execution_time_distribution(trace: Trace) -> MetricResult:
    """
    Checks whether any single span dominates trace duration (>80% of total).
    A balanced distribution (score near 1.0) suggests no pathological bottlenecks.
    Score: 1 - max_span_fraction (lower dominant span = higher score).
    """
    if not trace.spans or trace.duration_ms is None or trace.duration_ms == 0:
        return MetricResult(
            metric_name="execution_time_distribution",
            score=None,
            reason="Insufficient timing data.",
        )

    span_durations = [s.duration_ms or 0.0 for s in trace.spans]
    if not any(d > 0 for d in span_durations):
        return MetricResult(
            metric_name="execution_time_distribution",
            score=None,
            reason="No span durations recorded.",
        )

    max_duration = max(span_durations)
    max_fraction = max_duration / trace.duration_ms
    score = max(0.0, 1.0 - max_fraction)

    dominant_span = next(
        (s.name for s in trace.spans if (s.duration_ms or 0) == max_duration), "unknown"
    )
    return MetricResult(
        metric_name="execution_time_distribution",
        score=score,
        passed=max_fraction <= 0.8,
        reason=(
            f"Dominant span '{dominant_span}' took {max_fraction:.0%} of total trace time."
        ),
        metadata={"dominant_span": dominant_span, "dominant_fraction": max_fraction},
    )


# ---------------------------------------------------------------------------
# Batch runner — run all 13 at once
# ---------------------------------------------------------------------------


ALL_EXECUTION_METRICS = [
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
]


def run_execution_metrics(
    trace: Trace,
    *,
    max_steps: int = 10,
    target_ms: float = 5000.0,
    target_tokens: int = 4096,
    budget_usd: float = 0.10,
    min_output_length: int = 10,
) -> list[MetricResult]:
    """Run all 13 execution metrics and return results list."""
    return [
        task_completion_rate(trace),
        tool_call_success_rate(trace),
        error_rate(trace),
        step_efficiency(trace, max_steps=max_steps),
        latency_score(trace, target_ms=target_ms),
        token_efficiency(trace, target_tokens=target_tokens),
        cost_efficiency(trace, budget_usd=budget_usd),
        tool_diversity(trace),
        retry_rate(trace),
        hallucination_free_calls(trace),
        response_completeness(trace, min_length=min_output_length),
        tool_output_utilization(trace),
        execution_time_distribution(trace),
    ]
