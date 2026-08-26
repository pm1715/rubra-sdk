"""
11 Tool Orchestration metrics. All deterministic; no LLM calls required.

TruLens, RAGAS, and DeepEval each have some tool-call evaluation of their own
(RAGAS's ToolCallAccuracy/F1, DeepEval's ToolCorrectnessMetric, TruLens's
agent evaluators) — Rubra's differentiation is breadth (11 distinct signals
vs. 2-4) and two checks with no found equivalent elsewhere
(redundant_tool_call_rate, tool_error_recovery_rate), not exclusivity.
"""
from __future__ import annotations

from rubra.core.metrics.execution.metrics import MetricResult
from rubra.core.tracer.models import SpanStatus, SpanType, Trace


# ---------------------------------------------------------------------------
# 1. tool_selection_precision
# ---------------------------------------------------------------------------


def tool_selection_precision(trace: Trace) -> MetricResult:
    """
    Requires expected_tool_calls ground truth.
    Precision = TP / (TP + FP): fraction of called tools that were expected.
    """
    expected = trace.expected_tool_calls
    if not expected:
        return MetricResult(
            metric_name="tool_selection_precision",
            score=None,
            category="tool",
            reason="No expected_tool_calls ground truth provided.",
        )

    called = trace.unique_tools_used
    if not called:
        return MetricResult(
            metric_name="tool_selection_precision",
            score=0.0,
            passed=False,
            category="tool",
            reason="No tools were called.",
        )

    tp = sum(1 for t in called if t in expected)
    precision = tp / len(called)
    return MetricResult(
        metric_name="tool_selection_precision",
        score=precision,
        passed=precision >= 0.8,
        category="tool",
        reason=f"{tp}/{len(called)} called tools were in expected set.",
        metadata={"called": called, "expected": expected},
    )


# ---------------------------------------------------------------------------
# 2. tool_selection_recall
# ---------------------------------------------------------------------------


def tool_selection_recall(trace: Trace) -> MetricResult:
    """
    Recall = TP / (TP + FN): fraction of expected tools that were called.
    """
    expected = trace.expected_tool_calls
    if not expected:
        return MetricResult(
            metric_name="tool_selection_recall",
            score=None,
            category="tool",
            reason="No expected_tool_calls ground truth provided.",
        )

    called = trace.unique_tools_used
    tp = sum(1 for t in expected if t in called)
    recall = tp / len(expected)
    return MetricResult(
        metric_name="tool_selection_recall",
        score=recall,
        passed=recall >= 0.8,
        category="tool",
        reason=f"{tp}/{len(expected)} expected tools were called.",
        metadata={"called": called, "expected": expected},
    )


# ---------------------------------------------------------------------------
# 3. tool_selection_f1
# ---------------------------------------------------------------------------


def tool_selection_f1(trace: Trace) -> MetricResult:
    """
    F1 = 2 * precision * recall / (precision + recall).
    Handles non-deterministic trajectories — same goal, different valid paths.
    """
    precision_r = tool_selection_precision(trace)
    recall_r = tool_selection_recall(trace)

    if precision_r.score is None or recall_r.score is None:
        return MetricResult(
            metric_name="tool_selection_f1",
            score=None,
            category="tool",
            reason="Cannot compute F1 without ground truth expected_tool_calls.",
        )

    p = precision_r.score
    r = recall_r.score
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)

    return MetricResult(
        metric_name="tool_selection_f1",
        score=f1,
        passed=f1 >= 0.7,
        category="tool",
        reason=f"Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}.",
        metadata={"precision": p, "recall": r},
    )


# ---------------------------------------------------------------------------
# 4. tool_call_order_score
# ---------------------------------------------------------------------------


def tool_call_order_score(trace: Trace) -> MetricResult:
    """
    Compares actual tool call order to expected order (when provided), via
    longest common subsequence. When `expected_tool_args` ground truth is also
    available, uses a *weighted* LCS — a matched call contributes its argument
    correctness (0.0-1.0) rather than a flat 1.0, so a right-tool-wrong-args
    call scores partial credit instead of counting as a full match.
    """
    expected = trace.expected_tool_calls
    if not expected:
        return MetricResult(
            metric_name="tool_call_order_score",
            score=None,
            category="tool",
            reason="No expected_tool_calls provided.",
        )

    actual_spans = [s for s in trace.tool_call_spans if s.tool_call_data]
    actual_sequence = [s.tool_call_data.tool_name for s in actual_spans]

    if not actual_sequence:
        return MetricResult(
            metric_name="tool_call_order_score",
            score=0.0,
            passed=False,
            category="tool",
            reason="No tool calls made.",
        )

    expected_args = trace.expected_tool_args
    if expected_args:
        actual_args = [s.tool_call_data.arguments for s in actual_spans]
        weighted_len = _weighted_lcs(actual_sequence, expected, actual_args, expected_args)
        score = weighted_len / max(len(actual_sequence), len(expected))
        return MetricResult(
            metric_name="tool_call_order_score",
            score=score,
            passed=score >= 0.7,
            category="tool",
            reason=(
                f"Weighted LCS {weighted_len:.2f} (argument-aware) vs sequence lengths "
                f"actual={len(actual_sequence)}, expected={len(expected)}."
            ),
            metadata={"actual_sequence": actual_sequence, "expected": expected, "mode": "weighted"},
        )

    lcs_len = _lcs_length(actual_sequence, expected)
    score = lcs_len / max(len(actual_sequence), len(expected))
    return MetricResult(
        metric_name="tool_call_order_score",
        score=score,
        passed=score >= 0.7,
        category="tool",
        reason=(
            f"LCS length {lcs_len} vs sequence lengths "
            f"actual={len(actual_sequence)}, expected={len(expected)} "
            "(name-only — pass expected_tool_args for argument-aware scoring)."
        ),
        metadata={"actual_sequence": actual_sequence, "expected": expected, "mode": "name_only"},
    )


def _weighted_lcs(
    actual: list[str],
    expected: list[str],
    actual_args: list[dict],
    expected_args: dict[str, dict],
) -> float:
    """
    LCS where a match's weight is its argument-overlap score (1.0 if the tool
    has no known expected args) instead of a flat 1.0 — rewards right-tool
    right-args over right-tool wrong-args, without zeroing the whole sequence
    the way a hard gate would.
    """
    m, n = len(actual), len(expected)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if actual[i - 1] == expected[j - 1]:
                exp = expected_args.get(expected[j - 1])
                weight = _arg_overlap_score(actual_args[i - 1], exp) if exp else 1.0
                dp[i][j] = dp[i - 1][j - 1] + weight
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


_GROUNDING_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "are",
    "were", "has", "have", "had", "not", "you", "your", "true", "false", "null",
}


def _meaningful_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens of length >= 3, minus common stopwords."""
    import re

    tokens = re.findall(r"[A-Za-z0-9_.-]{3,}", text)
    return {t.lower() for t in tokens if t.lower() not in _GROUNDING_STOPWORDS}


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ---------------------------------------------------------------------------
# 5. tool_trajectory_equivalence
# ---------------------------------------------------------------------------


def tool_trajectory_equivalence(trace: Trace) -> MetricResult:
    """
    Evaluates whether different tool call sequences achieve equivalent outcomes.
    Checks: (a) same final output exists, (b) tool calls covered the same topic areas.
    A high score means the agent found a valid alternative path to the right answer.

    This handles the non-deterministic trajectory problem — agents validly
    take different routes to the same goal.
    """
    if not trace.final_output:
        return MetricResult(
            metric_name="tool_trajectory_equivalence",
            score=None,
            category="tool",
            reason="No final output to assess trajectory equivalence.",
        )

    expected_tools = trace.expected_tool_calls
    actual_tools = trace.unique_tools_used

    if not expected_tools:
        # Without ground truth, can only assess internal consistency:
        # did the agent use its tool outputs? (proxy for valid trajectory)
        from rubra.core.metrics.execution.metrics import tool_output_utilization
        util = tool_output_utilization(trace)
        return MetricResult(
            metric_name="tool_trajectory_equivalence",
            score=util.score,
            category="tool",
            reason=(
                "No expected trajectory provided; scoring by tool output utilization "
                f"as a proxy for trajectory validity. Score: {util.score}."
            ),
        )

    # With ground truth: LCS-based similarity between trajectories
    from rubra.core.metrics.tool.metrics import _lcs_length
    actual_seq = [s.tool_call_data.tool_name for s in trace.tool_call_spans if s.tool_call_data]
    lcs = _lcs_length(actual_seq, expected_tools)
    union = len(set(actual_seq) | set(expected_tools))
    jaccard = len(set(actual_seq) & set(expected_tools)) / union if union > 0 else 0.0
    order_score = lcs / max(len(actual_seq), len(expected_tools)) if actual_seq and expected_tools else 0.0
    score = (jaccard + order_score) / 2

    return MetricResult(
        metric_name="tool_trajectory_equivalence",
        score=score,
        passed=score >= 0.6,
        category="tool",
        reason=(
            f"Jaccard similarity={jaccard:.3f}, order alignment={order_score:.3f}. "
            f"Agent path is {score:.0%} equivalent to reference trajectory."
        ),
        metadata={"actual_tools": actual_tools, "expected_tools": expected_tools},
    )


# ---------------------------------------------------------------------------
# 6. redundant_tool_call_rate
# ---------------------------------------------------------------------------


def redundant_tool_call_rate(trace: Trace) -> MetricResult:
    """
    Detects repeated calls to the same tool with identical arguments.
    These represent wasted computation — no new information gained.
    Score: 1.0 = no redundant calls.
    """
    tool_spans = trace.tool_call_spans
    if not tool_spans:
        return MetricResult(
            metric_name="redundant_tool_call_rate",
            score=None,
            category="tool",
            reason="No tool calls in trace.",
        )

    seen: set[tuple] = set()
    redundant = 0

    for span in tool_spans:
        if not span.tool_call_data:
            continue
        key = (
            span.tool_call_data.tool_name,
            str(sorted(span.tool_call_data.arguments.items())),
        )
        if key in seen:
            redundant += 1
        else:
            seen.add(key)

    score = 1.0 - (redundant / len(tool_spans))
    return MetricResult(
        metric_name="redundant_tool_call_rate",
        score=score,
        passed=redundant == 0,
        category="tool",
        reason=(
            "No redundant tool calls detected."
            if redundant == 0
            else f"{redundant} redundant call(s) — same tool + same args called again."
        ),
        metadata={"redundant_count": redundant},
    )


# ---------------------------------------------------------------------------
# 7. tool_error_recovery_rate
# ---------------------------------------------------------------------------


def tool_error_recovery_rate(trace: Trace) -> MetricResult:
    """
    When a tool call fails, does the agent recover?
    Recovery = the agent continues and produces a final output despite the error.
    Score: 1.0 = recovered from all errors, 0.0 = errored and stopped.
    """
    error_tool_spans = [s for s in trace.tool_call_spans if s.status == SpanStatus.ERROR]

    if not error_tool_spans:
        return MetricResult(
            metric_name="tool_error_recovery_rate",
            score=None,
            category="tool",
            reason="No tool errors to recover from.",
        )

    # Check how many errors the agent recovered from. NOTE: this deliberately
    # checks position within tool_call_spans, not trace.spans — the error's
    # own paired TOOL_RESPONSE span always immediately follows it in
    # trace.spans, which made the old "any span after this index" check
    # almost always true regardless of whether the agent did anything
    # productive afterward. Recovery = another tool call follows, or the
    # trace still reached a final output.
    call_spans = trace.tool_call_spans
    recovered = 0
    for err_span in error_tool_spans:
        err_idx = call_spans.index(err_span) if err_span in call_spans else -1
        has_later_call = err_idx >= 0 and err_idx < len(call_spans) - 1
        if has_later_call or trace.final_output:
            recovered += 1

    # Final factor: did the agent produce output at all?
    produced_output = bool(trace.final_output)
    score = (recovered / len(error_tool_spans)) * (1.0 if produced_output else 0.5)

    return MetricResult(
        metric_name="tool_error_recovery_rate",
        score=score,
        passed=score >= 0.8,
        category="tool",
        reason=(
            f"Agent recovered from {recovered}/{len(error_tool_spans)} tool error(s). "
            f"Final output: {'present' if produced_output else 'absent'}."
        ),
        metadata={"errors": len(error_tool_spans), "recovered": recovered},
    )


# ---------------------------------------------------------------------------
# 8. intermediate_step_grounding
# ---------------------------------------------------------------------------


def intermediate_step_grounding(trace: Trace) -> MetricResult:
    """
    Measures whether each tool call builds on the previous step's output.
    A well-grounded agent uses retrieved information as the basis for next actions,
    not arbitrary tool calls.

    Proxy: check if tool call arguments contain substrings from the previous
    tool response output — i.e., the agent carried information forward.
    """
    tool_spans = trace.tool_call_spans
    response_spans = [s for s in trace.spans if s.span_type == SpanType.TOOL_RESPONSE]

    if len(tool_spans) < 2:
        return MetricResult(
            metric_name="intermediate_step_grounding",
            score=None,
            category="tool",
            reason="Need at least 2 tool calls to assess intermediate grounding.",
        )

    grounded_transitions = 0
    checked_transitions = 0

    for i in range(1, len(tool_spans)):
        prev_call = tool_spans[i - 1]
        curr_call = tool_spans[i]

        # Find the response for the previous call
        prev_response = next(
            (
                r for r in response_spans
                if r.tool_response_data
                and r.tool_response_data.tool_name == (
                    prev_call.tool_call_data.tool_name if prev_call.tool_call_data else ""
                )
            ),
            None,
        )

        if prev_response is None or not prev_response.tool_response_data:
            continue

        checked_transitions += 1
        prev_output = str(prev_response.tool_response_data.output or "").strip()
        curr_args = str(curr_call.tool_call_data.arguments if curr_call.tool_call_data else "").strip()

        # Token-overlap check: does ANY meaningful token from the previous
        # tool's output reappear in the next call's arguments? A fixed
        # substring probe (e.g. just the first N characters) misses grounding
        # evidence that isn't at the very start of the output — this checks
        # the whole thing.
        prev_tokens = _meaningful_tokens(prev_output)
        curr_tokens = _meaningful_tokens(curr_args)
        if prev_tokens and (prev_tokens & curr_tokens):
            grounded_transitions += 1

    if checked_transitions == 0:
        return MetricResult(
            metric_name="intermediate_step_grounding",
            score=None,
            category="tool",
            reason="Could not verify grounding — no matching tool response/call pairs.",
        )

    score = grounded_transitions / checked_transitions
    return MetricResult(
        metric_name="intermediate_step_grounding",
        score=score,
        passed=score >= 0.5,
        category="tool",
        reason=(
            f"{grounded_transitions}/{checked_transitions} step transitions show "
            "evidence of using previous tool output."
        ),
        metadata={"grounded": grounded_transitions, "checked": checked_transitions},
    )


# ---------------------------------------------------------------------------
# 9. tool_argument_completeness
# ---------------------------------------------------------------------------


def tool_argument_completeness(trace: Trace) -> MetricResult:
    """
    When `expected_tool_args` ground truth is provided, scores actual argument
    *correctness* per call (fraction of expected keys whose value matches).
    Without ground truth, falls back to a non-empty-values presence check —
    a proxy for parameter hallucination, not correctness.
    """
    tool_spans = trace.tool_call_spans
    if not tool_spans:
        return MetricResult(
            metric_name="tool_argument_completeness",
            score=None,
            category="tool",
            reason="No tool calls in trace.",
        )

    expected_args = trace.expected_tool_args
    if expected_args:
        scored_calls = 0
        total_score = 0.0
        for span in tool_spans:
            if not span.tool_call_data:
                continue
            exp = expected_args.get(span.tool_call_data.tool_name)
            if exp is None:
                continue  # no ground truth for this tool — excluded, not penalized
            scored_calls += 1
            total_score += _arg_overlap_score(span.tool_call_data.arguments, exp)

        if scored_calls == 0:
            return MetricResult(
                metric_name="tool_argument_completeness",
                score=None,
                category="tool",
                reason="expected_tool_args provided but no calls matched a tool with ground truth.",
            )

        score = total_score / scored_calls
        return MetricResult(
            metric_name="tool_argument_completeness",
            score=score,
            passed=score >= 0.8,
            category="tool",
            reason=(
                f"Argument correctness averaged {score:.0%} across {scored_calls} "
                "call(s) with known-expected arguments."
            ),
            metadata={"scored_calls": scored_calls, "mode": "correctness"},
        )

    # Fallback: no ground truth — presence-only proxy (original behavior).
    complete_calls = 0
    for span in tool_spans:
        if not span.tool_call_data:
            continue
        args = span.tool_call_data.arguments
        if not args:
            continue
        all_complete = all(
            v is not None and str(v).strip() != ""
            for v in args.values()
        )
        if all_complete:
            complete_calls += 1

    score = complete_calls / len(tool_spans)
    return MetricResult(
        metric_name="tool_argument_completeness",
        score=score,
        passed=score >= 0.9,
        category="tool",
        reason=(
            f"{complete_calls}/{len(tool_spans)} tool calls had complete, non-empty arguments "
            "(presence-only — pass expected_tool_args for correctness scoring)."
        ),
        metadata={"complete_calls": complete_calls, "total": len(tool_spans), "mode": "presence"},
    )


def _arg_overlap_score(actual: dict, expected: dict) -> float:
    """Fraction of expected key/value pairs the actual arguments got right."""
    if not expected:
        return 1.0
    matched = sum(
        1 for k, v in expected.items()
        if k in actual and str(actual[k]) == str(v)
    )
    return matched / len(expected)


# ---------------------------------------------------------------------------
# 10. tool_response_latency_score
# ---------------------------------------------------------------------------


def tool_response_latency_score(trace: Trace, max_tool_ms: float = 3000.0) -> MetricResult:
    """
    Checks whether individual tool calls completed within an acceptable latency.
    Flags tools that are pathologically slow — potential bottlenecks.
    Score: fraction of tool calls completing within max_tool_ms.
    """
    tool_spans = trace.tool_call_spans
    if not tool_spans:
        return MetricResult(
            metric_name="tool_response_latency_score",
            score=None,
            category="tool",
            reason="No tool calls in trace.",
        )

    fast_calls = sum(
        1 for s in tool_spans
        if s.duration_ms is not None and s.duration_ms <= max_tool_ms
    )
    measured = sum(1 for s in tool_spans if s.duration_ms is not None)

    if measured == 0:
        return MetricResult(
            metric_name="tool_response_latency_score",
            score=None,
            category="tool",
            reason="No tool call durations recorded.",
        )

    score = fast_calls / measured
    slow = [
        f"{s.name}({s.duration_ms:.0f}ms)"
        for s in tool_spans
        if s.duration_ms is not None and s.duration_ms > max_tool_ms
    ]

    return MetricResult(
        metric_name="tool_response_latency_score",
        score=score,
        passed=score == 1.0,
        category="tool",
        reason=(
            f"{fast_calls}/{measured} tool calls completed in <{max_tool_ms:.0f}ms."
            + (f" Slow: {', '.join(slow)}" if slow else "")
        ),
        metadata={"slow_tools": slow, "threshold_ms": max_tool_ms},
    )


# ---------------------------------------------------------------------------
# 11. tool_chain_validity
# ---------------------------------------------------------------------------


def tool_chain_validity(trace: Trace) -> MetricResult:
    """
    Checks structural validity of the tool call chain:
    - Every TOOL_CALL has a corresponding TOOL_RESPONSE
    - No TOOL_RESPONSE without a preceding TOOL_CALL
    - No orphaned tool calls (response missing)

    Score: 1.0 = perfectly paired. <1.0 = structural issues.
    """
    call_names = [
        s.tool_call_data.tool_name
        for s in trace.tool_call_spans
        if s.tool_call_data
    ]
    response_names = [
        s.tool_response_data.tool_name
        for s in trace.spans
        if s.span_type == SpanType.TOOL_RESPONSE and s.tool_response_data
    ]

    if not call_names and not response_names:
        return MetricResult(
            metric_name="tool_chain_validity",
            score=None,
            category="tool",
            reason="No tool calls or responses in trace.",
        )

    # Count matched pairs
    calls_copy = list(call_names)
    responses_copy = list(response_names)
    paired = 0

    for r in responses_copy:
        if r in calls_copy:
            calls_copy.remove(r)
            paired += 1

    total = max(len(call_names), len(response_names))
    score = paired / total if total > 0 else 1.0
    unpaired_calls = len(calls_copy)
    unpaired_responses = len([r for r in responses_copy if r not in [c for c in call_names]])

    return MetricResult(
        metric_name="tool_chain_validity",
        score=score,
        passed=score >= 0.9,
        category="tool",
        reason=(
            f"{paired} matched call/response pairs. "
            f"Unpaired calls: {unpaired_calls}, unpaired responses: {unpaired_responses}."
        ),
        metadata={
            "paired": paired,
            "unpaired_calls": unpaired_calls,
        },
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


ALL_TOOL_METRICS = [
    tool_selection_precision,
    tool_selection_recall,
    tool_selection_f1,
    tool_call_order_score,
    tool_trajectory_equivalence,
    redundant_tool_call_rate,
    tool_error_recovery_rate,
    intermediate_step_grounding,
    tool_argument_completeness,
    tool_response_latency_score,
    tool_chain_validity,
]


def run_tool_metrics(
    trace: Trace,
    *,
    max_tool_ms: float = 3000.0,
) -> list[MetricResult]:
    """Run all 11 tool orchestration metrics."""
    return [
        tool_selection_precision(trace),
        tool_selection_recall(trace),
        tool_selection_f1(trace),
        tool_call_order_score(trace),
        tool_trajectory_equivalence(trace),
        redundant_tool_call_rate(trace),
        tool_error_recovery_rate(trace),
        intermediate_step_grounding(trace),
        tool_argument_completeness(trace),
        tool_response_latency_score(trace, max_tool_ms=max_tool_ms),
        tool_chain_validity(trace),
    ]
