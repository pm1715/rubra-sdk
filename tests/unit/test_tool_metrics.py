"""
Unit tests for all 11 tool-orchestration metrics — Rubra's flagship category.
Builds synthetic traces directly; no decorators, no storage, no LLMs.

Covers both the original name-only behavior (default, no ground truth beyond
tool names) and the argument-aware upgrades that activate when
`expected_tool_args` is provided.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from rubra.core.tracer.models import (
    Span, SpanStatus, SpanType, Trace, TraceStatus,
    ToolCallData, ToolResponseData,
)
from rubra.core.metrics.tool.metrics import (
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
    run_tool_metrics,
)


def make_trace(
    calls: list[tuple[str, dict, bool]] | None = None,
    *,
    expected_tool_calls: list[str] | None = None,
    expected_tool_args: dict[str, dict] | None = None,
    final_output: str | None = "done",
    skip_response: set[int] | None = None,
    duration_ms_per_call: float = 100.0,
) -> Trace:
    """
    calls: list of (tool_name, arguments, is_error) tuples, in call order.
    skip_response: set of call indices to NOT emit a TOOL_RESPONSE for
                   (simulates an orphaned/dangling call).
    """
    t0 = datetime.now(timezone.utc)
    trace = Trace(
        agent_name="test_agent",
        task="test task",
        expected_tool_calls=expected_tool_calls,
        expected_tool_args=expected_tool_args,
        final_output=final_output,
    )
    for i, (tool_name, args, is_error) in enumerate(calls or []):
        started = t0 + timedelta(milliseconds=i * duration_ms_per_call)
        call = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.TOOL_CALL,
            name=f"tool:{tool_name}",
            started_at=started,
            tool_call_data=ToolCallData(tool_name=tool_name, arguments=args),
        )
        call.finish(status=SpanStatus.ERROR if is_error else SpanStatus.OK)
        trace.spans.append(call)

        if skip_response and i in skip_response:
            continue
        resp = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.TOOL_RESPONSE,
            name=f"tool_response:{tool_name}",
            started_at=started,
            tool_response_data=ToolResponseData(
                tool_name=tool_name,
                output=f"result-{tool_name}-{i}",
                error="boom" if is_error else None,
            ),
        )
        resp.finish()
        trace.spans.append(resp)

    trace.status = TraceStatus.COMPLETED
    return trace


# ---------------------------------------------------------------------------
# tool_selection_precision / recall / f1
# ---------------------------------------------------------------------------


def test_selection_precision_recall_f1_perfect_match():
    t = make_trace(
        calls=[("search", {}, False), ("calculate", {}, False)],
        expected_tool_calls=["search", "calculate"],
    )
    assert tool_selection_precision(t).score == 1.0
    assert tool_selection_recall(t).score == 1.0
    assert tool_selection_f1(t).score == 1.0


def test_selection_precision_penalizes_extra_tool():
    t = make_trace(
        calls=[("search", {}, False), ("weather", {}, False)],
        expected_tool_calls=["search"],
    )
    r = tool_selection_precision(t)
    assert r.score == 0.5  # 1 of 2 called tools was expected


def test_selection_recall_penalizes_missing_tool():
    t = make_trace(
        calls=[("search", {}, False)],
        expected_tool_calls=["search", "calculate"],
    )
    r = tool_selection_recall(t)
    assert r.score == 0.5  # 1 of 2 expected tools was called


def test_selection_metrics_none_without_ground_truth():
    t = make_trace(calls=[("search", {}, False)])
    assert tool_selection_precision(t).score is None
    assert tool_selection_recall(t).score is None
    assert tool_selection_f1(t).score is None


# ---------------------------------------------------------------------------
# tool_call_order_score — name-only (original) and argument-aware (new)
# ---------------------------------------------------------------------------


def test_order_score_perfect_sequence():
    t = make_trace(
        calls=[("search", {}, False), ("calculate", {}, False)],
        expected_tool_calls=["search", "calculate"],
    )
    r = tool_call_order_score(t)
    assert r.score == 1.0
    assert r.metadata["mode"] == "name_only"


def test_order_score_wrong_order_name_only():
    t = make_trace(
        calls=[("calculate", {}, False), ("search", {}, False)],
        expected_tool_calls=["search", "calculate"],
    )
    r = tool_call_order_score(t)
    # LCS of [calculate, search] vs [search, calculate] = 1 (either name alone)
    assert r.score == 0.5


def test_order_score_argument_aware_partial_credit():
    """
    Right tool, wrong argument value: name-only scoring would count this as a
    full match. Argument-aware scoring must give partial credit, not 1.0 —
    this is the exact gap identified vs. DeepEval's weighted LCS.
    """
    t = make_trace(
        calls=[("search", {"query": "wrong query"}, False)],
        expected_tool_calls=["search"],
        expected_tool_args={"search": {"query": "capital of France"}},
    )
    r = tool_call_order_score(t)
    assert r.metadata["mode"] == "weighted"
    assert r.score == 0.0  # single expected arg, wrong value -> 0 overlap


def test_order_score_argument_aware_correct_args_full_credit():
    t = make_trace(
        calls=[("search", {"query": "capital of France"}, False)],
        expected_tool_calls=["search"],
        expected_tool_args={"search": {"query": "capital of France"}},
    )
    r = tool_call_order_score(t)
    assert r.score == 1.0


def test_order_score_no_expected_calls_is_none():
    t = make_trace(calls=[("search", {}, False)])
    assert tool_call_order_score(t).score is None


# ---------------------------------------------------------------------------
# tool_argument_completeness — presence fallback (original) and correctness (new)
# ---------------------------------------------------------------------------


def test_argument_completeness_presence_fallback():
    t = make_trace(calls=[("search", {"query": "x"}, False), ("calc", {}, False)])
    r = tool_argument_completeness(t)
    assert r.metadata["mode"] == "presence"
    assert r.score == 0.5  # 1 of 2 calls had non-empty args


def test_argument_completeness_correctness_mode():
    t = make_trace(
        calls=[("search", {"query": "capital of France", "lang": "en"}, False)],
        expected_tool_args={"search": {"query": "capital of France", "lang": "fr"}},
    )
    r = tool_argument_completeness(t)
    assert r.metadata["mode"] == "correctness"
    assert r.score == 0.5  # 1 of 2 expected keys matched


def test_argument_completeness_excludes_tools_without_ground_truth():
    """A call to a tool with no entry in expected_tool_args must not be
    penalized — it's excluded from scoring, not counted as wrong."""
    t = make_trace(
        calls=[
            ("search", {"query": "capital of France"}, False),
            ("weather", {"city": "anything"}, False),
        ],
        expected_tool_args={"search": {"query": "capital of France"}},
    )
    r = tool_argument_completeness(t)
    assert r.score == 1.0
    assert r.metadata["scored_calls"] == 1


# ---------------------------------------------------------------------------
# tool_trajectory_equivalence
# ---------------------------------------------------------------------------


def test_trajectory_equivalence_identical_path():
    t = make_trace(
        calls=[("search", {}, False), ("calculate", {}, False)],
        expected_tool_calls=["search", "calculate"],
    )
    r = tool_trajectory_equivalence(t)
    assert r.score == 1.0


def test_trajectory_equivalence_no_ground_truth_uses_proxy():
    t = make_trace(calls=[("search", {}, False)], final_output="uses result")
    r = tool_trajectory_equivalence(t)
    assert r.score is not None  # falls back to tool_output_utilization proxy


# ---------------------------------------------------------------------------
# redundant_tool_call_rate
# ---------------------------------------------------------------------------


def test_redundant_call_detected():
    t = make_trace(
        calls=[
            ("search", {"query": "x"}, False),
            ("search", {"query": "x"}, False),  # exact repeat
        ]
    )
    r = redundant_tool_call_rate(t)
    assert r.score == 0.5
    assert r.passed is False


def test_redundant_call_different_args_not_flagged():
    t = make_trace(
        calls=[
            ("search", {"query": "x"}, False),
            ("search", {"query": "y"}, False),  # different args — not redundant
        ]
    )
    r = redundant_tool_call_rate(t)
    assert r.score == 1.0
    assert r.passed is True


# ---------------------------------------------------------------------------
# tool_error_recovery_rate
# ---------------------------------------------------------------------------


def test_error_recovery_agent_continues_after_error():
    t = make_trace(
        calls=[("search", {}, True), ("search", {"query": "retry"}, False)],
        final_output="recovered answer",
    )
    r = tool_error_recovery_rate(t)
    assert r.score == 1.0


def test_error_recovery_agent_stops_after_error():
    t = make_trace(calls=[("search", {}, True)], final_output=None)
    r = tool_error_recovery_rate(t)
    assert r.score == 0.0


def test_error_recovery_none_when_no_errors():
    t = make_trace(calls=[("search", {}, False)])
    assert tool_error_recovery_rate(t).score is None


# ---------------------------------------------------------------------------
# intermediate_step_grounding — token-overlap (upgraded from fixed substring)
# ---------------------------------------------------------------------------


def test_grounding_detects_overlap_anywhere_in_output():
    """
    The old implementation only probed the first 20 characters of the
    previous tool's output. This trace deliberately puts the useful token
    (Tokyo) *after* character 20, which the old substring probe would miss
    but real token-overlap correctly detects.
    """
    t = make_trace(
        calls=[
            ("get_weather", {"city": "Tokyo"}, False),
            ("convert_temp", {"city_from_step1": "Tokyo", "unit": "F"}, False),
        ]
    )
    # Manually set a long prefix before the useful token to prove the old
    # fixed-20-char probe would have failed here.
    t.spans[1].tool_response_data.output = (
        "Weather lookup completed successfully today: Tokyo is 26C, sunny"
    )
    r = intermediate_step_grounding(t)
    assert r.score == 1.0


def test_grounding_no_overlap_scores_zero():
    t = make_trace(
        calls=[
            ("get_weather", {"city": "Tokyo"}, False),
            ("send_email", {"to": "someone@example.com", "body": "hi"}, False),
        ]
    )
    r = intermediate_step_grounding(t)
    assert r.score == 0.0


def test_grounding_none_with_fewer_than_two_calls():
    t = make_trace(calls=[("search", {}, False)])
    assert intermediate_step_grounding(t).score is None


# ---------------------------------------------------------------------------
# tool_response_latency_score
# ---------------------------------------------------------------------------


def test_latency_score_all_fast():
    t = make_trace(calls=[("search", {}, False)], duration_ms_per_call=10.0)
    for s in t.tool_call_spans:
        s.duration_ms = 100.0
    r = tool_response_latency_score(t, max_tool_ms=3000.0)
    assert r.score == 1.0


def test_latency_score_flags_slow_call():
    t = make_trace(calls=[("search", {}, False)])
    t.tool_call_spans[0].duration_ms = 5000.0
    r = tool_response_latency_score(t, max_tool_ms=3000.0)
    assert r.score == 0.0
    assert "search" in r.metadata["slow_tools"][0]


# ---------------------------------------------------------------------------
# tool_chain_validity
# ---------------------------------------------------------------------------


def test_chain_validity_all_paired():
    t = make_trace(calls=[("search", {}, False), ("calculate", {}, False)])
    r = tool_chain_validity(t)
    assert r.score == 1.0


def test_chain_validity_detects_orphaned_call():
    t = make_trace(calls=[("search", {}, False)], skip_response={0})
    r = tool_chain_validity(t)
    assert r.score == 0.0
    assert r.metadata["unpaired_calls"] == 1


# ---------------------------------------------------------------------------
# run_tool_metrics — batch runner sanity check
# ---------------------------------------------------------------------------


def test_run_tool_metrics_returns_all_eleven():
    t = make_trace(
        calls=[("search", {"query": "x"}, False), ("calculate", {}, False)],
        expected_tool_calls=["search", "calculate"],
    )
    results = run_tool_metrics(t)
    assert len(results) == 11
    assert {r.metric_name for r in results} == {
        "tool_selection_precision",
        "tool_selection_recall",
        "tool_selection_f1",
        "tool_call_order_score",
        "tool_trajectory_equivalence",
        "redundant_tool_call_rate",
        "tool_error_recovery_rate",
        "intermediate_step_grounding",
        "tool_argument_completeness",
        "tool_response_latency_score",
        "tool_chain_validity",
    }
