"""
5 Goal & Intent metrics.
Require rubra[judge] (litellm). All raise ImportError with a clear message when
litellm is not installed — so the deterministic-only install stays clean.
"""
from __future__ import annotations

import json
from typing import Any

from rubra.core.metrics.execution.metrics import MetricResult
from rubra.core.tracer.models import Trace


def _require_litellm() -> Any:
    try:
        import litellm
        return litellm
    except ImportError:
        raise ImportError(
            "Goal metrics require LLM-as-judge. "
            "Install with: pip install rubra[judge]"
        ) from None


def _judge(
    prompt: str,
    model: str = "gpt-4o-mini",
    *,
    system: str = "You are an impartial evaluator. Respond ONLY with a JSON object.",
) -> dict[str, Any]:
    """Call the LLM judge and parse its JSON response."""
    litellm = _require_litellm()
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


# ---------------------------------------------------------------------------
# 1. goal_completion
# ---------------------------------------------------------------------------


def goal_completion(trace: Trace, model: str = "gpt-4o-mini") -> MetricResult:
    """
    LLM judge: did the agent fully achieve the stated task/goal?
    Score: 0.0–1.0. Reference-free — no ground truth needed.
    This is the metric that RAGAS AgentGoalAccuracy gets wrong.
    """
    task = trace.task or trace.task_description
    output = trace.final_output

    if not task:
        return MetricResult(
            metric_name="goal_completion",
            score=None,
            category="goal",
            is_deterministic=False,
            reason="No task specified — cannot evaluate goal completion.",
        )
    if not output:
        return MetricResult(
            metric_name="goal_completion",
            score=0.0,
            passed=False,
            category="goal",
            is_deterministic=False,
            reason="No final output produced by the agent.",
        )

    prompt = f"""
Evaluate whether the agent's response fully achieves the stated goal.

TASK: {task}

AGENT RESPONSE: {output[:2000]}

Rate on a scale of 0.0 to 1.0:
- 1.0 = task completely and correctly accomplished
- 0.7 = mostly accomplished, minor gaps
- 0.4 = partially accomplished
- 0.0 = not accomplished or completely wrong

Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence>"}}
"""
    result = _judge(prompt, model=model)
    score = float(result.get("score", 0.0))
    return MetricResult(
        metric_name="goal_completion",
        score=score,
        passed=result.get("passed", score >= 0.7),
        category="goal",
        is_deterministic=False,
        reason=result.get("reason", ""),
        metadata={"model": model},
    )


# ---------------------------------------------------------------------------
# 2. answer_correctness
# ---------------------------------------------------------------------------


def answer_correctness(trace: Trace, model: str = "gpt-4o-mini") -> MetricResult:
    """
    LLM judge: is the agent's answer factually correct?
    Requires expected_output ground truth for reference-based scoring.
    Falls back to reference-free if no expected_output provided.
    """
    output = trace.final_output
    expected = trace.expected_output

    if not output:
        return MetricResult(
            metric_name="answer_correctness",
            score=0.0,
            passed=False,
            category="goal",
            is_deterministic=False,
            reason="No output to evaluate.",
        )

    if expected:
        prompt = f"""
Compare the agent's response to the expected correct answer.

EXPECTED: {expected}
AGENT RESPONSE: {output[:2000]}

Score 0.0–1.0:
- 1.0 = semantically equivalent to expected
- 0.7 = mostly correct, minor difference
- 0.4 = partially correct
- 0.0 = incorrect or contradicts expected

Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence>"}}
"""
    else:
        task = trace.task or trace.task_description or ""
        prompt = f"""
Evaluate if the agent's response is factually accurate for the given task.

TASK: {task}
AGENT RESPONSE: {output[:2000]}

Score 0.0–1.0 for factual accuracy. Be strict.
Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence>"}}
"""

    result = _judge(prompt, model=model)
    score = float(result.get("score", 0.0))
    return MetricResult(
        metric_name="answer_correctness",
        score=score,
        passed=result.get("passed", score >= 0.7),
        category="goal",
        is_deterministic=False,
        reason=result.get("reason", ""),
        metadata={"model": model, "reference_based": expected is not None},
    )


# ---------------------------------------------------------------------------
# 3. reasoning_quality
# ---------------------------------------------------------------------------


def reasoning_quality(trace: Trace, model: str = "gpt-4o-mini") -> MetricResult:
    """
    LLM judge: did the agent reason through the problem systematically?
    Evaluates the quality of the agent's chain-of-thought, not just the final answer.
    """
    output = trace.final_output
    tool_sequence = [
        s.tool_call_data.tool_name
        for s in trace.tool_call_spans
        if s.tool_call_data
    ]

    if not output:
        return MetricResult(
            metric_name="reasoning_quality",
            score=None,
            category="goal",
            is_deterministic=False,
            reason="No output to evaluate reasoning.",
        )

    prompt = f"""
Evaluate the quality of reasoning shown by this agent.

TASK: {trace.task or 'not specified'}
TOOLS USED (in order): {tool_sequence}
FINAL RESPONSE: {output[:2000]}

Assess reasoning quality:
- Did the agent break down the problem logically?
- Did tool usage follow a coherent plan?
- Is the reasoning transparent in the response?

Score 0.0–1.0. Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence>"}}
"""
    result = _judge(prompt, model=model)
    score = float(result.get("score", 0.0))
    return MetricResult(
        metric_name="reasoning_quality",
        score=score,
        passed=result.get("passed", score >= 0.6),
        category="goal",
        is_deterministic=False,
        reason=result.get("reason", ""),
        metadata={"model": model, "tool_sequence": tool_sequence},
    )


# ---------------------------------------------------------------------------
# 4. task_understanding
# ---------------------------------------------------------------------------


def task_understanding(trace: Trace, model: str = "gpt-4o-mini") -> MetricResult:
    """
    LLM judge: did the agent correctly interpret what was being asked?
    A low score here explains why goal_completion fails even when tools succeed.
    """
    task = trace.task or trace.task_description
    output = trace.final_output

    if not task or not output:
        return MetricResult(
            metric_name="task_understanding",
            score=None,
            category="goal",
            is_deterministic=False,
            reason="Task or output missing.",
        )

    prompt = f"""
Did the agent correctly understand what the task was asking for?

TASK: {task}
AGENT'S RESPONSE: {output[:1500]}

Evaluate:
- Did the agent address the right question?
- Did it solve the stated problem (not a different one)?
- Did it miss important aspects of the request?

Score 0.0–1.0. Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence>"}}
"""
    result = _judge(prompt, model=model)
    score = float(result.get("score", 0.0))
    return MetricResult(
        metric_name="task_understanding",
        score=score,
        passed=result.get("passed", score >= 0.7),
        category="goal",
        is_deterministic=False,
        reason=result.get("reason", ""),
        metadata={"model": model},
    )


# ---------------------------------------------------------------------------
# 5. hallucination_score
# ---------------------------------------------------------------------------


def hallucination_score(trace: Trace, model: str = "gpt-4o-mini") -> MetricResult:
    """
    LLM judge: did the agent fabricate facts not supported by tool results?
    Score: 1.0 = no hallucinations. 0.0 = significant fabrication.
    """
    output = trace.final_output
    tool_outputs = "\n".join(
        f"- {s.tool_response_data.tool_name}: {str(s.tool_response_data.output)[:300]}"
        for s in trace.spans
        if s.tool_response_data and s.tool_response_data.output
    )

    if not output:
        return MetricResult(
            metric_name="hallucination_score",
            score=None,
            category="goal",
            is_deterministic=False,
            reason="No output to check for hallucinations.",
        )

    if not tool_outputs:
        return MetricResult(
            metric_name="hallucination_score",
            score=None,
            category="goal",
            is_deterministic=False,
            reason="No tool results to ground the response — cannot detect hallucinations.",
        )

    prompt = f"""
Determine whether the agent's final response contains hallucinated facts
not supported by the tool retrieval results.

TOOL RESULTS:
{tool_outputs}

AGENT'S FINAL RESPONSE:
{output[:2000]}

Check: are there claims in the response that contradict or are absent from
the tool results? Ignore style differences — focus on factual accuracy.

Score 0.0–1.0 where 1.0 = no hallucinations, 0.0 = major fabrication.
Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence>", "hallucinated_claims": ["..."]}}
"""
    result = _judge(prompt, model=model)
    score = float(result.get("score", 1.0))
    return MetricResult(
        metric_name="hallucination_score",
        score=score,
        passed=result.get("passed", score >= 0.8),
        category="goal",
        is_deterministic=False,
        reason=result.get("reason", ""),
        metadata={
            "model": model,
            "hallucinated_claims": result.get("hallucinated_claims", [])[:3],
        },
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_goal_metrics(
    trace: Trace,
    model: str = "gpt-4o-mini",
) -> list[MetricResult]:
    """Run all 5 goal metrics. Requires rubra[judge]."""
    return [
        goal_completion(trace, model=model),
        answer_correctness(trace, model=model),
        reasoning_quality(trace, model=model),
        task_understanding(trace, model=model),
        hallucination_score(trace, model=model),
    ]
