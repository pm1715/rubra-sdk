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
    Score: 0.0-1.0. Reference-free — no ground truth needed.
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

    prompt = f"""You are a strict, adversarial evaluator. Assume the goal was NOT fully achieved unless the response clearly proves otherwise — when uncertain, score lower, not higher.

TASK: {task}

AGENT RESPONSE: {output[:2000]}

Score on this rubric:
- 1.0 = goal completely and correctly accomplished, nothing missing
- 0.75 = accomplished with a minor, non-critical gap
- 0.5 = partially accomplished — a significant part of the goal is missing or wrong
- 0.25 = mostly failed — only a small, tangential part of the goal was addressed
- 0.0 = goal not accomplished, or the response is wrong or off-topic

Examples:
Task: "Book a flight from NYC to LA for tomorrow" / Response: "I found flights from NYC to LA tomorrow, the cheapest is $220 on Delta at 8am." -> {{"score": 0.5, "passed": false, "reason": "Found options but never actually booked — the goal was booking, not searching."}}
Task: "Summarize this article in 3 bullet points" / Response: exactly 3 accurate, well-formed bullet points covering the article's key ideas -> {{"score": 1.0, "passed": true, "reason": "Delivered exactly what was asked, accurately and in the right format."}}

Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence, cite specific evidence>"}}
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
        prompt = f"""You are a strict, adversarial evaluator comparing an answer against a known-correct reference. Assume the response is wrong unless it clearly matches — do not give credit for confident-sounding but unsupported claims.

EXPECTED: {expected}
AGENT RESPONSE: {output[:2000]}

Score 0.0-1.0:
- 1.0 = semantically equivalent to expected, no material difference
- 0.7 = mostly correct, one minor difference
- 0.4 = partially correct — gets the general idea but a key detail is wrong
- 0.0 = incorrect or contradicts expected

Example: Expected: "Paris" / Response: "The capital of France is Paris, a city on the Seine." -> {{"score": 1.0, "passed": true, "reason": "Correct answer; the extra detail doesn't change correctness."}}

Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence, cite the specific match or mismatch>"}}
"""
    else:
        task = trace.task or trace.task_description or ""
        prompt = f"""You are a strict, adversarial fact-checker. Assume claims are unverified unless clearly correct — when uncertain about a factual claim, score it down rather than giving benefit of the doubt.

TASK: {task}
AGENT RESPONSE: {output[:2000]}

Score 0.0-1.0 for factual accuracy:
- 1.0 = every factual claim checks out
- 0.5 = mix of correct and incorrect/unverifiable claims
- 0.0 = the central factual claim is wrong

Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence, name the claim that was checked>"}}
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

    prompt = f"""You are a strict, adversarial evaluator of agent reasoning quality. Do not reward a correct-looking final answer if the path to it was incoherent or unjustified — you are scoring the PROCESS, not just the outcome.

TASK: {trace.task or 'not specified'}
TOOLS USED (in order): {tool_sequence}
FINAL RESPONSE: {output[:2000]}

Assess:
- Did the agent break the problem down logically before acting?
- Did tool usage follow a coherent plan, or look arbitrary/redundant?
- Is the reasoning transparent and traceable in the response, or just an assertion?

Score 0.0-1.0. When the tool sequence looks arbitrary or the response doesn't explain itself, score low even if the final answer happens to be right.

Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence, cite specific evidence from the tool sequence or response>"}}
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

    prompt = f"""You are a strict, adversarial evaluator. Look specifically for the agent answering a DIFFERENT, easier, or narrower question than the one actually asked — this is a common, subtle failure mode. Assume misunderstanding unless the response clearly addresses the full task.

TASK: {task}
AGENT'S RESPONSE: {output[:1500]}

Evaluate:
- Did the agent address the right question, or a related-but-different one?
- Did it solve the stated problem in full, not just part of it?
- Did it miss any explicit constraint or aspect the task named?

Example: Task: "Compare X and Y, then recommend one" / Response: only describes X and Y without a recommendation -> {{"score": 0.4, "passed": false, "reason": "Understood the comparison but skipped the explicitly-requested recommendation."}}

Score 0.0-1.0. Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence, name the specific aspect addressed or missed>"}}
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

    prompt = f"""You are a strict, adversarial fact-checker. Your default assumption is that any specific claim NOT traceable to the tool results below is fabricated — do not give the agent benefit of the doubt for plausible-sounding details it could have inferred rather than retrieved.

TOOL RESULTS:
{tool_outputs}

AGENT'S FINAL RESPONSE:
{output[:2000]}

For every specific, checkable claim (numbers, names, dates, facts) in the response, verify it appears in or is a fair restatement of the tool results. Ignore style/phrasing differences — focus only on whether the substance is grounded.

Score 0.0-1.0 where 1.0 = every claim traces back to the tool results, 0.5 = a mix of grounded and fabricated claims, 0.0 = the central claim is fabricated or contradicts the tool results.

Return JSON: {{"score": <float>, "passed": <bool>, "reason": "<one sentence>", "hallucinated_claims": ["<specific claim not supported by tool results>", "..."]}}
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
