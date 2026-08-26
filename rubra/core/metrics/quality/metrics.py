"""
4 Output Quality metrics.
Deterministic text-analysis — no LLM calls required.
"""

from __future__ import annotations

import re

from rubra.core.metrics.execution.metrics import MetricResult
from rubra.core.tracer.models import Trace

# ---------------------------------------------------------------------------
# 1. answer_relevance_proxy
# ---------------------------------------------------------------------------


def answer_relevance_proxy(trace: Trace) -> MetricResult:
    """
    Checks whether the final output contains keywords from the task description.
    A lightweight, zero-cost proxy for relevance — no LLM needed.
    Score: fraction of task keywords found in the output.
    """
    task = trace.task or trace.task_description or ""
    output = trace.final_output or ""

    if not task or not output:
        return MetricResult(
            metric_name="answer_relevance_proxy",
            score=None,
            category="quality",
            reason="Task or output missing — cannot compute relevance proxy.",
        )

    # Extract meaningful words from task (filter stopwords)
    stopwords = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "of",
        "to",
        "in",
        "for",
        "and",
        "or",
        "what",
        "how",
        "why",
        "when",
        "where",
        "find",
        "get",
        "give",
        "tell",
        "me",
        "i",
        "you",
    }
    task_words = {
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{3,}\b", task)
        if w.lower() not in stopwords
    }

    if not task_words:
        return MetricResult(
            metric_name="answer_relevance_proxy",
            score=None,
            category="quality",
            reason="No meaningful keywords extracted from task.",
        )

    output_lower = output.lower()
    found = sum(1 for w in task_words if w in output_lower)
    score = found / len(task_words)

    return MetricResult(
        metric_name="answer_relevance_proxy",
        score=score,
        passed=score >= 0.5,
        category="quality",
        reason=f"{found}/{len(task_words)} task keywords present in output.",
        metadata={"task_keywords": list(task_words)[:10], "found": found},
    )


# ---------------------------------------------------------------------------
# 2. output_coherence_score
# ---------------------------------------------------------------------------


def output_coherence_score(trace: Trace) -> MetricResult:
    """
    Structural coherence check on the final output:
    - Not just a single word
    - Contains complete sentences (ends with punctuation)
    - Not excessively repetitive (no phrase repeated >3 times)
    Score: 0.0–1.0 based on these heuristics.
    """
    output = (trace.final_output or "").strip()

    if not output:
        return MetricResult(
            metric_name="output_coherence_score",
            score=0.0,
            passed=False,
            category="quality",
            reason="No output produced.",
        )

    score = 0.0
    checks = 0
    reasons: list[str] = []

    # Check 1: minimum length
    checks += 1
    if len(output.split()) >= 5:
        score += 1.0
    else:
        reasons.append("Output too short (< 5 words).")

    # Check 2: ends with sentence-ending punctuation
    checks += 1
    if re.search(r"[.!?]$", output):
        score += 1.0
    else:
        reasons.append("Output does not end with sentence punctuation.")

    # Check 3: no excessive phrase repetition
    checks += 1
    words = output.lower().split()
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if trigrams:
        max_repeat = max(trigrams.count(t) for t in set(trigrams))
        if max_repeat <= 3:
            score += 1.0
        else:
            reasons.append(f"Repetitive phrase detected ({max_repeat}x).")
    else:
        score += 1.0  # too short to check trigrams

    # Check 4: no truncation markers
    checks += 1
    if "[truncated]" not in output and "..." not in output[-10:]:
        score += 1.0
    else:
        reasons.append("Output appears truncated.")

    final_score = score / checks
    return MetricResult(
        metric_name="output_coherence_score",
        score=final_score,
        passed=final_score >= 0.75,
        category="quality",
        reason=(
            "Output passes all coherence checks." if not reasons else " ".join(reasons)
        ),
    )


# ---------------------------------------------------------------------------
# 3. format_compliance_score
# ---------------------------------------------------------------------------


def format_compliance_score(
    trace: Trace,
    expected_format: str | None = None,
) -> MetricResult:
    """
    Checks whether the output matches an expected format hint.
    Supported: 'json', 'markdown', 'list', 'code', or None (free-form, always pass).

    If no expected_format is provided, checks if the output format is self-consistent
    (e.g., if it starts with '{', it should end with '}').
    """
    output = (trace.final_output or "").strip()
    fmt = expected_format or trace.metadata.get("expected_format")

    if not output:
        return MetricResult(
            metric_name="format_compliance_score",
            score=0.0,
            passed=False,
            category="quality",
            reason="No output to check format compliance.",
        )

    if fmt is None:
        # Self-consistency check
        if output.startswith("{") and not output.endswith("}"):
            score, reason = 0.5, "Output looks like JSON but is not closed."
        elif output.startswith("[") and not output.endswith("]"):
            score, reason = 0.5, "Output looks like a JSON array but is not closed."
        else:
            score = 1.0
            reason = "No format constraint — output appears self-consistent."
        return MetricResult(
            metric_name="format_compliance_score",
            score=score,
            passed=score >= 0.9,
            category="quality",
            reason=reason,
        )

    fmt = fmt.lower()
    passed = False
    reason = ""

    if fmt == "json":
        import json

        try:
            json.loads(output)
            passed, reason = True, "Output is valid JSON."
        except json.JSONDecodeError as e:
            reason = f"Output is not valid JSON: {e}."

    elif fmt == "markdown":
        has_heading = bool(re.search(r"^#{1,6}\s", output, re.MULTILINE))
        has_list = bool(re.search(r"^[-*+]\s", output, re.MULTILINE))
        has_bold = bool(re.search(r"\*\*.+?\*\*", output))
        passed = has_heading or has_list or has_bold
        reason = (
            "Contains markdown formatting."
            if passed
            else "No markdown formatting detected."
        )

    elif fmt == "list":
        has_numbered = bool(re.search(r"^\d+[.)]\s", output, re.MULTILINE))
        has_bullet = bool(re.search(r"^[-*•]\s", output, re.MULTILINE))
        passed = has_numbered or has_bullet
        reason = "Output is a list." if passed else "No list formatting detected."

    elif fmt == "code":
        has_fence = "```" in output
        has_indent = bool(re.search(r"^ {4}", output, re.MULTILINE))
        passed = has_fence or has_indent
        reason = "Output contains code block." if passed else "No code block detected."

    else:
        return MetricResult(
            metric_name="format_compliance_score",
            score=None,
            category="quality",
            reason=(
                f"Unknown expected_format '{fmt}'. "
                "Supported: json, markdown, list, code."
            ),
        )

    return MetricResult(
        metric_name="format_compliance_score",
        score=1.0 if passed else 0.0,
        passed=passed,
        category="quality",
        reason=reason,
        metadata={"expected_format": fmt},
    )


# ---------------------------------------------------------------------------
# 4. response_groundedness
# ---------------------------------------------------------------------------


def response_groundedness(trace: Trace) -> MetricResult:
    """
    Checks whether the final output is grounded in tool retrieval results.
    A grounded response uses words/phrases retrieved from tools — not hallucinated.

    Proxy: fraction of sentences in the output that contain at least one word
    from any tool response (bigram overlap).
    """
    output = (trace.final_output or "").strip()
    if not output:
        return MetricResult(
            metric_name="response_groundedness",
            score=None,
            category="quality",
            reason="No final output.",
        )

    tool_text = " ".join(
        str(s.tool_response_data.output or "")
        for s in trace.spans
        if s.tool_response_data and s.tool_response_data.output
    ).lower()

    if not tool_text.strip():
        return MetricResult(
            metric_name="response_groundedness",
            score=None,
            category="quality",
            reason="No tool responses to ground output against.",
        )

    # Extract bigrams from tool text as a reference set
    tool_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", tool_text))

    # Check each output sentence
    sentences = re.split(r"[.!?]+", output)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return MetricResult(
            metric_name="response_groundedness",
            score=None,
            category="quality",
            reason="Could not parse sentences from output.",
        )

    grounded = 0
    for sentence in sentences:
        sent_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower()))
        if sent_words & tool_words:  # any overlap
            grounded += 1

    score = grounded / len(sentences)
    return MetricResult(
        metric_name="response_groundedness",
        score=score,
        passed=score >= 0.5,
        category="quality",
        reason=(
            f"{grounded}/{len(sentences)} output sentences grounded "
            "in tool retrieval results."
        ),
        metadata={"grounded_sentences": grounded, "total_sentences": len(sentences)},
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_quality_metrics(
    trace: Trace,
    *,
    expected_format: str | None = None,
) -> list[MetricResult]:
    return [
        answer_relevance_proxy(trace),
        output_coherence_score(trace),
        format_compliance_score(trace, expected_format=expected_format),
        response_groundedness(trace),
    ]
