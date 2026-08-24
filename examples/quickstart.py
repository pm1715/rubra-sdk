"""
Rubra Quickstart — 30-second demo.

Shows the complete loop: instrument -> run -> evaluate -> report.
Run with: python examples/quickstart.py
"""
import sys
sys.path.insert(0, ".")  # run from the rubra/ project root

import rubra
from rubra.core.storage.db import init_storage

# ── 0. Storage (optional — skip for stateless usage) ─────────────────────────
storage = init_storage("sqlite:///:memory:")

# ── 1. Instrument your tools with one decorator ───────────────────────────────

@rubra.tool
def search_web(query: str) -> str:
    """Simulated web search."""
    results = {
        "France capital": "Paris is the capital city of France, population 2.1M.",
        "Germany capital": "Berlin is the capital of Germany, population 3.6M.",
    }
    return results.get(query, "No results found.")


@rubra.tool
def summarize_text(text: str) -> str:
    """Simulated summarizer."""
    return text.split(".")[0] + "."  # return first sentence


# ── 2. Instrument your agent with one decorator ───────────────────────────────

@rubra.agent(
    task="Answer a capital city question by searching and summarizing.",
    expected_tool_calls=["search_web", "summarize_text"],
)
def capital_agent(question: str) -> str:
    country = question.split("of")[-1].strip().rstrip("?")
    raw = search_web(f"{country} capital")
    summary = summarize_text(raw)
    return f"Answer: {summary}"


# ── 3. Run the agent ──────────────────────────────────────────────────────────

output = capital_agent("What is the capital of France?")
print(f"\nAgent output: {output}\n")


# ── 4. Evaluate ───────────────────────────────────────────────────────────────

# Fetch the trace that was just captured
traces = storage.list_traces()
trace = traces[0]

report = rubra.evaluate(trace, metrics="all", persist=False)

# ── 5. Print the report ───────────────────────────────────────────────────────

print("=" * 70)
print(report.summary())
print("=" * 70)
print(f"\n{'METRIC':<40} {'SCORE':>8}  RESULT")
print("-" * 65)

for r in report.results:
    score_str = f"{r.score:.4f}" if r.score is not None else "    N/A"
    result_str = "✓" if r.passed else ("✗" if r.passed is False else "—")
    print(f"{r.metric_name:<40} {score_str:>8}  {result_str}")

print()
if report.rubra_score is not None:
    print(f"Rubra Score:              {report.rubra_score:.4f}")
if report.tool_intelligence_score is not None:
    print(f"Tool Intelligence Score:  {report.tool_intelligence_score:.4f}")
if report.agentic_efficiency_score is not None:
    print(f"Agentic Efficiency Score: {report.agentic_efficiency_score:.4f}")
