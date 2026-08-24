"""
Full example: rubra.patch() + @rubra.agent + @rubra.tool → evaluate()

Demonstrates automatic LLM span capture via rubra.patch(client)
so your existing OpenAI code needs zero changes after the one-liner.
"""
import sys
sys.path.insert(0, ".")  # run from project root

import openai
import rubra
from rubra.core.storage.db import init_storage

# ── 0. Init storage and patch OpenAI ─────────────────────────────────────────
storage = init_storage("sqlite:///:memory:")
client = rubra.patch(openai.OpenAI())   # <── the one line that enables LLM tracing


# ── 1. Instrument tools ───────────────────────────────────────────────────────

@rubra.tool
def search_web(query: str) -> str:
    """Your real search implementation here."""
    return f"Search result for: {query} — Paris is the capital of France."


# ── 2. Instrument agent ───────────────────────────────────────────────────────

@rubra.agent(
    task="Answer capital city questions using web search.",
    expected_tool_calls=["search_web"],
)
def capital_agent(question: str) -> str:
    context = search_web(question)

    # This call is automatically traced — no decorator needed
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer concisely using the context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


# ── 3. Run ────────────────────────────────────────────────────────────────────

answer = capital_agent("What is the capital of France?")
print(f"Answer: {answer}\n")

# ── 4. Evaluate ───────────────────────────────────────────────────────────────

trace = storage.list_traces()[0]
print(f"Spans captured: {len(trace.spans)}")
print(f"  Tool calls: {trace.total_tool_calls}")
print(f"  LLM calls:  {trace.total_llm_calls}")
print(f"  Tokens:     {trace.token_usage.total_tokens}")
print(f"  Cost:       ${trace.token_usage.estimated_cost_usd:.4f}")
print()

# Evaluate all deterministic metrics (zero LLM cost)
report = rubra.evaluate(trace, metrics="execution", persist=False)
print(report.summary())
print()

# Add goal metrics (requires OPENAI_API_KEY + rubra[judge])
# report = rubra.evaluate(trace, metrics="all")
