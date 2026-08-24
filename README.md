<p align="center">
  <img src="assets/wordmark.svg" alt="Rubra" width="380"/>
</p>

**Agentic evaluation framework. Every aspect, nothing missed.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![CI](https://github.com/pm1715/rubra-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/pm1715/rubra-sdk/actions)

Rubra is a **trace-first** agent evaluation framework. Decorate your agent — Rubra automatically captures every tool call, LLM call, token, and cost. Then evaluate with 36 metrics including 11 tool-orchestration metrics no other framework provides.

```python
import rubra

@rubra.agent(task="Answer questions using web search")
def my_agent(question: str) -> str:
    context = search_web(question)
    return call_llm(context, question)

my_agent("What is the capital of France?")

report = rubra.evaluate(rubra.get_last_trace())
print(f"Rubra Score: {report.rubra_score:.3f}")   # 0.923
print(f"Passed:      {report.passed}/{report.total_metrics}")
```

---

## Why Rubra

| Feature | **Rubra** | TruLens | RAGAS | DeepEval |
|---------|:---------:|:-------:|:-----:|:--------:|
| 1-line agent instrumentation | ✅ | ❌ | ❌ | ❌ |
| Tool orchestration metrics (11 unique) | ✅ | ❌ | ❌ | Partial |
| OpenAI + Anthropic auto-trace | ✅ | Manual | Manual | Manual |
| Reference-free goal evaluation | ✅ | ❌ | Partial | Partial |
| LangGraph + LangChain integration | ✅ | ✅ | ❌ | Partial |
| Safety metrics (injection, PII, scope) | ✅ | ❌ | ❌ | ✅ |
| OpenTelemetry export | ✅ | ❌ | ❌ | ❌ |
| Self-hosted REST API + Dashboard | ✅ | ✅ | ❌ | ✅ |
| Pytest plugin | ✅ | ❌ | ❌ | ✅ |
| Zero config (SQLite default) | ✅ | ❌ | ❌ | Partial |

---

## Installation

```bash
pip install rubra                    # core (4 deps, no LLM required)
pip install "rubra[judge]"           # + LLM-judge metrics via litellm
pip install "rubra[openai]"          # + OpenAI SDK interceptor
pip install "rubra[anthropic]"       # + Anthropic Claude interceptor
pip install "rubra[langgraph]"       # + LangGraph node tracing
pip install "rubra[langchain]"       # + LangChain callback handler
pip install "rubra[otel]"            # + OpenTelemetry export
pip install "rubra[all]"             # everything
```

---

## Quickstart

### 1. Basic agent (any framework)

```python
import rubra

@rubra.tool
def search_web(query: str) -> str:
    return my_search_api(query)

@rubra.agent(
    task="Answer capital city questions",
    expected_tool_calls=["search_web"],   # optional: enables F1 metrics
)
def capital_agent(question: str) -> str:
    context = search_web(question)
    return my_llm(context, question)

capital_agent("What is the capital of Japan?")

trace = rubra.get_last_trace()
report = rubra.evaluate(trace, metrics="all")
print(report.summary())
```

### 2. With OpenAI — zero-change LLM tracing

```python
import openai
import rubra

client = rubra.patch(openai.OpenAI())   # one line — that's it

@rubra.agent(task="Capital cities")
def agent(q: str) -> str:
    response = client.chat.completions.create(   # automatically traced
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": q}],
    )
    return response.choices[0].message.content
```

### 3. With Anthropic Claude

```python
import anthropic
import rubra

client = rubra.patch_anthropic(anthropic.Anthropic())

@rubra.agent(task="Summarise documents")
def agent(text: str) -> str:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": text}],
    )
    return response.content[0].text
```

### 4. In pytest — evaluate your agent in CI

```python
# test_agent.py  (no conftest.py needed — plugin registers automatically)

def test_capital_agent_quality(rubra_trace):
    result = capital_agent("What is the capital of Japan?")
    assert result == "Tokyo"

    report = rubra_trace.evaluate(metrics="execution")
    assert report.get("task_completion_rate").passed
    assert report.rubra_score >= 0.70

# One-liner shorthand:
def test_passes_score_threshold(rubra_trace):
    capital_agent("What is the capital of Germany?")
    rubra_trace.assert_score(min_rubra_score=0.70, min_pass_rate=0.80)
```

### 5. LangGraph

```python
from langgraph.graph import StateGraph
from rubra.integrations.langgraph import patch
import rubra

graph = StateGraph(MyState)
graph.add_node("search", search_node)
graph.add_node("answer", answer_node)
app = patch(graph).compile()   # wraps every node as a tool span

@rubra.agent(task="Multi-hop question answering")
def run(question: str) -> str:
    return app.invoke({"question": question})["answer"]
```

### 6. LangChain

```python
from rubra.integrations.langchain import RubraCallbackHandler
import rubra

handler = RubraCallbackHandler()

@rubra.agent(task="Chain execution")
def run(question: str) -> str:
    return my_chain.invoke({"question": question}, config={"callbacks": [handler]})
```

---

## Available Metrics

### Execution (13) — deterministic, no LLM needed
| Metric | Description |
|--------|-------------|
| `task_completion_rate` | Did the agent reach COMPLETED status? |
| `tool_call_success_rate` | Fraction of tool calls with no error |
| `error_rate` | 1 − (error spans / total spans) |
| `step_efficiency` | Penalty for exceeding max_steps |
| `latency_score` | Penalty for slow traces |
| `token_efficiency` | Penalty for excess token usage |
| `cost_efficiency` | Linear decay past budget |
| `tool_diversity` | Unique tools / total calls |
| `retry_rate` | Same-tool-after-error retries |
| `hallucination_free_calls` | Empty-argument proxy |
| `response_completeness` | Final output length check |
| `tool_output_utilization` | Tool output present in final response |
| `execution_time_distribution` | Dominant span fraction check |

### Tool Orchestration (11) — USP, unique to Rubra
| Metric | Description |
|--------|-------------|
| `tool_selection_precision` | TP / (TP + FP) vs expected tool calls |
| `tool_selection_recall` | TP / (TP + FN) |
| `tool_selection_f1` | Harmonic mean of precision + recall |
| `tool_call_order_score` | LCS-based sequence alignment |
| `tool_trajectory_equivalence` | Jaccard + order for non-deterministic paths |
| `redundant_tool_call_rate` | Same tool + args called twice |
| `tool_error_recovery_rate` | Does agent continue after tool failure? |
| `intermediate_step_grounding` | Next-call args reference prior response |
| `tool_argument_completeness` | All argument values non-empty |
| `tool_response_latency_score` | Per-tool latency check |
| `tool_chain_validity` | Every TOOL_CALL has a matching TOOL_RESPONSE |

### Safety (3)
`prompt_injection_resistance` · `scope_creep_score` · `pii_propagation_count`

### Quality (4)
`answer_relevance_proxy` · `output_coherence_score` · `format_compliance_score` · `response_groundedness`

### Goal / LLM-judge (5) — requires `rubra[judge]`
`goal_completion` · `answer_correctness` · `reasoning_quality` · `task_understanding` · `hallucination_score`

The judge model is configurable and works with any [litellm](https://docs.litellm.ai/docs/providers)-supported model — including free local models via Ollama, so you can exercise these metrics with zero API cost:

```python
report = rubra.evaluate(trace, metrics="all", judge_model="ollama/llama3.2")
```

### Composite scores (automatic)
- **`rubra_score`** — weighted average across all scored metrics
- **`tool_intelligence_score`** — average of tool-category metrics
- **`agentic_efficiency_score`** — completion × average efficiency

---

## REST API + Dashboard

See [rubra-server](https://github.com/pm1715/rubra-server) for the self-hosted FastAPI backend and live dashboard.

```bash
git clone https://github.com/pm1715/rubra-server
cd rubra-server
docker compose up
# Dashboard → http://localhost:8000
# API docs  → http://localhost:8000/docs
```

---

## CLI

```bash
rubra traces              # list recent traces
rubra eval                # evaluate latest trace
rubra eval <TRACE_ID>     # evaluate specific trace
rubra report -o out.html  # generate HTML report
```

---

## Architecture

Rubra uses Python `contextvars.ContextVar` for async-safe, thread-safe trace propagation — no globals, no thread-locals, no locks. Each `@rubra.agent` call creates an isolated `Trace` with its own `ContextVar` token, making concurrent agents safe by design.

```
@rubra.agent ──► Trace (ContextVar)
    @rubra.tool ──► TOOL_CALL + TOOL_RESPONSE spans
    rubra.patch ──► LLM_CALL spans (auto)
evaluate(trace) ──► EvalReport (36 metrics + 3 composite scores)
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
