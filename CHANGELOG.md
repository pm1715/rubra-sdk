# Changelog

All notable changes to Rubra are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] — 2026-08-25

### Added
- `evaluate(..., judge_model=...)` — goal metrics now accept any litellm-compatible model string (was hardcoded to `gpt-4o-mini`), enabling free local judge testing via Ollama (`judge_model="ollama/llama3.2"`).
- Author metadata (`[project.authors]`) in `pyproject.toml`, so PyPI shows a maintainer name instead of just the account username.

### Fixed
- `rubra[openai]` extra was missing from `pyproject.toml` even though `rubra.patch()` and `examples/with_openai.py` require the real `openai` package — added it (and included it in `[all]` and `[dev]`).

## [0.1.0] — 2026-08-25

### Added

**Core tracer**
- `@rubra.agent` decorator — wraps any sync/async agent function and captures a full `Trace`
- `@rubra.tool` decorator — captures `TOOL_CALL` + `TOOL_RESPONSE` span pairs
- `TraceContext` and `SpanContext` — context managers for manual trace control
- `ContextVar`-based context propagation (async-safe, no thread-locals)

**Metrics — 36 total across 6 categories**
- 13 execution metrics: task completion, latency, token/cost efficiency, tool diversity, retry rate, etc.
- 11 tool orchestration metrics (USP): trajectory F1, LCS-based order score, grounding, chain validity, etc.
- 3 safety metrics: prompt injection resistance, scope creep detection, PII propagation count
- 4 quality metrics: answer relevance, output coherence, format compliance, response groundedness
- 5 goal metrics (LLM-judge via `rubra[judge]`): goal completion, answer correctness, reasoning quality, task understanding, hallucination score
- 3 composite scores: `rubra_score`, `tool_intelligence_score`, `agentic_efficiency_score`

**Integrations**
- `rubra.patch(openai_client)` — one-line OpenAI SDK interceptor, auto-captures LLM spans + cost
- `rubra.patch_anthropic(anthropic_client)` — same for Anthropic Claude
- `rubra.integrations.langgraph` — `@rubra_node` decorator and `patch(graph)` for LangGraph
- `rubra.integrations.langchain` — `RubraCallbackHandler` for LangChain chains and agents
- `rubra.integrations.otel` — OpenTelemetry export via `enable_otel(endpoint)`

**Storage**
- SQLite auto-init at `.rubra/rubra.db` (zero config)
- PostgreSQL support via `RUBRA_DATABASE_URL` env var
- Upsert-safe `save_trace()` using SELECT + UPDATE/INSERT

**CLI**
- `rubra eval [TRACE_ID]` — evaluate latest or specific trace, print Rich table
- `rubra traces [--limit N] [--agent NAME]` — list recent traces
- `rubra report [TRACE_ID] [-o FILE]` — generate self-contained HTML report

**Pytest plugin**
- `rubra_trace` fixture — captures traces produced during a test
- `rubra_trace.evaluate(metrics="execution")` — run eval on captured trace
- `rubra_trace.assert_score(min_rubra_score=0.7)` — one-line assertion

**Utilities**
- `rubra.get_last_trace()` — retrieve most recently completed trace
- `EvalReport.to_html(path)` — self-contained HTML report with category breakdowns

### Infrastructure
- 114 SDK tests (unit + integration) + 24 server tests — all green
- GitHub Actions CI (Python 3.11 + 3.12, lint + test)
- Apache 2.0 license
