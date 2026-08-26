"""
Core trace and span data models.
All storage, metrics, and API layers speak these types — single source of truth.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SpanType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    AGENT_STEP = "agent_step"
    ERROR = "error"


class TraceStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class LLMCallData(BaseModel):
    """Structured payload for LLM_CALL spans."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    temperature: float | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    response: str | None = None
    finish_reason: str | None = None


class ToolCallData(BaseModel):
    """Structured payload for TOOL_CALL spans."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None


class ToolResponseData(BaseModel):
    """Structured payload for TOOL_RESPONSE spans."""

    tool_name: str
    output: Any = None
    error: str | None = None
    was_used_in_next_step: bool | None = None  # set during post-trace analysis


class Span(BaseModel):
    """A single unit of work within an agent trace."""

    span_id: str = Field(default_factory=_uuid)
    trace_id: str
    parent_span_id: str | None = None
    span_type: SpanType
    name: str

    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    duration_ms: float | None = None

    status: SpanStatus = SpanStatus.OK
    error_message: str | None = None
    error_type: str | None = None

    # Typed payload — one of these is populated depending on span_type
    llm_data: LLMCallData | None = None
    tool_call_data: ToolCallData | None = None
    tool_response_data: ToolResponseData | None = None

    # Arbitrary metadata (framework version, tags, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _compute_duration(self) -> "Span":
        if self.ended_at and self.duration_ms is None:
            delta = (self.ended_at - self.started_at).total_seconds()
            self.duration_ms = round(delta * 1000, 3)
        return self

    def finish(
        self,
        *,
        status: SpanStatus = SpanStatus.OK,
        error_message: str | None = None,
        error_type: str | None = None,
    ) -> "Span":
        self.ended_at = _now()
        self.status = status
        self.error_message = error_message
        self.error_type = error_type
        delta = (self.ended_at - self.started_at).total_seconds()
        self.duration_ms = round(delta * 1000, 3)
        return self

    @property
    def is_tool_call(self) -> bool:
        return self.span_type == SpanType.TOOL_CALL

    @property
    def is_llm_call(self) -> bool:
        return self.span_type == SpanType.LLM_CALL

    @property
    def is_error(self) -> bool:
        return self.status == SpanStatus.ERROR or self.span_type == SpanType.ERROR


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class Trace(BaseModel):
    """Root container for a single agent invocation."""

    trace_id: str = Field(default_factory=_uuid)
    agent_name: str
    task: str | None = None
    task_description: str | None = None  # richer version used for auto-rubric

    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    duration_ms: float | None = None

    status: TraceStatus = TraceStatus.RUNNING
    final_output: str | None = None
    error_message: str | None = None

    spans: list[Span] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Ground truth — optional; enables reference-based metrics
    expected_output: str | None = None
    expected_tool_calls: list[str] | None = None  # ordered list of tool names
    # Optional: expected arguments per tool name, e.g. {"search_web": {"query": "..."}}.
    # When provided, tool_call_order_score and tool_argument_completeness score
    # argument correctness, not just tool-name presence. Absent = name-only scoring.
    expected_tool_args: dict[str, dict[str, Any]] | None = None

    def finish(
        self,
        *,
        output: str | None = None,
        status: TraceStatus = TraceStatus.COMPLETED,
        error_message: str | None = None,
    ) -> "Trace":
        self.ended_at = _now()
        self.final_output = output
        self.status = status
        self.error_message = error_message
        delta = (self.ended_at - self.started_at).total_seconds()
        self.duration_ms = round(delta * 1000, 3)
        self._aggregate_token_usage()
        return self

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def _aggregate_token_usage(self) -> None:
        for span in self.spans:
            if span.llm_data:
                self.token_usage.prompt_tokens += span.llm_data.prompt_tokens
                self.token_usage.completion_tokens += span.llm_data.completion_tokens
                self.token_usage.total_tokens += span.llm_data.total_tokens
                self.token_usage.estimated_cost_usd += span.llm_data.cost_usd

    @property
    def tool_call_spans(self) -> list[Span]:
        return [s for s in self.spans if s.span_type == SpanType.TOOL_CALL]

    @property
    def llm_call_spans(self) -> list[Span]:
        return [s for s in self.spans if s.span_type == SpanType.LLM_CALL]

    @property
    def error_spans(self) -> list[Span]:
        return [s for s in self.spans if s.is_error]

    @property
    def unique_tools_used(self) -> list[str]:
        seen: list[str] = []
        for span in self.tool_call_spans:
            if span.tool_call_data and span.tool_call_data.tool_name not in seen:
                seen.append(span.tool_call_data.tool_name)
        return seen

    @property
    def total_steps(self) -> int:
        return len([s for s in self.spans if s.span_type == SpanType.AGENT_STEP])

    @property
    def total_llm_calls(self) -> int:
        return len(self.llm_call_spans)

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_call_spans)

    @property
    def had_errors(self) -> bool:
        return bool(self.error_spans) or self.status == TraceStatus.FAILED
