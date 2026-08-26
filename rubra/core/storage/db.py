"""
SQLite (default) / PostgreSQL storage layer.
Traces and spans serialized to JSON columns for schema flexibility.
Single-file SQLite — zero config for first run.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from rubra.core.tracer.models import Span, Trace, TraceStatus


class Base(DeclarativeBase):
    pass


class TraceRow(Base):
    __tablename__ = "traces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(36), unique=True, nullable=False, index=True)
    agent_name = Column(String(255), nullable=False, index=True)
    task = Column(Text, nullable=True)
    task_description = Column(Text, nullable=True)
    status = Column(String(32), nullable=False)
    final_output = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    total_spans = Column(Integer, default=0)
    total_llm_calls = Column(Integer, default=0)
    total_tool_calls = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    spans_json = Column(JSON, nullable=False, default=list)
    token_usage_json = Column(JSON, nullable=False, default=dict)
    metadata_json = Column(JSON, nullable=False, default=dict)
    tags_json = Column(JSON, nullable=False, default=list)
    expected_output = Column(Text, nullable=True)
    expected_tool_calls_json = Column(JSON, nullable=True)


class MetricResultRow(Base):
    __tablename__ = "metric_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(36), nullable=False, index=True)
    metric_name = Column(String(128), nullable=False, index=True)
    score = Column(Float, nullable=True)
    passed = Column(Integer, nullable=True)
    reason = Column(Text, nullable=True)
    category = Column(String(64), nullable=True)
    is_deterministic = Column(Integer, default=1)
    computed_at = Column(DateTime(timezone=True), nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)


class RubraStorage:
    def __init__(self, database_url: str) -> None:
        is_sqlite = "sqlite" in database_url
        is_memory = ":memory:" in database_url

        kwargs: dict = {"echo": False}
        if is_sqlite:
            kwargs["connect_args"] = {"check_same_thread": False}
        if is_memory:
            # StaticPool keeps a single connection so all sessions share the same DB
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool

        self._engine = create_engine(database_url, **kwargs)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def save_trace(self, trace: Trace) -> None:
        data = dict(
            agent_name=trace.agent_name,
            task=trace.task,
            task_description=trace.task_description,
            status=trace.status.value,
            final_output=trace.final_output,
            error_message=trace.error_message,
            started_at=trace.started_at,
            ended_at=trace.ended_at,
            duration_ms=trace.duration_ms,
            total_spans=len(trace.spans),
            total_llm_calls=trace.total_llm_calls,
            total_tool_calls=trace.total_tool_calls,
            total_tokens=trace.token_usage.total_tokens,
            estimated_cost_usd=trace.token_usage.estimated_cost_usd,
            spans_json=[s.model_dump(mode="json") for s in trace.spans],
            token_usage_json=trace.token_usage.model_dump(mode="json"),
            metadata_json=trace.metadata,
            tags_json=trace.tags,
            expected_output=trace.expected_output,
            expected_tool_calls_json=trace.expected_tool_calls,
        )
        with self._Session() as session:
            existing = session.execute(
                select(TraceRow).where(TraceRow.trace_id == trace.trace_id)
            ).scalar_one_or_none()
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
            else:
                session.add(TraceRow(trace_id=trace.trace_id, **data))
            session.commit()

    def save_metric_result(
        self,
        trace_id: str,
        metric_name: str,
        score: float | None,
        *,
        passed: bool | None = None,
        reason: str | None = None,
        category: str | None = None,
        is_deterministic: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row = MetricResultRow(
            trace_id=trace_id,
            metric_name=metric_name,
            score=score,
            passed=int(passed) if passed is not None else None,
            reason=reason,
            category=category,
            is_deterministic=int(is_deterministic),
            computed_at=datetime.now(UTC),
            metadata_json=metadata or {},
        )
        with self._Session() as session:
            session.add(row)
            session.commit()

    def get_trace(self, trace_id: str) -> Trace | None:
        with self._Session() as session:
            row = session.execute(
                select(TraceRow).where(TraceRow.trace_id == trace_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return _row_to_trace(row)

    def list_traces(
        self,
        agent_name: str | None = None,
        status: TraceStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Trace]:
        with self._Session() as session:
            stmt = select(TraceRow).order_by(TraceRow.started_at.desc())
            if agent_name:
                stmt = stmt.where(TraceRow.agent_name == agent_name)
            if status:
                stmt = stmt.where(TraceRow.status == status.value)
            stmt = stmt.limit(limit).offset(offset)
            rows = session.execute(stmt).scalars().all()
            return [_row_to_trace(r) for r in rows]

    def get_metric_results(self, trace_id: str) -> list[dict[str, Any]]:
        with self._Session() as session:
            rows = (
                session.execute(
                    select(MetricResultRow).where(MetricResultRow.trace_id == trace_id)
                )
                .scalars()
                .all()
            )
            return [
                {
                    "metric_name": r.metric_name,
                    "score": r.score,
                    "passed": bool(r.passed) if r.passed is not None else None,
                    "reason": r.reason,
                    "category": r.category,
                    "is_deterministic": bool(r.is_deterministic),
                    "computed_at": r.computed_at,
                }
                for r in rows
            ]


def _row_to_trace(row: TraceRow) -> Trace:
    from rubra.core.tracer.models import TokenUsage

    spans = [Span.model_validate(s) for s in (row.spans_json or [])]
    return Trace(
        trace_id=row.trace_id,
        agent_name=row.agent_name,
        task=row.task,
        task_description=row.task_description,
        status=TraceStatus(row.status),
        final_output=row.final_output,
        error_message=row.error_message,
        started_at=row.started_at,
        ended_at=row.ended_at,
        duration_ms=row.duration_ms,
        spans=spans,
        token_usage=TokenUsage.model_validate(row.token_usage_json or {}),
        metadata=row.metadata_json or {},
        tags=row.tags_json or [],
        expected_output=row.expected_output,
        expected_tool_calls=row.expected_tool_calls_json,
    )


_storage_instance: RubraStorage | None = None


def get_storage() -> RubraStorage | None:
    return _storage_instance


def init_storage(database_url: str | None = None) -> RubraStorage:
    """Initialize storage. Defaults to SQLite at .rubra/rubra.db"""
    global _storage_instance
    if database_url is None:
        db_dir = Path(".rubra")
        db_dir.mkdir(exist_ok=True)
        database_url = f"sqlite:///{db_dir / 'rubra.db'}"
    _storage_instance = RubraStorage(database_url)
    return _storage_instance


def auto_init_storage() -> RubraStorage:
    """Auto-init from RUBRA_DATABASE_URL env var, or default SQLite."""
    global _storage_instance
    if _storage_instance is None:
        url = os.environ.get("RUBRA_DATABASE_URL")
        _storage_instance = init_storage(url)
    return _storage_instance
