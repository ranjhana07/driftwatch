from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JsonType = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class Base(DeclarativeBase):
    pass


class ModelRow(Base):
    __tablename__ = "models"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(16))
    positive_label: Mapped[Any] = mapped_column(JsonType)
    config: Mapped[Any] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class EventRow(Base):
    __tablename__ = "prediction_events"
    __table_args__ = (Index("ix_prediction_events_model_ts", "model", "ts"),)

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    ts: Mapped[datetime] = mapped_column(DateTime)
    features: Mapped[Any] = mapped_column(JsonType)
    prediction: Mapped[Any] = mapped_column(JsonType)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[Any] = mapped_column(JsonType, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class LabelRow(Base):
    __tablename__ = "labels"
    __table_args__ = (Index("ix_labels_model_labeled_at", "model", "labeled_at"),)

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(64))
    label: Mapped[Any] = mapped_column(JsonType)
    labeled_at: Mapped[datetime] = mapped_column(DateTime)


class ReferenceRow(Base):
    __tablename__ = "reference_sets"

    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[Any] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class MetricRow(Base):
    __tablename__ = "metric_results"
    __table_args__ = (Index("ix_metric_results_model_window", "model", "window_end"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(64))
    window_start: Mapped[datetime] = mapped_column(DateTime)
    window_end: Mapped[datetime] = mapped_column(DateTime)
    kind: Mapped[str] = mapped_column(String(16))
    feature: Mapped[str] = mapped_column(String(128), default="")
    metric: Mapped[str] = mapped_column(String(48))
    value: Mapped[float] = mapped_column(Float)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_adjusted: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[float] = mapped_column(Float)
    level: Mapped[str] = mapped_column(String(16))
    flagged: Mapped[bool] = mapped_column(Boolean)
    n_ref: Mapped[int] = mapped_column(Integer, default=0)
    n_cur: Mapped[int] = mapped_column(Integer, default=0)


class AlertRow(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_model_created", "model", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(64))
    rule: Mapped[str] = mapped_column(String(64))
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    window_end: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
