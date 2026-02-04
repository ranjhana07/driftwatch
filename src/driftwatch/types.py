from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from driftwatch.schemas import MonitorConfig, Scalar

MetricKind = Literal["drift", "quality", "performance"]


class MetricResult(BaseModel):
    model: str = ""
    window_start: datetime
    window_end: datetime
    kind: MetricKind
    feature: str = ""
    metric: str
    value: float
    p_value: float | None = None
    p_adjusted: float | None = None
    severity: float = 0.0
    level: str = "none"
    flagged: bool = False
    n_ref: int = 0
    n_cur: int = 0


class ModelRecord(BaseModel):
    name: str
    task_type: Literal["classification", "regression"] = "classification"
    positive_label: Scalar = 1
    config: MonitorConfig = Field(default_factory=MonitorConfig)
    created_at: datetime | None = None


class Alert(BaseModel):
    id: int | None = None
    model: str
    rule: str
    level: str
    message: str
    window_end: datetime
    created_at: datetime
    delivered: bool = False
