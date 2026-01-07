from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from driftwatch.timeutil import as_utc

FeatureValue = float | int | str | bool | None
Scalar = float | int | str

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def validate_name(value: str) -> str:
    if not NAME_PATTERN.match(value):
        raise ValueError("names may contain letters, digits, '_', '.', '-' and be at most 64 characters")
    return value


class PredictionEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=64)
    model: str
    model_version: str = "1"
    timestamp: datetime
    features: dict[str, FeatureValue]
    prediction: Scalar
    score: float | None = None
    embedding: list[float] | None = None
    latency_ms: float | None = None

    @field_validator("model")
    @classmethod
    def _model_name(cls, value: str) -> str:
        return validate_name(value)

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return as_utc(value)


class LabelEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=64)
    model: str
    label: Scalar
    timestamp: datetime

    @field_validator("model")
    @classmethod
    def _model_name(cls, value: str) -> str:
        return validate_name(value)

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return as_utc(value)


class EventBatch(BaseModel):
    events: list[PredictionEvent] = Field(max_length=5000)


class LabelBatch(BaseModel):
    labels: list[LabelEvent] = Field(max_length=5000)


class IngestResult(BaseModel):
    accepted: int


class MonitorConfig(BaseModel):
    window_seconds: int = 3600
    interval_seconds: int = 300
    min_samples: int = 200
    min_labeled: int = 100
    alpha: float = 0.01
    psi_threshold: float = 0.2
    ks_effect: float = 0.1
    wasserstein_threshold: float = 0.1
    chi2_effect: float = 0.1
    js_threshold: float = 0.05
    mmd_effect: float = 0.02
    domain_auc_threshold: float = 0.65
    effect_floor: float = 0.5
    null_rate_increase: float = 0.05
    out_of_range_rate: float = 0.05
    unseen_category_rate: float = 0.02
    performance_drop: float = 0.05
    calibration_increase: float = 0.05
    rmse_ratio: float = 1.25


class ModelRegistration(BaseModel):
    name: str
    task_type: Literal["classification", "regression"] = "classification"
    positive_label: Scalar = 1
    config: MonitorConfig = Field(default_factory=MonitorConfig)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return validate_name(value)


class ReferenceRow(BaseModel):
    features: dict[str, FeatureValue]
    prediction: Scalar
    score: float | None = None
    embedding: list[float] | None = None
    label: Scalar | None = None


class ReferenceUpload(BaseModel):
    rows: list[ReferenceRow] = Field(min_length=1)
