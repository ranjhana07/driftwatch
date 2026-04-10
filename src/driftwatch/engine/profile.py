from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import numpy as np

import pandas as pd
from pydantic import BaseModel, Field

from driftwatch.engine.categorical import normalize
from driftwatch.engine.performance import classification_metrics, regression_metrics
from driftwatch.engine.window import OUTPUT_COLUMNS, OUTPUT_PREDICTION, OUTPUT_SCORE, Window
from driftwatch.timeutil import utcnow

MAX_TRACKED_CATEGORIES = 200


class FeatureProfile(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    null_rate: float
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    categories: dict[str, float] = Field(default_factory=dict)
    cardinality: int = 0


class ReferenceProfile(BaseModel):
    n_rows: int
    features: dict[str, FeatureProfile]
    outputs: dict[str, FeatureProfile]
    embedding_dim: int | None = None
    baseline: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def monitored(self) -> dict[str, FeatureProfile]:
        return {**self.features, **self.outputs}


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def profile_column(name: str, series: pd.Series, force_categorical: bool = False) -> FeatureProfile:
    null_rate = float(series.isna().mean()) if len(series) else 0.0
    if is_numeric(series) and not force_categorical:
        values = series.dropna().to_numpy(dtype=float)
        if values.size == 0:
            return FeatureProfile(name=name, kind="numeric", null_rate=null_rate)
        return FeatureProfile(
            name=name,
            kind="numeric",
            null_rate=null_rate,
            minimum=float(values.min()),
            maximum=float(values.max()),
            mean=float(values.mean()),
            std=float(values.std()),
        )
    observed = [v for v in normalize(series) if v != "__missing__"]
    counts = pd.Series(observed, dtype="object").value_counts(normalize=True)
    top = counts.head(MAX_TRACKED_CATEGORIES)
    return FeatureProfile(
        name=name,
        kind="categorical",
        null_rate=null_rate,
        categories={str(k): float(v) for k, v in top.items()},
        cardinality=int(len(counts)),
    )


def build_profile(window: Window, task_type: str = "classification", positive_label: Any = 1) -> ReferenceProfile:
    features = {name: profile_column(name, window.frame[name]) for name in window.feature_columns}
    outputs = {}
    for name in OUTPUT_COLUMNS:
        column = window.frame[name]
        if column.isna().all():
            continue
        categorical = name == OUTPUT_PREDICTION and task_type == "classification"
        outputs[name] = profile_column(name, column, force_categorical=categorical)
    baseline: dict[str, float] = {}
    labeled = window.with_labels_only()
    if labeled.size:
        baseline = performance_for(labeled, task_type, positive_label)
    return ReferenceProfile(
        n_rows=window.size,
        features=features,
        outputs=outputs,
        embedding_dim=None if window.embeddings is None else int(window.embeddings.shape[1]),
        baseline=baseline,
    )


def performance_for(labeled: Window, task_type: str, positive_label: Any) -> dict[str, float]:
    predictions = labeled.frame[OUTPUT_PREDICTION].to_numpy()
    labels = labeled.labels.to_numpy()
    if task_type == "regression":
        return regression_metrics(labels.astype(float), predictions.astype(float))
    scores = labeled.frame["__score"].to_numpy(dtype=float)
    return classification_metrics(labels, predictions, scores, positive_label)



def performance_with_errors(
    labeled: Window, task_type: str, positive_label: Any, resamples: int = 60, seed: int = 0
) -> dict[str, tuple[float, float]]:
    point = performance_for(labeled, task_type, positive_label)
    predictions = labeled.frame[OUTPUT_PREDICTION].to_numpy()
    labels = labeled.labels.to_numpy()
    scores = labeled.frame[OUTPUT_SCORE].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(resamples):
        index = rng.integers(0, len(labels), len(labels))
        if task_type == "regression":
            sample = regression_metrics(labels[index].astype(float), predictions[index].astype(float))
        else:
            sample = classification_metrics(labels[index], predictions[index], scores[index], positive_label)
        for name, value in sample.items():
            if name in draws:
                draws[name].append(value)
    return {
        name: (value, float(np.std(draws[name])) if len(draws[name]) > 1 else 0.0)
        for name, value in point.items()
    }
