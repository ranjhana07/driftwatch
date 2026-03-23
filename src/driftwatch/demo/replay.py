from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from driftwatch.client import Client
from driftwatch.demo.data import FEATURES
from driftwatch.demo.model import DemoModel


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def feature_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {name: json_value(row[name]) for name in FEATURES}


def reference_rows(model: DemoModel, features: pd.DataFrame, labels: pd.Series) -> list[dict[str, Any]]:
    predictions, scores = model.predict(features)
    embeddings = model.embed(features)
    records = features.to_dict("records")
    return [
        {
            "features": feature_dict(record),
            "prediction": int(predictions[i]),
            "score": float(scores[i]),
            "embedding": embeddings[i].tolist(),
            "label": int(labels.iloc[i]),
        }
        for i, record in enumerate(records)
    ]


def replay_stream(
    client: Client,
    model: DemoModel,
    stream: pd.DataFrame,
    start: datetime,
    seconds_per_event: float = 30.0,
    label_lag_seconds: float = 3600.0,
    id_prefix: str = "evt",
) -> int:
    began = time.perf_counter()
    predictions, scores = model.predict(stream)
    embeddings = model.embed(stream)
    latency_ms = (time.perf_counter() - began) * 1000.0 / max(len(stream), 1)
    records = stream[FEATURES].to_dict("records")
    labels = stream["label"].to_numpy()
    for i, record in enumerate(records):
        when = start + timedelta(seconds=seconds_per_event * i)
        event_id = client.log_prediction(
            feature_dict(record),
            int(predictions[i]),
            score=float(scores[i]),
            embedding=embeddings[i].tolist(),
            latency_ms=latency_ms,
            event_id=f"{id_prefix}-{i}",
            timestamp=when,
        )
        client.log_label(event_id, int(labels[i]), timestamp=when + timedelta(seconds=label_lag_seconds))
    client.flush()
    return len(records)
