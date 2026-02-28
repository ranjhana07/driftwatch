from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from driftwatch.schemas import EventBatch, LabelEvent, MonitorConfig, PredictionEvent


def event(**overrides):
    body = {
        "event_id": "e1",
        "model": "churn",
        "timestamp": "2026-03-01T10:00:00Z",
        "features": {"a": 1},
        "prediction": 1,
    }
    body.update(overrides)
    return body


def test_naive_timestamps_become_utc():
    parsed = PredictionEvent(**event(timestamp=datetime(2026, 3, 1, 10, 0)))
    assert parsed.timestamp.tzinfo is timezone.utc


def test_offset_timestamps_are_converted():
    ist = timezone(timedelta(hours=5, minutes=30))
    parsed = PredictionEvent(**event(timestamp=datetime(2026, 3, 1, 15, 30, tzinfo=ist)))
    assert parsed.timestamp == datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)


def test_non_finite_values_are_nulled():
    parsed = PredictionEvent(**event(features={"a": float("nan"), "b": float("inf"), "c": 2.5}, score=float("nan")))
    assert parsed.features == {"a": None, "b": None, "c": 2.5}
    assert parsed.score is None


def test_model_name_is_validated():
    with pytest.raises(ValidationError):
        PredictionEvent(**event(model="has space"))
    with pytest.raises(ValidationError):
        LabelEvent(event_id="e", model="x" * 65, label=1, timestamp="2026-03-01T10:00:00Z")


def test_batch_size_is_capped():
    with pytest.raises(ValidationError):
        EventBatch(events=[event(event_id=str(i)) for i in range(5001)])


def test_feature_types_are_preserved():
    parsed = PredictionEvent(**event(features={"i": 3, "f": 3.5, "s": "x", "b": True, "n": None}))
    assert parsed.features == {"i": 3, "f": 3.5, "s": "x", "b": True, "n": None}
    assert isinstance(parsed.features["i"], int)
    assert parsed.features["b"] is True


def test_monitor_config_defaults_are_sane():
    config = MonitorConfig()
    assert 0 < config.alpha < 0.1
    assert config.min_samples >= 50
