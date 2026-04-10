from __future__ import annotations

import numpy as np
import pytest
from synthetic import make_rows

from driftwatch.engine.profile import build_profile
from driftwatch.engine.window import OUTPUT_PREDICTION, OUTPUT_SCORE, Window


def test_window_from_rows_builds_frame_and_outputs():
    rows = make_rows(50, seed=1)
    window = Window.from_rows(rows)
    assert window.size == 50
    assert window.feature_columns == ["age", "income", "tenure", "plan", "region"]
    assert {OUTPUT_PREDICTION, OUTPUT_SCORE} <= set(window.frame.columns)
    assert window.embeddings.shape == (50, 4)
    assert window.labels.notna().all()


def test_window_handles_missing_keys_and_none():
    rows = make_rows(10, seed=2)
    rows[0]["features"].pop("plan")
    rows[1]["features"]["age"] = None
    window = Window.from_rows(rows)
    assert window.frame["plan"].isna().sum() == 1
    assert window.frame["age"].isna().sum() == 1


def test_empty_window():
    assert Window.from_rows([]).size == 0


def test_embeddings_dropped_when_inconsistent():
    rows = make_rows(5, seed=3)
    rows[2]["embedding"] = None
    assert Window.from_rows(rows).embeddings is None


def test_with_labels_only_filters_unlabeled_rows():
    rows = make_rows(20, seed=4)
    for row in rows[:5]:
        row["label"] = None
    labeled = Window.from_rows(rows).with_labels_only()
    assert labeled.size == 15
    assert labeled.embeddings.shape == (15, 4)
    assert len(labeled.raw_features) == 15


def test_profile_kinds_and_stats():
    window = Window.from_rows(make_rows(2000, seed=5))
    profile = build_profile(window)
    assert profile.features["age"].kind == "numeric"
    assert profile.features["tenure"].kind == "numeric"
    assert profile.features["plan"].kind == "categorical"
    assert profile.features["age"].mean == pytest.approx(window.frame["age"].mean())
    assert profile.features["plan"].categories["basic"] == pytest.approx(0.5, abs=0.05)
    assert profile.features["plan"].cardinality == 3
    assert profile.outputs[OUTPUT_PREDICTION].kind == "categorical"
    assert profile.outputs[OUTPUT_SCORE].kind == "numeric"
    assert profile.embedding_dim == 4


def test_profile_records_null_rate():
    rows = make_rows(1000, seed=6, null_rate=0.2)
    profile = build_profile(Window.from_rows(rows))
    assert profile.features["income"].null_rate == pytest.approx(0.2, abs=0.04)


def test_profile_baseline_performance():
    profile = build_profile(Window.from_rows(make_rows(3000, seed=7)))
    assert 0.5 < profile.baseline["auc"] < 1.0
    assert set(profile.baseline) == {"f1", "accuracy", "auc", "ece"}


def test_profile_without_labels_has_no_baseline():
    profile = build_profile(Window.from_rows(make_rows(100, seed=8, labeled=False)))
    assert profile.baseline == {}


def test_profile_round_trips_through_json():
    profile = build_profile(Window.from_rows(make_rows(200, seed=9)))
    restored = type(profile).model_validate_json(profile.model_dump_json())
    assert restored == profile
    assert np.isclose(restored.features["age"].mean, profile.features["age"].mean)


def test_performance_errors_shrink_with_more_labels():
    from driftwatch.engine.profile import performance_with_errors

    small = performance_with_errors(Window.from_rows(make_rows(150, seed=30)), "classification", 1)
    large = performance_with_errors(Window.from_rows(make_rows(3000, seed=31)), "classification", 1)
    assert small["auc"][1] > large["auc"][1] > 0
    assert 0.5 < large["auc"][0] < 1.0


def test_performance_errors_for_regression():
    from driftwatch.engine.profile import performance_with_errors

    rows = make_rows(400, seed=32)
    for row in rows:
        row["prediction"] = float(row["score"])
        row["label"] = float(row["score"]) + 0.1
    out = performance_with_errors(Window.from_rows(rows), "regression", 1)
    assert out["rmse"][0] == pytest.approx(0.1, abs=1e-6)
