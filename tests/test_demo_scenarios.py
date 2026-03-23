from __future__ import annotations

import numpy as np
import pytest
from adult_like import adult_like
from sklearn.metrics import roc_auc_score

from driftwatch.demo.data import FEATURES, split_adult
from driftwatch.demo.model import DemoModel, train_model
from driftwatch.demo.scenarios import SCENARIOS, StreamPlan, build_stream
from driftwatch.engine.numeric import psi


@pytest.fixture(scope="module")
def split():
    x, y = adult_like(6000, seed=1)
    return split_adult(x, y, seed=2, train_size=2500, reference_size=1000)


@pytest.fixture(scope="module")
def model(split):
    return train_model(split.train_x, split.train_y)


def stream(split, scenario, **kwargs):
    plan = StreamPlan(scenario=scenario, n_events=3000, onset=1500, seed=5, **kwargs)
    return build_stream(split.pool_x, split.pool_y, plan)


def test_split_sizes_are_disjoint(split):
    assert len(split.train_x) == 2500
    assert len(split.reference_x) == 1000
    assert len(split.pool_x) == 2500
    assert list(split.pool_x.columns) == FEATURES


def test_no_drift_scenario_has_no_drifted_rows(split):
    assert not stream(split, "none")["drifted"].any()


def test_drift_starts_at_the_onset(split):
    drifted = stream(split, "covariate_shift")["drifted"].to_numpy()
    assert not drifted[:1500].any()
    assert drifted[1500:].all()


def test_ramp_phases_drift_in(split):
    plan = StreamPlan("covariate_shift", n_events=3000, onset=1000, ramp=1000, seed=3)
    drifted = build_stream(split.pool_x, split.pool_y, plan)["drifted"].to_numpy()
    early = drifted[1000:1300].mean()
    late = drifted[1700:2000].mean()
    assert 0.05 < early < 0.55
    assert late > 0.7
    assert drifted[2000:].all()


def test_covariate_shift_moves_marginals_but_not_the_labeling_rule(split, model):
    data = stream(split, "covariate_shift")
    before, after = data[~data.drifted], data[data.drifted]
    assert psi(before["age"].to_numpy(), after["age"].to_numpy()) > 0.1
    assert after["age"].mean() > before["age"].mean()
    scores = model.predict(after)[1]
    assert roc_auc_score(after["label"], scores) > 0.7


def test_prior_shift_raises_the_positive_rate(split):
    data = stream(split, "prior_shift")
    before, after = data[~data.drifted], data[data.drifted]
    assert after["label"].mean() > before["label"].mean() + 0.1


def test_concept_drift_only_changes_labels(split, model):
    data = stream(split, "concept_drift")
    baseline = stream(split, "none")
    assert data[FEATURES].equals(baseline[FEATURES])
    before, after = data[~data.drifted], data[data.drifted]
    auc_before = roc_auc_score(before["label"], model.predict(before)[1])
    auc_after = roc_auc_score(after["label"], model.predict(after)[1])
    assert auc_after < auc_before - 0.03


def test_null_burst_nulls_only_after_onset(split):
    data = stream(split, "null_burst")
    before, after = data[~data.drifted], data[data.drifted]
    assert before["occupation"].isna().mean() < 0.2
    assert after["occupation"].isna().mean() > 0.5
    assert after["hours-per-week"].isna().mean() > 0.5


def test_stream_is_reproducible(split):
    a = stream(split, "prior_shift")
    b = stream(split, "prior_shift")
    assert a.equals(b)


def test_unknown_scenario_is_rejected(split):
    with pytest.raises(ValueError):
        stream(split, "earthquake")


def test_all_scenarios_build(split):
    for scenario in SCENARIOS:
        assert len(stream(split, scenario)) == 3000


def test_model_predictions_and_embeddings(split, model):
    labels, scores = model.predict(split.reference_x)
    assert set(np.unique(labels)) <= {0, 1}
    assert scores.min() >= 0 and scores.max() <= 1
    assert model.embed(split.reference_x).shape == (1000, 8)
    assert roc_auc_score(split.reference_y, scores) > 0.7


def test_model_tolerates_nulls(split, model):
    broken = stream(split, "null_burst")
    labels, scores = model.predict(broken)
    assert len(scores) == 3000 and np.isfinite(scores).all()


def test_model_round_trips_through_disk(split, model, tmp_path):
    path = tmp_path / "m.joblib"
    model.save(path)
    restored = DemoModel.load(path)
    assert np.allclose(restored.predict(split.reference_x)[1], model.predict(split.reference_x)[1])
