from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCENARIOS = ("none", "covariate_shift", "prior_shift", "concept_drift", "null_burst")
NULLED_FEATURES = ("hours-per-week", "occupation")
FLIPPED_RELATIONSHIPS = ("Husband", "Wife")


@dataclass(frozen=True)
class StreamPlan:
    scenario: str = "none"
    n_events: int = 9000
    onset: int = 4500
    ramp: int = 0
    seed: int = 0
    covariate_strength: float = 0.6
    prior_boost: float = 3.0
    flip_probability: float = 0.6
    null_probability: float = 0.6


def covariate_weights(pool_x: pd.DataFrame, strength: float) -> np.ndarray:
    age = pool_x["age"].to_numpy(dtype=float)
    hours = pool_x["hours-per-week"].to_numpy(dtype=float)
    education = pool_x["education-num"].to_numpy(dtype=float)
    logit = strength * (0.06 * (age - 38.0) + 0.05 * (hours - 40.0) + 0.25 * (education - 10.0))
    return np.exp(logit - logit.max())


def prior_weights(pool_y: pd.Series, boost: float) -> np.ndarray:
    return np.where(pool_y.to_numpy() == 1, boost, 1.0)


def sample_indices(weights: np.ndarray | None, size: int, population: int, rng: np.random.Generator) -> np.ndarray:
    if weights is None:
        return rng.choice(population, size=size, replace=size > population)
    return rng.choice(population, size=size, replace=True, p=weights / weights.sum())


def drift_mask(plan: StreamPlan, rng: np.random.Generator) -> np.ndarray:
    positions = np.arange(plan.n_events)
    if plan.scenario == "none":
        return np.zeros(plan.n_events, dtype=bool)
    after = positions >= plan.onset
    if plan.ramp <= 0:
        return after
    intensity = np.clip((positions - plan.onset + 1) / plan.ramp, 0.0, 1.0)
    return after & (rng.uniform(size=plan.n_events) < intensity)


def build_stream(pool_x: pd.DataFrame, pool_y: pd.Series, plan: StreamPlan) -> pd.DataFrame:
    if plan.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {plan.scenario}")
    rng = np.random.default_rng(plan.seed)
    population = len(pool_x)
    drifted = drift_mask(plan, rng)
    indices = sample_indices(None, plan.n_events, population, rng)
    if plan.scenario == "covariate_shift":
        shifted = sample_indices(covariate_weights(pool_x, plan.covariate_strength), plan.n_events, population, rng)
        indices = np.where(drifted, shifted, indices)
    elif plan.scenario == "prior_shift":
        shifted = sample_indices(prior_weights(pool_y, plan.prior_boost), plan.n_events, population, rng)
        indices = np.where(drifted, shifted, indices)
    stream = pool_x.iloc[indices].reset_index(drop=True)
    labels = pool_y.iloc[indices].reset_index(drop=True).to_numpy().copy()
    if plan.scenario == "concept_drift":
        married = stream["relationship"].isin(FLIPPED_RELATIONSHIPS).to_numpy()
        flip = drifted & married & (rng.uniform(size=plan.n_events) < plan.flip_probability)
        labels = np.where(flip, 1 - labels, labels)
    if plan.scenario == "null_burst":
        broken = drifted & (rng.uniform(size=plan.n_events) < plan.null_probability)
        for column in NULLED_FEATURES:
            values = stream[column].astype(object)
            stream[column] = values.where(~broken, None)
    stream["label"] = labels
    stream["drifted"] = drifted
    return stream
