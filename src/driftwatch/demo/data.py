from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
FEATURES = NUMERIC + CATEGORICAL
OPENML_ADULT_ID = 1590


@dataclass
class AdultSplit:
    train_x: pd.DataFrame
    train_y: pd.Series
    reference_x: pd.DataFrame
    reference_y: pd.Series
    pool_x: pd.DataFrame
    pool_y: pd.Series


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for column in CATEGORICAL:
        values = out[column].astype(object)
        out[column] = values.where(values.notna(), np.nan)
    for column in NUMERIC:
        out[column] = out[column].astype(float)
    return out.reset_index(drop=True)


def load_adult(cache_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_ADULT_ID, as_frame=True, data_home=str(cache_dir))
    frame = bunch.frame
    target = (frame["class"].astype(str) == ">50K").astype(int)
    return clean_frame(frame), target.reset_index(drop=True)


def split_adult(
    x: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
    train_size: int = 20000,
    reference_size: int = 4000,
) -> AdultSplit:
    order = np.random.default_rng(seed).permutation(len(x))
    train = order[:train_size]
    reference = order[train_size : train_size + reference_size]
    pool = order[train_size + reference_size :]

    def take(indices: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
        return x.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)

    train_x, train_y = take(train)
    reference_x, reference_y = take(reference)
    pool_x, pool_y = take(pool)
    return AdultSplit(train_x, train_y, reference_x, reference_y, pool_x, pool_y)
