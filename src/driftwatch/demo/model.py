from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from driftwatch.demo.data import CATEGORICAL, FEATURES, NUMERIC

EMBEDDING_DIM = 8


@dataclass
class DemoModel:
    preprocessor: ColumnTransformer
    classifier: LogisticRegression
    embedder: PCA

    def _encode(self, frame: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(_model_input(frame))

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        encoded = self._encode(frame)
        scores = self.classifier.predict_proba(encoded)[:, 1]
        return (scores >= 0.5).astype(int), scores

    def embed(self, frame: pd.DataFrame) -> np.ndarray:
        return self.embedder.transform(self._encode(frame))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "DemoModel":
        return joblib.load(path)


def _model_input(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for column in CATEGORICAL:
        values = out[column].astype(object)
        out[column] = values.where(values.notna(), np.nan)
    for column in NUMERIC:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return out


def train_model(x: pd.DataFrame, y: pd.Series, seed: int = 0) -> DemoModel:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=30)),
        ]
    )
    preprocessor = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    encoded = preprocessor.fit_transform(_model_input(x))
    classifier = LogisticRegression(max_iter=500, random_state=seed).fit(encoded, y)
    embedder = PCA(n_components=EMBEDDING_DIM, random_state=seed).fit(encoded)
    return DemoModel(preprocessor, classifier, embedder)


def holdout_auc(model: DemoModel, x: pd.DataFrame, y: pd.Series) -> float:
    return float(roc_auc_score(y, model.predict(x)[1]))
