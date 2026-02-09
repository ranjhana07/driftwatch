from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

OUTPUT_PREDICTION = "__prediction"
OUTPUT_SCORE = "__score"
OUTPUT_COLUMNS = (OUTPUT_PREDICTION, OUTPUT_SCORE)


@dataclass
class Window:
    frame: pd.DataFrame
    embeddings: np.ndarray | None = None
    labels: pd.Series | None = None
    raw_features: list[dict[str, Any]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.frame)

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c not in OUTPUT_COLUMNS]

    @classmethod
    def empty(cls) -> "Window":
        return cls(pd.DataFrame(columns=list(OUTPUT_COLUMNS)))

    @classmethod
    def from_rows(cls, rows: Sequence[dict[str, Any]]) -> "Window":
        if not rows:
            return cls.empty()
        raw = [dict(r["features"]) for r in rows]
        frame = pd.DataFrame(raw)
        frame[OUTPUT_PREDICTION] = [r["prediction"] for r in rows]
        frame[OUTPUT_SCORE] = pd.Series([r.get("score") for r in rows], dtype="float64")
        embeddings = _stack_embeddings([r.get("embedding") for r in rows])
        labels = None
        if any(r.get("label") is not None for r in rows):
            labels = pd.Series([r.get("label") for r in rows], index=frame.index)
        return cls(frame, embeddings, labels, raw)

    def with_labels_only(self) -> "Window":
        if self.labels is None:
            return Window.empty()
        mask = self.labels.notna().to_numpy()
        embeddings = self.embeddings[mask] if self.embeddings is not None and len(self.embeddings) == len(mask) else None
        return Window(
            self.frame[mask].reset_index(drop=True),
            embeddings,
            self.labels[mask].reset_index(drop=True),
            [r for r, keep in zip(self.raw_features, mask) if keep],
        )


def _stack_embeddings(values: Sequence[Any]) -> np.ndarray | None:
    present = [v for v in values if v is not None]
    if len(present) != len(values) or not present:
        return None
    width = len(present[0])
    if any(len(v) != width for v in present):
        return None
    return np.asarray(present, dtype=float)
