from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
PLANS = ["basic", "plus", "pro"]


def make_rows(
    n: int,
    seed: int = 0,
    age_shift: float = 0.0,
    plan_probs: tuple[float, float, float] = (0.5, 0.3, 0.2),
    null_rate: float = 0.0,
    flip_rate: float = 0.0,
    start: datetime = T0,
    spacing_seconds: float = 10.0,
    labeled: bool = True,
    model: str = "churn",
    embedding_shift: float = 0.0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    age = rng.normal(40 + age_shift, 10, n)
    income = rng.lognormal(10, 0.5, n)
    tenure = rng.integers(0, 60, n)
    plan = rng.choice(PLANS, n, p=list(plan_probs))
    region = rng.choice(["north", "south", "east", "west"], n)
    logit = 0.05 * (age - 40) + 0.03 * (tenure - 30) + np.where(plan == "pro", 1.0, 0.0) - 0.2
    probability = 1 / (1 + np.exp(-logit))
    truth = (rng.uniform(size=n) < probability).astype(int)
    if flip_rate:
        flipped = rng.uniform(size=n) < flip_rate
        truth = np.where(flipped, 1 - truth, truth)
    scores = np.clip(probability + rng.normal(0, 0.05, n), 0.001, 0.999)
    embedding = rng.normal(0, 1, (n, 4)) + embedding_shift
    rows = []
    for i in range(n):
        features: dict[str, Any] = {
            "age": float(age[i]),
            "income": float(income[i]),
            "tenure": int(tenure[i]),
            "plan": str(plan[i]),
            "region": str(region[i]),
        }
        if null_rate and rng.uniform() < null_rate:
            features["income"] = None
        rows.append(
            {
                "event_id": f"{model}-{seed}-{i}",
                "model": model,
                "model_version": "1",
                "timestamp": start + timedelta(seconds=spacing_seconds * i),
                "features": features,
                "prediction": int(scores[i] > 0.5),
                "score": float(scores[i]),
                "embedding": embedding[i].tolist(),
                "latency_ms": float(rng.uniform(1, 5)),
                "label": int(truth[i]) if labeled else None,
            }
        )
    return rows
