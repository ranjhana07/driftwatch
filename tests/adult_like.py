from __future__ import annotations

import numpy as np
import pandas as pd

from driftwatch.demo.data import clean_frame


def adult_like(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 80, n)
    education = rng.integers(1, 17, n)
    hours = np.clip(rng.normal(40, 10, n), 1, 99).round()
    relationship = rng.choice(["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried"], n, p=[0.4, 0.05, 0.25, 0.2, 0.1])
    occupation = rng.choice(["Sales", "Craft-repair", "Exec-managerial", "Adm-clerical", None], n, p=[0.25, 0.25, 0.2, 0.2, 0.1])
    logit = 0.04 * (age - 40) + 0.25 * (education - 10) + 0.03 * (hours - 40) + np.where(relationship == "Husband", 1.0, -0.5) - 1.0
    label = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame(
        {
            "age": age,
            "fnlwgt": rng.integers(20000, 400000, n),
            "education-num": education,
            "capital-gain": np.where(rng.uniform(size=n) < 0.08, rng.integers(500, 20000, n), 0),
            "capital-loss": np.where(rng.uniform(size=n) < 0.05, rng.integers(100, 3000, n), 0),
            "hours-per-week": hours,
            "workclass": rng.choice(["Private", "Self-emp", "Gov"], n, p=[0.7, 0.15, 0.15]),
            "education": rng.choice(["HS-grad", "Bachelors", "Masters", "Some-college"], n),
            "marital-status": rng.choice(["Married-civ-spouse", "Never-married", "Divorced"], n),
            "occupation": occupation,
            "relationship": relationship,
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], n, p=[0.85, 0.1, 0.05]),
            "sex": rng.choice(["Male", "Female"], n),
            "native-country": rng.choice(["United-States", "Mexico", "India"], n, p=[0.9, 0.06, 0.04]),
        }
    )
    return clean_frame(frame), pd.Series(label)
