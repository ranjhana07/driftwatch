from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.engine.distribution import compare_categorical, compare_numeric


def test_numeric_densities_sum_to_one_and_shift_is_visible():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(2, 1, 1000))
    body = compare_numeric(reference, current)
    assert sum(body["reference"]) == pytest.approx(1.0)
    assert sum(body["current"]) == pytest.approx(1.0)
    ref_peak = int(np.argmax(body["reference"]))
    cur_peak = int(np.argmax(body["current"]))
    assert cur_peak > ref_peak
    assert body["n_reference"] == 2000 and body["n_current"] == 1000


def test_numeric_with_empty_current():
    body = compare_numeric(pd.Series([1.0, 2.0, 3.0]), pd.Series(dtype=float))
    assert sum(body["current"]) == 0.0
    assert len(body["edges"]) == len(body["reference"]) + 1


def test_numeric_constant_column():
    body = compare_numeric(pd.Series([5.0] * 10), pd.Series([5.0] * 4))
    assert len(body["edges"]) == 21
    assert sum(body["reference"]) == pytest.approx(1.0)


def test_numeric_ignores_nulls():
    body = compare_numeric(pd.Series([1.0, None, 3.0]), pd.Series([None, 2.0]))
    assert body["n_reference"] == 2
    assert body["n_current"] == 1


def test_categorical_top_categories_and_other():
    reference = pd.Series(["a"] * 50 + ["b"] * 30 + [f"rare{i}" for i in range(20)])
    current = pd.Series(["a"] * 20 + ["b"] * 70 + ["new"] * 10)
    body = compare_categorical(reference, current)
    assert body["categories"][:2] == ["a", "b"]
    assert body["categories"][-1] == "other"
    assert sum(body["reference"]) == pytest.approx(1.0)
    assert sum(body["current"]) == pytest.approx(1.0)
    assert "new" in body["categories"]


def test_categorical_marks_missing():
    reference = pd.Series(["a", "b", None, "a"])
    body = compare_categorical(reference, pd.Series([None, None, "a"]))
    assert "(missing)" in body["categories"]
