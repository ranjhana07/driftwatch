from __future__ import annotations

import numpy as np
import pytest

from driftwatch.engine.multitest import adjust, benjamini_hochberg, bonferroni, holm

P = np.array([0.01, 0.04, 0.03, 0.005, 0.5])


def test_bonferroni():
    assert bonferroni(P) == pytest.approx([0.05, 0.2, 0.15, 0.025, 1.0])


def test_holm_known_values():
    assert holm(P) == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.5])


def test_benjamini_hochberg_known_values():
    assert benjamini_hochberg(P) == pytest.approx([0.025, 0.05, 0.05, 0.025, 0.5])


def test_adjusted_values_never_below_raw_and_capped():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=200)
    for method in ("bonferroni", "holm", "bh"):
        adjusted = adjust(p, method)
        assert np.all(adjusted >= p - 1e-12)
        assert np.all(adjusted <= 1.0)


def test_ordering_is_preserved():
    rng = np.random.default_rng(4)
    p = rng.uniform(size=50)
    adjusted = benjamini_hochberg(p)
    assert np.all(np.diff(adjusted[np.argsort(p)]) >= -1e-12)


def test_bh_controls_false_discoveries_under_the_null():
    rng = np.random.default_rng(5)
    hits = sum(bool(np.any(benjamini_hochberg(rng.uniform(size=50)) < 0.1)) for _ in range(400))
    assert hits / 400 < 0.15


def test_empty_input():
    assert benjamini_hochberg(np.array([])).size == 0
