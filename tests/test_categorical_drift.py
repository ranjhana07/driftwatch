from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.spatial.distance import jensenshannon

from driftwatch.engine.categorical import MISSING, aligned_counts, chi_square, cramers_v, js_divergence


def test_aligned_counts_union_of_categories():
    categories, ref, cur = aligned_counts(["a", "a", "b"], ["b", "c", None])
    assert categories == [MISSING, "a", "b", "c"]
    assert ref.tolist() == [0, 2, 1, 0]
    assert cur.tolist() == [1, 0, 1, 1]


def test_chi_square_matches_scipy_contingency():
    ref = np.array([120, 80, 50, 30], dtype=float)
    cur = np.array([90, 100, 70, 20], dtype=float)
    statistic, p_value, dof = chi_square(ref, cur)
    expected = stats.chi2_contingency(np.vstack([ref, cur]), correction=False)
    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue)
    assert dof == expected.dof


def test_chi_square_ignores_empty_categories():
    statistic, p_value, dof = chi_square(np.array([10, 20, 0.0]), np.array([12, 18, 0.0]))
    assert dof == 1
    assert 0.0 <= p_value <= 1.0
    assert statistic >= 0


def test_chi_square_degenerate_inputs():
    assert chi_square(np.array([5.0]), np.array([7.0])) == (0.0, 1.0, 0)
    assert chi_square(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == (0.0, 1.0, 0)


def test_chi_square_p_values_uniform_under_null():
    rng = np.random.default_rng(1)
    probs = [0.5, 0.3, 0.15, 0.05]
    p_values = []
    for _ in range(300):
        a = rng.multinomial(400, probs).astype(float)
        b = rng.multinomial(400, probs).astype(float)
        p_values.append(chi_square(a, b)[1])
    assert stats.kstest(p_values, "uniform").pvalue > 0.01


def test_cramers_v():
    assert cramers_v(40.0, 400.0) == pytest.approx(np.sqrt(0.1))
    assert cramers_v(1.0, 0.0) == 0.0


def test_js_matches_scipy_distance_squared():
    p = np.array([10, 20, 30, 40], dtype=float)
    q = np.array([25, 25, 25, 25], dtype=float)
    assert js_divergence(p, q) == pytest.approx(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


def test_js_zero_for_identical_and_one_for_disjoint():
    p = np.array([5, 5, 0, 0], dtype=float)
    assert js_divergence(p, p) == pytest.approx(0.0)
    assert js_divergence(p, np.array([0, 0, 3, 7], dtype=float)) == pytest.approx(1.0)


def test_js_is_symmetric_and_handles_zero_counts():
    p = np.array([3, 0, 7], dtype=float)
    q = np.array([0, 5, 5], dtype=float)
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert js_divergence(np.zeros(3), q) == 0.0
