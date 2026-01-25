from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from driftwatch.engine.numeric import ks_statistic, ks_test, normalized_wasserstein, psi, wasserstein


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def manual_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]
    total = 0.0
    for r, c in zip(ref_counts, cur_counts):
        e = max(r / ref_counts.sum(), 1e-4)
        a = max(c / cur_counts.sum(), 1e-4)
        total += (a - e) * np.log(a / e)
    return float(total)


def test_psi_matches_loop_implementation(rng):
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(0.3, 1.1, 2000)
    current = current[(current > reference.min()) & (current < reference.max())]
    assert psi(reference, current) == pytest.approx(manual_psi(reference, current))


def test_psi_is_near_zero_for_same_distribution(rng):
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(0, 1, 5000)
    assert psi(reference, current) < 0.02


def test_psi_grows_with_shift(rng):
    reference = rng.normal(0, 1, 5000)
    small = psi(reference, rng.normal(0.2, 1, 5000))
    large = psi(reference, rng.normal(1.0, 1, 5000))
    assert 0.02 < small < large
    assert large > 0.25


def test_psi_ignores_non_finite_values(rng):
    reference = rng.normal(0, 1, 2000)
    current = np.concatenate([rng.normal(0, 1, 1000), [np.nan, np.inf]])
    assert np.isfinite(psi(reference, current))


def test_ks_statistic_matches_scipy(rng):
    a = rng.normal(0, 1, 800)
    b = rng.normal(0.4, 1.3, 600)
    assert ks_statistic(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic, abs=1e-12)


def test_ks_statistic_with_ties(rng):
    a = rng.integers(0, 5, 500).astype(float)
    b = rng.integers(1, 6, 400).astype(float)
    assert ks_statistic(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic, abs=1e-12)


@pytest.mark.parametrize("shift", [0.0, 0.1, 0.3])
def test_ks_p_value_matches_scipy_asymptotic(rng, shift):
    a = rng.normal(0, 1, 500)
    b = rng.normal(shift, 1, 700)
    ours = ks_test(a, b)
    reference = stats.ks_2samp(a, b, method="asymp")
    assert ours[0] == pytest.approx(reference.statistic, abs=1e-12)
    assert ours[1] == pytest.approx(reference.pvalue, rel=1e-6, abs=1e-12)


def test_ks_p_values_are_uniform_under_the_null(rng):
    p_values = [ks_test(rng.normal(size=300), rng.normal(size=300))[1] for _ in range(300)]
    assert stats.kstest(p_values, "uniform").pvalue > 0.01


def test_wasserstein_matches_scipy(rng):
    a = rng.normal(0, 1, 900)
    b = rng.normal(0.5, 2, 400)
    assert wasserstein(a, b) == pytest.approx(stats.wasserstein_distance(a, b), rel=1e-10)


def test_wasserstein_of_pure_shift_is_the_shift():
    a = np.linspace(0, 1, 1001)
    assert wasserstein(a, a + 2.5) == pytest.approx(2.5)


def test_normalized_wasserstein_divides_by_reference_std(rng):
    a = rng.normal(0, 3, 2000)
    b = a + 1.5
    assert normalized_wasserstein(a, b) == pytest.approx(1.5 / np.std(a), rel=1e-6)


def test_normalized_wasserstein_handles_constant_reference():
    assert normalized_wasserstein(np.ones(50), np.ones(50) * 2) == pytest.approx(1.0)
