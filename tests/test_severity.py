from __future__ import annotations

import pytest

from driftwatch.engine.severity import LEVELS, level_for, level_rank, severity_from_ratio


def test_unflagged_results_stay_below_low():
    assert severity_from_ratio(0.9, flagged=False) < 0.25
    assert level_for(severity_from_ratio(50, flagged=False)) == "none"


def test_flagged_results_are_at_least_low():
    assert level_for(severity_from_ratio(0.1, flagged=True)) == "low"


@pytest.mark.parametrize(
    "ratio, level",
    [(1.0, "low"), (2.0, "medium"), (3.0, "high"), (3.7, "critical"), (40.0, "critical")],
)
def test_ratio_to_level(ratio, level):
    assert level_for(severity_from_ratio(ratio, flagged=True)) == level


def test_severity_is_capped_and_monotonic():
    values = [severity_from_ratio(r, True) for r in (0.5, 1, 2, 3, 4, 8)]
    assert values == sorted(values)
    assert max(values) == 1.0


def test_level_rank_orders_levels():
    assert [level_rank(level) for level in LEVELS] == [0, 1, 2, 3, 4]
