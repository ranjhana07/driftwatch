from __future__ import annotations

LEVELS = ("none", "low", "medium", "high", "critical")
_FLOORS = (0.0, 0.25, 0.5, 0.75, 0.9)


def severity_from_ratio(ratio: float, flagged: bool) -> float:
    score = min(max(ratio, 0.0) / 4.0, 1.0)
    if flagged:
        return max(score, 0.25)
    return min(score, 0.2)


def level_for(score: float) -> str:
    level = LEVELS[0]
    for name, floor in zip(LEVELS, _FLOORS):
        if score >= floor:
            level = name
    return level


def level_rank(level: str) -> int:
    return LEVELS.index(level)
