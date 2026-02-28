from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml
from pydantic import BaseModel, Field, field_validator

from driftwatch.engine.severity import LEVELS, level_rank
from driftwatch.types import MetricResult


class AlertRule(BaseModel):
    name: str
    model: str = "*"
    kinds: list[str] = Field(default_factory=lambda: ["drift", "quality", "performance"])
    metrics: list[str] | None = None
    features: list[str] | None = None
    min_level: str = "high"
    consecutive: int = Field(default=1, ge=1)
    cooldown_seconds: int = Field(default=3600, ge=0)
    notifiers: list[str] | None = None

    @field_validator("min_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        if value not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}")
        return value

    def applies_to(self, model: str) -> bool:
        return self.model in ("*", model)

    def select(self, results: Sequence[MetricResult]) -> list[MetricResult]:
        floor = level_rank(self.min_level)
        return [
            r
            for r in results
            if r.flagged
            and r.kind in self.kinds
            and level_rank(r.level) >= floor
            and (self.metrics is None or r.metric in self.metrics)
            and (self.features is None or r.feature in self.features)
        ]


def default_rules() -> list[AlertRule]:
    return [
        AlertRule(name="drift-high", kinds=["drift"], min_level="high"),
        AlertRule(name="quality-medium", kinds=["quality"], min_level="medium"),
        AlertRule(name="performance-medium", kinds=["performance"], min_level="medium"),
    ]


def load_rules(path: str | Path) -> list[AlertRule]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [AlertRule.model_validate(item) for item in raw.get("rules", [])]
