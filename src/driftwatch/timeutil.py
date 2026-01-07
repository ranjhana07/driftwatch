from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_naive_utc(value: datetime) -> datetime:
    return as_utc(value).replace(tzinfo=None)


def from_naive_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc)
