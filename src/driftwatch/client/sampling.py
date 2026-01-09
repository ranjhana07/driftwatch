from __future__ import annotations

import zlib


def is_sampled(event_id: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    bucket = zlib.crc32(event_id.encode("utf-8")) / 0xFFFFFFFF
    return bucket < rate
