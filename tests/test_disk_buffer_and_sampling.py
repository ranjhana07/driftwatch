from __future__ import annotations

import uuid

from driftwatch.client.buffer import DiskBuffer
from driftwatch.client.sampling import is_sampled


def test_buffer_round_trip_and_order(tmp_path):
    buffer = DiskBuffer(tmp_path)
    first = buffer.put("events", [{"a": 1}])
    second = buffer.put("labels", [{"b": 2}])
    assert buffer.pending() == [first, second]
    assert buffer.read(first) == ("events", [{"a": 1}])
    buffer.remove(first)
    assert buffer.pending() == [second]


def test_buffer_drops_oldest_when_over_limit(tmp_path):
    buffer = DiskBuffer(tmp_path, max_bytes=600)
    for i in range(10):
        buffer.put("events", [{"payload": "x" * 100, "i": i}])
    remaining = buffer.pending()
    assert 1 <= len(remaining) < 10
    assert buffer.size_bytes() <= 600 + 200
    newest = buffer.read(remaining[-1])[1][0]["i"]
    assert newest == 9


def test_sampling_is_deterministic():
    ids = [uuid.uuid4().hex for _ in range(50)]
    assert [is_sampled(i, 0.3) for i in ids] == [is_sampled(i, 0.3) for i in ids]


def test_sampling_extremes():
    assert is_sampled("abc", 1.0)
    assert not is_sampled("abc", 0.0)


def test_sampling_rate_is_close_to_target():
    ids = [uuid.uuid4().hex for _ in range(20000)]
    share = sum(is_sampled(i, 0.25) for i in ids) / len(ids)
    assert abs(share - 0.25) < 0.02
