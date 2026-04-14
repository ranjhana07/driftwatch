from __future__ import annotations

import json

import httpx
import pytest

from driftwatch.client import Client


class Collector:
    def __init__(self, fail_first: int = 0, status: int = 202) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_first = fail_first
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(self.requests) <= self.fail_first:
            return httpx.Response(503)
        return httpx.Response(self.status, json={"accepted": 1})

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


def make_client(collector: Collector, **kwargs) -> Client:
    defaults = dict(batch_size=10, flush_interval=0.05, max_retries=1, backoff_base=0.01, drain_interval=0.01)
    defaults.update(kwargs)
    return Client("http://collector", "churn", transport=httpx.MockTransport(collector), **defaults)


def test_events_are_batched_by_size():
    collector = Collector()
    with make_client(collector, batch_size=5, flush_interval=30) as client:
        for i in range(12):
            client.log_prediction({"x": i}, prediction=1, score=0.5)
        client.flush()
    sizes = sorted(len(b["events"]) for b in collector.bodies("/v1/events"))
    assert sizes == [2, 5, 5]
    assert client.stats.sent == 12


def test_event_payload_shape():
    collector = Collector()
    with make_client(collector) as client:
        event_id = client.log_prediction({"age": 31, "city": "Pune"}, prediction="yes", score=0.8, latency_ms=3.5)
        client.flush()
    event = collector.bodies("/v1/events")[0]["events"][0]
    assert event["event_id"] == event_id
    assert event["model"] == "churn"
    assert event["features"] == {"age": 31, "city": "Pune"}
    assert event["latency_ms"] == 3.5
    assert event["timestamp"].endswith("+00:00")


def test_labels_use_their_own_endpoint():
    collector = Collector()
    with make_client(collector) as client:
        client.log_label("abc", 1)
        client.flush()
    assert collector.bodies("/v1/labels")[0]["labels"][0]["event_id"] == "abc"


def test_sample_rate_zero_sends_nothing():
    collector = Collector()
    with make_client(collector, sample_rate=0.0) as client:
        for i in range(20):
            client.log_prediction({"x": i}, prediction=0)
        client.flush()
    assert collector.requests == []


def test_invalid_sample_rate_rejected():
    with pytest.raises(ValueError):
        Client("http://collector", "m", sample_rate=1.5)


def test_server_errors_are_retried():
    collector = Collector(fail_first=1)
    with make_client(collector, max_retries=2) as client:
        client.log_prediction({"x": 1}, prediction=1)
        client.flush()
    assert client.stats.sent == 1
    assert len(collector.requests) == 2


def test_failed_batches_spill_to_disk_and_replay(tmp_path):
    collector = Collector(fail_first=2)
    with make_client(collector, max_retries=1, buffer_dir=tmp_path / "buf") as client:
        client.log_prediction({"x": 1}, prediction=1)
        client.flush()
        assert client.stats.spilled == 1 or client.stats.replayed == 1
        client.flush()
    assert client.stats.replayed == 1
    assert list((tmp_path / "buf").glob("*.json")) == []


def test_unsent_events_are_counted_without_buffer():
    collector = Collector(fail_first=100)
    with make_client(collector, max_retries=0) as client:
        client.log_prediction({"x": 1}, prediction=1)
        client.flush()
    assert client.stats.dropped_unsent == 1


def test_client_errors_are_not_retried():
    collector = Collector(status=422)
    with make_client(collector, max_retries=3) as client:
        client.log_prediction({"x": 1}, prediction=1)
        client.flush()
    assert len(collector.requests) == 1
    assert client.stats.rejected == 1


def test_measure_reports_milliseconds():
    collector = Collector()
    with make_client(collector) as client:
        with client.measure() as timer:
            sum(range(1000))
    assert timer.ms > 0


def test_close_is_idempotent_and_flushes():
    collector = Collector()
    client = make_client(collector, flush_interval=30)
    client.log_prediction({"x": 1}, prediction=1)
    client.close()
    client.close()
    assert client.stats.sent == 1
