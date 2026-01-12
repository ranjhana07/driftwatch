from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import httpx

from driftwatch.client.buffer import DiskBuffer
from driftwatch.client.sampling import is_sampled
from driftwatch.timeutil import as_utc, utcnow

logger = logging.getLogger("driftwatch.client")

_PATHS = {"events": "/v1/events", "labels": "/v1/labels"}
_STOP = object()


@dataclass
class ClientStats:
    queued: int = 0
    sent: int = 0
    rejected: int = 0
    dropped_queue_full: int = 0
    dropped_unsent: int = 0
    spilled: int = 0
    replayed: int = 0


class _Flush:
    def __init__(self) -> None:
        self.done = threading.Event()


class Timer:
    def __init__(self) -> None:
        self.ms = 0.0


class Client:
    def __init__(
        self,
        endpoint: str,
        model: str,
        model_version: str = "1",
        api_key: str | None = None,
        batch_size: int = 100,
        flush_interval: float = 2.0,
        sample_rate: float = 1.0,
        max_queue: int = 10_000,
        buffer_dir: str | Path | None = None,
        buffer_max_bytes: int = 64 * 1024 * 1024,
        timeout: float = 5.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        drain_interval: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        self.model = model
        self.model_version = model_version
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.sample_rate = sample_rate
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.drain_interval = drain_interval
        self.stats = ClientStats()
        self._buffer = DiskBuffer(buffer_dir, buffer_max_bytes) if buffer_dir else None
        headers = {"X-API-Key": api_key} if api_key else {}
        self._http = httpx.Client(base_url=endpoint, headers=headers, timeout=timeout, transport=transport)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._next_drain = 0.0
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="driftwatch-client", daemon=True)
        self._worker.start()

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: float | int | str,
        score: float | None = None,
        embedding: list[float] | None = None,
        latency_ms: float | None = None,
        event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        event_id = event_id or uuid.uuid4().hex
        if not is_sampled(event_id, self.sample_rate):
            return event_id
        event = {
            "event_id": event_id,
            "model": self.model,
            "model_version": self.model_version,
            "timestamp": as_utc(timestamp or utcnow()).isoformat(),
            "features": features,
            "prediction": prediction,
            "score": score,
            "embedding": embedding,
            "latency_ms": latency_ms,
        }
        self._enqueue("events", event)
        return event_id

    def log_label(self, event_id: str, label: float | int | str, timestamp: datetime | None = None) -> None:
        if not is_sampled(event_id, self.sample_rate):
            return
        payload = {
            "event_id": event_id,
            "model": self.model,
            "label": label,
            "timestamp": as_utc(timestamp or utcnow()).isoformat(),
        }
        self._enqueue("labels", payload)

    @contextmanager
    def measure(self) -> Iterator[Timer]:
        timer = Timer()
        start = time.perf_counter()
        try:
            yield timer
        finally:
            timer.ms = (time.perf_counter() - start) * 1000.0

    def flush(self, timeout: float = 30.0) -> bool:
        request = _Flush()
        self._queue.put(request)
        return request.done.wait(timeout)

    def close(self, timeout: float = 30.0) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush(timeout)
        self._queue.put(_STOP)
        self._worker.join(timeout)
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _enqueue(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((kind, payload))
            self.stats.queued += 1
        except queue.Full:
            self.stats.dropped_queue_full += 1

    def _run(self) -> None:
        pending: dict[str, list[dict[str, Any]]] = {"events": [], "labels": []}
        deadline: float | None = None
        while True:
            wait = self.flush_interval if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=wait)
            except queue.Empty:
                item = None
            if item is _STOP:
                self._send_pending(pending)
                return
            if isinstance(item, _Flush):
                self._send_pending(pending)
                self._drain_buffer(force=True)
                deadline = None
                item.done.set()
                continue
            if item is not None:
                kind, payload = item
                pending[kind].append(payload)
                if deadline is None:
                    deadline = time.monotonic() + self.flush_interval
                if len(pending[kind]) >= self.batch_size:
                    self._deliver(kind, pending[kind])
                    pending[kind] = []
            if item is None or (deadline is not None and time.monotonic() >= deadline):
                self._send_pending(pending)
                self._drain_buffer()
                deadline = None

    def _send_pending(self, pending: dict[str, list[dict[str, Any]]]) -> None:
        for kind, items in pending.items():
            if items:
                self._deliver(kind, items)
                pending[kind] = []

    def _post(self, kind: str, items: list[dict[str, Any]]) -> str:
        try:
            response = self._http.post(_PATHS[kind], json={kind: items})
        except httpx.HTTPError as exc:
            logger.warning("delivery failed: %s", exc)
            return "retry"
        if response.is_success:
            return "ok"
        if response.status_code in (408, 429) or response.status_code >= 500:
            return "retry"
        logger.warning("collector rejected batch: %s %s", response.status_code, response.text[:200])
        return "reject"

    def _deliver(self, kind: str, items: list[dict[str, Any]]) -> None:
        for attempt in range(self.max_retries + 1):
            outcome = self._post(kind, items)
            if outcome == "ok":
                self.stats.sent += len(items)
                return
            if outcome == "reject":
                self.stats.rejected += len(items)
                return
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2**attempt))
        if self._buffer is not None:
            self._buffer.put(kind, items)
            self.stats.spilled += len(items)
            self._next_drain = time.monotonic() + self.drain_interval
        else:
            self.stats.dropped_unsent += len(items)

    def _drain_buffer(self, force: bool = False) -> None:
        if self._buffer is None:
            return
        if not force and time.monotonic() < self._next_drain:
            return
        for path in self._buffer.pending():
            kind, items = self._buffer.read(path)
            outcome = self._post(kind, items)
            if outcome == "retry":
                self._next_drain = time.monotonic() + self.drain_interval
                return
            self._buffer.remove(path)
            if outcome == "ok":
                self.stats.replayed += len(items)
            else:
                self.stats.rejected += len(items)
