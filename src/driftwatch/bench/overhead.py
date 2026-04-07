from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np

from driftwatch.bench.environment import describe
from driftwatch.client import Client
from driftwatch.schemas import PredictionEvent
from driftwatch.store.duck import DuckStore

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
CATEGORIES = {
    "workclass": ["Private", "Self-emp", "Gov", "Other"],
    "education": ["HS-grad", "Bachelors", "Masters", "Some-college", "Doctorate"],
    "occupation": ["Sales", "Craft-repair", "Exec-managerial", "Adm-clerical", "Tech-support"],
    "sex": ["Male", "Female"],
}
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def make_payloads(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    numeric = rng.normal(40, 12, (n, len(NUMERIC)))
    choices = {k: rng.integers(0, len(v), n) for k, v in CATEGORIES.items()}
    embeddings = rng.normal(0, 1, (n, 8))
    scores = rng.uniform(size=n)
    payloads = []
    for i in range(n):
        features: dict[str, Any] = {name: float(numeric[i, j]) for j, name in enumerate(NUMERIC)}
        for name, options in CATEGORIES.items():
            features[name] = options[int(choices[name][i])]
        payloads.append(
            {
                "features": features,
                "prediction": int(scores[i] > 0.5),
                "score": float(scores[i]),
                "embedding": embeddings[i].tolist(),
                "latency_ms": 2.0,
            }
        )
    return payloads


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(int(q * len(ordered)), len(ordered) - 1)]

    return {"mean": statistics.fmean(ordered), "p50": pick(0.5), "p95": pick(0.95), "p99": pick(0.99), "max": ordered[-1]}


def sdk_enqueue_overhead(calls: int = 100_000) -> dict[str, Any]:
    payloads = make_payloads(2000)
    transport = httpx.MockTransport(lambda request: httpx.Response(202, json={"accepted": 0}))
    client = Client(
        "http://bench", "bench", batch_size=500, flush_interval=0.5, max_queue=calls + 10, transport=transport
    )
    timings = []
    started = time.perf_counter()
    for i in range(calls):
        p = payloads[i % len(payloads)]
        t0 = time.perf_counter_ns()
        client.log_prediction(p["features"], p["prediction"], score=p["score"], embedding=p["embedding"], latency_ms=2.0)
        timings.append((time.perf_counter_ns() - t0) / 1000.0)
    enqueue_seconds = time.perf_counter() - started
    client.flush(timeout=300)
    client.close()
    stats = percentiles(timings)
    return {
        "calls": calls,
        "call_microseconds": {k: round(v, 2) for k, v in stats.items()},
        "calls_per_second": round(calls / enqueue_seconds),
        "sent_to_noop_transport": client.stats.sent,
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_collector(store_dir: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    env = {**os.environ, "DRIFTWATCH_LOCAL_DIR": str(store_dir), "DRIFTWATCH_DATABASE_URL": ""}
    process = subprocess.Popen(
        [sys.executable, "-m", "driftwatch", "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/healthz", timeout=1).status_code == 200:
                return process, url
        except httpx.HTTPError:
            time.sleep(0.2)
    process.kill()
    raise RuntimeError("collector did not start")


def end_to_end_throughput(events: int = 50_000, batch_size: int = 500) -> dict[str, Any]:
    payloads = make_payloads(5000)
    with tempfile.TemporaryDirectory() as tmp:
        process, url = start_collector(Path(tmp) / "store")
        try:
            client = Client(url, "bench", batch_size=batch_size, flush_interval=0.5, max_queue=events + 10)
            started = time.perf_counter()
            for i in range(events):
                p = payloads[i % len(payloads)]
                client.log_prediction(
                    p["features"],
                    p["prediction"],
                    score=p["score"],
                    embedding=p["embedding"],
                    latency_ms=2.0,
                    event_id=f"bench-{i}",
                    timestamp=START + timedelta(seconds=i),
                )
            enqueued = time.perf_counter() - started
            client.flush(timeout=600)
            total = time.perf_counter() - started
            stats = client.stats
            client.close()
            stored = httpx.get(f"{url}/v1/models", timeout=30).json()[0]["events"]
        finally:
            process.terminate()
            process.wait(timeout=20)
    return {
        "events": events,
        "batch_size": batch_size,
        "enqueue_seconds": round(enqueued, 3),
        "total_seconds": round(total, 3),
        "events_per_second": round(events / total),
        "sent": stats.sent,
        "rejected": stats.rejected,
        "dropped": stats.dropped_queue_full + stats.dropped_unsent,
        "stored": stored,
    }


def ingest_batch_latency(batches: int = 100, batch_size: int = 500) -> dict[str, Any]:
    payloads = make_payloads(batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        process, url = start_collector(Path(tmp) / "store")
        try:
            timings = []
            with httpx.Client(base_url=url, timeout=60) as http:
                for b in range(batches):
                    events = [
                        {
                            "event_id": f"lat-{b}-{i}",
                            "model": "bench",
                            "timestamp": (START + timedelta(seconds=b * batch_size + i)).isoformat(),
                            **p,
                        }
                        for i, p in enumerate(payloads)
                    ]
                    t0 = time.perf_counter()
                    response = http.post("/v1/events", json={"events": events})
                    timings.append((time.perf_counter() - t0) * 1000.0)
                    response.raise_for_status()
        finally:
            process.terminate()
            process.wait(timeout=20)
    return {
        "batches": batches,
        "batch_size": batch_size,
        "request_milliseconds": {k: round(v, 2) for k, v in percentiles(timings).items()},
        "per_event_microseconds_p50": round(percentiles(timings)["p50"] * 1000.0 / batch_size, 1),
    }


def store_read_write(events: int = 50_000, batch_size: int = 500, window_events: int = 480) -> dict[str, Any]:
    payloads = make_payloads(batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        store = DuckStore(Path(tmp) / "store")
        started = time.perf_counter()
        for b in range(events // batch_size):
            batch = [
                PredictionEvent(
                    event_id=f"st-{b}-{i}",
                    model="bench",
                    timestamp=START + timedelta(seconds=30 * (b * batch_size + i)),
                    **p,
                )
                for i, p in enumerate(payloads)
            ]
            store.add_events(batch)
        write_seconds = time.perf_counter() - started
        middle = START + timedelta(seconds=30 * (events // 2))
        window = (middle, middle + timedelta(seconds=30 * window_events))
        before = time_reads(lambda: store.read_events("bench", *window))
        parts_before = len(list((Path(tmp) / "store" / "events" / "bench").glob("*.parquet")))
        result: dict[str, Any] = {
            "events": events,
            "write_events_per_second": round(events / write_seconds),
            "parquet_parts": parts_before,
            "read_window_events": window_events,
            "read_window_milliseconds": before,
        }
        compact = getattr(store, "compact", None)
        if compact is not None:
            compact("bench")
            result["parquet_parts_after_compaction"] = len(list((Path(tmp) / "store" / "events" / "bench").glob("*.parquet")))
            result["read_window_milliseconds_after_compaction"] = time_reads(lambda: store.read_events("bench", *window))
        store.close()
    return result


def time_reads(read: Callable[[], Any], repeats: int = 15) -> dict[str, float]:
    read()
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        rows = read()
        timings.append((time.perf_counter() - t0) * 1000.0)
    assert rows
    stats = percentiles(timings)
    return {"p50": round(stats["p50"], 2), "p95": round(stats["p95"], 2)}


def run_overhead(out_dir: Path, scale: float = 1.0, log: Callable[[str], None] = print) -> dict[str, Any]:
    results: dict[str, Any] = {"environment": describe()}
    log("sdk enqueue overhead")
    results["sdk_enqueue"] = sdk_enqueue_overhead(int(100_000 * scale))
    log("end to end throughput against a collector subprocess")
    results["end_to_end"] = end_to_end_throughput(int(50_000 * scale))
    log("ingest batch latency")
    results["ingest_batch"] = ingest_batch_latency(max(int(100 * scale), 10))
    log("local store read and write")
    results["local_store"] = store_read_write(int(50_000 * scale))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "overhead.json", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(results, indent=2) + "\n")
    return results
