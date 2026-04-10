from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from adult_like import adult_like

from driftwatch.bench.server import start_server
from driftwatch.client import Client
from driftwatch.client.admin import AdminClient
from driftwatch.collector import create_app
from driftwatch.config import Settings
from driftwatch.demo.data import split_adult
from driftwatch.demo.model import train_model
from driftwatch.demo.replay import feature_dict, json_value, reference_rows, replay_stream
from driftwatch.demo.scenarios import StreamPlan, build_stream
from driftwatch.store.duck import DuckStore

START = datetime(2026, 4, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def prepared():
    x, y = adult_like(7000, seed=3)
    split = split_adult(x, y, seed=1, train_size=2500, reference_size=1200)
    return train_model(split.train_x, split.train_y), split


@pytest.fixture
def server(tmp_path):
    store = DuckStore(tmp_path / "store")
    handle = start_server(create_app(store, Settings()))
    yield handle
    handle.stop()
    store.close()


def test_json_value_conversions():
    import numpy as np

    assert json_value(np.int64(3)) == 3 and isinstance(json_value(np.int64(3)), int)
    assert json_value(np.float64(1.5)) == 1.5
    assert json_value(float("nan")) is None
    assert json_value(None) is None
    assert json_value("x") == "x"


def test_feature_dict_covers_every_feature(prepared):
    _, split = prepared
    record = split.reference_x.iloc[0].to_dict()
    body = feature_dict(record)
    assert set(body) == set(split.reference_x.columns)


def test_null_burst_flows_from_sdk_to_flagged_quality_check(prepared, server):
    model, split = prepared
    admin = AdminClient(server.url)
    admin.register_model("adult-null", config={"window_seconds": 14400, "min_samples": 200, "min_labeled": 100})
    summary = admin.upload_reference("adult-null", reference_rows(model, split.reference_x, split.reference_y))
    assert summary["n_rows"] == 1200
    plan = StreamPlan("null_burst", n_events=2400, onset=1200, seed=4)
    stream = build_stream(split.pool_x, split.pool_y, plan)
    with Client(server.url, "adult-null", batch_size=400, flush_interval=0.2) as client:
        sent = replay_stream(client, model, stream, START, seconds_per_event=30, label_lag_seconds=1800)
        assert client.stats.rejected == 0
    assert sent == 2400
    end = START + timedelta(seconds=30 * 2400 + 1800)
    windows = admin.backfill("adult-null", START, end, step_seconds=7200)["windows"]
    assert windows >= 3

    with httpx.Client(base_url=server.url) as http:
        quality = http.get("/v1/models/adult-null/metrics", params={"kind": "quality", "metric": "null_rate"}).json()
        flagged_features = {m["feature"] for m in quality if m["flagged"]}
        assert {"hours-per-week", "occupation"} <= flagged_features
        early = [m for m in quality if m["feature"] == "occupation" and m["window_end"] < (START + timedelta(hours=10)).isoformat()]
        assert early and not any(m["flagged"] for m in early)
        status = http.get("/v1/models").json()[0]["status"]
        assert status["flagged"] > 0
    admin.close()
