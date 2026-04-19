from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from synthetic import T0, make_rows

from driftwatch.collector import create_app
from driftwatch.config import Settings


def event_json(row):
    return {
        "event_id": row["event_id"],
        "model": row["model"],
        "model_version": row["model_version"],
        "timestamp": row["timestamp"].isoformat(),
        "features": row["features"],
        "prediction": row["prediction"],
        "score": row["score"],
        "embedding": row["embedding"],
        "latency_ms": row["latency_ms"],
    }


def label_json(row, delay=60):
    return {
        "event_id": row["event_id"],
        "model": row["model"],
        "label": row["label"],
        "timestamp": (row["timestamp"] + timedelta(seconds=delay)).isoformat(),
    }


def reference_json(rows):
    return {
        "rows": [
            {k: r[k] for k in ("features", "prediction", "score", "embedding", "label")} for r in rows
        ]
    }


@pytest.fixture
def client(store):
    with TestClient(create_app(store, Settings(api_key=None))) as c:
        yield c


def test_health(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"


def test_ingest_events_and_auto_register(client, store):
    rows = make_rows(30)
    response = client.post("/v1/events", json={"events": [event_json(r) for r in rows]})
    assert response.status_code == 202
    assert response.json() == {"accepted": 30}
    assert store.get_model("churn") is not None
    assert store.event_count("churn") == 30


def test_ingest_validation_errors(client):
    bad = event_json(make_rows(1)[0])
    bad["model"] = "bad model name"
    assert client.post("/v1/events", json={"events": [bad]}).status_code == 422
    assert client.post("/v1/events", json={"nope": []}).status_code == 422


def test_labels_endpoint(client):
    rows = make_rows(5)
    client.post("/v1/events", json={"events": [event_json(r) for r in rows]})
    response = client.post("/v1/labels", json={"labels": [label_json(r) for r in rows]})
    assert response.status_code == 202
    assert response.json()["accepted"] == 5


def test_register_and_list_models(client):
    response = client.post(
        "/v1/models", json={"name": "churn", "task_type": "classification", "config": {"window_seconds": 1800}}
    )
    assert response.status_code == 201
    assert response.json()["config"]["window_seconds"] == 1800
    listing = client.get("/v1/models").json()
    assert listing[0]["name"] == "churn"
    assert listing[0]["has_reference"] is False
    assert listing[0]["status"]["level"] == "none"
    assert client.get("/v1/models/churn").json()["name"] == "churn"
    assert client.get("/v1/models/missing").status_code == 404


def test_monitor_requires_reference(client):
    client.post("/v1/models", json={"name": "churn"})
    assert client.post("/v1/models/churn/monitor").status_code == 409
    assert client.put("/v1/models/missing/reference", json=reference_json(make_rows(5))).status_code == 404


def full_flow(client, shift=0.0):
    client.post("/v1/models", json={"name": "churn", "config": {"min_samples": 100, "min_labeled": 50}})
    reference = make_rows(2000, seed=300, start=T0 - timedelta(days=1))
    response = client.put("/v1/models/churn/reference", json=reference_json(reference))
    assert response.status_code == 200
    rows = make_rows(450, seed=1, spacing_seconds=8, age_shift=shift)
    client.post("/v1/events", json={"events": [event_json(r) for r in rows]})
    client.post("/v1/labels", json={"labels": [label_json(r) for r in rows]})
    end = (T0 + timedelta(hours=1)).isoformat()
    return client.post("/v1/models/churn/monitor", params={"window_end": end}), end


def test_reference_upload_summary(client):
    client.post("/v1/models", json={"name": "churn"})
    body = client.put("/v1/models/churn/reference", json=reference_json(make_rows(500))).json()
    assert body["n_rows"] == 500
    assert "age" in body["features"]
    assert "auc" in body["baseline"]


def test_monitor_run_and_queries(client):
    response, end = full_flow(client, shift=10)
    body = response.json()
    assert body["results"] > 0
    assert body["flagged"] >= 1
    metrics = client.get("/v1/models/churn/metrics", params={"kind": "drift", "feature": "age"}).json()
    assert {m["metric"] for m in metrics} == {"psi", "ks", "wasserstein"}
    heat = client.get("/v1/models/churn/heatmap").json()
    assert heat["windows"] and "age" in heat["features"]
    assert all(len(cell) == 5 for cell in heat["cells"])
    performance = client.get("/v1/models/churn/performance").json()
    assert "auc" in performance
    latest = client.get("/v1/models/churn/latest").json()
    assert len(latest) == body["results"]
    status = client.get("/v1/models").json()[0]["status"]
    assert status["flagged"] >= 1
    assert status["level"] != "none"


def test_distribution_numeric_and_categorical(client):
    full_flow(client, shift=10)
    numeric = client.get("/v1/models/churn/distribution", params={"feature": "age"}).json()
    assert numeric["kind"] == "numeric"
    assert len(numeric["edges"]) == len(numeric["reference"]) + 1
    assert sum(numeric["reference"]) == pytest.approx(1.0)
    assert sum(numeric["current"]) == pytest.approx(1.0)
    assert max(numeric["current"]) > 0
    categorical = client.get("/v1/models/churn/distribution", params={"feature": "plan"}).json()
    assert categorical["kind"] == "categorical"
    assert set(categorical["categories"]) >= {"basic", "plus", "pro"}
    assert sum(categorical["current"]) == pytest.approx(1.0)
    assert client.get("/v1/models/churn/distribution", params={"feature": "zzz"}).status_code == 404


def test_backfill_endpoint(client):
    full_flow(client)
    response = client.post(
        "/v1/models/churn/backfill",
        params={"start": T0.isoformat(), "end": (T0 + timedelta(hours=2)).isoformat(), "step_seconds": 1800},
    )
    assert response.status_code == 200
    assert response.json()["windows"] >= 1


def test_alerts_listing_empty(client):
    assert client.get("/v1/alerts").json() == []


def test_prometheus_metrics_exposed(client):
    full_flow(client, shift=10)
    body = client.get("/metrics").text
    assert 'driftwatch_events_ingested_total{model="churn"} 450.0' in body
    assert 'driftwatch_drift_score{detector="psi",feature="age",model="churn"}' in body
    assert "driftwatch_ingest_batch_seconds_count" in body
    assert 'driftwatch_performance{metric="auc",model="churn"}' in body


def test_api_key_is_enforced(store):
    with TestClient(create_app(store, Settings(api_key="s3cret"))) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_static_ui_is_served(store, tmp_path):
    (tmp_path / "index.html").write_text("<h1>hello</h1>")
    with TestClient(create_app(store, Settings(ui_dir=tmp_path))) as c:
        assert "hello" in c.get("/").text
        assert c.get("/healthz").status_code == 200


def test_cors_headers_when_configured(store):
    settings = Settings(cors_origins="http://localhost:8080")
    with TestClient(create_app(store, settings)) as c:
        response = c.get("/healthz", headers={"Origin": "http://localhost:8080"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:8080"
        other = c.get("/healthz", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in other.headers


def test_heatmap_lists_model_outputs_after_input_features(client):
    full_flow(client, shift=10)
    features = client.get("/v1/models/churn/heatmap").json()["features"]
    first_output = next(i for i, f in enumerate(features) if f.startswith("__"))
    assert all(f.startswith("__") for f in features[first_output:])
    assert features[:first_output] == sorted(features[:first_output])


def test_oversized_batches_are_rejected(client):
    row = event_json(make_rows(1)[0])
    events = [{**row, "event_id": f"e{i}"} for i in range(5001)]
    assert client.post("/v1/events", json={"events": events}).status_code == 422


def test_duplicate_ingest_does_not_double_count_events(client, store):
    rows = make_rows(10)
    body = {"events": [event_json(r) for r in rows]}
    client.post("/v1/events", json=body)
    client.post("/v1/events", json=body)
    assert store.event_count("churn") == 10
