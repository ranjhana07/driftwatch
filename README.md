# driftwatch

Monitoring for production ML models. A small client logs predictions from the serving process, a collector stores them, and a scheduler compares each window of live traffic against a reference dataset. It checks for feature drift, embedding drift, data quality problems and, once labels arrive, a drop in model performance. Findings become metrics, a dashboard and alerts.

Everything runs locally against DuckDB and Parquet with no services, or against PostgreSQL with Docker Compose.

## How it works

```
 serving process                      collector                          scheduler
+----------------+   HTTP batches   +------------------+              +-------------------+
| driftwatch     | ---------------> | FastAPI          |              | APScheduler       |
| Client         |  events, labels  |  /v1/events      |              |  one job per model|
|  queue + thread|                  |  /v1/labels      |              |                   |
|  disk buffer   |                  |  /v1/models/...  |              |  Monitor.run()    |
+----------------+                  +---------+--------+              |   window vs       |
                                              |                       |   reference       |
                                              v                       |  AlertManager     |
                                    +------------------+  read/write  |   rules, cooldown |
                                    | Store            | <----------> |   slack, email    |
                                    |  PostgreSQL      |              +---------+---------+
                                    |  or DuckDB +     |                        |
                                    |  Parquet (local) |                        v
                                    +---------+--------+              +-------------------+
                                              |                       | /metrics          |
                          +-------------------+---------+             | Prometheus -> Grafana
                          v                             v             +-------------------+
                  +---------------+             +---------------+
                  | static web UI |             | /metrics      |
                  | ECharts       |             | Prometheus    |
                  +---------------+             +---------------+
```

A model is registered with a reference dataset: rows of features, prediction, score, optional embedding and optional label. The reference gives each feature a profile (type, null rate, range, quantiles, category frequencies) and a baseline for AUC, F1, calibration error or RMSE.

Every `interval_seconds` the scheduler takes the events of the last `window_seconds` and runs the engine:

| Check | Applies to | Statistic | Flagged when |
| --- | --- | --- | --- |
| `psi` | numeric features, model score | population stability index, 10 reference-quantile bins, open-ended outer bins | PSI >= 0.2 |
| `ks` | numeric | two-sample Kolmogorov-Smirnov, asymptotic p-value | adjusted p < alpha and statistic >= 0.05 |
| `wasserstein` | numeric | 1-Wasserstein divided by the reference standard deviation | >= 0.2 |
| `chi2` | categorical, predicted class | chi-square test of homogeneity, effect size is Cramer's V | adjusted p < alpha and V >= 0.05 |
| `js` | categorical | Jensen-Shannon divergence, base 2 | >= 0.05 |
| `mmd` | embeddings | unbiased MMD with an RBF kernel, median bandwidth, permutation p-value | adjusted p < alpha and MMD^2 >= 0.01 |
| `domain_classifier` | embeddings | cross-validated AUC of a logistic regression separating reference from current | adjusted p < alpha and AUC >= 0.65 |
| quality | all features | null rate against the reference, out-of-range rate, missing keys, type mismatches, unexpected features, unseen categories, cardinality ratio | per-check thresholds in `MonitorConfig` |
| performance | labelled events | AUC, F1, accuracy, expected calibration error, RMSE | change from the reference baseline exceeds the threshold and 3 bootstrap standard errors |

p-values are corrected with Benjamini-Hochberg across the features of each detector within a window. Every result gets a severity score between 0 and 1 (the effect size relative to the detector's threshold, divided by four and capped) and a level: `none`, `low`, `medium`, `high`, `critical`.

Labels are delayed. A window only uses labels that had arrived by the end of the window, so performance results appear later than drift results and never use information from the future.

## Setup

Python 3.11 or newer.

```
pip install -e ".[server,dev]"
```

The base package (`pip install .`) only installs the client and its two dependencies, `httpx` and `pydantic`.

### Local mode

No database needed. Events are written as Parquet files and metadata goes into a DuckDB file under `.driftwatch/store`.

```
driftwatch serve --ui-dir ui
```

Open http://127.0.0.1:8000. In another terminal, replay the demo traffic (downloads the UCI Adult dataset from OpenML on first use):

```
driftwatch demo run
```

This trains a logistic regression on Adult, registers five models (`adult-control`, `adult-covariate-shift`, `adult-prior-shift`, `adult-concept-drift`, `adult-null-burst`), uploads a reference set, replays 9000 events per model through the client with drift injected half way, and backfills the monitor windows so the dashboard has history.

Run the scheduler to keep evaluating the latest window and to send alerts:

```
driftwatch scheduler
```

### Docker Compose

```
cp .env.example .env
docker compose up --build
```

| Service | Port | Purpose |
| --- | --- | --- |
| collector | 8000 | API, runs migrations on start |
| scheduler | 9102 | monitor runs, alerts, own Prometheus metrics |
| ui | 8080 | nginx serving `ui/`, proxying `/v1` to the collector |
| prometheus | 9090 | scrapes collector and scheduler |
| grafana | 3000 | dashboard provisioned from `deploy/grafana` |
| postgres | 5432 | storage |

The compose file was not run while writing this README because the machine it was developed on has no Docker. The configuration is plain, but treat it as untested.

## Configuration

All settings are environment variables with the `DRIFTWATCH_` prefix, or entries in a `.env` file. See `.env.example`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | unset | SQLAlchemy URL. Unset means local DuckDB and Parquet mode |
| `LOCAL_DIR` | `.driftwatch/store` | where local mode keeps its files |
| `API_KEY` | unset | when set, `/v1` requires an `X-API-Key` header |
| `HOST`, `PORT` | `127.0.0.1`, `8000` | collector bind address |
| `UI_DIR` | unset | serve a static UI from the collector |
| `CORS_ORIGINS` | unset | comma separated origins allowed to call the API |
| `ALERT_RULES_PATH` | unset | YAML file of alert rules, defaults are used when unset |
| `SLACK_WEBHOOK_URL` | unset | enables the Slack notifier |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_TO` | unset | enable the email notifier |
| `SCHEDULER_METRICS_PORT` | `9102` | scheduler Prometheus port |
| `SYNC_INTERVAL_SECONDS` | `60` | how often the scheduler picks up new or changed models |

Per-model monitoring settings (`MonitorConfig`) are sent when registering a model: `window_seconds`, `interval_seconds`, `min_samples`, `min_labeled`, `alpha` and the thresholds in the table above.

## Usage

### SDK

```python
from driftwatch import Client

client = Client("http://127.0.0.1:8000", model="churn", model_version="3", buffer_dir="/var/tmp/dw-buffer", sample_rate=0.5)

event_id = client.log_prediction(
    features={"age": 41, "plan": "pro"},
    prediction=1,
    score=0.83,
    embedding=[0.12, -0.4, 0.9],
    latency_ms=4.2,
)
client.log_label(event_id, 1)
client.close()
```

See `src/driftwatch/client/README.md` for queueing, retry and sampling behaviour.

### Registering a model and a reference

```python
from driftwatch.client.admin import AdminClient

admin = AdminClient("http://127.0.0.1:8000")
admin.register_model("churn", task_type="classification", config={"window_seconds": 3600})
admin.upload_reference("churn", rows)
```

Each reference row is `{"features": {...}, "prediction": ..., "score": ..., "embedding": [...], "label": ...}`. Score, embedding and label are optional; performance checks need labels and, for AUC and calibration, scores.

### CLI

```
driftwatch serve                       run the collector
driftwatch scheduler                   run monitors and alerts
driftwatch db upgrade                  apply Alembic migrations (PostgreSQL)
driftwatch models list
driftwatch models register NAME
driftwatch models compact NAME         merge small Parquet files in local mode
driftwatch monitor run NAME [--now T]  evaluate one window
driftwatch monitor backfill NAME       evaluate every window over the stored events
driftwatch alerts list [--json]
driftwatch demo run [--scenario S]
driftwatch bench detection | overhead
```

### Demo scenarios

| Scenario | What changes after the onset |
| --- | --- |
| `none` | nothing, used to measure false positives |
| `covariate_shift` | rows are resampled towards older, longer-hours, more educated people; P(y given x) is unchanged |
| `prior_shift` | positive class is resampled to about 48% instead of 24% |
| `concept_drift` | the label of 60% of married husbands and wives is flipped; features are untouched |
| `null_burst` | `hours-per-week` and `occupation` become null in 60% of rows, as if an upstream join broke |

### Alert rules

Rules live in a YAML file, see `deploy/README.md`. By default a drift result at `high` or above, a quality result at `medium` or above and a performance result at `medium` or above each raise an alert, with a one hour cooldown per rule.

## API reference

All routes except `/healthz` and `/metrics` need `X-API-Key` when `DRIFTWATCH_API_KEY` is set.

| Method and path | Purpose |
| --- | --- |
| `GET /healthz` | liveness |
| `GET /metrics` | Prometheus exposition |
| `POST /v1/events` | `{"events": [...]}` up to 5000 per request, returns 202 and `{"accepted": n}`. Unknown models are registered with defaults |
| `POST /v1/labels` | `{"labels": [{"event_id", "model", "label", "timestamp"}]}` |
| `POST /v1/models` | register or update a model: `name`, `task_type`, `positive_label`, `config` |
| `GET /v1/models` | models with event counts and the status of the latest window |
| `GET /v1/models/{name}` | one model record |
| `PUT /v1/models/{name}/reference` | upload the reference rows, returns the profile summary |
| `POST /v1/models/{name}/monitor?window_end=` | evaluate one window now |
| `POST /v1/models/{name}/backfill?start=&end=&step_seconds=` | evaluate many windows |
| `GET /v1/models/{name}/metrics` | stored results, filter by `start`, `end`, `kind`, `feature`, `metric` |
| `GET /v1/models/{name}/latest` | results of the most recent window |
| `GET /v1/models/{name}/heatmap` | window by feature matrix of the highest severity |
| `GET /v1/models/{name}/performance` | performance series |
| `GET /v1/models/{name}/distribution?feature=&window_end=` | reference and current histograms |
| `GET /v1/alerts?model=&limit=` | alert history |

Event body:

```json
{
  "event_id": "7c1f0e5a",
  "model": "churn",
  "model_version": "3",
  "timestamp": "2026-03-01T10:00:00Z",
  "features": {"age": 41, "plan": "pro"},
  "prediction": 1,
  "score": 0.83,
  "embedding": [0.12, -0.4, 0.9],
  "latency_ms": 4.2
}
```

Prometheus series exported by the collector: `driftwatch_drift_score`, `driftwatch_drift_flagged`, `driftwatch_quality_value`, `driftwatch_performance`, `driftwatch_model_severity`, `driftwatch_last_window_timestamp_seconds`, `driftwatch_events_ingested_total`, `driftwatch_labels_ingested_total`, `driftwatch_ingest_batch_seconds`. The scheduler adds `driftwatch_monitor_runs_total`, `driftwatch_monitor_run_seconds` and `driftwatch_alerts_raised_total`.

## Project layout

```
src/driftwatch/
  client/        SDK: queue, sender thread, disk buffer, sampling, admin helper
  collector/     FastAPI app and dashboard views
  engine/        detectors, quality checks, performance, profiles, window evaluation
  store/         Store interface, SQLAlchemy store, DuckDB and Parquet store
  alerting/      rules, manager, Slack and email notifiers
  migrations/    Alembic environment and versions
  demo/          Adult dataset, model, drift scenarios, replay and producer
  bench/         detection and overhead benchmarks
  monitor.py     reference handling, window runs, backfill
  scheduler.py   APScheduler jobs
  exporter.py    Prometheus collector over the store
  cli.py         Typer commands
ui/              static dashboard
deploy/          prometheus, nginx, grafana, alert rules
results/         raw benchmark output
tests/
```

## Testing

```
pip install -e ".[server,dev]"
pytest
```

The suite needs no network, API keys or running services. Statistical code is tested against scipy and scikit-learn reference values (KS statistic and p-value, Wasserstein distance, chi-square, Jensen-Shannon, AUC, F1, calibration curve, RMSE) and against loop-based reference implementations (PSI, MMD). The store contract tests run against SQLite and DuckDB; the collector, monitor, scheduler and CLI tests run on both. Notifiers are tested with `httpx.MockTransport` and a fake SMTP class. `tests/test_demo_end_to_end.py` starts a real collector in a thread and sends traffic through the SDK.

`make compile` byte-compiles the package and syntax-checks the UI scripts with Node.
