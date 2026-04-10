from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from driftwatch.client import Client
from driftwatch.client.admin import AdminClient
from driftwatch.demo.data import AdultSplit, load_adult, split_adult
from driftwatch.demo.model import DemoModel, holdout_auc, train_model
from driftwatch.demo.replay import reference_rows, replay_stream
from driftwatch.demo.scenarios import SCENARIOS, StreamPlan, build_stream
from driftwatch.timeutil import utcnow

MODEL_NAMES = {
    "none": "adult-control",
    "covariate_shift": "adult-covariate-shift",
    "prior_shift": "adult-prior-shift",
    "concept_drift": "adult-concept-drift",
    "null_burst": "adult-null-burst",
}


@dataclass
class DemoConfig:
    endpoint: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    scenarios: list[str] = field(default_factory=lambda: list(SCENARIOS))
    n_events: int = 9000
    onset: int = 4500
    ramp: int = 0
    seconds_per_event: float = 30.0
    label_lag_seconds: float = 3600.0
    window_seconds: int = 14400
    step_seconds: int = 7200
    reference_rows: int = 4000
    seed: int = 0
    cache_dir: Path = Path(".driftwatch/demo")
    end: datetime | None = None


def prepare(cache_dir: Path, seed: int = 0) -> tuple[DemoModel, AdultSplit]:
    features, labels = load_adult(cache_dir / "openml")
    split = split_adult(features, labels, seed=seed)
    path = cache_dir / f"adult-model-seed{seed}.joblib"
    if path.exists():
        return DemoModel.load(path), split
    model = train_model(split.train_x, split.train_y, seed=seed)
    model.save(path)
    return model, split


def run_scenario(
    config: DemoConfig,
    scenario: str,
    model: DemoModel,
    split: AdultSplit,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    name = MODEL_NAMES[scenario]
    span = timedelta(seconds=config.seconds_per_event * config.n_events)
    end = config.end or utcnow()
    start = end - span
    admin = AdminClient(config.endpoint, config.api_key)
    admin.register_model(
        name,
        config={"window_seconds": config.window_seconds, "interval_seconds": config.step_seconds // 4, "min_samples": 200},
    )
    reference = reference_rows(model, split.reference_x.head(config.reference_rows), split.reference_y.head(config.reference_rows))
    summary = admin.upload_reference(name, reference)
    log(f"{name}: reference uploaded ({summary['n_rows']} rows)")
    plan = StreamPlan(scenario=scenario, n_events=config.n_events, onset=config.onset, ramp=config.ramp, seed=config.seed + 1)
    stream = build_stream(split.pool_x, split.pool_y, plan)
    with Client(config.endpoint, name, api_key=config.api_key, batch_size=500, flush_interval=1.0) as client:
        sent = replay_stream(
            client, model, stream, start, config.seconds_per_event, config.label_lag_seconds, id_prefix=f"{name}-{config.seed}"
        )
        stats = client.stats
    log(f"{name}: replayed {sent} events (sent {stats.sent}, rejected {stats.rejected})")
    windows = admin.backfill(name, start, end + timedelta(seconds=config.label_lag_seconds), config.step_seconds)["windows"]
    log(f"{name}: evaluated {windows} windows")
    admin.close()
    return {"model": name, "events": sent, "windows": windows, "onset_at": (start + timedelta(seconds=config.seconds_per_event * config.onset)).isoformat()}


def run_demo(config: DemoConfig, log: Callable[[str], None] = print) -> list[dict[str, object]]:
    model, split = prepare(config.cache_dir, config.seed)
    log(f"model trained, reference auc {holdout_auc(model, split.reference_x, split.reference_y):.3f}")
    return [run_scenario(config, scenario, model, split, log) for scenario in config.scenarios]
