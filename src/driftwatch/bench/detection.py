from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from driftwatch.bench.environment import describe
from driftwatch.demo.data import FEATURES
from driftwatch.demo.producer import prepare
from driftwatch.demo.replay import feature_dict, reference_rows
from driftwatch.demo.scenarios import SCENARIOS, StreamPlan, build_stream
from driftwatch.engine.evaluate import evaluate_window
from driftwatch.engine.profile import build_profile
from driftwatch.engine.window import OUTPUT_PREDICTION, OUTPUT_SCORE, Window
from driftwatch.schemas import MonitorConfig
from driftwatch.types import MetricResult

DETECTORS = ["psi", "ks", "wasserstein", "chi2", "js", "mmd", "domain_classifier", "null_rate", "performance", "any"]
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass
class DetectionConfig:
    scenarios: list[str] = field(default_factory=lambda: list(SCENARIOS))
    seeds: int = 5
    n_events: int = 9000
    onset: int = 4500
    seconds_per_event: float = 30.0
    window_seconds: int = 14400
    step_seconds: int = 7200
    label_lag_seconds: float = 3600.0
    reference_rows: int = 4000
    cache_dir: Path = Path(".driftwatch/demo")
    out_dir: Path = Path("results")


def detector_flags(results: list[MetricResult]) -> dict[str, bool]:
    flags = {name: False for name in DETECTORS}
    for result in results:
        if not result.flagged:
            continue
        flags["any"] = True
        if result.kind == "drift" and result.metric in flags:
            flags[result.metric] = True
        elif result.kind == "quality" and result.metric == "null_rate":
            flags["null_rate"] = True
        elif result.kind == "performance":
            flags["performance"] = True
    return flags


def stream_window(
    frame: pd.DataFrame,
    raw: list[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    index: np.ndarray,
) -> Window:
    return Window(
        frame.iloc[index].reset_index(drop=True),
        embeddings[index],
        pd.Series(labels[index]),
        [raw[i] for i in index],
    )


def run_stream(
    config: DetectionConfig,
    scenario: str,
    seed: int,
    model,
    split,
    reference: Window,
    profile,
    monitor_config: MonitorConfig,
) -> list[dict]:
    plan = StreamPlan(scenario=scenario, n_events=config.n_events, onset=config.onset, seed=seed + 1)
    stream = build_stream(split.pool_x, split.pool_y, plan)
    predictions, scores = model.predict(stream)
    embeddings = model.embed(stream)
    raw = [feature_dict(r) for r in stream[FEATURES].to_dict("records")]
    frame = pd.DataFrame(raw)
    frame[OUTPUT_PREDICTION] = predictions
    frame[OUTPUT_SCORE] = scores
    labels = stream["label"].to_numpy()
    times = np.arange(config.n_events) * config.seconds_per_event
    label_times = times + config.label_lag_seconds
    onset_time = config.onset * config.seconds_per_event
    rows = []
    end = float(config.window_seconds)
    horizon = float(times[-1]) + config.seconds_per_event
    while end <= horizon:
        start = end - config.window_seconds
        inside = np.where((times >= start) & (times < end))[0]
        labeled = inside[label_times[inside] <= end]
        current = stream_window(frame, raw, embeddings, labels, inside)
        labeled_window = stream_window(frame, raw, embeddings, labels, labeled)
        started = time.perf_counter()
        results = evaluate_window(
            "bench",
            profile,
            reference,
            current,
            labeled_window,
            monitor_config,
            EPOCH + timedelta(seconds=start),
            EPOCH + timedelta(seconds=end),
            seed=seed,
        )
        elapsed = time.perf_counter() - started
        if scenario == "none" or end <= onset_time:
            phase = "pre"
        elif start >= onset_time:
            phase = "post"
        else:
            phase = "mixed"
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "window_end_s": end,
                "hours_after_onset": (end - onset_time) / 3600.0,
                "phase": phase,
                "n_current": int(len(inside)),
                "n_labeled": int(len(labeled)),
                "eval_seconds": elapsed,
                **detector_flags(results),
            }
        )
        end += config.step_seconds
    return rows


def summarize(windows: pd.DataFrame, onset_hours: float) -> pd.DataFrame:
    rows = []
    for scenario, group in windows.groupby("scenario", sort=False):
        runs = group.groupby("seed")
        for detector in DETECTORS:
            negative = group[group["phase"] == "pre"]
            positive = group[group["phase"] == "post"]
            delays = []
            for _, run in runs:
                after = run[(run["hours_after_onset"] > 0) & run[detector]]
                if scenario != "none" and len(after):
                    delays.append(float(after["hours_after_onset"].min()))
            rows.append(
                {
                    "scenario": scenario,
                    "detector": detector,
                    "negative_windows": len(negative),
                    "false_positive_rate": float(negative[detector].mean()) if len(negative) else np.nan,
                    "positive_windows": len(positive),
                    "true_positive_rate": float(positive[detector].mean()) if len(positive) else np.nan,
                    "runs": int(group["seed"].nunique()),
                    "runs_detected": len(delays),
                    "median_delay_hours": float(np.median(delays)) if delays else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_detection(config: DetectionConfig, log: Callable[[str], None] = print) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, split = prepare(config.cache_dir)
    reference = Window.from_rows(
        reference_rows(model, split.reference_x.head(config.reference_rows), split.reference_y.head(config.reference_rows))
    )
    profile = build_profile(reference)
    monitor_config = MonitorConfig(window_seconds=config.window_seconds)
    rows: list[dict] = []
    for scenario in config.scenarios:
        for seed in range(config.seeds):
            started = time.perf_counter()
            rows += run_stream(config, scenario, seed, model, split, reference, profile, monitor_config)
            log(f"{scenario} seed {seed}: {time.perf_counter() - started:.1f}s")
    windows = pd.DataFrame(rows)
    summary = summarize(windows, config.onset * config.seconds_per_event / 3600.0)
    return windows, summary


def write_results(config: DetectionConfig, windows: pd.DataFrame, summary: pd.DataFrame) -> None:
    out = config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out / "detection_windows.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "detection_summary.csv", index=False, float_format="%.4g")
    meta = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
        "monitor_config": MonitorConfig(window_seconds=config.window_seconds).model_dump(),
        "mean_eval_seconds_per_window": float(windows["eval_seconds"].mean()),
        "windows": int(len(windows)),
        "environment": describe(),
    }
    (out / "detection_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
