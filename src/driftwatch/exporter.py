from __future__ import annotations

from typing import Iterable

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from driftwatch.store.base import Store


class StoreCollector(Collector):
    def __init__(self, store: Store) -> None:
        self.store = store

    def collect(self) -> Iterable[Metric]:
        drift = GaugeMetricFamily(
            "driftwatch_drift_score", "Latest drift statistic per feature and detector", labels=["model", "feature", "detector"]
        )
        flagged = GaugeMetricFamily(
            "driftwatch_drift_flagged", "1 when the detector flagged the feature in the latest window", labels=["model", "feature", "detector"]
        )
        quality = GaugeMetricFamily(
            "driftwatch_quality_value", "Latest data quality check value", labels=["model", "feature", "check"]
        )
        performance = GaugeMetricFamily(
            "driftwatch_performance", "Latest delayed-label performance metric", labels=["model", "metric"]
        )
        severity = GaugeMetricFamily(
            "driftwatch_model_severity", "Highest severity score in the latest window", labels=["model"]
        )
        last_window = GaugeMetricFamily(
            "driftwatch_last_window_timestamp_seconds", "End of the latest evaluated window", labels=["model"]
        )
        for model in self.store.list_models():
            latest = self.store.latest_metrics(model.name)
            if not latest:
                continue
            for result in latest:
                if result.kind == "drift":
                    drift.add_metric([model.name, result.feature, result.metric], result.value)
                    flagged.add_metric([model.name, result.feature, result.metric], 1.0 if result.flagged else 0.0)
                elif result.kind == "quality":
                    quality.add_metric([model.name, result.feature, result.metric], result.value)
                else:
                    performance.add_metric([model.name, result.metric], result.value)
            severity.add_metric([model.name], max(r.severity for r in latest))
            last_window.add_metric([model.name], max(r.window_end for r in latest).timestamp())
        yield from (drift, flagged, quality, performance, severity, last_window)


class IngestMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.events = Counter("driftwatch_events_ingested_total", "Prediction events accepted", ["model"], registry=registry)
        self.labels = Counter("driftwatch_labels_ingested_total", "Labels accepted", ["model"], registry=registry)
        self.batch_seconds = Histogram(
            "driftwatch_ingest_batch_seconds",
            "Time to persist one ingest batch",
            ["kind"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=registry,
        )


def build_registry(store: Store) -> tuple[CollectorRegistry, IngestMetrics]:
    registry = CollectorRegistry()
    registry.register(StoreCollector(store))
    return registry, IngestMetrics(registry)
