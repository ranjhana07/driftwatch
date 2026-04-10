from __future__ import annotations

from datetime import timedelta

from synthetic import T0, make_rows, to_events, to_labels

from driftwatch.store.duck import DuckStore


def test_events_are_written_as_parquet_parts(tmp_path):
    store = DuckStore(tmp_path)
    store.add_events(to_events(make_rows(20)))
    store.add_events(to_events(make_rows(20, seed=1)))
    parts = list((tmp_path / "events" / "churn").glob("*.parquet"))
    assert len(parts) == 2
    store.close()


def test_data_survives_reopen(tmp_path):
    store = DuckStore(tmp_path)
    rows = make_rows(30)
    store.add_events(to_events(rows))
    store.add_labels(to_labels(rows))
    store.save_reference("churn", {"profile": {"n_rows": 30}})
    store.close()

    reopened = DuckStore(tmp_path)
    assert reopened.event_count("churn") == 30
    assert len(reopened.read_labeled_events("churn", T0, T0 + timedelta(days=1))) == 30
    assert reopened.load_reference("churn")["profile"]["n_rows"] == 30
    reopened.close()


def test_missing_embeddings_and_scores_round_trip(tmp_path):
    store = DuckStore(tmp_path)
    rows = make_rows(5)
    for row in rows:
        row["embedding"] = None
        row["score"] = None
    store.add_events(to_events(rows))
    read = store.read_events("churn", T0, T0 + timedelta(days=1))
    assert all(r["embedding"] is None and r["score"] is None for r in read)
    store.close()


def test_events_for_different_models_go_to_separate_directories(tmp_path):
    store = DuckStore(tmp_path)
    store.add_events(to_events(make_rows(5, model="a") + make_rows(5, model="b", seed=2)))
    assert (tmp_path / "events" / "a").is_dir()
    assert (tmp_path / "events" / "b").is_dir()
    store.close()


def test_compaction_merges_parts_and_keeps_rows(tmp_path):
    store = DuckStore(tmp_path)
    rows = make_rows(120, spacing_seconds=60)
    for i in range(0, 120, 20):
        store.add_events(to_events(rows[i : i + 20]))
    store.add_labels(to_labels(rows[:60]))
    store.add_labels(to_labels(rows[60:]))
    before = store.read_events("churn", T0, T0 + timedelta(days=1))
    merged = store.compact("churn")
    assert merged == 8
    assert len(list((tmp_path / "events" / "churn").glob("*.parquet"))) == 1
    assert len(list((tmp_path / "labels" / "churn").glob("*.parquet"))) == 1
    assert store.read_events("churn", T0, T0 + timedelta(days=1)) == before
    assert len(store.read_labeled_events("churn", T0, T0 + timedelta(days=1))) == 120
    store.close()


def test_compaction_drops_duplicate_events_and_splits_large_outputs(tmp_path):
    store = DuckStore(tmp_path)
    rows = make_rows(50)
    store.add_events(to_events(rows))
    store.add_events(to_events(rows))
    store.compact("churn", rows_per_file=20)
    assert store.event_count("churn") == 50
    assert len(list((tmp_path / "events" / "churn").glob("*.parquet"))) == 3
    assert store.compact("nothing") == 0
    store.close()
