from __future__ import annotations

from datetime import timedelta

import pytest
from synthetic import T0, make_rows, to_events, to_labels
from typer.testing import CliRunner

from driftwatch.cli import app
from driftwatch.config import Settings
from driftwatch.monitor import Monitor, reference_rows
from driftwatch.store.duck import DuckStore

runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_LOCAL_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("DRIFTWATCH_DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "store"


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_register_and_list_models(env):
    result = runner.invoke(app, ["models", "register", "churn", "--window-seconds", "1800"])
    assert result.exit_code == 0
    listing = runner.invoke(app, ["models", "list"])
    assert "churn" in listing.stdout
    assert "no events" in listing.stdout


def test_db_upgrade_needs_a_database_url(env):
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 1


def test_db_upgrade_applies_migrations(env, tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_DATABASE_URL", f"sqlite:///{(tmp_path / 'cli.db').as_posix()}")
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0
    assert "head" in result.stdout


def test_monitor_run_and_alerts_flow(env):
    runner.invoke(app, ["models", "register", "churn"])
    store = DuckStore(env)
    Monitor(store).set_reference("churn", reference_rows(make_rows(1500, seed=800, start=T0 - timedelta(days=1))))
    rows = make_rows(450, seed=1, spacing_seconds=8, age_shift=12)
    store.add_events(to_events(rows))
    store.add_labels(to_labels(rows))
    store.close()
    result = runner.invoke(app, ["monitor", "run", "churn", "--now", (T0 + timedelta(hours=1)).isoformat()])
    assert result.exit_code == 0, result.stdout
    assert "flagged" in result.stdout
    assert "age" in result.stdout


def test_monitor_run_reports_missing_reference(env):
    runner.invoke(app, ["models", "register", "churn"])
    result = runner.invoke(app, ["monitor", "run", "churn"])
    assert result.exit_code == 1


def test_backfill_uses_event_bounds(env):
    runner.invoke(app, ["models", "register", "churn"])
    store = DuckStore(env)
    Monitor(store).set_reference("churn", reference_rows(make_rows(1500, seed=801, start=T0 - timedelta(days=1))))
    store.add_events(to_events(make_rows(1000, seed=2, spacing_seconds=8)))
    store.close()
    result = runner.invoke(app, ["monitor", "backfill", "churn", "--step-seconds", "1800"])
    assert result.exit_code == 0, result.stdout
    assert "evaluated" in result.stdout


def test_alerts_list_empty_json(env):
    result = runner.invoke(app, ["alerts", "list", "--json"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "[]"


def test_settings_roundtrip_from_env(env):
    assert Settings().local_dir == env
