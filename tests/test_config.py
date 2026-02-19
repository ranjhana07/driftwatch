from __future__ import annotations

from driftwatch.config import Settings, make_store
from driftwatch.store.duck import DuckStore
from driftwatch.store.sql import SqlStore


def test_defaults_to_local_duckdb(tmp_path):
    settings = Settings(local_dir=tmp_path / "store", database_url=None)
    store = make_store(settings)
    assert isinstance(store, DuckStore)
    store.close()


def test_database_url_selects_sql_store():
    store = make_store(Settings(database_url="sqlite://"))
    assert isinstance(store, SqlStore)
    store.close()


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_PORT", "9001")
    monkeypatch.setenv("DRIFTWATCH_API_KEY", "secret")
    settings = Settings()
    assert settings.port == 9001
    assert settings.api_key == "secret"
