from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from driftwatch.store.base import Store


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DRIFTWATCH_", env_file=".env", extra="ignore")

    database_url: str | None = None
    local_dir: Path = Path(".driftwatch/store")
    api_key: str | None = None
    ui_dir: Path | None = None
    cors_origins: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    alert_rules_path: Path | None = None
    slack_webhook_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "driftwatch@localhost"
    smtp_to: str | None = None
    scheduler_metrics_port: int = 9102
    sync_interval_seconds: int = 60
    compact_interval_seconds: int = 21600


def make_store(settings: Settings) -> Store:
    if settings.database_url:
        from driftwatch.store.sql import SqlStore

        return SqlStore(settings.database_url)
    from driftwatch.store.duck import DuckStore

    return DuckStore(settings.local_dir)
