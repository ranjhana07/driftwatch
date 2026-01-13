from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx


class AdminClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._http = httpx.Client(base_url=endpoint, headers=headers, timeout=timeout, transport=transport)

    def register_model(
        self,
        name: str,
        task_type: str = "classification",
        positive_label: Any = 1,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {"name": name, "task_type": task_type, "positive_label": positive_label, "config": config or {}}
        return self._json(self._http.post("/v1/models", json=body))

    def upload_reference(self, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._json(self._http.put(f"/v1/models/{model}/reference", json={"rows": rows}))

    def run_monitor(self, model: str, window_end: datetime | None = None) -> dict[str, Any]:
        params = {"window_end": window_end.isoformat()} if window_end else {}
        return self._json(self._http.post(f"/v1/models/{model}/monitor", params=params))

    def backfill(self, model: str, start: datetime, end: datetime, step_seconds: int) -> dict[str, Any]:
        params = {"start": start.isoformat(), "end": end.isoformat(), "step_seconds": step_seconds}
        return self._json(self._http.post(f"/v1/models/{model}/backfill", params=params))

    def list_models(self) -> list[dict[str, Any]]:
        return self._json(self._http.get("/v1/models"))

    def close(self) -> None:
        self._http.close()

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        response.raise_for_status()
        return response.json()
