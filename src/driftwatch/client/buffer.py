from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


class DiskBuffer:
    def __init__(self, directory: str | Path, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.directory = Path(directory)
        self.max_bytes = max_bytes
        self.directory.mkdir(parents=True, exist_ok=True)

    def put(self, kind: str, items: list[dict[str, Any]]) -> Path:
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
        target = self.directory / name
        staging = self.directory / (name + ".tmp")
        staging.write_text(json.dumps({"kind": kind, "items": items}), encoding="utf-8")
        os.replace(staging, target)
        self._enforce_limit()
        return target

    def pending(self) -> list[Path]:
        return sorted(self.directory.glob("*.json"))

    def read(self, path: Path) -> tuple[str, list[dict[str, Any]]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["kind"], payload["items"]

    def remove(self, path: Path) -> None:
        path.unlink(missing_ok=True)

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.pending())

    def _enforce_limit(self) -> None:
        files = self.pending()
        total = sum(p.stat().st_size for p in files)
        while total > self.max_bytes and len(files) > 1:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
