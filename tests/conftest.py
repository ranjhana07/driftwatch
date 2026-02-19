from __future__ import annotations

import pytest

from driftwatch.store.duck import DuckStore
from driftwatch.store.sql import SqlStore


@pytest.fixture(params=["sql", "duck"])
def store(request, tmp_path):
    if request.param == "sql":
        instance = SqlStore("sqlite://", create_schema=True)
    else:
        instance = DuckStore(tmp_path / "duck")
    yield instance
    instance.close()
