from __future__ import annotations

import pytest

from driftwatch.store.sql import SqlStore


@pytest.fixture(params=["sql"])
def store(request, tmp_path):
    instance = SqlStore("sqlite://", create_schema=True)
    yield instance
    instance.close()
