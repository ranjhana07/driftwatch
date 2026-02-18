from __future__ import annotations

from sqlalchemy import create_engine, inspect

from driftwatch.migrate import downgrade, upgrade
from driftwatch.store.tables import Base


def test_upgrade_creates_every_mapped_table(tmp_path):
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade(url)
    inspector = inspect(create_engine(url))
    tables = set(inspector.get_table_names())
    assert set(Base.metadata.tables) <= tables
    for name, table in Base.metadata.tables.items():
        assert {c["name"] for c in inspector.get_columns(name)} == {c.name for c in table.columns}


def test_indexes_match_models(tmp_path):
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade(url)
    inspector = inspect(create_engine(url))
    for name, table in Base.metadata.tables.items():
        expected = {i.name for i in table.indexes}
        assert expected <= {i["name"] for i in inspector.get_indexes(name)}


def test_downgrade_removes_tables(tmp_path):
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade(url)
    downgrade(url)
    remaining = set(inspect(create_engine(url)).get_table_names())
    assert remaining <= {"alembic_version"}
