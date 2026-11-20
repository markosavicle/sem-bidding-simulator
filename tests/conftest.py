"""Shared fixtures: a small synthetic dataset written to DuckDB + SQLite once per session."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_generator as dg  # noqa: E402


@pytest.fixture(scope="session")
def small_tables():
    return dg.generate(n_days=42, seed=123, keyword_scale=0.5)


@pytest.fixture(scope="session")
def small_dbs(small_tables, tmp_path_factory):
    import duckdb

    d = tmp_path_factory.mktemp("db")
    duck, lite = d / "sem.duckdb", d / "sem.sqlite"
    dg.write_duckdb(small_tables, duck)
    dg.write_sqlite(small_tables, lite)
    dcon = duckdb.connect(str(duck), read_only=True)
    scon = sqlite3.connect(lite)
    yield {"duckdb": dcon, "sqlite": scon, "duck_path": duck, "sqlite_path": lite}
    dcon.close()
    scon.close()
