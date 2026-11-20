"""Run the analytical SQL files in ``sql/`` against DuckDB or SQLite.

    from src.sql_runner import connect, run_query, list_queries
    con = connect("data/sem.duckdb")
    df = run_query("05_wasted_spend_search_terms", con)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def list_queries() -> list[str]:
    return sorted(p.stem for p in SQL_DIR.glob("*.sql"))


def load_sql(name: str) -> str:
    """Read one of the bundled queries. Only names from ``list_queries()`` are
    accepted, so a caller can never point this at an arbitrary file
    (e.g. ``../../etc/passwd``) or run SQL that is not in the repository."""
    stem = name[:-4] if name.endswith(".sql") else name
    if stem not in list_queries():
        raise ValueError(f"unknown query {name!r}; available: {', '.join(list_queries())}")
    return (SQL_DIR / f"{stem}.sql").read_text(encoding="utf-8")


def connect(path: str | Path, read_only: bool = True):
    """DuckDB for *.duckdb, sqlite3 for *.sqlite / *.db. Read-only by default
    for both engines (analytics never writes)."""
    path = Path(path)
    if path.suffix == ".duckdb":
        import duckdb

        return duckdb.connect(str(path), read_only=read_only)
    if read_only:
        if not path.exists():  # sqlite3 would otherwise silently create an empty file
            raise FileNotFoundError(path)
        return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    return sqlite3.connect(path)


def run_query(name: str, con) -> pd.DataFrame:
    sql = load_sql(name).strip().rstrip(";")
    if isinstance(con, sqlite3.Connection):
        return pd.read_sql_query(sql, con)
    df = con.execute(sql).df()
    # DuckDB returns HUGEINT sums as float/object; DATE as datetime -> normalise
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df


def run_all(con) -> dict[str, pd.DataFrame]:
    return {name: run_query(name, con) for name in list_queries()}
