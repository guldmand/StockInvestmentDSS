"""
Minimal DuckDB connection check.

Works from both:
- project root: python system/scripts/check_duckdb_connection.py
- system folder: python scripts/check_duckdb_connection.py

Path priority:
1. DUCKDB_PATH from environment, if set
2. Local fallback resolved from repository structure
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb


def find_project_root() -> Path:
    """
    Resolve the project root based on this script location.

    Expected location:
    StockInvestmentDSS/system/scripts/check_duckdb_connection.py
    """
    return Path(__file__).resolve().parents[2]


def get_duckdb_path() -> Path:
    """
    Return active DuckDB path.

    If DUCKDB_PATH is set:
      - absolute paths are used as-is
      - relative paths are resolved from the current working directory

    If DUCKDB_PATH is not set:
      - default to system/runtime-data/market_research.duckdb
        resolved from the project root
    """
    env_path = os.getenv("DUCKDB_PATH")

    if env_path:
        return Path(env_path).expanduser().resolve()

    project_root = find_project_root()
    return project_root / "system" / "runtime-data" / "market_research.duckdb"


def main() -> None:
    db_path = get_duckdb_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS connection_check (
                id INTEGER,
                status VARCHAR
            )
            """
        )
        con.execute("DELETE FROM connection_check")
        con.execute("INSERT INTO connection_check VALUES (1, 'ok')")
        result = con.execute(
            "SELECT status FROM connection_check WHERE id = 1"
        ).fetchone()

    print(f"DuckDB connection OK: {db_path}")
    print(f"Status: {result[0]}")


if __name__ == "__main__":
    main()
