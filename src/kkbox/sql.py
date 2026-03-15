from __future__ import annotations

from pathlib import Path
import logging

import duckdb

logger = logging.getLogger(__name__)


def sql_path(path: str | Path) -> str:
    """Return DuckDB-friendly absolute path with forward slashes."""
    return str(Path(path).resolve()).replace("\\", "/")


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = con.execute(
        f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = '{table_name}'
        """
    ).fetchone()
    return bool(result[0])


def run_stage(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    sql: str,
    reuse_existing: bool = True,
) -> None:
    """Create/refresh a table if needed."""
    if reuse_existing and table_exists(con, table_name):
        logger.info("Reusing %s...", table_name)
        return
    if not reuse_existing and table_exists(con, table_name):
        logger.info("Dropping existing %s (reuse_existing=False)...", table_name)
        con.execute(f"DROP TABLE {table_name};")
    logger.info("Building %s...", table_name)
    con.execute(sql)


def configure_connection(duckdb_path: Path, temp_dir: Path) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with tuned defaults for KKBox builds."""
    con = duckdb.connect(str(duckdb_path))
    con.execute("SET threads = 1;")
    con.execute("SET preserve_insertion_order = false;")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}';")
    con.execute("SET max_temp_directory_size = '250GiB';")
    con.execute("SET memory_limit = '6GB';")
    con.execute("PRAGMA enable_progress_bar;")
    return con
