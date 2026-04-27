"""SQL utilities."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

if TYPE_CHECKING:
    from pathlib import Path


def export_to_sql(
    df: pd.DataFrame,
    db_filepath: str,
    table_name: str,
    if_exists: str = "append",
    index: bool = False,
) -> None:
    """
    Export a DataFrame to a SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object.
    db_filepath : str
        Database file path.
    table_name : str
        Table name.
    if_exists : str, optional (default='append')
        What to do if the table already exists.
        Options are 'fail', 'replace', 'append'.
    index : bool, optional (default=False)
        Whether to include the DataFrame index.

    """
    # initialize
    logger.info(
        f"{if_exists} data to SQLite database: "
        f"`{db_filepath}` and table: `{table_name}`"
    )

    # connect to the database
    conn = sqlite3.connect(db_filepath)

    # create the table if it doesn't exist
    df = df.copy()
    df["added_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=index)
    finally:
        conn.close()


def read_sql(
    db_filepath: str | Path, query: str, parse_dates: list | None = None
) -> pd.DataFrame:
    """
    Read a SQLite database.

    Parameters
    ----------
    db_filepath : str | Path
        Database file path.
    query : str
        Query to execute.
    parse_dates : list | None, optional
        List of column names to parse as dates.

    Returns
    -------
    df : pd.DataFrame
        DataFrame object.

    example:
    df = read_sql("data.db", "SELECT * FROM table")

    """
    # connect to the database
    conn = sqlite3.connect(db_filepath)

    # get table_name
    match = re.compile(
        r"\bFROM\s+[\"`\[]?(\w+)[\"`\]]?",
        re.IGNORECASE,
    ).search(query)
    table_name = match.group(1) if match else None

    # read the data
    try:
        # infer parse_dates if not provided
        if parse_dates is None and table_name is not None:
            dtypes = table_dtypes(db_filepath, table_name)
            parse_dates = [
                col
                for col, dtype in dtypes.items()
                if "DATE" in dtype.upper() or "TIME" in dtype.upper()
            ]
        df = pd.read_sql_query(query, conn, parse_dates=parse_dates or None)
    finally:
        conn.close()

    return df


def get_tables(db_filepath: str | Path) -> list:
    """
    Get the tables from a SQLite database.

    Parameters
    ----------
    db_filepath : str | Path
        Database file path.

    Returns
    -------
    tables : list
        List of tables.

    """
    # connect to the database
    conn = sqlite3.connect(db_filepath)

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        tables = [table[0] for table in tables]
    finally:
        conn.close()

    return tables


def table_remove(db_filepath: str, table_name: str) -> None:
    """
    Remove a table from a SQLite database.

    Parameters
    ----------
    db_filepath : str
        Database file path.
    table_name : str
        Table name.

    """
    # connect to the database
    conn = sqlite3.connect(db_filepath)

    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        logger.info(f"table `{table_name}` removed from `{db_filepath}`")
    finally:
        conn.close()


def table_dtypes(db_filepath: str | Path, table_name: str) -> dict:
    """
    Get the data types of a table from a SQLite database.

    Parameters
    ----------
    db_filepath : str | Path
        Database file path.
    table_name : str
        Table name.

    Returns
    -------
    dtypes : dict
        Dictionary of column names and data types.

    """
    # connect to the database
    conn = sqlite3.connect(db_filepath)

    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        dtypes = {col[1]: col[2] for col in columns}
    finally:
        conn.close()

    return dtypes
