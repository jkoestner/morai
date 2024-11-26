"""SQL utilities."""

import sqlite3
from datetime import datetime

import pandas as pd

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


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


def read_sql(db_filepath: str, query: str) -> pd.DataFrame:
    """
    Read a SQLite database.

    Parameters
    ----------
    db_filepath : str
        Database file path.
    query : str
        Query to execute.

    Returns
    -------
    df : pd.DataFrame
        DataFrame object.

    example:
    df = read_sql("data.db", "SELECT * FROM table")

    """
    # connect to the database
    conn = sqlite3.connect(db_filepath)

    # read the data
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


def get_tables(db_filepath: str) -> list:
    """
    Get the tables from a SQLite database.

    Parameters
    ----------
    db_filepath : str
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


def table_dtypes(db_filepath: str, table_name: str) -> dict:
    """
    Get the data types of a table from a SQLite database.

    Parameters
    ----------
    db_filepath : str
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
