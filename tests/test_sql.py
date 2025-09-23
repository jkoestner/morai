"""Tests the sql utils."""

import pandas as pd

from morai.utils import helpers, sql

test_utils_path = helpers.ROOT_PATH / "tests" / "files" / "utils"


def test_sql_get_tables():
    """Tests the get_tables function."""
    tables = sql.get_tables(db_filepath=test_utils_path / "hmd.sql")
    assert "hmd" in tables


def test_sql_read_table():
    """Tests the read_table function."""
    hmd = sql.read_sql(
        db_filepath=test_utils_path / "hmd.sql", query="SELECT * FROM hmd"
    )
    assert hmd.shape == (10212, 12)


def test_sql_get_dtypes():
    """Tests the get_dtypes function."""
    dtypes = sql.table_dtypes(db_filepath=test_utils_path / "hmd.sql", table_name="hmd")
    assert dtypes == {
        "year": "INTEGER",
        "age": "INTEGER",
        "female_death": "REAL",
        "male_death": "REAL",
        "total_death": "REAL",
        "last_modified_death": "TEXT",
        "female_exposure": "REAL",
        "male_exposure": "REAL",
        "total_exposure": "REAL",
        "last_modified_exposure": "TEXT",
        "country": "TEXT",
        "added_at": "TEXT",
    }


def test_sql_table_export():
    """Tests the export_table function."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    sql.export_to_sql(
        df=df,
        db_filepath=test_utils_path / "hmd.sql",
        table_name="test",
        if_exists="replace",
    )
    tables = sql.get_tables(test_utils_path / "hmd.sql")
    assert "test" in tables


def test_sql_table_remove():
    """Tests the table_remove function."""
    sql.table_remove(db_filepath=test_utils_path / "hmd.sql", table_name="test")
    tables = sql.get_tables(test_utils_path / "hmd.sql")
    assert "test" not in tables
