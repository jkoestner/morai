"""Tests the experience."""

import pandas as pd

from xact.experience import tables
from xact.utils import helpers

test_table_path = helpers.ROOT_PATH / "tests" / "files" / "tables"


def test_table_build():
    """Checks the soa table builds."""
    t1683 = pd.read_csv(test_table_path / "t1683.csv", index_col=0)
    t1683_e = pd.read_csv(test_table_path / "t1683_extend.csv", index_col=0)
    vbt15 = pd.read_csv(test_table_path / "vbt15.csv", index_col=0)
    vbt15_e = pd.read_csv(test_table_path / "vbt15_extend.csv", index_col=0)

    # ultimate table
    extra_dims = None
    table_list = [1683]
    test_1683 = tables.build_table(
        table_list=table_list, extra_dims=extra_dims, extend=False
    )
    test_1683_e = tables.build_table(
        table_list=table_list, extra_dims=extra_dims, extend=True
    )

    # select and ultimate table with multiple dimensions
    extra_dims = {"gender": ["F", "M"], "underwriting": ["NS", "S"]}
    table_list = [3224, 3234, 3252, 3262]
    test_vbt15 = tables.build_table(
        table_list=table_list, extra_dims=extra_dims, extend=False
    )
    test_vbt15_e = tables.build_table(
        table_list=table_list, extra_dims=extra_dims, extend=True
    )

    assert test_1683.equals(t1683)
    assert test_1683_e.equals(t1683_e)
    assert test_vbt15.equals(vbt15)
    assert test_vbt15_e.equals(vbt15_e)
