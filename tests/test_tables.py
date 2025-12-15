"""Tests the experience."""

import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd

from morai import models
from morai.experience import tables
from morai.forecast import preprocessors
from morai.utils import helpers

test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"

# create a test simple GLM model
experience_df = pd.read_csv(test_experience_path / "simple_experience.csv")
feature_dict = {
    "target": ["rate"],
    "weight": [],
    "passthrough": [],
    "ordinal": [
        "sex",
    ],
    "ohe": ["smoker_status"],
    "nominal": [],
}
preprocess_dict = preprocessors.preprocess_data(
    experience_df,
    feature_dict=feature_dict,
    standardize=False,
    add_constant=True,
)
GLM = models.core.GLM()
GLM.fit(X=preprocess_dict["X"], y=preprocess_dict["y"], r_style=False)


def test_generate_table_full():
    """Tests the generation of a table with no multiple."""
    test_rate_table, test_mult_table = tables.generate_table(
        model=GLM.model,
        mapping=preprocess_dict["mapping"],
        preprocess_feature_dict=preprocess_dict["feature_dict"],
        preprocess_params=preprocess_dict["params"],
        grid=None,
        mult_features=[],
    )

    rate_table_vals = np.array([0.071788, 0.088212, 0.108212, 0.131788])

    npt.assert_allclose(test_rate_table["vals"], rate_table_vals, atol=1e-3)
    assert test_mult_table is None


def test_generate_table_mult_with_mean():
    """Tests the generation of a table with multiples using mean."""
    test_rate_table, test_mult_table = tables.generate_table(
        model=GLM.model,
        mapping=preprocess_dict["mapping"],
        preprocess_feature_dict=preprocess_dict["feature_dict"],
        preprocess_params=preprocess_dict["params"],
        grid=None,
        mult_features=[
            "smoker_status",
        ],
        mult_method="mean",
    )

    rate_table_vals = np.array([0.071788, 0.108212])
    mult_table_vals = np.array([1.000000, 1.222222])

    npt.assert_allclose(test_rate_table["vals"], rate_table_vals, atol=1e-3)
    npt.assert_allclose(test_mult_table["multiple"], mult_table_vals, atol=1e-3)


def test_generate_table_mult_with_glm():
    """Tests the generation of a table with multiples using glm."""
    test_rate_table, test_mult_table = tables.generate_table(
        model=GLM.model,
        mapping=preprocess_dict["mapping"],
        preprocess_feature_dict=preprocess_dict["feature_dict"],
        preprocess_params=preprocess_dict["params"],
        grid=None,
        mult_features=[
            "smoker_status",
        ],
        mult_method="glm",
    )

    rate_table_vals = np.array([0.071788, 0.108212])
    mult_table_vals = np.array([1.000000, 1.250931])

    npt.assert_allclose(test_rate_table["vals"], rate_table_vals, atol=1e-3)
    npt.assert_allclose(test_mult_table["multiple"], mult_table_vals, atol=1e-3)


def test_build_table_soa():
    """Checks the soa table builds."""
    t1683 = pd.read_csv(test_experience_path / "tables" / "t1683.csv", index_col=0)
    t1683_e = pd.read_csv(
        test_experience_path / "tables" / "t1683_extend.csv", index_col=0
    )
    vbt15 = pd.read_csv(test_experience_path / "tables" / "vbt15.csv", index_col=0)
    vbt15_e = pd.read_csv(
        test_experience_path / "tables" / "vbt15_extend.csv", index_col=0
    )
    vbt15_j = pd.read_csv(
        test_experience_path / "tables" / "vbt15_juv.csv", index_col=0
    )

    # create the MortTable object
    MortTable = tables.MortTable()

    # ultimate table
    extra_dims = None
    juv_list = None
    table_list = [1683]
    test_1683 = MortTable.build_table_soa(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=False
    )
    test_1683_e = MortTable.build_table_soa(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=True
    )

    # select and ultimate table with multiple dimensions
    extra_dims = {"gender": ["F", "M"], "underwriting": ["NS", "S"]}
    table_list = [3224, 3234, 3252, 3262]
    test_vbt15 = MortTable.build_table_soa(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=False
    )
    test_vbt15_e = MortTable.build_table_soa(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=True
    )
    juv_list = [3273, 3273, 3274, 3274]
    test_vbt15_j = MortTable.build_table_soa(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=False
    )

    assert test_1683.equals(t1683)
    assert test_1683_e.equals(t1683_e)
    assert test_vbt15.equals(vbt15)
    assert test_vbt15_e.equals(vbt15_e)
    assert test_vbt15_j.equals(vbt15_j)


def test_build_table_workbook():
    """Checks the workbook table builds."""
    file_location = test_experience_path / "tables" / "simple_glm.xlsx"
    # create the MortTable object
    mt = tables.MortTable()
    test_rate_table, test_mult_table = mt.build_table_workbook(
        file_location=file_location, has_mults=True
    )

    rate_table_vals = np.array([0.071788, 0.108212])
    mult_table_vals = np.array([1.000000, 1.222222])

    npt.assert_allclose(test_rate_table["vals"], rate_table_vals, atol=1e-3)
    npt.assert_allclose(test_mult_table["multiple"], mult_table_vals, atol=1e-3)


def test_apply_mi_to_rate_table():
    """Checks the apply_mi_to_rate_table method."""
    mt = tables.MortTable(rate="simple_glm_mi_col", rate_filename="rate_map_test.yaml")
    mi_rate_table = mt.apply_mi_to_rate_table(mi_years=2)

    npt.assert_allclose(
        mi_rate_table.iloc[0]["vals"],
        mt.rate_table.iloc[0]["vals"] * (1 - mt.mi_table.iloc[0]["mi"]) ** 2,
        atol=1e-3,
        err_msg="The rates should have 2 years of 1% mi applied.",
    )


def test_calc_derived_table():
    """Checks the derived table calculation."""
    mt = tables.MortTable(rate="graded_glm", rate_filename="rate_map_test.yaml")
    test_derived_table = mt.calc_derived_table_from_mults(
        selected_dict={"smoker_status": ["S"]}
    )
    rate_table = mt.rate_table

    npt.assert_allclose(
        test_derived_table["vals"],
        rate_table["vals"] * 1.22222,
        atol=1e-3,
        err_msg="The derived table should be the rate table times the multiple (1.22).",
    )


def test_map_rates_simple():
    """Tests the mapping of rates with a simple rate_table."""
    test_experience_df = tables.map_rates(
        df=experience_df,
        rate="simple_glm",
        rate_filename="rate_map_test.yaml",
    )

    npt.assert_allclose(
        test_experience_df["rate"], test_experience_df["qx_simple_glm"], atol=1e-3
    )


def test_map_rates_simple_mi_year():
    """Tests the mapping of rates with a simple rate_table with mi by year."""
    test_experience_df = tables.map_rates(
        df=experience_df,
        rate="simple_glm_mi_year",
        rate_filename="rate_map_test.yaml",
    )

    npt.assert_allclose(
        test_experience_df["rate"] * 0.99**1,
        test_experience_df["qx_simple_glm_mi_year"],
        atol=1e-3,
        err_msg="The rates should have 1 years of 1% mi applied.",
    )


def test_map_rates_simple_mi_col():
    """Tests the mapping of rates with a simple rate_table with mi by column."""
    test_experience_df = tables.map_rates(
        df=experience_df,
        rate="simple_glm_mi_col",
        rate_filename="rate_map_test.yaml",
    )

    npt.assert_allclose(
        test_experience_df["rate"] * 0.99**2,
        test_experience_df["qx_simple_glm_mi_col"],
        atol=1e-3,
        err_msg="The rates should have 2 years of 1% mi applied.",
    )


def test_map_rates_grade():
    """Tests the mapping of rates with a graded rate_table."""
    test_experience_df = tables.map_rates(
        df=experience_df,
        rate="graded_glm",
        rate_filename="rate_map_test.yaml",
    )

    npt.assert_allclose(
        test_experience_df.iloc[1]["qx_graded_glm"],
        0.088,
        atol=1e-3,
        err_msg="Female Smoker should not be graded and still have multiple",
    )
    npt.assert_allclose(
        test_experience_df.iloc[5]["qx_graded_glm"],
        0.072,
        atol=1e-3,
        err_msg="Female Smoker should be graded and not have multiple",
    )


def test_formula_grade():
    """Tests the formula grading evaluation."""
    experience_df["test_multiple"] = 1

    # value multiple
    formula = "multiple"
    value_mult = tables._formula_grade(
        df=experience_df,
        multiple=1.25,
        formula=formula,
    )
    npt.assert_equal(
        value_mult,
        1.25,
        err_msg="The multiple should be 1.25.",
    )

    # simple multiple
    formula = "multiple * 1.25"
    simple_mult = tables._formula_grade(
        df=experience_df,
        multiple="test_multiple",
        formula=formula,
    )
    npt.assert_equal(
        simple_mult.iloc[0],
        1.25,
        err_msg="The multiple should be simple multiple times 1.25.",
    )

    # graded multiple
    formula = "multiple * 1.25 * ['smoker_status_encode']"
    graded_mult = tables._formula_grade(
        df=experience_df,
        multiple="test_multiple",
        formula=formula,
    )
    npt.assert_equal(
        graded_mult.iloc[1],
        1.25,
        err_msg="The multiple should be 1.25 when smoker_status_encode is 1.",
    )
    npt.assert_equal(
        graded_mult.iloc[0],
        0,
        err_msg="The multiple should be 0 when smoker_status_encode is 0.",
    )


def test_compare_tables():
    """Tests the compare tables function."""
    mt = tables.MortTable()
    tbl1 = mt.build_table_soa(table_list=[3224])
    tbl1["extra_key_1"] = 1
    tbl2 = mt.build_table_soa(table_list=[3219])
    tbl2["extra_key_2"] = 2
    compare_tbl = tables.compare_tables(table_1=tbl1, table_2=tbl2)

    assert (
        compare_tbl[(compare_tbl["issue_age"] == 19) & (compare_tbl["duration"] == 1)][
            "ratio"
        ].iloc[0]
        == 1.5
    ), "The ratio should be 1.5 for issue age 19 and duration 1."

    assert (
        compare_tbl[
            (compare_tbl["issue_age"] == 19) & (compare_tbl["duration"] == 100)
        ]["ratio"].iloc[0]
        == 1.0
    ), "The ratio should be 1.0 for issue age 19 and duration 100."


def test_check_aa_ia_dur_cols():
    """Tests the check aa ia dur cols function."""
    # setting up a dataframe with invalid rows
    # starts at attained age 0
    # ends at attained age 130 (above limit of 121)
    # duration is also invalid for the second row
    df = pd.DataFrame({"issue_age": 1, "attained_age": range(131)})
    df["duration"] = df["attained_age"] - df["issue_age"] + 1
    df.loc[1, "duration"] = 2
    checked_df = tables.check_aa_ia_dur_cols(df=df, max_age=121)

    assert len(checked_df) == 120, "The checked dataframe should have 120 rows."


def test_add_aa_ia_dur_cols():
    """Tests the add aa ia dur cols function."""
    # all
    df_all = pd.DataFrame({"issue_age": 1, "attained_age": range(123)})
    df_all["duration"] = df_all["attained_age"] - df_all["issue_age"] + 1
    check_df_all = tables.add_aa_ia_dur_cols(df_all)
    assert check_df_all.shape == (121, 3)

    # 2 columns
    df_ia_dur = df_all[["issue_age", "duration"]]
    check_df_ia_dur = tables.add_aa_ia_dur_cols(df_ia_dur)
    pd.testing.assert_frame_equal(check_df_ia_dur, check_df_all, check_like=True)

    df_aa_dur = df_all[["attained_age", "duration"]]
    check_df_aa_dur = tables.add_aa_ia_dur_cols(df_aa_dur)
    pd.testing.assert_frame_equal(check_df_aa_dur, check_df_all, check_like=True)

    df_ia_aa = df_all[["issue_age", "attained_age"]]
    check_df_ia_aa = tables.add_aa_ia_dur_cols(df_ia_aa)
    pd.testing.assert_frame_equal(check_df_ia_aa, check_df_all, check_like=True)

    # 1 column
    expected_values = np.arange(1, 122)

    df_ia = df_all[["issue_age"]].drop_duplicates()
    check_df_ia = tables.add_aa_ia_dur_cols(df_ia)
    assert np.array_equal(check_df_ia["issue_age"].unique(), np.array([1]))
    assert np.array_equal(check_df_ia["attained_age"].unique(), expected_values)
    assert np.array_equal(check_df_ia["duration"].unique(), expected_values)
    assert check_df_ia.shape == (121, 3)

    df_aa = df_all[["attained_age"]].drop_duplicates()
    check_df_aa = tables.add_aa_ia_dur_cols(df_aa)
    assert np.array_equal(check_df_aa["issue_age"].unique(), expected_values)
    assert np.array_equal(check_df_aa["attained_age"].unique(), expected_values)
    assert np.array_equal(check_df_aa["duration"].unique(), expected_values)
    assert check_df_aa.shape == (7381, 3)

    df_dur = df_all[["duration"]].drop_duplicates()
    check_df_dur = tables.add_aa_ia_dur_cols(df_dur)
    assert np.array_equal(check_df_dur["issue_age"].unique(), expected_values)
    assert np.array_equal(check_df_dur["attained_age"].unique(), expected_values)
    assert np.array_equal(check_df_dur["duration"].unique(), expected_values)
    assert check_df_dur.shape == (7381, 3)


def test_add_ultimate():
    """Tests the add ultimate function."""
    select_table = pd.read_csv(test_experience_path / "tables" / "vbt15_select.csv")
    ultimate_table = pd.read_csv(test_experience_path / "tables" / "vbt15_ultimate.csv")
    extend_tbl = tables.add_ultimate(select_table, ultimate_table)

    # check issue age 18 for reasonableness
    npt.assert_almost_equal(
        extend_tbl.loc[(extend_tbl["issue_age"] == 18), "vals"].sum(),
        50.65668,
        decimal=5,
    )


def test_output_table_csv():
    """Tests the output table csv function."""
    rate_table = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mult_table = pd.DataFrame({"m": [0.1, 0.2]})

    # test csv
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test.csv"
        tables.output_table(rate_table=rate_table, filename=str(tmp_path))

        # check exists
        assert tmp_path.exists()

        # check contents
        read_df = pd.read_csv(tmp_path)
        pd.testing.assert_frame_equal(read_df, rate_table)

    # test xlsx
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test.xlsx"
        tables.output_table(
            rate_table=rate_table, filename=str(tmp_path), mult_table=mult_table
        )

        # check exists
        assert tmp_path.exists()

        # check contents
        read_df = pd.read_excel(tmp_path, sheet_name="rate_table")
        read_mult_df = pd.read_excel(tmp_path, sheet_name="mult_table")
        pd.testing.assert_frame_equal(read_df, rate_table)
        pd.testing.assert_frame_equal(read_mult_df, mult_table)


def test_get_su_table():
    """Tests the get su table function."""
    mt = tables.MortTable()
    tbl1 = mt.build_table_soa(table_list=[3224])
    tbl1 = tbl1.dropna()
    su_table = tables.get_su_table(tbl1, "max")

    npt.assert_almost_equal(
        su_table.loc[
            (su_table["issue_age"] == 50) & (su_table["duration"] == 1), "su_ratio"
        ].values[0],
        4.333,
        decimal=3,
    )
    npt.assert_almost_equal(
        su_table.loc[
            (su_table["issue_age"] == 50) & (su_table["duration"] == 21), "su_ratio"
        ].values[0],
        1.0,
        decimal=3,
    )


def test_get_rates():
    """Tests the get rates function."""
    rates = tables.get_rates("rate_map_test.yaml")
    check_rates = [
        "simple_glm",
        "graded_glm",
        "simple_glm_mi_year",
        "simple_glm_mi_col",
    ]
    assert set(check_rates).issubset(set(rates))
