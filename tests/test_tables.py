"""Tests the experience."""

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
        "smoker_status",
    ],
    "ohe": [],
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


def test_generate_table_mult():
    """Tests the generation of a table with multiples."""
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
