"""Tests credibility functions."""

import pandas as pd
from pytest import approx

from morai.experience import credibility
from morai.utils import helpers

test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"
cred_df = pd.read_csv(test_experience_path / "simple_credibility.csv")


def test_limited_fluctuation():
    """Checks limited fluctuation function."""
    partial = credibility.limited_fluctuation(
        df=pd.DataFrame([{"lapses": 100}]), measure="lapses", p=0.9, r=0.05, sd=1, u=1
    )
    full = credibility.limited_fluctuation(
        df=pd.DataFrame([{"lapses": 1082}]),
        measure="lapses",
        p=0.9,
        r=0.05,
        sd=1,
        u=1,
    )
    assert partial.iloc[0]["credibility_lf"] == approx(0.3040, abs=1e-4)
    assert full.iloc[0]["credibility_lf"] == approx(1.00, abs=1e-2)


def test_asymptotic():
    """Checks asymptotic credibility."""
    partial = credibility.asymptotic(
        df=pd.DataFrame([{"lapses": 100}]), measure="lapses", k=270
    )
    test_partial = 100 / (100 + 270)
    assert partial.iloc[0]["credibility_as"] == test_partial


def test_vm20_buhlmann():
    """Checks vm20 buhlmann credibility."""
    partial = credibility.vm20_buhlmann(
        df=cred_df, amount_col="amount", rate_col="rate", exposure_col="exposure"
    )
    assert partial.iloc[0]["credibility_vm20"] == approx(0.1682, abs=1e-4)


def test_vm20_buhlmann_approx():
    """Checks vm20 buhlmann approximation credibility."""
    # calculate the moments
    cred_df["a"] = cred_df["amount"] * cred_df["exposure"] * cred_df["rate"]
    cred_df["b"] = cred_df["amount"] ** 2 * cred_df["exposure"] * cred_df["rate"]
    cred_df["c"] = cred_df["amount"] ** 2 * cred_df["exposure"] * cred_df["rate"] ** 2

    # calculate the credibility
    partial = credibility.vm20_buhlmann_approx(
        df=cred_df, a_col="a", b_col="b", c_col="c"
    )

    assert partial.iloc[0]["credibility_vm20_approx"] == approx(0.1682, abs=1e-4)


def test_buhlmann():
    """Checks buhlmann credibility."""
    partial = credibility.buhlmann(
        df=pd.DataFrame([{"lapses": 100}]), measure="lapses", k=270
    )
    test_partial = 100 / (100 + 270)
    assert partial.iloc[0]["credibility_bu"] == test_partial
