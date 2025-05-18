"""
HMD Data.

This module contains functions to work with HMD data. The HMD data is a
collection of data from the Human Mortality Database.

Link: https://www.mortality.org/Home/Index

"""

import re
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from morai.utils import config_helper, custom_logger, helpers, sql
from morai.utils.custom_logger import suppress_logs

logger = custom_logger.setup_logging(__name__)


def get_hmd_data(
    email: Optional[str] = None,
    password: Optional[str] = None,
    url: str = "https://www.mortality.org/File/GetDocument/hmd.v6/USA/STATS/Deaths_1x1.txt",
    clean_df: bool = True,
) -> pd.DataFrame:
    """
    Get HMD data from server.

    A user does need to have an HMD account to access.
    https://www.mortality.org/Account/Auth

    Parameters
    ----------
    email : str, optional
       The HMD email.
    password : str, optional
       The HMD password.
    url : str, optional
       The URL to get the data from.
    clean_df : bool, optional
       Clean the DataFrame.

    Returns
    -------
    hmd_df : pd.DataFrame
        The HMD data.

    """
    # if email or password are none use config
    if email is None or password is None:
        email = config_helper.HMD_EMAIL
        password = config_helper.HMD_PASSWORD

    # create a new session
    login_url = "https://www.mortality.org/Account/Login"
    with requests.Session() as s:
        soup = BeautifulSoup(s.get(login_url).text, "html.parser")
        token = soup.find("input", {"name": "__RequestVerificationToken"})["value"]

        # https://www.mortality.org/Account/UserAgreement
        s.post(
            login_url,
            data={
                "Email": email,
                "Password": password,
                "ReturnUrl": "",
                "__RequestVerificationToken": token,
            },
            headers={"Referer": login_url},
        )

        if "Logout" not in s.get(login_url).text:
            raise RuntimeError(f"Login failed: invalid credentials at `{login_url}`.")

        # get the data
        logger.info(f"requesting data from HMD: {url}")
        response = s.get(url)
        response.raise_for_status()
        txt = response.text

        match = re.search(r"Last modified:\s*(\d{1,2} \w{3} \d{4})", txt)
        last_modified = match.group(1) if match else None
        last_modified = pd.to_datetime(last_modified, format="%d %b %Y")

    # create df
    data_str = "\n".join(txt.splitlines())
    hmd_df = pd.read_csv(
        StringIO(data_str),
        skiprows=2,
        sep=r"\s+",
        na_values=["."],
    )
    hmd_df["last_modified"] = last_modified

    # clean the dataframe
    if clean_df:
        hmd_df = suppress_logs(helpers.clean_df)(hmd_df, update_cat=False)
        hmd_df["age"] = hmd_df["age"].str.replace("110+", "110").astype(int)

        # the US data has duplicate years in population (e.g. 1959+ and 1959-)
        hmd_df["year"] = (
            hmd_df["year"].astype(str).str.extract(r"(\d{4})").astype(float).astype(int)
        )
        hmd_df = hmd_df.drop_duplicates(subset=["year", "age"], keep="first")

    return hmd_df


def get_hmd_exposure_and_deaths(
    email: Optional[str] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get HMD exposure and deaths data.

    Parameters
    ----------
    email : str, optional
       The HMD email.
    password : str, optional
       The HMD password.

    Returns
    -------
    hmd_df : pd.DataFrame

    """
    # data links
    death_url = (
        "https://www.mortality.org/File/GetDocument/hmd.v6/USA/STATS/Deaths_1x1.txt"
    )
    exposure_url = (
        "https://www.mortality.org/File/GetDocument/hmd.v6/USA/STATS/Population.txt"
    )

    # getting the data
    deaths = get_hmd_data(email=email, password=password, url=death_url, clean_df=True)
    exposures = get_hmd_data(
        email=email, password=password, url=exposure_url, clean_df=True
    )

    # merge the dataframe
    hmd_df = pd.merge(
        deaths,
        exposures,
        on=["year", "age"],
        how="outer",
        suffixes=("_death", "_exposure"),
    )
    hmd_df["country"] = "USA"

    return hmd_df


def get_hmd_data_sql(db_filepath: str, table_name: str) -> pd.DataFrame:
    """
    Get HMD data from a SQLite database.

    Parameters
    ----------
    db_filepath : str
        Database file path.
    table_name : str
        Table name.

    Returns
    -------
    hmd_df : pd.DataFrame
        DataFrame object.

    """
    # read the data from the database
    query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY added_at
            DESC LIMIT 1
            """
    last_date_row = sql.read_sql(db_filepath, query)
    last_date = last_date_row["added_at"].iloc[0]

    query = f"""
            SELECT *
            FROM {table_name}
            """
    hmd_df = sql.read_sql(db_filepath, query)

    logger.debug(
        f"table `{table_name}` last updated at: {last_date}, rows: {len(hmd_df)}"
    )

    return hmd_df
