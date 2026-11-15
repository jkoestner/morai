"""
HMD Data.

This module contains functions to work with HMD data. The HMD data is a
collection of data from the Human Mortality Database.

Link: https://www.mortality.org/Home/Index

"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

from morai.utils import config_helper, custom_logger, helpers, sql
from morai.utils.custom_logger import suppress_logs

logger = custom_logger.setup_logging(__name__)

_FRENCH_MONTHS = {
    "janv": 1,
    "janvier": 1,
    "fevr": 2,
    "fevrier": 2,
    "mars": 3,
    "avr": 4,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "juillet": 7,
    "aout": 8,
    "sept": 9,
    "septembre": 9,
    "oct": 10,
    "octobre": 10,
    "nov": 11,
    "novembre": 11,
    "dec": 12,
    "decembre": 12,
}


def get_hmd_dataframe(
    email: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:  # pragma: no cover
    """
    Get HMD population, exposure, and deaths data.

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
        "https://www.mortality.org/File/GetDocument/hmd.v6/USA/STATS/Exposures_1x1.txt"
    )
    population_url = (
        "https://www.mortality.org/File/GetDocument/hmd.v6/USA/STATS/Population.txt"
    )

    # getting the data
    deaths = get_hmd_data(email=email, password=password, url=death_url, clean_df=True)
    deaths.rename(
        columns={
            "female": "female_death",
            "male": "male_death",
            "total": "total_death",
            "last_modified": "last_modified_death",
        },
        inplace=True,
    )
    exposures = get_hmd_data(
        email=email, password=password, url=exposure_url, clean_df=True
    )
    exposures.rename(
        columns={
            "female": "female_exposure",
            "male": "male_exposure",
            "total": "total_exposure",
            "last_modified": "last_modified_exposure",
        },
        inplace=True,
    )
    population = get_hmd_data(
        email=email, password=password, url=population_url, clean_df=True
    )
    population.rename(
        columns={
            "female": "female_population",
            "male": "male_population",
            "total": "total_population",
            "last_modified": "last_modified_population",
        },
        inplace=True,
    )
    # merge the dataframe
    hmd_df = pd.merge(
        deaths,
        exposures,
        on=["year", "age"],
        how="outer",
        suffixes=("_death", "_exposure"),
    )
    hmd_df = pd.merge(
        hmd_df,
        population,
        on=["year", "age"],
        how="outer",
        suffixes=("", "_population"),
    )
    hmd_df["country"] = "USA"

    return hmd_df


def get_hmd_data(
    email: str | None = None,
    password: str | None = None,
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
            raise RuntimeError(
                f"Login failed: invalid credentials at `{login_url}`. "
                f"Use config_helper.HMD_EMAIL and config_helper.HMD_PASSWORD "
                f"to check your credentials."
            )

        # get the data
        logger.info(f"requesting data from HMD: {url}")
        response = s.get(url)
        response.raise_for_status()
        response.encoding = "utf-8"
        txt = response.text

        last_modified = _parse_hmd_date(txt)

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
        # keeps 1959+
        if "year" in hmd_df:
            yr = hmd_df["year"].astype(str)
            hmd_df = hmd_df[~yr.str.endswith("-")]
            hmd_df["year"] = (
                yr[~yr.str.endswith("-")].str.extract(r"(\d{4})")[0].astype(int)
            )

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


def get_last_updated(table_name: str | None = None) -> dict:
    """
    Get the last updated meta of a table.

    Parameters
    ----------
    table_name : str, optional
        The name of the table to check. If None, defaults to "hmd".

    Returns
    -------
    last_update_meta : dict
        A dictionary containing:
        - "data_through": The maximum "data_through" date from the table.

    """
    if table_name is None:
        table_name = "hmd"
    db_filepath = helpers.FILES_PATH / "integrations" / "hmd" / "hmd.sql"

    # query the table
    query = f"SELECT MAX(last_modified_death) as data_through FROM {table_name}"
    try:
        result = sql.read_sql(db_filepath, query)
    except Exception:
        logger.error(
            f"Error querying table `{table_name}` for last updated date: {db_filepath}"
        )
        return {
            "data_through": None,
        }
    data_through = result["data_through"].iloc[0]

    # create the dict
    result = {
        "data_through": data_through,
    }

    return result


def _parse_hmd_date(txt: str) -> datetime | None:
    """
    Parse the 'Last modified' date from an HMD header (French or English).

    Parameters
    ----------
    txt : str
        The text content of the HMD file (usually the header).

    Returns
    -------
    datetime | None
        The parsed date as a datetime object, or None if parsing fails.

    """
    match = re.search(r"Last modified:\s*(\d{1,2})\s+([^\s\d;]+)\s+(\d{4})", txt)
    if not match:
        return None
    day, month, year = match.groups()

    # strip accents and trailing periods: "févr." -> "fevr"
    month_key = (
        unicodedata.normalize("NFKD", month)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .rstrip(".")
    )
    month_num = _FRENCH_MONTHS.get(month_key)
    if month_num is not None:
        return datetime(int(year), month_num, int(day))

    # fall back to english names/abbreviations
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(f"{day} {month.rstrip('.')} {year}", fmt)
        except ValueError:
            continue
    return None
