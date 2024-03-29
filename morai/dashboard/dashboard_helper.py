"""
Utilities for app.

Provides functions for common used functions in app.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def convert_to_short_number(number):
    """
    Convert number to short number.

    Parameters
    ----------
    number : float
        Number to convert to short number.

    Returns
    -------
    str
        Short number.

    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(number) < 1000:
            return f"{number:.1f}{unit}"
        number /= 1000
    return f"{number:.1f}T"


def generate_dash_dict(df):
    """
    Generate a dictionary of dashboard options from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to generate dropdown options.

    Returns
    -------
    dash_dict : dict
        Dictionary for the dashboard including following keys:
            - filters: list of filters
            - str_cols: list of string columns
            - num_cols: list of numeric columns

    """
    filters = []
    str_cols = []
    num_cols = []
    for col in df.columns:
        if isinstance(df[col][0], str):
            filter = dcc.Dropdown(
                id={"type": "str-filter", "index": col},
                options=[{"label": i, "value": i} for i in sorted(df[col].unique())],
                multi=True,
                placeholder=f"Select {col}",
            )
            str_cols.append(col)
        elif df[col].nunique() < 10:
            filter = dcc.Dropdown(
                id={"type": "str-filter", "index": col},
                options=[{"label": i, "value": i} for i in sorted(df[col].unique())],
                multi=True,
                placeholder=f"Select {col}",
            )
            str_cols.append(col)
        else:
            filter = dcc.RangeSlider(
                id={"type": "num-filter", "index": col},
                min=df[col].min(),
                max=df[col].max(),
                marks=None,
                value=[df[col].min(), df[col].max()],
                tooltip={"always_visible": True, "placement": "bottom"},
            )
            num_cols.append(col)
        filters.append(html.Label(col))
        filters.append(filter)
        dash_dict = {"filters": filters, "str_cols": str_cols, "num_cols": num_cols}

    return dash_dict


def generate_card(df, card_list, title="Data", color="Azure", inverse=False, **kwargs):
    """
    Generate cards.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to gather KPI from.
    card_list : list
        List of cards to generate.
    title : str
        Title of the card.
    color : str
        Color of the card.
    inverse : bool
        Inverse color of text.
    **kwargs
        Additional arguments for card.

    Returns
    -------
    cards : list
        List of cards.

    """
    card_body_content = []

    for column in card_list:
        value_text = f"{column}: {convert_to_short_number(df[column].sum())}"
        card_body_content.append(html.P(value_text, className="card-text"))

    card = dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(card_body_content)],
        color=color,
        inverse=inverse,
        style={"width": "auto"},
        **kwargs,
    )

    return card


def generate_selectors(df, dash_dict, actuals_col, exposure_col, tool="chart"):
    """
    Generate selectors for tool.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to generate selectors.
    dash_dict : dict
        Dictionary for the dashboard
    actuals_col : str
        Actuals column.
    exposure_col : str
        Exposure column.
    tool : str
        Tool name.

    Returns
    -------
    list
        List of selectors.

    """
    print(f"updating selectors for {tool}")
    if tool == "chart":
        selectors = [
            # selectors
            html.H5("Selectors"),
            html.Label("X-Axis"),
            dcc.Dropdown(
                id="x_axis_selector",
                options=df.columns,
                value="observation_year",
                clearable=False,
                placeholder="Select X-Axis",
            ),
            html.Label("Y-Axis"),
            dcc.Dropdown(
                id="y_axis_selector",
                options=["ratio", "risk"],
                value="ratio",
                clearable=False,
                placeholder="Select Y-Axis",
            ),
            html.Label("Color"),
            dcc.Dropdown(
                id="color_selector",
                options=df.columns,
                value=None,
                placeholder="Select Color",
            ),
            # additional options
            html.H5("Additional Options"),
            html.Label("Numerator"),
            dcc.Dropdown(
                id="numerator_selector",
                options=dash_dict["num_cols"],
                value=actuals_col,
                placeholder="Select Numerator",
            ),
            html.Label("Denominator"),
            dcc.Dropdown(
                id="denominator_selector",
                options=dash_dict["num_cols"],
                value=exposure_col,
                placeholder="Select Denominator",
            ),
            html.Label("X Bins"),
            dbc.Input(
                id="x_bins_selector",
                type="number",
                value=None,
                placeholder="Input X Bins",
            ),
            # unneeded selectors
            html.Label("Rates", style={"display": "none"}),
            dcc.Dropdown(
                id="rates_selector",
                options=dash_dict["num_cols"],
                value="ae_vbt15",
                clearable=False,
                placeholder="Select Rates",
                style={"display": "none"},
            ),
            html.Label("Weights", style={"display": "none"}),
            dcc.Dropdown(
                id="weights_selector",
                options=dash_dict["num_cols"],
                value=None,
                placeholder="Select Weights",
                style={"display": "none"},
            ),
            html.Label("Secondary", style={"display": "none"}),
            dcc.Dropdown(
                id="secondary_selector",
                options=dash_dict["num_cols"],
                value=None,
                placeholder="Select Secondary",
                style={"display": "none"},
            ),
        ]
    else:
        selectors = [
            # selectors
            html.H5("Selectors"),
            html.Label("X-Axis"),
            dcc.Dropdown(
                id="x_axis_selector",
                options=df.columns,
                value="observation_year",
                clearable=False,
                placeholder="Select X-Axis",
            ),
            html.Label("Rates"),
            dcc.Dropdown(
                id="rates_selector",
                options=dash_dict["num_cols"],
                value="ae_vbt15",
                clearable=False,
                placeholder="Select Rates",
            ),
            # additional options
            html.H5("Additional Options"),
            html.Label("Weights"),
            dcc.Dropdown(
                id="weights_selector",
                options=dash_dict["num_cols"],
                value=None,
                placeholder="Select Weights",
            ),
            html.Label("Secondary"),
            dcc.Dropdown(
                id="secondary_selector",
                options=dash_dict["num_cols"],
                value=None,
                placeholder="Select Secondary",
            ),
            html.Label("X Bins"),
            dbc.Input(
                id="x_bins_selector",
                type="number",
                value=None,
                placeholder="Input X Bins",
            ),
            # unneeded selectors
            html.Label("Y-Axis", style={"display": "none"}),
            dcc.Dropdown(
                id="y_axis_selector",
                options=["ratio", "risk"],
                value="ratio",
                clearable=False,
                placeholder="Select Y-Axis",
                style={"display": "none"},
            ),
            html.Label("Color", style={"display": "none"}),
            dcc.Dropdown(
                id="color_selector",
                options=df.columns,
                value=None,
                placeholder="Select Color",
                style={"display": "none"},
            ),
            html.Label("Numerator", style={"display": "none"}),
            dcc.Dropdown(
                id="numerator_selector",
                options=dash_dict["num_cols"],
                value=actuals_col,
                placeholder="Select Numerator",
                style={"display": "none"},
            ),
            html.Label("Denominator", style={"display": "none"}),
            dcc.Dropdown(
                id="denominator_selector",
                options=dash_dict["num_cols"],
                value=exposure_col,
                placeholder="Select Denominator",
                style={"display": "none"},
            ),
        ]

    return selectors
