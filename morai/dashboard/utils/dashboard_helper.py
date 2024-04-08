"""
Utilities for app.

Provides functions for common used functions in app.
"""

import os

import dash_bootstrap_components as dbc
import yaml
from dash import dcc, html

from morai.utils import helpers


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


def generate_filters(df):
    """
    Generate a dictionary of dashboard options from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to generate dropdown options.

    Returns
    -------
    filter_dict : dict
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
        filter_dict = {"filters": filters, "str_cols": str_cols, "num_cols": num_cols}

    return filter_dict


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


def generate_selectors(
    df,
    filter_dict,
    config,
):
    """
    Generate selectors for tool.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to generate selectors.
    filter_dict : dict
        Dictionary for the filters
    config : dict
        Configuration dictionary.

    Returns
    -------
    list
        List of selectors.

    """
    config_dataset = config["datasets"][config["general"]["dataset"]]
    selectors = [
        # selectors
        html.H5(
            "Selectors",
            style={
                "border-bottom": "1px solid black",
                "padding-bottom": "5px",
            },
        ),
        html.Div(
            id={"type": "selector-group", "index": "x_axis"},
            children=[
                html.Label("X-Axis"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "x_axis_selector"},
                    options=config_dataset["columns"]["features"],
                    value=config["defaults"]["x_axis"],
                    clearable=False,
                    placeholder="Select X-Axis",
                ),
            ],
        ),
        html.Div(
            id={"type": "selector-group", "index": "y_axis"},
            children=[
                html.Label("Y-Axis"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "y_axis_selector"},
                    options=["ratio", "risk"] + config_dataset["columns"]["measures"],
                    value=config["defaults"]["y_axis"],
                    clearable=False,
                    placeholder="Select Y-Axis",
                ),
            ],
        ),
        html.Div(
            id={"type": "selector-group", "index": "color"},
            children=[
                html.Label("Color"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "color_selector"},
                    options=config_dataset["columns"]["features"],
                    value=None,
                    placeholder="Select Color",
                ),
            ],
        ),
        html.Div(
            id={"type": "selector-group", "index": "chart_type"},
            children=[
                html.Label("Chart Type"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "chart_type_selector"},
                    options=["line", "bar", "heatmap"],
                    value="line",
                    placeholder="Select Chart Type",
                ),
            ],
        ),
        # additional options
        html.H5("Additional Options"),
        html.Div(
            id={"type": "selector-group", "index": "numerator"},
            children=[
                html.Label("Numerator"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "numerator_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=config["defaults"]["numerator"],
                    placeholder="Select Numerator",
                ),
            ],
        ),
        html.Div(
            id={"type": "selector-group", "index": "denominator"},
            children=[
                html.Label("Denominator"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "denominator_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=config["defaults"]["denominator"],
                    placeholder="Select Denominator",
                ),
            ],
        ),
        html.Div(
            id={"type": "selector-group", "index": "secondary"},
            children=[
                html.Label("Secondary"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "secondary_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=None,
                    placeholder="Select Secondary",
                ),
            ],
        ),
        html.Div(
            id={"type": "selector-group", "index": "x_bins"},
            children=[
                html.Label("X Bins"),
                dbc.Input(
                    id={"type": "selector", "index": "x_bins_selector"},
                    type="number",
                    value=None,
                    placeholder="Input X Bins",
                ),
            ],
        ),
        html.Div(
            id={"type": "selector-group", "index": "add_line"},
            children=[
                html.Label("Y=1 Line"),
                dbc.Checkbox(
                    id={"type": "selector", "index": "add_line_selector"},
                    value=False,
                    label="select on/off",
                ),
            ],
        ),
        # hidden selectors
        html.Div(
            id={"type": "selector-group", "index": "rates"},
            children=[
                html.Label("Rates"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "rates_selector"},
                    options=config_dataset["columns"]["rates"],
                    value=config["defaults"]["rates"],
                    clearable=False,
                    multi=True,
                    placeholder="Select Rates",
                ),
            ],
            style={"display": "none"},
        ),
        html.Div(
            id={"type": "selector-group", "index": "weights"},
            children=[
                html.Label("Weights"),
                dcc.Dropdown(
                    id={"type": "selector", "index": "weights_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=config["defaults"]["weights"],
                    multi=True,
                    placeholder="Select Weights",
                ),
            ],
            style={"display": "none"},
        ),
    ]

    return selectors


def _inputs_flatten_list(input_list):
    flat_list = []
    for item in input_list:
        if isinstance(item, dict):
            flat_list.append(item)
        elif isinstance(item, list):
            flat_list.extend(_inputs_flatten_list(item))
    return flat_list


def _inputs_parse_id(input_list, id_value):
    """
    Parse inputs for id value.

    Parameters
    ----------
    input_list : list
        List of inputs.
    id_value : str
        ID value to parse.

    Returns
    -------
    str
        Value of id value.

    """
    for input in input_list:
        input_id = input.get("id")
        # id is a dict
        if isinstance(input_id, dict):
            if input_id.get("index") == id_value:
                return input.get("value", None)
        elif isinstance(input_id, str):
            if input_id == id_value:
                return input.get("value", None)
    return None


def _inputs_parse_type(input_list, type_value):
    """
    Parse inputs for type value.

    Parameters
    ----------
    input_list : list
        List of inputs.
    type_value : str
        type to parse.

    Returns
    -------
    list
        List of inputs with type value.

    """
    type_list = []
    for input in input_list:
        input_id = input.get("id")
        # id is a dict
        if isinstance(input_id, dict):
            if input_id.get("type") == type_value:
                type_list.append(input)
    return type_list


def load_config(config_path=helpers.CONFIG_PATH):
    """
    Load the yaml configuration file.

    Parameters
    ----------
    config_path : str
        Path to the yaml configuration file.

    Returns
    -------
    config : dict
        Configuration dictionary.

    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def write_config(config, config_path=helpers.CONFIG_PATH):
    """
    Write the yaml configuration file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    config_path : str
        Path to the yaml configuration file.


    """
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)


def list_files(folder_path):
    """
    List files in the directory.

    Parameters
    ----------
    folder_path : str
        Path to the folder.

    Returns
    -------
    files : list
        List of files in the folder.

    """
    # List all entries in the given folder
    all_entries = os.listdir(folder_path)
    # Filter out entries that are files
    files = [
        entry
        for entry in all_entries
        if os.path.isfile(os.path.join(folder_path, entry))
    ]
    return files
