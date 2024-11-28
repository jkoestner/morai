"""
Utilities for app.

Provides functions for common used functions in app.
"""

import os
from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import polars as pl
import yaml
from dash import dcc, html

from morai.utils import helpers

num_to_str_count = 10


def convert_to_short_number(number: float) -> str:
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


def read_table(filepath: str) -> pl.DataFrame:
    """
    Read table from file.

    Parameters
    ----------
    filepath : str
        Path to the file.
    **kwargs
        Additional arguments for reading the file.

    Returns
    -------
    df : pd.DataFrame
        Dataframe from the file.

    """
    print(type(filepath))
    print(filepath)
    if filepath.suffix == ".csv":
        df = pl.read_csv(filepath)
    elif filepath.suffix == ".xlsx":
        df = pl.read_excel(filepath, sheet_name="rate_table")
    else:
        raise ValueError(f"File type not supported: {filepath}")
    return df


def filter_data(
    df: pl.DataFrame,
    callback_context: list[dict],
    num_to_str_count: int = num_to_str_count,
) -> pl.DataFrame:
    """
    Filter data based on the number of unique values.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe to filter.
    callback_context : list
        List of callback context.
    num_to_str_count : int
        Number of unique values to convert to string.

    Returns
    -------
    filtered_df : pl.DataFrame
        Filtered dataframe.

    """
    filtered_df = df
    str_cols = []
    num_cols = []
    for col in filtered_df.columns:
        if isinstance(filtered_df[col][0], str):
            str_cols.append(col)
        elif filtered_df[col].nunique() < num_to_str_count:
            str_cols.append(col)
        else:
            num_cols.append(col)

    # filter string columns
    for col in str_cols:
        str_values = _inputs_parse_id(callback_context, col)
        if str_values:
            filtered_df = filtered_df[filtered_df[col].isin(str_values)]

    # filter numeric columns
    for col in num_cols:
        num_values = _inputs_parse_id(callback_context, col)
        if num_values:
            filtered_df = filtered_df[
                (filtered_df[col] >= num_values[0])
                & (filtered_df[col] <= num_values[1])
            ]
    return filtered_df


def generate_filters(
    df: pl.DataFrame,
    prefix: str,
    num_to_str_count: int = num_to_str_count,
    config: Optional[Dict[str, Any]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> dict:
    """
    Generate a dictionary of dashboard options from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to generate dropdown options.
    prefix : str
        Prefix for the dropdown options.
    num_to_str_count : int
        Number of unique values to convert to string.
    config : dict
        Configuration dictionary.
    exclude_cols : list
        List of columns to exclude from the dropdown options.

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
    prefix_str_filter = f"{prefix}-str-filter"
    prefix_num_filter = f"{prefix}-num-filter"

    columns = list(df.columns)
    if config:
        config_dataset = config["datasets"][config["general"]["dataset"]]
        config_columns = config_dataset["columns"]["features"]
        columns = [col for col in columns if col in config_columns]

    if exclude_cols:
        columns = [col for col in columns if col not in exclude_cols]

    if not columns:
        return {}

    columns = sorted(columns)

    # create filters
    for col in columns:
        if isinstance(df[col][0], str) or df[col].nunique() < num_to_str_count:
            # Create collapsible checklist for string columns
            filter = html.Div(
                [
                    dbc.Button(
                        [
                            html.Span(col, style={"flex-grow": 1}),
                            html.I(className="fas fa-chevron-down"),
                        ],
                        id={"type": f"{prefix}-collapse-button", "index": col},
                        className="mb-2 w-100 text-start d-flex align-items-center",
                        color="light",
                    ),
                    dbc.Collapse(
                        dcc.Checklist(
                            id={"type": prefix_str_filter, "index": col},
                            options=[
                                {"label": str(i), "value": i}
                                for i in sorted(df[col].unique())
                            ],
                            value=[],
                            className="ms-2",
                            labelStyle={"display": "block"},
                        ),
                        id={"type": f"{prefix}-collapse", "index": col},
                        is_open=False,
                    ),
                ],
                className="mb-3",
            )
            str_cols.append(col)
        else:
            filter = html.Div(
                [
                    dbc.Button(
                        [
                            html.Span(col, style={"flex-grow": 1}),
                            html.I(className="fas fa-chevron-down"),
                        ],
                        id={"type": f"{prefix}-collapse-button", "index": col},
                        className="mb-2 w-100 text-start d-flex align-items-center",
                        color="light",
                    ),
                    dbc.Collapse(
                        dcc.RangeSlider(
                            id={"type": prefix_num_filter, "index": col},
                            min=df[col].min(),
                            max=df[col].max(),
                            marks=None,
                            value=[df[col].min(), df[col].max()],
                            tooltip={"always_visible": True, "placement": "bottom"},
                        ),
                        id={"type": f"{prefix}-collapse", "index": col},
                        is_open=False,
                    ),
                ],
                className="mb-3",
            )
            num_cols.append(col)
        filters.append(filter)

    filter_dict = {"filters": filters, "str_cols": str_cols, "num_cols": num_cols}
    return filter_dict


def get_card_list(config: Dict[str, Any]) -> List[str]:
    """
    Get list of variables to display in cards.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    card_list : list
        List of cards.

    """
    config_dataset = config["datasets"][config["general"]["dataset"]]
    card_list = [
        config_dataset["columns"]["exposure_cnt"],
        config_dataset["columns"]["actuals_cnt"],
        config_dataset["columns"]["exposure_amt"],
        config_dataset["columns"]["actuals_amt"],
    ]
    return card_list


def generate_card(
    df: pl.DataFrame,
    card_list: List[str],
    title: str = "Data",
    color: str = "Azure",
    inverse: bool = False,
    **kwargs,
) -> dbc.Card:
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
    config: Dict[str, Any],
    prefix: str,
    selector_dict: Dict[str, bool],
) -> List[html.Div]:
    """
    Generate selectors.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    prefix : str
        Prefix to add to the selectors.
    selector_dict : dict
        Dictionary of selectors with the key as the selector and value
        as the value for displaying the selector.

    Returns
    -------
    selectors : list
        List of selectors.

    """
    config_dataset = config["datasets"][config["general"]["dataset"]]
    prefix_group = f"{prefix}-selector-group"
    prefix_selector = f"{prefix}-selector"

    # html elements
    all_selectors = [
        html.Div(
            id={"type": prefix_group, "index": "model_file"},
            children=[
                html.Label("Model"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "model_file_selector"},
                    options=[
                        {"label": key, "value": key}
                        for key in list_files_in_folder(helpers.FILES_PATH / "models")
                        if key.endswith(".joblib")
                    ],
                    placeholder="Select a Model",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("model_file") == True
            else {"display": "none"},
        ),
        # selectors
        html.H5(
            "Selectors",
            style={
                "border-bottom": "1px solid black",
                "padding-bottom": "5px",
            },
        ),
        html.Div(
            id={"type": prefix_group, "index": "x_axis"},
            children=[
                html.Label("X-Axis"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "x_axis_selector"},
                    options=config_dataset["columns"]["features"],
                    value=config_dataset["defaults"]["x_axis"],
                    clearable=False,
                    placeholder="Select X-Axis",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("x_axis") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "y_axis"},
            children=[
                html.Label("Y-Axis"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "y_axis_selector"},
                    options=["ratio", "risk"] + config_dataset["columns"]["measures"],
                    value=config_dataset["defaults"]["y_axis"],
                    clearable=False,
                    placeholder="Select Y-Axis",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("color") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "color"},
            children=[
                html.Label("Color"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "color_selector"},
                    options=config_dataset["columns"]["features"],
                    value=None,
                    placeholder="Select Color",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("color") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "chart_type"},
            children=[
                html.Label("Chart Type"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "chart_type_selector"},
                    options=["line", "bar", "heatmap"],
                    value="line",
                    placeholder="Select Chart Type",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("color") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "target"},
            children=[
                html.Label("Target"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "target_selector"},
                    options=["ratio", "risk"] + config_dataset["columns"]["measures"],
                    value=config_dataset["defaults"]["y_axis"],
                    clearable=False,
                    placeholder="Select Target",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("target") == True
            else {"display": "none"},
        ),
        # additional options
        html.H5("Additional Options"),
        html.Div(
            id={"type": prefix_group, "index": "numerator"},
            children=[
                html.Label("Numerator"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "numerator_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=config_dataset["defaults"]["numerator"],
                    placeholder="Select Numerator",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("numerator") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "denominator"},
            children=[
                html.Label("Denominator"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "denominator_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=config_dataset["defaults"]["denominator"],
                    placeholder="Select Denominator",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("denominator") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "secondary"},
            children=[
                html.Label("Secondary"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "secondary_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=None,
                    placeholder="Select Secondary",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("secondary") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "x_bins"},
            children=[
                html.Label("X Bins"),
                dbc.Input(
                    id={"type": prefix_selector, "index": "x_bins_selector"},
                    type="number",
                    value=None,
                    placeholder="Input X Bins",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("x_bins") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "add_line"},
            children=[
                html.Label("Y=1 Line"),
                dbc.Checkbox(
                    id={"type": prefix_selector, "index": "add_line_selector"},
                    value=False,
                    label="select on/off",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("add_line") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "rates"},
            children=[
                html.Label("Rates"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "rates_selector"},
                    options=config_dataset["columns"]["rates"],
                    value=config_dataset["defaults"]["rates"],
                    clearable=False,
                    multi=True,
                    placeholder="Select Rates",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("rates") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "weights"},
            children=[
                html.Label("Weights"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "weights_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=config_dataset["defaults"]["weights"],
                    multi=True,
                    placeholder="Select Weights",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("weights") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "pdp_weight"},
            children=[
                html.Label("PDP Weight"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "pdp_weight_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=config_dataset["defaults"]["denominator"],
                    placeholder="Select Weight",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("pdp_weight") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "multi_numerator"},
            children=[
                html.Label("Multi Numerator"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "multi_numerator_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=[config_dataset["defaults"]["numerator"]],
                    multi=True,
                    placeholder="Select Numerators",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("multi_numerator") == True
            else {"display": "none"},
        ),
        html.Div(
            id={"type": prefix_group, "index": "multi_denominator"},
            children=[
                html.Label("Multi Denominator"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "multi_denominator_selector"},
                    options=config_dataset["columns"]["measures"],
                    value=[config_dataset["defaults"]["denominator"]],
                    multi=True,
                    placeholder="Select Denominators",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("multi_denominator") == True
            else {"display": "none"},
        ),
    ]

    # filter selectors
    selectors = [
        selector
        for selector in all_selectors
        if hasattr(selector, "id") and selector.id.get("index") in selector_dict.keys()
    ]

    return selectors


def load_config(config_path: str = helpers.CONFIG_PATH) -> Dict[str, Any]:
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


def write_config(
    config: Dict[str, Any], config_path: str = helpers.CONFIG_PATH
) -> None:
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


def list_files_in_folder(folder_path: str) -> List[str]:
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


def flatten_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flatten columns in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to flatten columns.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with flattened columns.

    """
    df.columns = [
        "__".join(col).strip() if isinstance(col, tuple) else col for col in df.columns
    ]
    return df


def _inputs_flatten_list(input_list: List[Any]) -> List[Any]:
    flat_list = []
    for item in input_list:
        if isinstance(item, dict):
            flat_list.append(item)
        elif isinstance(item, list):
            flat_list.extend(_inputs_flatten_list(item))
    return flat_list


def _inputs_parse_id(input_list: List[Any], id_value: str) -> Any:
    """
    Parse inputs for id value.

    This will get the value of an id from a dash callback_context list.

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


def _inputs_parse_type(input_list: List[Any], type_value: str) -> List[Any]:
    """
    Parse inputs for type value.

    This will get a list with a certain type from a dash callback_context list.

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
