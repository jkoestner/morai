"""
Utilities for app.

Provides functions for common used functions in app.
"""

import os
from typing import Any, Dict, List, Optional, Union

import dash_bootstrap_components as dbc
import pandas as pd
import polars as pl
import yaml
from dash import dcc, html

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)

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


def read_table(filepath: str) -> pl.LazyFrame:
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
    df : pl.LazyFrame
        Lazy Dataframe from the file.

    """
    if filepath.suffix == ".csv":
        df = pl.scan_csv(filepath)
    elif filepath.suffix == ".parquet":
        df = pl.scan_parquet(filepath)
    elif filepath.suffix == ".xlsx":
        df = pl.read_excel(filepath, sheet_name="rate_table").lazy()
    else:
        raise ValueError(f"File type not supported: {filepath}")

    # return dataframe
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    return df


def filter_data(
    df: Union[pd.DataFrame, pl.LazyFrame],
    callback_context: list[dict],
    num_to_str_count: int = num_to_str_count,
) -> Union[pd.DataFrame, pl.LazyFrame]:
    """
    Filter data based on the number of unique values.

    Parameters
    ----------
    df : pd.DataFrame or pl.LazyFrame
        Lazy dataframe to filter.
    callback_context : list
        List of callback context.
    num_to_str_count : int
        Number of unique values to convert to string.

    Returns
    -------
    filtered_df : pd.DataFrame or pl.LazyFrame
        Filtered dataframe.

    """
    is_lazy = isinstance(df, pl.LazyFrame)
    if not is_lazy:
        df = pl.from_pandas(df).lazy()
    schema = df.collect_schema()

    str_cols = []
    num_cols = []
    for col_name, dtype in schema.items():
        if dtype in pl.datatypes.group.NUMERIC_DTYPES:
            unique_count = df.select(pl.col(col_name).n_unique()).collect().item()
            if unique_count < num_to_str_count:
                str_cols.append(col_name)
            else:
                num_cols.append(col_name)
        else:
            str_cols.append(col_name)

    filtered_df = df

    # filter string columns
    for col in str_cols:
        str_values = _inputs_parse_id(callback_context, col)
        if str_values:
            filtered_df = filtered_df.filter(pl.col(col).is_in(str_values))

    # filter numeric columns
    for col in num_cols:
        num_values = _inputs_parse_id(callback_context, col)
        if num_values:
            filtered_df = filtered_df.filter(
                (pl.col(col) >= num_values[0]) & (pl.col(col) <= num_values[1])
            )

    # convert back to pandas
    if not is_lazy:
        filtered_df = filtered_df.collect().to_pandas()

    return filtered_df


def generate_filters(
    df: Union[pd.DataFrame, pl.LazyFrame],
    prefix: str,
    num_to_str_count: int = num_to_str_count,
    config: Optional[Dict[str, Any]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> dict:
    """
    Generate a dictionary of dashboard options from dataframe.

    Parameters
    ----------
    df : pd.DataFrame or pl.LazyFrame
        Lazy dataframe to generate dropdown options.
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
    # initialize
    filters = []
    str_cols = []
    num_cols = []
    prefix_str_filter = f"{prefix}-str-filter"
    prefix_num_filter = f"{prefix}-num-filter"

    # get column types
    is_lazy = isinstance(df, pl.LazyFrame)
    if not is_lazy:
        df = pl.from_pandas(df).lazy()
    schema = df.collect_schema()
    columns = list(schema.keys())

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
        col_dtype = schema[col]

        # check if categorical (non-numeric or low unique count)
        is_categorical = False
        if col_dtype not in pl.datatypes.group.NUMERIC_DTYPES:
            is_categorical = True
        else:
            unique_count = df.select(pl.col(col).n_unique()).collect().item()
            if unique_count < num_to_str_count:
                is_categorical = True

        # create options for categorical
        if is_categorical:
            unique_values = (
                df.select(pl.col(col)).drop_nulls().unique().collect().to_pandas()
            )
            options = [
                {"label": str(i), "value": i}
                for i in sorted(unique_values[col].astype(str).unique())
            ]

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
                            options=options,
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

        # create slider for numeric columns
        else:
            min_val = df.select(pl.col(col).min()).collect().item()
            max_val = df.select(pl.col(col).max()).collect().item()

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
                            min=min_val,
                            max=max_val,
                            step=1,
                            marks=None,
                            value=[min_val, max_val],
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


def get_active_filters(
    callback_context: Any,
    str_filters: Optional[list[Any]] = None,
    num_filters: Optional[list[Any]] = None,
) -> list[Any]:
    """
    Create a list of active filters for display.

    Parameters
    ----------
    callback_context : dash.callback_context
        The callback context containing states information
    str_filters : list, optional
        List of string filter values
    num_filters : list, optional
        List of numeric filter values (min/max pairs)

    Returns
    -------
    list
        List of html.Div elements representing active filters

    """
    active_filters_list = []

    # string filters
    if str_filters:
        for i, filter_value in enumerate(str_filters):
            if filter_value:
                col_name = [k["id"]["index"] for k in callback_context.states_list[0]][
                    i
                ]
                active_filters_list.append(
                    html.Div(
                        [
                            html.Strong(f"{col_name}: "),
                            ", ".join(str(v) for v in filter_value),
                        ],
                        className="mb-1",
                    )
                )

    # numeric filters
    if num_filters:
        for i, filter_value in enumerate(num_filters):
            if filter_value:
                col_name = [k["id"]["index"] for k in callback_context.states_list[1]][
                    i
                ]
                active_filters_list.append(
                    html.Div(
                        [
                            html.Strong(f"{col_name}: "),
                            f"{filter_value[0]} - {filter_value[1]}",
                        ],
                        className="mb-1",
                    )
                )

    return active_filters_list


def toggle_collapse(
    callback_context: Any, is_open: List[bool], children: List[dict]
) -> tuple[List[bool], List[List[dict]]]:
    """
    Toggle collapse state of filter checklists.

    Parameters
    ----------
    callback_context : dash.callback_context
        The callback context containing states information
    is_open : List[bool]
        List of current collapse states
    children : List[dict]
        List of current button children

    Returns
    -------
    tuple[List[bool], List[List[dict]]]
        Updated collapse states and button children

    """
    # callback not triggered
    ctx = callback_context
    if not ctx.triggered:
        return [False] * len(is_open), children

    # find which button was clicked
    button_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    button_idx = eval(button_id)["index"]

    # initialize
    new_is_open = []
    new_children = []

    # update collapse states and button content
    for _, (col, is_open_state, child) in enumerate(
        zip(
            [x["id"]["index"] for x in callback_context.inputs_list[0]],
            is_open,
            children,
            strict=False,
        )
    ):
        # collapse state
        new_state = not is_open_state if col == button_idx else is_open_state
        new_is_open.append(new_state)

        # button content
        label = child[0]["props"]["children"]  # Get the column name
        new_children.append(
            [
                html.Span(label, style={"flex-grow": 1}),
                html.I(className=f"fas fa-chevron-{'up' if new_state else 'down'}"),
            ]
        )

    return new_is_open, new_children


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
    df: pl.LazyFrame,
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
    df : pl.LazyFrame
        Lazy dataframe to gather KPI from.
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
        column_sum = df.select(pl.col(column).sum()).collect().item()
        value_text = f"{column}: {convert_to_short_number(column_sum)}"
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
            id={"type": prefix_group, "index": "normalize"},
            children=[
                html.Label("Normalize"),
                dcc.Dropdown(
                    id={"type": prefix_selector, "index": "normalize_selector"},
                    options=config_dataset["columns"]["features"],
                    value=None,
                    clearable=False,
                    multi=True,
                    placeholder="Normalize By",
                ),
            ],
            style={"display": "block"}
            if selector_dict.get("normalize") == True
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


def flatten_columns(
    df: Union[pd.DataFrame, pl.LazyFrame],
) -> Union[pd.DataFrame, pl.LazyFrame]:
    """
    Flatten columns in dataframe.

    Parameters
    ----------
    df : pd.DataFrame or pl.LazyFrame
        DataFrame to flatten columns.

    Returns
    -------
    df : pd.DataFrame or pl.LazyFrame
        DataFrame with flattened columns.

    """
    # check if lazy
    is_lazy = isinstance(df, pl.LazyFrame)

    if is_lazy:
        schema = df.collect_schema()
        columns = list(schema.keys())
        new_columns = []
        for col in columns:
            if isinstance(col, tuple):
                new_columns.append("__".join(str(c).strip() for c in col))
            else:
                new_columns.append(col)
        rename_dict = {
            old: new
            for old, new in zip(columns, new_columns, strict=False)
            if old != new
        }
        df = df.rename(rename_dict)
    else:  # pandas
        columns = df.columns
        new_columns = []
        for col in columns:
            if isinstance(col, tuple):
                new_columns.append("__".join(str(c).strip() for c in col))
            else:
                new_columns.append(col)
        df.columns = new_columns

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
