"""Data Input dashboard."""

import json

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import polars as pl
from dash_extensions.enrich import (
    Input,
    Output,
    Serverside,
    State,
    callback,
    dcc,
    html,
)

from morai.dashboard.utils import dashboard_helper as dh
from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/", title="morai - Input", order=0)


#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Input layout."""
    return html.Div(
        [
            dbc.Row(
                html.H4(
                    "Data Input",
                    className="bg-primary text-white p-2 mb-2 text-center",
                )
            ),
            dbc.Row(
                html.P(
                    [
                        "This page is used to load the configuration file "
                        "and display the configuration.",
                        html.Br(),
                        "The data must first be loaded and then can be used in "
                        "the other pages. The data will be available until the tab "
                        "is closed.",
                        html.Br(),
                        "The configuration file should be located in: ",
                        html.Span(
                            f"{helpers.FILES_PATH!s}",
                            style={"fontWeight": "bold"},
                        ),
                        html.Br(),
                        "The name should be: ",
                        html.Span(
                            "dashboard_config.yaml", style={"fontWeight": "bold"}
                        ),
                    ],
                ),
            ),
            # select file
            dbc.Container(
                [
                    dbc.Row(
                        html.Div(
                            html.H5(
                                "Select File",
                                style={
                                    "border-bottom": "1px solid black",
                                    "padding-bottom": "5px",
                                },
                            ),
                            style={
                                "width": "fit-content",
                                "padding": "0px",
                            },
                        ),
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="dataset-dropdown",
                                    options=[
                                        {"label": key, "value": key}
                                        for key in list(
                                            dh.load_config()["datasets"].keys()
                                        )
                                    ],
                                    placeholder="Select a dataset",
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Load Config",
                                    id="button-load-config",
                                    className="btn btn-primary",
                                ),
                                width=2,
                            ),
                        ],
                    ),
                ],
                className="m-1 bg-light border",
            ),
            # configuration
            dbc.Row(
                [
                    html.Div(
                        html.H5(
                            "Configuration",
                            style={
                                "border-bottom": "1px solid black",
                                "padding-bottom": "5px",
                            },
                        ),
                        style={
                            "width": "fit-content",
                            "padding": "0px",
                        },
                    ),
                    html.H6(
                        "General Config",
                    ),
                    dbc.Col(
                        dcc.Markdown(id="general-config-str"),
                    ),
                    html.H6(
                        "Dataset Config",
                    ),
                    dbc.Col(
                        dcc.Markdown(id="dataset-config-str"),
                    ),
                ],
                className="m-1 bg-light border",
            ),
        ],
        className="container",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


@callback(
    [
        Output("store-config", "data"),
        Output("store-dataset", "data"),
    ],
    [Input("button-load-config", "n_clicks")],
    [State("dataset-dropdown", "value")],
)
def load_config(n_clicks, dataset):
    """Load the configuration file."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    logger.debug("load config")
    config = dh.load_config()
    if dataset is not None:
        config["general"]["dataset"] = dataset

    # load dataset
    logger.debug("load data")
    config_dataset = config["datasets"][config["general"]["dataset"]]
    file_path = helpers.FILES_PATH / "dataset" / config_dataset["filename"]
    if file_path.suffix == ".parquet":
        pl.enable_string_cache()
        lzdf = pl.scan_parquet(file_path)
        dataset = lzdf.collect()
        dataset = dataset.to_pandas()

    return config, Serverside(dataset, key=config["general"]["dataset"])


@callback(
    [
        Output("general-config-str", "children"),
        Output("dataset-config-str", "children"),
    ],
    [Input("url", "pathname"), Input("store-config", "data")],
)
def display_config(pathname, config):
    """Display the configuration file."""
    if pathname != "/" or config is None:
        raise dash.exceptions.PreventUpdate

    # display config
    logger.debug("display config")
    general_dict = config["general"]
    general_json = json.dumps(general_dict, indent=2)
    general_config_str = f"```json\n{general_json}\n```"
    dataset_dict = config["datasets"][general_dict["dataset"]]
    dataset_json = json.dumps(dataset_dict, indent=2)
    dataset_config_str = f"```json\n{dataset_json}\n```"

    return (
        general_config_str,
        dataset_config_str,
    )
