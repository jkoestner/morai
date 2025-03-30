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
            # Header section with gradient background
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                [
                                    html.I(className="fas fa-database me-2"),
                                    "Data Input",
                                ],
                                className="mb-1",
                            ),
                            html.P(
                                "Load and configure your dataset",
                                className="text-white-50 mb-0 small",
                            ),
                        ],
                        className="bg-gradient bg-primary text-white p-4 mb-4 rounded-3 shadow-sm",
                    ),
                ],
            ),
            # Instructions Card
            dbc.Container(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5(
                                    [
                                        html.I(className="fas fa-info-circle me-2"),
                                        "Instructions",
                                    ],
                                    className="card-title mb-3",
                                ),
                                html.P(
                                    [
                                        "This page is used to load the configuration file and display the configuration.",
                                        html.Br(),
                                        "The data must first be loaded and then can be used in the other pages. The data will be available until the tab is closed.",
                                        html.Br(),
                                        html.Span(
                                            "Configuration Location:",
                                            className="fw-bold",
                                        ),
                                        " The configuration file should be located in: ",
                                        html.Code(
                                            f"{helpers.FILES_PATH!s}",
                                            className="bg-light px-2 py-1 rounded",
                                        ),
                                        html.Br(),
                                        html.Span("File Name:", className="fw-bold"),
                                        " The name should be: ",
                                        html.Code(
                                            "dashboard_config.yaml",
                                            className="bg-light px-2 py-1 rounded",
                                        ),
                                    ],
                                    className="card-text",
                                ),
                            ]
                        )
                    ],
                    className="shadow-sm mb-4",
                ),
                className="px-0",
            ),
            # Dataset Selection Card
            dbc.Container(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5(
                                [
                                    html.I(className="fas fa-file-alt me-2"),
                                    "Select Dataset",
                                ],
                                className="mb-0",
                            ),
                            className="bg-light",
                        ),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="dataset-dropdown",
                                                options=[
                                                    {"label": key, "value": key}
                                                    for key in list(
                                                        dh.load_config()[
                                                            "datasets"
                                                        ].keys()
                                                    )
                                                ],
                                                placeholder="Select a dataset",
                                                className="shadow-sm",
                                            ),
                                            width=8,
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(
                                                        className="fas fa-upload me-2"
                                                    ),
                                                    "Load Config",
                                                ],
                                                id="button-load-config",
                                                color="primary",
                                                className="w-100 shadow-sm",
                                            ),
                                            width=4,
                                        ),
                                    ],
                                    className="g-3",
                                ),
                            ]
                        ),
                    ],
                    className="shadow-sm mb-4",
                ),
                className="px-0",
            ),
            # Configuration Display Cards
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [
                                                    html.I(className="fas fa-cog me-2"),
                                                    "General Config",
                                                ],
                                                className="mb-0",
                                            ),
                                            className="bg-light",
                                        ),
                                        dbc.CardBody(
                                            dcc.Markdown(
                                                id="general-config-str",
                                                className="mb-0",
                                            ),
                                        ),
                                    ],
                                    className="shadow-sm h-100",
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [
                                                    html.I(
                                                        className="fas fa-table me-2"
                                                    ),
                                                    "Dataset Config",
                                                ],
                                                className="mb-0",
                                            ),
                                            className="bg-light",
                                        ),
                                        dbc.CardBody(
                                            dcc.Markdown(
                                                id="dataset-config-str",
                                                className="mb-0",
                                            ),
                                        ),
                                    ],
                                    className="shadow-sm h-100",
                                ),
                                width=6,
                            ),
                        ],
                        className="g-4",
                    ),
                ],
                className="px-0",
            ),
        ],
        className="container px-4 py-3",
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
    # only works with parquet, csv files
    # could add more
    logger.debug("load data")
    config_dataset = config["datasets"][config["general"]["dataset"]]
    file_path = helpers.FILES_PATH / "dataset" / config_dataset["filename"]
    if file_path.suffix == ".parquet":
        pl.enable_string_cache()
        lzdf = pl.scan_parquet(file_path)
    if file_path.suffix == ".csv":
        lzdf = pl.read_csv(file_path)

    return config, Serverside(lzdf, key=config["general"]["dataset"])


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
