"""Explore dashboard."""

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import polars as pl
from dash_extensions.enrich import (
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from morai.dashboard.components import sidebars
from morai.experience import charters
from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/explore", title="morai - Explore", order=1)

#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Explore layout."""
    return html.Div(
        [
            html.Div(  # Sidebar container
                sidebars.explore_sidebar,
            ),
            html.Div(  # Main content container
                [
                    html.Div(
                        [
                            html.H4(
                                [
                                    html.I(className="fas fa-search me-2"),
                                    "Data Exploration",
                                ],
                                className="mb-1",
                            ),
                            html.P(
                                "Analyze and visualize your data distributions",
                                className="text-white-50 mb-0 small",
                            ),
                        ],
                        className="bg-gradient bg-primary text-white p-4 mb-4 rounded-3 shadow-sm",
                    ),
                    html.Div(
                        [
                            # Dataset description card
                            dbc.Card(
                                dbc.CardBody(
                                    html.Div(
                                        id="data-description",
                                        className="mb-0",
                                    )
                                ),
                                className="shadow-sm mb-4",
                            ),
                            # Summary Statistics Table
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            [
                                                html.I(
                                                    className="fas fa-chart-bar me-2"
                                                ),
                                                "Summary Statistics",
                                            ],
                                            id="section-summary-stat",
                                            className="mb-0",
                                        ),
                                        className="bg-light",
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            id="loading-table-stats",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(id="table-stats"),
                                        ),
                                    ),
                                ],
                                className="shadow-sm mb-4",
                            ),
                            # Numerical Frequencies Section
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            [
                                                html.I(
                                                    className="fas fa-chart-bar me-2"
                                                ),
                                                "Numerical Frequencies",
                                            ],
                                            id="section-freq-num",
                                            className="mb-0",
                                        ),
                                        className="bg-light",
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            id="loading-chart-freq-num",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(id="chart-freq-num"),
                                        ),
                                    ),
                                ],
                                className="shadow-sm mb-4",
                            ),
                            # Categorical Frequencies Section
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            [
                                                html.I(
                                                    className="fas fa-chart-pie me-2"
                                                ),
                                                "Categorical Frequencies",
                                            ],
                                            id="section-freq-cat",
                                            className="mb-0",
                                        ),
                                        className="bg-light",
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            id="loading-chart-freq-cat",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(id="chart-freq-cat"),
                                        ),
                                    ),
                                ],
                                className="shadow-sm mb-4",
                            ),
                            # Target Analysis Section
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            [
                                                html.I(
                                                    className="fas fa-bullseye me-2"
                                                ),
                                                "Target Analysis",
                                            ],
                                            id="section-target",
                                            className="mb-0",
                                        ),
                                        className="bg-light",
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            id="loading-chart-target",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(id="chart-target"),
                                        ),
                                    ),
                                ],
                                className="shadow-sm mb-4",
                            ),
                        ],
                    ),
                ],
                style={"marginLeft": "250px"},
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
        Output("nav-freq-num", "active"),
        Output("nav-freq-cat", "active"),
        Output("nav-target", "active"),
    ],
    [Input("url", "hash"), Input("url", "href")],
)
def set_active_link(hash, href):
    """Set the active link in the sidebar based on the URL."""
    if not hash:
        return True, False, False
    elif "#section-freq-num" in hash:
        return True, False, False
    elif "#section-freq-cat" in hash:
        return False, True, False
    elif "#section-target" in hash:
        return False, False, True
    return False, False, False


@callback(
    [
        Output("table-stats", "children"),
        Output("chart-freq-num", "children"),
        Output("chart-freq-cat", "children"),
        Output("chart-target", "children"),
    ],
    [Input("url", "pathname")],
    [State("store-dataset", "data"), State("store-config", "data")],
)
def update_charts(pathname, dataset, config):
    """Update charts when page is loaded."""
    if pathname != "/explore" or dataset is None:
        raise dash.exceptions.PreventUpdate
    logger.debug("update explore charts")
    config_dataset = config["datasets"][config["general"]["dataset"]]
    features = config_dataset["columns"]["features"]

    is_lazy = isinstance(dataset, pl.LazyFrame)
    if is_lazy:
        schema = dataset.collect_schema()
        num_cols = [
            col
            for col in features
            if col in schema and schema[col] in pl.datatypes.group.NUMERIC_DTYPES
        ]
    else:
        num_cols = dataset[features].select_dtypes(include=["number"]).columns.tolist()

    measures = config_dataset["columns"]["measures"]

    # table
    stats_df = charters.get_stats(dataset, features=num_cols + measures)

    # charts
    chart_freq_num = charters.frequency(
        dataset,
        cols=3,
        features=num_cols,
        sum_var=config_dataset["columns"]["exposure_amt"],
    )

    chart_freq_cat = charters.frequency(
        dataset, cols=3, sum_var=config_dataset["columns"]["exposure_amt"]
    )

    chart_target = charters.target(
        df=dataset,
        target="risk",
        cols=3,
        features=features,
        numerator=[config_dataset["defaults"]["numerator"]],
        denominator=[config_dataset["defaults"]["denominator"]],
    )

    stats_table = dbc.Table.from_dataframe(
        stats_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="mb-0",
    )
    chart_freq_num = dcc.Graph(figure=chart_freq_num)
    chart_freq_cat = dcc.Graph(figure=chart_freq_cat)
    chart_target = dcc.Graph(figure=chart_target)

    return stats_table, chart_freq_num, chart_freq_cat, chart_target


@callback(
    [
        Output("data-description", "children"),
    ],
    [Input("url", "pathname")],
    [State("store-config", "data")],
)
def update_dataset_description(pathname, config):
    """Update charts when page is loaded."""
    if pathname not in ["/explore", "/experience", "/model"]:
        raise dash.exceptions.PreventUpdate

    filename = "No dataset loaded"
    description = "No dataset loaded"
    if config is not None:
        config_dataset = config["datasets"][config["general"]["dataset"]]
        filename = f"{config_dataset['filename']}"
        description = f"{config_dataset['description']}"

    div_description = html.Div(
        html.P(
            [
                html.Span(
                    "Dataset: ",
                    style={"fontWeight": "bold"},
                ),
                filename,
                html.Br(),
                html.Span(
                    "Description: ",
                    style={"fontWeight": "bold"},
                ),
                description,
            ],
        ),
    )

    return div_description
