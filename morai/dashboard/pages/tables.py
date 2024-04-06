"""
Tables dashboard.

Issue age, duration, and attained age are needed to compare mortality tables.
"""

from io import StringIO

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from morai.experience import charters, tables
from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/tables", title="morai - Tables")


#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Table layout."""
    return html.Div(
        [
            dcc.Store(id="user-config", storage_type="session"),
            dcc.Store(id="compare-df-store", storage_type="session"),
            # -----------------------------------------------------
            html.H4(
                "Table Viewer",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            dbc.Toast(
                "Need to enter two tables to compare.",
                id="toast-null-tables",
                header="Input Error",
                is_open=False,
                dismissable=True,
                icon="danger",
                style={"position": "fixed", "top": 100, "right": 10, "width": 350},
            ),
            dbc.Toast(
                "Table not found.",
                id="toast-table-not-found",
                header="Input Error",
                is_open=False,
                dismissable=True,
                icon="danger",
                style={"position": "fixed", "top": 100, "right": 10, "width": 350},
            ),
            html.P(
                [
                    "This page is used to compare mortality tables from " "the SOA.",
                    html.Br(),
                    "Mortality tables are sourced from: ",
                    html.A(
                        "mort.soa.org", href="https://mort.soa.org", target="_blank"
                    ),
                ],
            ),
            dbc.Col(
                dbc.Button("Compare", id="compare-button", color="primary"),
                width="auto",
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Morality Table 1"),
                                dbc.CardBody(
                                    dbc.Input(
                                        type="number",
                                        id="table-1-id",
                                        placeholder="example 3249",
                                    ),
                                ),
                            ],
                            color="light",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Table Description"),
                                dbc.CardBody(html.P(id="table-1-desc", children=" ")),
                            ],
                            color="light",
                        ),
                        width=6,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Morality Table 2"),
                                dbc.CardBody(
                                    dbc.Input(
                                        type="number",
                                        id="table-2-id",
                                        placeholder="example 3252",
                                    ),
                                ),
                            ],
                            color="light",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Table Description"),
                                dbc.CardBody(html.P(id="table-2-desc", children=" ")),
                            ],
                            color="light",
                        ),
                        width=6,
                    ),
                ],
                className="mb-2",
            ),
            dcc.Loading(
                id="loading-content",
                type="dot",
                children=html.Div(id="graph-contour"),
            ),
            dbc.Row(
                dbc.Col(
                    [
                        html.Label("Issue Age", className="text-center"),
                        dcc.Slider(
                            id="slider-issue-age",
                            min=0,
                            max=100,
                            value=0,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    width=6,
                    className="mx-auto",
                ),
                justify="center",
                align="center",
                className="my-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-content",
                            type="dot",
                            children=html.Div(id="graph-compare-duration"),
                        ),
                    ),
                    dbc.Col(
                        dcc.Loading(
                            id="loading-content",
                            type="dot",
                            children=html.Div(id="graph-compare-age"),
                        ),
                    ),
                ],
                className="mb-2",
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
        Output("compare-df-store", "data"),
        Output("table-1-desc", "children"),
        Output("table-2-desc", "children"),
        Output("toast-null-tables", "is_open"),
        Output("toast-table-not-found", "is_open"),
    ],
    [Input("compare-button", "n_clicks")],
    [State("table-1-id", "value"), State("table-2-id", "value")],
    prevent_initial_call=True,
)
def get_table_data(n_clicks, table1_id, table2_id):
    """Get the table data and create a compare dataframe."""
    logger.info(f"Retrieving tables {table1_id} and {table2_id}")

    if table1_id is None or table2_id is None:
        return dash.no_update, dash.no_update, dash.no_update, True, False

    # get the tables and compare them
    mt = tables.MortTable()
    try:
        table_1 = mt.build_table(table_list=[table1_id], extend=True)
        table_2 = mt.build_table(table_list=[table2_id], extend=True)
    except FileNotFoundError:
        logger.warning(f"Table not found: {table1_id} or {table2_id}")
        return dash.no_update, dash.no_update, dash.no_update, False, True
    table_1_desc = mt.get_soa_xml(table1_id).ContentClassification.TableDescription
    table_2_desc = mt.get_soa_xml(table2_id).ContentClassification.TableDescription
    compare_df = tables.compare_tables(table_1, table_2)

    # serialize the compare_df
    compare_df = compare_df.to_json(orient="split")

    return (
        compare_df,
        table_1_desc,
        table_2_desc,
        False,
        False,
    )


@callback(
    [
        Output("graph-contour", "children"),
        Output("slider-issue-age", "min"),
        Output("slider-issue-age", "max"),
        Output("slider-issue-age", "value"),
    ],
    [Input("compare-df-store", "data")],
    prevent_initial_call=True,
)
def create_contour_and_sliders(compare_df):
    """Graph the mortality tables with a contour and comparison."""
    # deserialize the compare_df
    compare_df = pd.read_json(StringIO(compare_df), orient="split")

    # get the slider values
    issue_age_min = compare_df["issue_age"].min()
    issue_age_max = compare_df["issue_age"].max()
    issue_age_value = issue_age_min

    # graph the tables
    graph_contour = charters.chart(
        compare_df,
        x_axis="issue_age",
        y_axis="duration",
        color="ratio",
        type="contour",
        y_special=False,
    )

    graph_contour = dcc.Graph(figure=graph_contour)

    return (
        graph_contour,
        issue_age_min,
        issue_age_max,
        issue_age_value,
    )


@callback(
    [
        Output("graph-compare-duration", "children"),
        Output("graph-compare-age", "children"),
    ],
    [Input("slider-issue-age", "value")],
    [State("compare-df-store", "data")],
    prevent_initial_call=True,
)
def update_graphs_from_slider(issue_age_value, compare_df):
    """Update the compare duration graph."""
    # deserialize the compare_df
    compare_df = pd.read_json(StringIO(compare_df), orient="split")

    # graph the tables
    graph_compare_duration = charters.compare_rates(
        compare_df[compare_df["issue_age"] == issue_age_value],
        x_axis="duration",
        rates=["table_1", "table_2"],
        y_log=True,
    )

    graph_compare_age = charters.compare_rates(
        compare_df[compare_df["issue_age"] == issue_age_value],
        x_axis="attained_age",
        rates=["table_1", "table_2"],
        y_log=True,
    )

    graph_compare_duration = dcc.Graph(figure=graph_compare_duration)
    graph_compare_age = dcc.Graph(figure=graph_compare_age)

    return graph_compare_duration, graph_compare_age
