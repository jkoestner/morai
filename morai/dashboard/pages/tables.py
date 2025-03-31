"""
Tables dashboard.

Issue age, duration, and attained age are needed to compare mortality tables.
"""

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash_extensions.enrich import (
    ALL,
    Input,
    Output,
    State,
    callback,
    callback_context,
    dcc,
    html,
)

from morai.dashboard.components import dash_formats
from morai.dashboard.utils import dashboard_helper as dh
from morai.experience import charters, tables
from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/tables", title="morai - Tables", order=4)


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
            dcc.Store(id="store-table-compare", storage_type="memory"),
            dcc.Store(id="store-table-1-select", storage_type="memory"),
            dcc.Store(id="store-table-2-select", storage_type="memory"),
            # Header section with gradient background
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                [
                                    html.I(className="fas fa-table me-2"),
                                    "Mortality Table Analysis",
                                ],
                                className="mb-1",
                            ),
                            html.P(
                                "Compare and analyze mortality tables",
                                className="text-white-50 mb-0 small",
                            ),
                        ],
                        className="bg-gradient bg-primary text-white p-4 mb-4 rounded-3 shadow-sm",
                    ),
                ],
            ),
            # Toast notifications
            dbc.Toast(
                id="tables-toast",
                header="Notification",
                is_open=False,
                dismissable=True,
                icon="danger",
                style={"position": "fixed", "top": 100, "right": 10, "width": 350},
            ),
            # Description Card
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(
                            [
                                html.I(className="fas fa-info-circle me-2"),
                                "About Mortality Tables",
                            ],
                            className="card-title mb-3",
                        ),
                        html.P(
                            [
                                "This page is used to compare mortality tables from the SOA. ",
                                html.Br(),
                                "The tables must include one of the following columns: ",
                                html.Code("issue_age, duration, and attained_age"),
                                html.Br(),
                                "The table must also have a rate value column: ",
                                html.Code("vals"),
                                html.Br(),
                                "Mortality tables are sourced from: ",
                                html.A(
                                    "mort.soa.org",
                                    href="https://mort.soa.org",
                                    target="_blank",
                                ),
                            ],
                            className="card-text mb-0",
                        ),
                    ]
                ),
                className="shadow-sm mb-4",
            ),
            # Action Buttons
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            [
                                html.I(className="fas fa-sync-alt me-2"),
                                "Compare",
                            ],
                            id="compare-button",
                            color="primary",
                            className="w-100 shadow-sm",
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        dbc.Button(
                            [
                                html.I(className="fas fa-filter me-2"),
                                "Filter",
                            ],
                            id="filter-button",
                            color="primary",
                            className="w-100 shadow-sm",
                        ),
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # Table Selection Cards
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H5(
                                        [
                                            html.I(className="fas fa-table me-2"),
                                            "Table 1 Selection",
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
                                                    dcc.RadioItems(
                                                        id="table-1-radio",
                                                        options=[
                                                            "soa table",
                                                            "file",
                                                            "rate_table",
                                                        ],
                                                        value="soa table",
                                                        className="mb-3",
                                                    ),
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    html.Div(id="table-1-card"),
                                                    width=9,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        html.Div(id="table-1-desc"),
                                        html.Div(id="table-1-filters"),
                                    ]
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
                                            html.I(className="fas fa-table me-2"),
                                            "Table 2 Selection",
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
                                                    dcc.RadioItems(
                                                        id="table-2-radio",
                                                        options=[
                                                            "soa table",
                                                            "file",
                                                            "rate_table",
                                                        ],
                                                        value="soa table",
                                                        className="mb-3",
                                                    ),
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    html.Div(id="table-2-card"),
                                                    width=9,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        html.Div(id="table-2-desc"),
                                        html.Div(id="table-2-filters"),
                                    ]
                                ),
                            ],
                            className="shadow-sm h-100",
                        ),
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Analysis Results Section
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.H5(
                            [
                                html.I(className="fas fa-chart-area me-2"),
                                "Analysis Results",
                            ],
                            className="mb-0",
                        ),
                        className="bg-light",
                    ),
                    dbc.CardBody(
                        [
                            # Contour Analysis
                            html.H5(
                                [
                                    html.I(className="fas fa-project-diagram me-2"),
                                    "Table Comparison Contour (Table 1 / Table 2)",
                                ],
                                id="section-table-contour",
                                className="mb-3",
                            ),
                            dcc.Loading(
                                id="loading-graph-contour",
                                type="default",
                                color="#007bff",
                                children=html.Div(
                                    id="graph-contour",
                                    className="bg-white rounded-3 shadow-sm p-3 mb-4",
                                ),
                            ),
                            # Age Analysis
                            html.H5(
                                [
                                    html.I(className="fas fa-chart-line me-2"),
                                    "Table Plot Age",
                                ],
                                id="section-table-plot",
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-graph-compare-age",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="graph-compare-age",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-graph-compare-age-log",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="graph-compare-age-log",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Issue Age",
                                                    className="text-center mb-2",
                                                ),
                                                dcc.Slider(
                                                    id="slider-issue-age",
                                                    min=0,
                                                    max=100,
                                                    value=-1,
                                                    step=1,
                                                    marks={
                                                        i: str(i)
                                                        for i in range(0, 100, 10)
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                    className="mb-4",
                                                ),
                                            ],
                                            id="slider-container",
                                            style={"display": "none"},
                                        ),
                                        width=6,
                                        className="mx-auto",
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-graph-compare-ratio",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="graph-compare-ratio",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Select/Ultimate Analysis
                            html.H5(
                                [
                                    html.I(className="fas fa-chart-bar me-2"),
                                    "Select/Ultimate Compare",
                                ],
                                id="section-table-su",
                                className="mb-3",
                            ),
                            html.P(
                                "The select and ultimate ratio is the ratio of the [ultimate attained age] / [select attained age].",
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-graph-su-table-1",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="graph-su-table-1",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-graph-su-table-2",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="graph-su-table-2",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Table Data
                            html.H5(
                                [
                                    html.I(className="fas fa-table me-2"),
                                    "Table Data",
                                ],
                                id="section-table-data",
                                className="mb-3",
                            ),
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Table-1",
                                        tab_id="tab-table-1",
                                        label_class_name="fw-bold",
                                        active_label_class_name="text-primary",
                                    ),
                                    dbc.Tab(
                                        label="Table-2",
                                        tab_id="tab-table-2",
                                        label_class_name="fw-bold",
                                        active_label_class_name="text-primary",
                                    ),
                                    dbc.Tab(
                                        label="Compare",
                                        tab_id="tab-table-compare",
                                        label_class_name="fw-bold",
                                        active_label_class_name="text-primary",
                                    ),
                                ],
                                id="tabs-tables",
                                active_tab="tab-table-compare",
                                className="mb-3",
                            ),
                            dcc.Loading(
                                id="loading-tables-tab-content",
                                type="default",
                                color="#007bff",
                                children=html.Div(
                                    id="tables-tab-content",
                                    className="bg-white rounded-3 shadow-sm p-3",
                                ),
                            ),
                        ]
                    ),
                ],
                className="shadow-sm",
            ),
        ],
        className="container-fluid px-4 py-3",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


@callback(
    [Output("table-1-card", "children")],
    [Input("table-1-radio", "value")],
)
def set_table_1_input(value):
    """Set the table 1 input based on the radio button."""
    if value == "soa table":
        input_box = dbc.Input(
            type="number",
            id="table-1-id",
            placeholder="example 3249",
        )
    elif value == "file":
        input_box = (
            dcc.Dropdown(
                id="table-1-id",
                options=[
                    {"label": key, "value": key}
                    for key in dh.list_files_in_folder(helpers.FILES_PATH / "rates")
                    if key.endswith((".csv", ".xlsx"))
                ],
                placeholder="Select a file",
            ),
        )
    else:
        input_box = dcc.Dropdown(
            id="table-1-id",
            options=tables.get_rates(),
            placeholder="Select a file",
        )
    return input_box


@callback(
    [Output("table-2-card", "children")],
    [Input("table-2-radio", "value")],
)
def set_table_2_input(value):
    """Set the table 2 input based on the radio button."""
    if value == "soa table":
        input_box = dbc.Input(
            type="number",
            id="table-2-id",
            placeholder="example 3252",
        )
    elif value == "file":
        input_box = (
            dcc.Dropdown(
                id="table-2-id",
                options=[
                    {"label": key, "value": key}
                    for key in dh.list_files_in_folder(helpers.FILES_PATH / "rates")
                    if key.endswith((".csv", ".xlsx"))
                ],
                placeholder="Select a file",
            ),
        )
    else:
        input_box = dcc.Dropdown(
            id="table-2-id",
            options=tables.get_rates(),
            placeholder="Select a file",
        )
    return input_box


@callback(
    [
        Output("store-table-1-select", "data"),
        Output("store-table-2-select", "data"),
        Output("table-1-filters", "children"),
        Output("table-2-filters", "children"),
        Output("store-table-compare", "data"),
        Output("table-1-desc", "children"),
        Output("table-2-desc", "children"),
        Output("graph-su-table-1", "children"),
        Output("graph-su-table-2", "children"),
        Output("tables-toast", "is_open", allow_duplicate=True),
        Output("tables-toast", "children", allow_duplicate=True),
    ],
    [Input("compare-button", "n_clicks"), Input("filter-button", "n_clicks")],
    [
        State("table-1-id", "value"),
        State("table-2-id", "value"),
        State({"type": "table-1-str-filter", "index": ALL}, "value"),
        State({"type": "table-1-num-filter", "index": ALL}, "value"),
        State({"type": "table-2-str-filter", "index": ALL}, "value"),
        State({"type": "table-2-num-filter", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def initialize_tables(
    n_clicks,
    filter_button,
    table1_id,
    table2_id,
    filter_table_1_str,
    filter_table_1_num,
    filter_table_2_str,
    filter_table_2_num,
):
    """Get the initial table data."""
    logger.debug(f"Retrieving tables {table1_id} and {table2_id}")
    no_upate_tuple = (dash.no_update,) * 9
    warning_tuple = (False, "")

    if table1_id is None or table2_id is None:
        return (*no_upate_tuple, True, "No table selected.")

    # load tables
    table_1, table_2, table_1_select_period, table_2_select_period, warning_tuple = (
        load_tables(table1_id, table2_id)
    )

    if True in warning_tuple:
        return (*no_upate_tuple, *warning_tuple)

    # generate filters
    if callback_context.triggered_id == "compare-button":
        filters_1 = dh.generate_filters(
            df=table_1,
            prefix="table-1",
            exclude_cols=["vals", "constant"],
        ).get("filters")
        filters_2 = dh.generate_filters(
            df=table_2,
            prefix="table-2",
            exclude_cols=["vals", "constant"],
        ).get("filters")
    else:
        filters_1 = dash.no_update
        filters_2 = dash.no_update

    # Get the filters from the callback context for description
    states_info = dh._inputs_flatten_list(callback_context.states_list)
    filters_table_1 = dh._inputs_parse_type(
        states_info, "table-1-num-filter"
    ) + dh._inputs_parse_type(states_info, "table-1-str-filter")
    filters_table_2 = dh._inputs_parse_type(
        states_info, "table-2-num-filter"
    ) + dh._inputs_parse_type(states_info, "table-2-str-filter")

    # filter the datasets
    filtered_table_1, filtered_table_2 = filter_tables(
        table_1, table_2, filter_list=callback_context.states_list
    )

    # group the datasets
    grouped_table_1 = (
        filtered_table_1.groupby(
            ["issue_age", "duration", "attained_age"], observed=True
        )["vals"]
        .agg("mean")
        .reset_index()
    )

    grouped_table_2 = (
        filtered_table_2.groupby(
            ["issue_age", "duration", "attained_age"], observed=True
        )["vals"]
        .agg("mean")
        .reset_index()
    )

    # compare the tables
    compare_df = tables.compare_tables(grouped_table_1, grouped_table_2)

    # table descriptions
    desc_1, desc_2 = get_table_desc(
        table1_id,
        table2_id,
        filtered_table_1,
        filtered_table_2,
        table_1_select_period,
        table_2_select_period,
        filters_table_1,  # Pass the filter info directly from states
        filters_table_2,
    )

    # select/ultimate graphs
    graph_su_table_1 = get_su_graph(
        grouped_table_1, table_1_select_period, title="Select/Ultimate Contour Table 1"
    )
    graph_su_table_2 = get_su_graph(
        grouped_table_2, table_2_select_period, title="Select/Ultimate Contour Table 2"
    )

    return (
        table_1_select_period,
        table_2_select_period,
        filters_1,
        filters_2,
        compare_df.to_dict("records"),
        desc_1,
        desc_2,
        graph_su_table_1,
        graph_su_table_2,
        False,
        False,
    )


@callback(
    [
        Output("graph-contour", "children"),
        Output("slider-container", "style"),
        Output("slider-issue-age", "min"),
        Output("slider-issue-age", "max"),
        Output("slider-issue-age", "value"),
        Output("slider-issue-age", "marks"),
    ],
    [Input("store-table-compare", "data"), Input("slider-issue-age", "value")],
    prevent_initial_call=True,
)
def create_contour(
    compare_df,
    issue_age_value,
):
    """Graph the mortality tables with a contour and comparison."""
    compare_df = pd.DataFrame(compare_df)

    # get the slider values
    issue_age_min = compare_df["issue_age"].min()
    issue_age_max = compare_df["issue_age"].max()
    if issue_age_value == -1:
        issue_age_value = round((issue_age_max + issue_age_min) / 2, 0)
    issue_age_marks = {i: str(i) for i in range(issue_age_min, issue_age_max, 10)}

    # adding attained_age to additional hover data
    grouped_data = (
        compare_df.groupby(["issue_age", "duration"], observed=True)["ratio"]
        .agg("sum")
        .reset_index()
    )
    grouped_data = grouped_data.pivot(
        index="duration", columns="issue_age", values="ratio"
    )
    issue_age, duration = np.meshgrid(grouped_data.columns, grouped_data.index)
    attained_age = issue_age + duration - 1

    # custom colorscale
    zmin = 0.5
    zmax = 1.5
    custom_colorscale = [
        [0.0, "blue"],
        [(1 - zmin - 0.01) / (zmax - zmin), "greenyellow"],
        [(1 - zmin) / (zmax - zmin), "darkviolet"],
        [min((1 - zmin + 0.01) / (zmax - zmin), 1), "yellow"],
        [1.0, "crimson"],
    ]
    contours = {
        "start": zmin,
        "end": zmax,
        "size": (zmax - zmin) / 11,
    }

    # graph the tables
    graph_contour = charters.chart(
        compare_df,
        x_axis="issue_age",
        y_axis="duration",
        color="ratio",
        type="contour",
        agg="mean",
        # add custom hover data
        customdata=attained_age,
        hovertemplate=(
            "issue_age: %{x}<br>"
            "duration: %{y}<br>"
            "attained_age: %{customdata}<br>"
            "ratio: %{z}<extra></extra>"
        ),
        # custom colorscale and contours
        colorscale=custom_colorscale,
        zmin=zmin,
        zmax=zmax,
        contours=contours,
    )

    # add the age line
    age_line = go.Scatter(
        x=[issue_age_value, issue_age_value],
        y=[compare_df["duration"].min(), compare_df["duration"].max()],
        mode="lines",
        line={"color": "black", "width": 2},
        name="age",
    )
    graph_contour.add_trace(age_line)

    graph_contour = dcc.Graph(figure=graph_contour)

    return (
        graph_contour,
        {"display": "block"},
        issue_age_min,
        issue_age_max,
        issue_age_value,
        issue_age_marks,
    )


@callback(
    [
        Output("graph-compare-age", "children"),
        Output("graph-compare-age-log", "children"),
        Output("graph-compare-ratio", "children"),
    ],
    [Input("slider-issue-age", "value")],
    [
        State("store-table-compare", "data"),
    ],
    prevent_initial_call=True,
)
def update_graphs_from_slider(issue_age_value, compare_df):
    """Update the compare duration graph."""
    compare_df = pd.DataFrame(compare_df)

    # graph the tables
    graph_compare_age = charters.compare_rates(
        compare_df[compare_df["issue_age"] == issue_age_value],
        x_axis="attained_age",
        rates=["table_1", "table_2"],
        y_log=False,
    )
    graph_compare_age_log = charters.compare_rates(
        compare_df[compare_df["issue_age"] == issue_age_value],
        x_axis="attained_age",
        rates=["table_1", "table_2"],
        y_log=True,
    )
    graph_compare_ratio = charters.chart(
        compare_df[compare_df["issue_age"] == issue_age_value],
        x_axis="attained_age",
        y_axis="ratio",
        type="line",
    )

    graph_compare_age = dcc.Graph(figure=graph_compare_age)
    graph_compare_age_log = dcc.Graph(figure=graph_compare_age_log)
    graph_compare_ratio = dcc.Graph(figure=graph_compare_ratio)

    return graph_compare_age, graph_compare_age_log, graph_compare_ratio


@callback(
    [Output("tables-tab-content", "children")],
    [
        Input("tabs-tables", "active_tab"),
        Input("store-table-compare", "data"),
    ],
    [
        State("table-1-id", "value"),
        State("table-2-id", "value"),
        State({"type": "table-1-str-filter", "index": ALL}, "value"),
        State({"type": "table-1-num-filter", "index": ALL}, "value"),
        State({"type": "table-2-str-filter", "index": ALL}, "value"),
        State({"type": "table-2-num-filter", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def update_table_tabs(
    active_tab,
    compare_df,
    table1_id,
    table2_id,
    table_1_filter_str,
    table_1_filter_num,
    table_2_filter_str,
    table_2_filter_num,
):
    """Update the tables tab content."""
    if callback_context.triggered_id == "tabs-tables":
        # load tables
        (
            table_1,
            table_2,
            table_1_select_period,
            table_2_select_period,
            warning_tuple,
        ) = load_tables(table1_id, table2_id)

        # filter the datasets
        filtered_table_1, filtered_table_2 = filter_tables(
            table_1, table_2, filter_list=callback_context.states_list
        )

        # group the datasets
        grouped_table_1 = (
            filtered_table_1.groupby(
                ["issue_age", "duration", "attained_age"], observed=True
            )["vals"]
            .agg("mean")
            .reset_index()
        )

        grouped_table_2 = (
            filtered_table_2.groupby(
                ["issue_age", "duration", "attained_age"], observed=True
            )["vals"]
            .agg("mean")
            .reset_index()
        )

        # compare the tables
        compare_df = tables.compare_tables(grouped_table_1, grouped_table_2)

        if active_tab == "tab-table-1":
            table = filtered_table_1
        elif active_tab == "tab-table-2":
            table = filtered_table_2
        else:
            table = compare_df
    else:
        table = pd.DataFrame(compare_df)

    # deserialize the table
    columnDefs = dash_formats.get_column_defs(table)
    tab_content = dag.AgGrid(
        rowData=table.to_dict("records"),
        columnDefs=columnDefs,
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        dashGridOptions={"pagination": True},
    )

    return tab_content


#   _____                 _   _
#  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
#  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def load_tables(table1_id, table2_id):
    """Get the table data and create a compare dataframe."""
    # process tables
    table_1 = pd.DataFrame()
    table_2 = pd.DataFrame()
    table_1_select_period = "Unknown"
    table_2_select_period = "Unknown"
    warning_tuple = (False, "")
    mt = tables.MortTable()

    # table_1
    logger.debug(f"loading table 1: {table1_id}")
    if isinstance(table1_id, str):
        # file
        if table1_id.endswith((".csv", ".xlsx")):
            try:
                filepath_1 = helpers.FILES_PATH / "rates" / table1_id
                table_1 = dh.read_table(filepath=filepath_1)
                table_1_select_period = "Unknown"
                table_1 = table_1.to_pandas()
            except FileNotFoundError:
                logger.warning(f"Table not found: {table1_id}")
                warning_tuple = (True, f"Table not found: {table1_id}")
        # rate table
        else:
            try:
                rate_table = tables.MortTable(rate=table1_id)
                table_1 = rate_table.rate_table
                table_1_select_period = "Unknown"
            except FileNotFoundError:
                logger.warning(f"Table not found: {table1_id}")
                warning_tuple = (True, f"Table not found: {table1_id}")
        # add age and duration columns if not present
        table_1 = tables.add_aa_ia_dur_cols(table_1)
    # soa table
    else:
        try:
            table_1 = mt.build_table_soa(table_list=[table1_id], extend=False)
            table_1_select_period = mt.select_period
        except FileNotFoundError:
            logger.warning(f"Table not found: {table1_id}")
            warning_tuple = (True, f"Table not found: {table1_id}")

    # table_2
    logger.debug(f"loading table 2: {table2_id}")
    if isinstance(table2_id, str):
        # file
        if table2_id.endswith((".csv", ".xlsx")):
            try:
                filepath_2 = helpers.FILES_PATH / "rates" / table2_id
                table_2 = dh.read_table(filepath=filepath_2)
                table_2_select_period = "Unknown"
                table_2 = table_2.to_pandas()
            except FileNotFoundError:
                logger.warning(f"Table not found: {table2_id}")
                warning_tuple = (True, f"Table not found: {table2_id}")
        # rate table
        else:
            try:
                rate_table = tables.MortTable(rate=table2_id)
                table_2 = rate_table.rate_table
                table_2_select_period = "Unknown"
            except FileNotFoundError:
                logger.warning(f"Table not found: {table1_id}")
                warning_tuple = (True, f"Table not found: {table2_id}")
        # add age and duration columns if not present
        table_2 = tables.add_aa_ia_dur_cols(table_2)
    # soa table
    else:
        try:
            table_2 = mt.build_table_soa(table_list=[table2_id], extend=False)
            table_2_select_period = mt.select_period
        except FileNotFoundError:
            logger.warning(f"Table not found: {table2_id}")
            warning_tuple = (True, f"Table not found: {table2_id}")

    return table_1, table_2, table_1_select_period, table_2_select_period, warning_tuple


def filter_tables(table_1, table_2, filter_list):
    """Load and filter the tables."""
    # callback context
    inputs_info = dh._inputs_flatten_list(filter_list)
    filters_table_1 = dh._inputs_parse_type(
        inputs_info, "table-1-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-1-str-filter")
    filters_table_2 = dh._inputs_parse_type(
        inputs_info, "table-2-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-2-str-filter")

    # filter the datasets
    filtered_table_1 = dh.filter_data(df=table_1, callback_context=filters_table_1)
    filtered_table_2 = dh.filter_data(df=table_2, callback_context=filters_table_2)

    return filtered_table_1, filtered_table_2


def get_table_desc(
    table1_id,
    table2_id,
    filtered_table_1,
    filtered_table_2,
    table_1_select_period,
    table_2_select_period,
    filters_table_1=None,
    filters_table_2=None,
):
    """Get the table descriptions."""
    mt = tables.MortTable()

    # Get active filters for table 1
    active_filters_1 = "None"
    if filters_table_1:
        active_filters = []
        for filter_info in filters_table_1:
            if isinstance(filter_info, dict) and "id" in filter_info:
                col = filter_info["id"]["index"]
                if filter_info.get("value"):  # Check if filter has a value
                    active_filters.append(f"{col}: {filter_info['value']}")
        if active_filters:
            active_filters_1 = ", ".join(active_filters)

    # Get active filters for table 2
    active_filters_2 = "None"
    if filters_table_2:
        active_filters = []
        for filter_info in filters_table_2:
            if isinstance(filter_info, dict) and "id" in filter_info:
                col = filter_info["id"]["index"]
                if filter_info.get("value"):  # Check if filter has a value
                    active_filters.append(f"{col}: {filter_info['value']}")
        if active_filters:
            active_filters_2 = ", ".join(active_filters)

    # table description 1
    if isinstance(table1_id, str):
        table_1_desc = table1_id
    else:
        soa_xml = mt.get_soa_xml(table1_id)
        table_1_desc = soa_xml.ContentClassification.TableDescription
    table_1_cols = {
        col: len(filtered_table_1[col].unique())
        for col in filtered_table_1.columns
        if col != "vals"
    }
    desc_1 = html.Div(
        [
            html.B("Table Description:"),
            html.Span(f" {table_1_desc}"),
            html.Br(),
            html.B("Table Shape:"),
            html.Span(f" {filtered_table_1.shape}"),
            html.Br(),
            html.B("Columns:"),
            html.Span(f" {table_1_cols}"),
            html.Br(),
            html.B("Select Period:"),
            html.Span(f" {table_1_select_period}"),
            html.Br(),
            html.B("Active Filters:"),
            html.Span(f" {active_filters_1}"),
        ]
    )

    # table description 2
    if isinstance(table2_id, str):
        table_2_desc = table2_id
    else:
        soa_xml = mt.get_soa_xml(table2_id)
        table_2_desc = soa_xml.ContentClassification.TableDescription
    table_2_cols = {
        col: len(filtered_table_2[col].unique())
        for col in filtered_table_2.columns
        if col != "vals"
    }
    desc_2 = html.Div(
        [
            html.B("Table Description:"),
            html.Span(f" {table_2_desc}"),
            html.Br(),
            html.B("Table Shape:"),
            html.Span(f" {filtered_table_2.shape}"),
            html.Br(),
            html.B("Columns:"),
            html.Span(f" {table_2_cols}"),
            html.Br(),
            html.B("Select Period:"),
            html.Span(f" {table_2_select_period}"),
            html.Br(),
            html.B("Active Filters:"),
            html.Span(f" {active_filters_2}"),
        ]
    )

    return desc_1, desc_2


def get_su_graph(df, select_period, title):
    """Get the select and ultimate graph."""
    df = tables.get_su_table(df, select_period)

    # plot the chart
    fig = charters.chart(
        df,
        x_axis="issue_age",
        y_axis="duration",
        color="su_ratio",
        type="contour",
        agg="mean",
        title=title,
        hovertemplate=(
            "issue_age: %{x}<br>" "duration: %{y}<br>" "ratio: %{z}<extra></extra>"
        ),
    )

    # colorscale that pivots around 1
    z_min = np.nanmin(fig.data[0].z)
    z_max = np.nanmax(fig.data[0].z)
    colorscale = [
        [0, "white"],
        [(1 - z_min) / (z_max - z_min), "dimgray"],
        [(1 - z_min + 0.01) / (z_max - z_min), "royalblue"],
        [min((2 - z_min) / (z_max - z_min), 1), "cornflowerblue"],
        [min((3 - z_min) / (z_max - z_min), 1), "lightseagreen"],
        [min((4 - z_min) / (z_max - z_min), 1), "yellowgreen"],
        [min((5 - z_min) / (z_max - z_min), 1), "sandybrown"],
        [1, "red"],
    ]
    fig.update_traces(contours_coloring="heatmap", colorscale=colorscale)

    return dcc.Graph(figure=fig)


@callback(
    Output({"type": "table-1-collapse", "index": ALL}, "is_open"),
    Output({"type": "table-1-collapse-button", "index": ALL}, "children"),
    Input({"type": "table-1-collapse-button", "index": ALL}, "n_clicks"),
    State({"type": "table-1-collapse", "index": ALL}, "is_open"),
    State({"type": "table-1-collapse-button", "index": ALL}, "children"),
    prevent_initial_call=True,
)
def toggle_table_1_collapse(n_clicks, is_open, children):
    """Toggle collapse state of filter checklists for table 1."""
    if not n_clicks or not any(n_clicks):
        raise dash.exceptions.PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return [False] * len(is_open), children

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_idx = eval(button_id)["index"]

    # Update the collapse states and button icons
    new_is_open = []
    new_children = []

    for _, (col, is_open_state, child) in enumerate(
        zip([x["id"]["index"] for x in ctx.inputs_list[0]], is_open, children)
    ):
        # Update collapse state
        new_state = not is_open_state if col == button_idx else is_open_state
        new_is_open.append(new_state)

        # Update button content
        label = child[0]["props"]["children"]  # Get the column name
        new_children.append(
            [
                html.Span(label, style={"flex-grow": 1}),
                html.I(className=f"fas fa-chevron-{'up' if new_state else 'down'}"),
            ]
        )

    return new_is_open, new_children


@callback(
    Output({"type": "table-2-collapse", "index": ALL}, "is_open"),
    Output({"type": "table-2-collapse-button", "index": ALL}, "children"),
    Input({"type": "table-2-collapse-button", "index": ALL}, "n_clicks"),
    State({"type": "table-2-collapse", "index": ALL}, "is_open"),
    State({"type": "table-2-collapse-button", "index": ALL}, "children"),
    prevent_initial_call=True,
)
def toggle_table_2_collapse(n_clicks, is_open, children):
    """Toggle collapse state of filter checklists for table 2."""
    if not n_clicks or not any(n_clicks):
        raise dash.exceptions.PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return [False] * len(is_open), children

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_idx = eval(button_id)["index"]

    # Update the collapse states and button icons
    new_is_open = []
    new_children = []

    for _, (col, is_open_state, child) in enumerate(
        zip([x["id"]["index"] for x in ctx.inputs_list[0]], is_open, children)
    ):
        # Update collapse state
        new_state = not is_open_state if col == button_idx else is_open_state
        new_is_open.append(new_state)

        # Update button content
        label = child[0]["props"]["children"]  # Get the column name
        new_children.append(
            [
                html.Span(label, style={"flex-grow": 1}),
                html.I(className=f"fas fa-chevron-{'up' if new_state else 'down'}"),
            ]
        )

    return new_is_open, new_children
