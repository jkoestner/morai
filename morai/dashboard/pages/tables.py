"""
Tables dashboard.

Issue age, duration, and attained age are needed to compare mortality tables.
"""

from io import StringIO

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import numpy as np
import pandas as pd
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
            dcc.Store(id="store-table-compare", storage_type="session"),
            dcc.Store(id="store-table-1", storage_type="session"),
            dcc.Store(id="store-table-2", storage_type="session"),
            dcc.Store(id="store-table-1-select", storage_type="session"),
            dcc.Store(id="store-table-2-select", storage_type="session"),
            # -----------------------------------------------------------
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
                        dcc.RadioItems(
                            id="table-1-radio",
                            options=["soa table", "file"],
                            value="soa table",
                        ),
                        width=1,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Morality Table 1"),
                                dbc.CardBody(
                                    children=None,
                                    id="table-1-card",
                                ),
                            ],
                            color="light",
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Table Description"),
                                dbc.CardBody(
                                    children=None,
                                    id="table-1-desc",
                                ),
                            ],
                            color="light",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Filters Table-1"),
                                dbc.CardBody(
                                    children=None,
                                    id="table-1-filters",
                                ),
                            ],
                            color="light",
                        ),
                        width=2,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.RadioItems(
                            id="table-2-radio",
                            options=["soa table", "file"],
                            value="soa table",
                        ),
                        width=1,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Morality Table 2"),
                                dbc.CardBody(
                                    children=None,
                                    id="table-2-card",
                                ),
                            ],
                            color="light",
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Table Description"),
                                dbc.CardBody(
                                    children=None,
                                    id="table-2-desc",
                                ),
                            ],
                            color="light",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Filters Table-2"),
                                dbc.CardBody(
                                    children=None,
                                    id="table-2-filters",
                                ),
                            ],
                            color="light",
                        ),
                        width=2,
                    ),
                ],
                className="mb-2",
            ),
            dcc.Loading(
                id="loading-graph-contour",
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
                            id="loading-graph-compare-age",
                            type="dot",
                            children=html.Div(id="graph-compare-age"),
                        ),
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Table-1", tab_id="tab-table-1"),
                            dbc.Tab(label="Table-2", tab_id="tab-table-2"),
                            dbc.Tab(label="Compare", tab_id="tab-table-compare"),
                        ],
                        id="tabs-tables",
                        active_tab="tab-table-1",
                    ),
                    dcc.Loading(
                        id="loading-tables-tab-content",
                        type="dot",
                        children=html.Div(id="tables-tab-content"),
                    ),
                ],
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
    else:
        input_box = (
            dcc.Dropdown(
                id="table-1-id",
                options=[
                    {"label": key, "value": key}
                    for key in dh.list_files_in_folder(
                        helpers.FILES_PATH / "dataset" / "tables"
                    )
                    if key.endswith(".csv")
                ],
                placeholder="Select a file",
            ),
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
    else:
        input_box = (
            dcc.Dropdown(
                id="table-2-id",
                options=[
                    {"label": key, "value": key}
                    for key in dh.list_files_in_folder(
                        helpers.FILES_PATH / "dataset" / "tables"
                    )
                    if key.endswith(".csv")
                ],
                placeholder="Select a file",
            ),
        )
    return input_box


@callback(
    [
        Output("store-table-1", "data"),
        Output("store-table-2", "data"),
        Output("store-table-1-select", "data"),
        Output("store-table-2-select", "data"),
        Output("toast-null-tables", "is_open"),
        Output("toast-table-not-found", "is_open"),
    ],
    [Input("compare-button", "n_clicks")],
    [
        State("table-1-id", "value"),
        State("table-2-id", "value"),
    ],
    prevent_initial_call=True,
)
def get_table_data(n_clicks, table1_id, table2_id):
    """Get the table data and create a compare dataframe."""
    logger.debug(f"Retrieving tables {table1_id} and {table2_id}")
    no_upate_tuple = (dash.no_update,) * 4

    if table1_id is None or table2_id is None:
        return (*no_upate_tuple, True, False)

    # process tables
    mt = tables.MortTable()

    # table_1
    if isinstance(table1_id, str):
        try:
            table_1 = pd.read_csv(helpers.FILES_PATH / "dataset" / "tables" / table1_id)
            table_1_select_period = "Unknown"
        except FileNotFoundError:
            logger.warning(f"Table not found: {table1_id}")
            return (*no_upate_tuple, False, True)
        # add age and duration columns if not present
        table_1 = tables.add_aa_ia_dur_cols(table_1)
    else:
        try:
            table_1 = mt.build_table(table_list=[table1_id], extend=True)
            table_1_select_period = mt.select_period
        except FileNotFoundError:
            logger.warning(f"Table not found: {table1_id}")
            return (*no_upate_tuple, False, True)

    # table_2
    if isinstance(table2_id, str):
        try:
            table_2 = pd.read_csv(helpers.FILES_PATH / "dataset" / "tables" / table2_id)
            table_2_select_period = "Unknown"
        except FileNotFoundError:
            logger.warning(f"Table not found: {table2_id}")
            return (*no_upate_tuple, False, True)
        # add age and duration columns if not present
        table_2 = tables.add_aa_ia_dur_cols(table_2)
    else:
        try:
            table_2 = mt.build_table(table_list=[table2_id], extend=True)
            table_2_select_period = mt.select_period
        except FileNotFoundError:
            logger.warning(f"Table not found: {table2_id}")
            return (*no_upate_tuple, False, True)

    # serialize the dataframes
    table_1 = table_1.to_json(orient="split")
    table_2 = table_2.to_json(orient="split")

    return (
        table_1,
        table_2,
        table_1_select_period,
        table_2_select_period,
        False,
        False,
    )


@callback(
    [Output("table-1-filters", "children"), Output("table-2-filters", "children")],
    [Input("store-table-1", "data"), Input("store-table-2", "data")],
    prevent_initial_call=True,
)
def create_filters(table_1, table_2):
    """Create the filters for the tables."""
    # deserialize the tables
    table_1 = pd.read_json(StringIO(table_1), orient="split")
    table_2 = pd.read_json(StringIO(table_2), orient="split")

    # create the filters
    filters_1 = dh.generate_filters(
        df=table_1,
        prefix="table-1",
        exclude_cols=["attained_age", "issue_age", "duration", "vals", "constant"],
    ).get("filters")
    filters_2 = dh.generate_filters(
        df=table_2,
        prefix="table-2",
        exclude_cols=["attained_age", "issue_age", "duration", "vals", "constant"],
    ).get("filters")

    return filters_1, filters_2


@callback(
    [Output("table-1-desc", "children"), Output("table-2-desc", "children")],
    [
        Input({"type": "table-1-str-filter", "index": ALL}, "value"),
        Input({"type": "table-1-num-filter", "index": ALL}, "value"),
        Input({"type": "table-2-str-filter", "index": ALL}, "value"),
        Input({"type": "table-2-num-filter", "index": ALL}, "value"),
        Input("table-1-filters", "children"),
    ],
    [
        State("store-table-1", "data"),
        State("store-table-2", "data"),
        State("table-1-id", "value"),
        State("table-2-id", "value"),
        State("store-table-1-select", "data"),
        State("store-table-2-select", "data"),
    ],
    prevent_initial_call=True,
)
def update_table_descriptions(
    table_1_filter_str,
    table_1_filter_num,
    table_2_filter_str,
    table_2_filter_num,
    table_1_filters,
    table_1,
    table_2,
    table1_id,
    table2_id,
    table_1_select_period,
    table_2_select_period,
):
    """Update the table descriptions."""
    mt = tables.MortTable()

    # callback context
    inputs_info = dh._inputs_flatten_list(callback_context.inputs_list)
    filters_table_1 = dh._inputs_parse_type(
        inputs_info, "table-1-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-1-str-filter")
    filters_table_2 = dh._inputs_parse_type(
        inputs_info, "table-2-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-2-str-filter")

    # deserialize the tables
    table_1 = pd.read_json(StringIO(table_1), orient="split")
    table_2 = pd.read_json(StringIO(table_2), orient="split")

    # filter the datasets
    filtered_table_1 = dh.filter_data(df=table_1, callback_context=filters_table_1)
    filtered_table_2 = dh.filter_data(df=table_2, callback_context=filters_table_2)

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
        ]
    )

    return desc_1, desc_2


@callback(
    [
        Output("graph-contour", "children"),
        Output("slider-issue-age", "min"),
        Output("slider-issue-age", "max"),
        Output("slider-issue-age", "value"),
    ],
    [
        Input({"type": "table-1-str-filter", "index": ALL}, "value"),
        Input({"type": "table-1-num-filter", "index": ALL}, "value"),
        Input({"type": "table-2-str-filter", "index": ALL}, "value"),
        Input({"type": "table-2-num-filter", "index": ALL}, "value"),
        Input("table-1-filters", "children"),
    ],
    [State("store-table-1", "data"), State("store-table-2", "data")],
    prevent_initial_call=True,
)
def create_contour_and_sliders(
    table_1_filter_str,
    table_1_filter_num,
    table_2_filter_str,
    table_2_filter_num,
    table_1_filters,
    table_1,
    table_2,
):
    """Graph the mortality tables with a contour and comparison."""
    # callback context
    inputs_info = dh._inputs_flatten_list(callback_context.inputs_list)
    filters_table_1 = dh._inputs_parse_type(
        inputs_info, "table-1-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-1-str-filter")
    filters_table_2 = dh._inputs_parse_type(
        inputs_info, "table-2-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-2-str-filter")

    # deserialize the tables
    table_1 = pd.read_json(StringIO(table_1), orient="split")
    table_2 = pd.read_json(StringIO(table_2), orient="split")

    # filter the datasets
    filtered_table_1 = dh.filter_data(df=table_1, callback_context=filters_table_1)
    filtered_table_2 = dh.filter_data(df=table_2, callback_context=filters_table_2)
    compare_df = tables.compare_tables(filtered_table_1, filtered_table_2)

    # get the slider values
    issue_age_min = compare_df["issue_age"].min()
    issue_age_max = compare_df["issue_age"].max()
    issue_age_value = issue_age_min

    # adding additional hover data
    grouped_data = (
        compare_df.groupby(["issue_age", "duration"], observed=True)["ratio"]
        .agg("sum")
        .reset_index()
    )
    grouped_data = grouped_data.pivot(
        index="duration", columns="issue_age", values="ratio"
    )
    issue_age, duration = np.meshgrid(grouped_data.columns, grouped_data.index)
    attained_age = issue_age + duration

    # graph the tables
    graph_contour = charters.chart(
        compare_df,
        x_axis="issue_age",
        y_axis="duration",
        color="ratio",
        type="contour",
        agg="mean",
        customdata=attained_age,
        hovertemplate=(
            "issue_age: %{x}<br>"
            "duration: %{y}<br>"
            "attained_age: %{customdata}<br>"
            "ratio: %{z}<extra></extra>"
        ),
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
        Output("graph-compare-age", "children"),
    ],
    [Input("slider-issue-age", "value")],
    [
        State("table-1-filters", "children"),
        State("table-2-filters", "children"),
        State("store-table-1", "data"),
        State("store-table-2", "data"),
    ],
    prevent_initial_call=True,
)
def update_graphs_from_slider(
    issue_age_value, filters_table_1, filters_table_2, table_1, table_2
):
    """Update the compare duration graph."""
    # callback context
    inputs_info = dh._inputs_flatten_list(callback_context.inputs_list)
    filters_table_1 = dh._inputs_parse_type(
        inputs_info, "table-1-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-1-str-filter")
    filters_table_2 = dh._inputs_parse_type(
        inputs_info, "table-2-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-2-str-filter")

    # deserialize the tables
    table_1 = pd.read_json(StringIO(table_1), orient="split")
    table_2 = pd.read_json(StringIO(table_2), orient="split")

    # filter the datasets
    filtered_table_1 = dh.filter_data(df=table_1, callback_context=filters_table_1)
    filtered_table_2 = dh.filter_data(df=table_2, callback_context=filters_table_2)
    compare_df = tables.compare_tables(filtered_table_1, filtered_table_2)

    # graph the tables
    graph_compare_age = charters.compare_rates(
        compare_df[compare_df["issue_age"] == issue_age_value],
        x_axis="attained_age",
        rates=["table_1", "table_2"],
        y_log=True,
    )

    graph_compare_age = dcc.Graph(figure=graph_compare_age)

    return graph_compare_age


@callback(
    [Output("tables-tab-content", "children")],
    [
        Input("tabs-tables", "active_tab"),
        Input({"type": "table-1-str-filter", "index": ALL}, "value"),
        Input({"type": "table-1-num-filter", "index": ALL}, "value"),
        Input({"type": "table-2-str-filter", "index": ALL}, "value"),
        Input({"type": "table-2-num-filter", "index": ALL}, "value"),
        Input("table-1-filters", "children"),
    ],
    [State("store-table-1", "data"), State("store-table-2", "data")],
    prevent_initial_call=True,
)
def update_table_tabs(
    active_tab,
    table_1_filter_str,
    table_1_filter_num,
    table_2_filter_str,
    table_2_filter_num,
    table_1_filters,
    table_1,
    table_2,
):
    """Update the tables tab content."""
    # callback context
    inputs_info = dh._inputs_flatten_list(callback_context.inputs_list)
    filters_table_1 = dh._inputs_parse_type(
        inputs_info, "table-1-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-1-str-filter")
    filters_table_2 = dh._inputs_parse_type(
        inputs_info, "table-2-num-filter"
    ) + dh._inputs_parse_type(inputs_info, "table-2-str-filter")

    # deserialize the tables
    table_1 = pd.read_json(StringIO(table_1), orient="split")
    table_2 = pd.read_json(StringIO(table_2), orient="split")

    # filter the datasets
    filtered_table_1 = dh.filter_data(df=table_1, callback_context=filters_table_1)
    filtered_table_2 = dh.filter_data(df=table_2, callback_context=filters_table_2)
    compare_df = tables.compare_tables(filtered_table_1, filtered_table_2)

    if active_tab == "tab-table-1":
        table = filtered_table_1
    elif active_tab == "tab-table-2":
        table = filtered_table_2
    else:
        table = compare_df

    # deserialize the table
    columnDefs = dash_formats.get_column_defs(table)
    tab_content = dag.AgGrid(
        rowData=table.to_dict("records"),
        columnDefs=columnDefs,
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        dashGridOptions={"pagination": True},
    )

    return tab_content
