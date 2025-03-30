"""Experience dashboard."""

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import pandas as pd
import polars as pl
from dash_extensions.enrich import (
    ALL,
    Input,
    Output,
    Serverside,
    State,
    callback,
    callback_context,
    dcc,
    html,
)

from morai.dashboard.components import dash_formats
from morai.dashboard.utils import dashboard_helper as dh
from morai.experience import charters, experience
from morai.forecast import metrics
from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


dash.register_page(__name__, path="/experience", title="morai - Experience", order=2)

#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Experience layout."""
    return html.Div(
        [
            dcc.Store(id="store-exp-filter", storage_type="session"),
            _build_header(),
            _build_cards_row(),
            _build_main_content(),
            dcc.Download(id="download-dataframe-csv"),
        ],
        className="container-fluid px-4 py-3",
    )


def _build_header():
    """Build the header section with a modern gradient design."""
    return html.Div(
        [
            html.Div(
                [
                    html.H4(
                        [
                            html.I(className="fas fa-chart-line me-2"),
                            "Experience Analysis",
                        ],
                        className="mb-1",
                    ),
                    html.P(
                        "Analyze and visualize your experience data",
                        className="text-white-50 mb-0 small",
                    ),
                ],
                className="bg-gradient bg-primary text-white p-4 mb-4 rounded-3 shadow-sm",
            ),
            dbc.Row(
                html.Div(
                    id="data-description",
                    className="mb-3",
                ),
            ),
        ]
    )


def _build_cards_row():
    """Build the cards section with improved styling."""
    return dbc.Row(
        [
            dbc.Col(
                html.Div(
                    id="card",
                    className="bg-white rounded-3 shadow-sm p-3 border border-light hover-shadow",
                ),
                width="auto",
            ),
            dbc.Col(
                dcc.Loading(
                    children=[
                        html.Div(
                            id="filtered-card",
                            className="bg-white rounded-3 shadow-sm p-3 border border-light hover-shadow",
                        ),
                    ],
                    id="loading-filtered-card",
                    type="default",
                    color="#007bff",
                ),
                width="auto",
            ),
            dbc.Col(
                html.Div(
                    id="active-filters-card",
                    className="bg-white rounded-3 shadow-sm p-3 border border-light hover-shadow",
                ),
                width="auto",
            ),
        ],
        className="mb-4 g-3",
    )


def _build_main_content():
    """Build the main content section with tabs."""
    return dbc.Row(
        [
            # Selectors column (left)
            _build_selectors_column(),
            # Chart column (middle)
            dbc.Col(
                [
                    # Filter toggle button
                    dbc.Button(
                        [
                            html.I(className="fas fa-filter me-2"),
                            "Show Filters",
                        ],
                        id="open-offcanvas-button",
                        className="mb-3",
                        color="primary",
                    ),
                    # Tabs and content
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                children=[],
                                label="Chart",
                                tab_id="tab-chart",
                                label_class_name="fw-bold",
                                active_label_class_name="text-primary",
                            ),
                            dbc.Tab(
                                children=[],
                                label="Table",
                                tab_id="tab-table",
                                label_class_name="fw-bold",
                                active_label_class_name="text-primary",
                            ),
                            dbc.Tab(
                                children=[],
                                label="Rank",
                                tab_id="tab-rank",
                                label_class_name="fw-bold",
                                active_label_class_name="text-primary",
                            ),
                        ],
                        id="tabs",
                        active_tab="tab-chart",
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                id="loading-tab-content",
                                type="default",
                                color="#007bff",
                                children=html.Div(
                                    id="tab-content",
                                    className="bg-white rounded-3 shadow-sm p-4 border border-light",
                                ),
                            ),
                            dcc.Loading(
                                id="loading-chart-secondary",
                                type="default",
                                color="#007bff",
                                children=html.Div(
                                    id="chart-secondary",
                                    className="mt-4 bg-white rounded-3 shadow-sm p-4 border border-light",
                                ),
                            ),
                        ],
                        className="h-100",
                    ),
                    # Offcanvas for filters
                    dbc.Offcanvas(
                        [
                            html.H4(
                                [
                                    html.I(className="fas fa-filter me-2"),
                                    "Data Filters",
                                ],
                                className="mb-4",
                            ),
                            html.Button(
                                [
                                    html.I(className="fas fa-undo me-2"),
                                    "Reset Filters",
                                ],
                                id="reset-filters-button",
                                n_clicks=0,
                                className="btn btn-outline-primary w-100 shadow-sm mb-4",
                            ),
                            html.Div(
                                id="chart-filters",
                                className="overflow-auto custom-scrollbar",
                                style={
                                    "max-height": "calc(100vh - 300px)",
                                    "backgroundColor": "#f0f7f0",
                                    "padding": "15px",
                                    "borderRadius": "8px",
                                    "width": "100%",
                                },
                            ),
                        ],
                        id="filters-offcanvas",
                        title="",
                        placement="end",
                        scrollable=True,
                        is_open=False,
                        style={
                            "width": "300px",
                        },
                    ),
                ],
                width=9,
            ),
        ],
        className="g-3",
    )


def _build_selectors_column():
    """Build the selectors column with improved styling and spacing."""
    return dbc.Col(
        html.Div(
            [
                html.Label(
                    [html.I(className="fas fa-tools me-2"), "Analysis Tool"],
                    className="text-dark fw-semibold mb-2 fs-5",
                ),
                dcc.Dropdown(
                    id="tool-selector",
                    options=[
                        {"label": "📊 Chart Analysis", "value": "chart"},
                        {"label": "📈 Rate Comparison", "value": "compare"},
                        {"label": "🎯 Target Analysis", "value": "target"},
                    ],
                    value="chart",
                    clearable=False,
                    className="border-0 shadow-sm mb-4",
                    style={
                        "borderRadius": "8px",
                        "fontSize": "15px",
                    },
                ),
                html.Button(
                    [html.I(className="fas fa-sync-alt me-2"), "Update Analysis"],
                    id="update-content-button",
                    n_clicks=0,
                    className="btn btn-primary w-100 shadow-sm mb-4",
                    style={"borderRadius": "8px", "fontSize": "15px"},
                ),
                html.H5(
                    [html.I(className="fas fa-sliders-h me-2"), "Analysis Options"],
                    className="text-dark fw-semibold mb-3 fs-5",
                ),
                html.Div(
                    id="chart-selectors",
                    className="overflow-auto custom-scrollbar selector-container",
                    style={
                        "max-height": "calc(100vh - 300px)",
                        "backgroundColor": "#f0f7f0",
                        "padding": "15px",
                        "borderRadius": "8px",
                        "width": "100%",
                    },
                ),
            ],
            className="bg-white rounded-3 shadow-sm p-4 border border-light h-100",
            style={"minWidth": "100%"},
        ),
        width=3,
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


@callback(
    [
        Output("store-exp-filter", "data"),
        Output("chart-selectors", "children"),
        Output("chart-filters", "children"),
        Output("card", "children"),
    ],
    [Input("store-dataset", "data")],
    [State("store-config", "data")],
)
def load_data(dataset, config):
    """Load data and create selectors."""
    if dataset is None:
        raise dash.exceptions.PreventUpdate

    logger.debug("generate selectors and filters")

    filter_dict = dh.generate_filters(df=dataset, prefix="chart", config=config)
    selectors_default = dh.generate_selectors(
        config=config,
        prefix="chart",
        selector_dict={
            "x_axis": True,
            "y_axis": True,
            "target": False,
            "color": True,
            "chart_type": True,
            "numerator": True,
            "denominator": True,
            "multi_numerator": False,
            "multi_denominator": False,
            "secondary": True,
            "x_bins": True,
            "add_line": True,
            "rates": False,
            "weights": False,
            "normalize": True,
        },
    )

    card = dh.generate_card(
        df=dataset,
        card_list=dh.get_card_list(config),
        title="All Data",
        color="LightSkyBlue",
    )

    return (
        # setting a key will avoid multiple versions of the same dataset
        Serverside(filter_dict, key=f"{config['general']['dataset']}_filter"),
        selectors_default,
        filter_dict["filters"],
        card,
    )


@callback(
    [
        Output("tab-content", "children"),
        Output("chart-secondary", "children"),
        Output("filtered-card", "children"),
    ],
    [
        # tabs
        Input("tabs", "active_tab"),
        # update button
        Input("update-content-button", "n_clicks"),
        # tools
        State("tool-selector", "value"),
        # selectors
        State({"type": "chart-selector", "index": ALL}, "value"),
        State({"type": "chart-str-filter", "index": ALL}, "value"),
        State({"type": "chart-num-filter", "index": ALL}, "value"),
        # stores
        State("store-dataset", "data"),
        State("store-exp-filter", "data"),
        State("store-config", "data"),
    ],
    prevent_initial_call=True,
)
def update_tab_content(
    active_tab,
    n_clicks,
    tool,
    selectors,
    str_filters,
    num_filters,
    dataset,
    filter_dict,
    config,
):
    """Create chart and table for experience analysis."""
    if not n_clicks and not active_tab:  # Prevent update if neither trigger is present
        raise dash.exceptions.PreventUpdate

    logger.debug("creating tab content")
    config_dataset = config["datasets"][config["general"]["dataset"]]

    # callback context
    states_info = dh._inputs_flatten_list(callback_context.states_list)
    x_axis = dh._inputs_parse_id(states_info, "x_axis_selector")
    y_axis = dh._inputs_parse_id(states_info, "y_axis_selector")

    # filter the dataset - only filter what's needed
    filtered_df = dh.filter_data(df=dataset, callback_context=states_info)

    # create cards
    card_list = dh.get_card_list(config)
    filtered_card = dh.generate_card(
        df=filtered_df,
        card_list=card_list,
        title="Filtered",
        color="LightGreen",
    )

    chart_secondary = None
    tab_content = None

    # Early validation for chart/table tabs
    if (active_tab in ["tab-chart", "tab-table"]) and (
        x_axis is None or y_axis is None
    ):
        return "Select X-Axis and Y-Axis to view content", None, filtered_card

    # update tab content based on tab and tool
    if active_tab == "tab-chart":
        if tool == "compare":
            chart = charters.compare_rates(
                df=filtered_df,
                x_axis=x_axis,
                rates=dh._inputs_parse_id(states_info, "rates_selector"),
                weights=dh._inputs_parse_id(states_info, "weights_selector"),
                secondary=dh._inputs_parse_id(states_info, "secondary_selector"),
                x_bins=dh._inputs_parse_id(states_info, "x_bins_selector"),
            )
        elif tool == "chart":
            if dh._inputs_parse_id(states_info, "normalize_selector"):
                filtered_df = experience.normalize(
                    df=filtered_df,
                    features=dh._inputs_parse_id(states_info, "normalize_selector"),
                    numerator=dh._inputs_parse_id(states_info, "numerator_selector"),
                    denominator=dh._inputs_parse_id(
                        states_info, "denominator_selector"
                    ),
                    add_norm_col=False,
                )
            chart = charters.chart(
                df=filtered_df,
                x_axis=x_axis,
                y_axis=y_axis,
                color=dh._inputs_parse_id(states_info, "color_selector"),
                type=dh._inputs_parse_id(states_info, "chart_type_selector"),
                numerator=dh._inputs_parse_id(states_info, "numerator_selector"),
                denominator=dh._inputs_parse_id(states_info, "denominator_selector"),
                x_bins=dh._inputs_parse_id(states_info, "x_bins_selector"),
                add_line=dh._inputs_parse_id(states_info, "add_line_selector"),
            )
            if dh._inputs_parse_id(states_info, "secondary_selector"):
                chart_secondary = charters.chart(
                    df=filtered_df,
                    x_axis=x_axis,
                    y_axis=dh._inputs_parse_id(states_info, "secondary_selector"),
                    color=dh._inputs_parse_id(states_info, "color_selector"),
                    type="bar",
                    x_bins=dh._inputs_parse_id(states_info, "x_bins_selector"),
                )
                chart_secondary = dcc.Graph(figure=chart_secondary)
        elif tool == "target":
            chart = charters.target(
                df=filtered_df,
                target=[dh._inputs_parse_id(states_info, "target_selector")],
                features=config_dataset["columns"]["features"],
                numerator=dh._inputs_parse_id(states_info, "multi_numerator_selector"),
                denominator=dh._inputs_parse_id(
                    states_info, "multi_denominator_selector"
                ),
                add_line=dh._inputs_parse_id(states_info, "add_line_selector"),
            )
        tab_content = dcc.Graph(figure=chart)

    elif active_tab == "tab-table":
        if tool == "compare":
            table = charters.compare_rates(
                df=filtered_df,
                x_axis=x_axis,
                rates=dh._inputs_parse_id(states_info, "rates_selector"),
                weights=dh._inputs_parse_id(states_info, "weights_selector"),
                secondary=dh._inputs_parse_id(states_info, "secondary_selector"),
                x_bins=dh._inputs_parse_id(states_info, "x_bins_selector"),
                display=False,
            )
        elif tool == "chart":
            table = charters.chart(
                df=filtered_df,
                x_axis=x_axis,
                y_axis=y_axis,
                color=dh._inputs_parse_id(states_info, "color_selector"),
                type=dh._inputs_parse_id(states_info, "chart_type_selector"),
                numerator=dh._inputs_parse_id(states_info, "numerator_selector"),
                denominator=dh._inputs_parse_id(states_info, "denominator_selector"),
                x_bins=dh._inputs_parse_id(states_info, "x_bins_selector"),
                display=False,
            )
        elif tool == "target":
            # give blank table
            table = filtered_df.head(0)

        columnDefs = dash_formats.get_column_defs(table)
        export_button = html.Button(
            "Export to CSV",
            id={"type": "export-button", "tab": "table"},
            className="btn btn-primary mt-2 mb-2",
        )

        grid = dag.AgGrid(
            id={"type": "data-table", "tab": "table"},
            rowData=table.to_dict("records"),
            columnDefs=columnDefs,
            dashGridOptions={
                "enableRangeSelection": True,
                "copyHeadersToClipboard": True,
                "enableCellTextSelection": True,
                "defaultColDef": {
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                },
            },
            className="ag-theme-alpine",
            columnSize="sizeToFit",
        )

        tab_content = html.Div([export_button, grid])

    elif active_tab == "tab-rank":
        rank = metrics.ae_rank(
            df=filtered_df,
            features=config_dataset["columns"]["features"],
            actuals=config_dataset["columns"]["actuals_amt"],
            expecteds=config_dataset["columns"]["expecteds_amt"],
            exposures=config_dataset["columns"]["exposure_amt"],
        )

        columnDefs = dash_formats.get_column_defs(rank)
        export_button = html.Button(
            "Export to CSV",
            id={"type": "export-button", "tab": "rank"},
            className="btn btn-primary mt-2 mb-2",
        )

        grid = dag.AgGrid(
            id={"type": "data-table", "tab": "rank"},
            rowData=rank.to_dict("records"),
            columnDefs=columnDefs,
            dashGridOptions={
                "enableRangeSelection": True,
                "copyHeadersToClipboard": True,
                "enableCellTextSelection": True,
                "defaultColDef": {
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                },
            },
            className="ag-theme-alpine",
            columnSize="sizeToFit",
        )

        tab_content = html.Div([export_button, grid])

    return tab_content, chart_secondary, filtered_card


@callback(
    [
        Output({"type": "chart-selector-group", "index": ALL}, "style"),
    ],
    [Input("tool-selector", "value")],
    [State({"type": "chart-selector-group", "index": ALL}, "id")],
    prevent_initial_call=True,
)
def toggle_tool(tool, all_selectors):
    """Toggle selectors based on tool."""
    logger.debug("toggle selectors")
    chart_selectors = [
        "x_axis",
        "y_axis",
        "color",
        "chart_type",
        "numerator",
        "denominator",
        "secondary",
        "x_bins",
        "add_line",
        "normalize",
    ]
    compare_selectors = [
        "x_axis",
        "rates",
        "weights",
        "secondary",
        "x_bins",
    ]
    target_selectors = [
        "target",
        "multi_numerator",
        "multi_denominator",
        "add_line",
    ]
    # determine which selectors to toggle
    if tool == "chart":
        show_selectors = chart_selectors
    elif tool == "compare":
        show_selectors = compare_selectors
    elif tool == "target":
        show_selectors = target_selectors
    else:
        return [dash.no_update for _ in all_selectors]

    # hide or show selectors
    style_updates = []
    for selector_id in all_selectors:
        selector_key = selector_id["index"]
        if selector_key in show_selectors:
            style_updates.append({"display": "block"})
        else:
            style_updates.append({"display": "none"})

    return style_updates


@callback(
    [
        Output({"type": "chart-str-filter", "index": ALL}, "value"),
        Output({"type": "chart-num-filter", "index": ALL}, "value"),
    ],
    [Input("reset-filters-button", "n_clicks")],
    [State("store-dataset", "data"), State("store-exp-filter", "data")],
    prevent_initial_call=True,
)
def reset_filters(n_clicks, dataset, filter_dict):
    """Reset all filters to default values."""
    logger.debug("resetting filters")
    str_reset_values = [[]] * len(filter_dict["str_cols"])
    num_reset_values = [
        [
            dataset.select(pl.col(col).min()).collect().item(),
            dataset.select(pl.col(col).max()).collect().item(),
        ]
        for col in filter_dict["num_cols"]
    ]
    return str_reset_values, num_reset_values


@callback(
    Output("download-dataframe-csv", "data"),
    Input({"type": "export-button", "tab": ALL}, "n_clicks"),
    State({"type": "data-table", "tab": ALL}, "rowData"),
    prevent_initial_call=True,
)
def export_table(n_clicks_list, table_data_list):
    """Export table data to CSV."""
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_id = eval(button_id)
    tab = button_id["tab"]

    # Find the matching table data
    for i, table_data in enumerate(table_data_list):
        if table_data and i == next(
            (idx for idx, btn in enumerate(n_clicks_list) if btn), None
        ):
            df = pd.DataFrame(table_data)
            filename = f"experience_{tab}.csv"
            return dcc.send_data_frame(df.to_csv, filename, index=False)

    return dash.no_update


@callback(
    Output("filters-offcanvas", "is_open"),
    [Input("open-offcanvas-button", "n_clicks")],
    [State("filters-offcanvas", "is_open")],
)
def toggle_filters_offcanvas(n_clicks, is_open):
    """Toggle the filters offcanvas."""
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output({"type": "chart-collapse", "index": ALL}, "is_open"),
    Output({"type": "chart-collapse-button", "index": ALL}, "children"),
    Input({"type": "chart-collapse-button", "index": ALL}, "n_clicks"),
    State({"type": "chart-collapse", "index": ALL}, "is_open"),
    State({"type": "chart-collapse-button", "index": ALL}, "children"),
    prevent_initial_call=True,
)
def toggle_collapse(n_clicks, is_open, children):
    """Toggle collapse state of filter checklists."""
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
        zip(
            [x["id"]["index"] for x in ctx.inputs_list[0]],
            is_open,
            children,
            strict=False,
        )
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
    Output("active-filters-card", "children"),
    [
        Input({"type": "chart-str-filter", "index": ALL}, "value"),
        Input({"type": "chart-num-filter", "index": ALL}, "value"),
    ],
    [
        State({"type": "chart-str-filter", "index": ALL}, "id"),
        State({"type": "chart-num-filter", "index": ALL}, "id"),
        State("store-dataset", "data"),
    ],
)
def update_active_filters_card(
    str_filters, num_filters, str_filter_ids, num_filter_ids, dataset
):
    """Update the card showing active filters."""
    active_filters = []

    # Check string filters
    for filter_val, filter_id in zip(str_filters, str_filter_ids, strict=False):
        col = filter_id["index"]
        unique_count = dataset.select(pl.col(col).unique()).collect().height
        if filter_val and len(filter_val) < unique_count:
            active_filters.append(col)

    # Check numeric filters
    for filter_val, filter_id in zip(num_filters, num_filter_ids, strict=False):
        if filter_val:
            col = filter_id["index"]
            min_val = dataset.select(pl.col(col).min()).collect().item()
            max_val = dataset.select(pl.col(col).max()).collect().item()
            if (filter_val[0] > min_val) or (filter_val[1] < max_val):
                active_filters.append(col)

    # Create card content
    if not active_filters:
        content = [
            html.H6("Active Filters", className="mb-2"),
            html.P("None", className="text-muted mb-0"),
        ]
    else:
        content = [
            html.H6("Active Filters", className="mb-2"),
            html.Ul(
                [html.Li(col) for col in sorted(active_filters)],
                className="mb-0 ps-3",
            ),
        ]

    return content
