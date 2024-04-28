"""Experience dashboard."""

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
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
from morai.experience import charters
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
            # -------------------------------------------------------
            html.H4(
                "Experience Analysis",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            dbc.Row(
                html.Div(
                    id="data-description",
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            id="card",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Loading(
                            children=[
                                html.Div(
                                    id="filtered-card",
                                ),
                            ],
                            id="loading-filtered-card",
                            type="dot",
                        ),
                        width="auto",
                    ),
                ],
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Label("tool selector", style={"margin": "10px"}),
                            dcc.Dropdown(
                                id="tool-selector",
                                options=["chart", "compare", "target"],
                                value="chart",
                                clearable=False,
                                placeholder="Select Tool",
                            ),
                        ],
                        className="m-2 bg-light border px-1",
                    ),
                    width=2,
                ),
            ),
            dbc.Row(
                [
                    # selectors
                    dbc.Col(
                        html.Div(
                            [
                                html.Button(
                                    "Update Content",
                                    id="update-content-button",
                                    n_clicks=0,
                                    className="btn btn-primary",
                                ),
                                html.H5(
                                    "Selectors",
                                    style={
                                        "border-bottom": "1px solid black",
                                        "padding-bottom": "5px",
                                    },
                                ),
                                html.Div(
                                    id="chart-selectors",
                                ),
                            ],
                            className="m-2 bg-light border p-1",
                        ),
                        width=2,
                    ),
                    # chart
                    dbc.Col(
                        [
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="Chart", tab_id="tab-chart"),
                                    dbc.Tab(label="Table", tab_id="tab-table"),
                                    dbc.Tab(label="Rank", tab_id="tab-rank"),
                                ],
                                id="tabs",
                                active_tab="tab-chart",
                            ),
                            dcc.Loading(
                                id="loading-tab-content",
                                type="dot",
                                children=html.Div(id="tab-content"),
                            ),
                            dcc.Loading(
                                id="loading-chart-secondary",
                                type="dot",
                                children=html.Div(id="chart-secondary"),
                            ),
                        ],
                        width=8,
                    ),
                    # filters
                    dbc.Col(
                        html.Div(
                            [
                                html.H5(
                                    "Filters",
                                    style={
                                        "border-bottom": "1px solid black",
                                        "padding-bottom": "5px",
                                    },
                                ),
                                html.Button(
                                    "Reset Filters",
                                    id="reset-filters-button",
                                    n_clicks=0,
                                    className="btn btn-primary",
                                ),
                                html.Div(
                                    id="chart-filters",
                                ),
                            ],
                            className="m-2 bg-light border p-1",
                        ),
                        width=2,
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

    # create filters and selectors
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
        },
    )
    filters = filter_dict["filters"]

    # create card
    card_list = dh.get_card_list(config)

    card = dh.generate_card(
        df=dataset,
        card_list=card_list,
        title="All Data",
        color="LightSkyBlue",
    )

    return (
        # setting a key will avoid multiple versions of the same dataset
        Serverside(filter_dict, key=f"{config['general']['dataset']}_filter"),
        selectors_default,
        filters,
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
    ],
    [
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
    logger.debug("creating tab content")
    config_dataset = config["datasets"][config["general"]["dataset"]]

    # callback context
    states_info = dh._inputs_flatten_list(callback_context.states_list)
    x_axis = dh._inputs_parse_id(states_info, "x_axis_selector")
    y_axis = dh._inputs_parse_id(states_info, "y_axis_selector")
    chart_secondary = None

    # filter the dataset
    filtered_df = dh.filter_data(df=dataset, callback_context=states_info)

    # create cards
    card_list = dh.get_card_list(config)
    filtered_card = dh.generate_card(
        df=filtered_df,
        card_list=card_list,
        title="Filtered",
        color="LightGreen",
    )

    # update tab content based on tab and tool
    if x_axis is None or y_axis is None:
        return "Select X-Axis and Y-Axis to view chart"
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
        tab_content = dag.AgGrid(
            rowData=table.to_dict("records"),
            columnDefs=columnDefs,
        )

    elif active_tab == "tab-rank":
        rank = metrics.ae_rank(
            df=filtered_df,
            features=config_dataset["columns"]["features"],
            actuals=config_dataset["columns"]["actuals_amt"],
            expecteds=config_dataset["columns"]["expecteds_amt"],
            exposures=config_dataset["columns"]["exposure_amt"],
        )

        columnDefs = dash_formats.get_column_defs(rank)
        tab_content = dag.AgGrid(
            rowData=rank.to_dict("records"),
            columnDefs=columnDefs,
        )

    else:
        tab_content = None

    return tab_content, chart_secondary, filtered_card


@callback(
    Output({"type": "chart-selector-group", "index": ALL}, "style"),
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
    str_reset_values = [None] * len(filter_dict["str_cols"])
    num_reset_values = [
        [dataset[col].min(), dataset[col].max()] for col in filter_dict["num_cols"]
    ]
    return str_reset_values, num_reset_values
