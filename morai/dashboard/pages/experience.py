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

# load config
config = dh.load_config()
config_dataset = config["datasets"][config["general"]["dataset"]]
# create cards
card_list = [
    config_dataset["columns"]["exposure_amt"],
    config_dataset["columns"]["actuals_cnt"],
    config_dataset["columns"]["exposure_amt"],
    config_dataset["columns"]["actuals_amt"],
]

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
                                options=["chart", "compare"],
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
                            id="selectors",
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
                                    id="filters",
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
        Output("selectors", "children"),
        Output("filters", "children"),
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
    filter_dict = dh.generate_filters(dataset)
    selectors_default = dh.generate_selectors(
        df=dataset,
        filter_dict=filter_dict,
        config=config,
    )
    filters = filter_dict["filters"]

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
        Output("filtered-card", "children"),
        Output("chart-secondary", "children"),
    ],
    [
        # tools
        Input("tool-selector", "value"),
        # selectors
        Input({"type": "selector", "index": ALL}, "value"),
        Input({"type": "str-filter", "index": ALL}, "value"),
        Input({"type": "num-filter", "index": ALL}, "value"),
        # tabs
        Input("tabs", "active_tab"),
    ],
    [State("store-dataset", "data"), State("store-exp-filter", "data")],
    prevent_initial_call=True,
)
def update_tab_content(
    tool,
    selectors,
    str_filters,
    num_filters,
    active_tab,
    dataset,
    filter_dict,
):
    """Create chart and table for experience analysis."""
    logger.debug("creating tab content")
    # inputs
    inputs_info = dh._inputs_flatten_list(callback_context.inputs_list)
    x_axis = dh._inputs_parse_id(inputs_info, "x_axis_selector")
    y_axis = dh._inputs_parse_id(inputs_info, "y_axis_selector")
    chart_secondary = None
    # filter the dataset
    filtered_df = dataset.copy()
    for col in filter_dict["str_cols"]:
        str_values = dh._inputs_parse_id(inputs_info, col)
        if str_values:
            filtered_df = filtered_df[filtered_df[col].isin(str_values)]
    for col in filter_dict["num_cols"]:
        num_values = dh._inputs_parse_id(inputs_info, col)
        if num_values:
            filtered_df = filtered_df[
                (filtered_df[col] >= num_values[0])
                & (filtered_df[col] <= num_values[1])
            ]
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
                rates=dh._inputs_parse_id(inputs_info, "rates_selector"),
                weights=dh._inputs_parse_id(inputs_info, "weights_selector"),
                secondary=dh._inputs_parse_id(inputs_info, "secondary_selector"),
                x_bins=dh._inputs_parse_id(inputs_info, "x_bins_selector"),
            )
        else:
            chart = charters.chart(
                df=filtered_df,
                x_axis=x_axis,
                y_axis=y_axis,
                color=dh._inputs_parse_id(inputs_info, "color_selector"),
                type=dh._inputs_parse_id(inputs_info, "chart_type_selector"),
                numerator=dh._inputs_parse_id(inputs_info, "numerator_selector"),
                denominator=dh._inputs_parse_id(inputs_info, "denominator_selector"),
                x_bins=dh._inputs_parse_id(inputs_info, "x_bins_selector"),
                add_line=dh._inputs_parse_id(inputs_info, "add_line_selector"),
            )
            if dh._inputs_parse_id(inputs_info, "secondary_selector"):
                chart_secondary = charters.chart(
                    df=filtered_df,
                    x_axis=x_axis,
                    y_axis=dh._inputs_parse_id(inputs_info, "secondary_selector"),
                    color=dh._inputs_parse_id(inputs_info, "color_selector"),
                    type="bar",
                    x_bins=dh._inputs_parse_id(inputs_info, "x_bins_selector"),
                )
                chart_secondary = dcc.Graph(figure=chart_secondary)

        tab_content = dcc.Graph(figure=chart)

    elif active_tab == "tab-table":
        if tool == "compare":
            table = charters.compare_rates(
                df=filtered_df,
                x_axis=x_axis,
                rates=dh._inputs_parse_id(inputs_info, "rates_selector"),
                weights=dh._inputs_parse_id(inputs_info, "weights_selector"),
                secondary=dh._inputs_parse_id(inputs_info, "secondary_selector"),
                x_bins=dh._inputs_parse_id(inputs_info, "x_bins_selector"),
                display=False,
            )
        else:
            table = charters.chart(
                df=filtered_df,
                x_axis=x_axis,
                y_axis=y_axis,
                color=dh._inputs_parse_id(inputs_info, "color_selector"),
                type=dh._inputs_parse_id(inputs_info, "chart_type_selector"),
                numerator=dh._inputs_parse_id(inputs_info, "numerator_selector"),
                denominator=dh._inputs_parse_id(inputs_info, "denominator_selector"),
                x_bins=dh._inputs_parse_id(inputs_info, "x_bins_selector"),
                display=False,
            )

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

    return tab_content, filtered_card, chart_secondary


@callback(
    Output({"type": "selector-group", "index": ALL}, "style"),
    [Input("tool-selector", "value")],
    [State({"type": "selector-group", "index": ALL}, "id")],
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
    # determine which selectors to toggle
    if tool == "chart":
        show_selectors = chart_selectors
    elif tool == "compare":
        show_selectors = compare_selectors
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
        Output({"type": "str-filter", "index": ALL}, "value"),
        Output({"type": "num-filter", "index": ALL}, "value"),
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
