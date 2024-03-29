"""Experience dashboard."""

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import polars as pl
from dash import Input, Output, callback, dcc, html

from xact.dashboard import dashboard_helper as dh
from xact.experience import charters
from xact.forecast import metrics
from xact.utils import helpers

dash.register_page(__name__, path="/")

# reading in the dataset
pl_parquet_path = helpers.ROOT_PATH / "files" / "dataset" / "mortality_grouped.parquet"
pl.enable_string_cache()
lzdf = pl.scan_parquet(
    pl_parquet_path,
)
mortality_df = lzdf.collect()
mortality_df = mortality_df.to_pandas()
# create dashboard dictionary
dash_dict = dh.generate_dash_dict(mortality_df)
# assign columns
actuals_amt_col = "death_claim_amount"
exposure_amt_col = "amount_exposed"
expecteds_amt_col = "exp_amt_vbt15"
actuals_cnt_col = "death_count"
exposure_cnt_col = "policies_exposed"
features = [
    "observation_year",
    "sex",
    "smoker_status",
    "insurance_plan",
    "issue_age",
    "duration",
    "face_amount_band",
    "issue_year",
    "attained_age",
    "soa_post_lvl_ind",
    "number_of_pfd_classes",
    "preferred_class",
]
# create cards
card = dh.generate_card(
    df=mortality_df,
    card_list=[exposure_cnt_col, actuals_cnt_col, exposure_amt_col, actuals_amt_col],
    title="Data",
    color="LightSkyBlue",
)
filtered_card_default = dh.generate_card(
    df=mortality_df,
    card_list=[exposure_cnt_col, actuals_cnt_col, exposure_amt_col, actuals_amt_col],
    title="Filtered",
    color="LightGreen",
)
selectors_default = dh.generate_selectors(
    df=mortality_df,
    dash_dict=dash_dict,
    actuals_col=actuals_amt_col,
    exposure_col=exposure_amt_col,
)


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
            dcc.Location(id="url", refresh=False),
            html.H4(
                "Experience Analysis",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            html.P(),
            dbc.Row(
                [
                    dbc.Col(card, width="auto"),
                    dbc.Col(
                        dcc.Loading(
                            children=[
                                html.Div(
                                    id="filtered-card",
                                    children=filtered_card_default,
                                ),
                            ],
                            id="loading-card",
                            type="dot",
                        ),
                        width="auto",
                    ),
                ],
            ),
            html.P(),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Label("tool selector"),
                            dcc.Dropdown(
                                id="tool_selector",
                                options=["chart", "compare"],
                                value="chart",
                                clearable=False,
                                placeholder="Select Tool",
                            ),
                        ],
                        className="m-1 bg-light border",
                    ),
                    width="auto",
                ),
            ),
            dbc.Row(
                [
                    # selectors
                    dbc.Col(
                        html.Div(
                            id="selectors",
                            className="m-1 bg-light border",
                            children=selectors_default,
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
                                id="loading-content",
                                type="dot",
                                children=html.Div(id="tab-content"),
                            ),
                        ],
                        width=8,
                    ),
                    # filters
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Filters"),
                                *dash_dict["filters"],
                            ],
                            className="m-1 bg-light border",
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
    Output("tab-content", "children"),
    Output("filtered-card", "children"),
    [
        # tools
        Input("tool_selector", "value"),
        # selectors
        Input("x_axis_selector", "value"),
        Input("y_axis_selector", "value"),
        Input("color_selector", "value"),
        Input("numerator_selector", "value"),
        Input("denominator_selector", "value"),
        Input({"type": "str-filter", "index": dash.dependencies.ALL}, "value"),
        Input({"type": "num-filter", "index": dash.dependencies.ALL}, "value"),
        Input("secondary_selector", "value"),
        Input("x_bins_selector", "value"),
        Input("weights_selector", "value"),
        Input("rates_selector", "value"),
        # tabs
        Input("tabs", "active_tab"),
    ],
)
def update_tab_content(
    tool,
    x_axis,
    y_axis,
    color,
    numerator,
    denominator,
    str_filters,
    num_filters,
    secondary,
    x_bins,
    weights,
    rates,
    active_tab,
):
    """Create chart and table for experience analysis."""
    # filter the dataset
    filtered_df = mortality_df.copy()
    for col, str_values in zip(dash_dict["str_cols"], str_filters):
        if str_values:
            filtered_df = filtered_df[filtered_df[col].isin(str_values)]
    for col, num_values in zip(dash_dict["num_cols"], num_filters):
        if num_values:
            filtered_df = filtered_df[
                (filtered_df[col] >= num_values[0])
                & (filtered_df[col] <= num_values[1])
            ]
    filtered_card = dh.generate_card(
        df=filtered_df,
        card_list=[
            exposure_cnt_col,
            actuals_cnt_col,
            exposure_amt_col,
            actuals_amt_col,
        ],
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
                rates=[rates],
                weights=[weights],
                secondary=secondary,
                x_bins=x_bins,
            )
        else:
            chart = charters.chart(
                df=filtered_df,
                x_axis=x_axis,
                y_axis=y_axis,
                color=color,
                type="line",
                numerator=numerator,
                denominator=denominator,
                x_bins=x_bins,
            )

        tab_content = dcc.Graph(figure=chart)

    elif active_tab == "tab-table":
        if tool == "compare":
            table = charters.compare_rates(
                df=filtered_df,
                x_axis=x_axis,
                rates=[rates],
                weights=[weights],
                secondary=secondary,
                x_bins=x_bins,
                display=False,
            )
        else:
            table = charters.chart(
                df=filtered_df,
                x_axis=x_axis,
                y_axis=y_axis,
                color=color,
                type="line",
                numerator=numerator,
                denominator=denominator,
                x_bins=x_bins,
                display=False,
            )

        tab_content = dag.AgGrid(
            rowData=table.to_dict("records"),
            columnDefs=[{"field": i} for i in table.columns],
        )

    elif active_tab == "tab-rank":
        rank = metrics.ae_rank(
            df=filtered_df,
            features=features,
            actuals=actuals_amt_col,
            expecteds=expecteds_amt_col,
            exposures=exposure_amt_col,
        )

        tab_content = dag.AgGrid(
            rowData=rank.to_dict("records"),
            columnDefs=[{"field": i} for i in rank.columns],
        )

    else:
        tab_content = None

    return tab_content, filtered_card


@callback(
    Output("selectors", "children"),
    Input("tool_selector", "value"),
)
def update_selectors(
    tool,
):
    """Create chart and table for experience analysis."""
    selectors = dh.generate_selectors(
        df=mortality_df,
        dash_dict=dash_dict,
        actuals_col=actuals_amt_col,
        exposure_col=exposure_amt_col,
        tool=tool,
    )

    return selectors
