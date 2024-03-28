"""Experience dashboard."""

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import polars as pl
from dash import Input, Output, callback, dash_table, dcc, html

from xact.dashboard import dashboard_helper as dh
from xact.experience import charters
from xact.utils import helpers

dash.register_page(__name__, path="/")

# reading in the dataset
pl_parquet_path = helpers.ROOT_PATH / "files" / "dataset" / "mortality_grouped.parquet"
# the partitioned columns need to be casted back correctly
pl.enable_string_cache()
lzdf = pl.scan_parquet(
    pl_parquet_path,
)
mortality_df = lzdf.collect()
mortality_df = mortality_df.to_pandas()
filter_dict = dh.generate_filters(mortality_df)


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
            html.H4(
                "Experience Analysis",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            html.P(),
            html.Div(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader("Exposure Counts"),
                            dbc.CardBody(
                                html.P(
                                    dh.convert_to_short_number(
                                        mortality_df["policies_exposed"].sum()
                                    ),
                                    className="card-text",
                                )
                            ),
                        ],
                        style={"width": "15rem"},
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader("Claim Counts"),
                            dbc.CardBody(
                                html.P(
                                    dh.convert_to_short_number(
                                        mortality_df["death_count"].sum()
                                    ),
                                    className="card-text",
                                )
                            ),
                        ],
                        style={"width": "15rem"},
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader("Observation Years"),
                            dbc.CardBody(
                                html.P(
                                    mortality_df["observation_year"].nunique(),
                                    className="card-text",
                                )
                            ),
                        ],
                        style={"width": "15rem"},
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    # selectors
                    html.Div(
                        [
                            html.H5("Selectors", className="m-1 bg-light border"),
                            html.P(),
                            html.Label("X-Axis"),
                            dcc.Dropdown(
                                id="x_axis_selector",
                                options=mortality_df.columns,
                                value="observation_year",
                                clearable=False,
                                placeholder="Select X-Axis",
                            ),
                            html.Label("Y-Axis"),
                            dcc.Dropdown(
                                id="y_axis_selector",
                                options=["ratio", "risk"],
                                value="ratio",
                                clearable=False,
                                placeholder="Select Y-Axis",
                            ),
                            html.Label("Color"),
                            dcc.Dropdown(
                                id="color_selector",
                                options=mortality_df.columns,
                                value=None,
                                clearable=True,
                                placeholder="Select Color",
                            ),
                            html.P(),
                            html.P(),
                            html.H5(
                                "Additional Options", className="m-1 bg-light border"
                            ),
                            html.Label("Numerator"),
                            dcc.Dropdown(
                                id="numerator_selector",
                                options=mortality_df.columns,
                                value="death_claim_amount",
                                clearable=True,
                                placeholder="Select Numerator",
                            ),
                            html.Label("Denominator"),
                            dcc.Dropdown(
                                id="denominator_selector",
                                options=mortality_df.columns,
                                value="amount_exposed",
                                clearable=True,
                                placeholder="Select Denominator",
                            ),
                        ],
                        className="two columns",
                    ),
                    # chart
                    html.Div(
                        [
                            dcc.Graph(id="experience_chart"),
                        ],
                        className="eight columns",
                    ),
                    # filters
                    html.Div(
                        [
                            html.H5("Filters", className="m-1 bg-light border"),
                            *filter_dict["filters"],
                        ],
                        className="two columns",
                    ),
                ],
                className="row",
            ),
            # html.Div(
            #     [
            #         dag.AgGrid(
            #             columnDefs=[{"field": i} for i in df.columns],
            #             rowData=df.to_dict("records"),
            #         )
            #     ],
            #     className="row",
            # ),
        ],
        className="container",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


@callback(
    Output("experience_chart", "figure"),
    [
        Input("x_axis_selector", "value"),
        Input("y_axis_selector", "value"),
        Input("color_selector", "value"),
        Input("numerator_selector", "value"),
        Input("denominator_selector", "value"),
        Input({"type": "str-filter", "index": dash.dependencies.ALL}, "value"),
        Input({"type": "num-filter", "index": dash.dependencies.ALL}, "value"),
    ],
)
def update_dashboard(
    x_axis, y_axis, color, numerator, denominator, str_filters, num_filters
):
    """Create chart and table for experience analysis."""
    if x_axis is None or y_axis is None:
        return None

    filtered_df = mortality_df.copy()
    for col, str_values in zip(filter_dict["str_cols"], str_filters):
        if str_values:
            filtered_df = filtered_df[filtered_df[col].isin(str_values)]
    for col, num_values in zip(filter_dict["num_cols"], num_filters):
        if num_values:
            filtered_df = filtered_df[
                (filtered_df[col] >= num_values[0])
                & (filtered_df[col] <= num_values[1])
            ]

    chart = charters.chart(
        df=filtered_df,
        x_axis=x_axis,
        y_axis=y_axis,
        color=color,
        type="line",
        numerator=numerator,
        denominator=denominator,
    )

    return chart
