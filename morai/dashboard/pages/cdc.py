"""CDC dashboard."""

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
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

from morai.dashboard.utils import dashboard_helper as dh
from morai.experience import charters
from morai.integrations import cdc
from morai.utils import custom_logger, helpers, sql

logger = custom_logger.setup_logging(__name__)


dash.register_page(__name__, path="/cdc", name="CDC", title="morai - CDC", order=5)


#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """CDC layout."""
    return html.Div(
        [
            dcc.Store(id="store-cdc-results", storage_type="session"),
            # -----------------------------------
            dbc.Toast(
                (
                    "Need to put table in `files/integrations/cdc/cdc.sql "
                    "to display results."
                ),
                id="toast-no-database",
                header="Input Error",
                is_open=False,
                dismissable=True,
                icon="danger",
                style={"position": "fixed", "top": 100, "right": 10, "width": 350},
            ),
            html.P(
                [
                    "This page is used to analyze cause of death stats from the CDC. ",
                    html.Br(),
                    "To be able to load results there needs to be a database in ",
                    html.Code("`files/integrations/cdc/cdc.sql`"),
                    html.Br(),
                    "Data sourced from: ",
                    html.A(
                        "wonder.cdc.gov",
                        href="https://wonder.cdc.gov/",
                        target="_blank",
                    ),
                ],
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                html.Div(id="cdc-monthly-description"),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Update",
                                                    id="button-monthly",
                                                    className="btn btn-primary p-1",
                                                ),
                                            ],
                                            className="mt-2 bg-light border p-1",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-monthly",
                                            type="dot",
                                            children=html.Div(id="cdc-monthly"),
                                        ),
                                        style={"overflow": "auto"},
                                        width=8,
                                    ),
                                ],
                            ),
                        ],
                        title="CDC - Monthly",
                    ),
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                html.Div(id="cdc-mi-description"),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Update",
                                                    id="button-mi",
                                                    className="btn btn-primary p-1",
                                                ),
                                            ],
                                            className="mt-2 bg-light border p-1",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-mi",
                                            type="dot",
                                            children=html.Div(id="cdc-mi"),
                                        ),
                                        style={"overflow": "auto"},
                                        width=8,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H5(
                                                "Filters",
                                                style={
                                                    "border-bottom": "1px solid black",
                                                    "padding-bottom": "5px",
                                                },
                                            ),
                                            html.Div(
                                                id="cdc-mi-filters",
                                            ),
                                        ],
                                        width=2,
                                        className="mt-2 bg-light border p-1",
                                    ),
                                ],
                            ),
                        ],
                        title="CDC - MI",
                    ),
                ],
                start_collapsed=True,
                always_open=True,
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
        Output("cdc-monthly", "children"),
        Output("cdc-monthly-description", "children"),
        Output("toast-no-database", "is_open"),
    ],
    Input("button-monthly", "n_clicks"),
)
def display_cdc_monthly(n_clicks):
    """Create cdc monthly."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    tables = sql.get_tables(db_filepath=db_filepath)

    # check if table does not exist in database
    if "mcd18_monthly" not in tables:
        logger.error("Table `mcd18_monthly` does not exist in database.")
        return dash.no_update, dash.no_update, True

    # get the data
    mcd18_monthly = cdc.get_cdc_data_sql(
        db_filepath=db_filepath, table_name="mcd18_monthly"
    )
    last_updated = mcd18_monthly["added_at"].max()
    mcd18_monthly = mcd18_monthly[mcd18_monthly["added_at"] == last_updated]
    cdc_monthly_chart = charters.chart(
        df=mcd18_monthly,
        x_axis="month",
        y_axis="deaths",
        type="area",
    )

    cdc_monthly_description = html.Div(
        [
            html.P(
                [f"The last update was: {last_updated}"],
            ),
        ],
    )

    return dcc.Graph(figure=cdc_monthly_chart), cdc_monthly_description, False


@callback(
    [
        Output("cdc-mi", "children"),
        Output("cdc-mi-description", "children"),
        Output("cdc-mi-filters", "children"),
    ],
    Input("button-mi", "n_clicks"),
    [
        State({"type": "cdc_mi-str-filter", "index": ALL}, "value"),
        State({"type": "cdc_mi-num-filter", "index": ALL}, "value"),
    ],
)
def display_cdc_mi(n_clicks, cdc_mi_str_filters, cdc_mi_num_filters):
    """Create cdc mi."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    tables = sql.get_tables(db_filepath=db_filepath)
    states_info = dh._inputs_flatten_list(callback_context.states_list)

    # check if table does not exist in database
    if "mi" not in tables:
        logger.error("Table `mi` does not exist in database.")
        return dash.no_update, dash.no_update

    # get the data
    mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mi")

    # filter the data
    filtered_mi = dh.filter_data(df=mi, callback_context=states_info)

    # create the charts
    last_updated = filtered_mi["added_at"].max()
    rolling = 10
    mi_df = cdc.calc_mi(df=filtered_mi, rolling=rolling)
    cdc_mi_chart = charters.compare_rates(
        df=mi_df,
        x_axis="year",
        rates=["1_year_mi", f"{rolling}_year_mi"],
    )

    cdc_mi_description = html.Div(
        [
            html.P(
                [f"The last update was: {last_updated}"],
            ),
        ],
    )

    # create the filters
    cdc_mi_filters = dash.no_update
    print(cdc_mi_num_filters)
    if not cdc_mi_num_filters:
        cdc_mi_filters = dh.generate_filters(
            df=mi,
            prefix="cdc_mi",
            config=None,
            exclude_cols=["deaths", "population", "crude_rate", "added_at"],
        )["filters"]

    return dcc.Graph(figure=cdc_mi_chart), cdc_mi_description, cdc_mi_filters
