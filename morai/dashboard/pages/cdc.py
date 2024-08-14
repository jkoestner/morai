"""CDC dashboard."""

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import pandas as pd
import plotly.express as px
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
from sklearn.linear_model import LinearRegression

from morai.dashboard.components import dash_formats
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
                                html.Div(id="cdc-cod-description"),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Update",
                                                    id="button-cod",
                                                    className="btn btn-primary p-1",
                                                ),
                                            ],
                                            className="mt-2 bg-light border p-1",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-cod",
                                            type="dot",
                                            children=html.Div(id="cdc-cod"),
                                        ),
                                        style={"overflow": "auto"},
                                        width=8,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-cod-heatmap",
                                            type="dot",
                                            children=html.Div(id="cdc-cod-heatmap"),
                                        ),
                                        style={"overflow": "auto"},
                                        width=8,
                                    ),
                                ],
                            ),
                        ],
                        title="CDC - COD",
                    ),
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                html.Div(id="cdc-cod-trends-description"),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Update",
                                                    id="button-cod-trends",
                                                    className="btn btn-primary p-1",
                                                ),
                                            ],
                                            className="mt-2 bg-light border p-1",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(
                                                        label="Trends",
                                                        tab_id="tab-trends-chart",
                                                    ),
                                                    dbc.Tab(
                                                        label="Table-Amt",
                                                        tab_id="tab-trends-table-amt",
                                                    ),
                                                    dbc.Tab(
                                                        label="Table-%",
                                                        tab_id="tab-trends-table-pct",
                                                    ),
                                                ],
                                                id="tabs-cod-trends",
                                                active_tab="tab-trends-chart",
                                            ),
                                            dcc.Loading(
                                                id="loading-cdc-cod-trends",
                                                type="dot",
                                                children=html.Div(id="cdc-cod-trends"),
                                            ),
                                        ],
                                        style={"overflow": "auto"},
                                        width=8,
                                    ),
                                ],
                            ),
                        ],
                        title="CDC - COD Trends",
                    ),
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
        Output("cdc-cod", "children"),
        Output("cdc-cod-heatmap", "children"),
        Output("cdc-cod-description", "children"),
    ],
    Input("button-cod", "n_clicks"),
)
def display_cdc_cod(n_clicks):
    """Create cdc cod."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    tables = sql.get_tables(db_filepath=db_filepath)

    # check if table does not exist in database
    if "mcd99_cod" not in tables or "mcd18_cod" not in tables:
        logger.error("Table `mcd99_cod` or `mcd18_cod` does not exist in database.")
        return dash.no_update, dash.no_update

    # get the data
    mcd99_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_cod")
    mcd18_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_cod")
    category_col = "simple_grouping"

    # filter and concat
    mcd18_cod = mcd18_cod[mcd18_cod["year"] >= 2021]
    cod_all = pd.concat([mcd99_cod, mcd18_cod], ignore_index=True)
    cod_all = cod_all[(cod_all["year"] != 2024)]
    cod_all = cdc.map_reference(
        df=cod_all,
        col=category_col,
        on_dict={"icd_-_sub-chapter": "wonder_sub_chapter"},
    )

    # create totals column
    totals = cod_all.groupby("year").sum(numeric_only=True).reset_index()
    totals[category_col] = "total"
    cod_all = pd.concat([cod_all, totals], ignore_index=True)
    category_orders = charters.get_category_orders(
        df=cod_all, category=category_col, measure="deaths"
    )

    # create the charts
    last_updated = mcd18_cod["added_at"].max()
    cdc_cod_chart = charters.chart(
        df=cod_all,
        x_axis="year",
        y_axis="deaths",
        color=category_col,
        type="area",
        category_orders=category_orders,
    )

    cdc_cod_heatmap = px.treemap(
        cod_all[(cod_all["simple_grouping"] != "total") & (cod_all["year"] == 2023)],
        path=[px.Constant("all"), "simple_grouping", "icd_-_sub-chapter"],
        values="deaths",
    )

    cdc_cod_description = html.Div(
        [
            html.P(
                [f"The last update was: {last_updated}"],
            ),
        ],
    )

    return (
        dcc.Graph(figure=cdc_cod_chart),
        dcc.Graph(figure=cdc_cod_heatmap),
        cdc_cod_description,
    )


@callback(
    [
        Output("cdc-cod-trends", "children"),
        Output("cdc-cod-trends-description", "children"),
    ],
    [
        Input("button-cod-trends", "n_clicks"),
        Input("tabs-cod-trends", "active_tab"),
    ],
)
def display_cdc_cod_trends(n_clicks, active_tab):
    """Create cdc cod trends."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    tables = sql.get_tables(db_filepath=db_filepath)

    # check if table does not exist in database
    if "mcd99_cod" not in tables or "mcd18_cod" not in tables:
        logger.error("Table `mcd99_cod` or `mcd18_cod` does not exist in database.")
        return dash.no_update, dash.no_update

    # get the data
    mcd99_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_cod")
    mcd18_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_cod")
    category_col = "simple_grouping"

    # filter and concat
    mcd18_cod = mcd18_cod[mcd18_cod["year"] >= 2021]
    cod_all = pd.concat([mcd99_cod, mcd18_cod], ignore_index=True)
    cod_all = cod_all[(cod_all["year"] != 2024)]
    cod_all = cdc.map_reference(
        df=cod_all,
        col=category_col,
        on_dict={"icd_-_sub-chapter": "wonder_sub_chapter"},
    )

    # create totals column
    totals = cod_all.groupby("year").sum(numeric_only=True).reset_index()
    totals[category_col] = "total"
    cod_all = pd.concat([cod_all, totals], ignore_index=True)
    category_orders = charters.get_category_orders(
        df=cod_all, category=category_col, measure="deaths"
    )

    # train the data based on year and the category using linear regression
    train_df = cod_all[(cod_all["year"] >= 2015) & (cod_all["year"] <= 2019)]
    train_df = train_df.groupby(["year", category_col])["deaths"].sum().reset_index()

    # create the models
    models = {}
    for cod in train_df[category_col].unique():
        cod_subset = train_df[train_df[category_col] == cod]
        X = (cod_subset["year"] - 2015).values.reshape(-1, 1)
        y = cod_subset["deaths"].values
        model = LinearRegression().fit(X, y)
        models[cod] = {
            "model": model,
            "coef": model.coef_[0],
            "intercept": model.intercept_,
        }

    # make the predictions
    test_df = cod_all[(cod_all["year"] >= 2020)]
    test_df = test_df.groupby(["year", category_col])["deaths"].sum().reset_index()

    for cod, model in models.items():
        mask = test_df[category_col] == cod
        if mask.sum() > 0:
            X = (test_df.loc[mask, "year"] - 2015).values.reshape(-1, 1)
            test_df.loc[mask, "pred"] = model["model"].predict(X)

    test_df["diff_abs"] = test_df["deaths"] - test_df["pred"]
    test_df["diff_pct"] = (test_df["deaths"] - test_df["pred"]) / test_df["pred"]

    # create the tab content
    if active_tab == "tab-trends-chart":
        display = True
        y_axis = "diff_abs"
    elif active_tab == "tab-trends-table-amt":
        display = False
        y_axis = "diff_abs"
    elif active_tab == "tab-trends-table-pct":
        display = False
        y_axis = "diff_pct"
    last_updated = mcd18_cod["added_at"].max()
    test_df["year"] = test_df["year"].astype(str)

    cdc_cod_trends_chart = charters.chart(
        df=test_df,
        x_axis="year",
        y_axis=y_axis,
        color=category_col,
        type="area",
        category_orders=category_orders,
        display=display,
    )

    if active_tab == "tab-trends-chart":
        tab_content = dcc.Graph(figure=cdc_cod_trends_chart)
    else:
        pivot = cdc_cod_trends_chart.pivot(
            index=category_col, columns="year", values=y_axis
        )
        pivot.index = pd.Categorical(
            pivot.index, categories=category_orders[category_col], ordered=True
        )
        pivot = pivot.sort_index().reset_index()
        columnDefs = dash_formats.get_column_defs(pivot)
        tab_content = dag.AgGrid(
            rowData=pivot.to_dict("records"),
            columnDefs=columnDefs,
        )

    cdc_cod_trends_description = html.Div(
        [
            html.P(
                [
                    f"The trends charts is based on a linear regression of 2015-2019 cause of deaths.",
                    html.Br(),
                    html.Br(),
                    f"The last update was: {last_updated}",
                ],
            ),
        ],
    )

    return tab_content, cdc_cod_trends_description


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
                [
                    "The MI rates are calculated using a 2000 age-adjusted rate.",
                    html.Br(),
                    f"The last update was: {last_updated}",
                ],
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
