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


def layout():
    """CDC layout."""
    return html.Div(
        [
            dcc.Store(id="store-cdc-results", storage_type="session"),
            # Header section with gradient background
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                [
                                    html.I(className="fas fa-heartbeat me-2"),
                                    "CDC Analysis",
                                ],
                                className="mb-1",
                            ),
                            html.P(
                                "Analyze CDC mortality data and trends",
                                className="text-white-50 mb-0 small",
                            ),
                        ],
                        className="bg-gradient bg-primary text-white p-4 mb-4 rounded-3 shadow-sm",
                    ),
                ],
            ),
            # Toast notification
            dbc.Toast(
                "Need to put table in `files/integrations/cdc/cdc.sql` to display results.",
                id="toast-no-database",
                header="Input Error",
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
                                "About CDC Data",
                            ],
                            className="card-title mb-3",
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
                            className="card-text mb-0",
                        ),
                    ]
                ),
                className="shadow-sm mb-4",
            ),
            # Main Content Accordion
            dbc.Accordion(
                [
                    # Cause of Death Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                html.Div(
                                    id="cdc-cod-description",
                                    className="mb-3",
                                ),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-sync-alt me-2"
                                                ),
                                                "Update Analysis",
                                            ],
                                            id="button-cod",
                                            color="primary",
                                            className="w-100 shadow-sm",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-cod",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="cdc-cod",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=10,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                dcc.Loading(
                                    id="loading-cdc-cod-heatmap",
                                    type="default",
                                    color="#007bff",
                                    children=html.Div(
                                        id="cdc-cod-heatmap",
                                        className="bg-white rounded-3 shadow-sm p-3",
                                    ),
                                ),
                            ),
                        ],
                        title=[
                            html.I(className="fas fa-chart-pie me-2"),
                            "Cause of Death Analysis",
                        ],
                    ),
                    # COD Trends Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                html.Div(
                                    id="cdc-cod-trends-description",
                                    className="mb-3",
                                ),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-sync-alt me-2"
                                                ),
                                                "Update Analysis",
                                            ],
                                            id="button-cod-trends",
                                            color="primary",
                                            className="w-100 shadow-sm",
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
                                                        label_class_name="fw-bold",
                                                        active_label_class_name="text-primary",
                                                    ),
                                                    dbc.Tab(
                                                        label="Table-Amt",
                                                        tab_id="tab-trends-table-amt",
                                                        label_class_name="fw-bold",
                                                        active_label_class_name="text-primary",
                                                    ),
                                                    dbc.Tab(
                                                        label="Table-%",
                                                        tab_id="tab-trends-table-pct",
                                                        label_class_name="fw-bold",
                                                        active_label_class_name="text-primary",
                                                    ),
                                                ],
                                                id="tabs-cod-trends",
                                                active_tab="tab-trends-chart",
                                                className="mb-3",
                                            ),
                                            dcc.Loading(
                                                id="loading-cdc-cod-trends",
                                                type="default",
                                                color="#007bff",
                                                children=html.Div(
                                                    id="cdc-cod-trends",
                                                    className="bg-white rounded-3 shadow-sm p-3",
                                                ),
                                            ),
                                        ],
                                        width=10,
                                    ),
                                ],
                            ),
                        ],
                        title=[
                            html.I(className="fas fa-chart-line me-2"),
                            "Mortality Trends",
                        ],
                    ),
                    # Monthly Analysis Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                html.Div(
                                    id="cdc-monthly-description",
                                    className="mb-3",
                                ),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-sync-alt me-2"
                                                ),
                                                "Update Analysis",
                                            ],
                                            id="button-monthly",
                                            color="primary",
                                            className="w-100 shadow-sm",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-monthly",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="cdc-monthly",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=10,
                                    ),
                                ],
                            ),
                        ],
                        title=[
                            html.I(className="fas fa-calendar-alt me-2"),
                            "Monthly Analysis",
                        ],
                    ),
                    # Mortality Improvement Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                html.Div(
                                    id="cdc-mi-description",
                                    className="mb-3",
                                ),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-sync-alt me-2"
                                                ),
                                                "Update Analysis",
                                            ],
                                            id="button-mi",
                                            color="primary",
                                            className="w-100 shadow-sm",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-mi",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                id="cdc-mi",
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=8,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    html.H5(
                                                        [
                                                            html.I(
                                                                className="fas fa-filter me-2"
                                                            ),
                                                            "Filters",
                                                        ],
                                                        className="mb-0",
                                                    ),
                                                    className="bg-light",
                                                ),
                                                dbc.CardBody(
                                                    html.Div(
                                                        id="cdc-mi-filters",
                                                    ),
                                                ),
                                            ],
                                            className="shadow-sm h-100",
                                        ),
                                        width=2,
                                    ),
                                ],
                            ),
                        ],
                        title=[
                            html.I(className="fas fa-chart-bar me-2"),
                            "Mortality Improvement",
                        ],
                    ),
                ],
                start_collapsed=True,
                always_open=True,
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
    if not db_filepath.exists():
        logger.error("Database does not exist.")
        return dash.no_update, dash.no_update, dash.no_update
    tables = sql.get_tables(db_filepath=db_filepath)

    # check if table does not exist in database
    if "mcd99_cod" not in tables or "mcd18_cod" not in tables:
        logger.error("Table `mcd99_cod` or `mcd18_cod` does not exist in database.")
        return dash.no_update, dash.no_update, dash.no_update

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
                    "The trends charts is based on a linear regression of 2015-2019 cause of deaths.",
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
