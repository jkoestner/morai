"""CDC dashboard."""

import threading
from concurrent.futures import ThreadPoolExecutor

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

# thread executor
executor = ThreadPoolExecutor(max_workers=2)
thread_lock = threading.Lock()

# initialize variables
# provides when to use data from 18 dataset as there is overlap in 99 dataset
NEW_DATASET_START_YEAR = 2021
# training for trend chart
TRAIN_START_YEAR = 2015
TRAIN_END_YEAR = 2019
# grouping for cod analysis
CATEGORY_COL = "simple_grouping"


def layout():
    """CDC layout."""
    last_updated = cdc.get_last_updated()
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
                id="cdc-toast",
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
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H5(
                                        [
                                            html.I(className="fas fa-info-circle me-2"),
                                            "About CDC Data",
                                        ],
                                        className="card-title mb-3",
                                    ),
                                    width=10,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-sync-alt me-2"
                                                ),
                                                "Update Data",
                                            ],
                                            id="button-update-cdc",
                                            color="primary",
                                            className="w-100 mb-2",
                                        ),
                                        html.Div(
                                            [
                                                "Last Updated: ",
                                                html.Span(
                                                    last_updated,
                                                    id="last-updated-text",
                                                    className="text-muted small",
                                                ),
                                            ],
                                            className="text-center small",
                                        ),
                                    ],
                                    width=2,
                                ),
                            ],
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
                                html.Br(),
                                html.Ul(
                                    [
                                        html.Li(
                                            [
                                                "1979-1998: ",
                                                html.A(
                                                    "https://wonder.cdc.gov/cmf-icd9.html",
                                                    href="https://wonder.cdc.gov/cmf-icd9.html",
                                                    target="_blank",
                                                ),
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                "1999-2020: ",
                                                html.A(
                                                    "https://wonder.cdc.gov/ucd-icd10.html",
                                                    href="https://wonder.cdc.gov/ucd-icd10.html",
                                                    target="_blank",
                                                ),
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                "2018-current: ",
                                                html.A(
                                                    "https://wonder.cdc.gov/mcd-icd10-provisional.html",
                                                    href="https://wonder.cdc.gov/mcd-icd10-provisional.html",
                                                    target="_blank",
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                "Note: The data will not match exactly to source data due to splitting the data more granularly"
                                " and losing a few claims (per year it is about .03%, 1000 claims off)",
                                html.Br(),
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
                                [
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                html.I(className="fas fa-sync-alt"),
                                                id="button-cod",
                                                color="primary",
                                                className="shadow-sm",
                                            ),
                                        ],
                                        width=1,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                dbc.Col(
                                    html.Div(
                                        id="cod-active-filters-card",
                                        className="bg-white rounded-3 shadow-sm p-3 border border-light hover-shadow",
                                    ),
                                    width="auto",
                                ),
                            ),
                            dbc.Row(
                                html.P(
                                    "The chart shows the amount of deaths by COD over the past few years. ",
                                ),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Loading(
                                                id="loading-cdc-cod",
                                                type="default",
                                                color="#007bff",
                                                children=html.Div(
                                                    id="cdc-cod",
                                                    className="bg-white rounded-3 shadow-sm p-3",
                                                ),
                                            ),
                                        ],
                                        width=10,
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
                                                        id="cdc-cod-filters",
                                                    ),
                                                ),
                                            ],
                                            className="shadow-sm h-100",
                                        ),
                                        width=2,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                html.P(
                                    [
                                        "Tree map that is a breakout of deaths from the most recent "
                                        "year in the data",
                                    ]
                                ),
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
                                className="mb-4",
                            ),
                            dbc.Row(
                                html.P(
                                    [
                                        "These tables show the top causes of death by age "
                                        "group for the most recent year in the data.",
                                    ]
                                ),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-top-causes",
                                            type="default",
                                            color="#007bff",
                                            children=html.Div(
                                                [
                                                    dbc.Tabs(
                                                        [
                                                            dbc.Tab(
                                                                html.Div(
                                                                    id="cdc-top-cause-names",
                                                                    className="bg-white rounded-3 shadow-sm p-3",
                                                                ),
                                                                label="Names",
                                                                tab_id="tab-names",
                                                            ),
                                                            dbc.Tab(
                                                                html.Div(
                                                                    id="cdc-top-cause-deaths",
                                                                    className="bg-white rounded-3 shadow-sm p-3",
                                                                ),
                                                                label="Deaths",
                                                                tab_id="tab-deaths",
                                                            ),
                                                        ],
                                                        id="cdc-top-causes-tabs",
                                                        active_tab="tab-names",
                                                    ),
                                                ],
                                                className="bg-white rounded-3 shadow-sm p-3",
                                            ),
                                        ),
                                        width=12,
                                    ),
                                ],
                                className="g-3 mb-3",
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
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            html.I(className="fas fa-sync-alt"),
                                            id="button-cod-trends",
                                            color="primary",
                                            className="shadow-sm",
                                        ),
                                        width=1,
                                    ),
                                    dbc.Col(
                                        html.P(
                                            "The trends charts is based on a linear regression of 2015-2019 cause of deaths.",
                                        ),
                                        width=11,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
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
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            html.I(className="fas fa-sync-alt"),
                                            id="button-monthly",
                                            color="primary",
                                            className="shadow-sm",
                                        ),
                                        width=1,
                                    ),
                                    dbc.Col(
                                        html.P(
                                            "This chart shows the deaths in US per month",
                                        ),
                                        width=11,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dcc.Loading(
                                        id="loading-cdc-monthly",
                                        type="default",
                                        color="#007bff",
                                        children=html.Div(
                                            id="cdc-monthly",
                                            className="bg-white rounded-3 shadow-sm p-3",
                                        ),
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
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            html.I(className="fas fa-sync-alt"),
                                            id="button-mi",
                                            color="primary",
                                            className="shadow-sm",
                                        ),
                                        width=1,
                                    ),
                                    dbc.Col(
                                        html.P(
                                            [
                                                "The MI rates are calculated using a 2000 age-adjusted rate.",
                                                html.Br(),
                                                "The crude adjusted rate is the total number of deaths divided by the "
                                                "total population weighted by the 2000 age distribution.",
                                                html.Br(),
                                                "The mortality improvement is the percentage change in the crude adjusted rate "
                                                "from one year to the next.",
                                                html.Br(),
                                                "The rolling average is a 10-year average of the mortality improvement.",
                                            ]
                                        ),
                                        width=11,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Chart and Filters Row
                            dbc.Row(
                                [
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
                                        width=10,
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
                                className="mb-4",
                            ),
                            # mi table
                            dbc.Row(
                                dcc.Loading(
                                    id="loading-cdc-mi-table",
                                    type="default",
                                    color="#007bff",
                                    children=html.Div(
                                        id="cdc-mi-table",
                                        className="bg-white rounded-3 shadow-sm p-3",
                                    ),
                                ),
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
        Output("cdc-toast", "is_open"),
        Output("cdc-toast", "children"),
        Output("cdc-toast", "icon"),
        Output("cdc-toast", "header"),
        Output("last-updated-text", "children"),
    ],
    Input("button-update-cdc", "n_clicks"),
    prevent_initial_call=True,
)
def update_cdc_data_async(n_clicks):
    """Update CDC data when button is clicked."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    def background_task():
        try:
            last_updated = pd.to_datetime(cdc.get_last_updated())
            days_since_update = (pd.Timestamp.now() - last_updated).days
            if days_since_update < 14:
                return "recent", None
            refresh_cdc_data()
            new_last_updated = cdc.get_last_updated()
            with thread_lock:
                return "success", new_last_updated
        except Exception as e:
            with thread_lock:
                return "error", str(e)

    future = executor.submit(background_task)
    status, result = future.result()  # This waits for the thread to complete
    if status == "recent":
        return (
            True,
            "Data was recently updated. Please wait 14 days before updating again.",
            "warning",
            "Warning",
            dash.no_update,
        )
    elif status == "success":
        return (
            True,
            "CDC data successfully updated!",
            "success",
            "Success",
            result,
        )
    else:
        return (
            True,
            f"Error updating CDC data: {result}",
            "danger",
            "Error",
            dash.no_update,
        )


@callback(
    [
        Output("cod-active-filters-card", "children"),
        Output("cdc-cod-filters", "children"),
        Output("cdc-cod", "children"),
        Output("cdc-cod-heatmap", "children"),
        Output("cdc-top-cause-names", "children"),
        Output("cdc-top-cause-deaths", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    Input("button-cod", "n_clicks"),
    [
        State({"type": "cdc_cod-str-filter", "index": ALL}, "value"),
        State({"type": "cdc_cod-num-filter", "index": ALL}, "value"),
    ],
)
def display_cdc_cod(n_clicks, cdc_cod_str_filters, cdc_cod_num_filters):
    """Create cdc cod."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    if not db_filepath.exists():
        logger.error("Database does not exist.")
        return dash.no_update, dash.no_update, True, "Database does not exist"
    tables = sql.get_tables(db_filepath=db_filepath)
    states_info = dh._inputs_flatten_list(callback_context.states_list)

    # check if table does not exist in database
    if "mcd99_cod" not in tables or "mcd18_cod" not in tables:
        logger.error("Table `mcd99_cod` or `mcd18_cod` does not exist in database.")
        return dash.no_update, dash.no_update, True, "Table does not exist in database"

    # get the data
    mcd99_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_cod")
    mcd18_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_cod")

    # filter and concat
    mcd18_cod = mcd18_cod[mcd18_cod["year"] >= NEW_DATASET_START_YEAR]
    cod_all = pd.concat([mcd99_cod, mcd18_cod], ignore_index=True)
    cod_all = cdc.map_reference(
        df=cod_all,
        col=CATEGORY_COL,
        on_dict={"icd_-_sub-chapter": "wonder_sub_chapter"},
    )
    cod_all = dh.filter_data(df=cod_all, callback_context=states_info)

    # create totals column
    totals = cod_all.groupby("year").sum(numeric_only=True).reset_index()
    totals[CATEGORY_COL] = "total"
    cod_all = pd.concat([cod_all, totals], ignore_index=True)
    category_orders = charters.get_category_orders(
        df=cod_all, category=CATEGORY_COL, measure="deaths"
    )

    most_recent_year = cod_all["year"].max()

    # create the charts
    cdc_cod_chart = charters.chart(
        df=cod_all,
        x_axis="year",
        y_axis="deaths",
        color=CATEGORY_COL,
        type="area",
        category_orders=category_orders,
    )

    cdc_cod_heatmap = px.treemap(
        cod_all[
            (cod_all[CATEGORY_COL] != "total") & (cod_all["year"] == most_recent_year)
        ],
        path=[px.Constant("all"), CATEGORY_COL, "icd_-_sub-chapter"],
        values="deaths",
        # skip first color to match the first chart
        color_discrete_sequence=px.colors.qualitative.Plotly[1:],
    )

    cdc_top_cause_deaths, cdc_top_cause_names = cdc.get_top_deaths_by_age_group(
        df=cod_all, year=most_recent_year
    )

    # create the filters
    cdc_cod_filters = dash.no_update
    if not cdc_cod_num_filters:
        cdc_cod_filters = dh.generate_filters(
            df=cod_all,
            prefix="cdc_cod",
            config=None,
            exclude_cols=[
                "deaths",
                "population",
                "crude_rate",
                "added_at",
                "icd_-_sub-chapter",
            ],
        )["filters"]

    # Create active filters display
    active_filters_list = dh.get_active_filters(
        callback_context=callback_context,
        str_filters=cdc_cod_str_filters,
        num_filters=cdc_cod_num_filters,
    )

    active_filters_card = dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    [
                        html.I(className="fas fa-list-ul me-2"),
                        "Active Filters",
                    ],
                    className="mb-0",
                ),
                className="bg-light",
            ),
            dbc.CardBody(
                html.Div(
                    active_filters_list if active_filters_list else "No active filters",
                    className="small",
                )
            ),
        ],
        className="shadow-sm mb-3",
    )

    return (
        active_filters_card,
        cdc_cod_filters,
        dcc.Graph(figure=cdc_cod_chart),
        dcc.Graph(figure=cdc_cod_heatmap),
        dag.AgGrid(
            rowData=cdc_top_cause_names.to_dict("records"),
            columnDefs=dash_formats.get_column_defs(cdc_top_cause_names),
            dashGridOptions={
                "defaultColDef": {
                    "width": 110,
                },
            },
        ),
        dag.AgGrid(
            rowData=cdc_top_cause_deaths.to_dict("records"),
            columnDefs=dash_formats.get_column_defs(cdc_top_cause_deaths),
            dashGridOptions={
                "defaultColDef": {
                    "width": 110,
                },
            },
        ),
        False,
        "",
    )


@callback(
    [
        Output("cdc-cod-trends", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
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
        return dash.no_update, dash.no_update, True, "Table does not exist in database"

    # get the data
    mcd99_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_cod")
    mcd18_cod = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_cod")

    # filter and concat
    mcd18_cod = mcd18_cod[mcd18_cod["year"] >= NEW_DATASET_START_YEAR]
    cod_all = pd.concat([mcd99_cod, mcd18_cod], ignore_index=True)
    cod_all = cdc.map_reference(
        df=cod_all,
        col=CATEGORY_COL,
        on_dict={"icd_-_sub-chapter": "wonder_sub_chapter"},
    )

    # create totals column
    totals = cod_all.groupby("year").sum(numeric_only=True).reset_index()
    totals[CATEGORY_COL] = "total"
    cod_all = pd.concat([cod_all, totals], ignore_index=True)
    category_orders = charters.get_category_orders(
        df=cod_all, category=CATEGORY_COL, measure="deaths"
    )

    # train the data based on year and the category using linear regression
    train_df = cod_all[
        (cod_all["year"] >= TRAIN_START_YEAR) & (cod_all["year"] <= TRAIN_END_YEAR)
    ]
    train_df = train_df.groupby(["year", CATEGORY_COL])["deaths"].sum().reset_index()

    # create the models
    models = {}
    for cod in train_df[CATEGORY_COL].unique():
        cod_subset = train_df[train_df[CATEGORY_COL] == cod]
        X = (cod_subset["year"] - TRAIN_START_YEAR).values.reshape(-1, 1)
        y = cod_subset["deaths"].values
        model = LinearRegression().fit(X, y)
        models[cod] = {
            "model": model,
            "coef": model.coef_[0],
            "intercept": model.intercept_,
        }

    # make the predictions
    test_df = cod_all[(cod_all["year"] >= (TRAIN_END_YEAR + 1))]
    test_df = test_df.groupby(["year", CATEGORY_COL])["deaths"].sum().reset_index()

    for cod, model in models.items():
        mask = test_df[CATEGORY_COL] == cod
        if mask.sum() > 0:
            X = (test_df.loc[mask, "year"] - TRAIN_START_YEAR).values.reshape(-1, 1)
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
    test_df["year"] = test_df["year"].astype(str)

    cdc_cod_trends_chart = charters.chart(
        df=test_df,
        x_axis="year",
        y_axis=y_axis,
        color=CATEGORY_COL,
        type="area",
        category_orders=category_orders,
        display=display,
    )

    if active_tab == "tab-trends-chart":
        tab_content = dcc.Graph(figure=cdc_cod_trends_chart)
    else:
        pivot = cdc_cod_trends_chart.pivot(
            index=CATEGORY_COL, columns="year", values=y_axis
        )
        pivot.index = pd.Categorical(
            pivot.index, categories=category_orders[CATEGORY_COL], ordered=True
        )
        pivot = pivot.sort_index().reset_index()
        columnDefs = dash_formats.get_column_defs(pivot)
        tab_content = dag.AgGrid(
            rowData=pivot.to_dict("records"),
            columnDefs=columnDefs,
        )

    return tab_content, False, ""


@callback(
    [
        Output("cdc-monthly", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
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
        return (
            dash.no_update,
            dash.no_update,
            True,
            "Table `mcd18_monthly` does not exist in database.",
        )

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

    return dcc.Graph(figure=cdc_monthly_chart), False, ""


@callback(
    [
        Output("cdc-mi", "children"),
        Output("cdc-mi-table", "children"),
        Output("cdc-mi-filters", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
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
    if "mcd18_mi" not in tables:
        logger.error("Table `mcd18_mi` does not exist in database.")
        return (
            dash.no_update,
            dash.no_update,
            True,
            "Table `mcd18_mi` does not exist in database",
        )

    # get the data
    mcd79_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd79_mi")
    mcd99_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_mi")
    mcd18_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_mi")
    mcd18_mi = mcd18_mi[mcd18_mi["year"] >= NEW_DATASET_START_YEAR]
    mi = pd.concat([mcd79_mi, mcd99_mi, mcd18_mi], axis=0, ignore_index=True)

    # remap the age groups
    mi = cdc.map_reference(
        df=mi, col="value", on_dict={"age_groups": "key"}, sheet_name="mapping"
    )
    mi = mi.drop("age_groups", axis=1)
    mi = mi.rename(columns={"value": "age_groups"})

    # filter the data
    filtered_mi = dh.filter_data(df=mi, callback_context=states_info)

    # create the charts
    rolling = 10
    mi_df = cdc.calc_mi(df=filtered_mi, rolling=rolling)
    cdc_mi_chart = charters.compare_rates(
        df=mi_df,
        x_axis="year",
        rates=["1_year_mi", f"{rolling}_year_mi", "whl_3"],
    )

    # mortality improvement table
    columnDefs = dash_formats.get_column_defs(mi_df)
    mi_table = dag.AgGrid(
        rowData=mi_df.sort_values(by="year", ascending=False).to_dict("records"),
        columnDefs=columnDefs,
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
    )

    # create the filters
    cdc_mi_filters = dash.no_update
    if not cdc_mi_num_filters:
        cdc_mi_filters = dh.generate_filters(
            df=mi,
            prefix="cdc_mi",
            config=None,
            exclude_cols=["deaths", "population", "crude_rate", "added_at"],
        )["filters"]

    return dcc.Graph(figure=cdc_mi_chart), mi_table, cdc_mi_filters, False, ""


@callback(
    Output({"type": "cdc_mi-collapse", "index": ALL}, "is_open"),
    Output({"type": "cdc_mi-collapse-button", "index": ALL}, "children"),
    Input({"type": "cdc_mi-collapse-button", "index": ALL}, "n_clicks"),
    State({"type": "cdc_mi-collapse", "index": ALL}, "is_open"),
    State({"type": "cdc_mi-collapse-button", "index": ALL}, "children"),
    prevent_initial_call=True,
)
def toggle_cdc_mi_collapse(n_clicks, is_open, children):
    """Toggle collapse state of filter checklists."""
    if not n_clicks or not any(n_clicks):
        raise dash.exceptions.PreventUpdate

    return dh.toggle_collapse(
        callback_context=callback_context,
        is_open=is_open,
        children=children,
    )


@callback(
    [
        Output({"type": "cdc_cod-collapse", "index": ALL}, "is_open"),
        Output({"type": "cdc_cod-collapse-button", "index": ALL}, "children"),
        Input({"type": "cdc_cod-collapse-button", "index": ALL}, "n_clicks"),
        State({"type": "cdc_cod-collapse", "index": ALL}, "is_open"),
        State({"type": "cdc_cod-collapse-button", "index": ALL}, "children"),
    ],
)
def toggle_cdc_cod_collapse(n_clicks, is_open, children):
    """Toggle collapse state of filter checklists."""
    if not n_clicks or not any(n_clicks):
        raise dash.exceptions.PreventUpdate

    return dh.toggle_collapse(
        callback_context=callback_context,
        is_open=is_open,
        children=children,
    )


#   _____                 _   _
#  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
#  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def refresh_cdc_data() -> None:
    """Refresh the cdc data."""
    try:
        db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
        mcd18_cod = cdc.get_cdc_data_xml(xml_filename="mcd18_cod.xml")
        sql.export_to_sql(
            df=mcd18_cod,
            db_filepath=db_filepath,
            table_name="mcd18_cod",
            if_exists="replace",
        )
        mcd18_monthly = cdc.get_cdc_data_xml(
            xml_filename="mcd18_monthly.xml", parse_date_col="Month"
        )
        sql.export_to_sql(
            df=mcd18_monthly,
            db_filepath=db_filepath,
            table_name="mcd18_monthly",
            if_exists="replace",
        )
        mcd18_mi = cdc.get_cdc_data_xml(xml_filename="mcd18_mi.xml")
        sql.export_to_sql(
            df=mcd18_mi,
            db_filepath=db_filepath,
            table_name="mcd18_mi",
            if_exists="replace",
        )
    except Exception as e:
        logger.error(f"Error refreshing cdc data: {e}")
