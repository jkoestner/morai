"""CDC dashboard."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import dash_mantine_components as dmc
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
from morai.models import core
from morai.utils import custom_logger, helpers, sql

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/cdc", name="CDC", title="morai - CDC", order=5)

# thread executor
executor = ThreadPoolExecutor(max_workers=2)
thread_lock = threading.Lock()

# initialize variables
# provides when to use data from 18 dataset as there is overlap in 99 dataset
# the 99 dataset ends in 2020 and the 18 dataset starts in 2018
NEW_DATASET_START_YEAR = 2021
# training for cod trend and population trend
TRAIN_START_YEAR = 2023
TRAIN_END_YEAR = 2025
# grouping for cod analysis
CATEGORY_COL = "simple_grouping"
# ibnr adjustment factors for the lag week
IBNR_FACTORS = {
    0: 0.27,
    1: 0.64,
    2: 0.76,
    3: 0.85,
    4: 0.90,
    5: 0.94,
    6: 0.96,
    7: 0.97,
    8: 0.98,
    9: 0.99,
    10: 0.99,
    11: 0.99,
    12: 0.99,
    13: 0.99,
}


def layout():
    """CDC layout."""
    last_updated = cdc.get_last_updated()
    return html.Div(
        [
            dcc.Store(id="store-cdc-results", storage_type="session"),
            dcc.Download(id="download-dataframe-csv"),
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
                className="toast",
            ),
            # Description Card
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5(
                                        [
                                            html.I(className="fas fa-database me-2"),
                                            "CDC Wonder Mortality Data",
                                        ],
                                        className="card-title mb-2",
                                    ),
                                    html.P(
                                        [
                                            "Requires a local database at ",
                                            html.Code("files/integrations/cdc/cdc.sql"),
                                            html.Br(),
                                            "Data sourced from ",
                                            html.A(
                                                "wonder.cdc.gov",
                                                href="https://wonder.cdc.gov/",
                                                target="_blank",
                                            ),
                                            ": ",
                                            html.Br(),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        [
                                                            html.A(
                                                                "1979-1998 (ICD-9)",
                                                                href="https://wonder.cdc.gov/cmf-icd9.html",
                                                                target="_blank",
                                                            ),
                                                        ]
                                                    ),
                                                    html.Li(
                                                        [
                                                            html.A(
                                                                "1999-2020 (ICD-10)",
                                                                href="https://wonder.cdc.gov/ucd-icd10.html",
                                                                target="_blank",
                                                            ),
                                                        ]
                                                    ),
                                                    html.Li(
                                                        [
                                                            html.A(
                                                                "2018-present",
                                                                href="https://wonder.cdc.gov/mcd-icd10-provisional.html",
                                                                target="_blank",
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                            ),
                                            "Notes: ",
                                            html.Br(),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Small rounding variance expected (~0.03% / year)."
                                                    ),
                                                    html.Li(
                                                        [
                                                            "Cause of death assignment takes time:",
                                                            html.Ul(
                                                                [
                                                                    html.Li(
                                                                        "'Other' deaths → ~4 months"
                                                                    ),
                                                                    html.Li(
                                                                        "'Delay' deaths → ~6 months (mainly external causes)"
                                                                    ),
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="card-text mb-0 small text-muted",
                                    ),
                                ],
                                xs=12,
                                md=9,
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync-alt me-2"),
                                            "Refresh Data",
                                        ],
                                        id="button-update-cdc",
                                        color="outline-primary",
                                        className="w-100 mb-2",
                                        size="sm",
                                    ),
                                    html.Div(
                                        [
                                            html.I(
                                                className="fas fa-clock me-1 text-muted"
                                            ),
                                            html.Span(
                                                last_updated,
                                                id="last-updated-text",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="text-center small",
                                    ),
                                ],
                                xs=12,
                                md=3,
                                className="d-flex flex-column justify-content-center",
                            ),
                        ],
                        className="align-items-center",
                    )
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
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-sync-alt me-2"
                                                ),
                                                "Load Analysis",
                                            ],
                                            id="button-cod",
                                            color="primary",
                                            className="shadow-sm",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            id="cod-active-filters-card",
                                        ),
                                    ),
                                ],
                                className="mb-3 align-items-center",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Deaths by cause of death over time. Current year is annualized (365 / ",
                                    html.Span(
                                        "days elapsed",
                                        id="cod-days-elapsed-text",
                                        className="fw-semibold",
                                    ),
                                    ").",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Loading(
                                                id="loading-cdc-cod",
                                                custom_spinner=dmc.Skeleton(
                                                    visible=True, h="100%"
                                                ),
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
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Treemap breakdown of deaths by category for year ",
                                    html.Span(
                                        "—",
                                        id="cod-tree-year-text",
                                        className="fw-semibold",
                                    ),
                                    ".",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                dcc.Loading(
                                    id="loading-cdc-cod-heatmap",
                                    custom_spinner=dmc.Skeleton(visible=True, h="100%"),
                                    children=html.Div(
                                        id="cdc-cod-heatmap",
                                        className="bg-white rounded-3 shadow-sm p-3",
                                    ),
                                ),
                                className="mb-4",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Top causes of death by age group for year ",
                                    html.Span(
                                        "—",
                                        id="cod-table-year-text",
                                        className="fw-semibold",
                                    ),
                                    ".",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-top-causes",
                                            custom_spinner=dmc.Skeleton(
                                                visible=True, h="100%"
                                            ),
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
                    # COD Excess Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync-alt me-2"),
                                            "Load Trends",
                                        ],
                                        id="button-cod-trends",
                                        color="primary",
                                        className="shadow-sm",
                                    ),
                                    width="auto",
                                ),
                                className="mb-3",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    f"Excess deaths vs. a linear regression baseline trained on {TRAIN_START_YEAR}-{TRAIN_END_YEAR}. "
                                    "Current year annualized (365 / ",
                                    html.Span(
                                        "days elapsed",
                                        id="cod-trend-days-elapsed-text",
                                        className="fw-semibold",
                                    ),
                                    "). Population differences are not accounted for.",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
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
                                        custom_spinner=dmc.Skeleton(
                                            visible=True, h="100%"
                                        ),
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
                            "Cause of Death - Excess Trends",
                        ],
                    ),
                    # Population Excess Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync-alt me-2"),
                                            "Load Population Trends",
                                        ],
                                        id="button-pop-trends",
                                        color="primary",
                                        className="shadow-sm",
                                    ),
                                    width="auto",
                                ),
                                className="mb-3",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    f"Lee-Carter population model trained on {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, extrapolated through the current year. "
                                    "Current year annualized (365 / ",
                                    html.Span(
                                        "days elapsed",
                                        id="pop-trend-days-elapsed-text",
                                        className="fw-semibold",
                                    ),
                                    f"). Note: population stats have a ~2-year lag; {TRAIN_START_YEAR}-{TRAIN_END_YEAR} trend is extrapolated to fill the gap.",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(
                                                label="Excess-Chart",
                                                tab_id="tab-pop-trends-chart",
                                                label_class_name="fw-bold",
                                                active_label_class_name="text-primary",
                                            ),
                                            dbc.Tab(
                                                label="Excess-Table",
                                                tab_id="tab-pop-trends-table",
                                                label_class_name="fw-bold",
                                                active_label_class_name="text-primary",
                                            ),
                                        ],
                                        id="tabs-pop-trends",
                                        active_tab="tab-pop-trends-chart",
                                        className="mb-3",
                                    ),
                                    dcc.Loading(
                                        id="loading-cdc-pop-trends",
                                        custom_spinner=dmc.Skeleton(
                                            visible=True, h="100%"
                                        ),
                                        children=html.Div(
                                            id="cdc-pop-trends",
                                            className="bg-white rounded-3 shadow-sm p-3",
                                        ),
                                    ),
                                ],
                            ),
                        ],
                        title=[
                            html.I(className="fas fa-chart-line me-2"),
                            "Population - Excess Trends",
                        ],
                    ),
                    # Monthly Analysis Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync-alt me-2"),
                                            "Load Monthly",
                                        ],
                                        id="button-monthly",
                                        color="primary",
                                        className="shadow-sm",
                                    ),
                                    width="auto",
                                ),
                                className="mb-3",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Total US deaths per month.",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dcc.Loading(
                                        id="loading-cdc-monthly",
                                        custom_spinner=dmc.Skeleton(
                                            visible=True, h="100%"
                                        ),
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
                    # Weekly Analysis Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync-alt me-2"),
                                            "Load Weekly",
                                        ],
                                        id="button-weekly",
                                        color="primary",
                                        className="shadow-sm",
                                    ),
                                    width="auto",
                                ),
                                className="mb-3",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    f"Total US deaths per week. The deaths also have an adjustment which accounts"
                                    f"for the lag in reporting. The lag adjustments are {list(IBNR_FACTORS.values())}",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dcc.Loading(
                                        id="loading-cdc-weekly",
                                        custom_spinner=dmc.Skeleton(
                                            visible=True, h="100%"
                                        ),
                                        children=html.Div(
                                            id="cdc-weekly",
                                            className="bg-white rounded-3 shadow-sm p-3",
                                        ),
                                    ),
                                ],
                            ),
                        ],
                        title=[
                            html.I(className="fas fa-calendar-alt me-2"),
                            "Weekly Analysis",
                        ],
                    ),
                    # Mortality Improvement Section
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-sync-alt me-2"),
                                            "Load MI",
                                        ],
                                        id="button-mi",
                                        color="primary",
                                        className="shadow-sm",
                                    ),
                                    width="auto",
                                ),
                                className="mb-3",
                            ),
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Age-adjusted mortality rates (2000 standard) and year-over-year improvement. "
                                    "Crude adjusted rate = deaths / population weighted by 2000 age distribution. "
                                    "Rolling average is a 10-year window.",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            # Chart and Filters Row
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-cdc-mi",
                                            custom_spinner=dmc.Skeleton(
                                                visible=True, h="100%"
                                            ),
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
                                    custom_spinner=dmc.Skeleton(visible=True, h="100%"),
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
            if days_since_update < 5:
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
            "Data was recently updated. Please wait 5 days before updating again.",
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
        Output("cod-days-elapsed-text", "children"),
        Output("cod-tree-year-text", "children"),
        Output("cod-table-year-text", "children"),
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
    logger.debug("Loading CDC COD charts")
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
        on_dict={"icd_sub_chapter": "wonder_sub_chapter"},
    )

    # normalize the partial deaths
    cod_all["deaths"] = cod_all["deaths"].astype(float)
    data_through = pd.Timestamp(mcd18_cod["data_through"].max())
    start_of_year = pd.Timestamp(f"{data_through.year}-01-01")
    days_elapsed = (data_through - start_of_year).days + 1
    factor_parial_year = 365 / days_elapsed
    mask = cod_all["year"] == data_through.year
    cod_all.loc[mask, "deaths"] *= factor_parial_year
    cod_all = dh.filter_data(df=cod_all, callback_context=states_info)

    # create totals column
    totals = cod_all.groupby("year").sum(numeric_only=True).reset_index()
    totals[CATEGORY_COL] = "total"
    cod_all = pd.concat([cod_all, totals], ignore_index=True)
    cod_all["age_groups"] = pd.Categorical(
        cod_all["age_groups"], categories=cdc.AGE_GROUP_ORDER, ordered=True
    )
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
        path=[px.Constant("all"), CATEGORY_COL, "icd_sub_chapter"],
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
                "icd_sub_chapter",
                "crude_95_confidence_interval",
                "m33",
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
    logger.debug("Completed CDC COD charts")

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
        days_elapsed,
        most_recent_year,
        most_recent_year,
    )


@callback(
    [
        Output("cdc-cod-trends", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
        Output("cod-trend-days-elapsed-text", "children"),
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
        on_dict={"icd_sub_chapter": "wonder_sub_chapter"},
    )

    # normalize the partial deaths
    cod_all["deaths"] = cod_all["deaths"].astype(float)
    data_through = pd.Timestamp(mcd18_cod["data_through"].max())
    start_of_year = pd.Timestamp(f"{data_through.year}-01-01")
    days_elapsed = (data_through - start_of_year).days + 1
    factor_parial_year = 365 / days_elapsed
    mask = cod_all["year"] == data_through.year
    cod_all.loc[mask, "deaths"] *= factor_parial_year

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

    return tab_content, False, "", days_elapsed


@callback(
    [
        Output("cdc-pop-trends", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
        Output("pop-trend-days-elapsed-text", "children"),
    ],
    [
        Input("button-pop-trends", "n_clicks"),
        Input("tabs-pop-trends", "active_tab"),
    ],
)
def display_cdc_pop_trends(n_clicks, active_tab):
    """Create cdc pop trends."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    tables = sql.get_tables(db_filepath=db_filepath)

    # check if table does not exist in database
    if "mcd99_mi" not in tables or "mcd18_mi" not in tables:
        logger.error("Table `mcd99_mi` or `mcd18_mi` does not exist in database.")
        return dash.no_update, True, "Table does not exist in database"

    # get the data
    mcd99_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_mi")
    mcd18_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_mi")

    # filter and concat
    mcd18_mi = mcd18_mi[mcd18_mi["year"] >= NEW_DATASET_START_YEAR]
    excess = pd.concat([mcd99_mi, mcd18_mi], ignore_index=True)
    excess = cdc.map_reference(
        df=excess,
        col="value",
        on_dict={"age_groups": "key"},
        sheet_name="mapping",
        category="bin_age_int",
    )
    excess = excess.rename(columns={"value": "mapped_age"})

    # normalize the partial deaths
    excess["deaths"] = excess["deaths"].astype(float)
    data_through = pd.Timestamp(mcd18_mi["data_through"].max())
    start_of_year = pd.Timestamp(f"{data_through.year}-01-01")
    days_elapsed = (data_through - start_of_year).days + 1
    factor_parial_year = 365 / days_elapsed
    mask = excess["year"] == data_through.year
    excess.loc[mask, "deaths"] *= factor_parial_year

    # group by year and mapped age
    excess_grouped = (
        excess.groupby(["year", "mapped_age"], observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    excess_grouped = excess_grouped[(excess_grouped["year"] >= TRAIN_START_YEAR)]
    max_year = excess_grouped["year"].max()
    base_year = max_year - 2

    # update population for provision years using trend
    train_df = excess_grouped[
        (excess_grouped["year"] >= TRAIN_START_YEAR)
        & (excess_grouped["year"] <= TRAIN_END_YEAR)
    ]
    train_df = (
        train_df.groupby(["year", "mapped_age"])["population"].sum().reset_index()
    )

    # create the linear age models
    models = {}
    for age in train_df["mapped_age"].unique():
        age_subset = train_df[train_df["mapped_age"] == age]
        X = (age_subset["year"] - TRAIN_START_YEAR).values.reshape(-1, 1)
        y = age_subset["population"].values
        model = LinearRegression().fit(X, y)
        models[age] = {
            "model": model,
            "coef": model.coef_[0],
            "intercept": model.intercept_,
        }
    bases = (
        excess_grouped[excess_grouped["year"] == base_year]
        .set_index("mapped_age")["population"]
        .to_dict()
    )
    coefs = {age: m["coef"] / m["intercept"] for age, m in models.items()}

    # update population for provision years
    for year in [base_year + 1, base_year + 2]:
        mask = excess_grouped["year"] == year
        for age in excess_grouped.loc[mask, "mapped_age"].unique():
            coef = coefs.get(age, 0)
            base = bases.get(age, 0)
            idx = (excess_grouped["year"] == year) & (
                excess_grouped["mapped_age"] == age
            )
            excess_grouped.loc[idx, "population"] = (1 + coef) ** (
                year - base_year
            ) * base

    # calculate qx_raw
    excess_grouped["qx_raw"] = excess_grouped["deaths"] / excess_grouped["population"]
    forecast_years = max_year - TRAIN_END_YEAR

    # model using Lee-Carter
    train_df = excess_grouped[
        (excess_grouped["year"] >= TRAIN_START_YEAR)
        & (excess_grouped["year"] <= TRAIN_END_YEAR)
    ]
    model = core.LeeCarter(
        age_col="mapped_age",
        year_col="year",
        actual_col="deaths",
        expose_col="population",
        interval=1,
    )
    fit_df = model.structure_df(train_df)
    fit_df = model.fit(fit_df)

    # forecast
    forecast_df = model.forecast(years=forecast_years)
    forecast_df = pd.concat(
        [fit_df[["year", "mapped_age", "qx_lc"]], forecast_df], axis=0
    ).reset_index()
    excess_grouped = pd.merge(
        excess_grouped,
        forecast_df[["mapped_age", "year", "qx_lc"]],
        on=["mapped_age", "year"],
        how="left",
    )
    excess_grouped["deaths_lc"] = excess_grouped["population"] * excess_grouped["qx_lc"]

    # create the chart
    chart = charters.compare_rates(
        excess_grouped,
        x_axis="year",
        rates=["qx_raw", "qx_lc"],
        weights=["population"],
    )

    # create the table
    table = (
        excess_grouped.groupby(["year"], observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    table["excess_lc_pct"] = table["deaths"] / table["deaths_lc"]
    table["qx_raw"] = table["deaths"] / table["population"]
    table["qx_lc"] = table["deaths_lc"] / table["population"]

    if active_tab == "tab-pop-trends-chart":
        tab_content = dcc.Graph(figure=chart)
    else:
        columnDefs = dash_formats.get_column_defs(table)
        tab_content = dag.AgGrid(
            rowData=table.to_dict("records"),
            columnDefs=columnDefs,
        )

    return tab_content, False, "", days_elapsed


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

    # color mapping
    years = sorted(mcd18_monthly["year"].unique())
    current_year = years[-1]
    prior = years[:-1]
    prior_color_map = px.colors.sample_colorscale(
        "Blues", [0.3 + 0.7 * i / max(len(prior) - 1, 1) for i in range(len(prior))]
    )
    color_discrete_map = dict(zip(prior, prior_color_map, strict=False))
    color_discrete_map[current_year] = "crimson"

    # chart
    cdc_monthly_chart = charters.chart(
        df=mcd18_monthly,
        x_axis="month",
        y_axis="deaths",
        color="year",
        type="line",
        color_discrete_map=color_discrete_map,
    )

    return dcc.Graph(figure=cdc_monthly_chart), False, ""


@callback(
    [
        Output("cdc-weekly", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    Input("button-weekly", "n_clicks"),
)
def display_cdc_weekly(n_clicks):
    """Create cdc monthly."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    tables = sql.get_tables(db_filepath=db_filepath)

    # check if table does not exist in database
    if "mcd18_weekly" not in tables:
        logger.error("Table `mcd18_weekly` does not exist in database.")
        return (
            dash.no_update,
            dash.no_update,
            True,
            "Table `mcd18_weekly` does not exist in database.",
        )

    # get the data
    mcd18_weekly = cdc.get_cdc_data_sql(
        db_filepath=db_filepath, table_name="mcd18_weekly"
    )
    last_updated = mcd18_weekly["added_at"].max()
    mcd18_weekly = mcd18_weekly[mcd18_weekly["added_at"] == last_updated]
    mcd18_weekly.dropna(subset=["mmwr_week"], inplace=True)
    mcd18_weekly = mcd18_weekly.sort_values("mmwr_week_date").reset_index(drop=True)
    mcd18_weekly["recent_week"] = range(len(mcd18_weekly) - 1, -1, -1)

    # add ibnr adjusted deaths to current year
    years = sorted(mcd18_weekly["mmwr_year"].unique())
    current_year = years[-1]
    prior_years = years[:-1]
    current_year_adj = mcd18_weekly[mcd18_weekly["mmwr_year"] == current_year].copy()
    current_year_adj["deaths"] = current_year_adj["deaths"] / current_year_adj[
        "recent_week"
    ].map(IBNR_FACTORS).fillna(1.0)
    current_year_adj = current_year_adj[current_year_adj["recent_week"] >= 3]
    current_year_adj["mmwr_year"] = f"{current_year}_adj"
    mcd18_weekly = pd.concat([mcd18_weekly, current_year_adj], ignore_index=True)

    # color mapping
    prior_color_map = px.colors.sample_colorscale(
        "Blues",
        [0.3 + 0.7 * i / max(len(prior_years) - 1, 1) for i in range(len(prior_years))],
    )
    color_discrete_map = dict(zip(prior_years, prior_color_map, strict=False))
    color_discrete_map[current_year] = "crimson"
    color_discrete_map[f"{current_year}_adj"] = "crimson"

    # chart
    cdc_weekly_chart = charters.chart(
        df=mcd18_weekly,
        x_axis="mmwr_week",
        y_axis="deaths",
        color="mmwr_year",
        type="line",
        color_discrete_map=color_discrete_map,
    )
    for trace in cdc_weekly_chart.data:
        if trace.name == f"{current_year}_adj":
            trace.line.dash = "dash"

    return dcc.Graph(figure=cdc_weekly_chart), False, ""


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
        df=mi,
        col="value",
        on_dict={"age_groups": "key"},
        sheet_name="mapping",
        category="bin_age",
    )
    mi = mi.drop("age_groups", axis=1)
    mi = mi.rename(columns={"value": "age_groups"})
    mi["age_groups"] = pd.Categorical(
        mi["age_groups"], categories=cdc.AGE_GROUP_ORDER, ordered=True
    )

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
    export_button = html.Button(
        "Export to CSV",
        id={"type": "export-button", "tab": "mi", "page": "cdc"},
        className="btn btn-primary mt-2 mb-2",
    )
    grid = dag.AgGrid(
        id={"type": "data-table", "tab": "mi", "page": "cdc"},
        rowData=mi_df.sort_values(by="year", ascending=False).to_dict("records"),
        columnDefs=columnDefs,
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
    )
    mi_table = html.Div([export_button, grid])

    # create the filters
    cdc_mi_filters = dash.no_update
    if not cdc_mi_num_filters:
        cdc_mi_filters = dh.generate_filters(
            df=mi,
            prefix="cdc_mi",
            config=None,
            exclude_cols=[
                "deaths",
                "population",
                "crude_rate",
                "added_at",
                "crude_95_confidence_interval",
                "m33",
            ],
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
    """
    Refresh the cdc data.

    Includes a 15 second sleep per call, due to CDC guidelines.
    """
    try:
        db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
        mcd18_cod = cdc.get_cdc_data_xml(xml_filename="mcd18_cod.xml")
        sql.export_to_sql(
            df=mcd18_cod,
            db_filepath=db_filepath,
            table_name="mcd18_cod",
            if_exists="replace",
        )
        time.sleep(15)

        mcd18_monthly = cdc.get_cdc_data_xml(
            xml_filename="mcd18_monthly.xml", parse_date_col="Month"
        )
        sql.export_to_sql(
            df=mcd18_monthly,
            db_filepath=db_filepath,
            table_name="mcd18_monthly",
            if_exists="replace",
        )
        time.sleep(15)

        mcd18_mi = cdc.get_cdc_data_xml(xml_filename="mcd18_mi.xml")
        sql.export_to_sql(
            df=mcd18_mi,
            db_filepath=db_filepath,
            table_name="mcd18_mi",
            if_exists="replace",
        )
        time.sleep(15)

        mcd18_weekly = cdc.get_cdc_data_xml(
            xml_filename="mcd18_weekly.xml", parse_date_col="mmwr_week_date"
        )
        sql.export_to_sql(
            df=mcd18_weekly,
            db_filepath=db_filepath,
            table_name="mcd18_weekly",
            if_exists="replace",
        )
    except Exception as e:
        logger.error(f"Error refreshing cdc data: {e}")
