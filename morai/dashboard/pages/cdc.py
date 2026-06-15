"""
CDC dashboard.

If running dashboard in container, ensure timeout is long enough for CDC queries
to run (can take up to 150 seconds). Timeout is setup in compose file.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

from morai.dashboard.components import common_build, dash_formats
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
DAYS_SINCE_LAST_UPDATE = 1
CDC_WAIT_TIME_SECONDS = 16
FILTER_MI_PREFIX = "cdc-mi"
FILTER_COD_PREFIX = "cdc-cod"
FILTER_COD_TRENDS_PREFIX = "cdc-cod-trends"
FILTER_MONTHLY_PREFIX = "cdc-monthly"
# provides when to use data from 18 dataset as there is overlap in 99 dataset
# the 99 dataset ends in 2020 and the 18 dataset starts in 2018
NEW_DATASET_START_YEAR = 2021
# training for cod trend and population trend
TRAIN_START_YEAR = 2010
TRAIN_END_YEAR = 2019
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
PARTIAL_WEEKS_TO_EXCLUDE = 7
# mi
ROLLING_MI_YEARS = 10


def layout():
    """CDC layout."""
    last_updated_meta = cdc.get_last_updated()
    recent_week = last_updated_meta["recent_week"]
    ibnr_factors_included = {
        recent_week - k: v
        for k, v in IBNR_FACTORS.items()
        if k >= PARTIAL_WEEKS_TO_EXCLUDE
    }
    ibnr_factors_excluded = {
        recent_week - k: v
        for k, v in IBNR_FACTORS.items()
        if k < PARTIAL_WEEKS_TO_EXCLUDE
    }
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
                                                                "2018-present (ICD-10)",
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
                                                                        "'Other' deaths → ~4 months (contributes to all causes)"
                                                                    ),
                                                                    html.Li(
                                                                        "'Delay' deaths → ~6 months (mainly external causes)"
                                                                    ),
                                                                    html.Li(
                                                                        f"The partial {last_updated_meta['data_through'].year} data is annualized using a factor of (365 / "
                                                                        f"{last_updated_meta['days_elapsed']}). "
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
                                            html.Label(
                                                "Last Updated: ",
                                            ),
                                            html.Span(
                                                pd.Timestamp(
                                                    last_updated_meta["last_updated"]
                                                ).date(),
                                                id="last-updated-text",
                                            ),
                                            html.Br(),
                                            html.I(
                                                className="fas fa-clock me-1 text-muted"
                                            ),
                                            html.Label(
                                                "Data Through: ",
                                            ),
                                            html.Span(
                                                last_updated_meta[
                                                    "data_through"
                                                ].date(),
                                                id="data-through-text",
                                            ),
                                        ],
                                        className="text-center small text-muted",
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
                                    "The heatmap and top causes use the most recent year of data.",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        common_build._build_tabbed_content(
                                            tabs=[
                                                common_build.TabSpec(
                                                    label="COD time", tab_id="cod-time"
                                                ),
                                                common_build.TabSpec(
                                                    label="COD Heatmap",
                                                    tab_id="cod-heatmap",
                                                ),
                                                common_build.TabSpec(
                                                    label="COD Top Causes - Names",
                                                    tab_id="cod-top-causes-names",
                                                ),
                                                common_build.TabSpec(
                                                    label="COD Top Causes - Deaths",
                                                    tab_id="cod-top-causes-deaths",
                                                ),
                                            ],
                                            prefix=FILTER_COD_PREFIX,
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
                                                        id=f"{FILTER_COD_PREFIX}-filters",
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
                                    "Standardized mortality rate compared to a linear regression baseline trained on "
                                    f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}. ",
                                    html.Br(),
                                    "The 'cause of death excess' won't match the 'population excess' as the 'cause of death' "
                                    "uses standardized deaths while the 'population' uses crude deaths. As the population gets "
                                    "older the 'population excess' metric will be a better indicator of excess mortality trends.",
                                    # lightbulb observations
                                    html.I(
                                        className="fas fa-lightbulb ms-2 text-warning",
                                        id="cod-trends-obs-trigger",
                                        style={"cursor": "pointer"},
                                    ),
                                    dbc.Popover(
                                        [
                                            dbc.PopoverHeader(
                                                [
                                                    html.I(
                                                        className="fas fa-lightbulb me-2 text-warning"
                                                    ),
                                                    "Observations (through 2025)",
                                                ]
                                            ),
                                            dbc.PopoverBody(
                                                html.Ul(
                                                    [
                                                        html.Li(
                                                            [
                                                                html.Span(
                                                                    "↓ ",
                                                                    className="text-success fw-bold",
                                                                ),
                                                                html.Strong(
                                                                    "Improving: "
                                                                ),
                                                                "respiratory, nervous, external",
                                                            ],
                                                            className="small",
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Span(
                                                                    "↑ ",
                                                                    className="text-danger fw-bold",
                                                                ),
                                                                html.Strong(
                                                                    "Deteriorating: "
                                                                ),
                                                                "neoplasms, circulatory",
                                                            ],
                                                            className="small mb-0",
                                                        ),
                                                    ],
                                                    className="mb-0 ps-3",
                                                ),
                                            ),
                                        ],
                                        target="cod-trends-obs-trigger",
                                        trigger="hover focus click",
                                        placement="bottom",
                                    ),
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        common_build._build_tabbed_content(
                                            tabs=[
                                                common_build.TabSpec(
                                                    label="Trends", tab_id="trends"
                                                ),
                                                common_build.TabSpec(
                                                    label="Actuals",
                                                    tab_id="actuals",
                                                ),
                                                common_build.TabSpec(
                                                    label="Actual - Predicted",
                                                    tab_id="table-diff",
                                                ),
                                                common_build.TabSpec(
                                                    label="Actual / Predicted",
                                                    tab_id="table-pct",
                                                ),
                                                common_build.TabSpec(
                                                    label="Data",
                                                    tab_id="data",
                                                ),
                                            ],
                                            prefix=FILTER_COD_TRENDS_PREFIX,
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
                                                        [
                                                            html.Label(
                                                                "Simple Grouping",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id={
                                                                    "type": "filter-str",
                                                                    "prefix": FILTER_COD_TRENDS_PREFIX,
                                                                    "index": "simple_grouping",
                                                                },
                                                                options=[
                                                                    {
                                                                        "label": "total",
                                                                        "value": "total",
                                                                    }
                                                                ],
                                                                value="total",
                                                                clearable=False,
                                                                className="ms-2",
                                                            ),
                                                            html.Label(
                                                                "Age Group",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id={
                                                                    "type": "filter-str",
                                                                    "prefix": FILTER_COD_TRENDS_PREFIX,
                                                                    "index": "age_groups",
                                                                },
                                                                options=[],
                                                                multi=True,
                                                                clearable=True,
                                                                className="ms-2",
                                                            ),
                                                        ],
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
                                    f"Lee-Carter population model trained on {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, extrapolated through the current year. ",
                                    html.Br(),
                                    f"Note: population stats have a ~2-year lag; {TRAIN_START_YEAR}-{TRAIN_END_YEAR} trend is extrapolated to fill the gap.",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                common_build._build_tabbed_content(
                                    tabs=[
                                        common_build.TabSpec(
                                            label="Excess Chart", tab_id="excess-chart"
                                        ),
                                        common_build.TabSpec(
                                            label="Excess Table",
                                            tab_id="excess-table",
                                        ),
                                    ],
                                    prefix="cdc-pop-trends",
                                ),
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
                                    html.Br(),
                                    "The COD chart defaults to using PIC (pneumonia, infections, covid) which shows the "
                                    "seasonal pattern. PIC may have an indirect effect on other COD such "
                                    "as circulatory",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        common_build._build_tabbed_content(
                                            tabs=[
                                                common_build.TabSpec(
                                                    label="All-causes Chart",
                                                    tab_id="chart",
                                                ),
                                                common_build.TabSpec(
                                                    label="All-causes Table",
                                                    tab_id="table",
                                                ),
                                                common_build.TabSpec(
                                                    label="COD Chart",
                                                    tab_id="pic-chart",
                                                ),
                                            ],
                                            prefix=FILTER_MONTHLY_PREFIX,
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
                                                        [
                                                            html.Label(
                                                                "Simple Chapter",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id={
                                                                    "type": "filter-str",
                                                                    "prefix": FILTER_MONTHLY_PREFIX,
                                                                    "index": "simple_chapter",
                                                                },
                                                                # default options
                                                                options=[
                                                                    {
                                                                        "label": "infectious",
                                                                        "value": "infectious",
                                                                    },
                                                                    {
                                                                        "label": "respiratory",
                                                                        "value": "respiratory",
                                                                    },
                                                                    {
                                                                        "label": "special",
                                                                        "value": "special",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "infectious",
                                                                    "respiratory",
                                                                    "special",
                                                                ],
                                                                multi=True,
                                                                clearable=True,
                                                                className="ms-2",
                                                            ),
                                                        ],
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
                                    "Total US deaths per week. The deaths also have an adjustment which accounts "
                                    "for the lag in reporting.",
                                    html.Br(),
                                    f"Excluding recent weeks due to incomplete factors, which are {ibnr_factors_excluded}.",
                                    html.Br(),
                                    f"Lag adjustments for recent weeks are {ibnr_factors_included}.",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                common_build._build_tabbed_content(
                                    tabs=[
                                        common_build.TabSpec(
                                            label="All-causes Chart", tab_id="chart"
                                        ),
                                        common_build.TabSpec(
                                            label="All-causes Table",
                                            tab_id="table",
                                        ),
                                        common_build.TabSpec(
                                            label="Flu Chart",
                                            tab_id="flu-chart",
                                        ),
                                        common_build.TabSpec(
                                            label="Flu Table",
                                            tab_id="flu-table",
                                        ),
                                    ],
                                    prefix="cdc-weekly",
                                ),
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
                                    "Mortality improvement calculated using the crude adjusted rate, which accounts for changes in age distribution over time. ",
                                    html.Br(),
                                    "Crude adjusted rate = weighted sum (deaths / population) weighted by 2000 population age distribution. (per 100,000 people) ",
                                    html.Br(),
                                    "whl = whittaker-henderson-lowrie method for smoothing (order = 3, lamda=400)",
                                ],
                                color="info",
                                className="py-2 mb-3 small",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        common_build._build_tabbed_content(
                                            tabs=[
                                                common_build.TabSpec(
                                                    label="Year-Chart",
                                                    tab_id="mi-year-chart",
                                                ),
                                                common_build.TabSpec(
                                                    label="Year-Table",
                                                    tab_id="mi-year-table",
                                                ),
                                                common_build.TabSpec(
                                                    label="Age-Chart",
                                                    tab_id="mi-age-chart",
                                                ),
                                            ],
                                            prefix=FILTER_MI_PREFIX,
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
                                                        id=f"{FILTER_MI_PREFIX}-filters",
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
            last_update_meta = cdc.get_last_updated()
            last_updated = pd.to_datetime(last_update_meta["last_updated"])
            days_since_update = (pd.Timestamp.now().date() - last_updated.date()).days
            # check if data was updated recently
            if days_since_update < DAYS_SINCE_LAST_UPDATE:
                return "recent", None
            # check if data is newer than the last update
            data_through = pd.to_datetime(last_update_meta["data_through"])
            new_data_through = cdc.get_cdc_data_xml(xml_filename="mcd18_check.xml")[
                "data_through"
            ].unique()
            new_last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if new_data_through <= data_through:
                return "no_update", None
            # get new data and update the database
            time.sleep(CDC_WAIT_TIME_SECONDS)
            refresh_cdc_data()
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
            f"Data was recently updated. Please wait {DAYS_SINCE_LAST_UPDATE} days before updating again.",
            "warning",
            "Warning",
            dash.no_update,
        )
    elif status == "no_update":
        return (
            True,
            "No new data available. Please check back later.",
            "warning",
            "warning",
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
        Output(f"{FILTER_COD_PREFIX}-tab-content", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    [
        Input(f"{FILTER_COD_PREFIX}-tabs", "active_tab"),
        Input("button-cod", "n_clicks"),
    ],
    [
        State(
            {"type": "filter-str", "prefix": FILTER_COD_PREFIX, "index": ALL}, "value"
        ),
        State(
            {"type": "filter-num", "prefix": FILTER_COD_PREFIX, "index": ALL}, "value"
        ),
    ],
)
def display_cdc_cod(active_tab, n_clicks, cdc_cod_str_filters, cdc_cod_num_filters):
    """Create cdc cod."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    logger.debug("Loading CDC COD charts")
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    if not db_filepath.exists():
        logger.error("Database does not exist.")
        return dash.no_update, dash.no_update, True, "Database does not exist"
    states_info = dh._inputs_flatten_list(callback_context.states_list)

    # get the data
    try:
        mcd99_cod = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd99_cod"
        )
        mcd18_cod = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd18_cod"
        )
    except Exception as e:
        logger.error(f"Failed to load table: {e}")
        return dash.no_update, dash.no_update, True, f"Failed to load data: {e}"

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

    # create tab content
    if active_tab == f"{FILTER_COD_PREFIX}-tab-cod-time":
        cdc_cod_chart = charters.chart(
            df=cod_all,
            x_axis="year",
            y_axis="deaths",
            color=CATEGORY_COL,
            type="area",
            category_orders=category_orders,
        )
        tab_content = html.Div([dcc.Graph(figure=cdc_cod_chart)])
    elif active_tab == f"{FILTER_COD_PREFIX}-tab-cod-heatmap":
        cdc_cod_heatmap = px.treemap(
            cod_all[
                (cod_all[CATEGORY_COL] != "total")
                & (cod_all["year"] == most_recent_year)
            ],
            path=[px.Constant("all"), CATEGORY_COL, "icd_sub_chapter"],
            values="deaths",
            # skip first color to match the first chart
            color_discrete_sequence=px.colors.qualitative.Plotly[1:],
        )
        tab_content = html.Div([dcc.Graph(figure=cdc_cod_heatmap)])
    elif active_tab == f"{FILTER_COD_PREFIX}-tab-cod-top-causes-names":
        cdc_top_cause_deaths, cdc_top_cause_names = cdc.get_top_deaths_by_age_group(
            df=cod_all, year=most_recent_year
        )
        columnDefs = dash_formats.get_column_defs(cdc_top_cause_names)
        grid = dag.AgGrid(
            id={
                "type": "data-table",
                "tab": "cod-top-causes-names",
                "page": FILTER_COD_PREFIX,
            },
            rowData=cdc_top_cause_names.to_dict("records"),
            columnDefs=columnDefs,
            dashGridOptions={
                "defaultColDef": {
                    "width": 110,
                },
            },
        )
        tab_content = html.Div([grid])
    elif active_tab == f"{FILTER_COD_PREFIX}-tab-cod-top-causes-deaths":
        cdc_top_cause_deaths, cdc_top_cause_names = cdc.get_top_deaths_by_age_group(
            df=cod_all, year=most_recent_year
        )
        columnDefs = dash_formats.get_column_defs(cdc_top_cause_deaths)
        grid = dag.AgGrid(
            id={
                "type": "data-table",
                "tab": "cod-top-causes-deaths",
                "page": FILTER_COD_PREFIX,
            },
            rowData=cdc_top_cause_deaths.to_dict("records"),
            columnDefs=columnDefs,
            dashGridOptions={
                "defaultColDef": {
                    "width": 110,
                },
            },
        )
        tab_content = html.Div([grid])
    else:
        tab_content = dash.no_update

    # create the filters
    cdc_cod_filters = dash.no_update
    if not cdc_cod_num_filters:
        cdc_cod_filters = dh.generate_filters(
            df=cod_all,
            prefix=FILTER_COD_PREFIX,
            config=None,
            exclude_cols=[
                "deaths",
                "population",
                "crude_rate",
                "added_at",
                "icd_sub_chapter",
                "crude_95_confidence_interval",
                "m33",
                "data_through",
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
        tab_content,
        False,
        "",
    )


@callback(
    [
        Output(f"{FILTER_COD_TRENDS_PREFIX}-tab-content", "children"),
        Output(
            {
                "type": "filter-str",
                "prefix": FILTER_COD_TRENDS_PREFIX,
                "index": "simple_grouping",
            },
            "options",
        ),
        Output(
            {
                "type": "filter-str",
                "prefix": FILTER_COD_TRENDS_PREFIX,
                "index": "age_groups",
            },
            "options",
        ),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    [
        Input("button-cod-trends", "n_clicks"),
        Input(f"{FILTER_COD_TRENDS_PREFIX}-tabs", "active_tab"),
        Input(
            {
                "type": "filter-str",
                "prefix": FILTER_COD_TRENDS_PREFIX,
                "index": "simple_grouping",
            },
            "value",
        ),
    ],
    State(
        {
            "type": "filter-str",
            "prefix": FILTER_COD_TRENDS_PREFIX,
            "index": "age_groups",
        },
        "value",
    ),
)
def display_cdc_cod_trends(n_clicks, active_tab, option_value, age_group_value):
    """Create cdc cod trends."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"

    # get the data
    try:
        mcd99_cod = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd99_cod"
        )
        mcd18_cod = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd18_cod"
        )
    except Exception as e:
        logger.error(f"Failed to load table: {e}")
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            True,
            f"Failed to load data: {e}",
        )

    # get options
    age_group_options = [
        {"label": str(v), "value": v} for v in mcd18_cod["age_groups"].dropna().unique()
    ]

    # filter and concat
    mcd18_cod = mcd18_cod[mcd18_cod["year"] >= NEW_DATASET_START_YEAR]
    cod_all = pd.concat([mcd99_cod, mcd18_cod], ignore_index=True)
    include_age_groups = age_group_value if age_group_value else cdc.AGE_GROUP_ORDER
    cod_all = cod_all[cod_all["age_groups"].isin(include_age_groups)]

    # map category_col and add every combo of [year, age_groups, category_col]
    cod_all = cdc.map_reference(
        df=cod_all,
        col=CATEGORY_COL,
        on_dict={"icd_sub_chapter": "wonder_sub_chapter"},
    )
    cod_all = (
        cod_all.groupby(["year", "age_groups", CATEGORY_COL])
        .agg({"deaths": "sum", "population": "first"})
        .reset_index()
    )
    idx = pd.MultiIndex.from_product(
        [
            cod_all["year"].unique(),
            cod_all[CATEGORY_COL].unique(),
            cod_all["age_groups"].unique(),
        ],
        names=["year", CATEGORY_COL, "age_groups"],
    )
    cod_all = (
        cod_all.set_index(["year", CATEGORY_COL, "age_groups"])
        .reindex(idx)
        .reset_index()
    )
    cod_all["population"] = cod_all.groupby(["year", "age_groups"])[
        "population"
    ].transform("first")

    # normalize the partial deaths
    cod_all["deaths"] = cod_all["deaths"].astype(float)
    data_through = pd.Timestamp(mcd18_cod["data_through"].max())
    start_of_year = pd.Timestamp(f"{data_through.year}-01-01")
    days_elapsed = (data_through - start_of_year).days + 1
    factor_parial_year = 365 / days_elapsed
    mask = cod_all["year"] == data_through.year
    cod_all.loc[mask, "deaths"] *= factor_parial_year

    # create totals column
    totals = (
        cod_all.groupby(["year", "age_groups"])
        .agg({"deaths": "sum", "population": "first"})
        .reset_index()
    )
    totals[CATEGORY_COL] = "total"
    cod_all = pd.concat([cod_all, totals], ignore_index=True)
    cod_all["age_groups"] = pd.Categorical(
        cod_all["age_groups"], categories=cdc.AGE_GROUP_ORDER, ordered=True
    )
    category_orders = charters.get_category_orders(
        df=cod_all, category=CATEGORY_COL, measure="deaths"
    )

    # calculate crude_adj
    cod_all = cdc.map_reference(
        df=cod_all,
        col="population_%",
        on_dict={"age_groups": "age_bucket"},
        sheet_name="age_std_2000",
    )
    total_weight = cod_all.groupby("age_groups")["population_%"].first().sum()
    cod_all["population_%"] = cod_all["population_%"] / total_weight
    cod_all["crude_adj"] = (
        cod_all["deaths"] / cod_all["population"] * cod_all["population_%"] * 100000
    )

    # train the data based on year and the category using linear regression
    train_df = cod_all[
        (cod_all["year"] >= TRAIN_START_YEAR) & (cod_all["year"] <= TRAIN_END_YEAR)
    ]
    train_df = (
        train_df.groupby(["year", CATEGORY_COL])[["crude_adj"]].sum().reset_index()
    )

    # create the models
    models = {}
    for cod in train_df[CATEGORY_COL].unique():
        cod_subset = train_df[train_df[CATEGORY_COL] == cod]
        X = (cod_subset["year"] - TRAIN_START_YEAR).values.reshape(-1, 1)
        y = cod_subset["crude_adj"].values
        model = LinearRegression().fit(X, y)
        models[cod] = {
            "model": model,
            "coef": model.coef_[0],
            "intercept": model.intercept_,
        }

    # make the predictions
    test_df = cod_all[(cod_all["year"] >= TRAIN_START_YEAR)]
    test_df = (
        test_df.groupby(["year", CATEGORY_COL])[["crude_adj", "deaths", "population"]]
        .sum()
        .reset_index()
    )

    for cod, model in models.items():
        mask = test_df[CATEGORY_COL] == cod
        X = (test_df.loc[mask, "year"] - TRAIN_START_YEAR).values.reshape(-1, 1)
        test_df.loc[mask, "predictions"] = model["model"].predict(X)

    test_df["diff_pct"] = (test_df["crude_adj"] - test_df["predictions"]) / test_df[
        "predictions"
    ]
    test_df["diff_amt"] = (
        (test_df["population"] * test_df["crude_adj"])
        - (test_df["population"] * test_df["predictions"])
    ) / 100000
    test_df["deaths_pred"] = test_df["deaths"] - test_df["diff_amt"]
    test_df["year"] = test_df["year"].astype(str)  # needed for column pivot

    # chart
    if active_tab == f"{FILTER_COD_TRENDS_PREFIX}-tab-trends":
        cdc_cod_trends_chart = charters.compare_rates(
            df=test_df[test_df[CATEGORY_COL] == option_value],
            x_axis="year",
            rates=["crude_adj", "predictions"],
            display=True,
        )
        tab_content = html.Div([dcc.Graph(figure=cdc_cod_trends_chart)])
    # table amount
    elif active_tab == f"{FILTER_COD_TRENDS_PREFIX}-tab-actuals":
        cdc_cod_trends_chart = charters.chart(
            df=test_df,
            x_axis="year",
            y_axis="deaths",
            color=CATEGORY_COL,
            type="area",
            category_orders=category_orders,
            display=False,
        )
        pivot = cdc_cod_trends_chart.pivot(
            index=CATEGORY_COL, columns="year", values="deaths"
        )
        pivot.index = pd.Categorical(
            pivot.index, categories=category_orders[CATEGORY_COL], ordered=True
        )
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        pivot = pivot.sort_index().reset_index()
        columnDefs = dash_formats.get_column_defs(pivot)
        grid = dag.AgGrid(
            rowData=pivot.to_dict("records"),
            columnDefs=columnDefs,
        )
        tab_content = html.Div([grid])
    elif active_tab == f"{FILTER_COD_TRENDS_PREFIX}-tab-table-diff":
        cdc_cod_trends_chart = charters.chart(
            df=test_df,
            x_axis="year",
            y_axis="diff_amt",
            color=CATEGORY_COL,
            type="area",
            category_orders=category_orders,
            display=False,
        )
        pivot = cdc_cod_trends_chart.pivot(
            index=CATEGORY_COL, columns="year", values="diff_amt"
        )
        pivot.index = pd.Categorical(
            pivot.index, categories=category_orders[CATEGORY_COL], ordered=True
        )
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        pivot = pivot.sort_index().reset_index()
        columnDefs = dash_formats.get_column_defs(pivot)
        grid = dag.AgGrid(
            id={
                "type": "data-table",
                "tab": "diff_amt",
                "page": FILTER_COD_TRENDS_PREFIX,
            },
            rowData=pivot.to_dict("records"),
            columnDefs=columnDefs,
        )
        export_button = common_build._build_export_button(
            "diff_amt", FILTER_COD_TRENDS_PREFIX
        )
        tab_content = html.Div([export_button, grid])
    # table pct
    elif active_tab == f"{FILTER_COD_TRENDS_PREFIX}-tab-table-pct":
        cdc_cod_trends_chart = charters.chart(
            df=test_df,
            x_axis="year",
            y_axis="diff_pct",
            color=CATEGORY_COL,
            type="area",
            category_orders=category_orders,
            display=False,
        )
        pivot = cdc_cod_trends_chart.pivot(
            index=CATEGORY_COL, columns="year", values="diff_pct"
        )
        pivot.index = pd.Categorical(
            pivot.index, categories=category_orders[CATEGORY_COL], ordered=True
        )
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        pivot = pivot.sort_index().reset_index()
        pivot.columns = [
            col if col == "index" else f"{col}_pct" for col in pivot.columns
        ]
        columnDefs = dash_formats.get_column_defs(pivot)
        grid = dag.AgGrid(
            rowData=pivot.to_dict("records"),
            columnDefs=columnDefs,
        )
        tab_content = html.Div([grid])
    elif active_tab == f"{FILTER_COD_TRENDS_PREFIX}-tab-data":
        columnDefs = dash_formats.get_column_defs(test_df)
        grid = dag.AgGrid(
            id={"type": "data-table", "tab": "data", "page": FILTER_COD_TRENDS_PREFIX},
            rowData=test_df.to_dict("records"),
            columnDefs=columnDefs,
        )
        export_button = common_build._build_export_button(
            "data", FILTER_COD_TRENDS_PREFIX
        )
        tab_content = html.Div([export_button, grid])
    else:
        tab_content = dash.no_update

    category_options = [
        {"label": str(v), "value": v}
        for v in sorted(test_df[CATEGORY_COL].dropna().unique())
    ]

    return tab_content, category_options, age_group_options, False, ""


@callback(
    [
        Output("cdc-pop-trends-tab-content", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    [
        Input("button-pop-trends", "n_clicks"),
        Input("cdc-pop-trends-tabs", "active_tab"),
    ],
)
def display_cdc_pop_trends(n_clicks, active_tab):
    """Create cdc pop trends."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"

    # get the data
    try:
        mcd99_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_mi")
        mcd18_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_mi")
    except Exception as e:
        logger.error(f"Failed to load table: {e}")
        return dash.no_update, True, f"Failed to load data: {e}"

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

    # result
    result = (
        excess_grouped.groupby(["year"], observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    result["excess_lc"] = result["deaths"] / result["deaths_lc"]
    result["crude_rt"] = result["deaths"] / result["population"] * 100000
    result["crude_rt_lc"] = result["deaths_lc"] / result["population"] * 100000

    # chart
    if active_tab == "cdc-pop-trends-tab-excess-chart":
        chart = charters.compare_rates(
            result,
            x_axis="year",
            rates=["crude_rt", "crude_rt_lc"],
        )
        tab_content = html.Div([dcc.Graph(figure=chart)])
    # table
    elif active_tab == "cdc-pop-trends-tab-excess-table":
        columnDefs = dash_formats.get_column_defs(result)
        grid = dag.AgGrid(
            id={"type": "data-table", "tab": "excess-table", "page": "cdc-pop-trends"},
            rowData=result.to_dict("records"),
            columnDefs=columnDefs,
        )
        export_button = common_build._build_export_button(
            "excess-table", "cdc-pop-trends"
        )
        tab_content = html.Div([export_button, grid])
    else:
        tab_content = dash.no_update

    return tab_content, False, ""


@callback(
    [
        Output(f"{FILTER_MONTHLY_PREFIX}-tab-content", "children"),
        Output(
            {
                "type": "filter-str",
                "prefix": FILTER_MONTHLY_PREFIX,
                "index": "simple_chapter",
            },
            "options",
        ),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    [
        Input("button-monthly", "n_clicks"),
        Input(f"{FILTER_MONTHLY_PREFIX}-tabs", "active_tab"),
    ],
    State(
        {
            "type": "filter-str",
            "prefix": FILTER_MONTHLY_PREFIX,
            "index": "simple_chapter",
        },
        "value",
    ),
)
def display_cdc_monthly(n_clicks, active_tab, chapter_values):
    """Create cdc monthly."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    simple_chapter_options = dash.no_update

    # get the data
    try:
        mcd18_monthly = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd18_monthly"
        )
    except Exception as e:
        logger.error(f"Failed to load table: {e}")
        return dash.no_update, dash.no_update, True, f"Failed to load data: {e}"

    # filter
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
    if active_tab == f"{FILTER_MONTHLY_PREFIX}-tab-chart":
        cdc_monthly_chart = charters.chart(
            df=mcd18_monthly,
            x_axis="month",
            y_axis="deaths",
            color="year",
            type="line",
            color_discrete_map=color_discrete_map,
        )
        tab_content = html.Div([dcc.Graph(figure=cdc_monthly_chart)])

    # table
    elif active_tab == f"{FILTER_MONTHLY_PREFIX}-tab-table":
        columnDefs = dash_formats.get_column_defs(mcd18_monthly)
        grid = dag.AgGrid(
            id={"type": "data-table", "tab": "table", "page": FILTER_MONTHLY_PREFIX},
            rowData=mcd18_monthly.to_dict("records"),
            columnDefs=columnDefs,
            dashGridOptions={
                "defaultColDef": {
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                },
            },
            className="ag-theme-alpine",
            columnSize="sizeToFit",
        )
        export_button = common_build._build_export_button(
            "table", FILTER_MONTHLY_PREFIX
        )
        tab_content = html.Div([export_button, grid])

    # pic chart
    elif active_tab == f"{FILTER_MONTHLY_PREFIX}-tab-pic-chart":
        # get the data
        try:
            mcd18_monthly_cod = cdc.get_cdc_data_sql(
                db_filepath=db_filepath, table_name="mcd18_monthly_cod"
            )
            mcd99_monthly_cod = cdc.get_cdc_data_sql(
                db_filepath=db_filepath, table_name="mcd99_monthly_cod"
            )
            mcd18_monthly_cod = mcd18_monthly_cod[mcd18_monthly_cod["year"] >= 2021]
            mcd99_monthly_cod = mcd99_monthly_cod[mcd99_monthly_cod["year"] >= 2010]
            monthly_cod = pd.concat(
                [mcd18_monthly_cod, mcd99_monthly_cod], ignore_index=True
            )
        except Exception as e:
            logger.error(f"Failed to load table: {e}")
            return dash.no_update, dash.no_update, True, f"Failed to load data: {e}"

        # format data
        monthly_cod = cdc.map_reference(
            df=monthly_cod,
            col="simple_chapter",
            on_dict={"icd_chapter": "wonder_chapter"},
        )

        # get options
        simple_chapter_options = [
            {"label": str(v), "value": v}
            for v in sorted(monthly_cod["simple_chapter"].dropna().unique())
        ]

        # filter
        monthly_cod_filtered = monthly_cod[
            monthly_cod["simple_chapter"].isin(chapter_values)
        ].copy()

        # zero out the pandemic (2020-2022)
        mask_excluded = monthly_cod_filtered["year"].between(2020, 2022)
        monthly_cod_filtered.loc[mask_excluded, "deaths"] = np.nan

        # compute monthly_avg using 2019 and prior
        max_date = monthly_cod_filtered["reported_date"].max()
        cutoff = max_date - pd.DateOffset(months=2)
        monthly_avg = (
            monthly_cod_filtered[monthly_cod_filtered["year"] <= 2019]
            .groupby(["reported_date", "month"])["deaths"]
            .sum()
            .groupby(level="month")
            .mean()
        )

        # map the average
        monthly_cod_filtered["avg"] = monthly_cod_filtered["month"].map(monthly_avg)
        monthly_cod_filtered.loc[
            monthly_cod_filtered["reported_date"] > cutoff, "avg"
        ] = np.nan
        monthly_cod_filtered.loc[mask_excluded, "avg"] = np.nan

        # create chart
        cdc_monthly_cod_chart = charters.chart(
            df=monthly_cod_filtered,
            x_axis="reported_date",
            y_axis="deaths",
            color="simple_chapter",
            type="area",
        )

        # add average
        cdc_monthly_cod_chart.add_trace(
            go.Scatter(
                x=monthly_cod_filtered["reported_date"],
                y=monthly_cod_filtered["avg"],
                mode="lines",
                name="monthly avg",
                line={"color": "black", "width": 2},
            )
        )

        tab_content = html.Div([dcc.Graph(figure=cdc_monthly_cod_chart)])
    else:
        tab_content = dash.no_update

    return tab_content, simple_chapter_options, False, ""


@callback(
    [
        Output("cdc-weekly-tab-content", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    [
        Input("button-weekly", "n_clicks"),
        Input("cdc-weekly-tabs", "active_tab"),
    ],
)
def display_cdc_weekly(n_clicks, active_tab):
    """Create cdc monthly."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"

    # get the data
    try:
        mcd18_weekly = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd18_weekly"
        )
    except Exception as e:
        logger.error(f"Failed to load table: {e}")
        return dash.no_update, True, f"Failed to load data: {e}"

    # filter
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
    current_year_adj = current_year_adj[
        current_year_adj["recent_week"] >= PARTIAL_WEEKS_TO_EXCLUDE
    ]
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

    # cdc weekly
    if active_tab == "cdc-weekly-tab-chart":
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
        tab_content = html.Div([dcc.Graph(figure=cdc_weekly_chart)])
    elif active_tab == "cdc-weekly-tab-table":
        mcd18_weekly = mcd18_weekly.sort_values(
            ["mmwr_week_date", "mmwr_year"], ascending=False
        )
        columnDefs = dash_formats.get_column_defs(mcd18_weekly)
        grid = dag.AgGrid(
            id={"type": "data-table", "tab": "table", "page": "cdc-weekly"},
            rowData=mcd18_weekly.to_dict("records"),
            columnDefs=columnDefs,
            dashGridOptions={
                "defaultColDef": {
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                },
            },
            className="ag-theme-alpine",
            columnSize="sizeToFit",
        )
        export_button = common_build._build_export_button("table", "cdc-weekly")
        tab_content = html.Div([export_button, grid])

    # cdc weekly influenza
    elif active_tab in {"cdc-weekly-tab-flu-chart", "cdc-weekly-tab-flu-table"}:
        try:
            mcd18_weekly_pic = cdc.get_cdc_data_sql(
                db_filepath=db_filepath, table_name="mcd18_weekly_pic"
            )
        except Exception as e:
            logger.error(f"Failed to load table: {e}")
            return dash.no_update, True, f"Failed to load data: {e}"

        # calculate average (excluding recent weeks)
        mcd18_weekly_pic = mcd18_weekly_pic[
            mcd18_weekly_pic["icd_chapter"] == "Diseases of the respiratory system"
        ]
        max_date = mcd18_weekly_pic["mmwr_week_date"].max()
        cutoff = max_date - pd.Timedelta(weeks=PARTIAL_WEEKS_TO_EXCLUDE)
        weekly_avg = (
            mcd18_weekly_pic[mcd18_weekly_pic["mmwr_week_date"] <= cutoff]
            .groupby("mmwr_week")["deaths"]
            .mean()
        )
        mcd18_weekly_pic["avg"] = mcd18_weekly_pic["mmwr_week"].map(weekly_avg)
        mcd18_weekly_pic.loc[mcd18_weekly_pic["mmwr_week_date"] > cutoff, "avg"] = None
        if active_tab == "cdc-weekly-tab-flu-chart":
            cdc_flu_chart = charters.compare_rates(
                df=mcd18_weekly_pic,
                x_axis="mmwr_week_date",
                rates=["deaths", "avg"],
            )
            tab_content = html.Div([dcc.Graph(figure=cdc_flu_chart)])
        else:
            columnDefs = dash_formats.get_column_defs(mcd18_weekly_pic)
            grid = dag.AgGrid(
                id={
                    "type": "data-table",
                    "tab": "table",
                    "page": "cdc-weekly-flu",
                },
                rowData=mcd18_weekly_pic.to_dict("records"),
                columnDefs=columnDefs,
                dashGridOptions={
                    "defaultColDef": {
                        "sortable": True,
                        "filter": True,
                        "resizable": True,
                    },
                },
                className="ag-theme-alpine",
                columnSize="sizeToFit",
            )
            export_button = common_build._build_export_button("table", "cdc-weekly-flu")
            tab_content = html.Div([export_button, grid])
    else:
        tab_content = dash.no_update

    return tab_content, False, ""


@callback(
    [
        Output(f"{FILTER_MI_PREFIX}-tab-content", "children"),
        Output(f"{FILTER_MI_PREFIX}-filters", "children"),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    [
        Input("button-mi", "n_clicks"),
        Input(f"{FILTER_MI_PREFIX}-tabs", "active_tab"),
    ],
    [
        State(
            {"type": "filter-str", "prefix": FILTER_MI_PREFIX, "index": ALL}, "value"
        ),
        State(
            {"type": "filter-num", "prefix": FILTER_MI_PREFIX, "index": ALL}, "value"
        ),
    ],
)
def display_cdc_mi(n_clicks, active_tab, cdc_mi_str_filters, cdc_mi_num_filters):
    """Create cdc mi."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    states_info = dh._inputs_flatten_list(callback_context.states_list)

    # get the data
    try:
        mcd79_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd79_mi")
        mcd99_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_mi")
        mcd18_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_mi")
    except Exception as e:
        logger.error(f"Failed to load table: {e}")
        return dash.no_update, dash.no_update, True, f"Failed to load data: {e}"

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
    mi["era"] = pd.cut(
        mi["year"],
        bins=[0, 2019, 2024, float("inf")],
        labels=["1979-2019", "2020-2024", "2025+"],
    )

    # filter the data
    filtered_mi = dh.filter_data(df=mi, callback_context=states_info)
    mi_df = cdc.calc_mi(df=filtered_mi, rolling=ROLLING_MI_YEARS)

    # chart
    if active_tab == f"{FILTER_MI_PREFIX}-tab-mi-year-chart":
        mi_df = mi_df[mi_df["age_groups"] == "All ages"]
        cdc_mi_chart = charters.compare_rates(
            df=mi_df,
            x_axis="year",
            rates=["1_year_mi_pct", f"{ROLLING_MI_YEARS}_year_mi_pct", "whl_3_pct"],
        )
        tab_content = html.Div([dcc.Graph(figure=cdc_mi_chart)])
    # table
    elif active_tab == f"{FILTER_MI_PREFIX}-tab-mi-year-table":
        mi_df = mi_df[mi_df["age_groups"] == "All ages"]
        columnDefs = dash_formats.get_column_defs(mi_df)
        grid = dag.AgGrid(
            id={"type": "data-table", "tab": "table", "page": "cdc-mi"},
            rowData=mi_df.sort_values(by="year", ascending=False).to_dict("records"),
            columnDefs=columnDefs,
            defaultColDef={"resizable": True, "sortable": True, "filter": True},
        )
        export_button = common_build._build_export_button("table", FILTER_MI_PREFIX)
        tab_content = html.Div([export_button, grid])
    elif active_tab == f"{FILTER_MI_PREFIX}-tab-mi-age-chart":
        age_cats = mi_df["age_groups"].cat.remove_unused_categories()
        mi_df["age_groups_order"] = age_cats.map(
            {
                cat: f"{i + 1:02d}: {cat}"
                for i, cat in enumerate(age_cats.cat.categories)
            }
        ).fillna(mi_df["age_groups"].astype(str))
        cdc_mi_chart = charters.compare_rates(
            df=mi_df[mi_df["year"] == mi_df["year"].max()],
            x_axis="age_groups_order",
            rates=["1_year_mi_pct", f"{ROLLING_MI_YEARS}_year_mi_pct", "whl_3_pct"],
        )
        tab_content = html.Div([dcc.Graph(figure=cdc_mi_chart)])
    else:
        tab_content = dash.no_update

    # create the filters (if the don't exist)
    if not cdc_mi_num_filters:
        cdc_mi_filters = dh.generate_filters(
            df=mi,
            prefix=FILTER_MI_PREFIX,
            config=None,
            exclude_cols=[
                "deaths",
                "population",
                "crude_rate",
                "added_at",
                "crude_95_confidence_interval",
                "m33",
                "data_through",
            ],
        )["filters"]
    else:
        cdc_mi_filters = dash.no_update

    return tab_content, cdc_mi_filters, False, ""


#   _____                 _   _
#  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
#  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def refresh_cdc_data() -> None:
    """
    Refresh the cdc data.

    Includes a ~16 second sleep per call, due to CDC guidelines.
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
        time.sleep(CDC_WAIT_TIME_SECONDS)

        mcd18_monthly = cdc.get_cdc_data_xml(
            xml_filename="mcd18_monthly.xml", parse_date_col="Month"
        )
        sql.export_to_sql(
            df=mcd18_monthly,
            db_filepath=db_filepath,
            table_name="mcd18_monthly",
            if_exists="replace",
        )
        time.sleep(CDC_WAIT_TIME_SECONDS)

        mcd18_mi = cdc.get_cdc_data_xml(xml_filename="mcd18_mi.xml")
        sql.export_to_sql(
            df=mcd18_mi,
            db_filepath=db_filepath,
            table_name="mcd18_mi",
            if_exists="replace",
        )
        time.sleep(CDC_WAIT_TIME_SECONDS)

        mcd18_weekly = cdc.get_cdc_data_xml(
            xml_filename="mcd18_weekly.xml", parse_date_col="mmwr_week_date"
        )
        sql.export_to_sql(
            df=mcd18_weekly,
            db_filepath=db_filepath,
            table_name="mcd18_weekly",
            if_exists="replace",
        )
        time.sleep(CDC_WAIT_TIME_SECONDS)

        mcd18_weekly_pic = cdc.get_cdc_data_xml(
            xml_filename="mcd18_weekly_pic.xml", parse_date_col="mmwr_week_date"
        )
        sql.export_to_sql(
            df=mcd18_weekly_pic,
            db_filepath=db_filepath,
            table_name="mcd18_weekly_pic",
            if_exists="replace",
        )
        time.sleep(CDC_WAIT_TIME_SECONDS)

        mcd18_monthly_cod = cdc.get_cdc_data_xml(
            xml_filename="mcd18_monthly_cod.xml", parse_date_col="Month"
        )
        sql.export_to_sql(
            df=mcd18_monthly_cod,
            db_filepath=db_filepath,
            table_name="mcd18_monthly_cod",
            if_exists="replace",
        )
    except Exception as e:
        logger.error(f"Error refreshing cdc data: {e}")
