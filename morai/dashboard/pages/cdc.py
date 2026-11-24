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
from morai.integrations import cdc, hmd
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
FILTER_POP_TRENDS_PREFIX = "cdc-pop-trends"
FILTER_MONTHLY_PREFIX = "cdc-monthly"
# provides when to use data from 18 dataset as there is overlap in 99 dataset
# the 99 dataset ends in 2020 and the 18 dataset starts in 2018
NEW_DATASET_START_YEAR = 2021
# training years for the death trend (cod linear baseline and Lee-Carter)
DEATH_TRAIN_START_YEAR = 2015
DEATH_TRAIN_END_YEAR = 2019
# training years for the population trend used to fill the ~2-year population lag
POP_TRAIN_START_YEAR = 2023
POP_TRAIN_END_YEAR = 2024
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
ROLLING_MI_YEARS = 10  # default for the rolling MI years filter


def layout():
    """CDC layout."""
    last_updated_meta = cdc.get_last_updated()
    hmd_meta = hmd.get_last_updated()
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
            dcc.Download(id="download-cdc-reference"),
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
                                            "Death data sourced from ",
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
                                            "Population data sourced from:",
                                            html.Br(),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        [
                                                            html.A(
                                                                "USA Exposures (HMD)",
                                                                href="https://www.mortality.org/Country/Country?cntr=USA",
                                                                target="_blank",
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                            ),
                                            "CDC Reference workbook: ",
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        [
                                                            html.A(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-file-excel me-1"
                                                                    ),
                                                                    "cdc_reference.xlsx",
                                                                ],
                                                                id="link-download-cdc-reference",
                                                                href="#",
                                                                n_clicks=0,
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
                                                    html.Li(
                                                        [
                                                            "HMD population data has better accuracy than CDC population data, "
                                                            "especially for older ages."
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
                                                "CDC Data Through: ",
                                            ),
                                            html.Span(
                                                last_updated_meta[
                                                    "data_through"
                                                ].date(),
                                                id="data-through-text",
                                            ),
                                            html.Br(),
                                            html.I(
                                                className="fas fa-clock me-1 text-muted"
                                            ),
                                            html.Label(
                                                "HMD Data Through: ",
                                            ),
                                            html.Span(
                                                hmd_meta["data_through"].date(),
                                                id="hmd-data-through-text",
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
                                    f"{DEATH_TRAIN_START_YEAR}-{DEATH_TRAIN_END_YEAR}. ",
                                    html.Br(),
                                    f"Note: population stats have a ~2-year lag; {POP_TRAIN_START_YEAR}-{POP_TRAIN_END_YEAR} population trend is extrapolated to fill the gap.",
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
                                                    label="Excess Trend - Rate",
                                                    tab_id="trends",
                                                ),
                                                common_build.TabSpec(
                                                    label="Excess Trend - Pct",
                                                    tab_id="trends-pct",
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
                                    f"Lee-Carter population model trained on {DEATH_TRAIN_START_YEAR}-{DEATH_TRAIN_END_YEAR}, extrapolated through the current year. ",
                                    html.Br(),
                                    f"Note: population stats have a ~2-year lag; {POP_TRAIN_START_YEAR}-{POP_TRAIN_END_YEAR} population trend is extrapolated to fill the gap.",
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
                                                    label="Excess Trend - Rate",
                                                    tab_id="excess-chart",
                                                ),
                                                common_build.TabSpec(
                                                    label="Excess Trend - Pct",
                                                    tab_id="excess-chart-pct",
                                                ),
                                                common_build.TabSpec(
                                                    label="Excess Table",
                                                    tab_id="excess-table",
                                                ),
                                            ],
                                            prefix=FILTER_POP_TRENDS_PREFIX,
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
                                                                "Age Group",
                                                                className="fw-bold",
                                                            ),
                                                            dcc.Dropdown(
                                                                id={
                                                                    "type": "filter-str",
                                                                    "prefix": FILTER_POP_TRENDS_PREFIX,
                                                                    "index": "age_groups",
                                                                },
                                                                options=[],
                                                                multi=True,
                                                                clearable=True,
                                                                className="ms-2",
                                                            ),
                                                            html.Label(
                                                                "Gender",
                                                                className="fw-bold mt-3",
                                                            ),
                                                            dcc.Dropdown(
                                                                id={
                                                                    "type": "filter-str",
                                                                    "prefix": FILTER_POP_TRENDS_PREFIX,
                                                                    "index": "gender",
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
                                                    [
                                                        html.Label(
                                                            "Rolling MI Years",
                                                            className="fw-bold",
                                                        ),
                                                        dbc.Input(
                                                            id=f"{FILTER_MI_PREFIX}-rolling-years",
                                                            type="number",
                                                            min=1,
                                                            step=1,
                                                            value=ROLLING_MI_YEARS,
                                                            debounce=True,
                                                            className="ms-2 mb-3",
                                                            style={"width": "6rem"},
                                                        ),
                                                        html.Div(
                                                            id=f"{FILTER_MI_PREFIX}-filters",
                                                        ),
                                                    ]
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


def _excess_pct_chart(x: pd.Series, pct: pd.Series, title: str) -> go.Figure:
    """
    Create a single-line chart of the excess percent (actual / expected - 1).

    Parameters
    ----------
    x : pd.Series
        The x-axis values (year).
    pct : pd.Series
        The excess percent values.
    title : str
        The chart title.

    Returns
    -------
    fig : go.Figure
        The excess percent chart.

    """
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=pct,
            mode="lines+markers",
            name="excess %",
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="year",
        yaxis_title="excess %",
        yaxis_tickformat=".0%",
    )
    return fig


@callback(
    Output("download-cdc-reference", "data"),
    Input("link-download-cdc-reference", "n_clicks"),
    prevent_initial_call=True,
)
def download_cdc_reference(n_clicks: int) -> dict:
    """Download the CDC reference workbook used for the simple grouping mapping."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc_reference.xlsx"
    if not filepath.exists():
        logger.error(f"The reference file was not found here: {filepath}")
        raise dash.exceptions.PreventUpdate
    return dcc.send_file(str(filepath))


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

            # get hmd data if the last update was more than a year ago
            hmd_data_through = pd.to_datetime(hmd.get_last_updated()["data_through"])
            if (pd.Timestamp.now().date() - hmd_data_through.date()).days > 365:
                logger.info("updating HMD data")
                hmd_df = hmd.get_hmd_dataframe()
                sql.export_to_sql(
                    df=hmd_df,
                    db_filepath=helpers.FILES_PATH / "integrations" / "hmd" / "hmd.sql",
                    table_name="hmd",
                    if_exists="replace",
                )

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
    hmd_db_filepath = helpers.FILES_PATH / "integrations" / "hmd" / "hmd.sql"

    # get the data
    try:
        mcd99_cod = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd99_cod"
        )
        mcd18_cod = cdc.get_cdc_data_sql(
            db_filepath=db_filepath, table_name="mcd18_cod"
        )
        hmd_df = hmd.get_hmd_data_sql(db_filepath=hmd_db_filepath, table_name="hmd")
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

    # merge in HMD population in place of CDC population
    # HMD population has more dynamic population data and
    # adjusts older age groups to be more accurate than CDC population data
    labels = [o["value"] for o in age_group_options if o["value"] != "Not Stated"]
    bins = [
        0 if v.startswith("<") else int(v.split("-")[0].split("+")[0].split()[0])
        for v in labels
    ] + [np.inf]

    hmd_grouped = (
        hmd_df.assign(
            age_groups=pd.cut(hmd_df["age"], bins=bins, labels=labels, right=False)
        )
        .groupby(["year", "age_groups"], observed=True, as_index=False)[
            "total_exposure"
        ]
        .sum()
    )
    hmd_grouped["age_groups"] = hmd_grouped["age_groups"].astype(str)
    cod_all = cod_all.merge(
        hmd_grouped,
        on=["year", "age_groups"],
        how="left",
        validate="many_to_one",
    )
    cod_all["population"] = cod_all.pop("total_exposure").combine_first(
        cod_all["population"]
    )
    cod_all.loc[(cod_all["age_groups"] == "Not Stated"), "population"] = np.nan

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

    # update population for provision years using trend
    pop_df = cod_all.groupby(["year", "age_groups"])["population"].first().reset_index()
    pop_df = _trend_population(
        pop_df=pop_df,
        group_col="age_groups",
        train_start_year=POP_TRAIN_START_YEAR,
        train_end_year=POP_TRAIN_END_YEAR,
    )
    cod_all = cod_all.drop(columns="population").merge(
        pop_df, on=["year", "age_groups"], how="left"
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
        (cod_all["year"] >= DEATH_TRAIN_START_YEAR)
        & (cod_all["year"] <= DEATH_TRAIN_END_YEAR)
    ]
    train_df = (
        train_df.groupby(["year", CATEGORY_COL])[["crude_adj"]].sum().reset_index()
    )

    # create the models
    models = {}
    for cod in train_df[CATEGORY_COL].unique():
        cod_subset = train_df[train_df[CATEGORY_COL] == cod]
        X = (cod_subset["year"] - DEATH_TRAIN_START_YEAR).values.reshape(-1, 1)
        y = cod_subset["crude_adj"].values
        model = LinearRegression().fit(X, y)
        models[cod] = {
            "model": model,
            "coef": model.coef_[0],
            "intercept": model.intercept_,
        }

    # make the predictions
    test_df = cod_all[(cod_all["year"] >= DEATH_TRAIN_START_YEAR)]
    test_df = (
        test_df.groupby(["year", CATEGORY_COL])[["crude_adj", "deaths", "population"]]
        .sum()
        .reset_index()
    )

    for cod, model in models.items():
        mask = test_df[CATEGORY_COL] == cod
        X = (test_df.loc[mask, "year"] - DEATH_TRAIN_START_YEAR).values.reshape(-1, 1)
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
            secondary="population",
            display=True,
        )
        tab_content = html.Div([dcc.Graph(figure=cdc_cod_trends_chart)])
    # chart pct
    elif active_tab == f"{FILTER_COD_TRENDS_PREFIX}-tab-trends-pct":
        cod_subset = test_df[test_df[CATEGORY_COL] == option_value]
        chart = _excess_pct_chart(
            x=cod_subset["year"],
            pct=cod_subset["diff_pct"],
            title=f"Excess % (crude_adj / predictions - 1): {option_value}",
        )
        tab_content = html.Div([dcc.Graph(figure=chart)])
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
        Output(f"{FILTER_POP_TRENDS_PREFIX}-tab-content", "children"),
        Output(
            {
                "type": "filter-str",
                "prefix": FILTER_POP_TRENDS_PREFIX,
                "index": "age_groups",
            },
            "options",
        ),
        Output(
            {
                "type": "filter-str",
                "prefix": FILTER_POP_TRENDS_PREFIX,
                "index": "gender",
            },
            "options",
        ),
        Output("cdc-toast", "is_open", allow_duplicate=True),
        Output("cdc-toast", "children", allow_duplicate=True),
    ],
    [
        Input("button-pop-trends", "n_clicks"),
        Input(f"{FILTER_POP_TRENDS_PREFIX}-tabs", "active_tab"),
    ],
    [
        State(
            {
                "type": "filter-str",
                "prefix": FILTER_POP_TRENDS_PREFIX,
                "index": "age_groups",
            },
            "value",
        ),
        State(
            {
                "type": "filter-str",
                "prefix": FILTER_POP_TRENDS_PREFIX,
                "index": "gender",
            },
            "value",
        ),
    ],
)
def display_cdc_pop_trends(n_clicks, active_tab, age_group_value, gender_value):
    """Create cdc pop trends."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    hmd_db_filepath = helpers.FILES_PATH / "integrations" / "hmd" / "hmd.sql"

    # get the data
    try:
        mcd99_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_mi")
        mcd18_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_mi")
        hmd_df = hmd.get_hmd_data_sql(db_filepath=hmd_db_filepath, table_name="hmd")
    except Exception as e:
        logger.error(f"Failed to load table: {e}")
        return dash.no_update, dash.no_update, True, f"Failed to load data: {e}"

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

    # get group options
    age_group_options = [
        {"label": str(v), "value": v}
        for v in excess.loc[excess["mapped_age"].notna(), "age_groups"]
        .dropna()
        .unique()
    ]
    gender_options = list(excess["gender"].dropna().unique())

    # filter the age groups
    include_age_groups = age_group_value if age_group_value else cdc.AGE_GROUP_ORDER
    excess = excess[excess["age_groups"].isin(include_age_groups)]

    # filter the genders
    include_genders = gender_value if gender_value else gender_options
    excess = excess[excess["gender"].isin(include_genders)]
    if excess.empty:
        logger.error("No data for the selected age groups and genders.")
        return (
            dash.no_update,
            age_group_options,
            True,
            "No data for the selected age groups and genders.",
        )

    # normalize the partial deaths
    excess["deaths"] = excess["deaths"].astype(float)
    data_through = pd.Timestamp(mcd18_mi["data_through"].max())
    start_of_year = pd.Timestamp(f"{data_through.year}-01-01")
    days_elapsed = (data_through - start_of_year).days + 1
    factor_parial_year = 365 / days_elapsed
    mask = excess["year"] == data_through.year
    excess.loc[mask, "deaths"] *= factor_parial_year

    # group by year, mapped age, and gender
    excess_grouped = (
        excess.groupby(["year", "mapped_age", "age_groups", "gender"], observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    max_year = excess_grouped["year"].max()

    # merge in HMD population in place of CDC population
    # HMD population has more dynamic population data and
    # adjusts older age groups to be more accurate than CDC population data
    labels = [o["value"] for o in age_group_options if o["value"] != "Not Stated"]
    hmd_grouped = _hmd_exposure_by_age_gender(hmd_df=hmd_df, age_groups=labels)
    excess_grouped = excess_grouped.merge(
        hmd_grouped,
        on=["year", "age_groups", "gender"],
        how="left",
        validate="many_to_one",
    )
    excess_grouped["population"] = excess_grouped.pop("hmd_exposure").combine_first(
        excess_grouped["population"]
    )
    excess_grouped.loc[(excess_grouped["age_groups"] == "Not Stated"), "population"] = (
        np.nan
    )

    # update population for provision years using trend
    excess_grouped = _trend_population(
        pop_df=excess_grouped,
        group_col=["mapped_age", "gender"],
        train_start_year=POP_TRAIN_START_YEAR,
        train_end_year=POP_TRAIN_END_YEAR,
    )
    excess_grouped = excess_grouped[(excess_grouped["year"] >= DEATH_TRAIN_START_YEAR)]

    # calculate qx_raw
    excess_grouped["qx_raw"] = excess_grouped["deaths"] / excess_grouped["population"]
    forecast_years = max_year - DEATH_TRAIN_END_YEAR

    # model using Lee-Carter, fit separately for each gender
    train_df = excess_grouped[
        (excess_grouped["year"] >= DEATH_TRAIN_START_YEAR)
        & (excess_grouped["year"] <= DEATH_TRAIN_END_YEAR)
    ]
    gender_forecasts = []
    for gender, gender_train_df in train_df.groupby("gender"):
        model = core.LeeCarter(
            age_col="mapped_age",
            year_col="year",
            actual_col="deaths",
            expose_col="population",
            interval=1,
        )
        fit_df = model.structure_df(gender_train_df)
        fit_df = model.fit(fit_df)

        # forecast
        gender_forecast_df = model.forecast(years=forecast_years)
        gender_forecast_df = pd.concat(
            [fit_df[["year", "mapped_age", "qx_lc"]], gender_forecast_df], axis=0
        )
        gender_forecast_df["gender"] = gender
        gender_forecasts.append(gender_forecast_df)
    forecast_df = pd.concat(gender_forecasts, axis=0, ignore_index=True)
    excess_grouped = pd.merge(
        excess_grouped,
        forecast_df[["mapped_age", "gender", "year", "qx_lc"]],
        on=["mapped_age", "gender", "year"],
        how="left",
    )
    excess_grouped["deaths_lc"] = excess_grouped["population"] * excess_grouped["qx_lc"]

    # result
    result = (
        excess_grouped.groupby(["year"], observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    result["excess_lc_pct"] = result["deaths"] / result["deaths_lc"]
    result["crude_rt"] = result["deaths"] / result["population"] * 100000
    result["crude_rt_lc"] = result["deaths_lc"] / result["population"] * 100000

    # chart
    if active_tab == f"{FILTER_POP_TRENDS_PREFIX}-tab-excess-chart":
        chart = charters.compare_rates(
            result,
            x_axis="year",
            rates=["crude_rt", "crude_rt_lc"],
            secondary="population",
        )
        tab_content = html.Div([dcc.Graph(figure=chart)])
    # chart pct
    elif active_tab == f"{FILTER_POP_TRENDS_PREFIX}-tab-excess-chart-pct":
        chart = _excess_pct_chart(
            x=result["year"],
            pct=result["crude_rt"] / result["crude_rt_lc"] - 1,
            title="Excess % (crude_rt / crude_rt_lc - 1)",
        )
        tab_content = html.Div([dcc.Graph(figure=chart)])
    # table
    elif active_tab == f"{FILTER_POP_TRENDS_PREFIX}-tab-excess-table":
        columnDefs = dash_formats.get_column_defs(result)
        grid = dag.AgGrid(
            id={
                "type": "data-table",
                "tab": "excess-table",
                "page": FILTER_POP_TRENDS_PREFIX,
            },
            rowData=result.to_dict("records"),
            columnDefs=columnDefs,
        )
        export_button = common_build._build_export_button(
            "excess-table", FILTER_POP_TRENDS_PREFIX
        )
        tab_content = html.Div([export_button, grid])
    else:
        tab_content = dash.no_update

    return tab_content, age_group_options, gender_options, False, ""


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
        Input(f"{FILTER_MI_PREFIX}-rolling-years", "value"),
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
def display_cdc_mi(
    n_clicks, active_tab, rolling_years, cdc_mi_str_filters, cdc_mi_num_filters
):
    """Create cdc mi."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # rolling years for the average mi (fall back to default if blank/invalid)
    try:
        rolling_years = int(rolling_years)
    except (TypeError, ValueError):
        rolling_years = ROLLING_MI_YEARS
    if rolling_years < 1:
        rolling_years = ROLLING_MI_YEARS

    # initialize
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    hmd_db_filepath = helpers.FILES_PATH / "integrations" / "hmd" / "hmd.sql"
    states_info = dh._inputs_flatten_list(callback_context.states_list)

    # get the data
    try:
        mcd79_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd79_mi")
        mcd99_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd99_mi")
        mcd18_mi = cdc.get_cdc_data_sql(db_filepath=db_filepath, table_name="mcd18_mi")
        hmd_df = hmd.get_hmd_data_sql(db_filepath=hmd_db_filepath, table_name="hmd")
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

    # collapse to the remapped age groups (e.g. 5-9 and 10-14 -> 5-14) so each
    # year / age group / gender has a single row before merging HMD exposure
    mi = mi.groupby(["year", "age_groups", "gender"], as_index=False)[
        ["deaths", "population"]
    ].sum()

    # merge in HMD exposure in place of CDC population
    # HMD population has more dynamic population data and
    # adjusts older age groups to be more accurate than CDC population data
    labels = [v for v in mi["age_groups"].unique() if v != "Not Stated"]
    hmd_grouped = _hmd_exposure_by_age_gender(hmd_df=hmd_df, age_groups=labels)
    mi = mi.merge(
        hmd_grouped,
        on=["year", "age_groups", "gender"],
        how="left",
        validate="one_to_one",
    )
    mi["population"] = mi.pop("hmd_exposure").combine_first(mi["population"])
    mi.loc[(mi["age_groups"] == "Not Stated"), "population"] = np.nan

    # update population for provisional years using trend
    mi = _trend_population(
        pop_df=mi,
        group_col=["age_groups", "gender"],
        train_start_year=POP_TRAIN_START_YEAR,
        train_end_year=POP_TRAIN_END_YEAR,
    )

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
    mi_df = cdc.calc_mi(df=filtered_mi, rolling=rolling_years)

    # chart
    if active_tab == f"{FILTER_MI_PREFIX}-tab-mi-year-chart":
        mi_df = mi_df[mi_df["age_groups"] == "All ages"]
        cdc_mi_chart = charters.compare_rates(
            df=mi_df,
            x_axis="year",
            rates=["1_year_mi_pct", f"{rolling_years}_year_mi_pct", "whl_3_pct"],
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
            rates=["1_year_mi_pct", f"{rolling_years}_year_mi_pct", "whl_3_pct"],
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


def _hmd_exposure_by_age_gender(
    hmd_df: pd.DataFrame,
    age_groups: list[str],
) -> pd.DataFrame:
    """
    Aggregate HMD single-age exposure into CDC age groups by gender.

    Parameters
    ----------
    hmd_df : pd.DataFrame
        HMD DataFrame with "year", "age", "female_exposure", and
        "male_exposure" columns.
    age_groups : list[str]
        CDC age group labels (e.g. "< 1 year", "1-4 years", "85+ years"),
        excluding "Not Stated". The lower bound of each label is used as the
        bin edge, so the labels must be contiguous.

    Returns
    -------
    hmd_grouped : pd.DataFrame
        DataFrame with "year", "age_groups", "gender" ("Female" / "Male"),
        and "hmd_exposure" columns.

    """
    lower_bounds = [
        0 if v.startswith("<") else int(v.split("-")[0].split("+")[0].split()[0])
        for v in age_groups
    ]
    order = np.argsort(lower_bounds)
    labels = [age_groups[i] for i in order]
    bins = [lower_bounds[i] for i in order] + [np.inf]

    hmd_grouped = (
        hmd_df.assign(
            age_groups=pd.cut(hmd_df["age"], bins=bins, labels=labels, right=False)
        )
        .groupby(["year", "age_groups"], observed=True, as_index=False)[
            ["female_exposure", "male_exposure"]
        ]
        .sum()
        .rename(columns={"female_exposure": "Female", "male_exposure": "Male"})
        .melt(
            id_vars=["year", "age_groups"],
            value_vars=["Female", "Male"],
            var_name="gender",
            value_name="hmd_exposure",
        )
    )
    hmd_grouped["age_groups"] = hmd_grouped["age_groups"].astype(str)

    # HMD stores exposure as 0 for years without exposure yet (e.g. latest year)
    hmd_grouped["hmd_exposure"] = hmd_grouped["hmd_exposure"].replace(0, np.nan)

    return hmd_grouped


def _trend_population(
    pop_df: pd.DataFrame,
    group_col: str | list,
    train_start_year: int,
    train_end_year: int,
) -> pd.DataFrame:
    """
    Extrapolate population for the provisional years using a linear trend.

    Population stats have a lag, so every year after `train_end_year` through
    the current (max) year is replaced by the `train_end_year` population grown
    at the trended rate fit on train_start_year-train_end_year for each group.
    The number of forecast years is current year - train_end_year.

    Parameters
    ----------
    pop_df : pd.DataFrame
        DataFrame with "year", group_col, and "population" columns, with one
        row per year and group.
    group_col : str | list
        The column(s) to fit a separate population trend for
        (e.g. "age group", or ["mapped_age", "gender"]).
    train_start_year : int
        The first year (inclusive) used to fit the population trend.
    train_end_year : int
        The last year (inclusive) used to fit the population trend. This is
        also the base year the trend is projected from.

    Returns
    -------
    pop_df : pd.DataFrame
        DataFrame with the population updated for the provisional years.

    """
    pop_df = pop_df.copy()
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    max_year = pop_df["year"].max()
    forecast_years = max_year - train_end_year
    base_year = max_year - forecast_years

    # train the population trend
    train_df = pop_df[
        (pop_df["age_groups"] != "Not Stated")
        & (pop_df["year"] >= train_start_year)
        & (pop_df["year"] <= train_end_year)
    ]
    train_df = (
        train_df.groupby(["year", *group_cols], observed=True)["population"]
        .sum()
        .reset_index()
    )

    # create the linear group models
    models = {}
    for group, group_subset in train_df.groupby(group_cols, observed=True):
        X = (group_subset["year"] - train_start_year).values.reshape(-1, 1)
        y = group_subset["population"].values
        model = LinearRegression().fit(X, y)
        models[group] = {
            "model": model,
            "coef": model.coef_[0],
            "intercept": model.intercept_,
        }
    bases = (
        pop_df[pop_df["year"] == base_year]
        .groupby(group_cols, observed=True)["population"]
        .sum()
        .to_dict()
    )
    coefs = {group: m["coef"] / m["intercept"] for group, m in models.items()}

    # prediction
    for year in [base_year + 1, base_year + 2]:
        for group, base in bases.items():
            group_key = group if isinstance(group, tuple) else (group,)
            idx = (pop_df["age_groups"] != "Not Stated") & (pop_df["year"] == year)
            for col, val in zip(group_cols, group_key, strict=True):
                idx &= pop_df[col] == val
            pop_df.loc[idx, "population"] = (1 + coefs.get(group, 0)) ** (
                year - base_year
            ) * base

    return pop_df


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
