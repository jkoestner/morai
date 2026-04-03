"""
Analytics page for dashboard health metrics.

All metrics are computed at page load.
"""

from __future__ import annotations

import os
from typing import Any

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import psutil
from dash_extensions.enrich import html

from morai.integrations import cdc
from morai.utils import config_helper, custom_logger

logger = custom_logger.setup_logging(__name__)

ANALYTICS_TOKEN = config_helper.ANALYTICS_TOKEN

dash.register_page(
    __name__,
    path=f"/analytics/{ANALYTICS_TOKEN}",
    title="morai - Analytics",
    hidden=True,
)

# Registry of all lru_cache-decorated functions to report on
_CACHED_FUNCTIONS: list[Any] = [
    ("get_cdc_reference", "morai.integrations.cdc", cdc.get_cdc_reference),
]

#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout() -> html.Div:
    """Return the analytics page layout, computed fresh on each page load."""
    return html.Div(
        [
            dbc.Container(
                [
                    html.H3("Dashboard Analytics", className="mt-4 mb-4"),
                    html.P(
                        "Metrics are captured at page load. Refresh the browser to update.",
                        className="text-muted mb-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                _system_card(),
                                width=12,
                                lg=6,
                            ),
                            dbc.Col(
                                _cache_card(),
                                width=12,
                                lg=6,
                            ),
                        ]
                    ),
                ],
                fluid=True,
            ),
        ]
    )


#   _____                 _   _
#  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
#  | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def _system_card() -> dbc.Card:
    """Build the system resources card."""
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    rss_mb = mem_info.rss / 1024**2
    total_ram_mb = psutil.virtual_memory().total / 1024**2
    mem_pct = rss_mb / total_ram_mb * 100
    cpu_pct = proc.cpu_percent(interval=None)

    rows = [
        ("Memory (RSS)", f"{rss_mb:.1f} MB"),
        ("Memory %", f"{mem_pct:.2f}%"),
        ("CPU %", f"{cpu_pct:.1f}%"),
    ]

    table_rows = [
        html.Tr([html.Th(label, style={"width": "50%"}), html.Td(value)])
        for label, value in rows
    ]

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("System Resources", className="mb-0")),
            dbc.CardBody(
                dbc.Table(
                    html.Tbody(table_rows),
                    bordered=False,
                    striped=True,
                    size="sm",
                    className="mb-0",
                )
            ),
        ],
        className="mb-4",
    )


def _cache_card() -> dbc.Card:
    """Build the LRU cache statistics card."""
    header = html.Thead(
        html.Tr(
            [
                html.Th("Function"),
                html.Th("Module"),
                html.Th("Hits"),
                html.Th("Misses"),
                html.Th("Size"),
                html.Th("Max"),
                html.Th("Hit Rate"),
            ]
        )
    )

    body_rows = []
    for func_name, module, func in _CACHED_FUNCTIONS:
        info = func.cache_info()  # type: ignore[union-attr]
        total = info.hits + info.misses
        hit_rate = f"{info.hits / total * 100:.1f}%" if total > 0 else "—"
        body_rows.append(
            html.Tr(
                [
                    html.Td(func_name),
                    html.Td(module, style={"fontSize": "0.8rem", "color": "gray"}),
                    html.Td(info.hits),
                    html.Td(info.misses),
                    html.Td(info.currsize),
                    html.Td(info.maxsize),
                    html.Td(hit_rate),
                ]
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("LRU Cache", className="mb-0")),
            dbc.CardBody(
                dbc.Table(
                    [header, html.Tbody(body_rows)],
                    bordered=False,
                    striped=True,
                    size="sm",
                    responsive=True,
                    className="mb-0",
                )
            ),
        ],
        className="mb-4",
    )
