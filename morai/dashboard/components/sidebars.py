"""Components - sidebar."""

import dash_bootstrap_components as dbc
from dash import html

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


explore_sidebar = html.Div(
    [
        html.H5("Navigation", className="hide-mobile"),
        html.Hr(className="hide-mobile"),
        dbc.Nav(
            [
                dbc.NavLink(
                    "Summary Statistics",
                    id="nav-summary-stat",
                    href="#section-summary-stat",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Numerical Frequencies",
                    id="nav-freq-num",
                    href="#section-freq-num",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Categorical Frequencies",
                    id="nav-freq-cat",
                    href="#section-freq-cat",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Target Analysis",
                    id="nav-target",
                    href="#section-target",
                    external_link=True,
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    className="hide-mobile",
    style={
        "position": "fixed",
        "left": "0px",
        "width": "250px",
        "top": "60px",
        "bottom": 0,
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
        "overflowY": "auto",
    },
)
