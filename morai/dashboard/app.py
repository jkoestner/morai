"""
Building plotly dashboard.

Builds plotly pages with call backs. There are 2 options the user has for running code.
1. Local running
2. Docker running (coming soon)

To run locally:
1. cd into directory with app.py
2. run plotly dashboard - `python app.py`

The ascii text is generated using https://patorjk.com/software/taag/
with "standard font"
"""

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
from dash_extensions.enrich import DashProxy, ServersideOutputTransform, dcc, html

from morai.utils import custom_logger, helpers

#      _    ____  ____
#     / \  |  _ \|  _ \
#    / _ \ | |_) | |_) |
#   / ___ \|  __/|  __/
#  /_/   \_\_|   |_|

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = DashProxy(
    __name__,
    use_pages=True,
    external_stylesheets=[
        # "https://codepen.io/chriddyp/pen/bWLwgP.css",
        dbc_css,
        dbc.themes.FLATLY,
    ],
    transforms=[ServersideOutputTransform()],
)
server = app.server
app.config.suppress_callback_exceptions = True

app.title = "Morai Dashboard"
app._favicon = "morai_logo.ico"

# creating the navbar
page_links = [
    dbc.NavItem(dbc.NavLink(page["name"], href=page["relative_path"]))
    for page in dash.page_registry.values()
]

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src=app.get_asset_url("morai_logo.ico"), height="30px")
                    ),
                    dbc.Col(dbc.NavbarBrand("morai", className="ms-3 fs-3")),
                ],
                align="center",
                className="g-0",
            ),
            dbc.Row(
                dbc.Nav(
                    page_links,
                    className="ms-auto",
                    navbar=True,
                ),
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
    sticky="top",
)

app.layout = html.Div(
    [
        dcc.Store(id="store-config", storage_type="session"),
        dcc.Store(id="store-dataset", storage_type="session"),
        dcc.Location(id="url", refresh=False),
        navbar,
        dash.page_container,
    ]
)


if __name__ == "__main__":
    custom_logger.set_log_level("DEBUG", module_prefix="pages")
    app.run_server(debug=True, host=helpers.HOST)
