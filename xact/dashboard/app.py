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

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

#      _    ____  ____
#     / \  |  _ \|  _ \
#    / _ \ | |_) | |_) |
#   / ___ \|  __/|  __/
#  /_/   \_\_|   |_|

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://codepen.io/chriddyp/pen/bWLwgP.css",
    ],
)
server = app.server
app.config.suppress_callback_exceptions = True

app.title = "Xact Dashboard"
app._favicon = "xact_logo.ico"
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    dcc.Link(
                        f"{page['name']} - {page['path']}", href=page["relative_path"]
                    )
                )
                for page in dash.page_registry.values()
            ]
        ),
        dash.page_container,
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
