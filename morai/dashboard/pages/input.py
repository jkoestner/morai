"""Data Input dashboard."""

import os

import dash
import dash_bootstrap_components as dbc
from dash import (
    dcc,
    html,
)

from morai.utils import helpers

dash.register_page(__name__, path="/input", title="morai - Input")


#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Input layout."""
    return html.Div(
        [
            html.H4(
                "Data Input",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            html.P("Root Directory: " + str(helpers.ROOT_PATH)),
            dcc.Dropdown(
                id="file-dropdown",
                options=[{"label": file, "value": file} for file in list_files()],
                value=list_files()[0],
            ),
            dbc.Button("Load Data", id="load-data", className="btn btn-primary"),
            html.P(),
        ],
        className="container",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


def list_files():
    """List files in the directory."""
    # List all entries in the given folder
    folder_path = helpers.ROOT_PATH / "files" / "dataset"
    all_entries = os.listdir(folder_path)
    # Filter out entries that are files
    files = [
        entry
        for entry in all_entries
        if os.path.isfile(os.path.join(folder_path, entry))
    ]
    return files
