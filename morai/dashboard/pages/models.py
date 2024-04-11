"""Experience dashboard."""

import json

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
from dash_extensions.enrich import (
    Input,
    Output,
    Serverside,
    State,
    callback,
    dcc,
    html,
)

from morai.dashboard.components import dash_formats
from morai.dashboard.utils import dashboard_helper as dh
from morai.forecast import metrics
from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


dash.register_page(__name__, path="/model", title="morai - Model", order=3)


#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Model layout."""
    return html.Div(
        [
            dcc.Store(id="store-model-results", storage_type="session"),
            # -----------------------------------
            html.H4(
                "Model Analysis",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            html.Div(
                id="model-results",
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": key, "value": key}
                        for key in dh.list_files(helpers.FILES_PATH / "models")
                    ],
                    placeholder="Select a Model",
                ),
                width=2,
                className="p-1",
            ),
        ],
        className="container",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


@callback(
    Output("store-model-results", "data"),
    Input("url", "pathname"),
)
def load_model_results(pathname):
    """Load model results."""
    if pathname != "/model":
        raise dash.exceptions.PreventUpdate

    # load models
    model_results = metrics.ModelResults(filepath="model_results.json")

    return Serverside(model_results, key="model_results")


@callback(
    Output("model-results", "children"),
    Input("url", "pathname"),
    State("store-model-results", "data"),
)
def display_model_results(pathname, model_results):
    """Load model results."""
    if pathname != "/model" or model_results is None:
        raise dash.exceptions.PreventUpdate

    logger.debug("Display model results")
    # load models
    models = model_results.model
    scorecard = model_results.scorecard

    # update column defs
    model_column_defs = dash_formats.get_column_defs(models)
    model_col_def = {
        "headerName": "model_name",
        "field": "model_name",
        "pinned": "left",
    }
    model_column_defs = dash_formats.remove_column_defs(model_column_defs, "model_name")
    model_column_defs.insert(0, model_col_def)

    score_column_defs = dash_formats.get_column_defs(scorecard)
    score_col_def = {
        "headerName": "model_name",
        "field": "model_name",
        "pinned": "left",
    }
    score_column_defs = dash_formats.remove_column_defs(score_column_defs, "model_name")
    score_column_defs.insert(0, score_col_def)

    # create the divs
    model_results_div = html.Div(
        [
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dag.AgGrid(
                                                id="grid-model-dictionary",
                                                rowData=models.to_dict("records"),
                                                columnDefs=model_column_defs,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Markdown(
                                            id="markdown-model-dictionary",
                                            children=(
                                                "### Model Details ",
                                                "Select a cell to see value",
                                            ),
                                        ),
                                        width=6,
                                        style={"height": "400px", "overflowY": "auto"},
                                    ),
                                ],
                            ),
                        ],
                        title="Model Dictionary",
                    ),
                    dbc.AccordionItem(
                        [
                            dag.AgGrid(
                                rowData=scorecard.to_dict("records"),
                                columnDefs=score_column_defs,
                            ),
                        ],
                        title="Model Scorecard",
                    ),
                ],
                start_collapsed=True,
            )
        ]
    )
    return model_results_div


@callback(
    Output("markdown-model-dictionary", "children"),
    Input("grid-model-dictionary", "cellClicked"),
)
def clicked_cell_model_dictionary(cell):
    """Clicked cell model dictionary."""
    if not cell:
        raise dash.exceptions.PreventUpdate

    markdown = json.dumps(cell["value"], indent=2)
    markdown = f"```json\n{markdown}\n```"
    return markdown
