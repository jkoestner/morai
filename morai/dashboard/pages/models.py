"""Experience dashboard."""

import dash_ag_grid as dag
import dash_extensions.enrich as dash
from dash_extensions.enrich import (
    Input,
    Output,
    callback,
    html,
)

from morai.dashboard.components import dash_formats
from morai.forecast import metrics
from morai.utils import custom_logger

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
            html.H4(
                "Model Analysis",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            html.Div(
                id="model-results",
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
    Output("model-results", "children"),
    Input("url", "pathname"),
)
def load_model_results(pathname):
    """Load model results."""
    if pathname != "/model":
        raise dash.exceptions.PreventUpdate

    # load models
    logger.debug("Loading model results")
    model_results = metrics.ModelResults(filepath="model_results.json")
    models = model_results.model
    scorecard = model_results.scorecard

    model_results_div = html.Div(
        [
            dag.AgGrid(
                rowData=models.to_dict("records"),
                columnDefs=dash_formats.get_column_defs(models),
            ),
            dag.AgGrid(
                rowData=scorecard.to_dict("records"),
                columnDefs=dash_formats.get_column_defs(scorecard),
            ),
        ]
    )
    return model_results_div
